//! A self-contained verifier for leanVM execution proofs: single threaded, std only, plain integer arithmetic.
//!
//! Kept close enough to the Lean transcription (`formal/Statement/LeanVmVerifier.lean`) to read side by side: the same functions, in the same order, with the same constants.
//!
//! What the transcription has no counterpart for sits in modules of its own: the prepared [`Bytecode`], the proof's byte encoding ([`Proof::to_bytes`]) and what the statement of an aggregate adds, [`da`].
//!
//! One walk of the protocol serves two entry points. [`verify_execution`] checks the proof. [`replay_unchecked`] reads the same [`Summary`] out of it and checks nothing: every check it leaves out is behind `Transcript::checked`.

#![forbid(unsafe_code)]

mod bus;
mod bytecode;
pub mod da;
mod field;
mod hash;
pub mod keccak;
pub mod layout;
mod merkle;
mod multilinear;
mod ring_switch;
mod transcript;
mod whir;
mod wire;

pub use bytecode::Bytecode;
pub use field::{E, K};
pub use layout::{
    MAX_LOG_BYTECODE, MAX_LOG_MEMORY, MAX_STACKED_LOG, MAX_TABLE_LOG, MIN_LOG_MEMORY, MIN_STACKED_LOG, TAG, Truncation,
    min_execution_stacked_log,
};
pub use merkle::MerklePath;
pub use transcript::{Proof, RATE_SCALARS, SECURITY_BITS, grinding_bits};
pub use whir::{
    BATCH_GRINDING_BITS, FOLD_GRINDING_BITS, INITIAL_FOLDING_FACTOR, MAX_LOG_INV_RATE, MIN_LOG_INV_RATE,
    OOD_GRINDING_BITS, QUERY_GRINDING_BITS, RESIDUAL_MAX_LOG, RS_DOMAIN_INITIAL_REDUCTION_FACTOR,
    RS_DOMAIN_SUBSEQUENT_REDUCTION_FACTOR, SUBSEQUENT_FOLDING_FACTOR, whir_queries,
};
pub use wire::{MAX_HEIGHT, MAX_LEAF_WORDS, MAX_PHASES, MAX_QUERIES};

// What the provers and the aggregation's guest constants and witness read of the protocol.
pub use bus::{N_TREES as N_BUS_TREES, bus_grinding_bits};
pub use hash::Digest;
pub use multilinear::stack_offsets;
pub use whir::{Subspace, WhirConfig, derive_config, subspace_roots};

use bus::{ColumnClaim, N_TREES, table_sumcheck, verify_bus_balance};
use field::{ZERO, powers};
use hash::{ROUNDS, STATE_LANES};
use keccak::{KeccakShape, verify_keccak};
use layout::{
    GLOBAL_COLUMN_BASES, HASH_TABLES, K_BITS, KECCAK_OUT, MEMORY_0, MEMORY_1, N_TABLES, QKECCAK, build_layout,
};
use multilinear::{dot, eq_eval, multilinear_eval};
use ring_switch::{ring_switch, ring_switch_many};
use std::ops::Range;
use transcript::Transcript;
use whir::verify_whir;

/// Why a proof was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifierError {
    MultilinearSize,
    NotPowerOfTwo,
    StreamExhausted,
    GrindingNonce,
    MerkleOpeningMissing,
    MerkleOpeningShape,
    MerkleOpeningPosition,
    MerkleRootMismatch,
    StreamNotConsumed,
    MerkleNotConsumed,
    /// The GKR layer whose children do not match its sumcheck.
    GkrLayer(usize),
    BusCountZero,
    BytecodeShare,
    BytecodeValue,
    TableSumcheckTerminal,
    AnnouncedTableSizes,
    WhirShape,
    QueryCountLadder,
    InducedBasisDimensions,
    WhirFinalQuery,
    WhirTerminal,
    WhirNoTerminalLevel,
    KeccakProduct,
    KeccakLayer,
    KeccakOutputOpening,
    KeccakInputOpening,
    AnnouncedHighLimb,
    PcsInverseRate,
    CommittedSize,
    StackedClaims,
}

impl std::fmt::Display for VerifierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::MultilinearSize => "multilinear table has the wrong size",
            Self::NotPowerOfTwo => "expected a power of two",
            Self::StreamExhausted => "proof stream exhausted",
            Self::GrindingNonce => "invalid grinding nonce",
            Self::MerkleOpeningMissing => "Merkle opening missing",
            Self::MerkleOpeningShape => "Merkle opening has the wrong shape",
            Self::MerkleOpeningPosition => "Merkle opening at the wrong position",
            Self::MerkleRootMismatch => "Merkle root mismatch",
            Self::StreamNotConsumed => "proof stream not fully consumed",
            Self::MerkleNotConsumed => "Merkle openings not fully consumed",
            Self::GkrLayer(layer) => return write!(f, "GKR layer {layer}: children do not match the sumcheck"),
            Self::BusCountZero => "a bus count is zero",
            Self::BytecodeShare => "bytecode sumcheck terminal mismatch",
            Self::BytecodeValue => "bytecode evaluation mismatch",
            Self::TableSumcheckTerminal => "table sumcheck terminal mismatch",
            Self::AnnouncedTableSizes => "invalid announced table sizes",
            Self::WhirShape => "invalid WHIR shape",
            Self::QueryCountLadder => "tabulated query count does not match the ladder",
            Self::InducedBasisDimensions => "bad induced-basis dimensions",
            Self::WhirFinalQuery => "WHIR final query does not match the final polynomial",
            Self::WhirTerminal => "WHIR terminal check failed",
            Self::WhirNoTerminalLevel => "WHIR verification ended without a terminal level",
            Self::KeccakProduct => "Keccak product polynomial does not carry the claim",
            Self::KeccakLayer => "Keccak layer evaluations do not match the sumcheck",
            Self::KeccakOutputOpening => "Keccak output opening ends on the wrong value",
            Self::KeccakInputOpening => "Keccak input opening ends on the wrong value",
            Self::AnnouncedHighLimb => "announced size has a nonzero high limb",
            Self::PcsInverseRate => "invalid PCS inverse rate",
            Self::CommittedSize => "committed size outside the PCS window",
            Self::StackedClaims => "too many stacked claims",
        };
        f.write_str(message)
    }
}

impl std::error::Error for VerifierError {}

fn require(condition: bool, error: VerifierError) -> Result<(), VerifierError> {
    if condition { Ok(()) } else { Err(error) }
}

/// The stacked opening's batching is accounted for up to this many claims, the ring-switched one included. The count
/// follows from the announced shape, so a replay checks it too, like the committed size.
pub const MAX_STACKED_CLAIMS: usize = 512;

/// The weight a claim puts on the stack, and the value it claims for it.
type StackClaim<'a> = (Box<dyn Fn(&[E]) -> E + 'a>, E);

/// Discharge every claim on the committed stack in one opening: the same powers of one challenge
/// batch the values into the target, and the weights into the basis WHIR evaluates at its terminal point.
///
/// The challenge is drawn right after the ring switch's map challenges, one verifier message under their one grind
/// (`BATCH_GRINDING_BITS`), so it takes no grind of its own.
fn verify_stacked_opening(
    transcript: &mut Transcript,
    root: Digest,
    stack_log: usize,
    log_inv_rate: usize,
    claims: &[StackClaim],
) -> Result<(), VerifierError> {
    let values: Vec<E> = claims.iter().map(|(_, value)| *value).collect();
    require(claims.len() <= MAX_STACKED_CLAIMS, VerifierError::StackedClaims)?;
    let scales = powers(transcript.sample(), claims.len());
    let evaluate_basis = |point: &[E]| {
        let weights: Vec<E> = claims.iter().map(|(weight, _)| weight(point)).collect();
        dot(&scales, &weights)
    };
    let target = dot(&scales, &values);
    verify_whir(transcript, stack_log, log_inv_rate, target, root, &evaluate_basis)
}

/// An OOD sample, read right after its commitment: a grind of `bits`, a point `z` of the committed variables, then the
/// committed multilinear at `z`. Level 0's claim follows every other stacked claim.
fn read_ood(transcript: &mut Transcript, vars: usize, bits: u32) -> Result<StackClaim<'static>, VerifierError> {
    transcript.grind_check(bits)?;
    let z = transcript.samples(vars);
    let value = transcript.next_scalar()?;
    Ok((Box::new(move |x| eq_eval(&z, x)), value))
}

/// What a walk of the proof leaves behind, for a prover that proves this verification in turn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    /// The announced sizes: the memory's log size and each table's row count.
    pub log_memory: usize,
    pub table_rows: Vec<u64>,
    pub log_inv_rate: usize,
    /// What the rate word announces of a truncated proof, `None` for a full one.
    pub truncation: Option<Truncation>,
    /// The committed stack's log size.
    pub stack_log: usize,
    /// Where the bus's bytecode sumcheck leaves the stacked bytecode, `(zeta_low, r)`. Its value there is the claim an aggregate defers.
    pub bytecode_point: Vec<E>,
    /// The stream positions the Keccak reduction reads.
    pub keccak_scalars: Range<usize>,
}

/// Verify that `proof` shows an execution of `bytecode` (the stacked bytecode multilinear) on the 256-bit `public_input`.
pub fn verify_execution(bytecode: &Bytecode, public_input: &[u8; 32], proof: &Proof) -> Result<Summary, VerifierError> {
    walk(bytecode, public_input, proof, true)
}

/// NOT a verification: the [`Summary`] of `proof`, read out without checking it. An invalid proof is accepted.
///
/// For a prover that reads its guest's witness out of a proof the guest then verifies in full. It walks the same transcript as [`verify_execution`] and leaves out every check: the sumcheck, GKR, constraint, Keccak and WHIR terminal identities, the bus count's nonzero check, the proof of work, the bytecode evaluation and the Merkle roots. It still fails on a proof it cannot walk: announced sizes out of range, a stream or openings of the wrong length or shape, a Merkle path at a position the protocol did not open.
pub fn replay_unchecked(bytecode: &Bytecode, public_input: &[u8; 32], proof: &Proof) -> Result<Summary, VerifierError> {
    walk(bytecode, public_input, proof, false)
}

/// Exposed for the Keccak reduction's benchmark and tests: `2^table_bits` tables of `2^log_steps` permutations each
/// proved alone, a state committed every `depth` rounds, the committed states the whole stack.
///
/// The proof is the commitment root and its OOD sample, the output lanes of every table claimed at a challenge point,
/// the reduction, then the ring switch and its opening, as in an execution proof. Its public input is the shape, the
/// words `log_steps`, `depth`, `log_inv_rate` and `table_bits`, so a proof of one shape verifies for no other.
pub fn verify_keccak_batch(
    fiat_shamir_iv: &[u8; 32],
    log_steps: usize,
    depth: usize,
    table_bits: usize,
    log_inv_rate: usize,
    proof: &Proof,
) -> Result<(), VerifierError> {
    require(
        log_steps >= layout::KECCAK_MIN_STEPS_LOG
            && depth > 0
            && ROUNDS.is_multiple_of(depth)
            && table_bits <= 2
            && (table_bits == 0 || (ROUNDS / depth).is_power_of_two()),
        VerifierError::CommittedSize,
    )?;
    let shape = KeccakShape {
        log_steps,
        depth,
        table_bits,
    };
    let stack_log = log_steps + table_bits + shape.slot_bits();
    require(
        (MIN_STACKED_LOG..=MAX_STACKED_LOG).contains(&stack_log),
        VerifierError::CommittedSize,
    )?;
    let mut statement = [0u8; 32];
    for (bytes, word) in statement
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip([log_steps, depth, log_inv_rate, table_bits])
    {
        *bytes = (word as u64).to_le_bytes();
    }
    let mut transcript = Transcript::new(proof, Digest(*fiat_shamir_iv), Digest(statement), true);
    let halves = transcript.next_scalars(2)?;
    let root = Digest::from_halves(halves[0], halves[1]);
    let ood = read_ood(&mut transcript, stack_log, OOD_GRINDING_BITS)?;
    // The instance pins the committed middle states, a multilinear identity of degree `log_steps` in it.
    transcript.grind_check(grinding_bits(log_steps as u128))?;
    let instance = transcript.samples(log_steps);
    let output = transcript.next_scalars(shape.tables() * STATE_LANES)?;
    let openings = verify_keccak(&mut transcript, shape, &instance, &output)?;
    let (point, slices) = openings.inputs;
    let (target, weight) = ring_switch(point, &slices, &mut transcript)?;
    let mut claims: Vec<StackClaim> = vec![(Box::new(weight), target)];
    if let Some((point, word)) = openings.outputs {
        claims.push((Box::new(move |x| eq_eval(&point, x)), word));
    }
    claims.push(ood);
    verify_stacked_opening(&mut transcript, root, stack_log, log_inv_rate, &claims)?;
    transcript.finish()?;
    Ok(())
}

fn walk(bytecode: &Bytecode, public_input: &[u8; 32], proof: &Proof, checked: bool) -> Result<Summary, VerifierError> {
    let public_input = Digest(*public_input);
    let mut transcript = Transcript::new(proof, bytecode.fiat_shamir_iv, public_input, checked);

    // 1] memory log-size, table row counts, and log-inv-rate in WHIR
    let announced = transcript.next_scalars(2 + N_TABLES)?;
    // The last word is `(rate, (mem_count << 2) | flag)`. Flag 1 announces a truncated proof over `mem_count`
    // memory cells. Flag 2 announces one whose q_keccak is a region (QT). Flag 3 is refused.
    let rate_word = announced[1 + N_TABLES];
    let flag = rate_word.c1 & 3;
    require(
        announced[..1 + N_TABLES].iter().all(|value| value.c1 == 0) && flag != 3 && (flag != 0 || rate_word.c1 == 0),
        VerifierError::AnnouncedHighLimb,
    )?;
    let log_memory = announced[0].c0;
    let table_rows: Vec<u64> = announced[1..1 + N_TABLES].iter().map(|value| value.c0).collect();
    let log_inverse_rate = rate_word.c0;
    require(
        (MIN_LOG_INV_RATE as u64..=MAX_LOG_INV_RATE as u64).contains(&log_inverse_rate),
        VerifierError::PcsInverseRate,
    )?;
    let truncation = (flag != 0).then_some(Truncation {
        mem_count: rate_word.c1 >> 2,
        qt: flag == 2,
    });
    let layout = build_layout(bytecode, log_memory, &table_rows, truncation)?;
    require(
        (MIN_STACKED_LOG..=MAX_STACKED_LOG).contains(&layout.stack_log),
        VerifierError::CommittedSize,
    )?;

    // 2] parse WHIR commitment: one Merkle root (its OOD sample follows, outside the PCS).
    let halves = transcript.next_scalars(2)?;
    let root = Digest::from_halves(halves[0], halves[1]);
    let ood = read_ood(&mut transcript, layout.stack_log, OOD_GRINDING_BITS)?;

    // 3] Bus: one batched GKR over the push, pull and count trees, then the leaf decomposition, which leaves each table
    // a degree-2 claim per tree.
    let bus = verify_bus_balance(&layout, &mut transcript)?;

    // 4] One batched (back-loaded) "table sumcheck" over all six tables, at the bus point, proving the target the
    // trees' leaf claims derive and that constraints vanish. Every table takes a disjoint range of xi powers for its
    // constraints, then one power per bus tree, shared by every table. The batch is degree n_powers - 1 in xi.
    let n_constraints: usize = layout.tables.iter().map(|table| table.n_constraints()).sum();
    let n_powers = n_constraints + N_TREES;
    transcript.grind_check(grinding_bits(n_powers as u128 - 1))?;
    let xi = transcript.sample();
    let xi_powers = powers(xi, n_powers);
    let (constraint_powers, form_powers) = xi_powers.split_at(n_constraints);
    let target = dot(form_powers, &bus.totals);
    let evaluations = table_sumcheck(
        &layout.tables,
        &layout.table_log_heights,
        &bus.forms,
        constraint_powers,
        form_powers,
        &bus.point,
        target,
        &mut transcript,
    )?;
    let mut claims = bus.claims;
    for (base, (point, values)) in GLOBAL_COLUMN_BASES.into_iter().zip(&evaluations) {
        claims.extend(values.iter().enumerate().map(|(local, &value)| ColumnClaim {
            column: base + local,
            point: point.clone(),
            value,
        }));
    }

    // 5] binding the public input: each memory limb's claim is that limb's public line at the challenge, which the verifier
    // evaluates itself, so the prover sends nothing. A line: numerator one, no grind.
    let public_challenge = transcript.sample();
    let (first, second) = public_input.halves();
    let mut public_point = vec![ZERO; layout.placements[MEMORY_0].expect("memory is committed").variables];
    public_point[0] = public_challenge;
    let public_pairs = [(first.c0, second.c0), (first.c1, second.c1)];
    for (column, (low, high)) in [MEMORY_0, MEMORY_1].into_iter().zip(public_pairs) {
        claims.push(ColumnClaim {
            column,
            point: public_point.clone(),
            value: multilinear_eval(&[K(low), K(high)], &[public_challenge])?,
        });
    }

    // 6] KECCAK and KECCAK64 validity, by GKR over the rounds. The tables' claims on their output lanes are the reduction's
    // claim on the permutations' output: they leave the batch, their columns having no placement. Each table's claim
    // is at a prefix of the tallest's point, so a shorter table is completed with its padding step: the claim at the
    // tallest's point is `eq(0) claim + (1 + eq(0)) padding`. `KECCAK64` claims its output lanes 0 to 3, and the prover
    // sends the others, which only the reduction reads.
    let keccak_start = transcript.stream_offset();
    let kappa = layout.keccak.log_steps;
    let tallest = HASH_TABLES
        .iter()
        .copied()
        .find(|&table| layout.table_log_heights[table] == kappa)
        .expect("a tallest hash table");
    let rho = evaluations[tallest].0.clone();
    let keccak64_rest = transcript.next_scalars(STATE_LANES - layout::KECCAK64_OUT_LANES)?;
    let mut output = Vec::with_capacity(HASH_TABLES.len() * STATE_LANES);
    for (half, (&table, padding)) in HASH_TABLES.iter().zip(layout::hash_padding_outputs()).enumerate() {
        let (_, values) = &evaluations[table];
        let uncovered = rho[layout.table_log_heights[table]..]
            .iter()
            .fold(field::ONE, |e, &x| e * (field::ONE + x));
        for (lane, &pad) in padding.iter().enumerate() {
            let claimed = match (half, lane) {
                (0, lane) => values[KECCAK_OUT + lane],
                (_, lane) if lane < layout::KECCAK64_OUT_LANES => values[layout::KECCAK64_OUT + lane],
                (_, lane) => keccak64_rest[lane - layout::KECCAK64_OUT_LANES],
            };
            output.push(uncovered * claimed + (field::ONE + uncovered) * E::new(pad, 0));
        }
    }
    let openings = verify_keccak(&mut transcript, layout.keccak, &rho, &output)?;
    let keccak_scalars = keccak_start..transcript.stream_offset();
    if let Some((point, value)) = openings.outputs {
        claims.push(ColumnClaim {
            column: QKECCAK,
            point,
            value,
        });
    }

    // 7] Ring-switching. A claim is supported on q_keccak's block of the stack, so its weight carries the
    // placement's selector; the claims lead the batch, taking the first powers. A truncated q_keccak is pieces,
    // each a claim of its own: the prover sends the slices of all but the last, which the reduction's slices fix.
    let (point, slices) = openings.inputs;
    let mut stack_claims: Vec<StackClaim> = Vec::new();
    match &layout.qt {
        None => {
            let (target, weight) = ring_switch(point, &slices, &mut transcript)?;
            let qkeccak = layout.placements[QKECCAK].expect("q_keccak is committed");
            stack_claims.push((
                Box::new(move |x| qkeccak.eq_above(x) * weight(&x[..qkeccak.variables])),
                target,
            ));
        }
        Some(qt) => {
            let split = qt.ring_split(&point);
            let pieces = split.len();
            let share = qt.pad_share(&point);
            // The derived piece is the largest, the first of them.
            let derived = (0..pieces)
                .max_by_key(|&j| (qt.pieces[j].b(), std::cmp::Reverse(j)))
                .expect("a piece");
            let sent = transcript.next_scalars(K_BITS * (pieces - 1))?;
            let mut slices_of: Vec<Vec<E>> = Vec::with_capacity(pieces);
            let mut sent_chunks = sent.chunks(K_BITS);
            for j in 0..pieces {
                slices_of.push(if j == derived {
                    Vec::new()
                } else {
                    sent_chunks.next().expect("a piece's slices").to_vec()
                });
            }
            let mut last: Vec<E> = slices.iter().zip(&share).map(|(&s, &pad)| s + pad).collect();
            for ((_, coefficient), own) in split.iter().zip(&slices_of) {
                for (slice, &value) in last.iter_mut().zip(own) {
                    *slice += *coefficient * value;
                }
            }
            let coefficient = split[derived].1;
            require(coefficient != ZERO, VerifierError::KeccakInputOpening)?;
            let inverse = coefficient.inv();
            slices_of[derived] = last.into_iter().map(|slice| slice * inverse).collect();
            let all = split.into_iter().map(|(point, _)| point).zip(slices_of).collect();
            let rings = ring_switch_many(all, &mut transcript)?;
            for (piece, (target, weight)) in qt.pieces.iter().zip(rings) {
                let placement = piece.placement;
                stack_claims.push((
                    Box::new(move |x| placement.eq_above(x) * weight(&x[..placement.variables])),
                    target,
                ));
            }
        }
    }
    stack_claims.extend(claims.iter().filter_map(|claim| claim.on_stack(&layout)));
    stack_claims.push(ood);
    verify_stacked_opening(
        &mut transcript,
        root,
        layout.stack_log,
        log_inverse_rate as usize,
        &stack_claims,
    )?;
    transcript.finish()?;
    Ok(Summary {
        log_memory: layout.log_memory,
        table_rows,
        log_inv_rate: log_inverse_rate as usize,
        truncation,
        stack_log: layout.stack_log,
        bytecode_point: bus.bytecode_point,
        keccak_scalars,
    })
}
