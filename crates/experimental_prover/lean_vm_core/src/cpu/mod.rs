//! Whole-program assembly over GF(2^64) (`doc/leanvm/main.tex`): the instruction tables
//! sharing the state / memory / bytecode buses, bound to one field-valued
//! commitment and verified oracle-free. Addresses, the program counter, and read
//! counts are g-powers, so every increment is a free ×g. Machine-word arithmetic
//! is over `E = F128 = K[y]/(y²+x·y+1)` (XOR degree 1, MUL_NATIVE degree 2),
//! with each word carried by two committed `K = F64` limbs. `KECCAK` adds the
//! memory/state/bytecode plumbing for one Keccak-f sponge step whose relation
//! is discharged by the Keccak reduction (see [`crate::keccak`]). All
//! challenges and transcript scalars live in the same tower E.

use crate::constraints;
use crate::leaf::{self, Block, ColumnClaim, Coord};
use crate::pcs;
use crate::tables::{self, FillCtx, FlushBuilder, SEP_BYTECODE, SEP_MEM, SEP_STATE};
use crate::transcript::ProverState;
use crate::witness;
use primitives::field::{F64, F128, g_pow};
use verifier::layout::{BYTECODE_FINAL_COUNTERS, MEMORY_0, MEMORY_1, MEMORY_FINAL_COUNTERS, NUM_GLOBAL_COLUMNS};

mod execute;
#[cfg(test)]
pub(crate) mod forge;
pub mod hints;
pub mod layout;
mod segments;
mod trace;
mod truncate;
pub(crate) use execute::Cell;
pub use execute::{ExecError, Execution, Fault, Memory, execute};
use isa::Program;
pub use isa::{DerefMode, Op, OutCells, Stats, ZeroScale};
pub use layout::*;
pub(crate) use trace::{Drow, Jrow, Krow, Rows, Srow, Trace, Trow, Xrow};
pub use truncate::piece_digits;
pub use verifier::qt_pad;

/// Announce the prover's sizes (`log_mem`, every table's row count, the PCS rate) by
/// writing them onto the scalar stream, which binds them into the state and lets the
/// verifier reconstruct the layout, every table's height following from its rows
/// ([`layout::table_logs`]). The public statement (program + input) is not announced
/// here; it seeds the transcript at construction (the program's Fiat-Shamir IV). The boundary
/// states are derived from the program, so they need no binding.
///
/// A truncated proof (`truncate` holds its memory count) packs `(mem_count << 2) | 1` in the rate word's high limb.
/// A full proof leaves it zero. q_keccak is pieces in both.
fn announce_public(
    ps: &mut ProverState,
    log_mem: usize,
    rows: [usize; tables::N_TABLES],
    log_inv_rate: usize,
    truncate: Option<usize>,
) {
    ps.add_scalar(F128::new(log_mem as u64, 0));
    for n in rows {
        ps.add_scalar(F128::new(n as u64, 0));
    }
    let memory = truncate.map_or(0, |mem_count| ((mem_count as u64) << 2) | 1);
    ps.add_scalar(F128::new(log_inv_rate as u64, memory));
}

pub use verifier::Proof;

/// Why [`prove`] produced no proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProveError {
    /// `log_inv_rate` is outside the range the WHIR configuration accepts (1, 2, 3 or 4).
    InvalidRate { log_inv_rate: usize },
    /// The program has no execution on this input and advice: a failed `assert`,
    /// a wild pointer, advice that does not fit, ...
    Execution(ExecError),
    /// The run read these cells before anything wrote them, so their values would be
    /// the prover's ([`Execution::unconstrained_reads`]).
    UnconstrainedReads(Vec<u32>),
    /// The committed witness is `2^log_committed` words, outside the
    /// `2^MIN_STACKED_LOG..=2^MAX_STACKED_LOG` every verifier accepts
    WitnessOutOfRange { log_committed: usize },
}

impl std::fmt::Display for ProveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRate { log_inv_rate } => write!(f, "log_inv_rate {log_inv_rate} is not supported"),
            Self::Execution(e) => write!(f, "{e}"),
            Self::UnconstrainedReads(cells) => write!(
                f,
                "the program read {} cell(s) before anything wrote them, first at {:?}: a source read \
                 of a cell nothing stores, or a store the lowering dropped or misplaced",
                cells.len(),
                &cells[..cells.len().min(8)]
            ),
            Self::WitnessOutOfRange { log_committed } => write!(
                f,
                "the committed witness would be 2^{log_committed} words, outside the verifiable 2^{}..=2^{}",
                pcs::MIN_STACKED_LOG,
                pcs::MAX_STACKED_LOG
            ),
        }
    }
}

impl std::error::Error for ProveError {}

/// Per side, which table (if any) owns each bus block, as `(table, column base)`.
type BlockOwners = [Vec<Option<(usize, usize)>>; 3];
/// Each table's `(column base, committed column count)` in the global schema.
type TableSpans = Vec<(usize, usize)>;

/// Blocks sourced from a table's height belong to it; the boundary, memory and
/// bytecode blocks belong to none and keep their own column claims at ζ.
fn block_owners(log_bytecode: usize, sides: [usize; 3]) -> BlockOwners {
    let sch = schema();
    let src = block_kappa_sources(log_bytecode);
    let mut it = src
        .into_iter()
        .map(|(source, _)| source.checked_sub(2).map(|t| (t, sch.base[t])));
    sides.map(|n| it.by_ref().take(n).collect())
}

/// The bus's public wiring: per side which table owns each block, and each table's
/// column span. Derived from the program and the announced layout alone, so a
/// verifier builds the same one.
fn bus_wiring(program: &Program, l: &Layout) -> (BlockOwners, TableSpans) {
    let owners = block_owners(
        crate::log2_strict_usize(program.prog.len()),
        [l.push.len(), l.pull.len(), l.count.len()],
    );
    (owners, table_spans())
}

/// The table sumcheck carries every committed column of a table, because its bus
/// forms reference the flushed ones and its constraint the rest.
fn table_spans() -> TableSpans {
    let sch = schema();
    tables::tables()
        .iter()
        .enumerate()
        .map(|(t, tb)| (sch.base[t], tb.n_committed_columns()))
        .collect()
}

/// The per-table inputs to the table sumcheck (§constraints), in schema order: each
/// table's committed columns and its summand as one form over them (its identities
/// weighted by its own `ξ`-range, its bus forms, one per tree, by the shared powers at
/// [`xi_form_base`]).
fn airs<'a>(
    taus: &[usize; tables::N_TABLES],
    rows: &[usize; tables::N_TABLES],
    bus: &leaf::BusProof,
    cols: &'a [&'a [F64]],
    spans: &[(usize, usize)],
    xi: F128,
) -> Vec<constraints::Air<'a>> {
    let base = xi_form_base();
    let pows = primitives::field::powers(xi, n_xi_powers());
    let offsets = constraints::xi_offsets(tables::tables().iter().map(|t| t.n_constraints()));
    tables::tables()
        .iter()
        .zip(spans)
        .enumerate()
        .map(|(t, (&table, &(start, n)))| {
            let identities = table.constraint(&pows[offsets[t]..][..table.n_constraints()]);
            let bus_forms = (0..verifier::N_BUS_TREES).map(|s| bus.forms[s][t].scaled(pows[base + s]));
            constraints::Air {
                tau: taus[t],
                rows: rows[t],
                padding: table.padding(),
                cols: &cols[start..start + n],
                form: leaf::BusForm::sum(bus_forms.chain([identities])),
                public: table.public_columns().into_iter().map(|(column, _)| column).collect(),
            }
        })
        .collect()
}

/// Where the bus forms sit in the batch's `ξ`-powers: the last [`verifier::N_BUS_TREES`], one
/// per tree, AFTER every table's identity range, and shared by all tables rather than
/// one set per table. That sharing is what keeps the batch tied to the bus: with a common
/// `ξ^{base+s}` per tree, the batch's target is `Σ_s ξ^{base+s}·R_s` for
/// the trees' table shares `R_s`, which the verifier DERIVES from the leaf claims
/// (a mismatch fails its table sumcheck's terminal check). Were the
/// powers per table, the target
/// would not factor through the `R_s` and nothing would pin the tables' share of
/// the bus.
pub fn xi_form_base() -> usize {
    tables::tables().iter().map(|t| t.n_constraints()).sum()
}

/// The batch's `ξ`-powers: one per identity, then one per bus tree. Its error is their count less one over `|E|`.
pub fn n_xi_powers() -> usize {
    xi_form_base() + verifier::N_BUS_TREES
}

/// The hash tables' claims on their output lanes, which are the Keccak reduction's claim on
/// the permutations' output: the batch claims every column of a table at one point, a prefix of the point of the
/// tallest, and `keccak64_rest` holds `KECCAK64`'s lanes the table does not claim, at its own point.
///
/// A table shorter than the tallest is completed with its padding step, so its claim at the tallest's point is
/// `eq(0) * claim + (1 - eq(0)) * padding` for `eq(0)` the product of `1 - rho_i` over the completed variables.
fn keccak_output_claim(
    table_claims: &[constraints::Claims],
    taus: &[usize; tables::N_TABLES],
    keccak64_rest: &[F128; KECCAK64_SENT_LANES],
) -> keccak_gkr::OutputClaim {
    let kappa = layout::hash_kappa(taus);
    let tallest = verifier::layout::HASH_TABLES
        .iter()
        .find(|&&t| taus[t] == kappa)
        .expect("a tallest hash table");
    let rho = table_claims[*tallest].chi[..kappa].to_vec();
    let scalars = |values: &[F128]| values.iter().map(to_scalar).collect::<Vec<_>>();
    let evals = verifier::layout::HASH_TABLES.map(|t| scalars(&table_claims[t].evals));
    let heights = verifier::layout::HASH_TABLES.map(|t| taus[t]);
    let lanes = verifier::hash_output_claims(
        &scalars(&rho),
        heights,
        evals.each_ref().map(Vec::as_slice),
        &scalars(keccak64_rest),
    );
    keccak_gkr::OutputClaim {
        instance: rho,
        lanes: lanes.map(|half| half.map(from_scalar)).to_vec(),
    }
}

/// The lanes of `KECCAK64`'s output the table's claims leave out: 4 to 24.
const KECCAK64_SENT_LANES: usize = primitives::hash::STATE_LANES - tables::keccak64::OUT_LANES;

/// `KECCAK64`'s output lanes 4 to 24 as a multilinear in its steps, at `point`, from the input lanes `q_keccak` holds:
/// the values the prover sends before the reduction folds the tables' claims.
fn keccak64_output_lanes(q_keccak: &[F64], kappa: usize, tau: usize, point: &[F128]) -> [F128; KECCAK64_SENT_LANES] {
    use primitives::field::F128Unreduced;
    let eq = primitives::multilinear::eq_table(point);
    let starts: [usize; primitives::hash::STATE_LANES] =
        std::array::from_fn(|lane| layout::qkeccak_slot_start(kappa, crate::keccak::KECCAK64_HALF, lane));
    let sums = parallel::fold_reduce(
        1 << tau,
        || [F128Unreduced::ZERO; KECCAK64_SENT_LANES],
        |acc, row| {
            let mut state: [u64; primitives::hash::STATE_LANES] =
                std::array::from_fn(|lane| q_keccak[starts[lane] + row].0);
            primitives::hash::keccak_f(&mut state);
            for (acc, &word) in acc.iter_mut().zip(&state[tables::keccak64::OUT_LANES..]) {
                *acc ^= eq[row].mul_base_unreduced(F64(word));
            }
        },
        |mut a, b| {
            for (a, b) in a.iter_mut().zip(b) {
                *a ^= b;
            }
            a
        },
    );
    sums.map(F128Unreduced::reduce)
}

/// Prove the program on the given public input and witness ([`execute`]): run it (witness generation),
/// then emit everything the verifier needs through the returned [`Proof`]
/// (scalar stream + PCS commitment / opening hints). Returns the proof and the
/// run [`Stats`], or why no proof was made ([`ProveError`]). `log_inv_rate`
/// selects the PCS rate and is announced in the Fiat-Shamir transcript before
/// the commitment.
#[tracing::instrument(name = "Prove", skip_all, fields(log_inv_rate))]
pub fn prove(
    program: &Program,
    witness: &[(String, Vec<Vec<F128>>)],
    public_input: [F128; 2],
    log_inv_rate: usize,
) -> Result<(Proof, Stats), ProveError> {
    if ::pcs::whir::validate_log_inv_rate(log_inv_rate).is_err() {
        return Err(ProveError::InvalidRate { log_inv_rate });
    }
    let (exec, keccak_blocks) = crate::stage!("Execute program", || {
        crate::keccak::while_executing(|step| execute::execute_with(program, witness, public_input, step))
    });
    let exec = exec.map_err(ProveError::Execution)?;
    // A live value that came from outside the constraint system would make the proof
    // about a weaker statement than the program text, so it is refused here, on the one
    // path every proof takes, rather than left to whichever test happens to look.
    if !exec.unconstrained_reads.is_empty() {
        return Err(ProveError::UnconstrainedReads(exec.unconstrained_reads));
    }
    // The layout follows from the run's sizes alone: a run past the window is refused
    // before its witness is built.
    let layout = layout_of(
        crate::log2_strict_usize(program.prog.len()),
        crate::log2_strict_usize(exec.mem.len()),
        exec.trace.row_counts(),
        public_input,
    );
    // q_keccak is pieces in every proof. A truncated program also cuts the memory and the table columns, unless that
    // leaves the stack below what a recursion guest opens: then it keeps them whole, as a full proof does.
    let truncated = if program.truncate {
        let floor = verifier::min_execution_stacked_log(crate::log2_strict_usize(program.prog.len()));
        truncate::Compact::reaching(&layout, exec.mem_used, floor)?
    } else {
        None
    };
    let compact = match truncated {
        Some(compact) => compact,
        None => truncate::Compact::new(&layout, 1 << layout.placements[MEMORY_FINAL_COUNTERS].n_vars, false)?,
    };
    let shape = compact.shape;
    let cycles = exec.cycles;
    let mut w = crate::stage!("Build witness", || build(program, &exec, layout, keccak_blocks));
    #[cfg(test)]
    forge::tamper(&mut w);
    let mem_used = exec.mem_used;
    drop(exec);
    let committed_size = compact.placed;
    // The public statement (program digest + input) seeds the transcript, so
    // every challenge depends on the exact program and public input.
    let mut ps = ProverState::new(
        verifier::Digest(program.bytecode().fiat_shamir_iv()).words(),
        verifier::Digest(public_input_bytes(&public_input)).words(),
    );

    // Announce sizes and commit before sampling any challenge.
    announce_public(&mut ps, w.log_mem, w.layout.rows, log_inv_rate, compact.announced());
    let compact_q = crate::stage!("Compact copy", || compact.copy(&w.q, &w.layout));
    #[cfg(test)]
    let compact_q = {
        let mut q = compact_q;
        forge::tamper_compact(&compact, &mut q);
        q
    };
    let committed = crate::stage!("Commit", || {
        pcs::commit(&mut ps, &compact_q[..shape.committed_len()], shape, log_inv_rate)
    });

    // KECCAK (§keccak), single PCS: q_keccak is ALWAYS a column in `w.q` (the
    // KECCAK table is filled to the reduction's floor, so the proof shape is uniform
    // and there is no has/hasn't-KECCAK fork). The Keccak reduction's claims and
    // EVERY leanVM point claim are discharged together by ONE
    // WHIR over this commitment (below). Input and output lanes bind through the
    // memory bus: their virtual value columns route to q_keccak (inputs) and to the
    // reduction's output claim (outputs), so no separate pin claims are needed.
    let (owners, spans) = bus_wiring(program, &w.layout);
    // The columns are windows into `w.q`, so both stages read them in place: the
    // table sumcheck builds a table's few `E` state columns from them on the round
    // it joins and never writes the K-columns back.
    let (bus, table_claims) = {
        let l = &w.layout;
        let cols = w.columns();
        let bus = crate::stage!("Prove bus", || {
            leaf::prove_balance(
                &l.push,
                &l.pull,
                &l.count,
                &cols,
                program.stacked_bytecode(),
                &owners,
                &spans,
                &mut ps,
            )
        });
        let table_claims = crate::stage!("Prove constraints", || {
            // One sumcheck for all six tables (§constraints).
            // The eq point is the bus GKR's ζ, not a fresh one: that is what lets the
            // batch settle the bus forms alongside the constraints.
            ps.grind(fiat_shamir::grinding_bits(n_xi_powers() as u128 - 1));
            let xi = ps.sample();
            constraints::prove(&airs(&l.taus, &l.rows, &bus, &cols, &spans, xi), &bus.point, &mut ps)
        });
        (bus, table_claims)
    };
    // The bus and the constraints were the output lanes' last readers.
    drop(std::mem::take(&mut w.virt));
    let l = &w.layout;

    // The PI binding (§sec:e2e-pi) transmits nothing: its claims are the public
    // input's limb lines at `r_pi`, which the verifier evaluates itself. `r_pi`
    // is still a challenge, squeezed after the commitment so the memory cannot
    // be chosen to fit it. Its claims are lines in it, numerator one: no grind.
    let r_pi = ps.sample();
    // Memory binds every step's input and output words. Their value columns are
    // virtual: the inputs route to q_keccak through `slot_claims`, the outputs are
    // the reduction's claim on the permutation's output.
    let mut slots = finish_claims(l, bus.claims, &table_claims, r_pi);

    // Run the Keccak reduction over the witness rebuilt from q_keccak's input
    // states; it returns two claims on the committed `q_keccak`, discharged by the
    // PCS below in the SAME WHIR as every leanVM point claim: one ring-switched,
    // one appended to the point claims.
    let keccak_reduction = w.keccak_reduction;
    let witness::Placement { offset, n_vars } = w.layout.placements[QKECCAK];
    let q_keccak = &w.q[offset..][..1 << n_vars];
    // `KECCAK64`'s table claims only its output lanes 0 to 3; the others are sent at its own point, before the
    // reduction draws the challenge that folds the tables.
    let keccak64_rest = {
        let kappa = layout::hash_kappa(&w.layout.taus);
        let tau = w.layout.taus[verifier::layout::OP_KECCAK64];
        #[cfg_attr(not(test), allow(unused_mut))]
        let mut rest = keccak64_output_lanes(q_keccak, kappa, tau, &table_claims[verifier::layout::OP_KECCAK64].chi);
        #[cfg(test)]
        forge::tamper_sent(&mut rest);
        ps.add_scalars(&rest);
        rest
    };
    let output = keccak_output_claim(&table_claims, &w.layout.taus, &keccak64_rest);
    let openings = crate::stage!("Keccak reduction", || {
        keccak_reduction.prove(&mut ps, q_keccak, &output)
    });
    let (ring, outputs) = openings.into_pcs_claims(offset);
    slots.extend(outputs);
    // The reduction was the last reader of the full stack: the proof opens the compact one.
    drop(std::mem::take(&mut w.q));
    let (slots, ring) = compact.map_claims(&w.layout, slots, ring);
    crate::stage!("PCS open", || {
        pcs::open(&mut ps, committed, compact_q, &slots, &ring)
    });
    Ok((
        ps.into_proof(),
        Stats {
            cycles,
            rows: w.layout.rows,
            committed: committed_size,
            log_mem: w.log_mem,
            mem_used,
        },
    ))
}

/// Everything the PCS has to open, in the ORDER that feeds the batch's weights:
/// the bus's framework claims, then the table sumcheck's per-table column claims (less the
/// `KECCAK` output lanes', which go to the reduction), then the two public-input limb
/// claims, each located in its committed slot. Every verifier assembles the same list in
/// the same order.
fn finish_claims(
    l: &Layout,
    bus_claims: Vec<ColumnClaim>,
    table_claims: &[constraints::Claims],
    r_pi: F128,
) -> Vec<pcs::StackClaim> {
    let mut claims = bus_claims;
    let sch = schema();
    claims.reserve(sch.n - NUM_GLOBAL_COLUMNS);
    for (t, table) in tables::tables().iter().enumerate() {
        let public = table.public_columns();
        for c in 0..table.n_committed_columns() {
            let col = sch.base[t] + c;
            if let Some((_, KeccakLane::Output(_))) = keccak_value(col) {
                continue; // the Keccak reduction's claim on the output (`keccak_output_claim`)
            }
            if public.iter().any(|&(p, _)| p == c) {
                continue; // a constant on every row: no claim, the verifier knows its value
            }
            claims.push(ColumnClaim {
                col,
                point: table_claims[t].chi.clone(),
                value: table_claims[t].evals[c],
            });
        }
    }
    claims.extend(bind_pi_claim(r_pi, l));
    slot_claims(l, claims)
}

/// The public-input binding (§sec:e2e-pi): each committed `MEM` limb at
/// `(r, 0,…,0)` must equal the line through that limb of `pi[0]` and `pi[1]`.
/// The public input is the statement, so both sides evaluate the lines
/// themselves and build byte-identical claims from their layouts; the prover
/// sends nothing.
fn bind_pi_claim(r: F128, l: &Layout) -> [ColumnClaim; 2] {
    let [a, b] = l.pi;
    let limbs = [(a.c0, b.c0), (a.c1, b.c1)];
    let mut point = vec![F128::ZERO; l.placements[MEMORY_0].n_vars];
    point[0] = r;
    [MEMORY_0, MEMORY_1].map(|col| {
        let (x, y) = limbs[col - MEMORY_0];
        ColumnClaim {
            col,
            point: point.clone(),
            value: primitives::multilinear::interp_k(F64(x), F64(y), r),
        }
    })
}

/// A field element in the `verifier` crate's type, and back: the same limbs, no arithmetic.
pub fn to_scalar(value: &F128) -> verifier::E {
    verifier::E::new(value.c0, value.c1)
}

pub fn from_scalar(value: verifier::E) -> F128 {
    F128::new(value.c0, value.c1)
}

/// The statement as the 32 bytes the `verifier` crate takes: the two limbs of each half.
pub fn public_input_bytes(public_input: &[F128; 2]) -> [u8; 32] {
    verifier::Digest::from_halves(to_scalar(&public_input[0]), to_scalar(&public_input[1])).0
}

/// Verify a proof against the public statement (program + public input). The `verifier` crate does it: this only
/// builds its inputs.
#[tracing::instrument(name = "Verify", skip_all)]
pub fn verify(
    program: &Program,
    public_input: &[F128; 2],
    proof: &Proof,
) -> Result<verifier::Summary, verifier::VerifierError> {
    verifier::verify_execution(program.bytecode(), &public_input_bytes(public_input), proof)
}

/// Lift `ColumnClaim`s to located PCS claims: a claim on column `c` lives in
/// the slot at `placements[c].offset`, with the claim's point as the low point.
///
/// The columns `q_keccak` carries are virtual: they have no committed placement. A claim
/// `col(r) = v` on one (at the `n_log`-dim instance point `r`) is re-routed to the equal
/// `q_keccak` slot evaluation: an ordinary claim on the committed `QKECCAK` column at the
/// point whose low coords are `r` and whose next `SLOT_LOG` are the slot's bits. No
/// downstream special-casing: it folds into the one opening like every other point claim.
fn slot_claims(l: &Layout, claims: Vec<ColumnClaim>) -> Vec<pcs::StackClaim> {
    let kappa = layout::hash_kappa(&l.taus);
    claims
        .into_iter()
        .map(|c| {
            // A hash table's value column (always virtual): its bus claim at instance point
            // `c.point` is a claim on the first `2^tau` words of its slot of its half of QKECCAK.
            if let Some((half, KeccakLane::Slot(slot))) = keccak_value(c.col) {
                debug_assert_eq!(c.point.len(), l.taus[verifier::layout::HASH_TABLES[half]]);
                return pcs::StackClaim::Point {
                    offset: l.placements[QKECCAK].offset + layout::qkeccak_slot_start(kappa, half, slot),
                    low_point: c.point,
                    value: c.value,
                };
            }
            pcs::StackClaim::Point {
                offset: l.placements[c.col].offset,
                low_point: c.point,
                value: c.value,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use verifier::K;

    /// `SET` of a prover word.
    fn set(o: u32, k: F128) -> Op {
        Op::Set { o, k: to_scalar(&k) }
    }

    /// A K-embedded immediate (extension limb zero).
    fn w(x: u64) -> F128 {
        F128::new(x, 0)
    }

    /// A hand-assembled program: no hints, no debug information, started at `(pc, fp) = (0, 0)`. Straight-line
    /// programs that never change the frame pointer and touch only the first `main_frame` cells need no
    /// allocation. It halts on reaching the final state `(g^{len-1}, g^0)`, the last slot in main's frame
    /// (§sec:state).
    fn from_bytecode(prog: Vec<Op>, main_frame: u32) -> Program {
        Program::from(isa::CompiledProgram {
            prog,
            hints: Default::default(),
            main_frame,
            fn_ranges: Vec::new(),
            src_lines: Vec::new(),
        })
    }

    /// A hand-built straight-line program with two KECCAK rows. The first reads, in the even
    /// layout, two message cells (2, 3) twice as `m`, a four-cell `tail` (4..8), `b8` (8) and a
    /// zero `cap` (9..13), and writes 13..26 in the odd layout; the second reads that output in
    /// the odd layout and writes 26..39 in the even one. Two filler SETs step the pc to 15, whose
    /// slot is the never-executed sentinel.
    fn keccak_program(a: F128, b: F128, tail: [F128; 4], b8: F128) -> Program {
        let mut prog = vec![set(2, a), set(3, b)];
        for (i, &t) in tail.iter().enumerate() {
            prog.push(set(4 + i as u32, t));
        }
        prog.push(set(8, b8));
        for c in 0..4u32 {
            prog.push(set(9 + c, F128::ZERO));
        }
        prog.push(Op::Keccak {
            m: [2, 3, 2, 3],
            tail: 4,
            b8: 8,
            cap: 9,
            out: 13,
            in_odd: false,
            out_odd: true,
            out_cells: OutCells::All,
        });
        prog.push(Op::Keccak {
            m: [13, 14, 15, 16],
            tail: 17,
            b8: 21,
            cap: 22,
            out: 26,
            in_odd: true,
            out_odd: false,
            out_cells: OutCells::All,
        });
        for k in 0..2u32 {
            prog.push(set(39 + k, F128::ONE));
        }
        prog.push(Op::Xor { a: 0, b: 0, c: 0 }); // sentinel
        assert_eq!(prog.len(), 16);
        from_bytecode(prog, 64)
    }

    /// Two KECCAK rows over one input that write only part of their output: the first its cell 0 at 13, the
    /// second its cells 0 and 1 at 26. An XOR reads cells 13 and 27 into 40.
    fn partial_keccak_program() -> Program {
        let mut prog = vec![set(2, w(3)), set(3, w(4))];
        for c in 4..13u32 {
            prog.push(set(c, w(u64::from(c))));
        }
        for (out, out_cells) in [(13, OutCells::One), (26, OutCells::Two)] {
            prog.push(Op::Keccak {
                m: [2, 3, 2, 3],
                tail: 4,
                b8: 8,
                cap: 9,
                out,
                in_odd: false,
                out_odd: false,
                out_cells,
            });
        }
        prog.push(Op::Xor { a: 13, b: 27, c: 40 });
        prog.push(set(41, F128::ONE));
        prog.push(Op::Xor { a: 0, b: 0, c: 0 }); // sentinel
        assert_eq!(prog.len(), 16);
        from_bytecode(prog, 64)
    }

    /// Every committed column holds its table's padding value in every row past the table's rows: the table
    /// sumcheck takes those rows from [`tables::Table::padding`] and never reads them.
    #[test]
    fn padding_rows_hold_the_padding_values() {
        let a = F128::new(0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210);
        let pad = F128::new(primitives::hash::PAD_FIRST as u64, 0);
        let end = F128::new(primitives::hash::END_BIT, 0x5555);
        let programs = [
            keccak_program(a, w(5), [pad, F128::ZERO, F128::ZERO, F128::ZERO], end),
            partial_keccak_program(),
            keccak64_mix_program([a, w(1), w(2), w(3)], 3, 2, true, false).0,
            keccak64_mix_program([a, w(1), w(2), w(3)], 5, 0, false, false).0,
        ];
        let pi = [w(7), w(11)];
        for program in programs {
            let (exec, blocks) = crate::keccak::while_executing(|step| execute::execute_with(&program, &[], pi, step));
            let exec = exec.unwrap();
            let layout = layout_of(
                crate::log2_strict_usize(program.prog.len()),
                crate::log2_strict_usize(exec.mem.len()),
                exec.trace.row_counts(),
                pi,
            );
            let witness = build(&program, &exec, layout, blocks);
            let cols = witness.columns();
            for (t, table) in tables::tables().iter().enumerate() {
                let (start, n) = (schema().base[t], table.n_committed_columns());
                let (rows, padding) = (witness.layout.rows[t], table.padding());
                for c in 0..n {
                    assert!(
                        cols[start + c][rows..].iter().all(|&v| v == padding[c]),
                        "table {t} column {c}"
                    );
                }
            }
        }
    }

    /// A KECCAK writes the output cells its bytecode requests and no other, and the proof verifies.
    #[test]
    fn keccak_writes_only_requested_cells() {
        use crate::keccak::permute_cells;
        let program = partial_keccak_program();
        let pi = [w(7), w(11)];
        let exec = execute(&program, &[], pi).unwrap();
        let input = std::array::from_fn(|c| exec.mem.word(if c < 4 { 2 + c % 2 } else { c }));
        let out = permute_cells(&input, false, false);
        assert_eq!(exec.mem.word(13), out[0]);
        assert_eq!([exec.mem.word(26), exec.mem.word(27)], [out[0], out[1]]);
        assert!(
            (14..26).chain(28..39).all(|c| exec.mem.word(c) == F128::ZERO),
            "a cell not requested is not written"
        );
        let (proof, _) = prove(&program, &[], pi, crate::pcs::TEST_LOG_INV_RATE).unwrap();
        verify(&program, &pi, &proof).expect("partial outputs verify");
    }

    /// The opcode's execution semantics: the output cells hold the permutation of the lanes the
    /// input cells hold, each side in its layout, the two `m` pairs aliasing one another and
    /// `b8`'s free high limb ignored. With a zero `cap` and the padding in `tail` and `b8`, the
    /// first permutation's first four lanes are the hash of the 64 message bytes. Hand-assembled,
    /// the program is provable all the same: its tables are not powers of two.
    #[test]
    fn keccak_permutes_in_both_layouts() {
        use crate::keccak::{STATE_CELLS, permute_cells, state_of_cells};
        let a = F128::new(0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210);
        let b = F128::new(0xdead_beef_cafe_babe, 0x0bad_f00d_0bad_f00d);
        let pad = F128::new(primitives::hash::PAD_FIRST as u64, 0);
        let end = F128::new(primitives::hash::END_BIT, 0x5555);
        let program = keccak_program(a, b, [pad, F128::ZERO, F128::ZERO, F128::ZERO], end);
        let pi = [w(7), w(11)];
        let exec = execute(&program, &[], pi).unwrap();

        let mut input = [F128::ZERO; STATE_CELLS];
        input[..9].copy_from_slice(&[a, b, a, b, pad, F128::ZERO, F128::ZERO, F128::ZERO, end]);
        let out = permute_cells(&input, false, true);
        let words = |at: usize| -> Vec<F128> { (at..at + STATE_CELLS).map(|i| exec.mem.word(i)).collect() };
        assert_eq!(words(13), out);
        assert_eq!(words(26), permute_cells(&out, true, false));

        let bytes: Vec<u8> = [a, b, a, b]
            .iter()
            .flat_map(|c| [c.c0, c.c1])
            .flat_map(u64::to_le_bytes)
            .collect();
        assert_eq!(
            primitives::hash::digest_of(&state_of_cells(true, |c| out[c])),
            primitives::hash::hash(&bytes)
        );

        let (proof, _) = prove(&program, &[], pi, crate::pcs::TEST_LOG_INV_RATE).unwrap();
        verify(&program, &pi, &proof).expect("a hand-assembled program verifies");
    }

    /// A straight-line program of `keccak64` `KECCAK64` rows and `keccak` `KECCAK` rows on the same four message cells (2..6),
    /// the `KECCAK`s with the constants a 64-byte message leaves in `tail` (6..10), `b8` (10) and `cap` (11..15).
    /// Row `i` of `KECCAK64` writes cell `base + 2 i` (and the one after when `out_two`), and of `KECCAK` cell `kbase + 13 i`,
    /// two cells. The last slot is the sentinel.
    fn keccak64_mix_program(
        message: [F128; 4],
        keccak64: usize,
        keccak: usize,
        out_two: bool,
        truncate: bool,
    ) -> (Program, u32, u32) {
        keccak64_mix_program_with_cap(message, F128::ZERO, keccak64, keccak, out_two, truncate)
    }

    /// [`keccak64_mix_program`] whose `KECCAK` rows hold `cap` in their first capacity cell: not a message `KECCAK64` could hash.
    fn keccak64_mix_program_with_cap(
        message: [F128; 4],
        cap: F128,
        keccak64: usize,
        keccak: usize,
        out_two: bool,
        truncate: bool,
    ) -> (Program, u32, u32) {
        let plain = (F64::ONE, F64::ZERO, F64::ZERO);
        keccak64_mix_program_with_immediates(message, cap, keccak64, keccak, out_two, truncate, plain)
    }

    /// [`keccak64_mix_program_with_cap`] whose `KECCAK64` rows carry the immediates `(s_a, k_a, k_y)`: they hash another message
    /// than the `KECCAK` rows do, unless the immediates are `(1, 0, 0)`.
    fn keccak64_mix_program_with_immediates(
        message: [F128; 4],
        cap: F128,
        keccak64: usize,
        keccak: usize,
        out_two: bool,
        truncate: bool,
        (s_a, k_a, k_y): (F64, F64, F64),
    ) -> (Program, u32, u32) {
        let (base, kbase) = (64u32, 64 + 2 * keccak64 as u32 + 8);
        let mut prog = Vec::new();
        for (c, &m) in message.iter().enumerate() {
            prog.push(set(2 + c as u32, m));
        }
        prog.push(set(6, F128::new(primitives::hash::PAD_FIRST as u64, 0)));
        for o in [7, 8, 9] {
            prog.push(set(o, F128::ZERO));
        }
        prog.push(set(10, F128::new(primitives::hash::END_BIT, 0x5555)));
        for o in 11..15 {
            prog.push(set(o, if o == 11 { cap } else { F128::ZERO }));
        }
        // Odd rows hash the message rotated by a cell, held in 16..20, so that rows have different outputs.
        for (c, o) in (16..20u32).enumerate() {
            prog.push(set(o, message[(c + 1) % 4]));
        }
        for i in 0..keccak64 as u32 {
            prog.push(
                Op::keccak64(
                    if i % 2 == 0 { [2, 3, 4, 5] } else { [16, 17, 18, 19] },
                    base + 2 * i,
                    out_two,
                    verifier::K(s_a.0),
                    verifier::K(k_a.0),
                    verifier::K(k_y.0),
                )
                .expect("a nonzero scale"),
            );
        }
        for i in 0..keccak as u32 {
            prog.push(Op::Keccak {
                m: [2, 3, 4, 5],
                tail: 6,
                b8: 10,
                cap: 11,
                out: kbase + 13 * i,
                in_odd: false,
                out_odd: false,
                out_cells: OutCells::Two,
            });
        }
        let frame = kbase + 13 * keccak as u32 + 16;
        // Filler SETs, then the sentinel in the last slot.
        let len = (prog.len() + 2).next_power_of_two();
        let mut spare = frame;
        while prog.len() < len - 1 {
            prog.push(set(spare, F128::ONE));
            spare += 1;
        }
        prog.push(Op::Xor { a: 0, b: 0, c: 0 });
        let mut program = from_bytecode(prog, spare + 1);
        program.truncate = truncate;
        (program, base, kbase)
    }

    /// `KECCAK64` is the `KECCAK` of its four cells with the padding in the others: the same output cells, for both values
    /// of `out_two`, on random messages, and a cell it does not write stays unwritten.
    #[test]
    fn keccak64_equals_keccak_on_the_same_cells() {
        let mut state = 0x1234_5678_9abc_def1u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let pi = [w(7), w(11)];
        for round in 0..8 {
            let message: [F128; 4] = std::array::from_fn(|_| F128::new(next(), next()));
            let out_two = round % 2 == 0;
            let (program, base, kbase) = keccak64_mix_program(message, 3, 1, out_two, false);
            let exec = execute(&program, &[], pi).unwrap();
            assert_eq!(exec.rows()[verifier::layout::OP_KECCAK64], 3);
            let (k0, k1) = (exec.mem.word(kbase as usize), exec.mem.word(kbase as usize + 1));
            // Row 1 hashes the rotated message.
            assert_ne!(exec.mem.word(base as usize + 2), k0);
            for i in [0usize, 2] {
                assert_eq!(exec.mem.word(base as usize + 2 * i), k0, "cell 0 of row {i}");
                let second = exec.mem.word(base as usize + 2 * i + 1);
                assert_eq!(second, if out_two { k1 } else { F128::ZERO }, "cell 1 of row {i}");
            }
            let mut input = [F128::ZERO; crate::keccak::STATE_CELLS];
            input[..4].copy_from_slice(&message);
            let bytes: Vec<u8> = message
                .iter()
                .flat_map(|c| [c.c0, c.c1])
                .flat_map(u64::to_le_bytes)
                .collect();
            let digest = primitives::hash::hash(&bytes);
            let cells: Vec<u8> = [k0, k1]
                .iter()
                .flat_map(|c| [c.c0, c.c1])
                .flat_map(u64::to_le_bytes)
                .collect();
            assert_eq!(cells, digest);
        }
    }

    /// `KECCAK64` with the immediates `(g^(-n), c g^(-n), d)` is the `KECCAK` of `P`, `A g^n + c`, `X`, `Y + d`, the two cells
    /// made by `MUL` and `XOR`: random cells, shifts 0 to 7, random immediates, both values of `out_two`. `A` is
    /// K-valued where it is shifted, as the compiler's `a_shift` asks.
    #[test]
    fn tha_equals_keccak_on_the_shifted_cells() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let pi = [w(7), w(11)];
        for n in 0..8usize {
            for out_two in [false, true] {
                let mut message: [F128; 4] = std::array::from_fn(|_| F128::new(next(), next()));
                if n > 0 {
                    message[1] = F128::new(next(), 0);
                }
                let (c, d) = (F64(next()), F64(next()));
                let g_n = g_pow(n);
                let s_a = g_n.inv();
                let mut prog = Vec::new();
                for (cell, &m) in message.iter().enumerate() {
                    prog.push(set(2 + cell as u32, m));
                }
                // The padding a 64-byte message leaves in the cells the `KECCAK` reads past its four.
                prog.push(set(6, F128::new(primitives::hash::PAD_FIRST as u64, 0)));
                for o in [7, 8, 9] {
                    prog.push(set(o, F128::ZERO));
                }
                prog.push(set(10, F128::new(primitives::hash::END_BIT, 0x5555)));
                for o in 11..15 {
                    prog.push(set(o, F128::ZERO));
                }
                // `A g^n + c` and `Y + d` by `MUL` and `XOR`.
                for (o, k) in [(16, F128::from(g_n)), (17, F128::from(c)), (18, F128::from(d))] {
                    prog.push(set(o, k));
                }
                prog.push(Op::Mul { a: 3, b: 16, c: 19 });
                prog.push(Op::Xor { a: 19, b: 17, c: 20 });
                prog.push(Op::Xor { a: 5, b: 18, c: 21 });
                prog.push(Op::Keccak {
                    m: [2, 20, 4, 21],
                    tail: 6,
                    b8: 10,
                    cap: 11,
                    out: 40,
                    in_odd: false,
                    out_odd: false,
                    out_cells: OutCells::Two,
                });
                prog.push(
                    Op::keccak64([2, 3, 4, 5], 30, out_two, K(s_a.0), K((c * s_a).0), K(d.0)).expect("a nonzero scale"),
                );
                let len = (prog.len() + 2).next_power_of_two();
                while prog.len() < len - 1 {
                    prog.push(set(60, F128::ONE));
                }
                prog.push(Op::Xor { a: 0, b: 0, c: 0 });
                let program = from_bytecode(prog, 64);
                let exec = execute(&program, &[], pi).unwrap();
                assert_eq!(exec.rows()[verifier::layout::OP_KECCAK64], 1);
                let (k0, k1) = (exec.mem.word(40), exec.mem.word(41));
                assert_eq!(exec.mem.word(30), k0, "output cell 0, shift {n}");
                let second = exec.mem.word(31);
                assert_eq!(
                    second,
                    if out_two { k1 } else { F128::ZERO },
                    "output cell 1, shift {n}"
                );
                // The proof of the run verifies: the bus carries the shifted lane.
                let (proof, _) = prove(&program, &[], pi, crate::pcs::TEST_LOG_INV_RATE).unwrap();
                verify(&program, &pi, &proof).unwrap_or_else(|e| panic!("shift {n}: {e}"));
            }
        }
    }

    /// A `KECCAK64` with a zero scale cannot be built, and the interpreter refuses one that is built by hand, or whose
    /// stored inverse is not the scale's.
    #[test]
    fn a_keccak64_with_a_zero_scale_is_refused() {
        assert_eq!(
            Op::keccak64([2, 3, 4, 5], 6, false, K(0), K(1), K(1)).err(),
            Some(ZeroScale)
        );
        let pi = [w(7), w(11)];
        let run = |s_a: K, s_inv: K| {
            let mut prog = vec![set(2, F128::ONE)];
            prog.push(Op::Keccak64 {
                o: [2, 2, 2, 2],
                out: 6,
                out_two: false,
                s_a,
                s_inv,
                k_a: K(0),
                k_y: K(0),
            });
            while prog.len() < 3 {
                prog.push(set(8, F128::ONE));
            }
            prog.push(Op::Xor { a: 0, b: 0, c: 0 });
            execute(&from_bytecode(prog, 9), &[], pi).err().map(|e| e.fault)
        };
        assert_eq!(run(K(0), K(0)), Some(Fault::BadInstruction));
        assert_eq!(run(K(2), K(2)), Some(Fault::BadInstruction));
        assert_eq!(run(K(2), K(2).inv()), None);
    }

    /// A proof with only `KECCAK64` rows, one with only `KECCAK` rows, and ones with both at different heights, all verify.
    #[test]
    fn proofs_with_keccak64_and_keccak_rows_verify() {
        let message: [F128; 4] = std::array::from_fn(|i| F128::new(0x1111 * (i as u64 + 1), 0x7777 * (i as u64 + 3)));
        let pi = [w(7), w(11)];
        for (keccak64, keccak, out_two, truncate) in [
            (3, 0, true, false),
            (3, 0, false, false),
            (0, 3, true, false),
            (3, 3, true, false),
            (600, 20, true, false),
            (20, 600, false, false),
            (70_000, 40, true, true),
            (40, 70_000, true, true),
            (70_000, 70_000, false, true),
            // Four row pieces of `KECCAK64` (12 pieces by slot block) and two of `KECCAK`: fourteen ring blocks, more than
            // the eight an opening used to carry.
            (122_000, 98_000, true, true),
        ] {
            let (program, _, _) = keccak64_mix_program(message, keccak64, keccak, out_two, truncate);
            let (proof, stats) = prove(&program, &[], pi, crate::pcs::TEST_LOG_INV_RATE).unwrap();
            assert_eq!(
                [
                    stats.rows[verifier::layout::OP_KECCAK],
                    stats.rows[verifier::layout::OP_KECCAK64]
                ],
                [keccak, keccak64]
            );
            let summary = verify(&program, &pi, &proof)
                .unwrap_or_else(|e| panic!("{keccak64} KECCAK64 and {keccak} KECCAK rows: {e}"));
            assert_eq!(
                summary.truncation.is_some(),
                truncate,
                "{keccak64} KECCAK64 and {keccak} KECCAK rows: the rate word announces the truncation"
            );
        }
    }

    /// Each forgery of the witness is refused by the verifier: the prover makes a proof, which does not verify.
    #[test]
    fn forged_keccak64_witnesses_are_rejected() {
        let message: [F128; 4] = std::array::from_fn(|i| F128::new(0x1111 * (i as u64 + 1), 0x7777 * (i as u64 + 3)));
        let cap = F128::new(0xcafe, 0xbabe);
        // The six forgeries of `KECCAK64`, with the plain instruction and with one that shifts and adds.
        let (s_a, k_a, k_y) = (g_pow(3).inv(), F64(0x0123_4567), F64(0x31));
        for imm in [(F64::ONE, F64::ZERO, F64::ZERO), (s_a, k_a * s_a, k_y)] {
            forged_keccak64_cases(message, cap, imm);
        }
    }

    /// The forgeries of `KECCAK64`'s operands: a lane that is not what the immediates say, and an immediate column that
    /// differs from the bytecode. Each is refused with a proof the prover completes.
    #[test]
    fn forged_keccak64_operands_are_rejected() {
        use super::forge::{Forge, set};
        let message: [F128; 4] = std::array::from_fn(|i| {
            F128::new(
                0x1111 * (i as u64 + 1),
                if i == 1 { 0 } else { 0x7777 * (i as u64 + 3) },
            )
        });
        let pi = [w(7), w(11)];
        let (n, c, d) = (3usize, F64(0x0123_4567), F64(0x31));
        let s_a = g_pow(n).inv();
        let imm = (s_a, c * s_a, d);
        let (a_lo, y_lo) = (F64(message[1].c0), F64(message[3].c0));
        let honest_2 = (a_lo + imm.1) * g_pow(n);
        let (program, _, _) = keccak64_mix_program_with_immediates(message, F128::ZERO, 3, 2, true, false, imm);
        set(None);
        let (proof, _) = prove(&program, &[], pi, crate::pcs::TEST_LOG_INV_RATE).unwrap();
        verify(&program, &pi, &proof).expect("the honest proof verifies");
        let cases = [
            // Lane 2 is `A.lo g^n + c'` with another constant, and is built without the shift.
            (
                Forge::Keccak64Lane {
                    lane: 2,
                    value: (honest_2 + F64(1)).0,
                },
                "lane 2, another constant",
            ),
            (
                Forge::Keccak64Lane {
                    lane: 2,
                    value: (a_lo + c).0,
                },
                "lane 2, no shift",
            ),
            // Lane 6 is `Y.lo` without `k_y`, and carries another chain's index.
            (Forge::Keccak64Lane { lane: 6, value: y_lo.0 }, "lane 6, no offset"),
            (
                Forge::Keccak64Lane {
                    lane: 6,
                    value: (y_lo + d + F64(2)).0,
                },
                "lane 6, another index",
            ),
            // Each immediate column of a row differs from the bytecode, alone.
            (
                Forge::Keccak64Column {
                    column: tables::keccak64::S_A,
                    xor: 4,
                },
                "S_A",
            ),
            (
                Forge::Keccak64Column {
                    column: tables::keccak64::K_A,
                    xor: 4,
                },
                "K_A",
            ),
            (
                Forge::Keccak64Column {
                    column: tables::keccak64::K_Y,
                    xor: 4,
                },
                "K_Y",
            ),
        ];
        for (forge, what) in cases {
            set(Some(forge));
            let forged = prove(&program, &[], pi, crate::pcs::TEST_LOG_INV_RATE);
            set(None);
            let (proof, _) = forged.unwrap_or_else(|e| panic!("{what}: no proof: {e}"));
            assert!(verify(&program, &pi, &proof).is_err(), "{what} is accepted");
        }
        // An immediate of the program the verifier holds differs from the one proved, each alone.
        for (what, other) in [
            ("s_a", (g_pow(n + 1).inv(), imm.1, imm.2)),
            ("k_a", (imm.0, imm.1 + F64(1), imm.2)),
            ("k_y", (imm.0, imm.1, imm.2 + F64(1))),
        ] {
            let (changed, _, _) = keccak64_mix_program_with_immediates(message, F128::ZERO, 3, 2, true, false, other);
            assert!(
                verify(&changed, &pi, &proof).is_err(),
                "the bytecode's {what} is not bound"
            );
        }
    }

    /// The forgeries of `KECCAK64`'s own rows, each refused, for a program whose `KECCAK64` rows carry `imm`.
    fn forged_keccak64_cases(message: [F128; 4], cap: F128, imm: (F64, F64, F64)) {
        use super::forge::{Forge, set};
        let pi = [w(7), w(11)];
        let rejects = |forge: Forge, keccak64: usize, keccak: usize, out_two: bool, cap: F128| {
            let (program, _, _) =
                keccak64_mix_program_with_immediates(message, cap, keccak64, keccak, out_two, false, imm);
            // The honest proof of the same program verifies.
            set(None);
            let (proof, _) = prove(&program, &[], pi, crate::pcs::TEST_LOG_INV_RATE).unwrap();
            verify(&program, &pi, &proof).expect("the honest proof verifies");
            set(Some(forge));
            // The prover goes on past its own check that the bus balances, as a malicious one would.
            let forged = prove(&program, &[], pi, crate::pcs::TEST_LOG_INV_RATE);
            set(None);
            let (proof, _) = forged.unwrap_or_else(|e| panic!("{forge:?}: no proof: {e}"));
            assert!(verify(&program, &pi, &proof).is_err(), "{forge:?} is accepted");
        };
        // A lane the padding fixes: lanes 8, 16 and 20 are public, so the committed pieces stay honest and the
        // reduction runs on the forged state. Lane 24 is committed and pinned by its identity.
        for lane in [8, 16, 20, 24] {
            rejects(Forge::Keccak64InputLane(lane), 3, 2, true, F128::ZERO);
        }
        rejects(Forge::Keccak64OtherRowsOutput, 3, 2, true, F128::ZERO);
        // The rows of one table in the other's half.
        rejects(Forge::SwapRows(2), 2, 2, true, cap);
        rejects(Forge::SentLane, 3, 2, true, F128::ZERO);
        rejects(Forge::Keccak64PaddingRowIsReal, 3, 2, true, F128::ZERO);
        // The flag against the bytecode, both ways.
        rejects(Forge::Keccak64OutTwo(true), 3, 2, false, F128::ZERO);
        rejects(Forge::Keccak64OutTwo(false), 3, 2, true, F128::ZERO);
    }

    /// `q_keccak` as pieces of its halves (70,000 `KECCAK64` rows are two row pieces of three slot blocks, 40 `KECCAK` rows
    /// one): the one ring claim over them refuses a word of a piece changed in the committed stack, a pad taken from
    /// the other half, two slot blocks of one row piece that change places, and a block that holds another row
    /// piece's.
    #[test]
    fn forged_pieces_of_q_keccak_are_rejected() {
        use super::forge::{Forge, set};
        let message: [F128; 4] = std::array::from_fn(|i| F128::new(0x1111 * (i as u64 + 1), 0x7777 * (i as u64 + 3)));
        let pi = [w(7), w(11)];
        let (program, _, _) = keccak64_mix_program(message, 70_000, 40, true, true);
        set(None);
        let (honest, _) = prove(&program, &[], pi, crate::pcs::TEST_LOG_INV_RATE).unwrap();
        verify(&program, &pi, &honest).expect("the honest proof verifies");
        for forge in [
            // A word in `KECCAK`'s block, and in each of `KECCAK64`'s three slot blocks.
            Forge::PieceWord(crate::keccak::KECCAK_HALF, (0, 6), 0),
            Forge::PieceWord(crate::keccak::KECCAK64_HALF, (0, 3), 0),
            Forge::PieceWord(crate::keccak::KECCAK64_HALF, (24, 3), 0),
            Forge::PieceWord(crate::keccak::KECCAK64_HALF, (32, 5), 0),
            Forge::SwapPads,
            Forge::SwapBlocks,
            Forge::BorrowBlock,
        ] {
            set(Some(forge));
            let forged = prove(&program, &[], pi, crate::pcs::TEST_LOG_INV_RATE);
            set(None);
            let (proof, _) = forged.unwrap_or_else(|e| panic!("{forge:?}: no proof: {e}"));
            assert!(verify(&program, &pi, &proof).is_err(), "{forge:?} is accepted");
        }
    }

    /// The last slot in a frame other than main's is an ordinary slot: the machine halts in the
    /// final STATE, so this program reaches the last slot at `fp = g`, executes the JUMP there,
    /// and halts on landing back in main's frame. Its five rows are exactly the walk the state
    /// bus balances (§sec:state), so the verifier accepts the run the prover proves.
    #[test]
    fn halts_in_the_final_state() {
        let pi = [w(7), w(11)];
        let (last_pc, other_fp) = (F128::from(g_pow(7)), F128::from(g_pow(1)));
        let fill = Op::Xor { a: 0, b: 0, c: 0 };
        // The last slot is reached at `fp = g`, which is not main's, so it runs like any other.
        let program = |last| {
            let code = vec![
                set(2, F128::ONE),                // the jump condition, and main's fp
                set(3, last_pc),                  // the last slot's pc
                set(4, other_fp),                 // a frame pointer other than main's
                Op::Jump { oc: 2, od: 3, of: 4 }, // -> (g^7, g)
                fill,
                fill,
                fill,
                last, // at fp = g it reads cells 2, 3, 2, so -> (g^7, g^0): the final state
            ];
            from_bytecode(code, 8)
        };

        let jumping = program(Op::Jump { oc: 1, od: 2, of: 1 });
        let exec = execute(&jumping, &[], pi).expect("the run halts in the final state");
        assert_eq!(exec.cycles, 5, "three SETs and both JUMPs, the last slot's included");
        assert_eq!(
            exec.rows(),
            [0, 0, 3, 0, 2, 0, 0],
            "XOR, MUL, SET, DEREF, JUMP, KECCAK, KECCAK64"
        );
        let (proof, _) = prove(&jumping, &[], pi, crate::pcs::TEST_LOG_INV_RATE).unwrap();
        verify(&jumping, &pi, &proof).expect("the verifier accepts the run");

        // With the compiler's SET in the last slot the run has nowhere to go from there, its
        // successor g^8 being no bytecode address.
        let err = execute(&program(set(5, F128::ONE)), &[], pi)
            .err()
            .expect("the run must fail");
        assert!(matches!(err.fault, Fault::HaltOutsideMain { fp: 1 }), "{err}");
    }

    /// A 128-bit-word MUL: the E-product of two full machine words.
    #[test]
    fn mul_128bit_word() {
        let x = F128::new(0x0123_4567_89ab_cdef, 0xfeed_face_dead_beef);
        let y = F128::new(0x9999_aaaa_bbbb_cccc, 0x1357_9bdf_2468_ace0);
        let prog = vec![
            set(2, x),
            set(3, y),
            Op::Mul { a: 2, b: 3, c: 4 },
            Op::Xor { a: 0, b: 0, c: 0 }, // sentinel (never executed)
        ];
        let program = from_bytecode(prog, 5);
        let pi = [w(1), w(2)];
        let exec = execute(&program, &[], pi).unwrap();
        assert_eq!(exec.mem.word(4), x * y, "MUL computes the E product");
    }

    /// A frame past the first page of cells keeps its words and access counts, and its program proves
    /// the same bytes twice.
    #[test]
    fn dynamic_high_frame_keeps_its_memory_and_counts() {
        let high = 1 << 16;
        let fill = Op::Xor { a: 0, b: 0, c: 0 };
        let jump = Op::Jump { oc: 0, od: 0, of: 1 };
        let pi = [F128::from(g_pow(3)), F128::from(g_pow(high))];
        let program = from_bytecode(
            vec![
                jump,
                fill,
                fill,
                set(0, F128::from(g_pow(7))),
                set(1, F128::ONE),
                jump,
                fill,
                fill,
            ],
            2,
        );
        let exec = execute(&program, &[], pi).unwrap();
        assert_eq!(exec.mem_used, high + 2);
        assert_eq!(exec.mem.len(), (high + 2).next_power_of_two());
        assert_eq!(exec.mem.word(high), F128::from(g_pow(7)));
        assert_eq!(exec.mem.word(high + 1), F128::ONE);
        assert_eq!(exec.mem.page(high >> Memory::page_log()).unwrap()[0].count, 3);
        let (first, _) = prove(&program, &[], pi, crate::pcs::TEST_LOG_INV_RATE).unwrap();
        verify(&program, &pi, &first).unwrap();
        let (second, _) = prove(&program, &[], pi, crate::pcs::TEST_LOG_INV_RATE).unwrap();
        assert_eq!(first.to_bytes(), second.to_bytes());
    }
}
