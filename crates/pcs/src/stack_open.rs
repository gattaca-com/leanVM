// CREDIT: https://github.com/succinctlabs/flock (flock-core), MIT OR Apache-2.0.
//! Stacked batch-mixed opening for the F64-committed PCS.
//!
//! The committed witness is a stack of `2^log_n` [`F64`] words (committed via
//! [`super::whir::commit`], which only encodes and only transmits the lane blocks
//! that carry data: every claim lives inside them and the weight is zero past
//! them), and one WHIR run discharges
//!
//! - **point claims** ([`StackClaim`]): plain multilinear evaluations of
//!   aligned sub-slices of the stack (a claim's weight is `eq(low_point, .)`
//!   supported on `[offset, offset + 2^|low_point|)`),
//! - **ring-switched claims** ([`RingSwitchOpen`]): bit-MLE evaluation claims
//!   on the packed sub-block `q_keccak = stack[offset .. offset + 2^qkeccak_vars]`,
//!   of which only the first `len` words are weighed, reduced per claim by [`super::ring_switch::prove_prepare`] and the
//!   deferred finish path to an inner-product
//!   claim `<q_keccak, rs_eq_ind> = sumcheck_claim` against the transparent
//!   E-valued weight `rs_eq_ind`.
//!
//! All claims are lambda-folded into ONE combined weight `b_stack` over the
//! whole stack plus one `target`, then proved by
//! `whir::recursive_prover_with_prepared_basis`. The weight is never
//! materialized over the stack: the lane rounds run on its terms (see the `basis`
//! module), and only the one lane block that survives them is built. A verifier recomputes
//! the target from the claims and evaluates `MLE(b_stack)` once, at the final fold
//! point, from the closed form under "The combined weight".
//!
//! ## Transcript order
//!
//! Sample the shared linear map, then one batching challenge for both claim families, then run WHIR. The caller already bound each claim's slices and value through the transcript, so none is observed again here.
//!
//! ## The combined weight
//!
//! With `sel = offset >> qkeccak_vars` the selector coords of the q_keccak slice,
//! the lifted weight at a full-stack point `x = (x_lo, x_hi)` (split at
//! `qkeccak_vars`, LSB-first) is
//!
//! ```text
//! b(x) = eq(sel, x_hi) * sum_i lambda^i * MLE(rs_eq_ind_i)(x_lo)
//!      + sum_j lambda^(n_rs + j) * eq(claim_j, x)
//! ```
//!
//! with `rs_eq_ind_i` cut to zero past the ring's `len` words.
//!
//! which is exactly what a dense `b_stack` scatter would produce (each claim's
//! weight lives on its aligned slice, so scattering the low-dimensional eq /
//! rs_eq_ind tensor at the slice offset IS multiplying by the boolean
//! selector eq).
//!
//! Both families take DISJOINT power ranges of ONE challenge, as the table
//! sumcheck's xi ranges do, so every claim carries a distinct power (the
//! batching step of `thm:rbr`).

use fiat_shamir::transcript::ProverState;
use flux_profiler::timed;
use primitives::field::{F64, F192, powers};

use super::ring_switch;
use super::whir::{ProverConfig, ProverData};

mod basis;

/// Bits of the block the repeat map tags: it holds `2^EQUAL_BLOCK_LOG` words.
pub const EQUAL_BLOCK_LOG: usize = basis::LOW_VARS;

/// The exact repeat map of `stack`: for each block of `2^EQUAL_BLOCK_LOG` words, whether it equals the block
/// before it. The reference a producer's map is checked against.
pub fn equal_blocks(stack: &[F64]) -> Vec<bool> {
    let blocks: Vec<_> = stack.chunks_exact(1 << EQUAL_BLOCK_LOG).collect();
    (0..blocks.len()).map(|b| b > 0 && blocks[b] == blocks[b - 1]).collect()
}

// ---------------------------------------------------------------------------
// Claim types
// ---------------------------------------------------------------------------

/// Rows of a jagged chunk: a column's run inside one chunk of its group.
pub const CHUNK_LOG: usize = 10;

/// An owning point claim folded into the stacked mixed opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StackClaim {
    /// `eq(low_point, .)` on the aligned slice
    /// `[offset, offset + 2^|low_point|)`; `offset` must be a multiple of
    /// `2^|low_point|`.
    Point {
        offset: usize,
        low_point: Vec<F192>,
        value: F192,
    },
    /// A column of a jagged group at `point`. The group is `chunks` chunks of `2^CHUNK_LOG` rows from `offset`, a
    /// chunk holding `2^log_width` column runs one after another, so row `i` of the column sits at
    /// `offset + (i >> CHUNK_LOG)·2^(CHUNK_LOG + log_width) + column·2^CHUNK_LOG + (i mod 2^CHUNK_LOG)`.
    /// Its weight is `eq(point, i)` there, the point zero-extended; `offset` is a multiple of a chunk.
    Jagged {
        offset: usize,
        log_width: usize,
        column: usize,
        chunks: usize,
        point: Vec<F192>,
        value: F192,
    },
    /// A [`StackClaim::Jagged`] on every column of its group, column `j` weighted by `eq(column, j)`.
    JaggedAt {
        offset: usize,
        log_width: usize,
        chunks: usize,
        point: Vec<F192>,
        column: Vec<F192>,
        value: F192,
    },
}

impl StackClaim {
    #[inline]
    pub fn value(&self) -> F192 {
        match self {
            StackClaim::Point { value, .. } | StackClaim::Jagged { value, .. } | StackClaim::JaggedAt { value, .. } => {
                *value
            }
        }
    }

    /// The range of the stack its weight is supported on.
    fn range(&self) -> (usize, usize) {
        match self {
            StackClaim::Point { offset, low_point, .. } => (*offset, *offset + (1usize << low_point.len())),
            StackClaim::Jagged {
                offset,
                log_width,
                chunks,
                ..
            }
            | StackClaim::JaggedAt {
                offset,
                log_width,
                chunks,
                ..
            } => (*offset, *offset + (chunks << (CHUNK_LOG + log_width))),
        }
    }
}

/// One ring-switched claim on the q_keccak sub-block: the 64 bit-slice MLEs of
/// q_keccak at `suffix_point` (see [`super::ring_switch`]), which has
/// `qkeccak_vars` coords, the same for every claim.
///
/// `s_hat_v` holds those 64 values. The caller transmits and checks them itself
/// (the Keccak reduction sends them and checks them against its last claim), so this
/// layer only binds them to the commitment. `None` asks the prover to fold the slices from the witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingSwitchClaim {
    pub suffix_point: Vec<F192>,
    pub s_hat_v: Option<Vec<F192>>,
}

/// Prover-side bundle of the ring-switched claims discharged in the same
/// stacked opening as the [`StackClaim`]s. Each claim may carry its
/// precomputed `s_hat_v`.
#[derive(Clone, Debug)]
pub struct RingSwitchOpen {
    /// q_keccak's offset inside the committed stack; must be a multiple of
    /// `2^qkeccak_vars` (an aligned slice).
    pub offset: usize,
    /// The words from `offset` the claims weigh: at most `2^qkeccak_vars`, a whole number of `2^CHUNK_LOG`-word
    /// blocks unless it is all of them. The weight is zero past them, the claims being on those words alone.
    pub len: usize,
    pub claims: Vec<RingSwitchClaim>,
}

impl RingSwitchOpen {
    /// log2 of q_keccak's length in F64 words, its claims' point length; the
    /// opener slices `q_keccak = stack[offset .. offset + 2^qkeccak_vars]` (no
    /// separate copy).
    pub fn qkeccak_vars(&self) -> usize {
        self.claims.first().map_or(0, |claim| claim.suffix_point.len())
    }
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

/// Open the committed `F64` stack: discharge every `point_claims` slice
/// evaluation AND the ring-switched q_keccak claims (`ring`) in ONE WHIR
/// run, reusing the caller's [`super::whir::commit`] output as L0.
///
/// `repeats` is the producer's map of the stack: for each block of `2^EQUAL_BLOCK_LOG` words, `true` only if
/// the block equals the block before it. A missing repeat costs a read and a false one is a bug, so the map is
/// conservative. It may be empty, which claims no repeats.
///
/// `stack` is the committed message (not stored in [`ProverData`]), consumed: the
/// lane rounds drop it before the level-1 commit. It is the `2^log_n`-word stack truncated to the lane blocks that
/// carry data, so every claim must live inside it and the padding past it is
/// exactly the weight's zero region. `config.initial_k` / `config.log_inv_rates[0]`
/// must match the commit's `log_batch_size` / `log_inv_rate` (enforced by shape
/// asserts inside the WHIR prover).
#[timed]
pub fn open_batch_mixed_whir_stacked(
    ps: &mut ProverState,
    log_n: usize,
    stack: Vec<F64>,
    repeats: Vec<bool>,
    prover_data: ProverData,
    config: &ProverConfig,
    point_claims: &[StackClaim],
    ring: &RingSwitchOpen,
) {
    let qkeccak_len = 1usize << ring.qkeccak_vars();
    assert!(
        ring.offset.is_multiple_of(qkeccak_len),
        "q_keccak offset must be 2^qkeccak_vars-aligned"
    );
    assert!(
        ring.claims.is_empty() || (ring.len <= qkeccak_len && ring.offset + ring.len <= stack.len()),
        "q_keccak's words must fit inside the stack"
    );
    assert!(
        ring.len == qkeccak_len || ring.len.is_multiple_of(1 << CHUNK_LOG),
        "a cut ring weight ends on a block"
    );
    assert!(
        point_claims.iter().all(|c| c.range().1 <= stack.len()),
        "every claim must live inside the committed lanes"
    );
    assert!(
        repeats.is_empty() || repeats.len() == stack.len() >> EQUAL_BLOCK_LOG,
        "the repeat map tags every block of the stack"
    );
    let span = tracing::info_span!("Ring switch").entered();

    // 1. Ring-switch reduction: prepare every claim's s_hat_v (the caller bound
    //    them upstream), then sample one shared linear map.
    let qkeccak = stack.get(ring.offset..ring.offset + qkeccak_len).unwrap_or_default();
    let mut rs_states = Vec::with_capacity(ring.claims.len());
    for claim in &ring.claims {
        assert_eq!(
            claim.suffix_point.len(),
            ring.qkeccak_vars(),
            "every ring-switch suffix point must have qkeccak_vars coords"
        );
        rs_states.push(ring_switch::prove_prepare(
            qkeccak,
            &claim.suffix_point,
            claim.s_hat_v.as_deref(),
        ));
    }
    let map_challenges = ring_switch::sample_map_challenges(ps);
    let coordinate_weights = ring_switch::build_coordinate_weights(&map_challenges);

    // 2. The ONE batching challenge both families take disjoint power ranges of. Nothing is
    //    observed first: every claim value reached the caller through a binding stream read, so
    //    the challenge already depends on all of them (`leanvm_core::pcs::open`).
    let lambdas = powers(ps.sample(), ring.claims.len() + point_claims.len());
    let (lambdas_rs, lambdas_pd) = lambdas.split_at(ring.claims.len());

    let rs_outputs: Vec<_> = rs_states
        .into_iter()
        .zip(lambdas_rs.iter().copied())
        .map(|(state, lambda)| ring_switch::prove_finish_deferred(state, &coordinate_weights, lambda))
        .collect();
    drop(span);

    // 3. Combined target and lifted stack weight b_stack: the lambda-weighted
    //    rs_eq_ind sum scattered at the q_keccak slice, plus the point-claim
    //    eq tensors scattered at their offsets (the weight itself stays factored).
    let target = rs_outputs
        .iter()
        .fold(F192::ZERO, |acc, out| acc + out.batched_sumcheck_claim)
        + point_claims
            .iter()
            .zip(lambdas_pd)
            .fold(F192::ZERO, |sum, (claim, &lambda)| sum + lambda * claim.value());

    // The lane rounds run on the weight's terms and a projection of the stack onto them, so the
    // lifted weight over the stack is never built.
    let lane_vars = log_n - config.initial_k;
    let rings: Vec<_> = rs_outputs.into_iter().map(|output| output.weight).collect();
    let initial_basis = tracing::info_span!("Basis")
        .in_scope(|| basis::project(&stack, repeats, lane_vars, point_claims, lambdas_pd, ring, &rings));

    // 4. One WHIR over the full stack against the combined claim (the
    //    prover takes the stack; no copy).
    super::whir::recursive_prover_with_prepared_basis(config, log_n, stack, initial_basis, target, prover_data, ps);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use primitives::test_rng::Rng;

    use crate::whir::{SumcheckMessage, SumcheckProver};

    /// `(u_0, u_2)` of the pairs of blocks `(2i, 2i + 1)`, a last odd block against zeros.
    fn blocks_msg(f: &[F192], b: &[F192], block: usize) -> SumcheckMessage {
        let (mut u_0, mut u_2) = (F192::ZERO, F192::ZERO);
        for (f, b) in f.chunks(2 * block).zip(b.chunks(2 * block)) {
            for x in 0..block {
                let (f0, b0) = (f[x], b[x]);
                let (f1, b1) = (
                    f.get(block + x).copied().unwrap_or(F192::ZERO),
                    b.get(block + x).copied().unwrap_or(F192::ZERO),
                );
                u_0 += f0 * b0;
                u_2 += (f0 + f1) * (b0 + b1);
            }
        }
        SumcheckMessage { u_0, u_2 }
    }

    /// `(u_0, u_2)` of the adjacent pairs `(2j, 2j + 1)`.
    fn adjacent_msg(f: &[F192], b: &[F192]) -> SumcheckMessage {
        let (mut u_0, mut u_2) = (F192::ZERO, F192::ZERO);
        for (f, b) in f.chunks_exact(2).zip(b.chunks_exact(2)) {
            u_0 += f[0] * b[0];
            u_2 += (f[0] + f[1]) * (b[0] + b[1]);
        }
        SumcheckMessage { u_0, u_2 }
    }

    /// One lane round: block `2j` and block `2j + 1` fold into block `j` as `x0 + r (x0 + x1)`,
    /// a last odd block against zeros.
    fn fold_blocks(x: &[F192], block: usize, r: F192) -> Vec<F192> {
        x.chunks(2 * block)
            .flat_map(|pair| (0..block).map(move |i| (pair[i], pair.get(block + i).copied().unwrap_or(F192::ZERO))))
            .map(|(x0, x1)| x0 + r * (x0 + x1))
            .collect()
    }

    /// The lane rounds of a projected basis must send the messages of the naive lane rounds on
    /// the dense weight and leave the same witness, weight and next message, across claims inside
    /// one lane and spanning lanes, strides above the lane bit, shared and distinct points, and a
    /// q_flock ring both spanning lanes and selected by lane bits.
    #[test]
    fn projected_lane_rounds_match_the_naive_dense_rounds() {
        const LANE_ROUNDS: usize = 6;
        let mut rng = Rng::new(0xBA515);
        for (lane_vars, lanes, qkeccak_vars) in [
            (6usize, 1usize, 6usize),
            (6, 3, 7),
            (8, 13, 10),
            (8, 13, 8),
            (8, 5, 6),
            (10, 37, 12),
            (12, 5, 13),
            (11, 5, 15),
        ] {
            let lane_block = 1usize << lane_vars;
            let mut stack: Vec<F64> = (0..lanes * lane_block).map(|_| F64(rng.next_u64())).collect();
            // Runs of equal blocks, which the map hands to the projection.
            for lane in stack.chunks_mut(lane_block).filter(|_| lane_vars >= EQUAL_BLOCK_LOG) {
                for b in [1, 2, 6, 7].into_iter().filter(|&b| b << EQUAL_BLOCK_LOG < lane_block) {
                    lane.copy_within((b - 1) << EQUAL_BLOCK_LOG..b << EQUAL_BLOCK_LOG, b << EQUAL_BLOCK_LOG);
                }
            }
            let repeats = equal_blocks(&stack);
            let qkeccak_len = 1usize << qkeccak_vars;
            // The whole block, and blocks cut short: on a lane's end, inside a lane, and past the stack's bits.
            let chunk = 1usize << CHUNK_LOG;
            let cases = [0, qkeccak_len]
                .into_iter()
                .filter(|offset| offset + qkeccak_len <= stack.len())
                .map(|offset| (offset, qkeccak_len))
                .chain(
                    [3 * chunk, 5 * chunk, 9 * chunk, 10 * chunk]
                        .into_iter()
                        .filter_map(|len| (len < qkeccak_len && len <= stack.len()).then_some((0, len))),
                );
            for (offset, len) in cases {
                let ring = RingSwitchOpen {
                    offset,
                    len,
                    claims: (0..2)
                        .map(|_| RingSwitchClaim {
                            suffix_point: rng.ext_vec(qkeccak_vars),
                            s_hat_v: None,
                        })
                        .collect(),
                };
                let coordinates = rng.ext_vec(192);
                let rs_outputs: Vec<_> = ring
                    .claims
                    .iter()
                    .map(|claim| {
                        let mut words = stack[offset..offset + len].to_vec();
                        words.resize(qkeccak_len, F64::ZERO);
                        let state = ring_switch::prove_prepare(&words, &claim.suffix_point, None);
                        ring_switch::prove_finish_deferred(state, &coordinates, rng.ext())
                    })
                    .collect();
                let last_lane = (lanes - 1) * lane_block;
                let shared = rng.ext_vec(lane_vars);
                let ring_shape = (offset, qkeccak_vars, None);
                let mut shapes = vec![
                    ring_shape.clone(),
                    ring_shape,
                    (last_lane, lane_vars, Some(shared.clone())),
                    (0, lane_vars, Some(shared)),
                    (last_lane, lane_vars, None),
                    (8, 3, None),
                    (8, 3, None),
                    (0, 0, None),
                    (last_lane + 4, 2, None),
                ];
                for spanned in 1..=2 {
                    if lanes >= 1 << spanned {
                        shapes.push((0, lane_vars + spanned, None));
                    }
                }
                let mut claims: Vec<_> = shapes
                    .into_iter()
                    .filter(|&(offset, vars, _)| offset + (1 << vars) <= stack.len())
                    .map(|(offset, vars, point)| StackClaim::Point {
                        offset,
                        low_point: point.unwrap_or_else(|| rng.ext_vec(vars)),
                        value: rng.ext(),
                    })
                    .collect();
                // Jagged groups: two columns of one group at one point, a lone column of a wider group whose point
                // is shorter than a chunk, and a column whose point is longer than the group.
                if stack.len() >= 8 * chunk {
                    let shared = rng.ext_vec(CHUNK_LOG + 1);
                    let groups = [
                        (0, 1, 0, 3, shared.clone()),
                        (0, 1, 1, 3, shared),
                        (6 * chunk, 1, 1, 1, rng.ext_vec(CHUNK_LOG - 2)),
                        (2 * chunk, 0, 0, 1, rng.ext_vec(CHUNK_LOG + 3)),
                    ];
                    claims.extend(groups.into_iter().map(|(offset, log_width, column, chunks, point)| {
                        StackClaim::Jagged {
                            offset,
                            log_width,
                            column,
                            chunks,
                            point,
                            value: rng.ext(),
                        }
                    }));
                    // A claim on every column of a group at once.
                    claims.push(StackClaim::JaggedAt {
                        offset: 4 * chunk,
                        log_width: 1,
                        chunks: 2,
                        point: rng.ext_vec(CHUNK_LOG + 1),
                        column: rng.ext_vec(1),
                        value: rng.ext(),
                    });
                }
                let lambdas = rng.ext_vec(claims.len());

                // Oracle: the dense weight written out naively, one eq entry at a time.
                let mut dense = vec![F192::ZERO; stack.len()];
                ring_switch::combine_deferred_chunk(&rs_outputs, 0, &mut dense[offset..offset + len]);
                let eq_at = |point: &[F192], j: usize, lambda: F192| {
                    point.iter().enumerate().fold(lambda, |w, (i, &p_i)| {
                        w * if (j >> i) & 1 == 1 { p_i } else { F192::ONE + p_i }
                    })
                };
                for (claim, &lambda) in claims.iter().zip(&lambdas) {
                    let jagged = |dense: &mut [F192],
                                  offset: usize,
                                  log_width: usize,
                                  column: usize,
                                  chunks: usize,
                                  point: &[F192],
                                  lambda: F192| {
                        for row in (0..chunks << CHUNK_LOG).filter(|row| row >> point.len() == 0) {
                            let at = offset
                                + (row >> CHUNK_LOG << (CHUNK_LOG + log_width))
                                + (column << CHUNK_LOG)
                                + (row & ((1 << CHUNK_LOG) - 1));
                            dense[at] += eq_at(point, row, lambda);
                        }
                    };
                    match claim {
                        StackClaim::Point { offset, low_point, .. } => {
                            for j in 0..1usize << low_point.len() {
                                dense[offset + j] += eq_at(low_point, j, lambda);
                            }
                        }
                        StackClaim::Jagged {
                            offset,
                            log_width,
                            column,
                            chunks,
                            point,
                            ..
                        } => jagged(&mut dense, *offset, *log_width, *column, *chunks, point, lambda),
                        StackClaim::JaggedAt {
                            offset,
                            log_width,
                            chunks,
                            point,
                            column,
                            ..
                        } => {
                            for j in 0..1 << log_width {
                                jagged(
                                    &mut dense,
                                    *offset,
                                    *log_width,
                                    j,
                                    *chunks,
                                    point,
                                    eq_at(column, j, lambda),
                                );
                            }
                        }
                    }
                }

                let rings: Vec<_> = rs_outputs.iter().map(|output| output.weight.clone()).collect();
                let projected = basis::project(&stack, repeats.clone(), lane_vars, &claims, &lambdas, &ring, &rings);
                let target = rng.ext();
                let (mut prover, msg) = SumcheckProver::new(stack.clone(), projected, target, lane_block);
                let context = format!(
                    "lane_vars={lane_vars}, lanes={lanes}, qkeccak_vars={qkeccak_vars}, offset={offset}, len={len}"
                );
                let mut f: Vec<F192> = stack.iter().map(|&word| F192::from(word)).collect();
                assert_eq!(msg, blocks_msg(&f, &dense, lane_block), "{context}, round 0");
                for round in 1..=LANE_ROUNDS {
                    let r = rng.ext();
                    let last = round == LANE_ROUNDS;
                    f = fold_blocks(&f, lane_block, r);
                    dense = fold_blocks(&dense, lane_block, r);
                    let want = if last {
                        adjacent_msg(&f, &dense)
                    } else {
                        blocks_msg(&f, &dense, lane_block)
                    };
                    assert_eq!(prover.fold_lane(r, last), want, "{context}, round {round}");
                }
                assert_eq!(prover.f_ext(), f, "{context}, witness");
                assert_eq!(prover.basis(), dense, "{context}, weight");
            }
        }
    }
}
