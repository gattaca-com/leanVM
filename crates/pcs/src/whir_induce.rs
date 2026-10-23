// CREDIT: https://github.com/succinctlabs/flock (flock-core), MIT OR Apache-2.0.
// Copyright (c) 2026 Bain Capital Crypto, LP and Ron Rothblum
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! LCH novel-basis induction of the per-level sumcheck basis polynomial: the
//! dense per-query expansion, its succinct residual evaluator, and the sparse
//! transposed-NTT fast path with the dispatch between them.

use flux_profiler::timed;
use crate::ntt::additive_ntt_f64::transpose_butterfly_lanes;
use crate::ntt::AdditiveNttF64;
use crate::whir_ntt_ext::words_mut;
use crate::whir::build_eq_table_ext;
use primitives::field::{F64, F192, Weights8, dot_base};

// ===================================================================
// LCH novel-basis evaluations over K (mirror of whir's extension-field block)
// ===================================================================
//
// The subspace-polynomial recurrence runs entirely over the K evaluation
// domain (F64 values); results are lifted into E with `mul_base` only where
// they scale E-accumulators. Standard basis only (v_i = x^i = F64(1 << i)).

#[inline]
fn next_s(s: F64, s_at_root: F64) -> F64 {
    s * s + s_at_root * s
}

/// `sks_vks[k] = s_k(v_k)` for `k = 0..=log_n`, over K. Mirror of
/// `whir::eval_sk_at_vks`. Public for the recursion harness, which dumps
/// these vanishing-polynomial values as guest hints (base-field, embedded into
/// the tower with both extension limbs zero).
pub fn eval_sk_at_vks(log_n: usize) -> Vec<F64> {
    let mut sks_vks = vec![F64::ZERO; log_n + 1];
    sks_vks[0] = F64::ONE;
    if log_n == 0 {
        return sks_vks;
    }
    let mut layer: Vec<F64> = (1..=log_n).map(|i| F64(1u64 << i)).collect();
    let mut cur_len = log_n;
    for i in 0..log_n {
        for j in 0..cur_len {
            let sk_at_vk = next_s(layer[j], sks_vks[i]);
            if j == 0 {
                sks_vks[i + 1] = sk_at_vk;
            } else {
                layer[j - 1] = sk_at_vk;
            }
        }
        cur_len -= 1;
    }
    sks_vks
}

/// `out[i] = W-hat_i(x) = s_i(x) / s_i(v_i)`, the normalized LCH basis
/// exponents at the K point `x`. Stays entirely in K.
fn normalized_sks_at(x: F64, sks_vks: &[F64], inv_sks_vks: &[F64], out: &mut [F64]) {
    if out.is_empty() {
        return;
    }
    out[0] = x;
    for i in 1..out.len() {
        out[i] = next_s(out[i - 1], sks_vks[i - 1]);
    }
    for (v, &inv) in out.iter_mut().zip(inv_sks_vks) {
        *v *= inv;
    }
}

/// Row entries of an opened level: `F64` at L0 (mixed `mul_base` dot against
/// the E-valued eq weights), `F192` at every deeper level (full E dot).
pub(crate) trait RowElem: Copy + Sync {
    fn dot(row: &[Self], eq: &[F192]) -> F192;
}

impl RowElem for F64 {
    #[inline]
    fn dot(row: &[Self], eq: &[F192]) -> F192 {
        row.iter()
            .zip(eq.iter())
            .map(|(&r, &e)| e.mul_base(r))
            .fold(F192::ZERO, |a, v| a + v)
    }
}

impl RowElem for F192 {
    #[inline]
    fn dot(row: &[Self], eq: &[F192]) -> F192 {
        row.iter()
            .zip(eq.iter())
            .map(|(&r, &e)| r * e)
            .fold(F192::ZERO, |a, v| a + v)
    }
}

fn invert_sks(sks_vks: &[F64]) -> Vec<F64> {
    sks_vks
        .iter()
        .map(|&v| if v.is_zero() { F64::ZERO } else { v.inv() })
        .collect()
}

/// Dense induce: `basis_poly[j] = Σ_i w_i · W-hat_j(q_i)`,
/// `enforced_sum = Σ_i w_i · <row_i, eq(v_challenges, ·)>`, for the per-query
/// batching weights `w` of [`primitives::field::powers`].
///
/// `W-hat_j(q) = Π_k W-hat_k(q)^(bit k of j)` is a K value and a tensor over the
/// bits of `j`. Split at `lo` bits, `basis_poly[j] = Σ_i (w_i · high_i(j >> lo)) ·
/// low_i(j mod 2^lo)`: one packed mixed dot over the queries per output.
#[timed]
pub(crate) fn induce_sumcheck_poly<T: RowElem>(
    log_msg_cols: usize,
    sks_vks: &[F64],
    opened_rows: &[Vec<T>],
    v_challenges: &[F192],
    queries: &[usize],
    weights: &[F192],
) -> (Vec<F192>, F192) {
    let n_queries = queries.len();
    assert_eq!(opened_rows.len(), n_queries);
    assert_eq!(weights.len(), n_queries);
    let eq = build_eq_table_ext(v_challenges);
    let enforced_sum = opened_rows
        .iter()
        .zip(weights)
        .fold(F192::ZERO, |sum, (row, &w)| sum + T::dot(row, &eq) * w);

    let inv_sks_vks = invert_sks(sks_vks);
    let lo = log_msg_cols / 2;
    let padded = n_queries.next_multiple_of(8);
    // `low[j * padded + i] = low_i(j)`, `high[i << (log_msg_cols - lo) | j] = high_i(j)`.
    let mut low = vec![F64::ZERO; padded << lo];
    let mut high = Vec::with_capacity(n_queries << (log_msg_cols - lo));
    let mut sks = vec![F64::ZERO; log_msg_cols];
    for (i, &q) in queries.iter().enumerate() {
        normalized_sks_at(F64(q as u64), sks_vks, &inv_sks_vks, &mut sks);
        for (j, v) in tensor_k(&sks[..lo]).into_iter().enumerate() {
            low[j * padded + i] = v;
        }
        high.extend(tensor_k(&sks[lo..]));
    }
    let mut basis = vec![F192::ZERO; 1 << log_msg_cols];
    parallel::chunks_mut(&mut basis, 1 << lo, |j, out: &mut [F192]| {
        let scaled = |i: usize| match weights.get(i) {
            Some(w) => w.mul_base(high[(i << (log_msg_cols - lo)) | j]),
            None => F192::ZERO,
        };
        let packed: Vec<Weights8> = (0..padded / 8)
            .map(|g| Weights8::new(&std::array::from_fn(|t| scaled(8 * g + t))))
            .collect();
        for (out, low) in out.iter_mut().zip(low.chunks_exact(padded)) {
            *out = dot_base(&packed, low).reduce();
        }
    });
    (basis, enforced_sum)
}

/// `t[j] = Π_k s[k]^(bit k of j)`, over K.
fn tensor_k(s: &[F64]) -> Vec<F64> {
    let mut t = Vec::with_capacity(1 << s.len());
    t.push(F64::ONE);
    for &x in s {
        for j in 0..t.len() {
            t.push(t[j] * x);
        }
    }
    t
}

/// Just the `enforced_sum` half of [`induce_sumcheck_poly`]:
///   `enforced_sum = Σ_i w_i · <opened_rows[i], eq(v_challenges, ·)>`
/// Cheap: O(num_queries x num_interleaved). The prover needs it for its last
/// level, whose induced basis it never materializes.
#[timed]
pub(crate) fn induce_sumcheck_enforced_sum<T: RowElem>(
    opened_rows: &[Vec<T>],
    v_challenges: &[F192],
    queries: &[usize],
    weights: &[F192],
) -> F192 {
    assert_eq!(opened_rows.len(), queries.len());
    let eq = build_eq_table_ext(v_challenges);
    debug_assert_eq!(weights.len(), queries.len());
    let mut sum = F192::ZERO;
    for (i, row) in opened_rows.iter().enumerate() {
        debug_assert_eq!(row.len(), eq.len());
        sum += weights[i] * T::dot(row, &eq);
    }
    sum
}

/// SUCCINCT evaluator for the induced basis poly's MLE at residual points,
/// in place of the dense basis, via the closed form:
///   `MLE(basis_poly)(p) = Σ_i w_i · Π_k (1 + p[k] · (1 + W-hat_k(q_i)))`
/// where `q_i = F64(queries[i])` and the K-valued `W-hat_k(q_i)` lifts into E
/// through the char-2 factor. `ris_for_basis` is the fixed residual prefix
/// (length `log_msg_cols - yr_log_n`); returns evaluations at the `2^yr_log_n`
/// points `ris_for_basis ++ y_bits`.
#[timed]
pub(crate) fn induce_sumcheck_evaluate_at_residual(
    log_msg_cols: usize,
    sks_vks: &[F64],
    queries: &[usize],
    weights: &[F192],
    ris_for_basis: &[F192],
    yr_log_n: usize,
) -> Vec<F192> {
    assert_eq!(ris_for_basis.len() + yr_log_n, log_msg_cols);
    let n_queries = queries.len();
    let yr_len = 1usize << yr_log_n;

    debug_assert_eq!(weights.len(), n_queries);
    let inv_sks_vks = invert_sks(sks_vks);
    let prefix_len = ris_for_basis.len();

    // Per-query precomputation: W-hat_k(q) for all k over K, split into a
    // fixed prefix product (E scalar) and the suffix W-hat values varied per y.
    struct PerQuery {
        prefix_prod: F192,
        suffix_w: Vec<F64>, // length = yr_log_n
    }
    let compute_query = |&q: &usize| -> PerQuery {
        let mut sks_at_x = vec![F64::ZERO; log_msg_cols];
        normalized_sks_at(F64(q as u64), sks_vks, &inv_sks_vks, &mut sks_at_x);
        // Prefix product: Π_{k<prefix_len} (1 + ris[k] · (1 + W-hat_k(q)))
        let mut prefix_prod = F192::ONE;
        for k in 0..prefix_len {
            prefix_prod *= F192::ONE + ris_for_basis[k] * (F192::ONE + F192::from(sks_at_x[k]));
        }
        let suffix_w = if log_msg_cols > prefix_len {
            sks_at_x[prefix_len..].to_vec()
        } else {
            Vec::new()
        };
        PerQuery { prefix_prod, suffix_w }
    };
    // Once per opening over query-count-sized inputs; stay serial below
    // the dispatch crossover.
    const PAR_FLOOR: usize = 1024;
    let per_query: Vec<PerQuery> = if n_queries > PAR_FLOOR {
        parallel::map_collect(n_queries, |i| compute_query(&queries[i]))
    } else {
        queries.iter().map(compute_query).collect()
    };

    // For each residual position y, accumulate the suffix product per query.
    let compute_y = |y: usize| -> F192 {
        let mut sum = F192::ZERO;
        for i in 0..n_queries {
            let pq = &per_query[i];
            let mut suffix_prod = F192::ONE;
            for j in 0..yr_log_n {
                let p_j = if (y >> j) & 1 == 1 { F192::ONE } else { F192::ZERO };
                suffix_prod *= F192::ONE + p_j * (F192::ONE + F192::from(pq.suffix_w[j]));
            }
            sum += weights[i] * pq.prefix_prod * suffix_prod;
        }
        sum
    };
    if yr_len > PAR_FLOOR {
        primitives::par_collect(yr_len, compute_y)
    } else {
        (0..yr_len).map(compute_y).collect()
    }
}

/// Elements per cache-resident window of the layer-blocked run below.
const TRANSPOSE_CHUNK: usize = 1 << 16;

/// The transposed-butterfly sweep over `layers`, in the order given: parallel
/// over blocks once there are enough of them, over rows within a block
/// otherwise.
///
/// Layers arrive in descending order, so blocks GROW along the run and the ones
/// that fit a cache-resident window are a prefix. Blocks also nest, so such a
/// window holds complete blocks of every layer in that prefix and the whole
/// prefix runs back-to-back inside it: one read and one write of DRAM traffic
/// for the run instead of one sweep per layer. This is the sub-group
/// decomposition the interleaved encode already uses for its deep layers,
/// applied in the transpose direction.
///
/// `data` starts at element `offset` of the domain, aligned to every block it pairs within.
fn transpose_layers_ext(
    ntt: &AdditiveNttF64,
    data: &mut [F192],
    offset: usize,
    log_d: usize,
    layers: impl Iterator<Item = usize>,
) {
    transpose_layers_ext_windowed(ntt, data, offset, log_d, layers, TRANSPOSE_CHUNK);
}

/// The transposed forward butterfly, applied to each pair of E elements as K words. The forward
/// butterfly is `M = [[1, t], [1, t+1]]`, so `M^T = [[1, 1], [t, t+1]]` is
/// `s = a + b; top = s; bot = t*s + b`.
fn transpose_butterfly_ext(t: F64, top: &mut [F192], bot: &mut [F192]) {
    transpose_butterfly_lanes(words_mut(top), words_mut(bot), t);
}

fn transpose_layers_ext_windowed(
    ntt: &AdditiveNttF64,
    data: &mut [F192],
    offset: usize,
    log_d: usize,
    layers: impl Iterator<Item = usize>,
    window_len: usize,
) {
    let n_threads = parallel::num_threads();
    let layers: Vec<usize> = layers.collect();
    debug_assert!(layers.windows(2).all(|w| w[0] > w[1]), "layers descend");
    // Keep a window per worker, or the blocked run costs more in lost
    // parallelism than it saves in traffic.
    let blocked = if data.len() >= window_len.saturating_mul(n_threads) {
        layers
            .iter()
            .position(|&l| (1usize << (log_d - l)) > window_len)
            .unwrap_or(layers.len())
    } else {
        0
    };
    if blocked > 1 {
        parallel::chunks_mut(data, window_len, |c, window: &mut [F192]| {
            let base = offset + c * window_len;
            for &layer in &layers[..blocked] {
                let block_size = 1usize << (log_d - layer);
                let bsh = block_size >> 1;
                for (b, block) in window.chunks_mut(block_size).enumerate() {
                    let (top, bot) = block.split_at_mut(bsh);
                    transpose_butterfly_ext(ntt.twiddle(layer, base / block_size + b), top, bot);
                }
            }
        });
    }
    let mut deep = &layers[if blocked > 1 { blocked } else { 0 }..];
    while let Some(&layer) = deep.first() {
        let block_size = 1usize << (log_d - layer);
        let (first, num_blocks) = (offset / block_size, data.len() / block_size);
        let bsh = block_size >> 1;
        // Fuse consecutive layers into one pass over each super-block, when it keeps every worker busy.
        let fused = deep.iter().take(MAX_FUSED).enumerate().take_while(|&(j, &l)| l + j == layer).count();
        let super_len = block_size << (fused - 1);
        if fused > 1 && data.len() / super_len >= n_threads {
            parallel::chunks_mut(data, super_len, |sb, chunk: &mut [F192]| {
                fused_transpose_layers(ntt, chunk, layer, fused, offset / super_len + sb)
            });
            deep = &deep[fused..];
            continue;
        }
        deep = &deep[1..];
        if num_blocks >= n_threads {
            parallel::chunks_mut(data, block_size, |block, chunk: &mut [F192]| {
                let (top, bot) = chunk.split_at_mut(bsh);
                transpose_butterfly_ext(ntt.twiddle(layer, first + block), top, bot);
            });
        } else {
            for block in 0..num_blocks {
                let t = ntt.twiddle(layer, first + block);
                let chunk = &mut data[block * block_size..(block + 1) * block_size];
                let (top, bot) = chunk.split_at_mut(bsh);
                let chunk_len = parallel::recommended_chunk_size(bsh);
                parallel::chunks_mut2(top, bot, chunk_len, |_, top_c, bot_c| transpose_butterfly_ext(t, top_c, bot_c));
            }
        }
    }
}

/// Layers one fused pass takes, which bounds the rows it keeps in L1.
const MAX_FUSED: usize = 3;
/// Elements per row strip of a fused pass: `2^MAX_FUSED` strips of this width fit L1.
const FUSED_STRIP: usize = 128;

/// `fused` consecutive layers `layer, layer-1, ..` over one super-block, the
/// `sb`-th, in a single pass. Its `2^fused` rows pair at distance `1, 2, 4, ..`
/// rows, one layer each. The rows are independent column by column, so each
/// strip of columns runs every layer while it sits in L1.
fn fused_transpose_layers(ntt: &AdditiveNttF64, sb_data: &mut [F192], layer: usize, fused: usize, sb: usize) {
    let rows = 1usize << fused;
    let row_len = sb_data.len() >> fused;
    let base = sb_data.as_mut_ptr();
    for c in (0..row_len).step_by(FUSED_STRIP.min(row_len)) {
        let w = FUSED_STRIP.min(row_len);
        for j in 0..fused {
            for r in (0..rows).filter(|r| r >> j & 1 == 0) {
                let t = ntt.twiddle(layer - j, (sb << (fused - 1 - j)) + (r >> (j + 1)));
                // SAFETY: rows `r` and `r + 2^j` are distinct, so the two strips are disjoint and inside `sb_data`.
                let (top, bot) = unsafe {
                    (
                        std::slice::from_raw_parts_mut(base.add(r * row_len + c), w),
                        std::slice::from_raw_parts_mut(base.add((r + (1 << j)) * row_len + c), w),
                    )
                };
                transpose_butterfly_ext(t, top, bot);
            }
        }
    }
}

/// `F^T` of a sparse input, keeping only the low `2^(log_d - log_inv_rate)` outputs.
///
/// The input has `positions.len()` nonzeros. Transpose step `s` mixes only within
/// `2^(s+1)`-aligned blocks, so the early steps run on the blocks that hold a
/// nonzero while those are a minority, and the rest sweep densely down to layer
/// `log_inv_rate`. Every step acts within one of the `2^log_inv_rate` replicas of
/// the message domain. The last `log_inv_rate` steps pair across replicas and
/// every kept output is a `top = a + b` of them, so they collapse to a sum over
/// the replicas: the transpose of the encode's free replication.
/// Mirror of `whir::transpose_forward_ntt_sparse`.
fn transpose_forward_ntt_sparse_ext(
    ntt: &AdditiveNttF64,
    positions: &[usize],
    values: &[F192],
    log_d: usize,
    log_inv_rate: usize,
) -> Vec<F192> {
    let _span = tracing::info_span!(
        "NTT",
        kind = "transpose induce",
        log_domain = log_d,
        nonzero = positions.len()
    )
    .entered();
    // One replica at a time, so the working set is a quarter of the domain and stays in cache.
    let n_keep = 1usize << (log_d - log_inv_rate);
    let mut kept = vec![F192::ZERO; n_keep];
    transpose_replica(ntt, positions, values, &mut kept, 0, log_d, log_inv_rate);
    let mut replica = vec![F192::ZERO; n_keep];
    for offset in (n_keep..1usize << log_d).step_by(n_keep) {
        replica.fill(F192::ZERO);
        transpose_replica(ntt, positions, values, &mut replica, offset, log_d, log_inv_rate);
        for (out, &x) in kept.iter_mut().zip(&replica) {
            *out += x;
        }
    }
    kept
}

/// The steps of [`transpose_forward_ntt_sparse_ext`] that act within the replica at `offset`, on a zeroed `data`.
fn transpose_replica(
    ntt: &AdditiveNttF64,
    positions: &[usize],
    values: &[F192],
    data: &mut [F192],
    offset: usize,
    log_d: usize,
    log_inv_rate: usize,
) {
    let mut local = Vec::with_capacity(positions.len());
    for (&p, &v) in positions.iter().zip(values) {
        if let Some(p) = p.checked_sub(offset).filter(|&p| p < data.len()) {
            data[p] += v;
            local.push(p);
        }
    }
    let mut blocks: Vec<usize> = Vec::with_capacity(local.len());
    let mut s = 0;
    while s < log_d - log_inv_rate {
        blocks.clear();
        blocks.extend(local.iter().map(|&p| p >> (s + 1)));
        blocks.sort_unstable();
        blocks.dedup();
        if 2 * (blocks.len() << (s + 1)) > data.len() {
            break;
        }
        let (layer, half) = (log_d - 1 - s, 1usize << s);
        for &b in &blocks {
            let (top, bot) = data[2 * b * half..2 * (b + 1) * half].split_at_mut(half);
            transpose_butterfly_ext(ntt.twiddle(layer, (offset >> (s + 1)) + b), top, bot);
        }
        s += 1;
    }
    transpose_layers_ext(ntt, data, offset, log_d, (log_inv_rate..log_d - s).rev());
}

/// `F^T`-based fast path for [`induce_sumcheck_poly`]: scatter per-query
/// E-weights into the codeword domain, apply `F^T` with K-twiddles, keep the
/// low `2^log_msg_cols` outputs. Byte-identical output to the dense path
/// (pinned by `induce_via_ntt_matches_dense`). Mirror of
/// `whir::induce_sumcheck_poly_via_ntt` with the L0 mixed row dot.
pub(crate) fn induce_sumcheck_poly_via_ntt_base(
    log_msg_cols: usize,
    log_inv_rate: usize,
    opened_rows: &[Vec<F64>],
    v_challenges: &[F192],
    queries: &[usize],
    weights: &[F192],
) -> (Vec<F192>, F192) {
    let log_block = log_msg_cols + log_inv_rate;
    let block_len = 1usize << log_block;
    let n_queries = queries.len();
    assert_eq!(opened_rows.len(), n_queries);

    let eq = build_eq_table_ext(v_challenges);
    debug_assert_eq!(weights.len(), n_queries);

    let mut enforced_sum = F192::ZERO;
    for i in 0..n_queries {
        enforced_sum += F64::dot(&opened_rows[i], &eq) * weights[i];
    }

    let coeffs = if log_block == 0 {
        // SAFETY: zero is a valid F192, and the loop below reads these slots.
        let mut c = vec![F192::ZERO; block_len];
        for i in 0..n_queries {
            c[queries[i]] += weights[i];
        }
        c
    } else {
        let ntt = AdditiveNttF64::standard(log_block);
        transpose_forward_ntt_sparse_ext(&ntt, queries, weights, log_block, log_inv_rate)
    };
    (coeffs, enforced_sum)
}

/// Query density threshold for choosing the sparse NTT over dense expansion.
const NTT_QUERIES_PER_BLOWUP: usize = 8;

/// Cost-based dispatch: the NTT overtakes the dense expansion once a level
/// opens more than [`NTT_QUERIES_PER_BLOWUP`] queries per unit of blowup.
#[inline]
pub(crate) fn induce_use_ntt_heuristic(log_msg_cols: usize, log_inv_rate: usize, n_queries: usize) -> bool {
    log_msg_cols >= 12 && n_queries > NTT_QUERIES_PER_BLOWUP * (1usize << log_inv_rate)
}

/// Dispatch between the dense [`induce_sumcheck_poly`] and the sparse
/// [`induce_sumcheck_poly_via_ntt_base`] for L0 (base-field rows). Mirror of
/// `whir::induce_sumcheck_poly_auto`: in the recursive PCS this fires
/// only at the top level (large message domain, many queries); deeper levels
/// stay dense. Both paths produce identical output, so a mis-dispatch only
/// costs time.
pub(crate) fn induce_sumcheck_poly_auto_base(
    log_msg_cols: usize,
    log_inv_rate: usize,
    sks_vks: &[F64],
    opened_rows: &[Vec<F64>],
    v_challenges: &[F192],
    queries: &[usize],
    weights: &[F192],
) -> (Vec<F192>, F192) {
    if induce_use_ntt_heuristic(log_msg_cols, log_inv_rate, queries.len()) {
        induce_sumcheck_poly_via_ntt_base(log_msg_cols, log_inv_rate, opened_rows, v_challenges, queries, weights)
    } else {
        induce_sumcheck_poly(log_msg_cols, sks_vks, opened_rows, v_challenges, queries, weights)
    }
}
