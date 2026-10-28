//! Additive NTT over GF(2^64) in the Lin-Chung-Han novel polynomial basis.
//!
//! It is the Reed-Solomon encoder of every WHIR commitment over K = F_{2^64}.
//!
//! - Layers run neighbors-last: layer 0 pairs rows half the domain apart.
//! - Many independent transforms share one buffer, interleaved row by row.
//! - The E-valued encodes of deeper WHIR levels reuse it, one F64 lane per F192 coefficient.
//! - Large transforms are bound by memory bandwidth, so the driver minimizes sweeps of the buffer.

use flux_profiler::timed;
use std::cell::RefCell;

use primitives::field::F64;
use crate::stack_open::EQUAL_BLOCK_LOG;
use primitives::log2_strict_usize;
use primitives::stream::Stream;

/// Where an encode stopped writing its buffer back: the deep pass's layers, still to run on any row read.
#[derive(Clone, Copy, Debug)]
pub struct DeepRows {
    log_d: usize,
    num_ntts: usize,
    /// The first layer of the encode, where the cosets split.
    start: usize,
    deep_start: usize,
}

impl DeepRows {
    /// Words of the buffer the encode left.
    pub fn words(&self) -> usize {
        self.num_ntts << self.log_d
    }

    /// Words of one deep sub-block.
    fn sub_len(&self) -> usize {
        self.num_ntts << (self.log_d - self.deep_start)
    }

    /// The runs of the sorted `positions` that fall in one deep sub-block, ascending.
    fn runs<'a>(&self, positions: &'a [usize]) -> impl Iterator<Item = &'a [usize]> {
        debug_assert!(positions.is_sorted());
        let log_sub = self.log_d - self.deep_start;
        positions.chunk_by(move |a, b| a >> log_sub == b >> log_sub)
    }

    /// The sorted, distinct deep sub-blocks that hold the sorted `positions`.
    fn sub_blocks(&self, positions: &[usize]) -> Vec<usize> {
        self.runs(positions).map(|run| run[0] >> (self.log_d - self.deep_start)).collect()
    }
}

/// `out`, `n` words per position, as one disjoint slice per run of positions, paired with the run.
fn split_runs<'a, 'b>(
    mut out: &'a mut [F64],
    n: usize,
    runs: impl Iterator<Item = &'b [usize]>,
) -> Vec<(&'a mut [F64], &'b [usize])> {
    runs.map(|run| {
        let (head, tail) = std::mem::take(&mut out).split_at_mut(run.len() * n);
        out = tail;
        (head, run)
    })
    .collect()
}

/// What [`AdditiveNttF64::encode_lane_major`] is asked to keep of its codeword.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Keep {
    /// All of it: the deep pass is written back.
    Whole,
    /// The buffer as the gathered passes leave it: [`AdditiveNttF64::finish_rows`] runs the deep pass again on a row read.
    BeforeDeep,
    /// None of it, if the plan allows: [`AdditiveNttF64::rebuild_rows`] recomputes a row from the message.
    Nothing,
}

/// What [`AdditiveNttF64::encode_lane_major`] kept of its codeword.
pub struct Encoded {
    /// The buffer kept, or empty when nothing was.
    pub words: Vec<F64>,
    /// Set if the deep pass is left to run on a row read, from `words` or, when empty, from the message.
    pub deep: Option<DeepRows>,
}

/// Table of the normalized subspace polynomials at the basis, rows end to end.
///
/// - Row `i` holds `s_i(b_j)` for every basis element `b_j` with `j >= i`, so `l - i` values.
/// - Each row is scaled so that its first entry is one.
fn generate_evals_from_subspace(basis: &[F64]) -> Vec<F64> {
    let l = basis.len();
    let mut evals = Vec::with_capacity(l * (l + 1) / 2);
    evals.extend_from_slice(basis);
    let mut prev = 0;
    for i in 1..l {
        for k in 1..=l - i {
            let (v, first) = (evals[prev + k], evals[prev]);
            evals.push(v * (v + first));
        }
        prev += l - i + 1;
    }
    let mut rest = &mut evals[..];
    for i in 0..l {
        let (row, tail) = rest.split_at_mut(l - i);
        let inv = row[0].inv();
        row.iter_mut().for_each(|v| *v *= inv);
        rest = tail;
    }
    evals
}

/// `Σ_j bit_j(idx) · basis[j]`.
#[inline]
fn span_get(basis: &[F64], idx: usize) -> F64 {
    let mut acc = F64::ZERO;
    for (j, &b) in basis.iter().enumerate() {
        if (idx >> j) & 1 == 1 {
            acc += b;
        }
    }
    acc
}

/// Receives a finished block of codeword rows, as `(first_row, rows)`.
///
/// - It is called from pool tasks.
/// - Each row is handed over exactly once.
/// - Blocks are aligned, and all of one power-of-two size.
/// - A block is handed over while its rows are still in cache.
pub type RowSink<'a> = dyn Fn(usize, &[F64]) + Sync + 'a;

/// Additive NTT over F_{2^64} with the standard polynomial-basis subspace
/// `{1, x, x², …}`: the F_2-subspace is `{0, 1, …, 2^ℓ−1}` under the natural
/// integer encoding.
#[derive(Clone, Debug)]
pub struct AdditiveNttF64 {
    /// The rows of [`generate_evals_from_subspace`], end to end.
    evals: Vec<F64>,
    /// Per eval row, the subset sums of `row[1..]` over each byte of a block index:
    /// a twiddle of a block below `2^24` is three loads and two XORs.
    tables: Vec<[[F64; 256]; 3]>,
}

impl AdditiveNttF64 {
    /// Row `i`, which starts after `l + (l - 1) + ... + (l - i + 1)` values.
    fn row(&self, i: usize) -> &[F64] {
        let l = self.tables.len();
        let start = i * l - i * i.saturating_sub(1) / 2;
        &self.evals[start..start + l - i]
    }

    fn new(basis: &[F64]) -> Self {
        let evals = generate_evals_from_subspace(basis);
        let mut rest = &evals[..];
        let tables = (0..basis.len())
            .map(|i| {
                let (row, tail) = rest.split_at(basis.len() - i);
                rest = tail;
                let basis = &row[1..];
                let mut t = [[F64::ZERO; 256]; 3];
                for (k, part) in t.iter_mut().enumerate() {
                    let sub = &basis[(8 * k).min(basis.len())..(8 * k + 8).min(basis.len())];
                    for (x, slot) in part.iter_mut().enumerate() {
                        *slot = span_get(sub, x);
                    }
                }
                t
            })
            .collect();
        Self { evals, tables }
    }

    /// Standard NTT with basis `{1, x, …, x^(dim-1)}`. Requires `dim ≤ 63` so
    /// the evaluation domain (and the twiddles) stay inside F_{2^64} without
    /// wrap; far beyond any codeword size in use.
    pub fn standard(dim: usize) -> Self {
        assert!(dim <= 63, "standard NTT requires dim ≤ 63");
        let basis: Vec<F64> = (0..dim).map(|i| F64(1u64 << i)).collect();
        Self::new(&basis)
    }

    pub fn log_domain_size(&self) -> usize {
        self.tables.len()
    }

    /// Twiddle of one block at one layer.
    ///
    /// - It is `s_i(sum_j bit_j(block) * b_(i+1+j))`, with `i = L - layer - 1` on a `2^L`-point domain.
    /// - The normalized `s_i` is F_2-linear, so this is a subset sum of row `i` of the table.
    #[inline]
    pub fn twiddle(&self, layer: usize, block: usize) -> F64 {
        let row = self.log_domain_size() - layer - 1;
        if block < 1 << 24 {
            let t = &self.tables[row];
            t[0][block & 255] + t[1][(block >> 8) & 255] + t[2][(block >> 16) & 255]
        } else {
            span_get(&self.row(row)[1..], block)
        }
    }

    /// The seven twiddles a radix-8 group needs, breadth-first: layer `layer`,
    /// then `layer + 1` (one per half), then `layer + 2` (one per quarter).
    ///
    /// `span_get` is F_2-linear in the block index, so the six deeper twiddles are
    /// the block's own contribution plus a fixed correction per sub-block index:
    /// one scan of the three basis rows replaces seven.
    pub(crate) fn twiddles_radix8(&self, layer: usize, block: usize) -> [F64; 7] {
        let l = self.log_domain_size();
        let (v1, v2) = (self.row(l - layer - 2), self.row(l - layer - 3));
        // `a` and `c` are the deeper layers' twiddles of this block's first sub-block.
        let t0 = self.twiddle(layer, block);
        let a = self.twiddle(layer + 1, 2 * block);
        let c = self.twiddle(layer + 2, 4 * block);
        let (d, e0, e1) = (v1[1], v2[1], v2[2]);
        [t0, a, a + d, c, c + e0, c + e1, c + e0 + e1]
    }

    /// RS-encode a message already stored in the codeword's first replica.
    ///
    /// # Overview
    ///
    /// - The message fills the first `1 / 2^r` of the buffer, `r` the log inverse rate.
    /// - It is row-major: word `row * n + lane` for `n` interleaved lanes.
    /// - The buffer ends as `2^r` copies of it, each transformed from layer `r` on.
    ///
    /// The replication is free: the first pass reads its rows from the message itself, see [`Self::transform`].
    pub fn encode_interleaved_in_place(&self, data: &mut [F64], num_ntts: usize, log_inv_rate: usize) {
        // The message is the buffer's own first replica.
        let msg = parallel::SendPtr(data.as_mut_ptr());
        self.transform(data, num_ntts, log_inv_rate, Some(msg), None);
    }

    /// Encode in place, handing `on_rows` every finished block of rows, see [`RowSink`].
    #[timed]
    pub fn encode_interleaved_in_place_with(
        &self,
        data: &mut [F64],
        num_ntts: usize,
        log_inv_rate: usize,
        on_rows: &RowSink<'_>,
    ) {
        let msg = parallel::SendPtr(data.as_mut_ptr());
        self.transform(data, num_ntts, log_inv_rate, Some(msg), Some(on_rows));
    }

    /// [`Self::encode_interleaved_in_place_with`] without writing the deep pass back.
    ///
    /// - Each deep sub-block is finished in scratch and handed to `on_rows` from there.
    /// - The buffer keeps what the gathered passes left, and [`Self::finish_rows`] finishes any row again.
    /// - With no separate deep pass in the plan, the buffer is the whole codeword and this returns `None`.
    ///
    /// A queried row costs one sub-block transform, against a write of the whole codeword saved.
    pub fn encode_interleaved_in_place_without_deep(
        &self,
        data: &mut [F64],
        num_ntts: usize,
        log_inv_rate: usize,
        on_rows: &RowSink<'_>,
    ) -> Option<DeepRows> {
        let msg = parallel::SendPtr(data.as_mut_ptr());
        self.transform_deferring_deep(data, num_ntts, log_inv_rate, Some(msg), Some(on_rows), true, None)
    }

    /// [`Self::encode_interleaved_with`] without writing the deep pass back, as
    /// [`Self::encode_interleaved_in_place_without_deep`] does.
    pub fn encode_interleaved_without_deep(
        &self,
        data: &mut [F64],
        msg: &[F64],
        num_ntts: usize,
        log_inv_rate: usize,
        on_rows: &RowSink<'_>,
    ) -> Option<DeepRows> {
        assert_eq!(
            msg.len() << log_inv_rate,
            data.len(),
            "the codeword is 2^log_inv_rate messages"
        );
        let msg = parallel::SendPtr(msg.as_ptr().cast_mut());
        self.transform_deferring_deep(data, num_ntts, log_inv_rate, Some(msg), Some(on_rows), true, None)
    }

    /// Encode a lane-major message, handing `on_rows` every finished block, and keep what `keep` asks for.
    ///
    /// - The result equals [`transpose_lane_major`] into block 0 followed by the in-place encode.
    /// - With a gathered first pass, that pass reads the message itself and writes every block.
    /// - So no transposed copy of the message is written and read back.
    /// - With [`Keep::Nothing`], if a single such pass covers the layers before the deep pass, the codeword is never whole:
    ///   the encode runs once per half of the cosets, in one half-size buffer, and the deep pass goes to `on_rows` only.
    /// - Otherwise the codeword is whole and the encode is as the in-place one.
    ///
    /// `repeats` tags the message's blocks of `2^EQUAL_BLOCK_LOG` words that equal their predecessor, or is empty.
    /// The stage fill copies the lines of a repeat from the stage instead of reading them, see [`stage_copies`].
    pub fn encode_lane_major(
        &self,
        msg: &[F64],
        repeats: &[bool],
        n_lanes: usize,
        num_ntts: usize,
        log_inv_rate: usize,
        keep: Keep,
        on_rows: &RowSink<'_>,
    ) -> Encoded {
        assert!(0 < n_lanes && n_lanes <= num_ntts, "a row holds its lanes");
        assert!(log_inv_rate >= 1, "a codeword has at least two cosets");
        let msg_rows = msg.len() / n_lanes;
        assert_eq!(msg.len(), n_lanes * msg_rows, "the message is n_lanes lane blocks");
        let log_d = log2_strict_usize(msg_rows) + log_inv_rate;
        let rebuildable = lane_major_deep_start(log_d, num_ntts, log_inv_rate)
            .is_some_and(|deep_start| deep_start - log_inv_rate <= fit(L2_WORDS, num_ntts));
        let by_halves = keep == Keep::Nothing && rebuildable;
        let defer_deep = by_halves || keep == Keep::BeforeDeep;
        // The cosets one pass holds: half of them when the opening rebuilds the rows.
        let held = (1usize << log_inv_rate) >> usize::from(by_halves);
        // Read-only from here on: the pointer only feeds the first pass's reads.
        let msg_ptr = parallel::SendPtr(msg.as_ptr().cast_mut());
        let mut deep = None;
        // Every word is written by the gathered pass before the deep pass reads it.
        let data = primitives::filled_vec((held * msg_rows) * num_ntts, |data| {
            for first in (0..1usize << log_inv_rate).step_by(held).rev() {
                let cosets = (msg, first..first + held, repeats);
                deep = self.transform_deferring_deep(
                    data,
                    num_ntts,
                    log_inv_rate,
                    Some(msg_ptr),
                    Some(on_rows),
                    defer_deep,
                    Some(cosets),
                );
            }
        });
        Encoded { words: if by_halves { Vec::new() } else { data }, deep }
    }

    /// RS-encode a message coset by coset, handing `on_rows` every finished block of rows and writing no codeword.
    ///
    /// - The codeword is `2^r` cosets, each an independent transform of one copy of the message, from layer `r` on.
    /// - A task builds each of its cosets in scratch and hashes it there, so no codeword is ever written.
    /// - Several cosets that are small share a scratch, so the blocks are all one size and reach the hasher in batches.
    /// - [`Self::coset_rows`] recomputes any row from the message.
    ///
    /// Use it when [`Self::encodes_by_coset`] holds: the scratch is then cache-resident.
    pub fn encode_cosets(&self, msg: &[F64], num_ntts: usize, log_inv_rate: usize, on_rows: &RowSink<'_>) {
        let log_d = self.coset_log_d(msg.len(), num_ntts, log_inv_rate);
        let rows = msg.len() / num_ntts;
        let group = 1usize << (L2_WORDS / msg.len()).max(1).ilog2().min(log_inv_rate as u32);
        parallel::for_each_chunk((1 << log_inv_rate) / group, |lo, hi| {
            with_scratch(group * msg.len(), |scratch| {
                for g in lo..hi {
                    for (i, coset) in scratch.chunks_exact_mut(msg.len()).enumerate() {
                        self.build_coset(coset, msg, num_ntts, log_d, log_inv_rate, g * group + i);
                    }
                    on_rows(g * group * rows, scratch);
                }
            });
        });
    }

    /// Codeword rows `positions` (sorted) of the encode [`Self::encode_cosets`] ran, back to back, one transform per
    /// coset they touch.
    pub fn coset_rows(&self, msg: &[F64], num_ntts: usize, log_inv_rate: usize, positions: &[usize]) -> Vec<F64> {
        debug_assert!(positions.is_sorted());
        let log_d = self.coset_log_d(msg.len(), num_ntts, log_inv_rate);
        let rows = msg.len() / num_ntts;
        let mut out = vec![F64::ZERO; positions.len() * num_ntts];
        let mut parts = split_runs(&mut out, num_ntts, positions.chunk_by(|a, b| a / rows == b / rows));
        parallel::chunks_mut(&mut parts, 1, |_, part| {
            let (dst, run) = &mut part[0];
            with_scratch(msg.len(), |coset| {
                self.build_coset(coset, msg, num_ntts, log_d, log_inv_rate, run[0] / rows);
                for (dst, &q) in dst.chunks_exact_mut(num_ntts).zip(run.iter()) {
                    dst.copy_from_slice(&coset[(q % rows) * num_ntts..][..num_ntts]);
                }
            });
        });
        out
    }

    /// Whether a message of `msg_words` words encodes coset by coset: one coset fits the cache budget [`COSET_WORDS`].
    pub fn encodes_by_coset(msg_words: usize) -> bool {
        msg_words <= COSET_WORDS
    }

    /// The log of the codeword's rows, for a message of `msg_words` words.
    fn coset_log_d(&self, msg_words: usize, num_ntts: usize, log_inv_rate: usize) -> usize {
        assert!(num_ntts > 0 && msg_words.is_multiple_of(num_ntts));
        let log_d = log2_strict_usize(msg_words / num_ntts) + log_inv_rate;
        assert!(log_d <= self.log_domain_size());
        log_d
    }

    /// Coset `coset` of the codeword, built into `buf` (one message long) from the message.
    fn build_coset(&self, buf: &mut [F64], msg: &[F64], num_ntts: usize, log_d: usize, log_inv_rate: usize, coset: usize) {
        self.run_layers_from(msg, buf, log_d, num_ntts, log_inv_rate, log_d, log_inv_rate, coset, no_hook);
    }

    /// Codeword rows `positions` (sorted) of a buffer left by [`Self::encode_interleaved_in_place_without_deep`],
    /// back to back.
    pub fn finish_rows(&self, data: &[F64], deep: &DeepRows, positions: &[usize]) -> Vec<F64> {
        let sub_len = deep.sub_len();
        self.rows_from(deep, positions, |_, sub| &data[sub * sub_len..][..sub_len])
    }

    /// Codeword rows `positions` (sorted) of an encode that kept no buffer, recomputed from its lane-major message,
    /// back to back.
    pub fn rebuild_rows(
        &self,
        msg: &[F64],
        repeats: &[bool],
        n_lanes: usize,
        deep: &DeepRows,
        positions: &[usize],
    ) -> Vec<F64> {
        let built = self.rebuild_sub_blocks(msg, repeats, n_lanes, deep, &deep.sub_blocks(positions));
        let sub_len = deep.sub_len();
        self.rows_from(deep, positions, |i, _| &built[i * sub_len..][..sub_len])
    }

    /// The rows at `positions` (sorted), back to back, from the sub-blocks before their deep layers.
    ///
    /// `src(i, sub)` is the `i`-th sub-block, number `sub` in the codeword. Each finishes into the thread's
    /// scratch, so only the queried rows are written.
    fn rows_from<'a>(
        &self,
        deep: &DeepRows,
        positions: &[usize],
        src: impl Fn(usize, usize) -> &'a [F64] + Sync,
    ) -> Vec<F64> {
        let (sub_len, log_sub, n) = (deep.sub_len(), deep.log_d - deep.deep_start, deep.num_ntts);
        let mut out = vec![F64::ZERO; positions.len() * n];
        let mut parts = split_runs(&mut out, n, deep.runs(positions));
        parallel::chunks_mut(&mut parts, 1, |i, part| {
            let (dst, run) = &mut part[0];
            let sub_idx = run[0] >> log_sub;
            with_scratch(sub_len, |sub| {
                self.finish_sub_block(Some(src(i, sub_idx)), sub, deep, sub_idx);
                for (dst, &q) in dst.chunks_exact_mut(n).zip(run.iter()) {
                    dst.copy_from_slice(&sub[(q & ((1 << log_sub) - 1)) * n..][..n]);
                }
            });
        });
        out
    }

    /// The deep layers over sub-block `sub_idx` of the codeword, which `src` holds as the gathered pass left it.
    ///
    /// With no `src`, `sub` holds it and the layers run in place.
    fn finish_sub_block(&self, src: Option<&[F64]>, sub: &mut [F64], deep: &DeepRows, sub_idx: usize) {
        let DeepRows { log_d, num_ntts, deep_start, .. } = *deep;
        self.run_layers_to(sub, src, log_d, num_ntts, deep_start, log_d, deep_start, sub_idx, None, no_hook);
    }

    /// The deep sub-blocks `subs` (sorted, distinct) of an encode that kept no buffer, back to back, before their deep layers.
    ///
    /// - The encode ran one gathered pass over its lane-major message: a group is one residue of one coset,
    ///   and its transform leaves row `i` of the group as row `r` of sub-block `coset << g | i`, `g` layers deep.
    /// - So each group transforms again here, skipping the blocks that feed no row of `subs`,
    ///   and keeps only the rows that fall in a sub-block of `subs`.
    /// - Those rows are every word of the sub-blocks before their deep layers, which [`Self::rebuild_rows`] runs.
    ///
    /// # Panics
    ///
    /// Panics unless the encode had the plan this reads back: one lane-major gathered pass.
    #[timed]
    fn rebuild_sub_blocks(
        &self,
        msg: &[F64],
        repeats: &[bool],
        n_lanes: usize,
        deep: &DeepRows,
        subs: &[usize],
    ) -> Vec<F64> {
        let DeepRows { log_d, num_ntts, start, deep_start } = *deep;
        let g = deep_start - start;
        let (rows, log_step) = (1usize << g, log_d - deep_start);
        let (sub_len, msg_rows) = (deep.sub_len(), 1usize << (log_d - start));
        assert!(0 < g && g <= (L2_WORDS / num_ntts).max(1).ilog2() as usize, "one gathered pass");
        assert!(num_ntts.is_multiple_of(8) && log_step >= LOG_BATCH, "the gathered pass is lane-major");
        assert_eq!(msg.len(), n_lanes * msg_rows);
        // Each coset with queried sub-blocks: its index, those sub-blocks' places in `subs`, and their group rows.
        let mut next = 0;
        let cosets: Vec<(usize, std::ops::Range<usize>, Vec<usize>)> = subs
            .chunk_by(|a, b| a >> g == b >> g)
            .map(|run| {
                next += run.len();
                (run[0] >> g, next - run.len()..next, run.iter().map(|sub| sub & (rows - 1)).collect())
            })
            .collect();
        let copies = stage_copies(repeats, log_step, rows, msg_rows, n_lanes, num_ntts);
        // A group writes row `r` of every queried sub-block, and the groups cover every residue `r`.
        primitives::filled_vec(subs.len() * sub_len, |built| {
            let out = parallel::SendPtr(built.as_mut_ptr());
            parallel::for_each_chunk((1usize << log_step) / BATCH, |lo, hi| {
                // One octet of lanes of a batch of residues is 64 lanes of one transform.
                let stage_len = rows * BATCH * 8;
                // `built` is far larger than a node's L3 share and is read only after every task: its lines stream.
                let stream = primitives::stream::Stream::new();
                with_scratch(2 * stage_len, |all| {
                    let (stage, scratch) = all.split_at_mut(stage_len);
                    for t in lo..hi {
                        for octet in 0..num_ntts / 8 {
                            fill_stage(stage, msg, t * BATCH, log_step, msg_rows, n_lanes, num_ntts, octet, &copies, 1);
                            for (coset, run, live) in &cosets {
                                let live_rows = Some(&live[..]);
                                self.run_layers_to(scratch, Some(stage), deep_start, BATCH * 8, start, deep_start, start, *coset, live_rows, no_hook);
                                for (pos, &i) in run.clone().zip(live) {
                                    for (s, line) in scratch[i * BATCH * 8..][..BATCH * 8].chunks_exact(8).enumerate() {
                                        // SAFETY: this task alone owns row `t * BATCH + s` of every sub-block,
                                        // and `built` outlives the tasks.
                                        let dst = unsafe { out.slice(pos * sub_len + (t * BATCH + s) * num_ntts + 8 * octet, 8) };
                                        stream.copy(dst, line);
                                    }
                                }
                            }
                        }
                    }
                });
            });
        })
    }

    /// RS-encode a message held in a buffer of its own, handing `on_rows` every finished block of rows.
    ///
    /// - The result equals encoding in place.
    /// - Every codeword word is written before it is read, so the codeword may start uninitialized.
    /// - The blocks are as for the in-place encode.
    ///
    /// # Panics
    ///
    /// Panics unless the codeword is exactly `2^r` messages long.
    #[timed]
    pub fn encode_interleaved_with(
        &self,
        data: &mut [F64],
        msg: &[F64],
        num_ntts: usize,
        log_inv_rate: usize,
        on_rows: &RowSink<'_>,
    ) {
        assert_eq!(
            msg.len() << log_inv_rate,
            data.len(),
            "the codeword is 2^log_inv_rate messages"
        );
        // Read-only from here on: the pointer only feeds the first pass's reads.
        let msg = parallel::SendPtr(msg.as_ptr().cast_mut());
        self.transform(data, num_ntts, log_inv_rate, Some(msg), Some(on_rows));
    }

    /// Run layers `start..d` of a `2^d`-row transform in as few sweeps of the buffer as possible.
    ///
    /// A large transform is bound by memory bandwidth, so its cost is its number of sweeps.
    ///
    /// # Plan
    ///
    /// ```text
    ///     layers:  start ............ deep_start ............ d
    ///              |---- gathered passes ----|---- deep pass ----|
    ///
    ///     gathered pass:  one task per row group of 2^g rows, 2^(d - layer - g) apart
    ///                     gathered into L2 scratch, all g layers run there, written back
    ///     deep pass:      one task per run of contiguous sub-blocks, at least an L2 budget of rows,
    ///                     its layers run in place (or in scratch, when a separate message has to be copied in)
    /// ```
    ///
    /// - When every layer fits one L3-sized sub-block, the deep pass is the whole transform: one sweep.
    /// - Otherwise gathered passes run until the rest fits one L2-sized sub-block.
    /// - With `n` words a row, a block of up to `(L2 words / n)^2` rows then costs two sweeps.
    ///
    /// # With a message
    ///
    /// - The layer-`start` blocks are not in the buffer yet: each is a copy of the message.
    /// - The first pass reads its rows from the message instead.
    ///
    /// # Why block 0 goes last
    ///
    /// - The message may be the buffer's own block 0.
    /// - So a gathered task visits blocks in descending order, and overwrites block 0 last.
    /// - Tasks own disjoint row residues, so no task reads rows another task writes.
    ///
    /// # Finished rows
    ///
    /// - Each deep sub-block is final, and hands its rows to `on_rows` while they are in L2.
    /// - Without a split deep pass, the rows go over in parallel blocks at the end.
    fn transform(
        &self,
        data: &mut [F64],
        num_ntts: usize,
        start: usize,
        msg: Option<parallel::SendPtr<F64>>,
        on_rows: Option<&RowSink<'_>>,
    ) {
        self.transform_deferring_deep(data, num_ntts, start, msg, on_rows, false, None);
    }

    /// [`Self::transform`]; with `defer_deep`, the deep pass's results go only to `on_rows`, not back into the buffer.
    ///
    /// With `lane_major` lanes, `msg` is a separate lane-major message, which the first gathered pass reads.
    /// A plan with no such pass transposes it into block 0 of the buffer and encodes in place.
    /// Its `cosets` are the layer-`start` blocks the buffer holds, which need that one pass and no more.
    /// Twiddles and rows handed to `on_rows` stay global.
    #[allow(clippy::too_many_arguments)]
    fn transform_deferring_deep(
        &self,
        data: &mut [F64],
        num_ntts: usize,
        start: usize,
        msg: Option<parallel::SendPtr<F64>>,
        on_rows: Option<&RowSink<'_>>,
        defer_deep: bool,
        lane_major: Option<(&[F64], std::ops::Range<usize>, &[bool])>,
    ) -> Option<DeepRows> {
        // The buffer is 2^log_d rows of `num_ntts` words, or the layer-`start` blocks `cosets` of them.
        let (lane_major, cosets, repeats) = match lane_major {
            Some((words, cosets, repeats)) => (Some(words), cosets, repeats),
            None => (None, 0..1 << start, &[][..]),
        };
        assert!(num_ntts > 0);
        assert_eq!(data.len() % num_ntts, 0);
        let log_d = log2_strict_usize(data.len() / num_ntts / cosets.len()) + start;
        assert!(log_d <= self.log_domain_size());
        assert!(start <= log_d);
        // The message is `n_lanes` lane blocks of `2^(log_d - start)` words.
        let mut lane_major = lane_major.map(|words| (words, words.len() >> (log_d - start)));

        let l2_words = l2_words(lane_major.is_some());
        let (par_log, deep_start) = (par_log(log_d), deep_start(log_d, num_ntts, start, l2_words));
        let fit2 = fit(l2_words, num_ntts);

        // A buffer this large is evicted before the next pass reads it back.
        let stream = data.len() >= STREAM_MIN_WORDS;

        // Phase 1: gathered passes, each at most one L2 group of layers.
        //
        // Only the first one reads the message.
        let mut msg = msg;
        // A lane-major message is read by the first gathered pass, a batch of residues at a time.
        // Without such a pass, or with too few residues to batch, it is transposed into block 0 first.
        if let (Some((words, n_lanes)), Some(_)) = (lane_major, msg) && lane_major_deep_start(log_d, num_ntts, start).is_none() {
            assert_eq!(cosets.len(), 1 << start, "a partial buffer needs the gathered pass");
            let log_rows = log_d - start;
            transpose_lane_major(&mut data[..num_ntts << log_rows], words, n_lanes, num_ntts, log_rows);
            msg = Some(parallel::SendPtr(data.as_mut_ptr()));
            lane_major = None;
        }
        let mut layer = start;
        while layer < deep_start {
            let g = (deep_start - layer).min(fit2);
            let lane_major = lane_major.take().map(|(words, n_lanes)| (words, n_lanes, repeats));
            self.gathered_pass(data, log_d, num_ntts, layer, g, msg.take(), lane_major, stream, cosets.start);
            layer += g;
        }

        // Phase 2: replicas a deep-only plan cannot build inside its tasks.
        //
        //     message is block 0 of the buffer  ->  a task could overwrite it while others still copy it
        //     no layer left to run              ->  no deep task would write the buffer at all
        if let Some(m) = msg.take_if(|m| std::ptr::eq(m.0, data.as_mut_ptr()) || deep_start == log_d) {
            replicate(data, m, data.len() >> start);
        }

        // A lone sub-block runs on this thread, so its rows go over in parallel afterwards.
        let fuse = 0 < deep_start && deep_start < log_d;
        let deep_rows = on_rows.filter(|_| fuse);

        // Phase 3: the deep pass, one task per run of contiguous sub-blocks.
        if deep_start < log_d {
            // A task takes at least an L2 budget of rows, so a high rate's tiny sub-blocks
            // still reach the hasher in whole batches.
            let log_sub = (log_d - deep_start).max(fit2.min(log_d - par_log));
            let sub_len = num_ntts << log_sub;
            let block_len = data.len() >> start;
            let outer = log_d - log_sub;
            // While a sub-block transforms, the next one is prefetched into L2, spread over its row groups.
            let (base, total) = (parallel::SendPtr(data.as_mut_ptr()), data.len());
            let hooks = hook_calls(log_d, deep_start, log_d, outer);
            let prefetch_next = |sub_idx: usize| {
                let next = (sub_idx + 1) * sub_len;
                prefetch_spread(&base, next, if next < total { sub_len } else { 0 }, hooks)
            };
            match msg {
                // A separate message: build each sub-block in scratch, hash it, then write it out once.
                //
                // Streamed out whole, the codeword is written without ever being read.
                Some(m) => {
                    // Whole cache lines in every row, so a finished sub-block can drain a line at a time.
                    let lines_ok = num_ntts.is_multiple_of(8) && base.0.align_offset(64) == 0;
                    let deferred = stream && lines_ok;
                    let drain_per_hook = (sub_len / 8).div_ceil(hooks);
                    parallel::for_each_chunk(1 << outer, |lo, hi| {
                        // A draining task has two scratches: one transforms while the other empties.
                        with_scratch(sub_len * (1 + usize::from(deferred)), |both| {
                            // One fence at the end of the task covers all its streaming stores.
                            let stream = stream.then(Stream::new);
                            let (mut scratch, mut idle) = both.split_at_mut(sub_len);
                            let mut drain = Drain::default();
                            for sub_idx in lo..hi {
                                // A task sits at the same offset in every replica, or covers whole replicas.
                                //
                                //     task at offset off of its replica  <-  message words [off, off + sub_len)
                                let off = (sub_idx * sub_len) % block_len;
                                // Inside one replica the first sweep reads the message itself; across several it is copied in.
                                let mut from = None;
                                for part in scratch.chunks_mut(block_len) {
                                    // SAFETY:
                                    // - The external message is valid for one whole replica of words.
                                    // - It is disjoint from the codeword.
                                    // - Each part ends inside its replica, so the read stays in bounds.
                                    let words = unsafe { std::slice::from_raw_parts(m.add(off), part.len()) };
                                    if sub_len <= block_len {
                                        from = Some(words);
                                    } else {
                                        part.copy_from_slice(words);
                                    }
                                }
                                match &stream {
                                    Some(s) if deferred => {
                                        // The previous sub-block drains while this one transforms.
                                        let hook = || drain.step(Some(s), drain_per_hook);
                                        self.run_layers_to(
                                            scratch,
                                            from,
                                            log_d,
                                            num_ntts,
                                            deep_start,
                                            log_d,
                                            outer,
                                            sub_idx,
                                            None,
                                            hook,
                                        );
                                        drain.step(Some(s), usize::MAX);
                                        if let Some(f) = deep_rows {
                                            f(sub_idx << log_sub, scratch);
                                        }
                                        // This one drains during the next transform, while the other scratch is filled.
                                        drain = Drain {
                                            // SAFETY: the sub-block lies inside the codeword.
                                            dst: unsafe { base.add(sub_idx * sub_len) },
                                            src: scratch.as_ptr(),
                                            row_len: num_ntts,
                                            dst_stride: num_ntts,
                                            rows: 1 << log_sub,
                                            ..Drain::default()
                                        };
                                        std::mem::swap(&mut scratch, &mut idle);
                                    }
                                    _ => {
                                        self.run_layers_to(
                                            scratch,
                                            from,
                                            log_d,
                                            num_ntts,
                                            deep_start,
                                            log_d,
                                            outer,
                                            sub_idx,
                                            None,
                                            no_hook,
                                        );
                                        if let Some(f) = deep_rows {
                                            f(sub_idx << log_sub, scratch);
                                        }
                                        // SAFETY: this task alone owns the sub-block of the codeword.
                                        let dst = unsafe { base.slice(sub_idx * sub_len, sub_len) };
                                        match &stream {
                                            Some(s) => s.copy(dst, scratch),
                                            None => dst.copy_from_slice(scratch),
                                        }
                                    }
                                }
                            }
                            if let Some(s) = &stream {
                                drain.step(Some(s), usize::MAX);
                            }
                        });
                    });
                }
                None => parallel::chunks_mut(data, sub_len, |local, sub| {
                    // The buffer's first sub-block, in the whole domain.
                    let sub_idx = local + (cosets.start << (log_d - start - log_sub));
                    let hook = prefetch_next(local);
                    if defer_deep && deep_rows.is_some() {
                        // The rows are hashed from scratch and the buffer keeps its pre-deep state.
                        with_scratch(sub_len, |scratch| {
                            self.run_layers_from(sub, scratch, log_d, num_ntts, deep_start, log_d, outer, sub_idx, hook);
                            if let Some(f) = deep_rows {
                                f(sub_idx << log_sub, scratch);
                            }
                        });
                    } else {
                        // The replicas are already in place: run the layers where they are.
                        self.run_layers(sub, log_d, num_ntts, deep_start, log_d, outer, sub_idx, hook);
                        if let Some(f) = deep_rows {
                            f(sub_idx << log_sub, sub);
                        }
                    }
                }),
            }
        }

        if let Some(f) = on_rows.filter(|_| !fuse) {
            // Four blocks per worker.
            let log_tasks = (4 * parallel::num_threads()).next_power_of_two().ilog2() as usize;
            let log_rows = log_d - log_tasks.min(log_d);
            parallel::chunks_mut(data, num_ntts << log_rows, |i, rows| f(i << log_rows, rows));
        }
        (defer_deep && deep_rows.is_some() && msg.is_none()).then_some(DeepRows {
            log_d,
            num_ntts,
            start,
            deep_start,
        })
    }

    /// Run a range of layers in place over one sub-block of the domain.
    ///
    /// - The domain has `2^d` rows and splits into `2^o` equal sub-blocks.
    /// - The buffer is one of them; its index fixes the global block index, and so the twiddle, of each block.
    /// - Layers fuse into radix-8 sweeps, with radix-4 sweeps where a lone radix-2 would be left over.
    /// - `hook` runs once after each row group of each sweep.
    #[allow(clippy::too_many_arguments)]
    fn run_layers(
        &self,
        buf: &mut [F64],
        log_d: usize,
        num_ntts: usize,
        first_layer: usize,
        end_layer: usize,
        outer_log: usize,
        sub_idx: usize,
        hook: impl FnMut(),
    ) {
        self.run_layers_to(buf, None, log_d, num_ntts, first_layer, end_layer, outer_log, sub_idx, None, hook);
    }

    /// [`Self::run_layers`] over a copy of `src` that lands in `buf`: the first sweep reads `src` and writes `buf`.
    #[allow(clippy::too_many_arguments)]
    fn run_layers_from(
        &self,
        src: &[F64],
        buf: &mut [F64],
        log_d: usize,
        num_ntts: usize,
        first_layer: usize,
        end_layer: usize,
        outer_log: usize,
        sub_idx: usize,
        hook: impl FnMut(),
    ) {
        self.run_layers_to(buf, Some(src), log_d, num_ntts, first_layer, end_layer, outer_log, sub_idx, None, hook);
    }

    /// [`Self::run_layers`] that leaves the blocks alone which hold none of the buffer rows `live` (sorted).
    ///
    /// - Only those rows come out right: a block feeds only the rows inside it, in every later layer.
    /// - A fused sweep skips whole blocks of its first layer, so a sparse `live` prunes coarsely.
    ///
    /// With a `src` as long as `buf`, the layers run over a copy of it, which the first sweep makes.
    /// That sweep loads from `src` and stores to `buf`, and the blocks it skips are not copied.
    /// Later sweeps work inside the blocks of the first, so they never read a word that was not copied.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_layers_to(
        &self,
        buf: &mut [F64],
        mut src: Option<&[F64]>,
        log_d: usize,
        num_ntts: usize,
        first_layer: usize,
        end_layer: usize,
        outer_log: usize,
        sub_idx: usize,
        live: Option<&[usize]>,
        mut hook: impl FnMut(),
    ) {
        if let Some(src) = src.filter(|_| first_layer >= end_layer) {
            buf.copy_from_slice(src);
        }
        let mut layer = first_layer;
        while layer < end_layer {
            // This layer's blocks inside the buffer: how many, and their size in rows and words.
            let num_blocks_in_buf = 1usize << (layer - outer_log);
            let block_size = 1usize << (log_d - layer);
            let block_elems = block_size * num_ntts;
            // A block's index in the whole domain, which picks its twiddle.
            let global = |block_in_buf: usize| sub_idx * num_blocks_in_buf + block_in_buf;

            let radix = radix_layers(end_layer - layer);
            if radix == 3 {
                // One radix-8 sweep.
                let eighth = block_size >> 3;
                for_each_block(num_blocks_in_buf, log_d - layer, live, |block_in_buf| {
                    let t = self.twiddles_radix8(layer, global(block_in_buf));
                    let start = block_in_buf * block_elems;
                    butterfly_interleaved_fused_3layer(
                        &mut buf[start..start + block_elems],
                        src.map(|s| &s[start..start + block_elems]),
                        &t,
                        eighth,
                        num_ntts,
                        &mut hook,
                    );
                });
                layer += 3;
            } else if radix == 2 {
                // One radix-4 sweep.
                let quarter = block_size >> 2;
                for_each_block(num_blocks_in_buf, log_d - layer, live, |block_in_buf| {
                    let global_block = global(block_in_buf);
                    let t = [
                        self.twiddle(layer, global_block),
                        self.twiddle(layer + 1, 2 * global_block),
                        self.twiddle(layer + 1, 2 * global_block + 1),
                    ];
                    let start = block_in_buf * block_elems;
                    butterfly_interleaved_fused_2layer(
                        &mut buf[start..start + block_elems],
                        src.map(|s| &s[start..start + block_elems]),
                        &t,
                        quarter,
                        num_ntts,
                        &mut hook,
                    );
                });
                layer += 2;
            } else {
                // One layer: a plain butterfly sweep.
                let block_size_half = block_size >> 1;
                for_each_block(num_blocks_in_buf, log_d - layer, live, |block_in_buf| {
                    let twiddle = self.twiddle(layer, global(block_in_buf));
                    let start = block_in_buf * block_elems;
                    // A lone layer is rare: its block is copied before it runs, not fused with it.
                    if let Some(src) = src {
                        buf[start..start + block_elems].copy_from_slice(&src[start..start + block_elems]);
                    }
                    butterfly_interleaved_block(
                        &mut buf[start..start + block_elems],
                        twiddle,
                        block_size_half,
                        num_ntts,
                        &mut hook,
                    );
                });
                layer += 1;
            }
            src = None;
        }
    }

    /// Recover novel-basis coefficients from evaluations with a scalar inverse NTT.
    #[timed]
    pub fn inverse_transform(&self, data: &mut [F64]) {
        let log_d = log2_strict_usize(data.len());
        assert!(log_d <= self.log_domain_size());
        for layer in (0..log_d).rev() {
            let num_blocks = 1usize << layer;
            let block_size_half = 1usize << (log_d - layer - 1);
            for block in 0..num_blocks {
                let twiddle = self.twiddle(layer, block);
                let block_start = block << (log_d - layer);
                for idx0 in block_start..(block_start + block_size_half) {
                    let idx1 = idx0 | block_size_half;
                    let u = data[idx0];
                    let new_v = data[idx1] + u;
                    data[idx1] = new_v;
                    data[idx0] = u + new_v * twiddle;
                }
            }
        }
    }
}

/// How many rows, as a log, a cache budget of `words` holds.
///
/// At 51 lanes: 2^17 / 51 -> 2^11 rows in L2,  2^18 / 51 -> 2^12 rows in L3.
fn fit(words: usize, num_ntts: usize) -> usize {
    (words / num_ntts).max(1).ilog2() as usize
}

/// From 2^12 rows on, a deep pass cuts at least 2^LOG_SUBS_PER_WORKER sub-blocks per worker: this log of them.
fn par_log(log_d: usize) -> usize {
    /// Below this domain the transform is too small to split across the pool.
    const PARALLEL_FLOOR_LOG_D: usize = 12;
    /// Fewest rows a deep sub-block takes when splitting for parallelism.
    const MIN_SUB_LOG: usize = 8;
    if log_d >= PARALLEL_FLOOR_LOG_D {
        (log2_strict_usize(parallel::num_threads().next_power_of_two()) + LOG_SUBS_PER_WORKER).min(log_d - MIN_SUB_LOG)
    } else {
        0
    }
}

/// Where the deep pass of a `2^log_d`-row transform from layer `start` starts.
///
/// - If every layer fits one L3 sub-block, the deep pass is the whole transform: one sweep.
/// - Otherwise the deep pass keeps one L2 sub-block of layers.
fn deep_start(log_d: usize, num_ntts: usize, start: usize, l2_words: usize) -> usize {
    let deep = if log_d - start <= fit(L3_WORDS, num_ntts) {
        fit(L3_WORDS, num_ntts)
    } else {
        fit(l2_words, num_ntts)
    };
    log_d.saturating_sub(deep).max(par_log(log_d)).max(start)
}

/// The deep pass start of a lane-major encode, if its first gathered pass can read the message:
/// a layer for it to run, whole cache lines in a row, and enough residues to batch.
fn lane_major_deep_start(log_d: usize, num_ntts: usize, start: usize) -> Option<usize> {
    let deep_start = deep_start(log_d, num_ntts, start, L2_WORDS);
    let g = (deep_start - start).min(fit(L2_WORDS, num_ntts));
    (start < deep_start && num_ntts.is_multiple_of(8) && log_d - start - g >= LOG_BATCH).then_some(deep_start)
}

/// Call `f` with each block of `1 << log_rows` rows that holds a row of `live` (sorted), or with every block of `num_blocks`.
fn for_each_block(num_blocks: usize, log_rows: usize, live: Option<&[usize]>, mut f: impl FnMut(usize)) {
    match live {
        None => (0..num_blocks).for_each(f),
        Some(rows) => rows.chunk_by(|a, b| a >> log_rows == b >> log_rows).for_each(|run| f(run[0] >> log_rows)),
    }
}

/// How many times [`AdditiveNttF64::run_layers`] calls its hook over `first_layer..end_layer`.
pub(super) fn hook_calls(log_d: usize, first_layer: usize, end_layer: usize, outer_log: usize) -> usize {
    let (mut layer, mut calls) = (first_layer, 0);
    while layer < end_layer {
        let block_size = 1usize << (log_d - layer);
        let radix = radix_layers(end_layer - layer);
        // Every block of the sweep calls once per row group, and a group takes `2^radix` rows.
        calls += (1usize << (layer - outer_log)) * (block_size >> radix);
        layer += radix;
    }
    calls
}

/// A finished row group, leaving its scratch for the buffer a piece at a time.
///
/// - `dst` is the group's first row in the buffer, `dst_stride` words before its next.
/// - A scratch row is `row_len` words and lands in pieces of `piece` words, `piece_stride` words apart.
/// - `src` is the group's contiguous scratch, and `row`, `k` are the next piece to move.
/// - A streaming drain moves whole aligned cache lines: `piece` is 8, and the caller checks the alignment.
pub(super) struct Drain {
    pub(super) dst: *mut F64,
    pub(super) src: *const F64,
    pub(super) row_len: usize,
    pub(super) dst_stride: usize,
    pub(super) piece: usize,
    pub(super) piece_stride: usize,
    pub(super) rows: usize,
    pub(super) row: usize,
    pub(super) k: usize,
}

impl Default for Drain {
    /// A group with no rows left to move, in whole lines.
    fn default() -> Self {
        Self {
            dst: std::ptr::null_mut(),
            src: std::ptr::null(),
            row_len: 0,
            dst_stride: 0,
            piece: 8,
            piece_stride: 8,
            rows: 0,
            row: 0,
            k: 0,
        }
    }
}

impl Drain {
    /// Move up to `pieces` more pieces: streamed, or stored normally without a stream.
    #[inline(always)]
    pub(super) fn step(&mut self, stream: Option<&Stream>, pieces: usize) {
        let per_row = self.row_len / self.piece;
        for _ in 0..pieces.min((self.rows - self.row) * per_row - self.k) {
            // SAFETY:
            // - The cursor is inside the group, so both pieces are in bounds, and line-aligned when streamed.
            // - Only this group's owner writes these rows of the buffer.
            // - The scratch is neither written nor gathered into until its group has drained.
            unsafe {
                let dst = self.dst.add(self.row * self.dst_stride + self.k * self.piece_stride);
                let src = self.src.add(self.row * self.row_len + self.k * self.piece);
                match stream {
                    Some(s) => s.copy_lines(dst.cast::<Line>(), &*src.cast::<Line>()),
                    None => dst.copy_from_nonoverlapping(src, self.piece),
                }
            }
            self.k += 1;
            if self.k == per_row {
                (self.row, self.k) = (self.row + 1, 0);
            }
        }
    }
}

/// A hook that prefetches `words` words at `base + start` into L2, evenly over `calls` calls.
///
/// - A sub-block that fills the L2 cannot be prefetched whole before its predecessor is done.
/// - Spread over the predecessor's sweeps, the lines arrive while it is still working.
fn prefetch_spread(base: &parallel::SendPtr<F64>, start: usize, words: usize, calls: usize) -> impl FnMut() {
    let lines = words / 8;
    let per_call = lines.div_ceil(calls.max(1));
    // Only an address is formed, never a reference.
    let first = base.0.wrapping_add(start);
    let mut done = 0;
    move || {
        let end = (done + per_call).min(lines);
        #[cfg(target_arch = "x86_64")]
        for line in done..end {
            // SAFETY: a prefetch of any address is harmless.
            unsafe {
                let line = first.wrapping_add(8 * line).cast();
                core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T1 }>(line)
            };
        }
        done = end;
    }
}

/// The hook of a sweep nobody listens to.
///
/// One named function, so every such caller shares one instance of the sweeps.
fn no_hook() {}

/// How many layers the next sweep fuses, with `left` layers still to run.
///
/// - Radix 8 while three or more remain, but never leave one layer for a lone radix-2 sweep.
/// - A radix-2 sweep costs about twice a radix-8 layer, a radix-4 sweep about 1.4 times.
/// - So when one layer would be left over, a radix-4 sweep goes first and another ends the run.
/// - A block of the next sweep's first layer has `2^left` rows or more, so it always fits the radix.
fn radix_layers(left: usize) -> usize {
    if left >= 4 && left % 3 == 1 { 2 } else { left.min(3) }
}

/// Visit every row group of a fused multi-layer block.
///
/// ```text
///     the block is N slabs of equal height
///
///     slab 0:     row 0   row 1   ...   row r   ...
///     slab 1:     row 0   row 1   ...   row r   ...
///     ...
///     slab N-1:   row 0   row 1   ...   row r   ...
///
///     group r  =  row r of every slab
/// ```
///
/// - Each slab hands out its rows in turn, so the groups are disjoint without any pointer arithmetic.
/// - A source, when given, is a separate buffer of the block's length, cut the same way.
/// - With no source, a group is read in place.
#[inline(always)]
fn fused_rows<const N: usize>(
    block: &mut [F64],
    src: Option<&[F64]>,
    stride_rows: usize,
    num_ntts: usize,
    do_one: impl Fn(Option<[&[F64]; N]>, &mut [&mut [F64]; N]),
    mut hook: impl FnMut(),
) {
    // Words from one slab to the next.
    let stride = stride_rows * num_ntts;
    let len = block.len();
    debug_assert_eq!(len, N * stride);
    let mut slabs = block.chunks_exact_mut(stride);
    let mut block_rows: [_; N] = std::array::from_fn(|_| slabs.next().unwrap().chunks_exact_mut(num_ntts));
    let mut src_rows = src.map(|src| {
        debug_assert_eq!(src.len(), len);
        let mut slabs = src.chunks_exact(stride);
        std::array::from_fn::<_, N, _>(|_| slabs.next().unwrap().chunks_exact(num_ntts))
    });
    for _ in 0..stride_rows {
        let mut rows: [&mut [F64]; N] = block_rows.each_mut().map(|slab| slab.next().unwrap());
        let srcs = src_rows
            .as_mut()
            .map(|slabs| slabs.each_mut().map(|slab| slab.next().unwrap()));
        do_one(srcs, &mut rows);
        hook();
    }
}

/// Layers L, L+1 and L+2 fused into one sweep over a layer-L block.
///
/// ```text
///     a group is 8 rows, e = (block rows) / 8 apart:  r, r + e, ..., r + 7e
///
///     layer L     pairs rows 4e apart
///     layer L+1   pairs rows 2e apart
///     layer L+2   pairs rows  e apart
/// ```
///
/// - The eight rows stay in L1 across all twelve butterflies.
/// - The seven twiddles are breadth-first: one for layer L, two for L+1, four for L+2.
fn butterfly_interleaved_fused_3layer(
    block: &mut [F64],
    src: Option<&[F64]>,
    t: &[F64; 7],
    eighth: usize,
    num_ntts: usize,
    hook: impl FnMut(),
) {
    // On AVX-512 the whole sweep is one target-feature function, so each group's butterflies run on registers
    // inside the sweep's loop: a call per group would pass the vectors through memory.
    #[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
    {
        #[inline]
        #[target_feature(enable = "vpclmulqdq", enable = "avx512f", enable = "avx512bw")]
        fn sweep(
            block: &mut [F64],
            src: Option<&[F64]>,
            t: &[F64; 7],
            eighth: usize,
            num_ntts: usize,
            hook: impl FnMut(),
        ) {
            fused_sweep_avx512::<8, 7>(block, src, eighth, num_ntts, t, hook, #[inline(always)] |v, t| {
                // Layer L: rows 4 apart, one twiddle for the whole block.
                butterfly_regs(v, 0, 4, t[0]);
                butterfly_regs(v, 1, 5, t[0]);
                butterfly_regs(v, 2, 6, t[0]);
                butterfly_regs(v, 3, 7, t[0]);
                // Layer L+1: rows 2 apart, one twiddle per half.
                butterfly_regs(v, 0, 2, t[1]);
                butterfly_regs(v, 1, 3, t[1]);
                butterfly_regs(v, 4, 6, t[2]);
                butterfly_regs(v, 5, 7, t[2]);
                // Layer L+2: adjacent rows, one twiddle per quarter.
                butterfly_regs(v, 0, 1, t[3]);
                butterfly_regs(v, 2, 3, t[4]);
                butterfly_regs(v, 4, 5, t[5]);
                butterfly_regs(v, 6, 7, t[6]);
            });
        }
        // SAFETY: this code is compiled only where VPCLMULQDQ and AVX-512F are enabled, and every such
        // machine runs the AVX-512BW that the sweep adds.
        unsafe { sweep(block, src, t, eighth, num_ntts, hook) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
    fused_rows::<8>(block, src, eighth, num_ntts, |src, rows| radix8_butterflies(src, rows, t), hook);
}

/// The sweep of [`fused_rows`] with `kernel` on registers: [`fused_lanes_avx512`] for each row group.
#[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
#[inline]
#[target_feature(enable = "vpclmulqdq", enable = "avx512f", enable = "avx512bw")]
fn fused_sweep_avx512<const N: usize, const T: usize>(
    block: &mut [F64],
    src: Option<&[F64]>,
    stride_rows: usize,
    num_ntts: usize,
    twiddles: &[F64; T],
    hook: impl FnMut(),
    kernel: impl Fn(&mut RowVectors<N>, &[Twiddle; T]),
) {
    fused_rows::<N>(
        block,
        src,
        stride_rows,
        num_ntts,
        |src, rows| fused_lanes_avx512(src, rows, twiddles, #[inline(always)] |v, t| kernel(v, t)),
        hook,
    );
}

/// The twelve butterflies of one radix-8 row group, with its seven twiddles breadth-first.
#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
#[inline(always)]
fn radix8_butterflies(src: Option<[&[F64]; 8]>, rows: &mut [&mut [F64]; 8], t: &[F64; 7]) {
    copy_rows(src, rows);
    let [r0, r1, r2, r3, r4, r5, r6, r7] = rows;
    // Layer L: rows 4 apart, one twiddle for the whole block.
    butterfly_lanes(r0, r4, t[0]);
    butterfly_lanes(r1, r5, t[0]);
    butterfly_lanes(r2, r6, t[0]);
    butterfly_lanes(r3, r7, t[0]);
    // Layer L+1: rows 2 apart, one twiddle per half.
    butterfly_lanes(r0, r2, t[1]);
    butterfly_lanes(r1, r3, t[1]);
    butterfly_lanes(r4, r6, t[2]);
    butterfly_lanes(r5, r7, t[2]);
    // Layer L+2: adjacent rows, one twiddle per quarter.
    butterfly_lanes(r0, r1, t[3]);
    butterfly_lanes(r2, r3, t[4]);
    butterfly_lanes(r4, r5, t[5]);
    butterfly_lanes(r6, r7, t[6]);
}

/// Copy the rows that are read from a source into the rows that are written, where the two differ.
#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
#[inline(always)]
fn copy_rows<const N: usize>(src: Option<[&[F64]; N]>, rows: &mut [&mut [F64]; N]) {
    for (from, row) in src.iter().flatten().zip(rows.iter_mut()) {
        row.copy_from_slice(from);
    }
}

/// Transpose a lane-major message into the row-major order the encoder reads.
///
/// # Layout
///
/// ```text
///     input, lane-major:   [ block 0 | block 1 | ... | block n-1 ]   each 2^rows_log words
///     output, row-major:   row x = [ 0, ..., 0, block n-1 word x, ..., block 0 word x ]
/// ```
///
/// - A row is `row_words >= n` words: the first `row_words - n` lanes are zero padding.
/// - Codeword lane `t` past the padding takes message block `row_words - 1 - t`.
/// - Any lane count works: a commitment encodes only the lanes that carry data, and a caller pads
///   the row to the vector width.
///
/// # Why descending
///
/// - The absent all-zero lanes lead every Merkle leaf image, so the proof can leave them out.
/// - `whir::commit` has the full reason.
///
/// # Performance
///
/// - Rows are handled one tile at a time.
/// - Each lane adds a contiguous burst of words to an L1-resident tile.
/// - The finished tile goes out as one contiguous run.
/// - That keeps an n-way gather at a `2^rows_log` stride near memory bandwidth.
#[timed]
pub(super) fn transpose_lane_major(out: &mut [F64], msg: &[F64], n_lanes: usize, row_words: usize, log_rows: usize) {
    let rows = 1usize << log_rows;
    assert!(n_lanes > 0, "a commitment needs at least one lane");
    assert!(n_lanes <= row_words, "a row holds its lanes");
    assert_eq!(msg.len(), n_lanes * rows, "message is n_lanes contiguous lane blocks");
    assert_eq!(out.len(), row_words * rows, "the output is whole padded rows");

    /// Words per cache-resident row tile.
    const TILE_WORDS: usize = 4096;
    assert!(row_words <= TILE_WORDS, "a codeword row must fit the transpose tile");
    // Largest power-of-two row count whose tile fits: both it and `rows` are then
    // powers of two, so the tiles cover every row. They have to: this is the sole
    // initializer of an uninitialized codeword's message region, and a truncating
    // tile count would leave the tail reading the previous phase's plausible bytes,
    // whose symptom is a proof that stops verifying rather than a crash.
    let tile_rows = (1usize << (TILE_WORDS / row_words).ilog2()).min(rows);
    assert_eq!(rows % tile_rows, 0, "row tiles must cover every row");

    // Nothing reads the output back before a buffer this large is evicted.
    let stream = out.len() >= STREAM_MIN_WORDS;
    let pad = row_words - n_lanes;
    parallel::chunks_mut(out, tile_rows * row_words, |t, out_tile| {
        with_scratch(out_tile.len(), |tile| {
            // First message row this tile covers.
            let r0 = t * tile_rows;
            // Fill the tile column by column: lane `lane` takes every `row_words`-th word.
            for lane in 0..row_words {
                let slots = tile[lane..].iter_mut().step_by(row_words);
                if lane < pad {
                    slots.for_each(|slot| *slot = F64::ZERO);
                } else {
                    let src = &msg[(row_words - 1 - lane) * rows + r0..][..tile_rows];
                    for (slot, &word) in slots.zip(src) {
                        *slot = word;
                    }
                }
            }
            // Write the finished tile out as one contiguous run.
            if stream {
                Stream::new().copy(out_tile, tile);
            } else {
                out_tile.copy_from_slice(tile);
            }
        });
    });
}

/// Words one L2-resident unit of work may span: a gathered row group or a deep sub-block.
///
/// # Why this value
///
/// - 2^17 words is 1 MiB, a whole Zen 5 L2.
/// - Two SMT threads then spill each other's units into L3, which costs far less than a DRAM sweep.
/// - At 51 lanes it plans one gathered pass plus the deep pass, where 2^16 words planned two plus one.
///
/// # On Apple silicon
///
/// - A deep layer costs about the same whether its sub-block fits a cache or not: the transform is compute bound.
/// - A gathered layer costs more than a deep one, its rows scattered over more streams than the prefetcher follows.
/// - So the budget is sized to push layers into the deep pass, not to fit a cache.
/// - It exceeds [`L3_WORDS`], which then changes no plan.
#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
const L2_WORDS: usize = 1 << 17;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
const L2_WORDS: usize = 1 << 21;

/// [`L2_WORDS`] for an encode that has no lane-major message, on a pool of at most 4 threads.
///
/// # Why this value
///
/// - 2^15 words is 256 KiB: a deep sub-block of 2^9 rows of 48 words is 192 KiB.
/// - One thread per core owns its whole 1 MiB L2, and the sub-block leaves room for the twiddles and the prefetched next sub-block.
/// - The plan then has one more gathered layer and fewer deep ones.
/// - Measured on the 8:1 extension encode of 2^22 words at 4 threads: 130 -> 123 ms.
/// - 2^14 words plans a second gathered pass and costs 184 ms.
/// - The lane-major base encode keeps [`L2_WORDS`]: 2^16 words plans a second gathered pass there and costs 40 %.
/// - Larger pools keep [`L2_WORDS`]: their nodes share the L3 and the SMT threads share an L2.
fn l2_words(lane_major: bool) -> usize {
    /// Apple silicon keeps [`L2_WORDS`] whatever the plan.
    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    const FEW_THREADS_L2_WORDS: usize = 1 << 15;
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    const FEW_THREADS_L2_WORDS: usize = L2_WORDS;
    if lane_major || parallel::num_threads() > 4 { L2_WORDS } else { FEW_THREADS_L2_WORDS }
}

/// Deep sub-blocks cut per worker, as a log.
///
/// - Apple silicon mixes performance and efficiency cores.
/// - With one sub-block each, the performance cores would wait on the efficiency cores' sub-blocks.
/// - Elsewhere every core is alike, and one each suffices.
#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
const LOG_SUBS_PER_WORKER: usize = 0;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
const LOG_SUBS_PER_WORKER: usize = 2;

/// Words a deep sub-block may span when it alone covers every layer.
///
/// # Why this value
///
/// - 2^18 words is 2 MiB, one thread's share of L3.
/// - Such a sub-block spills from L2 into L3.
/// - That costs extra L3 traffic, but saves a whole sweep of DRAM.
const L3_WORDS: usize = 1 << 18;

/// Words of one coset a coset-major encode may span.
///
/// # Why this value
///
/// - 2^19 words is 4 MiB: the largest coset it admits is 2^13 rows of 48 words, 3 MiB.
/// - A coset runs in a scratch that lives in L2 and L3, and never reaches DRAM.
/// - Four nodes of two threads each share one 32 MiB L3, so a worker's share is 4 MiB.
/// - A larger coset spills to DRAM, and then the codeword's write pass is the cheaper plan.
const COSET_WORDS: usize = 1 << 19;

/// Buffers of at least this many words get streaming stores for data a pass does not read back.
///
/// # Why this value
///
/// - 2^23 words is 64 MiB, more than L3 keeps until the next pass.
/// - A streaming store then skips the read an ordinary store makes before it writes.
/// - A smaller buffer stays in L3 for the next pass, which a streaming store would give up.
const STREAM_MIN_WORDS: usize = 1 << 23;

/// Residues a lane-major first pass takes at once, as a log: one cache line of a lane block.
const LOG_BATCH: usize = 3;
pub(super) const BATCH: usize = 1 << LOG_BATCH;
/// Most batches a lane-major task fills side by side, so a lane's adjacent lines are read together.
const STAGE_SUBS: usize = 2;

/// Batches a lane-major task fills side by side, as one stage of that many batches.
///
/// # Why this value
///
/// - A batch reads one 64 byte line per lane block and row, 16 KB from the next row's.
/// - Each such line opens its own DRAM page: 4 threads read it at 20 GB/s, where 128 byte runs reach 31 GB/s.
/// - A second batch's line is the neighbour of the first's, so the fill reads both in one pass.
/// - The stage doubles to 2^(g + 10) words, which spills from L2 into L3.
/// - Measured on the base encode at 8:1 (rate 2 / rate 1, ms): 1 thread 944 -> 919 / 540 -> 504,
///   2 threads 496 -> 482 / 287 -> 270, 4 threads 294 -> 286 / 169 -> 162.
/// - At 8 threads it changes nothing, and at 16 threads it costs 3 %: the stage then crowds the shared L2 and L3.
/// - So a pool of more than 4 threads keeps one batch.
/// - A step under two batches has no neighbour line to read.
pub(super) fn stage_subs(lane_major: bool, log_step: usize) -> usize {
    let fits = log_step >= LOG_BATCH + STAGE_SUBS.ilog2() as usize;
    if lane_major && fits && parallel::num_threads() <= 4 { STAGE_SUBS } else { 1 }
}

/// One cache line of words.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Line([F64; 8]);

/// Lend this thread's scratch buffer, grown to the requested length.
///
/// - It is cache-line aligned, so a full-width load never splits a line.
/// - It lives as long as the thread, so no pass allocates in its hot loop.
pub(super) fn with_scratch<R>(len: usize, f: impl FnOnce(&mut [F64]) -> R) -> R {
    thread_local! {
        static SCRATCH: RefCell<Vec<Line>> = const { RefCell::new(Vec::new()) };
    }
    SCRATCH.with_borrow_mut(|lines| {
        // Grow only; a later, smaller request reuses the same lines.
        if lines.len() * 8 < len {
            lines.resize(len.div_ceil(8), Line([F64::ZERO; 8]));
        }
        // SAFETY:
        // - A line is exactly eight contiguous words, with no padding.
        // - The buffer holds at least the requested words, all initialized.
        f(unsafe { std::slice::from_raw_parts_mut(lines.as_mut_ptr().cast::<F64>(), len) })
    })
}

/// The tiles of a stage that repeat an earlier tile, as a bit per lane of each octet.
pub(super) struct StageCopies {
    /// A repeat tile is a copy of the tile this many rows before it.
    back: usize,
    rows: usize,
    n_octets: usize,
    /// Per `t0 >> EQUAL_BLOCK_LOG`, octet of lanes and row of the stage: bit `k` is set if lane `8 * octet + k`
    /// of the tile repeats the tile `back` rows before.
    masks: Vec<u8>,
}

impl StageCopies {
    pub(super) fn none() -> Self {
        Self { back: 0, rows: 0, n_octets: 0, masks: Vec::new() }
    }

    /// The masks of one octet of the batch of residues at `t0`, or none when no tile of the stage repeats.
    fn octet(&self, octet: usize, t0: usize) -> &[u8] {
        let at = ((t0 >> EQUAL_BLOCK_LOG) * self.n_octets + octet) * self.rows;
        self.masks.get(at..at + self.rows).unwrap_or(&[])
    }
}

/// Which tiles of a lane-major stage are copies of the tile `back` rows before them.
///
/// - Stage row `i` of residue `r` is message row `r + i * 2^log_step` of every lane. Going back `back` rows
///   is going back `D = max(2^log_step, 2^EQUAL_BLOCK_LOG)` words: a whole number `m` of blocks.
/// - The line of row `i` equals the line of row `i - back` if the lane's `m` blocks up to the one holding
///   the line each repeat their predecessor: a run of repeats at least `m` long ends there.
/// - The block holding the line is `(i << log_step >> EQUAL_BLOCK_LOG) + u`, with `u = r >> EQUAL_BLOCK_LOG`
///   below `m`, so one mask table serves every residue of a `u`.
/// - A padding lane is zero in both rows, so its bit is set too.
/// - Rows before `back` read the message, as do all rows of a message with no map.
///
/// `repeats` is the message's map of blocks that equal their predecessor. A lane's first block never repeats.
pub(super) fn stage_copies(
    repeats: &[bool],
    log_step: usize,
    rows: usize,
    msg_rows: usize,
    n_lanes: usize,
    row_words: usize,
) -> StageCopies {
    if repeats.is_empty() || !msg_rows.is_multiple_of(1 << EQUAL_BLOCK_LOG) {
        return StageCopies::none();
    }
    let distance = (1usize << log_step).max(1 << EQUAL_BLOCK_LOG);
    let (back, m) = (distance >> log_step, distance >> EQUAL_BLOCK_LOG);
    if back >= rows {
        return StageCopies::none();
    }
    let (pad, n_octets, per_lane) = (row_words - n_lanes, row_words / 8, msg_rows >> EQUAL_BLOCK_LOG);
    let mut masks = vec![0u8; m * n_octets * rows];
    // The length of the run of repeats ending at each block of the lane.
    let mut run = vec![0usize; per_lane];
    for lane in pad..row_words {
        let blocks = &repeats[(row_words - 1 - lane) * per_lane..][..per_lane];
        let mut length = 0;
        for (h, (&repeat, run)) in blocks.iter().zip(&mut run).enumerate() {
            length = if repeat && h > 0 { length + 1 } else { 0 };
            *run = length;
        }
        for u in 0..m {
            let column = &mut masks[(u * n_octets + lane / 8) * rows..][..rows];
            for (i, mask) in column.iter_mut().enumerate().skip(back) {
                if run[(i << log_step >> EQUAL_BLOCK_LOG) + u] >= m {
                    *mask |= 1 << (lane % 8);
                }
            }
        }
    }
    for lane in 0..pad {
        for u in 0..m {
            let column = &mut masks[(u * n_octets + lane / 8) * rows..][..rows];
            column.iter_mut().skip(back).for_each(|mask| *mask |= 1 << (lane % 8));
        }
    }
    StageCopies { back, rows, n_octets, masks }
}

/// Transpose one octet of lanes of `subs` batches of [`BATCH`] consecutive residues of a lane-major message into `stage`.
///
/// - `stage` is `subs` stages of one batch each, and batch `u` starts at residue `t0 + u * BATCH`.
/// - A row's tiles of all batches go together: the lines of one lane block that sit next to each other are read next to each other.
/// - Message row `j` is `msg_rows` words apart per lane block, and word `w` of a row of `row_words` words
///   is lane block `row_words - 1 - w` of the message, or zero while `w` is below the padding.
/// - Residue `t0 + s` takes rows `t0 + s + i * step` for `i < rows`, `step = 2^log_step`.
/// - The octet is lanes `8 * octet..8 * octet + 8`. Together with the residues, they are 64 lanes of one transform.
/// - `stage` has `rows` rows of 64 words: lane `8 * octet + k` of row `t0 + s + i * step` is word `(i * BATCH + s) * 8 + k`.
/// - One tile is 8 words of 8 lane blocks, so each 64 byte line of a lane block is read once, whole.
/// - A lane of tile `i` with its bit set in the octet's [`StageCopies`] is copied from the tile `back` rows
///   before, which is already in the stage, and its line is not read.
pub(super) fn fill_stage(
    stage: &mut [F64],
    msg: &[F64],
    t0: usize,
    log_step: usize,
    msg_rows: usize,
    n_lanes: usize,
    row_words: usize,
    octet: usize,
    copies: &StageCopies,
    subs: usize,
) {
    // SAFETY: AVX-512F is enabled at compile time.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe {
        fill_stage_tiles(stage, msg, t0, log_step, msg_rows, n_lanes, row_words, octet, copies, subs)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    fill_stage_tiles(stage, msg, t0, log_step, msg_rows, n_lanes, row_words, octet, copies, subs);
}

/// [`fill_stage`], a target-feature function where AVX-512 is enabled, so its tiles move through registers.
#[cfg_attr(all(target_arch = "x86_64", target_feature = "avx512f"), target_feature(enable = "avx512f"), inline)]
fn fill_stage_tiles(
    stage: &mut [F64],
    msg: &[F64],
    t0: usize,
    log_step: usize,
    msg_rows: usize,
    n_lanes: usize,
    row_words: usize,
    octet: usize,
    copies: &StageCopies,
    subs: usize,
) {
    let back = copies.back;
    let sub_words = stage.len() / subs;
    let rows = sub_words / (BATCH * 8);
    let pad = row_words - n_lanes;
    // Lane block of line `k` of the tile, which is lane `8 * octet + k`.
    let block = |k: usize| row_words - 1 - (8 * octet + k);
    // The lines of a tile that are padding lanes.
    let pad_lines = (0..8).filter(|k| 8 * octet + k < pad).fold(0u8, |m, k| m | 1 << k);
    let masks: [&[u8]; STAGE_SUBS] = std::array::from_fn(|u| if u < subs { copies.octet(octet, t0 + u * BATCH) } else { &[] });
    for i in 0..rows {
        // The sub-batches' lines of one lane are adjacent: a row's tiles go together.
        for (u, masks) in masks.iter().enumerate().take(subs) {
            let j = t0 + u * BATCH + (i << log_step);
            let sub = &mut stage[u * sub_words..][..sub_words];
            let copy = masks.get(i).copied().unwrap_or(0);
            if copy != 0xFF {
                // Line `k` of the tile: 8 residues of one lane block, or zeros for a padding lane or a copy.
                let src: [usize; 8] = std::array::from_fn(|k| block(k) * msg_rows + j);
                fill_tile(sub, i * BATCH, msg, src, pad_lines | copy);
            }
            if copy != 0 {
                // Tile `i - back` is a tile of the stage, and `copies` has no bit before row `back`.
                copy_tile(sub, i, i - back, copy);
            }
        }
    }
}

/// Write `sub[tile + s][k] = msg[src[k] + s]` for the 8 x 8 words of eight lines, a line in `zeros` being zeros.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline]
#[target_feature(enable = "avx512f")]
fn fill_tile(sub: &mut [F64], tile: usize, msg: &[F64], src: [usize; 8], zeros: u8) {
    use core::arch::x86_64::*;
    use primitives::simd::{load, store};
    let r: [__m512i; 8] =
        std::array::from_fn(|k| if zeros >> k & 1 == 1 { _mm512_setzero_si512() } else { load(&msg[src[k]..], 0) });
    // Pairs of columns: lanes `[(r0, r1), (r2, r3)]`, then 128-bit lane picks to finish.
    let (a0, a1) = (_mm512_unpacklo_epi64(r[0], r[1]), _mm512_unpackhi_epi64(r[0], r[1]));
    let (a2, a3) = (_mm512_unpacklo_epi64(r[2], r[3]), _mm512_unpackhi_epi64(r[2], r[3]));
    let (a4, a5) = (_mm512_unpacklo_epi64(r[4], r[5]), _mm512_unpackhi_epi64(r[4], r[5]));
    let (a6, a7) = (_mm512_unpacklo_epi64(r[6], r[7]), _mm512_unpackhi_epi64(r[6], r[7]));
    let b0 = _mm512_shuffle_i64x2::<0x88>(a0, a2);
    let b1 = _mm512_shuffle_i64x2::<0x88>(a1, a3);
    let b2 = _mm512_shuffle_i64x2::<0xDD>(a0, a2);
    let b3 = _mm512_shuffle_i64x2::<0xDD>(a1, a3);
    let b4 = _mm512_shuffle_i64x2::<0x88>(a4, a6);
    let b5 = _mm512_shuffle_i64x2::<0x88>(a5, a7);
    let b6 = _mm512_shuffle_i64x2::<0xDD>(a4, a6);
    let b7 = _mm512_shuffle_i64x2::<0xDD>(a5, a7);
    let cols = [
        _mm512_shuffle_i64x2::<0x88>(b0, b4),
        _mm512_shuffle_i64x2::<0x88>(b1, b5),
        _mm512_shuffle_i64x2::<0x88>(b2, b6),
        _mm512_shuffle_i64x2::<0x88>(b3, b7),
        _mm512_shuffle_i64x2::<0xDD>(b0, b4),
        _mm512_shuffle_i64x2::<0xDD>(b1, b5),
        _mm512_shuffle_i64x2::<0xDD>(b2, b6),
        _mm512_shuffle_i64x2::<0xDD>(b3, b7),
    ];
    for (s, col) in cols.into_iter().enumerate() {
        store(sub, tile + s, col);
    }
}

/// [`fill_tile`] word by word.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn fill_tile(sub: &mut [F64], tile: usize, msg: &[F64], src: [usize; 8], zeros: u8) {
    for s in 0..8 {
        for k in 0..8 {
            sub[(tile + s) * 8 + k] = if zeros >> k & 1 == 1 { F64::ZERO } else { msg[src[k] + s] };
        }
    }
}

/// Copy the lanes of tile `from` of a stage into tile `i`, for the bits set in `mask`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline]
#[target_feature(enable = "avx512f")]
fn copy_tile(sub: &mut [F64], i: usize, from: usize, mask: u8) {
    use core::arch::x86_64::*;
    use primitives::simd::{load, store};
    for s in 0..BATCH {
        let (kept, copied) = (load(sub, i * BATCH + s), load(sub, from * BATCH + s));
        store(sub, i * BATCH + s, _mm512_mask_blend_epi64(mask, kept, copied));
    }
}

/// [`copy_tile`] word by word.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn copy_tile(sub: &mut [F64], i: usize, from: usize, mask: u8) {
    for s in 0..BATCH {
        for k in (0..8).filter(|k| mask >> k & 1 == 1) {
            sub[(i * BATCH + s) * 8 + k] = sub[(from * BATCH + s) * 8 + k];
        }
    }
}

/// Fill every replica of the buffer with a copy of the message.
///
/// - This is the unfused form of a gathered first pass.
/// - The message may be the buffer's own first replica, which is then left as it is.
fn replicate(data: &mut [F64], msg: parallel::SendPtr<F64>, msg_len: usize) {
    // Copy granularity: small enough to spread a short message over every worker.
    const CHUNK: usize = 1 << 14;
    let replicas = data.len() / msg_len;
    let in_place = std::ptr::eq(msg.0, data.as_mut_ptr());
    let chunks = msg_len.div_ceil(CHUNK);
    let dst = parallel::SendPtr(data.as_mut_ptr());
    // Replica index innermost, so one worker copies a chunk into several replicas while it is cached.
    parallel::for_each(chunks * replicas, |t| {
        let (c, replica) = (t / replicas, t % replicas);
        // The message already occupies replica 0.
        if in_place && replica == 0 {
            return;
        }
        let start = c * CHUNK;
        let len = CHUNK.min(msg_len - start);
        // SAFETY:
        // - Only this task writes this range.
        // - The range skips replica 0, so it never overlaps a message stored there.
        unsafe {
            let src = std::slice::from_raw_parts(msg.add(start), len);
            dst.slice(replica * msg_len + start, len).copy_from_slice(src);
        }
    });
}

/// Test oracle: fill every replica with a copy of a message held elsewhere.
#[cfg(test)]
fn replicate_rows(data: &mut [F64], msg: &[F64]) {
    for replica in data.chunks_mut(msg.len()) {
        replica.copy_from_slice(msg);
    }
}

/// Layers L and L+1 fused into one sweep over a layer-L block, four rows at a time.
///
/// The three twiddles are breadth-first: one for layer L, two for L+1.
fn butterfly_interleaved_fused_2layer(
    block: &mut [F64],
    src: Option<&[F64]>,
    t: &[F64; 3],
    quarter: usize,
    num_ntts: usize,
    hook: impl FnMut(),
) {
    // On AVX-512 the sweep is one target-feature function, like the radix-8 one.
    #[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
    {
        #[inline]
        #[target_feature(enable = "vpclmulqdq", enable = "avx512f", enable = "avx512bw")]
        fn sweep(
            block: &mut [F64],
            src: Option<&[F64]>,
            t: &[F64; 3],
            quarter: usize,
            num_ntts: usize,
            hook: impl FnMut(),
        ) {
            fused_sweep_avx512::<4, 3>(block, src, quarter, num_ntts, t, hook, #[inline(always)] |v, t| {
                // Layer L: rows 2 apart, one twiddle for the block.
                butterfly_regs(v, 0, 2, t[0]);
                butterfly_regs(v, 1, 3, t[0]);
                // Layer L+1: adjacent rows, one twiddle per half.
                butterfly_regs(v, 0, 1, t[1]);
                butterfly_regs(v, 2, 3, t[2]);
            });
        }
        // SAFETY: as in `butterfly_interleaved_fused_3layer`.
        unsafe { sweep(block, src, t, quarter, num_ntts, hook) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
    fused_rows::<4>(block, src, quarter, num_ntts, |src, rows| radix4_butterflies(src, rows, t), hook);
}

/// The four butterflies of one radix-4 row group.
#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
#[inline(always)]
fn radix4_butterflies(src: Option<[&[F64]; 4]>, rows: &mut [&mut [F64]; 4], t: &[F64; 3]) {
    copy_rows(src, rows);
    let [r0, r1, r2, r3] = rows;
    // Layer L: rows 2 apart, one twiddle for the block.
    butterfly_lanes(r0, r2, t[0]);
    butterfly_lanes(r1, r3, t[0]);
    // Layer L+1: adjacent rows, one twiddle per half.
    butterfly_lanes(r0, r1, t[1]);
    butterfly_lanes(r2, r3, t[2]);
}

#[inline]
fn butterfly_interleaved_block(
    block: &mut [F64],
    twiddle: F64,
    block_size_half: usize,
    num_ntts: usize,
    mut hook: impl FnMut(),
) {
    let half_offset = block_size_half * num_ntts;
    let (top, bot) = block.split_at_mut(half_offset);
    for r in 0..block_size_half {
        let off = r * num_ntts;
        butterfly_lanes(&mut top[off..off + num_ntts], &mut bot[off..off + num_ntts], twiddle);
        hook();
    }
}

/// Butterfly all `num_ntts` lanes of one (top row, bottom row) pair with a
/// shared twiddle: new_u = u + v*t; new_v = v + new_u.
///
/// On NEON this processes eight lanes per iteration. The pair kernel handles a short even tail,
/// and the scalar path handles an odd tail.
#[inline]
fn butterfly_lanes(top: &mut [F64], bot: &mut [F64], twiddle: F64) {
    debug_assert_eq!(top.len(), bot.len());
    // The lanes the vector kernels cover; the scalar loop at the end takes the rest.
    let done: usize;
    // Every layer's block 0 has twiddle zero: the butterfly is then `v' = v + u`, no product.
    if twiddle == F64::ZERO {
        for (v, &u) in bot.iter_mut().zip(top.iter()) {
            *v += u;
        }
        return;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
    {
        // SAFETY: this code is compiled only where VPCLMULQDQ and AVX-512F are enabled, and every such
        // machine runs the AVX-512BW that the kernel adds.
        unsafe { butterfly_lanes_avx512(top, bot, twiddle.0) };
        done = top.len();
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "pclmulqdq",
        target_feature = "avx2",
        not(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))
    ))]
    {
        // SAFETY: this code is compiled only where PCLMULQDQ and AVX2 are enabled, and every such
        // machine runs the kernel's features.
        unsafe { butterfly_lanes_avx2(top, bot, twiddle.0) };
        done = top.len();
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        // SAFETY: the aes target feature is enabled at compile time, and NEON is in the aarch64 baseline.
        done = unsafe { butterfly_lanes_neon(top, bot, twiddle.0) };
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq", target_feature = "avx2")
    )))]
    {
        done = 0;
    }
    for lane in done..top.len() {
        let v = bot[lane];
        let new_u = top[lane] + v * twiddle;
        top[lane] = new_u;
        bot[lane] = v + new_u;
    }
}

/// The transpose of [`butterfly_lanes`]: `s = u + v; u' = s; v' = s * t + v` over a pair of rows.
/// Only AVX-512 has a kernel; other targets take the scalar loop.
pub(crate) fn transpose_butterfly_lanes(top: &mut [F64], bot: &mut [F64], twiddle: F64) {
    debug_assert_eq!(top.len(), bot.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
    // SAFETY: as in `butterfly_lanes`.
    unsafe {
        transpose_butterfly_lanes_avx512(top, bot, twiddle.0)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
    for (u, v) in top.iter_mut().zip(bot.iter_mut()) {
        let s = *u + *v;
        *u = s;
        *v += s * twiddle;
    }
}

/// [`butterfly_lanes_avx512`] at half the width, for a machine with AVX2 but not both
/// VPCLMULQDQ and AVX-512.
///
/// A tail of fewer than four lanes goes through a zero-padded vector, where the butterfly of zeros is zero.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "pclmulqdq",
    target_feature = "avx2",
    not(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))
))]
#[inline]
#[target_feature(enable = "pclmulqdq", enable = "avx2")]
#[cfg_attr(target_feature = "vpclmulqdq", target_feature(enable = "vpclmulqdq"))]
fn butterfly_lanes_avx2(top: &mut [F64], bot: &mut [F64], twiddle: u64) {
    use core::arch::x86_64::*;
    use primitives::{load_at, store_at};

    let vectors = top.len() / 4;
    for i in 0..vectors {
        let (new_u, new_v) = butterfly_ymm(load_at(top, 4 * i), load_at(bot, 4 * i), twiddle);
        store_at(top, 4 * i, new_u);
        store_at(bot, 4 * i, new_v);
    }
    let tail = top.len() - 4 * vectors;
    if tail != 0 {
        let (mut u, mut v) = ([F64::ZERO; 4], [F64::ZERO; 4]);
        u[..tail].copy_from_slice(&top[4 * vectors..]);
        v[..tail].copy_from_slice(&bot[4 * vectors..]);
        let (new_u, new_v) = butterfly_ymm(load_at::<_, __m256i>(&u, 0), load_at::<_, __m256i>(&v, 0), twiddle);
        store_at(&mut u, 0, new_u);
        store_at(&mut v, 0, new_v);
        top[4 * vectors..].copy_from_slice(&u[..tail]);
        bot[4 * vectors..].copy_from_slice(&v[..tail]);
    }
}

/// Four butterflies `u' = u + v * t`, `v' = v + u'` in registers: the products one per 128-bit
/// lane, even lanes then odd lanes as in `butterfly_zmm`, then one shift-and-XOR reduction.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "pclmulqdq",
    target_feature = "avx2",
    not(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))
))]
#[inline]
#[target_feature(enable = "pclmulqdq", enable = "avx2")]
#[cfg_attr(target_feature = "vpclmulqdq", target_feature(enable = "vpclmulqdq"))]
fn butterfly_ymm(
    u: core::arch::x86_64::__m256i,
    v: core::arch::x86_64::__m256i,
    twiddle: u64,
) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
    use core::arch::x86_64::*;
    use primitives::field::gf2_64x3::x86_64::reduce_halves256;

    #[cfg(target_feature = "vpclmulqdq")]
    let (even, odd) = {
        let tw = _mm256_set1_epi64x(twiddle as i64);
        (
            _mm256_clmulepi64_epi128::<0x00>(v, tw),
            _mm256_clmulepi64_epi128::<0x11>(v, tw),
        )
    };
    // Four 128-bit multiplies; immediate bit 0 picks the qword of `v`.
    #[cfg(not(target_feature = "vpclmulqdq"))]
    let (even, odd) = {
        let tw = _mm_cvtsi64_si128(twiddle as i64);
        let (v01, v23) = (_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
        (
            _mm256_set_m128i(
                _mm_clmulepi64_si128::<0x00>(v23, tw),
                _mm_clmulepi64_si128::<0x00>(v01, tw),
            ),
            _mm256_set_m128i(
                _mm_clmulepi64_si128::<0x01>(v23, tw),
                _mm_clmulepi64_si128::<0x01>(v01, tw),
            ),
        )
    };
    let new_u = _mm256_xor_si256(u, reduce_halves256(even, odd));
    (new_u, _mm256_xor_si256(v, new_u))
}

/// The butterflies of one row pair that the NEON kernels cover: whole groups of eight lanes, then pairs.
/// Returns how many lanes it did.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
#[target_feature(enable = "neon,aes")]
fn butterfly_lanes_neon(top: &mut [F64], bot: &mut [F64], twiddle: u64) -> usize {
    let mut lane = top.len() / 8 * 8;
    for (top, bot) in top.chunks_exact_mut(8).zip(bot.chunks_exact_mut(8)) {
        butterfly_lanes_neon_8(top, bot, twiddle);
    }
    while lane + 2 <= top.len() {
        butterfly_lane_pair_neon(&mut top[lane..lane + 2], &mut bot[lane..lane + 2], twiddle);
        lane += 2;
    }
    lane
}

/// Eight F64 butterflies as four independent NEON lane-pair reductions.
/// Loading all four bottom vectors before reducing them gives the out-of-order
/// core four independent PMULL chains to schedule, while one call amortizes
/// loop control and the duplicated twiddle/reduction constants over 8 lanes.
///
/// `top` and `bot` are eight words each.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
#[target_feature(enable = "neon,aes")]
fn butterfly_lanes_neon_8(top: &mut [F64], bot: &mut [F64], twiddle: u64) {
    use core::arch::aarch64::*;
    use primitives::field::gf2_64::aarch64::{pmull, pmull_hi, reduce_pair_pmull4};
    use primitives::{load_at, store_at};

    let v: [uint64x2_t; 4] = std::array::from_fn(|i| load_at(bot, 2 * i));
    let tw = vdupq_n_u64(twiddle);
    for i in 0..4 {
        let prod = reduce_pair_pmull4(pmull(vgetq_lane_u64::<0>(v[i]), twiddle), pmull_hi(v[i], tw));
        let new_u = veorq_u64(load_at(top, 2 * i), prod);
        store_at(top, 2 * i, new_u);
        store_at(bot, 2 * i, veorq_u64(v[i], new_u));
    }
}

/// `g(spill)` per top nibble `n` of a product's high word: `spill = n ^ n>>1 ^ n>>3`, the bits
/// `g(hi) = hi * (x^4 + x^3 + x + 1)` pushes past x^63, then `g(spill)` itself, which fits a byte.
///
/// The 16 entries repeat in each 128-bit lane, the table of an in-lane byte shuffle.
#[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
const G_SPILL: core::arch::x86_64::__m512i = {
    let mut words = [0u64; 8];
    let mut n = 0;
    while n < 16 {
        let s = n ^ (n >> 1) ^ (n >> 3);
        let g = (s ^ (s << 1) ^ (s << 3) ^ (s << 4)) as u8 as u64;
        words[n / 8] |= g << (8 * (n % 8));
        n += 1;
    }
    let mut lane = 1;
    while lane < 4 {
        (words[2 * lane], words[2 * lane + 1]) = (words[0], words[1]);
        lane += 1;
    }
    primitives::simd::from_words(words)
};

/// `u ^ v * tw` in each of the eight 64-bit lanes of an AVX-512 register, for the two kernels below.
///
/// # Algorithm
///
/// ```text
///     products:   even lanes 0,2,4,6 -> CLMUL 0x00      odd lanes 1,3,5,7 -> CLMUL 0x11
///                 each 128-bit lane then holds one whole 128-bit product
///
///     unpack:     lo = [ lo(p_0), lo(p_1), ..., lo(p_7) ]     hi = [ hi(p_0), ..., hi(p_7) ]
///
///     reduce:     p = lo + hi * x^64,  x^64 = x^4 + x^3 + x + 1
///                 p mod f = lo ^ g(hi) ^ g(spill),  g(y) = y ^ y<<1 ^ y<<3 ^ y<<4 = z ^ z<<3, z = y ^ y<<1
///                 spill = hi>>63 ^ hi>>61 ^ hi>>60, the bits g pushes past x^63
///                 g(spill) is a byte fixed by hi's top nibble: one `vpshufb` lookup
/// ```
///
/// - The unpacks and the lookup stay inside 128-bit lanes, so no shuffle crosses lanes.
/// - The reduction takes three shifts and three-way XORs, `vpternlogq 0x96`, instead of two more carry-less multiplies.
/// - Zen 5 has one carry-less multiplier per core, so those two would dominate the kernel.
#[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
#[inline]
#[target_feature(enable = "vpclmulqdq", enable = "avx512f", enable = "avx512bw")]
fn mul_xor_avx512(
    u: core::arch::x86_64::__m512i,
    v: core::arch::x86_64::__m512i,
    tw: core::arch::x86_64::__m512i,
) -> core::arch::x86_64::__m512i {
    use core::arch::x86_64::*;

    // Products v * t: even lanes, then odd lanes, one 128-bit product per 128-bit lane.
    let even = _mm512_clmulepi64_epi128::<0x00>(v, tw);
    let odd = _mm512_clmulepi64_epi128::<0x11>(v, tw);
    // Back to lane order: qword i of lo / hi is the low / high half of lane i's product.
    let lo = _mm512_unpacklo_epi64(even, odd);
    let hi = _mm512_unpackhi_epi64(even, odd);

    // Reduce modulo x^64 + x^4 + x^3 + x + 1: p mod f = lo ^ g(hi) ^ g(spill), g linear.
    const XOR3: i32 = 0x96;
    // g(spill) depends on hi's top nibble alone, and fits a byte: one in-lane table lookup.
    let nibble = _mm512_srli_epi64::<60>(hi);
    let g_spill = _mm512_shuffle_epi8(G_SPILL, nibble);
    // g(hi) = y ^ y<<3, with y = hi ^ hi<<1.
    let y = _mm512_xor_si512(hi, _mm512_slli_epi64::<1>(hi));
    _mm512_ternarylogic_epi64::<XOR3>(
        _mm512_ternarylogic_epi64::<XOR3>(u, lo, g_spill),
        y,
        _mm512_slli_epi64::<3>(y),
    )
}

/// The butterflies of a pair of equal-length rows with a shared twiddle, eight lanes per AVX-512 register.
///
/// The last register is masked to the row's end.
#[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
#[inline]
#[target_feature(enable = "vpclmulqdq", enable = "avx512f", enable = "avx512bw")]
fn butterfly_lanes_avx512(top: &mut [F64], bot: &mut [F64], twiddle: u64) {
    use core::arch::x86_64::*;
    use primitives::simd::{load_prefix, store_prefix, xor};

    let tw = _mm512_set1_epi64(twiddle as i64);
    let step = |top: &mut [F64], bot: &mut [F64], i: usize, n: usize| {
        let u = load_prefix(top, i, n);
        let v = load_prefix(bot, i, n);
        // Butterfly top: u' = u + v * t. Bottom: v' = v + u'.
        let new_u = mul_xor_avx512(u, v, tw);
        store_prefix(top, i, n, new_u);
        store_prefix(bot, i, n, xor(v, new_u));
    };
    let vectors = top.len() / 8;
    for i in 0..vectors {
        step(top, bot, i, 8);
    }
    let tail = top.len() - 8 * vectors;
    if tail > 0 {
        step(top, bot, vectors, tail);
    }
}

/// Eight-lane vectors of every row that a register kernel keeps in flight.
///
/// One vector's butterflies form a dependent chain, so a second vector covers its latency.
#[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
const LANE_VECTORS: usize = 2;

/// A broadcast twiddle, and whether it is zero: the butterfly is then `v' = v + u`, with no product.
#[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
type Twiddle = (core::arch::x86_64::__m512i, bool);

/// The vectors of `N` rows held in registers: `v[row][k]` is vector `k` of the row.
#[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
type RowVectors<const N: usize> = [[core::arch::x86_64::__m512i; LANE_VECTORS]; N];

/// Butterfly rows `top` and `bot` of every vector in flight.
#[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
#[inline]
#[target_feature(enable = "vpclmulqdq", enable = "avx512f", enable = "avx512bw")]
fn butterfly_regs<const N: usize>(v: &mut RowVectors<N>, top: usize, bot: usize, (tw, zero): Twiddle) {
    for k in 0..LANE_VECTORS {
        let (u, w) = (v[top][k], v[bot][k]);
        let new_u = if zero { u } else { mul_xor_avx512(u, w, tw) };
        v[top][k] = new_u;
        v[bot][k] = primitives::simd::xor(w, new_u);
    }
}

/// Runs `kernel` on `N` equal-length rows held in registers, [`LANE_VECTORS`] eight-lane vectors of every row at a time.
///
/// - The vectors load from `src`, or from `rows` when there is none, and store to `rows`.
/// - A vector is loaded and stored once, however many butterflies `kernel` applies to it.
/// - The kernel orders its butterflies breadth-first, so the vectors in flight hide each other's latency.
/// - The last chunk may be short: its vectors past the row's end load as zeros and are not stored.
#[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
#[inline]
#[target_feature(enable = "vpclmulqdq", enable = "avx512f", enable = "avx512bw")]
fn fused_lanes_avx512<const N: usize, const T: usize>(
    src: Option<[&[F64]; N]>,
    rows: &mut [&mut [F64]; N],
    twiddles: &[F64; T],
    kernel: impl Fn(&mut RowVectors<N>, &[Twiddle; T]),
) {
    use core::arch::x86_64::*;
    use primitives::simd::{load, load_prefix, store, store_prefix};

    const CHUNK: usize = 8 * LANE_VECTORS;
    let len = rows[0].len();
    debug_assert!(rows.iter().all(|row| row.len() == len));
    debug_assert!(src.iter().flatten().all(|row| row.len() == len));
    let tw = twiddles.map(|t| (_mm512_set1_epi64(t.0 as i64), t == F64::ZERO));
    let mut lane = 0;
    while lane + CHUNK <= len {
        // The chunk of every row, as arrays: the loads and stores below need no bounds check of their own.
        let chunks: [&mut [F64; CHUNK]; N] =
            rows.each_mut().map(|row| (&mut row[lane..lane + CHUNK]).try_into().unwrap());
        let mut v: RowVectors<N> = match &src {
            Some(src) => std::array::from_fn(|i| {
                let from: &[F64; CHUNK] = src[i][lane..lane + CHUNK].try_into().unwrap();
                std::array::from_fn(|k| load(from, k))
            }),
            None => std::array::from_fn(|i| std::array::from_fn(|k| load(&*chunks[i], k))),
        };
        kernel(&mut v, &tw);
        for i in 0..N {
            for k in 0..LANE_VECTORS {
                store(chunks[i], k, v[i][k]);
            }
        }
        lane += CHUNK;
    }
    if lane < len {
        let reg = lane / 8;
        // The lanes of vector `k` of the short chunk: all eight, the row's last few, or none.
        let n = |k: usize| len.saturating_sub(lane + 8 * k).min(8);
        let mut v: RowVectors<N> = std::array::from_fn(|i| {
            std::array::from_fn(|k| match &src {
                Some(src) => load_prefix(src[i], reg + k, n(k)),
                None => load_prefix(&*rows[i], reg + k, n(k)),
            })
        });
        kernel(&mut v, &tw);
        for i in 0..N {
            for k in 0..LANE_VECTORS {
                store_prefix(rows[i], reg + k, n(k), v[i][k]);
            }
        }
    }
}

/// [`butterfly_lanes_avx512`] transposed: `s = u + v; u' = s; v' = s * t + v`.
#[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
#[inline]
#[target_feature(enable = "vpclmulqdq", enable = "avx512f", enable = "avx512bw")]
fn transpose_butterfly_lanes_avx512(top: &mut [F64], bot: &mut [F64], twiddle: u64) {
    use core::arch::x86_64::*;
    use primitives::simd::{load_prefix, store_prefix, xor};

    let tw = _mm512_set1_epi64(twiddle as i64);
    let step = |top: &mut [F64], bot: &mut [F64], i: usize, n: usize| {
        let u = load_prefix(top, i, n);
        let v = load_prefix(bot, i, n);
        let s = xor(u, v);
        store_prefix(top, i, n, s);
        store_prefix(bot, i, n, mul_xor_avx512(v, s, tw));
    };
    let vectors = top.len() / 8;
    for i in 0..vectors {
        step(top, bot, i, 8);
    }
    let tail = top.len() - 8 * vectors;
    if tail > 0 {
        step(top, bot, vectors, tail);
    }
}

/// Two F64 butterflies with a shared twiddle, NEON-resident end to end.
/// The two products issue as PMULL/PMULL2 on the loaded row (no lane
/// extraction) and reduce through the all-PMULL lane-pair fold
/// ([`primitives::field::gf2_64::aarch64::reduce_pair_pmull4`]).
///
/// `top` and `bot` are two words each.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
#[target_feature(enable = "neon,aes")]
fn butterfly_lane_pair_neon(top: &mut [F64], bot: &mut [F64], twiddle: u64) {
    use core::arch::aarch64::*;
    use primitives::field::gf2_64::aarch64::{pmull, pmull_hi, reduce_pair_pmull4};
    use primitives::{load_at, store_at};

    let (u, v): (uint64x2_t, uint64x2_t) = (load_at(top, 0), load_at(bot, 0));
    // Products v_lane * twiddle: PMULL on the low lane, PMULL2 on the high one.
    let prod = reduce_pair_pmull4(pmull(vgetq_lane_u64::<0>(v), twiddle), pmull_hi(v, vdupq_n_u64(twiddle)));
    let new_u = veorq_u64(u, prod);
    store_at(top, 0, new_u);
    store_at(bot, 0, veorq_u64(v, new_u));
}

#[cfg(test)]
mod tests {
    use super::*;
    use primitives::test_rng::Rng;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A row sink that copies each handed-over block into `dst` at its row, without locking.
    /// Every row is handed over exactly once, so the blocks written from pool tasks are disjoint.
    fn sink_into(dst: &mut [F64], row_words: usize) -> impl Fn(usize, &[F64]) + Sync + '_ {
        let len = dst.len();
        let base = parallel::SendPtr(dst.as_mut_ptr());
        move |row, rows| {
            assert!((row * row_words).checked_add(rows.len()).is_some_and(|end| end <= len));
            // SAFETY: in bounds, and the sink contract hands each row over once, so no
            // other task writes this range. `dst` is borrowed for as long as the sink lives.
            unsafe { base.slice(row * row_words, rows.len()) }.copy_from_slice(rows);
        }
    }

    /// Scalar reference: one butterfly at a time over `num_ntts` interleaved lanes.
    fn forward_scalar_from_layer(ntt: &AdditiveNttF64, data: &mut [F64], num_ntts: usize, start_layer: usize) {
        let n_total = data.len();
        let log_d = log2_strict_usize(n_total / num_ntts);

        for layer in start_layer..log_d {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (log_d - layer);
            let block_size_half = block_size >> 1;
            let block_elems = block_size * num_ntts;
            for block in 0..num_blocks {
                let twiddle = ntt.twiddle(layer, block);
                let block_start = block * block_elems;
                for row in 0..block_size_half {
                    let off_top = block_start + row * num_ntts;
                    let off_bot = off_top + block_size_half * num_ntts;
                    for lane in 0..num_ntts {
                        let v = data[off_bot + lane];
                        let new_u = data[off_top + lane] + v * twiddle;
                        data[off_top + lane] = new_u;
                        data[off_bot + lane] = v + new_u;
                    }
                }
            }
        }
    }

    /// Check forward∘inverse = id and scalar == parallel.
    #[test]
    fn inverse_roundtrip_and_variants_agree() {
        let ntt = AdditiveNttF64::standard(12);
        let mut rng = Rng::new(1);
        for log_d in [1usize, 3, 6, 10] {
            let n = 1usize << log_d;
            let orig: Vec<F64> = (0..n).map(|_| F64(rng.next_u64())).collect();

            let mut a = orig.clone();
            forward_scalar_from_layer(&ntt, &mut a, 1, 0);
            let mut c = orig.clone();
            ntt.transform(&mut c, 1, 0, None, None);
            assert_eq!(a, c, "parallel == scalar at log_d={log_d}");

            ntt.inverse_transform(&mut a);
            assert_eq!(a, orig, "inverse roundtrip at log_d={log_d}");
        }
    }

    #[test]
    fn interleaved_parallel_matches_scalar() {
        // Invariant: the parallel transform equals the scalar reference, word for word.
        //
        // Each shape forces one plan of the driver, under the budgets used off Apple silicon:
        //
        //     (log_d, lanes, start)   plan
        //     (7, 3, 0)               deep pass only, a single sub-block
        //     (12, 8, 0 or 1)         one gathered pass, then deep sub-blocks
        //     (14, 64, 0 or 1)        one gathered pass, then deep sub-blocks
        //     (10, 2048, 0)           wide rows shrink the cache budgets: one gathered pass
        //     (12, 2048, 1)           two gathered passes, with streaming stores
        //     (18, 48, 1)             one gathered pass whose groups drain a line at a time
        //     (20, 9, 1)              streaming stores, rows that are not whole cache lines
        //
        // A non-zero start is the commit path, which enters at the rate layer.
        let mut rng = Rng::new(0xC0FFEE);
        for (log_d, lanes, start_layer) in [
            (7usize, 3usize, 0usize),
            (12, 8, 0),
            (12, 8, 1),
            (14, 64, 0),
            (14, 64, 1),
            (10, 2048, 0),
            (12, 2048, 1),
            (18, 48, 1),
            (20, 9, 1),
        ] {
            let ntt = AdditiveNttF64::standard(log_d);
            let original: Vec<F64> = (0..lanes << log_d).map(|_| F64(rng.next_u64())).collect();

            // Reference: one butterfly at a time.
            let mut want = original.clone();
            forward_scalar_from_layer(&ntt, &mut want, lanes, start_layer);
            // Under test: the pass-planning driver.
            let mut got = original;
            ntt.transform(&mut got, lanes, start_layer, None, None);

            assert_eq!(got, want, "log_d={log_d}, lanes={lanes}, start_layer={start_layer}");
        }
    }

    #[test]
    fn fused_encode_matches_replicate_then_transform() {
        // Invariant: encoding in place equals replicating the message, then transforming.
        //
        // The in-place encode never materializes the replicas up front.
        // Its message is the buffer's own first replica, which the plan must overwrite last.
        // The plans are those of the budgets used off Apple silicon.
        //
        //     (log_d, lanes, rate)   plan
        //     (9, 8, 1)              replicate first, then deep pass only
        //     (12, 64, 2)            one gathered pass reading the first replica
        //     (14, 8, 1)             one gathered pass reading the first replica
        //     (14, 64, 2)            one gathered pass reading the first replica
        //     (12, 2048, 1)          two gathered passes, with streaming stores
        //     (18, 48, 2)            one gathered pass reading the first replica, groups draining a line at a time
        //     (20, 9, 2)             the same with rows that are not whole cache lines
        //     (4, 8, 4)              rate = log_d: no layer left, rows handed over at the end
        //
        // The handed-over rows must tile the codeword once, each block already final.
        let mut rng = Rng::new(0xE0C0DE);
        for (log_d, lanes, log_inv_rate) in [
            (9usize, 8usize, 1usize),
            (12, 64, 2),
            (14, 8, 1),
            (14, 64, 2),
            (12, 2048, 1),
            (18, 48, 2),
            (20, 9, 2),
            (4, 8, 4),
        ] {
            let ntt = AdditiveNttF64::standard(log_d);
            let msg_len = (lanes << log_d) >> log_inv_rate;
            let msg: Vec<F64> = (0..msg_len).map(|_| F64(rng.next_u64())).collect();

            // Reference: 2^rate explicit copies, then the scalar transform from the rate layer.
            let mut want = vec![F64::ZERO; msg_len << log_inv_rate];
            replicate_rows(&mut want, &msg);
            forward_scalar_from_layer(&ntt, &mut want, lanes, log_inv_rate);

            // Under test: only the first replica holds the message, the rest is zero.
            let mut got = vec![F64::ZERO; msg_len << log_inv_rate];
            got[..msg_len].copy_from_slice(&msg);
            let mut handed = vec![F64::ZERO; got.len()];
            let hits: Vec<AtomicUsize> = (0..1usize << log_d).map(|_| AtomicUsize::new(0)).collect();
            let write = sink_into(&mut handed, lanes);
            ntt.encode_interleaved_in_place_with(&mut got, lanes, log_inv_rate, &|row, rows| {
                for hit in &hits[row..row + rows.len() / lanes] {
                    hit.fetch_add(1, Ordering::Relaxed);
                }
                write(row, rows)
            });
            drop(write);
            assert_eq!(got, want, "log_d={log_d}, lanes={lanes}, rate={log_inv_rate}");
            // Every row handed over exactly once, so the blocks tile the rows, and each is final.
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "blocks tile the rows, log_d={log_d}");
            assert_eq!(handed, want, "a handed-over block is final, log_d={log_d}");
        }
    }

    /// Every lane of the codeword must be exactly the single-lane RS codeword of
    /// that lane's contiguous message block, which is what makes a commitment over
    /// `n_lanes` lanes equal to the `2^log_batch_size`-lane one with a zero tail.
    /// The shapes cover the transposing fused first pass, its fallback, and lane
    /// counts that are not powers of two (the padding-free commit's whole point).
    #[test]
    fn lane_major_msg_encode_matches_per_lane_reference() {
        let mut rng = Rng::new(0x1A2E);
        for (log_rows, log_inv_rate, n_lanes) in [
            (2usize, 1usize, 3usize),
            (3, 1, 1),
            (5, 2, 7),
            (9, 1, 5),
            (12, 2, 37),
            (14, 1, 64),
        ] {
            let log_d = log_rows + log_inv_rate;
            let ntt = AdditiveNttF64::standard(log_d);
            let rows = 1usize << log_rows;
            let msg: Vec<F64> = (0..rows * n_lanes).map(|_| F64(rng.next_u64())).collect();

            let row_words = n_lanes.next_multiple_of(8);
            let mut got = vec![F64::ZERO; (row_words << log_rows) << log_inv_rate];
            transpose_lane_major(&mut got[..row_words << log_rows], &msg, n_lanes, row_words, log_rows);
            ntt.encode_interleaved_in_place(&mut got, row_words, log_inv_rate);

            let block_len = 1usize << log_d;
            for lane in 0..row_words {
                // The leading lanes are zero padding, the rest encode message block `row_words - 1 - lane`.
                let mut want = vec![F64::ZERO; block_len];
                if lane >= row_words - n_lanes {
                    let block = row_words - 1 - lane;
                    replicate_rows(&mut want, &msg[block * rows..(block + 1) * rows]);
                    ntt.transform(&mut want, 1, log_inv_rate, None, None);
                }
                for pos in 0..block_len {
                    assert_eq!(
                        got[pos * row_words + lane],
                        want[pos],
                        "lane {lane} pos {pos} at log_rows={log_rows}, rate={log_inv_rate}, n_lanes={n_lanes}"
                    );
                }
            }
        }
    }

    /// The lane-major encode reads the message in its first gathered pass, so it must equal the
    /// transpose into block 0 followed by the in-place encode: rows handed over and rows opened.
    /// The shapes cover a fused first pass (with and without padding lanes, at both rates),
    /// a deep-only plan, and a domain too small for any gathered pass.
    #[test]
    fn lane_major_encode_matches_transpose_then_in_place() {
        let mut rng = Rng::new(0x7A5E);
        for (log_rows, log_inv_rate, n_lanes) in [
            (13usize, 2usize, 47usize),
            (14, 1, 40),
            (13, 1, 17),
            (12, 2, 64),
            (5, 2, 7),
        ] {
            let log_d = log_rows + log_inv_rate;
            let ntt = AdditiveNttF64::standard(log_d);
            let mut msg: Vec<F64> = (0..n_lanes << log_rows).map(|_| F64(rng.next_u64())).collect();
            // Runs of equal blocks, which the stage fill copies instead of reading.
            let block = 1usize << EQUAL_BLOCK_LOG;
            for lane in msg.chunks_mut(1 << log_rows).filter(|lane| lane.len() >= 8 * block) {
                for b in [1, 2, 3, 5, 6] {
                    lane.copy_within((b - 1) * block..b * block, b * block);
                }
            }
            let repeats = crate::stack_open::equal_blocks(&msg);
            let row_words = n_lanes.next_multiple_of(8);
            let mut want = vec![F64::ZERO; (row_words << log_rows) << log_inv_rate];
            transpose_lane_major(&mut want[..row_words << log_rows], &msg, n_lanes, row_words, log_rows);
            ntt.encode_interleaved_in_place(&mut want, row_words, log_inv_rate);

            for keep in [Keep::Whole, Keep::BeforeDeep, Keep::Nothing] {
                let mut handed = vec![F64::ZERO; want.len()];
                let on_rows = sink_into(&mut handed, row_words);
                let Encoded { words, deep } = ntt.encode_lane_major(&msg, &repeats, n_lanes, row_words, log_inv_rate, keep, &on_rows);
                drop(on_rows);
                let context = format!("log_rows={log_rows}, rate={log_inv_rate}, n_lanes={n_lanes}, keep_nothing={}", keep == Keep::Nothing);
                assert_eq!(handed, want, "handed-over rows, {context}");
                let mut positions: Vec<usize> = (0..40).map(|_| rng.next_u64() as usize % (1 << log_d)).collect();
                positions.sort_unstable();
                let rows = match deep {
                    Some(deep) if words.is_empty() => ntt.rebuild_rows(&msg, &repeats, n_lanes, &deep, &positions),
                    Some(deep) => ntt.finish_rows(&words, &deep, &positions),
                    None => {
                        assert_eq!(words, want, "codeword, {context}");
                        positions.iter().flat_map(|&q| words[q * row_words..][..row_words].iter().copied()).collect()
                    }
                };
                for (&q, row) in positions.iter().zip(rows.chunks_exact(row_words)) {
                    assert_eq!(row[..], want[q * row_words..][..row_words], "row {q}, {context}");
                }
            }
        }
    }

    /// Without the deep writeback, the handed-over rows and the rows finished on demand
    /// must equal the full encode's, wherever the plan splits off a deep pass.
    #[test]
    fn deferred_deep_rows_match_the_full_encode() {
        let mut rng = Rng::new(0xDEE9);
        for (log_d, lanes, log_inv_rate, split) in [(14usize, 51usize, 1usize, true), (14, 64, 2, true), (9, 8, 1, true), (4, 8, 4, false)] {
            let ntt = AdditiveNttF64::standard(log_d);
            let msg_len = (lanes << log_d) >> log_inv_rate;
            let msg: Vec<F64> = (0..msg_len).map(|_| F64(rng.next_u64())).collect();
            let mut want = vec![F64::ZERO; msg_len << log_inv_rate];
            want[..msg_len].copy_from_slice(&msg);
            ntt.encode_interleaved_in_place(&mut want, lanes, log_inv_rate);

            let mut got = vec![F64::ZERO; msg_len << log_inv_rate];
            got[..msg_len].copy_from_slice(&msg);
            let mut handed = vec![F64::ZERO; got.len()];
            let deep = ntt.encode_interleaved_in_place_without_deep(&mut got, lanes, log_inv_rate, &sink_into(&mut handed, lanes));
            assert_eq!(handed, want, "handed-over rows, log_d={log_d}, lanes={lanes}");
            assert_eq!(deep.is_some(), split, "plan, log_d={log_d}, lanes={lanes}");
            let mut positions: Vec<usize> = (0..40).map(|_| rng.next_u64() as usize % (1 << log_d)).collect();
            positions.sort_unstable();
            let rows = match deep {
                Some(deep) => ntt.finish_rows(&got, &deep, &positions),
                None => positions.iter().flat_map(|&q| got[q * lanes..][..lanes].iter().copied()).collect(),
            };
            for (&q, row) in positions.iter().zip(rows.chunks_exact(lanes)) {
                assert_eq!(row[..], want[q * lanes..][..lanes], "row {q}, log_d={log_d}, lanes={lanes}");
            }
        }
    }

    /// The cosets' handed-over rows, and the rows recomputed from the message, equal the full encode's.
    #[test]
    fn coset_rows_match_the_full_encode() {
        let mut rng = Rng::new(0xC05E7);
        // Small cosets group in one scratch; the last shape is a single coset.
        for (log_rows, lanes, log_inv_rate) in [(5usize, 48usize, 6usize), (9, 24, 4), (7, 3, 3), (6, 8, 1)] {
            let log_d = log_rows + log_inv_rate;
            let ntt = AdditiveNttF64::standard(log_d + 1);
            let msg_len = lanes << log_rows;
            let msg: Vec<F64> = (0..msg_len).map(|_| F64(rng.next_u64())).collect();
            let mut want = vec![F64::ZERO; msg_len << log_inv_rate];
            want[..msg_len].copy_from_slice(&msg);
            ntt.encode_interleaved_in_place(&mut want, lanes, log_inv_rate);

            let mut handed = vec![F64::ZERO; want.len()];
            ntt.encode_cosets(&msg, lanes, log_inv_rate, &sink_into(&mut handed, lanes));
            assert_eq!(handed, want, "handed-over rows, log_rows={log_rows}, lanes={lanes}");

            // Repeats included.
            let mut positions: Vec<usize> = (0..60).map(|_| rng.next_u64() as usize % (1 << log_d)).collect();
            positions.extend_from_within(..5);
            positions.sort_unstable();
            let rows = ntt.coset_rows(&msg, lanes, log_inv_rate, &positions);
            for (&q, row) in positions.iter().zip(rows.chunks_exact(lanes)) {
                assert_eq!(row[..], want[q * lanes..][..lanes], "row {q}, log_rows={log_rows}, lanes={lanes}");
            }
        }
    }

    #[test]
    fn interleaved_lanes_are_independent_ntts() {
        let ntt = AdditiveNttF64::standard(10);
        let mut rng = Rng::new(2);
        let log_d = 7;
        let n = 1usize << log_d;
        for lanes in [1usize, 2, 4, 8, 17, 24, 64] {
            // SoA buffer + per-lane copies.
            let mut soa = vec![F64::ZERO; n * lanes];
            let mut per_lane: Vec<Vec<F64>> = vec![vec![F64::ZERO; n]; lanes];
            for pos in 0..n {
                for lane in 0..lanes {
                    let v = F64(rng.next_u64());
                    soa[pos * lanes + lane] = v;
                    per_lane[lane][pos] = v;
                }
            }
            ntt.transform(&mut soa, lanes, 0, None, None);
            for (lane, lane_data) in per_lane.iter_mut().enumerate() {
                forward_scalar_from_layer(&ntt, lane_data, 1, 0);
                for pos in 0..n {
                    assert_eq!(soa[pos * lanes + lane], lane_data[pos]);
                }
            }
        }
    }
}
