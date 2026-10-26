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

    /// The sorted, distinct deep sub-blocks that hold `positions`.
    fn sub_blocks(&self, positions: &[usize]) -> Vec<usize> {
        let mut subs: Vec<usize> = positions.iter().map(|&q| q >> (self.log_d - self.deep_start)).collect();
        subs.sort_unstable();
        subs.dedup();
        subs
    }

    /// The rows at `positions`, out of the finished sub-blocks `subs`, held back to back in `built`.
    fn pick(&self, subs: &[usize], built: &[F64], positions: &[usize]) -> Vec<Vec<F64>> {
        let log_sub = self.log_d - self.deep_start;
        positions
            .iter()
            .map(|&q| {
                let sub = subs.binary_search(&(q >> log_sub)).expect("every queried sub-block is finished");
                built[sub * self.sub_len() + (q & ((1 << log_sub) - 1)) * self.num_ntts..][..self.num_ntts].to_vec()
            })
            .collect()
    }
}

/// Table of the normalized subspace polynomials at the basis.
///
/// - Row `i` holds `s_i(b_j)` for every basis element `b_j` with `j >= i`.
/// - Each row is scaled so that its first entry is one.
fn generate_evals_from_subspace(basis: &[F64]) -> Vec<Vec<F64>> {
    let l = basis.len();
    let mut evals: Vec<Vec<F64>> = Vec::with_capacity(l);
    evals.push(basis.to_vec());
    for i in 1..l {
        let mut row = Vec::with_capacity(l - i);
        for k in 1..evals[i - 1].len() {
            let val = evals[i - 1][k] * (evals[i - 1][k] + evals[i - 1][0]);
            row.push(val);
        }
        evals.push(row);
    }
    for row in evals.iter_mut() {
        let inv = row[0].inv();
        for v in row.iter_mut() {
            *v *= inv;
        }
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
pub type RowSink<'a> = dyn Fn(usize, &[F64]) + Sync + 'a;

/// Additive NTT over F_{2^64} with the standard polynomial-basis subspace
/// `{1, x, x², …}`: the F_2-subspace is `{0, 1, …, 2^ℓ−1}` under the natural
/// integer encoding, exactly as in the extension-field version (whose domain already
/// lived inside this very subfield).
#[derive(Clone, Debug)]
pub struct AdditiveNttF64 {
    evals: Vec<Vec<F64>>,
    /// Per eval row, the subset sums of `row[1..]` over each byte of a block index:
    /// a twiddle of a block below `2^24` is three loads and two XORs.
    tables: Vec<Box<[[F64; 256]; 3]>>,
}

impl AdditiveNttF64 {
    fn new(basis: &[F64]) -> Self {
        let evals = generate_evals_from_subspace(basis);
        let tables = evals
            .iter()
            .map(|row| {
                let basis = &row[1..];
                let mut t = Box::new([[F64::ZERO; 256]; 3]);
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
        self.evals.len()
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
            span_get(&self.evals[row][1..], block)
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
        let (v1, v2) = (&self.evals[l - layer - 2], &self.evals[l - layer - 3]);
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
    /// # Why the replication is free
    ///
    /// - A butterfly whose bottom input is zero copies its top input to both outputs.
    /// - So the first `r` layers on the zero-padded message leave `2^r` copies of it.
    /// - The first pass therefore reads its rows from the message itself.
    /// - No pass fills the replicas only to read them back.
    ///
    /// # Why the transpose stays separate
    ///
    /// - A lane-major caller transposes its message into place first.
    /// - The transpose wants one long contiguous run per lane.
    /// - The first pass wants hundreds of scattered rows at once.
    pub fn encode_interleaved_in_place(&self, data: &mut [F64], num_ntts: usize, log_inv_rate: usize) {
        // The message is the buffer's own first replica.
        let msg = parallel::SendPtr(data.as_mut_ptr());
        self.transform(data, num_ntts, log_inv_rate, Some(msg), None);
    }

    /// Encode in place, handing `on_rows` every finished block of rows.
    ///
    /// - It is called as `on_rows(first_row, rows)`, from pool tasks.
    /// - Each row is handed over exactly once.
    /// - Blocks are aligned, and all of one power-of-two size.
    /// - A block is handed over while its rows are still in cache.
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
    /// - The buffer keeps what the gathered passes left, and [`DeepRows::rows`] finishes any row again.
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

    /// Encode a lane-major message into a codeword that starts uninitialized, handing `on_rows` every finished block.
    ///
    /// - The result equals [`transpose_lane_major`] into block 0 followed by the in-place encode.
    /// - With a gathered first pass, that pass reads the message itself and writes every block.
    /// - So no transposed copy of the message is written and read back.
    /// - With `defer_deep`, the deep pass goes to `on_rows` only, as in [`Self::encode_interleaved_in_place_without_deep`].
    #[allow(clippy::too_many_arguments)]
    pub fn encode_lane_major(
        &self,
        data: &mut [F64],
        msg: &[F64],
        n_lanes: usize,
        num_ntts: usize,
        log_inv_rate: usize,
        on_rows: &RowSink<'_>,
        defer_deep: bool,
    ) -> Option<DeepRows> {
        assert!(0 < n_lanes && n_lanes <= num_ntts, "a row holds its lanes");
        assert_eq!(data.len() % num_ntts, 0);
        assert_eq!(
            msg.len() * num_ntts,
            n_lanes * (data.len() >> log_inv_rate),
            "the message is n_lanes lane blocks"
        );
        // Read-only from here on: the pointer only feeds the first pass's reads.
        let msg = parallel::SendPtr(msg.as_ptr().cast_mut());
        self.transform_deferring_deep(
            data,
            num_ntts,
            log_inv_rate,
            Some(msg),
            Some(on_rows),
            defer_deep,
            Some(n_lanes),
        )
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

    /// Codeword rows `positions` of the encode [`Self::encode_cosets`] ran, one transform per coset they touch.
    pub fn coset_rows(&self, msg: &[F64], num_ntts: usize, log_inv_rate: usize, positions: &[usize]) -> Vec<Vec<F64>> {
        let log_d = self.coset_log_d(msg.len(), num_ntts, log_inv_rate);
        let rows = msg.len() / num_ntts;
        // Positions grouped by coset, in runs.
        let mut order: Vec<usize> = (0..positions.len()).collect();
        order.sort_unstable_by_key(|&i| positions[i]);
        let runs: Vec<&[usize]> = order.chunk_by(|&a, &b| positions[a] / rows == positions[b] / rows).collect();
        let found = parallel::map_collect(runs.len(), |r| {
            with_scratch(msg.len(), |coset| {
                self.build_coset(coset, msg, num_ntts, log_d, log_inv_rate, positions[runs[r][0]] / rows);
                runs[r].iter().map(|&i| coset[(positions[i] % rows) * num_ntts..][..num_ntts].to_vec()).collect::<Vec<_>>()
            })
        });
        let mut out = vec![Vec::new(); positions.len()];
        for (run, rows) in runs.iter().zip(found) {
            for (&i, row) in run.iter().zip(rows) {
                out[i] = row;
            }
        }
        out
    }

    /// Whether a message of `msg_words` words encodes coset by coset: one coset fits the cache budget [`COSET_WORDS`].
    pub fn encodes_by_coset(msg_words: usize) -> bool {
        msg_words <= COSET_WORDS
    }

    /// The log of the codeword's rows, for a message of `msg_words` words.
    fn coset_log_d(&self, msg_words: usize, num_ntts: usize, log_inv_rate: usize) -> usize {
        assert!(num_ntts > 0 && msg_words % num_ntts == 0);
        let log_d = log2_strict_usize(msg_words / num_ntts) + log_inv_rate;
        assert!(log_d <= self.log_domain_size());
        log_d
    }

    /// Coset `coset` of the codeword, built into `buf` (one message long) from the message.
    fn build_coset(&self, buf: &mut [F64], msg: &[F64], num_ntts: usize, log_d: usize, log_inv_rate: usize, coset: usize) {
        buf.copy_from_slice(msg);
        self.run_layers(buf, log_d, num_ntts, log_inv_rate, log_d, log_inv_rate, coset, no_hook);
    }

    /// Codeword rows `positions` of a buffer left by [`Self::encode_interleaved_in_place_without_deep`].
    pub fn finish_rows(&self, data: &[F64], deep: &DeepRows, positions: &[usize]) -> Vec<Vec<F64>> {
        let (subs, sub_len) = (deep.sub_blocks(positions), deep.sub_len());
        let mut built = vec![F64::ZERO; subs.len() * sub_len];
        parallel::chunks_mut(&mut built, sub_len, |i, sub| {
            sub.copy_from_slice(&data[subs[i] * sub_len..][..sub_len]);
            self.finish_sub_block(sub, deep, subs[i]);
        });
        deep.pick(&subs, &built, positions)
    }

    /// Codeword rows `positions` of an encode that kept no buffer, recomputed from its lane-major message.
    pub fn rebuild_rows(&self, msg: &[F64], n_lanes: usize, deep: &DeepRows, positions: &[usize]) -> Vec<Vec<F64>> {
        let subs = deep.sub_blocks(positions);
        let built = self.rebuild_sub_blocks(msg, n_lanes, deep, &subs);
        deep.pick(&subs, &built, positions)
    }

    /// The deep layers over sub-block `sub_idx` of the codeword, which `sub` holds as the gathered pass left it.
    fn finish_sub_block(&self, sub: &mut [F64], deep: &DeepRows, sub_idx: usize) {
        let DeepRows { log_d, num_ntts, deep_start, .. } = *deep;
        self.run_layers(sub, log_d, num_ntts, deep_start, log_d, deep_start, sub_idx, no_hook);
    }

    /// The finished deep sub-blocks `subs` (sorted, distinct) of an encode that kept no buffer, back to back.
    ///
    /// - The encode ran one gathered pass over its lane-major message: a group is one residue of one coset,
    ///   and its transform leaves row `i` of the group as row `r` of sub-block `coset << g | i`, `g` layers deep.
    /// - So each group transforms again here, skipping the blocks that feed no row of `subs`,
    ///   and keeps only the rows that fall in a sub-block of `subs`.
    /// - Those rows are every word of the sub-blocks before their deep layers, which then run in place.
    ///
    /// # Panics
    ///
    /// Panics unless the encode had the plan this reads back: one lane-major gathered pass.
    #[timed]
    pub fn rebuild_sub_blocks(&self, msg: &[F64], n_lanes: usize, deep: &DeepRows, subs: &[usize]) -> Vec<F64> {
        let DeepRows { log_d, num_ntts, start, deep_start } = *deep;
        let g = deep_start - start;
        let (rows, log_step) = (1usize << g, log_d - deep_start);
        let (sub_len, msg_rows) = (deep.sub_len(), 1usize << (log_d - start));
        assert!(0 < g && g <= (L2_WORDS / num_ntts).max(1).ilog2() as usize, "one gathered pass");
        assert!(num_ntts % 8 == 0 && log_step >= LOG_BATCH, "the gathered pass is lane-major");
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
        // SAFETY: a group writes row `r` of every queried sub-block, and the groups cover every residue `r`.
        let mut built = unsafe { primitives::uninit_vec::<F64>(subs.len() * sub_len) };
        let (out, msg) = (parallel::SendPtr(built.as_mut_ptr()), parallel::SendPtr(msg.as_ptr().cast_mut()));
        parallel::for_each_chunk((1usize << log_step) / BATCH, |lo, hi| {
            let stage_len = BATCH * rows * num_ntts;
            with_scratch(stage_len + rows * num_ntts, |all| {
                let (stage, scratch) = all.split_at_mut(stage_len);
                for t in lo..hi {
                    fill_stage(stage, unsafe { msg.add(0) }.cast_const(), t * BATCH, log_step, msg_rows, n_lanes, num_ntts);
                    for (s, staged) in stage.chunks_exact(rows * num_ntts).enumerate() {
                        for (coset, run, live) in &cosets {
                            scratch.copy_from_slice(staged);
                            let live_rows = Some(&live[..]);
                            self.run_layers_to(scratch, deep_start, num_ntts, start, deep_start, start, *coset, live_rows, no_hook);
                            for (pos, &i) in run.clone().zip(live) {
                                // SAFETY: this task alone owns row `t * BATCH + s` of every sub-block.
                                let dst = unsafe { out.slice(pos * sub_len + (t * BATCH + s) * num_ntts, num_ntts) };
                                dst.copy_from_slice(&scratch[i * num_ntts..][..num_ntts]);
                            }
                        }
                    }
                }
            });
        });
        parallel::chunks_mut(&mut built, sub_len, |i, sub| self.finish_sub_block(sub, deep, subs[i]));
        built
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
    #[allow(clippy::too_many_arguments)]
    fn transform_deferring_deep(
        &self,
        data: &mut [F64],
        num_ntts: usize,
        start: usize,
        msg: Option<parallel::SendPtr<F64>>,
        on_rows: Option<&RowSink<'_>>,
        defer_deep: bool,
        mut lane_major: Option<usize>,
    ) -> Option<DeepRows> {
        // The buffer is 2^log_d rows of `num_ntts` words.
        assert!(num_ntts > 0);
        assert_eq!(data.len() % num_ntts, 0);
        let log_d = log2_strict_usize(data.len() / num_ntts);
        assert!(log_d <= self.log_domain_size());
        assert!(start <= log_d);

        /// Below this domain the transform is too small to split across the pool.
        const PARALLEL_FLOOR_LOG_D: usize = 12;
        /// Fewest rows a deep sub-block takes when splitting for parallelism.
        const MIN_SUB_LOG: usize = 8;

        // How many rows, as a log, each cache budget holds.
        //
        //     51 lanes:  2^17 / 51 -> 2^11 rows in L2,  2^18 / 51 -> 2^12 rows in L3
        let fit = |words: usize| (words / num_ntts).max(1).ilog2() as usize;
        let (fit2, fit3) = (fit(L2_WORDS), fit(L3_WORDS));

        // From 2^12 rows on, cut at least 2^LOG_SUBS_PER_WORKER deep sub-blocks per worker.
        let par_log = if log_d >= PARALLEL_FLOOR_LOG_D {
            (log2_strict_usize(parallel::num_threads().next_power_of_two()) + LOG_SUBS_PER_WORKER)
                .min(log_d - MIN_SUB_LOG)
        } else {
            0
        };

        // Where the deep pass starts.
        //
        //     every layer fits one L3 sub-block  ->  deep pass only, one sweep
        //     otherwise                          ->  the deep pass keeps one L2 sub-block of layers
        let deep = if log_d - start <= fit3 { fit3 } else { fit2 };
        let deep_start = log_d.saturating_sub(deep).max(par_log).max(start);

        // A buffer this large is evicted before the next pass reads it back.
        let stream = data.len() >= STREAM_MIN_WORDS;

        // Phase 1: gathered passes, each at most one L2 group of layers.
        //
        // Only the first one reads the message.
        let mut msg = msg;
        let from_lane_major = lane_major.is_some();
        // A lane-major message is read by the first gathered pass, a batch of residues at a time.
        // Without such a pass, or with too few residues to batch, it is transposed into block 0 first.
        if let (Some(n_lanes), Some(m)) = (lane_major, msg) {
            let g = (deep_start - start).min(fit2);
            if start == deep_start || num_ntts % 8 != 0 || log_d - start - g < LOG_BATCH {
                let log_rows = log_d - start;
                // SAFETY: the message is `n_lanes` lane blocks of `2^log_rows` words, disjoint from the buffer.
                let words = unsafe { std::slice::from_raw_parts(m.0.cast_const(), n_lanes << log_rows) };
                transpose_lane_major(&mut data[..num_ntts << log_rows], words, n_lanes, num_ntts, log_rows);
                msg = Some(parallel::SendPtr(data.as_mut_ptr()));
                lane_major = None;
            }
        }
        // An opening rebuilds the deep sub-blocks of a lane-major encode from its message: one gathered pass over it.
        let defer_deep = defer_deep && (!from_lane_major || (lane_major.is_some() && deep_start - start <= fit2));
        let mut layer = start;
        while layer < deep_start {
            let g = (deep_start - layer).min(fit2);
            self.gathered_pass(data, log_d, num_ntts, layer, g, msg.take(), lane_major.take(), stream);
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
                                for part in scratch.chunks_mut(block_len) {
                                    // SAFETY:
                                    // - The external message is valid for one whole replica of words.
                                    // - It is disjoint from the codeword.
                                    // - Each part ends inside its replica, so the read stays in bounds.
                                    let words = unsafe { std::slice::from_raw_parts(m.add(off), part.len()) };
                                    part.copy_from_slice(words);
                                }
                                match &stream {
                                    Some(s) if deferred => {
                                        // The previous sub-block drains while this one transforms.
                                        let hook = || drain.step(s, drain_per_hook);
                                        self.run_layers(
                                            scratch,
                                            log_d,
                                            num_ntts,
                                            deep_start,
                                            log_d,
                                            outer,
                                            sub_idx,
                                            hook,
                                        );
                                        drain.step(s, usize::MAX);
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
                                        self.run_layers(
                                            scratch,
                                            log_d,
                                            num_ntts,
                                            deep_start,
                                            log_d,
                                            outer,
                                            sub_idx,
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
                                drain.step(s, usize::MAX);
                            }
                        });
                    });
                }
                None => parallel::chunks_mut(data, sub_len, |sub_idx, sub| {
                    let hook = prefetch_next(sub_idx);
                    if defer_deep && deep_rows.is_some() {
                        // The rows are hashed from scratch and the buffer keeps its pre-deep state.
                        with_scratch(sub_len, |scratch| {
                            scratch.copy_from_slice(sub);
                            self.run_layers(scratch, log_d, num_ntts, deep_start, log_d, outer, sub_idx, hook);
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

    /// Run layers `layer..layer + g` in one sweep of the buffer, one row group at a time.
    ///
    /// # Row groups
    ///
    /// - A group takes rows `r + i * step`, for `i < 2^g`, of one layer-`layer` block.
    /// - Here `step = 2^(d - layer - g)` and the residue `r < step`.
    /// - Those are exactly the rows the next `g` layers pair with each other.
    ///
    /// ```text
    ///     g = 2, a 16-row block, step = 4, residue r = 1:
    ///
    ///     rows:     0  1  2  3 |  4  5  6  7 |  8  9 10 11 | 12 13 14 15
    ///                  ^              ^              ^              ^
    ///     scratch:  [ row 1, row 5, row 9, row 13 ]   ->  a 4-row transform of its own
    /// ```
    ///
    /// # Why the twiddles still match
    ///
    /// - At layer `L`, a row's block index is `row >> (d - L)`.
    /// - The residue is below `step`, so that shift drops it.
    /// - The scratch transform therefore takes the same twiddles as the full one.
    ///
    /// # With a message
    ///
    /// - Every block's rows come from the message.
    /// - One task takes one residue across all blocks, block 0 last.
    ///
    /// # Streaming the scatter
    ///
    /// - A burst of streaming stores after each group saturates the write-combining buffers and overlaps nothing.
    /// - So a streaming task keeps two scratches and drains the previous group during this group's transform.
    /// - The sweeps call a hook once per row group, and each call moves an equal share of the previous group's lines.
    /// - The shares round up, so the group is out by the last call.
    /// - The task's last group drains after its loop, before the fence.
    ///
    /// # Why a late scatter is safe
    ///
    /// - Group k's rows reach the codeword after group k+1 gathered, so group k+1 must not read them.
    /// - Without a message, a group reads only its own rows, and groups are disjoint.
    /// - With a message, group k+1 reads the message rows of one residue `t`, and group k writes one block's rows.
    /// - If group k is block `b > 0` of residue `t`, it writes block `b`, which is not the message.
    /// - If group k is block 0 of residue `t`, group k+1 is residue `t + 1`, whose rows it does not write.
    /// - Group k+2 gathers into the scratch group k drained, and that drain ended with group k+1's transform.
    ///
    /// Without whole cache lines (rows or codeword start off 64 bytes), a group scatters at once.
    ///
    /// # With a lane-major message
    ///
    /// - Word `w` of message row `j` is word `j` of lane block `num_ntts - 1 - w`, and zero below the padding.
    /// - A lane block's cache line holds [`BATCH`] consecutive rows, so a task takes that many residues.
    /// - It first transposes their rows into a stage of `BATCH` contiguous row groups.
    /// - The groups then gather from the stage as they would from a row-major message.
    /// - Every block, block 0 too, is written from the stage, so the message is never overwritten.
    #[allow(clippy::too_many_arguments)]
    #[timed]
    fn gathered_pass(
        &self,
        data: &mut [F64],
        log_d: usize,
        num_ntts: usize,
        layer: usize,
        g: usize,
        msg: Option<parallel::SendPtr<F64>>,
        lane_major: Option<usize>,
        stream: bool,
    ) {
        // A group is 2^g rows, `step` rows apart.
        let log_step = log_d - layer - g;
        let (rows, step) = (1usize << g, 1usize << log_step);
        // With a message, a task is one residue across every block, or a batch of residues when it is lane-major.
        // Without one, a task is one (block, residue) pair.
        let batch = if lane_major.is_some() { BATCH } else { 1 };
        let n_tasks = if msg.is_some() { step / batch } else { step << layer };
        // The message's rows, and the stage that holds a batch of residues' row groups.
        let msg_rows = 1usize << (log_d - layer);
        let stage_len = if lane_major.is_some() {
            BATCH * rows * num_ntts
        } else {
            0
        };
        let base = parallel::SendPtr(data.as_mut_ptr());
        // Whole cache lines in every row, so a finished group can drain a line at a time.
        let lines_ok = num_ntts.is_multiple_of(8) && base.0.align_offset(64) == 0;
        let hooks = hook_calls(layer + g, layer, layer + g, layer);
        // A small group's rows arrive without help, and its bookkeeping would cost more than it saves.
        let fetch_ahead = rows * num_ntts >= FETCH_AHEAD_MIN_WORDS;
        let drain_per_hook = (rows * num_ntts / 8).div_ceil(hooks);
        parallel::for_each_chunk(n_tasks, |lo, hi| {
            // One L2-resident scratch of 2^g rows serves every group of the task.
            // A draining task has two: one transforms while the other empties.
            let deferred = stream && lines_ok;
            with_scratch(stage_len + rows * num_ntts * (1 + usize::from(deferred)), |all| {
                // One fence at the end of the task covers all its streaming stores.
                let stream = stream.then(Stream::new);
                let (stage, both) = all.split_at_mut(stage_len);
                let (mut scratch, mut idle) = both.split_at_mut(rows * num_ntts);
                let mut drain = Drain::default();
                // `next`: the first row of the group after this one, and which share of its lines this group fetches.
                // `staged`: the group's rows, when a stage holds them.
                let mut group =
                    |block: usize, r: usize, next: Option<(*const F64, std::ops::Range<usize>)>, staged: &[F64]| {
                        // Row `i` of the group, as a word offset inside its block.
                        let row = |i: usize| (r + (i << log_step)) * num_ntts;
                        let block_off = (block << (log_d - layer)) * num_ntts;
                        // Gather: the scattered rows become one contiguous 2^g-row buffer.
                        for (i, dst) in scratch.chunks_exact_mut(num_ntts).enumerate() {
                            // SAFETY:
                            // - The message and the codeword both cover every row addressed here.
                            // - Tasks own disjoint residues, so no other task writes these rows.
                            // - Block 0 is written last, after every read of the message in it.
                            let src = unsafe {
                                match msg {
                                    _ if !staged.is_empty() => &staged[i * num_ntts..][..num_ntts],
                                    Some(m) => std::slice::from_raw_parts(m.add(row(i)), num_ntts),
                                    None => base.slice(block_off + row(i), num_ntts),
                                }
                            };
                            dst.copy_from_slice(src);
                        }
                        // Transform: a (layer + g)-layer domain whose sub-block index is the global block.
                        let (log_n, outer) = (layer + g, layer);
                        match &stream {
                            Some(s) if deferred => {
                                // The previous group drains while this one transforms; the rest drains now.
                                let (first, lines) = next.unwrap_or((std::ptr::null(), 0..0));
                                let mut fetch = prefetch_rows(first, num_ntts, step * num_ntts, lines, hooks);
                                self.run_layers(scratch, log_n, num_ntts, layer, log_n, outer, block, || {
                                    drain.step(s, drain_per_hook);
                                    fetch();
                                });
                                drain.step(s, usize::MAX);
                                // This group drains during the next transform, while the other scratch is gathered into.
                                drain = Drain {
                                    // SAFETY: the group's first row lies inside the codeword.
                                    dst: unsafe { base.0.add(block_off + row(0)) },
                                    src: scratch.as_ptr(),
                                    row_len: num_ntts,
                                    dst_stride: step * num_ntts,
                                    rows,
                                    ..Drain::default()
                                };
                                std::mem::swap(&mut scratch, &mut idle);
                            }
                            _ => {
                                self.run_layers(scratch, log_n, num_ntts, layer, log_n, outer, block, no_hook);
                                // Scatter: every row returns to its place.
                                for (i, src) in scratch.chunks_exact(num_ntts).enumerate() {
                                    // SAFETY: this group alone owns these rows of the codeword.
                                    let dst = unsafe { base.slice(block_off + row(i), num_ntts) };
                                    match &stream {
                                        Some(s) => s.copy(dst, src),
                                        None => dst.copy_from_slice(src),
                                    }
                                }
                            }
                        }
                    };
                let lines = rows * num_ntts / 8;
                for t in lo..hi {
                    if let (Some(m), Some(n_lanes)) = (msg, lane_major) {
                        // A task is a batch of residues: transpose their row groups into the stage, then run each.
                        let t0 = t * BATCH;
                        fill_stage(stage, m.0.cast_const(), t0, log_step, msg_rows, n_lanes, num_ntts);
                        // No fetch-ahead: a batch's lines sit 16 KB apart in blocks 16 MB apart, so they share
                        // a few L2 and L3 sets. Lines fetched a batch early are evicted before the fill reads them.
                        for (s, staged) in stage.chunks_exact(rows * num_ntts).enumerate() {
                            for block in (0..1usize << layer).rev() {
                                group(block, t0 + s, None, staged);
                            }
                        }
                    } else if let Some(m) = msg {
                        // Block 0 may be the message itself, so it is transformed last.
                        // The residue after this one reads fresh rows: its lines are fetched over this one's groups.
                        let blocks = 1usize << layer;
                        for (k, block) in (0..blocks).rev().enumerate() {
                            let next = (fetch_ahead && t + 1 < hi).then(|| {
                                let first = m.0.cast_const().wrapping_add((t + 1) * num_ntts);
                                (first, (k * lines) >> layer..((k + 1) * lines) >> layer)
                            });
                            group(block, t, next, &[]);
                        }
                    } else {
                        // Task index = block * step + residue; the next task reads its own, disjoint rows.
                        let next = (fetch_ahead && t + 1 < hi).then(|| {
                            let (block, r) = ((t + 1) >> log_step, (t + 1) & (step - 1));
                            (base.0.cast_const().wrapping_add(((block << (log_d - layer)) + r) * num_ntts), 0..lines)
                        });
                        group(t >> log_step, t & (step - 1), next, &[]);
                    }
                }
                if let Some(s) = &stream {
                    drain.step(s, usize::MAX);
                }
            });
        });
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
        self.run_layers_to(buf, log_d, num_ntts, first_layer, end_layer, outer_log, sub_idx, None, hook);
    }

    /// [`Self::run_layers`] that leaves the blocks alone which hold none of the buffer rows `live` (sorted).
    ///
    /// - Only those rows come out right: a block feeds only the rows inside it, in every later layer.
    /// - A fused sweep skips whole blocks of its first layer, so a sparse `live` prunes coarsely.
    #[allow(clippy::too_many_arguments)]
    fn run_layers_to(
        &self,
        buf: &mut [F64],
        log_d: usize,
        num_ntts: usize,
        first_layer: usize,
        end_layer: usize,
        outer_log: usize,
        sub_idx: usize,
        live: Option<&[usize]>,
        mut hook: impl FnMut(),
    ) {
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

/// Call `f` with each block of `1 << log_rows` rows that holds a row of `live` (sorted), or with every block of `num_blocks`.
fn for_each_block(num_blocks: usize, log_rows: usize, live: Option<&[usize]>, mut f: impl FnMut(usize)) {
    match live {
        None => (0..num_blocks).for_each(f),
        Some(rows) => rows.chunk_by(|a, b| a >> log_rows == b >> log_rows).for_each(|run| f(run[0] >> log_rows)),
    }
}

/// How many times [`AdditiveNttF64::run_layers`] calls its hook over `first_layer..end_layer`.
fn hook_calls(log_d: usize, first_layer: usize, end_layer: usize, outer_log: usize) -> usize {
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

/// A finished row group of a gathered pass, leaving its scratch for the codeword a few lines at a time.
///
/// - `dst` is the group's first row in the codeword, `dst_stride` words before its next.
/// - `src` is the group's contiguous scratch.
/// - `row`, `line` are the next cache line to move.
/// - Both sides are whole, aligned cache lines: the caller checks the codeword's alignment and `row_len % 8 == 0`.
struct Drain {
    dst: *mut F64,
    src: *const F64,
    row_len: usize,
    dst_stride: usize,
    rows: usize,
    row: usize,
    line: usize,
}

impl Default for Drain {
    /// A group with no rows left to move.
    fn default() -> Self {
        Self {
            dst: std::ptr::null_mut(),
            src: std::ptr::null(),
            row_len: 0,
            dst_stride: 0,
            rows: 0,
            row: 0,
            line: 0,
        }
    }
}

impl Drain {
    /// Stream out up to `lines` more cache lines.
    #[inline(always)]
    fn step(&mut self, stream: &Stream, lines: usize) {
        for _ in 0..lines.min((self.rows - self.row) * (self.row_len / 8) - self.line) {
            // SAFETY:
            // - The cursor is inside the group, so both lines are in bounds and line-aligned.
            // - Only this group's owner writes these rows of the codeword.
            // - The scratch half is neither written nor gathered into until its group has drained.
            unsafe {
                let dst = self.dst.add(self.row * self.dst_stride + self.line * 8);
                let src = self.src.add(self.row * self.row_len + self.line * 8);
                stream.copy_lines(dst.cast::<Line>(), &*src.cast::<Line>());
            }
            self.line += 1;
            if self.line == self.row_len / 8 {
                (self.row, self.line) = (self.row + 1, 0);
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

/// A hook that prefetches cache lines `lines` of a row group into L2, evenly over `calls` calls.
///
/// - The group's rows are `row_words` words long and `stride` words apart, from `first`.
/// - Line `n` is line `n % (row_words / 8)` of row `n / (row_words / 8)`.
fn prefetch_rows(
    first: *const F64,
    row_words: usize,
    stride: usize,
    lines: std::ops::Range<usize>,
    calls: usize,
) -> impl FnMut() {
    let per_row = row_words / 8;
    let mut left = lines.len();
    // A group with nothing to fetch divides nothing.
    let per_call = if left == 0 { 0 } else { left.div_ceil(calls) };
    // The cursor: the next line's row and its offset inside the row, so no call divides.
    let (mut row, mut col) = if left == 0 { (0, 0) } else { (lines.start / per_row, lines.start % per_row) };
    move || {
        for _ in 0..per_call.min(left) {
            // Only an address is formed, never a reference.
            let line = first.wrapping_add(row * stride + col * 8);
            // SAFETY: a prefetch of any address is harmless.
            #[cfg(target_arch = "x86_64")]
            unsafe {
                core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T1 }>(line.cast())
            };
            col += 1;
            if col == per_row {
                (row, col) = (row + 1, 0);
            }
        }
        left = left.saturating_sub(per_call);
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
/// # Why the rows are disjoint
///
/// - Distinct groups take distinct rows inside each slab.
/// - The slabs do not overlap.
/// - So the groups are pairwise disjoint, and one base pointer can stand in for N nested splits.
fn fused_rows<const N: usize>(
    block: &mut [F64],
    stride_rows: usize,
    num_ntts: usize,
    do_one: impl Fn(&mut [&mut [F64]; N]),
    mut hook: impl FnMut(),
) {
    // Words from one slab to the next.
    let stride = stride_rows * num_ntts;
    debug_assert_eq!(block.len(), N * stride);
    let base = block.as_mut_ptr();
    for r in 0..stride_rows {
        // Row `r` of each slab starts this many words into it.
        let off = r * num_ntts;
        // SAFETY:
        // - The groups are disjoint, as argued above.
        // - The row ends inside its slab, since the row index is below the slab height.
        let mut rows: [&mut [F64]; N] =
            std::array::from_fn(|i| unsafe { std::slice::from_raw_parts_mut(base.add(i * stride + off), num_ntts) });
        do_one(&mut rows);
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
    t: &[F64; 7],
    eighth: usize,
    num_ntts: usize,
    hook: impl FnMut(),
) {
    fused_rows::<8>(block, eighth, num_ntts, |rows| radix8_butterflies(rows, t), hook);
}

/// The twelve butterflies of one radix-8 row group, with its seven twiddles breadth-first.
///
/// On AVX-512 they run on registers: each row's vectors are loaded and stored once per group.
#[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
#[inline(always)]
fn radix8_butterflies(rows: &mut [&mut [F64]; 8], t: &[F64; 7]) {
    // Inlined into the sweep: a call would pass the vectors through memory.
    #[allow(clippy::redundant_closure)] // the closure carries the inline attribute
    fused_lanes_avx512(
        rows,
        t,
        #[inline(always)]
        |v, t| radix8_regs(v, t),
    );
}

/// [`radix8_butterflies`] on registers.
#[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
#[inline(always)]
fn radix8_regs(v: &mut RowVectors<8>, t: &[Twiddle; 7]) {
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
}

/// The twelve butterflies of one radix-8 row group, with its seven twiddles breadth-first.
#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
#[inline(always)]
fn radix8_butterflies(rows: &mut [&mut [F64]; 8], t: &[F64; 7]) {
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
/// - A Merkle leaf reads its lanes from the top index down.
/// - So the absent all-zero lanes lead every leaf image.
/// - Their hash prefix is then one chaining value that every leaf shares.
/// - The proof can leave them out.
/// - Each lane is an independent codeword, so the order costs nothing.
///
/// # Performance
///
/// - Rows are handled one tile at a time.
/// - Each lane adds a contiguous burst of words to an L1-resident tile.
/// - The finished tile goes out as one contiguous run.
/// - That keeps an n-way gather at a `2^rows_log` stride near memory bandwidth.
#[timed]
pub fn transpose_lane_major(out: &mut [F64], msg: &[F64], n_lanes: usize, row_words: usize, log_rows: usize) {
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
const BATCH: usize = 1 << LOG_BATCH;

/// Gathered groups of at least this many words fetch the next group's source rows while they transform.
const FETCH_AHEAD_MIN_WORDS: usize = 1 << 13;

/// One cache line of words.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Line([F64; 8]);

/// Lend this thread's scratch buffer, grown to the requested length.
///
/// - It is cache-line aligned, so a full-width load never splits a line.
/// - It lives as long as the thread, so no pass allocates in its hot loop.
fn with_scratch<R>(len: usize, f: impl FnOnce(&mut [F64]) -> R) -> R {
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

/// Transpose the row groups of [`BATCH`] consecutive residues of a lane-major message into `stage`.
///
/// - Message row `j` is `msg_rows` words apart per lane block, and word `w` of a row of `row_words` words
///   is lane block `row_words - 1 - w` of the message, or zero while `w` is below the padding.
/// - Residue `t0 + s` takes rows `t0 + s + i * step` for `i < rows`, `step = 2^log_step`.
/// - `stage` holds the residues' groups back to back: row `i` of residue `s` is at `(s * rows + i) * row_words`.
/// - One tile is 8 words of 8 lane blocks, so each 64 byte line of a lane block is read once, whole.
fn fill_stage(
    stage: &mut [F64],
    msg: *const F64,
    t0: usize,
    log_step: usize,
    msg_rows: usize,
    n_lanes: usize,
    row_words: usize,
) {
    let rows = stage.len() / (BATCH * row_words);
    let pad = row_words - n_lanes;
    let out = stage.as_mut_ptr();
    // Octet by octet, so the hardware prefetcher follows 8 lane streams, not a row's 48 lines in one L2 set.
    for octet in 0..row_words / 8 {
        for i in 0..rows {
            let j = t0 + (i << log_step);
            // Line `k` of the tile: 8 residues of one lane block, or null for a padding lane.
            let src: [*const F64; 8] = std::array::from_fn(|k| {
                let w = 8 * octet + k;
                if w < pad {
                    std::ptr::null()
                } else {
                    msg.wrapping_add((row_words - 1 - w) * msg_rows + j)
                }
            });
            // SAFETY:
            // - Every non-null source is `BATCH` words inside lane block `row_words - 1 - w`, as the caller's rows are.
            // - Every destination is a whole line of the stage: `s < BATCH`, `i < rows`, `octet < row_words / 8`.
            unsafe { transpose_tile(&src, |s| out.add((s * rows + i) * row_words + 8 * octet)) };
        }
    }
}

/// Write `dst(s)[k] = src[k][s]` for the 8 x 8 words of eight lines, a null line being zeros.
///
/// # Safety
///
/// Every non-null `src[k]` is valid for 8 reads, and every `dst(s)` for 8 writes, apart from the sources.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn transpose_tile(src: &[*const F64; 8], dst: impl Fn(usize) -> *mut F64) {
    use core::arch::x86_64::*;
    // SAFETY: the caller's bounds; AVX-512F is enabled at compile time.
    unsafe {
        let r: [__m512i; 8] = std::array::from_fn(|k| {
            if src[k].is_null() {
                _mm512_setzero_si512()
            } else {
                _mm512_loadu_si512(src[k].cast())
            }
        });
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
            _mm512_storeu_si512(dst(s).cast(), col);
        }
    }
}

/// [`transpose_tile`] word by word.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
unsafe fn transpose_tile(src: &[*const F64; 8], dst: impl Fn(usize) -> *mut F64) {
    for s in 0..8 {
        for (k, line) in src.iter().enumerate() {
            // SAFETY: the caller's bounds.
            unsafe {
                dst(s)
                    .add(k)
                    .write(if line.is_null() { F64::ZERO } else { line.add(s).read() })
            };
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
    t: &[F64; 3],
    quarter: usize,
    num_ntts: usize,
    hook: impl FnMut(),
) {
    fused_rows::<4>(block, quarter, num_ntts, |rows| radix4_butterflies(rows, t), hook);
}

/// The four butterflies of one radix-4 row group.
///
/// On AVX-512 they run on registers, like [`radix8_butterflies`].
#[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
#[inline(always)]
fn radix4_butterflies(rows: &mut [&mut [F64]; 4], t: &[F64; 3]) {
    #[allow(clippy::redundant_closure)] // the closure carries the inline attribute
    fused_lanes_avx512(
        rows,
        t,
        #[inline(always)]
        |v, t| {
            // Layer L: rows 2 apart, one twiddle for the block.
            butterfly_regs(v, 0, 2, t[0]);
            butterfly_regs(v, 1, 3, t[0]);
            // Layer L+1: adjacent rows, one twiddle per half.
            butterfly_regs(v, 0, 1, t[1]);
            butterfly_regs(v, 2, 3, t[2]);
        },
    );
}

/// The four butterflies of one radix-4 row group.
#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
#[inline(always)]
fn radix4_butterflies(rows: &mut [&mut [F64]; 4], t: &[F64; 3]) {
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
/// On NEON this processes eight lanes per iteration. Four independent pair
/// reductions stay in the vector register file, exposing their PMULL chains
/// in parallel and amortizing the loop branch and constant setup. The pair
/// kernel handles a short even tail, and the scalar path handles an odd tail.
#[inline]
fn butterfly_lanes(top: &mut [F64], bot: &mut [F64], twiddle: F64) {
    debug_assert_eq!(top.len(), bot.len());
    // Every layer's block 0 has twiddle zero: the butterfly is then `v' = v + u`, no product.
    if twiddle == F64::ZERO {
        for (v, &u) in bot.iter_mut().zip(top.iter()) {
            *v += u;
        }
        return;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
    {
        let vectors = top.len() / 8;
        // SAFETY: the target features are enabled at compile time and each
        // iteration reads and writes exactly eight elements from both rows.
        unsafe {
            for i in 0..vectors {
                butterfly_lanes_avx512(top.as_mut_ptr().add(8 * i), bot.as_mut_ptr().add(8 * i), twiddle.0, 0xFF);
            }
            let tail = top.len() - 8 * vectors;
            if tail > 0 {
                let mask = (1u8 << tail) - 1;
                butterfly_lanes_avx512(top.as_mut_ptr().add(8 * vectors), bot.as_mut_ptr().add(8 * vectors), twiddle.0, mask);
            }
        }
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "vpclmulqdq",
        target_feature = "avx2",
        not(target_feature = "avx512f")
    ))]
    {
        let vectors = top.len() / 4;
        // SAFETY: the target features are enabled at compile time and each
        // iteration reads and writes exactly four elements from both rows.
        unsafe {
            for i in 0..vectors {
                butterfly_lanes_avx2(top.as_mut_ptr().add(4 * i), bot.as_mut_ptr().add(4 * i), twiddle.0);
            }
        }
        for lane in 4 * vectors..top.len() {
            let v = bot[lane];
            let new_u = top[lane] + v * twiddle;
            top[lane] = new_u;
            bot[lane] = v + new_u;
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        let vectors = top.len() / 8;
        // SAFETY: aes target feature is enabled at compile time; the kernel
        // reads/writes exactly lanes [8i, 8i+8) of each row.
        unsafe {
            for i in 0..vectors {
                butterfly_lanes_neon_8(top.as_mut_ptr().add(8 * i), bot.as_mut_ptr().add(8 * i), twiddle.0);
            }
            let mut lane = 8 * vectors;
            while lane + 2 <= top.len() {
                butterfly_lane_pair_neon(top.as_mut_ptr().add(lane), bot.as_mut_ptr().add(lane), twiddle.0);
                lane += 2;
            }
        }
        if top.len() % 2 == 1 {
            let last = top.len() - 1;
            let v = bot[last];
            let new_u = top[last] + v * twiddle;
            top[last] = new_u;
            bot[last] = v + new_u;
        }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx2")
    )))]
    {
        for lane in 0..top.len() {
            let v = bot[lane];
            let new_u = top[lane] + v * twiddle;
            top[lane] = new_u;
            bot[lane] = v + new_u;
        }
    }
}

/// The transpose of [`butterfly_lanes`]: `s = u + v; u' = s; v' = s * t + v` over a pair of rows.
/// Only AVX-512 has a kernel; other targets take the scalar loop.
pub(crate) fn transpose_butterfly_lanes(top: &mut [F64], bot: &mut [F64], twiddle: F64) {
    debug_assert_eq!(top.len(), bot.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
    {
        // SAFETY: the target features are enabled at compile time and each call
        // touches only the words its mask selects, all within both rows.
        unsafe {
            let vectors = top.len() / 8;
            for i in 0..vectors {
                transpose_butterfly_lanes_avx512(top.as_mut_ptr().add(8 * i), bot.as_mut_ptr().add(8 * i), twiddle.0, 0xFF);
            }
            let tail = top.len() - 8 * vectors;
            if tail > 0 {
                let mask = (1u8 << tail) - 1;
                transpose_butterfly_lanes_avx512(top.as_mut_ptr().add(8 * vectors), bot.as_mut_ptr().add(8 * vectors), twiddle.0, mask);
            }
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
    for (u, v) in top.iter_mut().zip(bot.iter_mut()) {
        let s = *u + *v;
        *u = s;
        *v += s * twiddle;
    }
}

/// [`butterfly_lanes_avx512`] at half the width, for a machine with VPCLMULQDQ
/// but no AVX-512. Without it the base encode's innermost loop is scalar there,
/// which costs it about half again as much.
///
/// # Safety
/// Requires VPCLMULQDQ + AVX2; `top` and `bot` must each address four readable
/// and writable F64 values.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "vpclmulqdq",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
#[inline]
#[target_feature(enable = "vpclmulqdq", enable = "avx2")]
unsafe fn butterfly_lanes_avx2(top: *mut F64, bot: *mut F64, twiddle: u64) {
    use core::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "vpclmulqdq", enable = "avx2")]
    unsafe fn reduce(p: __m256i, r: __m256i) -> __m256i {
        let t = _mm256_clmulepi64_epi128::<0x01>(p, r);
        let u = _mm256_clmulepi64_epi128::<0x01>(t, r);
        _mm256_xor_si256(_mm256_xor_si256(p, t), u)
    }

    // SAFETY: the caller supplies valid four-element rows and the function's
    // target features cover every intrinsic below.
    unsafe {
        let u = _mm256_loadu_si256(top.cast());
        let v = _mm256_loadu_si256(bot.cast());
        let tw = _mm256_set1_epi64x(twiddle as i64);
        let r = _mm256_set1_epi64x(0x1b);

        let even = reduce(_mm256_clmulepi64_epi128::<0x00>(v, tw), r);
        let odd = reduce(_mm256_clmulepi64_epi128::<0x11>(v, tw), r);
        // The odd products land in each lane's low half; swap them up, then take
        // the odd qwords (32-bit elements 2, 3, 6, 7) from them.
        let odd = _mm256_shuffle_epi32::<0x4e>(odd);
        let product = _mm256_blend_epi32::<0b1100_1100>(even, odd);

        let new_u = _mm256_xor_si256(u, product);
        let new_v = _mm256_xor_si256(v, new_u);
        _mm256_storeu_si256(top.cast(), new_u);
        _mm256_storeu_si256(bot.cast(), new_v);
    }
}

/// Eight F64 butterflies as four independent NEON lane-pair reductions.
/// Loading all four bottom vectors before reducing them gives the out-of-order
/// core four independent PMULL chains to schedule, while one call amortizes
/// loop control and the duplicated twiddle/reduction constants over 8 lanes.
///
/// # Safety
/// Requires the `aes` target feature; `top`/`bot` must each point at eight
/// readable+writable F64 values.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn butterfly_lanes_neon_8(top: *mut F64, bot: *mut F64, twiddle: u64) {
    use core::arch::aarch64::*;
    use primitives::field::gf2_64::aarch64::reduce_pair_pmull4;

    // SAFETY: caller guarantees the two eight-element regions; F64 is
    // repr(transparent) over u64 and this function carries the aes feature.
    unsafe {
        let v0 = vld1q_u64(bot.cast());
        let v1 = vld1q_u64(bot.cast::<u64>().add(2));
        let v2 = vld1q_u64(bot.cast::<u64>().add(4));
        let v3 = vld1q_u64(bot.cast::<u64>().add(6));
        let tw = vdupq_n_u64(twiddle);

        let p00: uint64x2_t = core::mem::transmute(vmull_p64(vgetq_lane_u64::<0>(v0), twiddle));
        let p01: uint64x2_t = core::mem::transmute(vmull_high_p64(
            core::mem::transmute::<uint64x2_t, poly64x2_t>(v0),
            core::mem::transmute::<uint64x2_t, poly64x2_t>(tw),
        ));
        let p10: uint64x2_t = core::mem::transmute(vmull_p64(vgetq_lane_u64::<0>(v1), twiddle));
        let p11: uint64x2_t = core::mem::transmute(vmull_high_p64(
            core::mem::transmute::<uint64x2_t, poly64x2_t>(v1),
            core::mem::transmute::<uint64x2_t, poly64x2_t>(tw),
        ));
        let p20: uint64x2_t = core::mem::transmute(vmull_p64(vgetq_lane_u64::<0>(v2), twiddle));
        let p21: uint64x2_t = core::mem::transmute(vmull_high_p64(
            core::mem::transmute::<uint64x2_t, poly64x2_t>(v2),
            core::mem::transmute::<uint64x2_t, poly64x2_t>(tw),
        ));
        let p30: uint64x2_t = core::mem::transmute(vmull_p64(vgetq_lane_u64::<0>(v3), twiddle));
        let p31: uint64x2_t = core::mem::transmute(vmull_high_p64(
            core::mem::transmute::<uint64x2_t, poly64x2_t>(v3),
            core::mem::transmute::<uint64x2_t, poly64x2_t>(tw),
        ));

        let prod0 = reduce_pair_pmull4(p00, p01);
        let prod1 = reduce_pair_pmull4(p10, p11);
        let prod2 = reduce_pair_pmull4(p20, p21);
        let prod3 = reduce_pair_pmull4(p30, p31);

        let u0 = vld1q_u64(top.cast());
        let u1 = vld1q_u64(top.cast::<u64>().add(2));
        let u2 = vld1q_u64(top.cast::<u64>().add(4));
        let u3 = vld1q_u64(top.cast::<u64>().add(6));
        let new_u0 = veorq_u64(u0, prod0);
        let new_u1 = veorq_u64(u1, prod1);
        let new_u2 = veorq_u64(u2, prod2);
        let new_u3 = veorq_u64(u3, prod3);
        let new_v0 = veorq_u64(v0, new_u0);
        let new_v1 = veorq_u64(v1, new_u1);
        let new_v2 = veorq_u64(v2, new_u2);
        let new_v3 = veorq_u64(v3, new_u3);

        vst1q_u64(top.cast(), new_u0);
        vst1q_u64(top.cast::<u64>().add(2), new_u1);
        vst1q_u64(top.cast::<u64>().add(4), new_u2);
        vst1q_u64(top.cast::<u64>().add(6), new_u3);
        vst1q_u64(bot.cast(), new_v0);
        vst1q_u64(bot.cast::<u64>().add(2), new_v1);
        vst1q_u64(bot.cast::<u64>().add(4), new_v2);
        vst1q_u64(bot.cast::<u64>().add(6), new_v3);
    }
}

/// `g(spill)` per top nibble `n` of a product's high word: `spill = n ^ n>>1 ^ n>>3`, the bits
/// `g(hi) = hi * (x^4 + x^3 + x + 1)` pushes past x^63, then `g(spill)` itself, which fits a byte.
#[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
const G_SPILL: [u8; 16] = {
    let mut t = [0u8; 16];
    let mut n = 0;
    while n < 16 {
        let s = n ^ (n >> 1) ^ (n >> 3);
        t[n] = (s ^ (s << 1) ^ (s << 3) ^ (s << 4)) as u8;
        n += 1;
    }
    t
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
///
/// # Safety
///
/// Callable only where VPCLMULQDQ, AVX-512F and AVX-512BW are enabled, which the kernels' target features guarantee.
#[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn mul_xor_avx512(
    u: core::arch::x86_64::__m512i,
    v: core::arch::x86_64::__m512i,
    tw: core::arch::x86_64::__m512i,
) -> core::arch::x86_64::__m512i {
    use core::arch::x86_64::*;

    // SAFETY: the caller's target features cover every intrinsic below.
    unsafe {
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
        let g_spill = _mm512_shuffle_epi8(_mm512_broadcast_i32x4(_mm_loadu_si128(G_SPILL.as_ptr().cast())), nibble);
        // g(hi) = y ^ y<<3, with y = hi ^ hi<<1.
        let y = _mm512_xor_si512(hi, _mm512_slli_epi64::<1>(hi));
        _mm512_ternarylogic_epi64::<XOR3>(
            _mm512_ternarylogic_epi64::<XOR3>(u, lo, g_spill),
            y,
            _mm512_slli_epi64::<3>(y),
        )
    }
}

/// Eight F64 butterflies with a shared twiddle, one per 64-bit lane of an AVX-512 register.
///
/// # Safety
///
/// - Requires VPCLMULQDQ, AVX-512F and AVX-512BW.
/// - Each pointer must address the words `mask` selects, readable and writable.
#[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
#[inline]
#[target_feature(enable = "vpclmulqdq", enable = "avx512f", enable = "avx512bw", enable = "avx2")]
unsafe fn butterfly_lanes_avx512(top: *mut F64, bot: *mut F64, twiddle: u64, mask: u8) {
    use core::arch::x86_64::*;

    // SAFETY:
    // - The caller supplies two rows valid for the lanes `mask` selects.
    // - This function's target features cover every intrinsic below.
    unsafe {
        // Load both rows and broadcast the twiddle to every lane.
        let u = _mm512_maskz_loadu_epi64(mask, top.cast());
        let v = _mm512_maskz_loadu_epi64(mask, bot.cast());
        let tw = _mm512_set1_epi64(twiddle as i64);
        // Butterfly top: u' = u + v * t. Bottom: v' = v + u'.
        let new_u = mul_xor_avx512(u, v, tw);
        let new_v = _mm512_xor_si512(v, new_u);
        _mm512_mask_storeu_epi64(top.cast(), mask, new_u);
        _mm512_mask_storeu_epi64(bot.cast(), mask, new_v);
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
#[inline(always)]
fn butterfly_regs<const N: usize>(v: &mut RowVectors<N>, top: usize, bot: usize, (tw, zero): Twiddle) {
    for k in 0..LANE_VECTORS {
        let (u, w) = (v[top][k], v[bot][k]);
        // SAFETY: the target features are enabled at compile time.
        let new_u = if zero { u } else { unsafe { mul_xor_avx512(u, w, tw) } };
        v[top][k] = new_u;
        v[bot][k] = unsafe { core::arch::x86_64::_mm512_xor_si512(w, new_u) };
    }
}

/// Runs `kernel` on `N` equal-length rows held in registers, [`LANE_VECTORS`] eight-lane vectors of every row at a time.
///
/// - A vector is loaded and stored once, however many butterflies `kernel` applies to it.
/// - The kernel orders its butterflies breadth-first, so the vectors in flight hide each other's latency.
/// - The last chunk may be short: its vectors past the row's end load as zeros and are not stored.
#[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
#[inline(always)]
fn fused_lanes_avx512<const N: usize, const T: usize>(
    rows: &mut [&mut [F64]; N],
    twiddles: &[F64; T],
    kernel: impl Fn(&mut RowVectors<N>, &[Twiddle; T]),
) {
    use core::arch::x86_64::*;

    const CHUNK: usize = 8 * LANE_VECTORS;
    let len = rows[0].len();
    debug_assert!(rows.iter().all(|row| row.len() == len));
    let ptrs: [*mut F64; N] = std::array::from_fn(|i| rows[i].as_mut_ptr());
    // SAFETY: the target features are enabled at compile time, and every load and store
    // stays inside its row: a whole chunk ends by `len`, and a short one is masked at `len`.
    unsafe {
        let tw = twiddles.map(|t| (_mm512_set1_epi64(t.0 as i64), t == F64::ZERO));
        let mut lane = 0;
        while lane + CHUNK <= len {
            let mut v: RowVectors<N> =
                std::array::from_fn(|i| std::array::from_fn(|k| _mm512_loadu_si512(ptrs[i].add(lane + 8 * k).cast())));
            kernel(&mut v, &tw);
            for i in 0..N {
                for k in 0..LANE_VECTORS {
                    _mm512_storeu_si512(ptrs[i].add(lane + 8 * k).cast(), v[i][k]);
                }
            }
            lane += CHUNK;
        }
        if lane < len {
            // The lanes of vector `k` of the short chunk: all eight, the row's last few, or none.
            let mask = |k: usize| ((1u16 << len.saturating_sub(lane + 8 * k).min(8)) - 1) as u8;
            let mut v: RowVectors<N> = std::array::from_fn(|i| {
                std::array::from_fn(|k| _mm512_maskz_loadu_epi64(mask(k), ptrs[i].add(lane + 8 * k).cast()))
            });
            kernel(&mut v, &tw);
            for i in 0..N {
                for k in 0..LANE_VECTORS {
                    _mm512_mask_storeu_epi64(ptrs[i].add(lane + 8 * k).cast(), mask(k), v[i][k]);
                }
            }
        }
    }
}

/// [`butterfly_lanes_avx512`] transposed: `s = u + v; u' = s; v' = s * t + v`.
///
/// # Safety
///
/// Same as [`butterfly_lanes_avx512`].
#[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
#[inline]
#[target_feature(enable = "vpclmulqdq", enable = "avx512f", enable = "avx512bw", enable = "avx2")]
unsafe fn transpose_butterfly_lanes_avx512(top: *mut F64, bot: *mut F64, twiddle: u64, mask: u8) {
    use core::arch::x86_64::*;

    // SAFETY: as in `butterfly_lanes_avx512`.
    unsafe {
        let u = _mm512_maskz_loadu_epi64(mask, top.cast());
        let v = _mm512_maskz_loadu_epi64(mask, bot.cast());
        let tw = _mm512_set1_epi64(twiddle as i64);
        let s = _mm512_xor_si512(u, v);
        _mm512_mask_storeu_epi64(top.cast(), mask, s);
        _mm512_mask_storeu_epi64(bot.cast(), mask, mul_xor_avx512(v, s, tw));
    }
}

/// Two F64 butterflies with a shared twiddle, NEON-resident end to end.
/// The two products issue as PMULL/PMULL2 on the loaded row (no lane
/// extraction) and reduce through the all-PMULL lane-pair fold
/// ([`primitives::field::gf2_64::aarch64::reduce_pair_pmull4`]), replacing the
/// old 10-op shift-XOR fold chain.
///
/// # Safety
/// Requires the `aes` target feature; `top`/`bot` must each point at two
/// readable+writable F64 values.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn butterfly_lane_pair_neon(top: *mut F64, bot: *mut F64, twiddle: u64) {
    use core::arch::aarch64::*;
    use primitives::field::gf2_64::aarch64::reduce_pair_pmull4;
    // SAFETY: caller guarantees the pointees; F64 is repr(transparent) u64.
    unsafe {
        let u = vld1q_u64(top as *const u64);
        let v = vld1q_u64(bot as *const u64);
        // Products v_lane * twiddle: PMULL on the low lanes, PMULL2 on the
        // highs (the dup is loop-invariant and hoisted after inlining).
        let tw = vdupq_n_u64(twiddle);
        let p0: uint64x2_t = core::mem::transmute(vmull_p64(vgetq_lane_u64::<0>(v), twiddle));
        let p1: uint64x2_t = core::mem::transmute(vmull_high_p64(
            core::mem::transmute::<uint64x2_t, poly64x2_t>(v),
            core::mem::transmute::<uint64x2_t, poly64x2_t>(tw),
        ));
        let prod = reduce_pair_pmull4(p0, p1);
        let new_u = veorq_u64(u, prod);
        let new_v = veorq_u64(v, new_u);
        vst1q_u64(top as *mut u64, new_u);
        vst1q_u64(bot as *mut u64, new_v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use primitives::test_rng::Rng;

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
            let blocks = std::sync::Mutex::new(Vec::new());
            ntt.encode_interleaved_in_place_with(&mut got, lanes, log_inv_rate, &|row, rows| {
                blocks.lock().unwrap().push((row, rows.to_vec()))
            });
            assert_eq!(got, want, "log_d={log_d}, lanes={lanes}, rate={log_inv_rate}");

            let mut blocks = blocks.into_inner().unwrap();
            blocks.sort_by_key(|b| b.0);
            let mut next = 0;
            for (row, rows) in blocks {
                assert_eq!(row, next, "blocks tile the rows, log_d={log_d}");
                assert_eq!(
                    rows[..],
                    want[row * lanes..][..rows.len()],
                    "a handed-over block is final"
                );
                next += rows.len() / lanes;
            }
            assert_eq!(next, 1 << log_d, "every row handed over, log_d={log_d}");
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
            let msg: Vec<F64> = (0..n_lanes << log_rows).map(|_| F64(rng.next_u64())).collect();
            let row_words = n_lanes.next_multiple_of(8);
            let mut want = vec![F64::ZERO; (row_words << log_rows) << log_inv_rate];
            transpose_lane_major(&mut want[..row_words << log_rows], &msg, n_lanes, row_words, log_rows);
            ntt.encode_interleaved_in_place(&mut want, row_words, log_inv_rate);

            for defer_deep in [false, true] {
                let mut got = vec![F64(0xBAD); want.len()];
                let handed = std::sync::Mutex::new(vec![F64::ZERO; got.len()]);
                let on_rows = |row: usize, rows: &[F64]| {
                    handed.lock().unwrap()[row * row_words..][..rows.len()].copy_from_slice(rows)
                };
                let deep =
                    ntt.encode_lane_major(&mut got, &msg, n_lanes, row_words, log_inv_rate, &on_rows, defer_deep);
                let context =
                    format!("log_rows={log_rows}, rate={log_inv_rate}, n_lanes={n_lanes}, defer_deep={defer_deep}");
                assert_eq!(handed.into_inner().unwrap(), want, "handed-over rows, {context}");
                let positions: Vec<usize> = (0..40).map(|_| rng.next_u64() as usize % (1 << log_d)).collect();
                let rows = match deep {
                    // A lane-major deferral leaves a codeword the message rebuilds.
                    Some(deep) => {
                        assert_eq!(ntt.finish_rows(&got, &deep, &positions), ntt.rebuild_rows(&msg, n_lanes, &deep, &positions));
                        ntt.rebuild_rows(&msg, n_lanes, &deep, &positions)
                    }
                    None => {
                        assert_eq!(got, want, "codeword, {context}");
                        positions
                            .iter()
                            .map(|&q| got[q * row_words..][..row_words].to_vec())
                            .collect()
                    }
                };
                for (&q, row) in positions.iter().zip(&rows) {
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
            let handed = std::sync::Mutex::new(vec![F64::ZERO; got.len()]);
            let deep = ntt.encode_interleaved_in_place_without_deep(&mut got, lanes, log_inv_rate, &|row, rows| {
                handed.lock().unwrap()[row * lanes..][..rows.len()].copy_from_slice(rows)
            });
            assert_eq!(handed.into_inner().unwrap(), want, "handed-over rows, log_d={log_d}, lanes={lanes}");
            assert_eq!(deep.is_some(), split, "plan, log_d={log_d}, lanes={lanes}");
            let positions: Vec<usize> = (0..40).map(|_| rng.next_u64() as usize % (1 << log_d)).collect();
            let rows = match deep {
                Some(deep) => ntt.finish_rows(&got, &deep, &positions),
                None => positions.iter().map(|&q| got[q * lanes..][..lanes].to_vec()).collect(),
            };
            for (&q, row) in positions.iter().zip(&rows) {
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

            let handed = std::sync::Mutex::new(vec![F64::ZERO; want.len()]);
            ntt.encode_cosets(&msg, lanes, log_inv_rate, &|row, rows| {
                handed.lock().unwrap()[row * lanes..][..rows.len()].copy_from_slice(rows)
            });
            assert_eq!(handed.into_inner().unwrap(), want, "handed-over rows, log_rows={log_rows}, lanes={lanes}");

            // Repeats and unsorted positions included.
            let mut positions: Vec<usize> = (0..60).map(|_| rng.next_u64() as usize % (1 << log_d)).collect();
            positions.extend_from_within(..5);
            let rows = ntt.coset_rows(&msg, lanes, log_inv_rate, &positions);
            for (&q, row) in positions.iter().zip(&rows) {
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
