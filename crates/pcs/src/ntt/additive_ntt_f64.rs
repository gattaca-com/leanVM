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

use super::gathered::GatheredPass;
use super::layers::{Layers, no_hook};
use super::plan::{L2_WORDS, Plan, l2_words};
use super::rows::built_rows;
use super::twiddles::Twiddles;
use super::source::{Lanes, Source};
use super::window::Window;

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
    pub deep: Option<Plan>,
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
/// Additive NTT over F_{2^64} with the standard polynomial-basis subspace
/// `{1, x, x², …}`: the F_2-subspace is `{0, 1, …, 2^ℓ−1}` under the natural
/// integer encoding.
///
/// - It owns the [`Twiddles`] of the domain.
/// - Every method here forwards to the component that does the work.
#[derive(Clone, Debug)]
pub struct AdditiveNttF64 {
    tw: Twiddles,
}

impl AdditiveNttF64 {
    /// The layer sweeps over this domain.
    pub(super) fn layers(&self) -> Layers<'_> {
        Layers::new(&self.tw)
    }

    /// Standard NTT with basis `{1, x, …, x^(dim-1)}`. Requires `dim ≤ 63` so
    /// the evaluation domain (and the twiddles) stay inside F_{2^64} without
    /// wrap; far beyond any codeword size in use.
    pub fn standard(dim: usize) -> Self {
        Self { tw: Twiddles::standard(dim) }
    }

    pub fn log_domain_size(&self) -> usize {
        self.tw.log_domain_size()
    }

    /// Twiddle of one block at one layer, see [`Twiddles::twiddle`].
    #[inline]
    pub fn twiddle(&self, layer: usize, block: usize) -> F64 {
        self.tw.twiddle(layer, block)
    }

    /// Recover novel-basis coefficients from evaluations with a scalar inverse NTT.
    pub fn inverse_transform(&self, data: &mut [F64]) {
        self.tw.inverse_transform(data)
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
        self.transform(&mut Window::new(data, num_ntts, 0), log_inv_rate, Source::InPlace, None, false);
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
        let rebuildable =
            Plan::for_lanes(log_d, num_ntts, log_inv_rate).is_some_and(|plan| plan.single_gathered_pass());
        let by_halves = keep == Keep::Nothing && rebuildable;
        let defer_deep = by_halves || keep == Keep::BeforeDeep;
        // The cosets one pass holds: half of them when the opening rebuilds the rows.
        let held = (1usize << log_inv_rate) >> usize::from(by_halves);
        let mut deep = None;
        // Every word is written by the gathered pass before the deep pass reads it.
        let data = primitives::filled_vec((held * msg_rows) * num_ntts, |data| {
            for first in (0..1usize << log_inv_rate).step_by(held).rev() {
                let src = Source::Lanes(Lanes { words: msg, repeats, n_lanes });
                let mut win = Window::new(data, num_ntts, first);
                deep = self.transform(&mut win, log_inv_rate, src, Some(on_rows), defer_deep);
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
        let log_d = self.coset_log_d(msg.len(), num_ntts, log_inv_rate);
        built_rows(num_ntts, log_d - log_inv_rate, positions, |_, coset, buf| {
            self.build_coset(buf, msg, num_ntts, log_d, log_inv_rate, coset)
        })
    }

    /// Whether a message of `msg_words` words encodes coset by coset: one coset fits the cache budget [`COSET_WORDS`].
    pub fn encodes_by_coset(msg_words: usize) -> bool {
        msg_words <= COSET_WORDS
    }

    /// The log of the codeword's rows, for a message of `msg_words` words.
    fn coset_log_d(&self, msg_words: usize, num_ntts: usize, log_inv_rate: usize) -> usize {
        assert!(num_ntts > 0 && msg_words.is_multiple_of(num_ntts));
        let log_d = log2_strict_usize(msg_words / num_ntts) + log_inv_rate;
        assert!(log_d <= self.tw.log_domain_size());
        log_d
    }

    /// Coset `coset` of the codeword, built into `buf` (one message long) from the message.
    fn build_coset(&self, buf: &mut [F64], msg: &[F64], num_ntts: usize, log_d: usize, log_inv_rate: usize, coset: usize) {
        let (first, end) = (log_inv_rate, log_d);
        self.layers().run(buf, Some(msg), log_d, num_ntts, first, end, log_inv_rate, coset, None, no_hook);
    }

    /// RS-encode a message held in a buffer of its own, handing `on_rows` every finished block of rows.
    ///
    /// - The result equals encoding in place.
    /// - Every codeword word is written before it is read, so the codeword may start uninitialized.
    /// - The blocks are as for the in-place encode.
    /// - With `defer_deep`, the deep pass is not written back: each deep sub-block is finished in scratch
    ///   and handed to `on_rows` from there.
    /// - The buffer then keeps what the gathered passes left, and [`Self::finish_rows`] finishes any row again.
    /// - With no separate deep pass in the plan, the buffer is the whole codeword and this returns `None`.
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
        defer_deep: bool,
        on_rows: &RowSink<'_>,
    ) -> Option<Plan> {
        assert_eq!(
            msg.len() << log_inv_rate,
            data.len(),
            "the codeword is 2^log_inv_rate messages"
        );
        let mut win = Window::new(data, num_ntts, 0);
        self.transform(&mut win, log_inv_rate, Source::Rows(msg), Some(on_rows), defer_deep)
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
    /// - The layer-`start` blocks are not in the buffer yet: each is a copy of the message, see [`Source`].
    /// - The first pass reads its rows from the message instead.
    /// - A lane-major message is read by that pass directly.
    ///   A plan with no such pass transposes it into block 0 of the buffer and encodes in place.
    /// - Its `first_coset` and the buffer's size give the layer-`start` blocks the buffer holds.
    /// - Twiddles and rows handed to `on_rows` stay global.
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
    /// - With `defer_deep`, the deep pass's results go only to `on_rows`, not back into the buffer.
    fn transform(
        &self,
        win: &mut Window,
        start: usize,
        mut src: Source<'_>,
        on_rows: Option<&RowSink<'_>>,
        defer_deep: bool,
    ) -> Option<Plan> {
        // The buffer is 2^log_d rows of `num_ntts` words, or the layer-`start` blocks of a lane-major message.
        let (num_ntts, log_d) = (win.num_ntts(), src.log_d(win, start));
        assert!(log_d <= self.tw.log_domain_size());
        assert!(start <= log_d);

        // The plan reads the original message, before a lane-major one is transposed.
        let plan = Plan::new(log_d, num_ntts, start, l2_words(matches!(src, Source::Lanes(_))));

        // Phase 1: gathered passes, each at most one L2 group of layers.
        //
        // Only the first one reads the message.
        // A lane-major message is read by the first gathered pass, a batch of residues at a time.
        // Without such a pass, or with too few residues to batch, it is transposed into block 0 first.
        if matches!(src, Source::Lanes(_)) && Plan::for_lanes(log_d, num_ntts, start).is_none() {
            assert_eq!(win.len() / num_ntts, 1 << log_d, "a partial buffer needs the gathered pass");
            src = src.transpose_into(win, log_d - start);
        }
        for (layer, g) in plan.gathered() {
            GatheredPass::new(&plan, win, layer, g, src.take()).run(self.layers(), win);
        }

        // Phase 2: replicas a deep-only plan cannot build inside its tasks.
        src = src.replicate(win, start, plan.has_deep());

        // A lone sub-block runs on this thread, so its rows go over in parallel afterwards.
        let deep_rows = on_rows.filter(|_| plan.fuses_rows());

        // Phase 3: the deep pass, one task per run of contiguous sub-blocks.
        if plan.has_deep() {
            self.deep_pass(win, &plan, plan.log_sub(), src, deep_rows, defer_deep);
        }

        if let Some(f) = on_rows.filter(|_| !plan.fuses_rows()) {
            let log_rows = plan.log_handover_rows();
            parallel::chunks_mut(win.words_mut(), num_ntts << log_rows, |i, rows| f(i << log_rows, rows));
        }
        (defer_deep && deep_rows.is_some() && matches!(src, Source::Replicas)).then_some(plan)
    }
}

/// Words of one coset a coset-major encode may span.
///
/// # Why this value
///
/// - 2^19 words is 4 MiB: the largest coset it admits is 2^13 rows of 48 words, 3 MiB.
/// - A coset runs in a scratch that lives in L2 and L3, and never reaches DRAM.
/// - Four nodes of two threads each share one 32 MiB L3, so a worker's share is 4 MiB.
/// - A larger coset spills to DRAM, and then the codeword's write pass is the cheaper plan.
const COSET_WORDS: usize = 1 << 19;

/// One cache line of words.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub(super) struct Line([F64; 8]);

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

/// Test oracle: fill every replica with a copy of a message held elsewhere.
#[cfg(test)]
fn replicate_rows(data: &mut [F64], msg: &[F64]) {
    for replica in data.chunks_mut(msg.len()) {
        replica.copy_from_slice(msg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::stage::transpose_lane_major;
    use crate::stack_open::EQUAL_BLOCK_LOG;
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
            ntt.transform(&mut Window::new(&mut c, 1, 0), 0, Source::Replicas, None, false);
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
        //     (7, 17 or 24, 0)        the same, with a vector tail or whole vectors
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
            (7, 17, 0),
            (7, 24, 0),
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
            ntt.transform(&mut Window::new(&mut got, lanes, 0), start_layer, Source::Replicas, None, false);

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
            ntt.encode_interleaved_in_place(&mut got, lanes, log_inv_rate);
            assert_eq!(got, want, "log_d={log_d}, lanes={lanes}, rate={log_inv_rate}");

            // Under test: a separate message, with the rows handed to a sink.
            let mut sep = vec![F64::ZERO; got.len()];
            let mut handed = vec![F64::ZERO; got.len()];
            let hits: Vec<AtomicUsize> = (0..1usize << log_d).map(|_| AtomicUsize::new(0)).collect();
            let write = sink_into(&mut handed, lanes);
            ntt.encode_interleaved_with(&mut sep, &msg, lanes, log_inv_rate, false, &|row, rows| {
                for hit in &hits[row..row + rows.len() / lanes] {
                    hit.fetch_add(1, Ordering::Relaxed);
                }
                write(row, rows)
            });
            drop(write);
            assert_eq!(sep, want, "separate message, log_d={log_d}, lanes={lanes}, rate={log_inv_rate}");
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
                    ntt.transform(&mut Window::new(&mut want, 1, 0), log_inv_rate, Source::Replicas, None, false);
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
    ///
    /// A deep-only plan with a separate message builds the whole codeword and returns `None`.
    /// The in-place, deep-only, deferred branch runs in `lane_major_encode_matches_transpose_then_in_place`,
    /// at shape `(5, 2, 7)`.
    #[test]
    fn deferred_deep_rows_match_the_full_encode() {
        let mut rng = Rng::new(0xDEE9);
        let shapes = [(14usize, 51usize, 1usize, true), (15, 64, 2, true), (9, 8, 1, false), (4, 8, 4, false)];
        for (log_d, lanes, log_inv_rate, split) in shapes {
            let ntt = AdditiveNttF64::standard(log_d);
            let msg_len = (lanes << log_d) >> log_inv_rate;
            let msg: Vec<F64> = (0..msg_len).map(|_| F64(rng.next_u64())).collect();
            let mut want = vec![F64::ZERO; msg_len << log_inv_rate];
            want[..msg_len].copy_from_slice(&msg);
            ntt.encode_interleaved_in_place(&mut want, lanes, log_inv_rate);

            let mut got = vec![F64::ZERO; msg_len << log_inv_rate];
            let mut handed = vec![F64::ZERO; got.len()];
            let sink = sink_into(&mut handed, lanes);
            let deep = ntt.encode_interleaved_with(&mut got, &msg, lanes, log_inv_rate, true, &sink);
            drop(sink);
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
}
