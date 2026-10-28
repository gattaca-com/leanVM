//! Codeword rows read back from an encode that did not write its whole codeword.

use flux_profiler::timed;
use primitives::field::F64;

use super::additive_ntt_f64::{AdditiveNttF64, no_hook, with_scratch};
use super::plan::Plan;
use super::stage::{BATCH, LOG_BATCH, fill_stage, stage_copies};

/// The rows at the sorted `positions`, `n` words each, back to back.
///
/// - The rows group by the block of `2^log_block` rows that holds them, one pool task per block.
/// - `build(i, block, scratch)` writes block number `block`, the `i`-th one touched, into the thread's scratch.
/// - Only the queried rows are copied out of it.
pub(super) fn built_rows(
    n: usize,
    log_block: usize,
    positions: &[usize],
    build: impl Fn(usize, usize, &mut [F64]) + Sync,
) -> Vec<F64> {
    debug_assert!(positions.is_sorted());
    let mut out = vec![F64::ZERO; positions.len() * n];
    let mut rest = &mut out[..];
    let mut parts = positions
        .chunk_by(|a, b| a >> log_block == b >> log_block)
        .map(|run| {
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(run.len() * n);
            rest = tail;
            (head, run)
        })
        .collect::<Vec<_>>();
    parallel::chunks_mut(&mut parts, 1, |i, part| {
        let (dst, run) = &mut part[0];
        let block = run[0] >> log_block;
        with_scratch(n << log_block, |scratch| {
            build(i, block, scratch);
            for (dst, &q) in dst.chunks_exact_mut(n).zip(run.iter()) {
                dst.copy_from_slice(&scratch[(q & ((1 << log_block) - 1)) * n..][..n]);
            }
        });
    });
    out
}

impl AdditiveNttF64 {
    /// Codeword rows `positions` (sorted) of a buffer left by [`Self::encode_interleaved_with`] with `defer_deep`,
    /// back to back.
    pub fn finish_rows(&self, data: &[F64], deep: &Plan, positions: &[usize]) -> Vec<F64> {
        let sub_len = deep.sub_len();
        self.finished_rows(deep, positions, |_, sub| &data[sub * sub_len..][..sub_len])
    }

    /// Codeword rows `positions` (sorted) of an encode that kept no buffer, recomputed from its lane-major message,
    /// back to back.
    pub fn rebuild_rows(
        &self,
        msg: &[F64],
        repeats: &[bool],
        n_lanes: usize,
        deep: &Plan,
        positions: &[usize],
    ) -> Vec<F64> {
        let log_sub = deep.log_d - deep.deep_start;
        let mut subs: Vec<usize> = positions.iter().map(|q| q >> log_sub).collect();
        subs.dedup();
        let built = self.rebuild_sub_blocks(msg, repeats, n_lanes, deep, &subs);
        let sub_len = deep.sub_len();
        self.finished_rows(deep, positions, |i, _| &built[i * sub_len..][..sub_len])
    }

    /// The rows at `positions` (sorted), back to back, from the sub-blocks before their deep layers.
    ///
    /// `src(i, sub)` is the `i`-th sub-block, number `sub` in the codeword. Each finishes into the thread's
    /// scratch, so only the queried rows are written.
    fn finished_rows<'a>(
        &self,
        deep: &Plan,
        positions: &[usize],
        src: impl Fn(usize, usize) -> &'a [F64] + Sync,
    ) -> Vec<F64> {
        let Plan { log_d, num_ntts, deep_start, .. } = *deep;
        built_rows(num_ntts, log_d - deep_start, positions, |i, sub_idx, sub| {
            let src = Some(src(i, sub_idx));
            self.run_layers_to(sub, src, log_d, num_ntts, deep_start, log_d, deep_start, sub_idx, None, no_hook);
        })
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
        deep: &Plan,
        subs: &[usize],
    ) -> Vec<F64> {
        let Plan { log_d, num_ntts, start, deep_start, .. } = *deep;
        let g = deep_start - start;
        let (rows, log_step) = (1usize << g, log_d - deep_start);
        let (sub_len, msg_rows) = (deep.sub_len(), 1usize << (log_d - start));
        assert!(deep.single_gathered_pass(), "one gathered pass");
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
}
