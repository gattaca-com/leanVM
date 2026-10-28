//! Codeword rows read back from an encode that did not write its whole codeword.

use flux_profiler::timed;
use primitives::field::F64;

use super::cosets::CosetEncoder;
use super::layers::{Layers, no_hook};
use super::plan::Plan;
use super::scratch::with_scratch;
use super::source::Lanes;
use super::stage::{BATCH, LOG_BATCH, Stage};

/// What an encode kept of its codeword, for the openings to read rows from.
///
/// A codeword kept whole needs no layers: [`gather_rows`] reads it.
pub enum Kept<'a> {
    /// The buffer as the gathered passes left it: the deep layers run again on each row read.
    BeforeDeep { words: &'a [F64], plan: Plan },
    /// No buffer: each row is recomputed from the lane-major message, which `plan` says how to encode.
    Nothing { msg: &'a [F64], repeats: &'a [bool], n_lanes: usize, plan: Plan },
    /// A coset-major encode: each row is recomputed from the message, coset by coset.
    Cosets { msg: &'a [F64], row_words: usize, log_inv_rate: usize },
}

/// The rows at the sorted `positions` of a codeword kept whole, `row_words` words each, back to back.
pub fn gather_rows(words: &[F64], row_words: usize, positions: &[usize]) -> Vec<F64> {
    let mut out = Vec::with_capacity(positions.len() * row_words);
    for &q in positions {
        out.extend_from_slice(&words[q * row_words..][..row_words]);
    }
    out
}

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

/// Codeword rows of an encode that left rows out of its codeword.
///
/// - It owns the [`Layers`] that finish a deep sub-block or rebuild a coset.
/// - [`Openings::rows`] says which rows are read, and [`Kept`] says what the encode left to read them from.
pub struct Openings<'t> {
    layers: Layers<'t>,
}

impl<'t> Openings<'t> {
    pub(super) fn new(layers: Layers<'t>) -> Self {
        Self { layers }
    }

    /// The rows at the sorted `positions`, back to back, each as the words of one codeword row.
    pub fn rows(&self, kept: Kept<'_>, positions: &[usize]) -> Vec<F64> {
        match kept {
            Kept::BeforeDeep { words, plan } => {
                let sub_len = plan.sub_len();
                self.finished_rows(&plan, positions, |_, sub| &words[sub * sub_len..][..sub_len])
            }
            Kept::Nothing { msg, repeats, n_lanes, plan } => {
                let log_sub = plan.log_d - plan.deep_start;
                let mut subs: Vec<usize> = positions.iter().map(|q| q >> log_sub).collect();
                subs.dedup();
                let built = Rebuild::new(&plan, Lanes { words: msg, repeats, n_lanes }, &subs).run(self.layers);
                let sub_len = plan.sub_len();
                self.finished_rows(&plan, positions, |i, _| &built[i * sub_len..][..sub_len])
            }
            Kept::Cosets { msg, row_words, log_inv_rate } => {
                CosetEncoder::new(self.layers, msg.len(), row_words, log_inv_rate).rows(msg, positions)
            }
        }
    }

    /// The rows at `positions` (sorted), back to back, from the sub-blocks before their deep layers.
    ///
    /// `src(i, sub)` is the `i`-th sub-block, number `sub` in the codeword. Each finishes into the thread's
    /// scratch, so only the queried rows are written.
    fn finished_rows<'a>(
        &self,
        plan: &Plan,
        positions: &[usize],
        src: impl Fn(usize, usize) -> &'a [F64] + Sync,
    ) -> Vec<F64> {
        let Plan { log_d, num_ntts, deep_start, .. } = *plan;
        let layers = self.layers;
        built_rows(num_ntts, log_d - deep_start, positions, |i, sub_idx, sub| {
            let src = Some(src(i, sub_idx));
            layers.run(sub, src, log_d, num_ntts, deep_start, log_d, deep_start, sub_idx, None, no_hook);
        })
    }
}

/// The re-run of the gathered pass of an encode that kept no buffer, for the sub-blocks a query reads.
///
/// - The encode ran one gathered pass over its lane-major message: a group is one residue of one coset,
///   and its transform leaves row `i` of the group as row `r` of sub-block `coset << g | i`, `g` layers deep.
/// - So each group transforms again here, skipping the blocks that feed no row of the sub-blocks,
///   and keeps only the rows that fall in one of them.
/// - It owns the [`Stage`] of that pass and, per coset with queried sub-blocks, the rows it keeps.
struct Rebuild<'a> {
    plan: Plan,
    lanes: Lanes<'a>,
    stage: Stage,
    /// Each coset with queried sub-blocks: its index, those sub-blocks' places in `subs`, and their group rows.
    cosets: Vec<(usize, std::ops::Range<usize>, Vec<usize>)>,
    n_subs: usize,
}

impl<'a> Rebuild<'a> {
    /// The rebuild of the deep sub-blocks `subs` (sorted, distinct).
    ///
    /// # Panics
    ///
    /// Panics unless the encode had the plan this reads back: one lane-major gathered pass.
    fn new(plan: &Plan, lanes: Lanes<'a>, subs: &[usize]) -> Self {
        let Plan { log_d, num_ntts, start, deep_start, .. } = *plan;
        let g = deep_start - start;
        let (rows, log_step) = (1usize << g, log_d - deep_start);
        let msg_rows = 1usize << (log_d - start);
        assert!(plan.single_gathered_pass(), "one gathered pass");
        assert!(num_ntts.is_multiple_of(8) && log_step >= LOG_BATCH, "the gathered pass is lane-major");
        assert_eq!(lanes.words.len(), lanes.n_lanes * msg_rows);
        let mut next = 0;
        let cosets = subs
            .chunk_by(|a, b| a >> g == b >> g)
            .map(|run| {
                next += run.len();
                (run[0] >> g, next - run.len()..next, run.iter().map(|sub| sub & (rows - 1)).collect())
            })
            .collect();
        let stage = Stage::new(&lanes, log_step, rows, msg_rows, num_ntts, 1);
        Self { plan: *plan, lanes, stage, cosets, n_subs: subs.len() }
    }

    /// The sub-blocks back to back, before their deep layers.
    #[timed]
    fn run(&self, layers: Layers) -> Vec<F64> {
        let Plan { log_d, num_ntts, start, deep_start, .. } = self.plan;
        let (rows, log_step) = (1usize << (deep_start - start), log_d - deep_start);
        let sub_len = self.plan.sub_len();
        // A group writes row `r` of every queried sub-block, and the groups cover every residue `r`.
        primitives::filled_vec(self.n_subs * sub_len, |built| {
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
                            self.stage.fill(stage, self.lanes.words, t * BATCH, octet);
                            for (coset, run, live) in &self.cosets {
                                let live_rows = Some(&live[..]);
                                layers.run(
                                    scratch,
                                    Some(stage),
                                    deep_start,
                                    BATCH * 8,
                                    start,
                                    deep_start,
                                    start,
                                    *coset,
                                    live_rows,
                                    no_hook,
                                );
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
