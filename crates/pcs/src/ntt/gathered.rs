//! The gathered pass: a few layers in one sweep of the buffer, one row group at a time.

use flux_profiler::timed;
use primitives::field::F64;

use super::additive_ntt_f64::with_scratch;
use super::drain::{Drain, LateDrain};
use super::layers::Layers;
use super::plan::Plan;
use super::source::Source;
use super::stage::{BATCH, Stage, stage_subs};
use super::window::Window;

/// One gathered pass: layers `layer..layer + g` in one sweep of the buffer, one row group at a time.
///
/// - It owns the geometry of the pass: the group, block and task sizes, and how a finished group drains.
/// - It owns its [`Source`] and, for a lane-major message, its [`Stage`].
/// - [`GatheredPass::new`] computes the geometry, and [`GatheredPass::run`] sweeps the buffer once.
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
/// # Tasks
///
/// - A task is one residue over every block the buffer holds, from `first_block` on, block 0 last.
/// - With a message, every block's rows come from the message, which may be the buffer's block 0.
/// - Tasks own disjoint residues, so no task reads rows another task writes.
///
/// # With a lane-major message
///
/// - Word `w` of message row `j` is word `j` of lane block `num_ntts - 1 - w`, and zero below the padding.
/// - A lane block's cache line holds [`BATCH`] consecutive rows, so a task takes that many residues.
/// - It goes one octet of lanes at a time, and the octet of the residues is 64 lanes of one transform.
/// - It transposes those lanes into a stage of `2^g` rows of 64 words, one line per residue.
/// - Each block transforms a copy of the stage and scatters it line by line.
///
/// # Draining a group late
///
/// - A group drains during the next group's transform, as [`Drain`] describes.
/// - Group k+1 never reads what group k writes.
/// - A block's rows are its own, and the message is read before block 0 is written.
pub(super) struct GatheredPass<'a> {
    src: Source<'a>,
    /// The transpose of a lane-major message, if the source is one.
    staging: Option<Stage>,
    layer: usize,
    g: usize,
    num_ntts: usize,
    /// The first block the buffer holds, and the blocks it holds from there.
    first_block: usize,
    blocks: std::ops::Range<usize>,
    /// Rows of a layer-`layer` block, and the rows between the rows of a group.
    block_rows: usize,
    step: usize,
    /// Rows of a group, its words a row, and its words in all.
    rows: usize,
    width: usize,
    group_len: usize,
    /// Residues one task step takes, and the words of its stage.
    batch: usize,
    stage_len: usize,
    /// How a finished group drains: in pieces of `piece` words, `piece_stride` apart, `per_hook` a hook call.
    piece: usize,
    piece_stride: usize,
    per_hook: usize,
    stream: bool,
}

impl<'a> GatheredPass<'a> {
    /// The geometry of the pass over layers `layer..layer + g` of `plan`, reading `src` and writing `win`.
    pub(super) fn new(plan: &Plan, win: &Window, layer: usize, g: usize, src: Source<'a>) -> Self {
        let (log_d, num_ntts) = (plan.log_d, win.num_ntts());
        // A group is 2^g rows, `step` rows apart, of a block of `block_rows` rows.
        let log_step = log_d - layer - g;
        let (rows, step, block_rows) = (1usize << g, 1usize << log_step, 1usize << (log_d - layer));
        // The blocks the buffer holds.
        let first_block = win.first_coset();
        let blocks = first_block..first_block + (win.len() / num_ntts) / block_rows;
        // A lane-major group is 64 lanes of a batch of residues, transposed into a stage: one line per residue in each row.
        let staged = matches!(src, Source::Lanes(_));
        let subs = stage_subs(staged, log_step);
        let batch = if staged { BATCH * subs } else { 1 };
        let width = if staged { BATCH * 8 } else { num_ntts };
        let (group_len, stage_len) = (rows * width, if staged { subs * rows * width } else { 0 });
        let staging = match src {
            Source::Lanes(lanes) => Some(Stage::new(&lanes, log_step, rows, block_rows, num_ntts, subs)),
            _ => None,
        };
        // Whole aligned cache lines in every row stream out a line at a time; other rows are stored whole.
        let stream = win.streams_lines();
        let piece = if staged || stream { 8 } else { num_ntts };
        let piece_stride = if staged { num_ntts } else { piece };
        let per_hook = (group_len / piece).div_ceil(Layers::hook_calls(layer + g, layer, layer + g, layer));
        Self {
            src,
            staging,
            layer,
            g,
            num_ntts,
            first_block,
            blocks,
            block_rows,
            step,
            rows,
            width,
            group_len,
            batch,
            stage_len,
            piece,
            piece_stride,
            per_hook,
            stream,
        }
    }

    /// Sweep the buffer once, one task per range of residues.
    #[timed]
    pub(super) fn run(&self, layers: Layers, win: &mut Window) {
        let base = win.base();
        let win = &*win;
        parallel::for_each_chunk(self.step / self.batch, |lo, hi| {
            with_scratch(self.stage_len + 2 * self.group_len, |all| {
                let (stage, both) = all.split_at_mut(self.stage_len);
                // One fence at the end of the task covers all its streaming stores.
                let mut late = LateDrain::new(both, self.group_len, self.stream, true, self.per_hook);
                for t in lo..hi {
                    match (self.src, &self.staging) {
                        (Source::Lanes(lanes), Some(staging)) => {
                            for octet in 0..self.num_ntts / 8 {
                                staging.fill(stage, lanes.words, t * self.batch, octet);
                                for block in self.blocks.clone().rev() {
                                    for (u, staged) in stage.chunks_exact(self.group_len).enumerate() {
                                        let r = t * self.batch + u * BATCH;
                                        let dst = win.row(self.block_rows, block, r) + 8 * octet;
                                        self.group(layers, &mut late, win, base, block, r, Some(staged), dst);
                                    }
                                }
                            }
                        }
                        _ => self.blocks.clone().rev().for_each(|block| {
                            let dst = win.row(self.block_rows, block, t);
                            self.group(layers, &mut late, win, base, block, t, None, dst)
                        }),
                    }
                }
                late.finish();
            });
        });
    }

    /// Transform residue `r`'s group as layer-`layer` block `block`, and send it back to word `dst` on.
    ///
    /// The rows are `from` the stage, or gathered from the message or the buffer.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn group(
        &self,
        layers: Layers,
        late: &mut LateDrain,
        win: &Window,
        base: parallel::SendPtr<F64>,
        block: usize,
        r: usize,
        from: Option<&[F64]>,
        dst: usize,
    ) {
        let (layer, g, num_ntts, step) = (self.layer, self.g, self.num_ntts, self.step);
        if from.is_none() {
            // Every block copies the message's rows, which the buffer's block 0 may hold, or its own rows.
            let own = win.row(self.block_rows, block, r);
            let first = win.row(self.block_rows, self.first_block, r);
            let src = self.src.group_src(base.0.cast_const(), own, first);
            for (i, dst) in late.scratch().chunks_exact_mut(num_ntts).enumerate() {
                // SAFETY:
                // - The message and the buffer both cover every row addressed here.
                // - Tasks own disjoint residues, so no other task writes these rows.
                // - Block 0 is written last, after every read of the message in it.
                dst.copy_from_slice(unsafe { std::slice::from_raw_parts(src.add(i * step * num_ntts), num_ntts) });
            }
        }
        late.transform(|scratch, mut hook| {
            layers.run(
                scratch,
                from,
                layer + g,
                self.width,
                layer,
                layer + g,
                layer,
                block,
                None,
                move || hook.step(),
            )
        });
        late.send(Drain {
            // SAFETY: the group's first row lies inside the buffer.
            dst: unsafe { base.add(dst) },
            row_len: self.width,
            dst_stride: step * num_ntts,
            piece: self.piece,
            piece_stride: self.piece_stride,
            rows: self.rows,
            ..Drain::default()
        });
    }
}
