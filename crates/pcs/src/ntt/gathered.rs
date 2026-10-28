//! The gathered pass of [`AdditiveNttF64`]: a few layers in one sweep of the buffer, one row group at a time.

use flux_profiler::timed;
use primitives::field::F64;

use super::additive_ntt_f64::{AdditiveNttF64, with_scratch};
use super::drain::{Drain, LateDrain};
use super::layers::Layers;
use super::source::Source;
use super::stage::{BATCH, Stage, stage_subs};
use super::window::Window;

impl AdditiveNttF64 {
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
    /// - Group k+1 never reads what group k writes: a block's rows are its own, and the message is read before block 0 is written.
    #[timed]
    pub(super) fn gathered_pass(&self, win: &mut Window, log_d: usize, layer: usize, g: usize, src: Source<'_>) {
        let (num_ntts, first_block, stream) = (win.num_ntts(), win.first_coset(), win.stream());
        // A group is 2^g rows, `step` rows apart, of a block of `block_rows` rows.
        let log_step = log_d - layer - g;
        let (rows, step, block_rows) = (1usize << g, 1usize << log_step, 1usize << (log_d - layer));
        // The blocks the buffer holds, and the first word of row `r` of one of them in the buffer.
        let blocks = first_block..first_block + (win.len() / num_ntts) / block_rows;
        let base = win.base();
        let win = &*win;
        let row = |block: usize, r: usize| win.row(block_rows, block, r);
        let layers = self.layers();
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
        let stream = stream && num_ntts.is_multiple_of(8) && base.0.align_offset(64) == 0;
        let piece = if staged || stream { 8 } else { num_ntts };
        let piece_stride = if staged { num_ntts } else { piece };
        let per_hook = (group_len / piece).div_ceil(Layers::hook_calls(layer + g, layer, layer + g, layer));
        parallel::for_each_chunk(step / batch, |lo, hi| {
            with_scratch(stage_len + 2 * group_len, |all| {
                let (stage, both) = all.split_at_mut(stage_len);
                let mut late = LateDrain::new(both, group_len, stream, true, per_hook);
                // Transform residue `r`'s group as layer-`layer` block `block`, and send it back to word `dst` on.
                // The rows are `from` the stage, or gathered from the message or the buffer.
                let mut group = |block: usize, r: usize, from: Option<&[F64]>, dst: usize| {
                    if from.is_none() {
                        // Every block copies the message's rows, which the buffer's block 0 may hold, or its own rows.
                        let src = src.group_src(base.0.cast_const(), row(block, r), row(first_block, r));
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
                            width,
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
                        row_len: width,
                        dst_stride: step * num_ntts,
                        piece,
                        piece_stride,
                        rows,
                        ..Drain::default()
                    });
                };
                for t in lo..hi {
                    match (src, &staging) {
                        (Source::Lanes(lanes), Some(staging)) => {
                            for octet in 0..num_ntts / 8 {
                                staging.fill(stage, lanes.words, t * batch, octet);
                                for block in blocks.clone().rev() {
                                    for (u, staged) in stage.chunks_exact(group_len).enumerate() {
                                        let r = t * batch + u * BATCH;
                                        group(block, r, Some(staged), row(block, r) + 8 * octet);
                                    }
                                }
                            }
                        }
                        _ => blocks.clone().rev().for_each(|block| group(block, t, None, row(block, t))),
                    }
                }
                late.finish();
            });
        });
    }
}
