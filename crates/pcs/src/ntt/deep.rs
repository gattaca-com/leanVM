//! The deep pass of [`AdditiveNttF64`], and the drain that writes a finished group back late.

use primitives::field::F64;
use primitives::stream::Stream;

use super::additive_ntt_f64::{AdditiveNttF64, Line, Msg, RowSink, no_hook, radix_layers, with_scratch};
use super::rows::DeepRows;

/// How many times [`AdditiveNttF64::run_layers_to`] calls its hook over `first_layer..end_layer`.
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
///
/// # Draining late
///
/// - A burst of streaming stores after each group saturates the write-combining buffers and overlaps nothing.
/// - So a task keeps two scratches, and the previous group drains during this group's transform.
/// - The sweeps call a hook once per row group, and each call moves an equal share of the previous group.
/// - The shares round up, so the group is out by the last call, and the task's last group drains at its end.
/// - Group k+2 gathers into the scratch group k drained, and that drain ended with group k+1's transform.
/// - Only whole aligned cache lines stream; other rows take ordinary stores, whole, on the same schedule.
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

impl AdditiveNttF64 {
    /// The deep layers of `plan` over sub-blocks of `2^log_sub` rows, one task per run of contiguous sub-blocks.
    ///
    /// - With a separate message, each sub-block is built in scratch and written out once.
    /// - Otherwise the buffer holds the replicas, and the layers run in place.
    /// - With `defer_deep`, an in-place sub-block is finished in scratch and the buffer keeps its pre-deep state.
    /// - Each finished sub-block goes to `deep_rows` while it is in L2.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn deep_pass(
        &self,
        data: &mut [F64],
        plan: &DeepRows,
        log_sub: usize,
        msg: Msg<'_>,
        first_coset: usize,
        deep_rows: Option<&RowSink<'_>>,
        defer_deep: bool,
        stream: bool,
    ) {
        let DeepRows { log_d, num_ntts, start, deep_start } = *plan;
        let sub_len = num_ntts << log_sub;
        let outer = log_d - log_sub;
        let (base, total) = (parallel::SendPtr(data.as_mut_ptr()), data.len());
        let hooks = hook_calls(log_d, deep_start, log_d, outer);
        match msg {
            // A separate message: build each sub-block in scratch, hash it, then write it out once.
            //
            // Streamed out whole, the codeword is written without ever being read.
            Msg::Rows(m) => {
                // With whole cache lines in every row, a finished sub-block drains late, as in a gathered pass.
                let late = stream && num_ntts.is_multiple_of(8) && base.0.align_offset(64) == 0;
                let per_hook = (sub_len / 8).div_ceil(hooks);
                let block_len = total >> start;
                parallel::for_each_chunk(1 << outer, |lo, hi| {
                    with_scratch(sub_len << usize::from(late), |both| {
                        // One fence at the end of the task covers all its streaming stores.
                        let stream = stream.then(Stream::new);
                        let (mut scratch, mut idle) = both.split_at_mut(sub_len);
                        let mut drain = Drain::default();
                        for sub_idx in lo..hi {
                            // A task sits at the same offset in every replica, or covers whole replicas.
                            //
                            //     task at offset off of its replica  <-  message words [off, off + sub_len)
                            let off = (sub_idx * sub_len) % block_len;
                            // Inside one replica the first sweep reads the message; across several it is copied in.
                            let from = (sub_len <= block_len).then(|| &m[off..][..sub_len]);
                            if from.is_none() {
                                scratch.chunks_mut(block_len).for_each(|part| part.copy_from_slice(&m[..part.len()]));
                            }
                            if late {
                                // The previous sub-block drains while this one transforms.
                                let hook = || drain.step(stream.as_ref(), per_hook);
                                self.run_layers_to(
                                    scratch, from, log_d, num_ntts, deep_start, log_d, outer, sub_idx, None, hook,
                                );
                                drain.step(stream.as_ref(), usize::MAX);
                            } else {
                                self.run_layers_to(
                                    scratch, from, log_d, num_ntts, deep_start, log_d, outer, sub_idx, None, no_hook,
                                );
                            }
                            if let Some(f) = deep_rows {
                                f(sub_idx << log_sub, scratch);
                            }
                            match &stream {
                                Some(_) if late => {
                                    // This one drains during the next transform, while the other scratch is filled.
                                    drain = Drain {
                                        // SAFETY: the sub-block lies inside the codeword.
                                        dst: unsafe { base.add(sub_idx * sub_len) },
                                        src: scratch.as_ptr(),
                                        row_len: sub_len,
                                        rows: 1,
                                        ..Drain::default()
                                    };
                                    std::mem::swap(&mut scratch, &mut idle);
                                }
                                other => {
                                    // SAFETY: this task alone owns the sub-block of the codeword.
                                    let dst = unsafe { base.slice(sub_idx * sub_len, sub_len) };
                                    match other {
                                        Some(s) => s.copy(dst, scratch),
                                        None => dst.copy_from_slice(scratch),
                                    }
                                }
                            }
                        }
                        drain.step(stream.as_ref(), usize::MAX);
                    });
                });
            }
            _ => {
                // While a sub-block transforms, the next one is prefetched into L2, spread over its row groups.
                let prefetch_next = |sub_idx: usize| {
                    let next = (sub_idx + 1) * sub_len;
                    prefetch_spread(&base, next, if next < total { sub_len } else { 0 }, hooks)
                };
                parallel::chunks_mut(data, sub_len, |local, sub| {
                    // The buffer's first sub-block, in the whole domain.
                    let sub_idx = local + (first_coset << (log_d - start - log_sub));
                    let hook = prefetch_next(local);
                    if defer_deep && deep_rows.is_some() {
                        // The rows are hashed from scratch and the buffer keeps its pre-deep state.
                        with_scratch(sub_len, |scratch| {
                            let (src, end) = (Some(&*sub), log_d);
                            self.run_layers_to(scratch, src, log_d, num_ntts, deep_start, end, outer, sub_idx, None, hook);
                            if let Some(f) = deep_rows {
                                f(sub_idx << log_sub, scratch);
                            }
                        });
                    } else {
                        // The replicas are already in place: run the layers where they are.
                        self.run_layers_to(sub, None, log_d, num_ntts, deep_start, log_d, outer, sub_idx, None, hook);
                        if let Some(f) = deep_rows {
                            f(sub_idx << log_sub, sub);
                        }
                    }
                });
            }
        }
    }
}
