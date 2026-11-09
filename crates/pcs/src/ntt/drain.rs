//! Write-back of a finished row group, overlapped with the transform of the next one.
//!
//! - A burst of streaming stores after each group saturates the write-combining buffers and overlaps nothing.
//! - So a task keeps two scratches, and the previous group drains during this group's transform.
//! - The sweeps call a hook once per row group, and each call moves an equal share of the previous group.
//! - The shares round up, so the group is out by the last call, and the task's last group drains at its end.
//! - Group k+2 is built in the scratch group k drained, and that drain ended with group k+1's transform.

use primitives::field::F64;
use primitives::stream::Stream;

use super::scratch::Line;

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
    fn step(&mut self, stream: Option<&Stream>, pieces: usize) {
        let per_row = self.row_len / self.piece;
        for _ in 0..pieces.min((self.rows - self.row) * per_row - self.k) {
            // SAFETY:
            // - The cursor is inside the group, so both pieces are in bounds, and line-aligned when streamed.
            // - Only this group's owner writes these rows of the buffer.
            // - The scratch is not written again until its group has drained.
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

/// The two scratches of a task, and the drain that moves the finished one out during the next transform.
///
/// - It owns the task's [`Stream`], so one fence at the end of the task covers all its streaming stores.
/// - Build it inside the task's scratch lend, and call [`LateDrain::finish`] at the end of the task.
/// - With `late` unset it has no idle scratch: [`LateDrain::send`] copies the scratch out at once.
pub(super) struct LateDrain<'s> {
    scratch: &'s mut [F64],
    idle: &'s mut [F64],
    drain: Drain,
    stream: Option<Stream>,
    per_hook: usize,
    late: bool,
}

/// The hook of a sweep that drains: each call moves one share of the previous group.
///
/// A plain struct, so the sweeps get a static closure and never a `dyn`.
pub(super) struct Hook<'a> {
    drain: &'a mut Drain,
    stream: Option<&'a Stream>,
    per_hook: usize,
}

impl Hook<'_> {
    #[inline(always)]
    pub(super) fn step(&mut self) {
        self.drain.step(self.stream, self.per_hook);
    }
}

impl<'s> LateDrain<'s> {
    /// A task's scratches: the first `len` words of `both` are the scratch, and the rest the idle one.
    ///
    /// - With `stream`, the finished groups leave as streaming stores.
    /// - `per_hook` is the pieces one hook call moves.
    pub(super) fn new(both: &'s mut [F64], len: usize, stream: bool, late: bool, per_hook: usize) -> Self {
        let (scratch, idle) = both.split_at_mut(len);
        Self {
            scratch,
            idle,
            drain: Drain::default(),
            stream: stream.then(Stream::new),
            per_hook,
            late,
        }
    }

    /// The scratch the next group is built in.
    #[inline(always)]
    pub(super) fn scratch(&mut self) -> &mut [F64] {
        self.scratch
    }

    /// Run `f` on the scratch while the previous group drains from the hook.
    ///
    /// Whatever the hook left of that drain is moved out when `f` returns.
    #[inline(always)]
    pub(super) fn transform(&mut self, f: impl FnOnce(&mut [F64], Hook<'_>)) {
        let hook = Hook {
            drain: &mut self.drain,
            stream: self.stream.as_ref(),
            per_hook: self.per_hook,
        };
        f(self.scratch, hook);
        self.drain.step(self.stream.as_ref(), usize::MAX);
    }

    /// Hand over the finished scratch. `drain` says where it goes, and its `src` is filled in here.
    ///
    /// - Late: the group drains during the next transform, and the idle scratch takes its place.
    /// - Otherwise: the scratch is copied to the one contiguous row at `drain.dst` now.
    #[inline(always)]
    pub(super) fn send(&mut self, mut drain: Drain) {
        if self.late {
            drain.src = self.scratch.as_ptr();
            self.drain = drain;
            std::mem::swap(&mut self.scratch, &mut self.idle);
        } else {
            // SAFETY: the caller owns the one row of `row_len` words at `dst`, which lies inside the buffer.
            let dst = unsafe { std::slice::from_raw_parts_mut(drain.dst, drain.row_len) };
            match &self.stream {
                Some(s) => s.copy(dst, self.scratch),
                None => dst.copy_from_slice(self.scratch),
            }
        }
    }

    /// Move out what is left of the last group, at the end of the task.
    pub(super) fn finish(&mut self) {
        self.drain.step(self.stream.as_ref(), usize::MAX);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::scratch::with_scratch;

    #[test]
    fn drains_preserve_gaps_and_guards() {
        #[repr(align(64))]
        struct Buffer([F64; 1024]);
        let sentinel = F64(0xDEAD_BEEF);
        for (late, stream, rows, row_len, piece, piece_stride, dst_stride) in [
            (false, false, 1usize, 13usize, 13usize, 13usize, 13usize),
            (false, true, 1, 24, 8, 8, 24),
            (true, false, 3, 21, 7, 11, 40),
            (true, true, 3, 24, 8, 16, 64),
        ] {
            for calls in [0, 1, 7, 40] {
                let mut output = Buffer([sentinel; 1024]);
                let mut expected = output.0;
                let len = rows * row_len;
                let span = (rows - 1) * dst_stride + (row_len / piece - 1) * piece_stride + piece;
                let stride = span.next_multiple_of(8) + 8;
                with_scratch(len << usize::from(late), |both| {
                    let mut drain = LateDrain::new(both, len, stream, late, 2);
                    for group in 0..3 {
                        let value = |i| F64((group * len + i + 1) as u64);
                        drain.transform(|scratch, mut hook| {
                            for (i, word) in scratch.iter_mut().enumerate() {
                                *word = value(i);
                            }
                            for _ in 0..calls {
                                hook.step();
                            }
                        });
                        let start = 8 + group * stride;
                        assert!(start + span <= output.0.len() - 8);
                        for row in 0..rows {
                            for k in 0..row_len / piece {
                                for word in 0..piece {
                                    expected[start + row * dst_stride + k * piece_stride + word] =
                                        value(row * row_len + k * piece + word);
                                }
                            }
                        }
                        drain.send(Drain {
                            dst: output.0.as_mut_ptr().wrapping_add(start),
                            row_len,
                            dst_stride,
                            piece,
                            piece_stride,
                            rows,
                            ..Drain::default()
                        });
                    }
                    drain.finish();
                    drain.finish();
                });
                assert_eq!(output.0, expected, "late={late}, stream={stream}, calls={calls}");
            }
        }
    }
}
