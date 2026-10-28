//! The deep pass of [`AdditiveNttF64`], and the drain that writes a finished group back late.

use primitives::field::F64;

use super::additive_ntt_f64::{AdditiveNttF64, RowSink, with_scratch};
use super::drain::{Drain, LateDrain};
use super::layers::{Layers, no_hook};
use super::plan::Plan;
use super::source::Source;
use super::window::Window;

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
    pub(super) fn deep_pass(
        &self,
        win: &mut Window,
        plan: &Plan,
        log_sub: usize,
        src: Source<'_>,
        deep_rows: Option<&RowSink<'_>>,
        defer_deep: bool,
    ) {
        let (first_coset, stream) = (win.first_coset(), win.stream());
        let Plan { log_d, num_ntts, start, deep_start, .. } = *plan;
        let layers = self.layers();
        let sub_len = num_ntts << log_sub;
        let outer = log_d - log_sub;
        let (base, total) = (win.base(), win.len());
        let hooks = Layers::hook_calls(log_d, deep_start, log_d, outer);
        match src {
            // A separate message: build each sub-block in scratch, hash it, then write it out once.
            //
            // Streamed out whole, the codeword is written without ever being read.
            Source::Rows(m) => {
                // With whole cache lines in every row, a finished sub-block drains late, as in a gathered pass.
                let late = stream && num_ntts.is_multiple_of(8) && base.0.align_offset(64) == 0;
                let per_hook = (sub_len / 8).div_ceil(hooks);
                let block_len = total >> start;
                parallel::for_each_chunk(1 << outer, |lo, hi| {
                    with_scratch(sub_len << usize::from(late), |both| {
                        // One fence at the end of the task covers all its streaming stores.
                        let mut ld = LateDrain::new(both, sub_len, stream, late, per_hook);
                        for sub_idx in lo..hi {
                            // A task sits at the same offset in every replica, or covers whole replicas.
                            //
                            //     task at offset off of its replica  <-  message words [off, off + sub_len)
                            let off = (sub_idx * sub_len) % block_len;
                            // Inside one replica the first sweep reads the message; across several it is copied in.
                            let from = (sub_len <= block_len).then(|| &m[off..][..sub_len]);
                            if from.is_none() {
                                let copy = |part: &mut [F64]| part.copy_from_slice(&m[..part.len()]);
                                ld.scratch().chunks_mut(block_len).for_each(copy);
                            }
                            if ld.is_late() {
                                // The previous sub-block drains while this one transforms.
                                ld.transform(|scratch, mut hook| {
                                    layers.run(
                                        scratch,
                                        from,
                                        log_d,
                                        num_ntts,
                                        deep_start,
                                        log_d,
                                        outer,
                                        sub_idx,
                                        None,
                                        move || hook.step(),
                                    )
                                });
                            } else {
                                layers.run(
                                    ld.scratch(),
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
                            }
                            if let Some(f) = deep_rows {
                                f(sub_idx << log_sub, ld.scratch());
                            }
                            // A late one drains during the next transform, while the other scratch is filled.
                            ld.send(Drain {
                                // SAFETY: the sub-block lies inside the codeword, and this task alone owns it.
                                dst: unsafe { base.add(sub_idx * sub_len) },
                                row_len: sub_len,
                                rows: 1,
                                ..Drain::default()
                            });
                        }
                        ld.finish();
                    });
                });
            }
            _ => {
                // While a sub-block transforms, the next one is prefetched into L2, spread over its row groups.
                let prefetch_next = |sub_idx: usize| {
                    let next = (sub_idx + 1) * sub_len;
                    prefetch_spread(&base, next, if next < total { sub_len } else { 0 }, hooks)
                };
                parallel::chunks_mut(win.words_mut(), sub_len, |local, sub| {
                    // The buffer's first sub-block, in the whole domain.
                    let sub_idx = local + (first_coset << (log_d - start - log_sub));
                    let hook = prefetch_next(local);
                    if defer_deep && deep_rows.is_some() {
                        // The rows are hashed from scratch and the buffer keeps its pre-deep state.
                        with_scratch(sub_len, |scratch| {
                            let (src, end) = (Some(&*sub), log_d);
                            layers.run(scratch, src, log_d, num_ntts, deep_start, end, outer, sub_idx, None, hook);
                            if let Some(f) = deep_rows {
                                f(sub_idx << log_sub, scratch);
                            }
                        });
                    } else {
                        // The replicas are already in place: run the layers where they are.
                        layers.run(sub, None, log_d, num_ntts, deep_start, log_d, outer, sub_idx, None, hook);
                        if let Some(f) = deep_rows {
                            f(sub_idx << log_sub, sub);
                        }
                    }
                });
            }
        }
    }
}
