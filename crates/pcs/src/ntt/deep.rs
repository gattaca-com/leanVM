//! The deep pass: the near layers, which hand their finished rows to the hasher while they are in L2.

use primitives::field::F64;

use super::drain::{Drain, LateDrain};
use super::encode::RowSink;
use super::layers::hook_calls;
use super::plan::Plan;
use super::scratch::with_scratch;
use super::source::Source;
use super::twiddles::AdditiveNttF64;
use super::window::Window;

/// The deep pass: layers `deep_start..log_d`, one task of `2^log_task` rows at a time, a run of sub-blocks.
///
/// - It owns the sub-block geometry, and the sink that takes each finished sub-block.
/// - With a separate message, each sub-block is built in scratch and written out once.
/// - Otherwise the buffer holds the replicas, and the layers run in place.
/// - With `defer`, an in-place sub-block is finished in scratch and the buffer keeps its pre-deep state.
/// - Each finished sub-block goes to the sink while it is in L2.
pub(super) struct DeepPass<'s> {
    plan: Plan,
    /// Log of the rows of a task, its words, and the log of the tasks the domain holds.
    log_task: usize,
    task_len: usize,
    outer: usize,
    /// Hook calls of one task's layers.
    hooks: usize,
    defer: bool,
    sink: Option<&'s RowSink<'s>>,
}

impl<'s> DeepPass<'s> {
    pub(super) fn new(plan: &Plan, sink: Option<&'s RowSink<'s>>, defer: bool) -> Self {
        let log_task = plan.log_task();
        let outer = plan.log_d - log_task;
        Self {
            plan: *plan,
            log_task,
            task_len: plan.num_ntts << log_task,
            outer,
            hooks: hook_calls(plan.log_d, plan.deep_start, outer),
            defer,
            sink,
        }
    }

    pub(super) fn run(&self, ntt: &AdditiveNttF64, win: &mut Window, src: Source<'_>) {
        match src {
            Source::Rows(msg) => self.with_message(ntt, win, msg),
            _ => self.in_place(ntt, win),
        }
    }

    /// A separate message: build each task in scratch, hash it, then write it out once.
    ///
    /// - Streamed out whole, the codeword is written without ever being read.
    /// - With whole cache lines in every row, a task drains late, as a gathered group does.
    fn with_message(&self, ntt: &AdditiveNttF64, win: &mut Window, msg: &[F64]) {
        let Plan {
            log_d,
            num_ntts,
            start,
            deep_start,
            ..
        } = self.plan;
        let (task_len, outer, log_task, sink) = (self.task_len, self.outer, self.log_task, self.sink);
        let (base, total, stream, late) = (win.base(), win.len(), win.stream(), win.streams_lines());
        let per_hook = (task_len / 8).div_ceil(self.hooks);
        let block_len = total >> start;
        parallel::for_each_chunk(1 << outer, |lo, hi| {
            with_scratch(task_len << usize::from(late), |both| {
                let mut ld = LateDrain::new(both, task_len, stream, late, per_hook);
                for task in lo..hi {
                    // A task sits at the same offset in every replica, or covers whole replicas.
                    //
                    //     task at offset off of its replica  <-  message words [off, off + task_len)
                    let off = (task * task_len) % block_len;
                    // Inside one replica the first sweep reads the message; across several it is copied in.
                    let from = (task_len <= block_len).then(|| &msg[off..][..task_len]);
                    if from.is_none() {
                        let copy = |part: &mut [F64]| part.copy_from_slice(&msg[..part.len()]);
                        ld.scratch().chunks_mut(block_len).for_each(copy);
                    }
                    // The previous task drains while this one transforms, if it drains late.
                    ld.transform(|scratch, mut hook| {
                        ntt.run_layers(
                            scratch,
                            from,
                            log_d,
                            num_ntts,
                            deep_start,
                            outer,
                            task,
                            None,
                            move || hook.step(),
                        )
                    });
                    if let Some(f) = sink {
                        f(task << log_task, ld.scratch());
                    }
                    ld.send(Drain {
                        // SAFETY: the task's rows lie inside the codeword, and this task alone owns them.
                        dst: unsafe { base.add(task * task_len) },
                        row_len: task_len,
                        rows: 1,
                        ..Drain::default()
                    });
                }
                ld.finish();
            });
        });
    }

    /// The buffer holds the replicas: run the layers where they are, or in scratch when deferred.
    ///
    /// While a task transforms, the next one is prefetched into L2, spread over its row groups.
    fn in_place(&self, ntt: &AdditiveNttF64, win: &mut Window) {
        let Plan {
            log_d,
            num_ntts,
            deep_start,
            ..
        } = self.plan;
        let (task_len, outer, log_task, sink) = (self.task_len, self.outer, self.log_task, self.sink);
        let first_task = win.first_row() >> log_task;
        let (base, total) = (win.base(), win.len());
        parallel::chunks_mut(win.words_mut(), task_len, |local, words| {
            let task = first_task + local;
            let next = (local + 1) * task_len;
            let hook = prefetch_spread(base, next, if next < total { task_len } else { 0 }, self.hooks);
            if self.defer {
                // The rows are hashed from scratch and the buffer keeps its pre-deep state.
                with_scratch(task_len, |scratch| {
                    ntt.run_layers(
                        scratch,
                        Some(&*words),
                        log_d,
                        num_ntts,
                        deep_start,
                        outer,
                        task,
                        None,
                        hook,
                    );
                    if let Some(f) = sink {
                        f(task << log_task, scratch);
                    }
                });
            } else {
                // The replicas are already in place: run the layers where they are.
                ntt.run_layers(words, None, log_d, num_ntts, deep_start, outer, task, None, hook);
                if let Some(f) = sink {
                    f(task << log_task, words);
                }
            }
        });
    }
}

/// A hook that prefetches `words` words at `base + start` into L2, evenly over `calls` calls.
///
/// - A task that fills the L2 cannot be prefetched whole before its predecessor is done.
/// - Spread over the predecessor's sweeps, the lines arrive while it is still working.
fn prefetch_spread(base: parallel::SendPtr<F64>, start: usize, words: usize, calls: usize) -> impl FnMut() {
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
        #[cfg(not(target_arch = "x86_64"))]
        let _ = first;
        done = end;
    }
}
