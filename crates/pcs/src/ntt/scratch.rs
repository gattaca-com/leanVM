//! The thread-local scratch every pass transforms in.

use primitives::field::F64;
use std::cell::RefCell;
use std::thread::LocalKey;

/// One cache line of words.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub(super) struct Line([F64; 8]);

thread_local! {
    static SCRATCH: RefCell<Vec<Line>> = const { RefCell::new(Vec::new()) };
    static SINK_SCRATCH: RefCell<Vec<Line>> = const { RefCell::new(Vec::new()) };
}

/// Lend this thread's scratch buffer, grown to the requested length.
///
/// - It is cache-line aligned, so a full-width load never splits a line.
/// - It lives as long as the thread, so no pass allocates in its hot loop.
pub(super) fn with_scratch<R>(len: usize, f: impl FnOnce(&mut [F64]) -> R) -> R {
    lend(&SCRATCH, len, f)
}

/// Lend this thread's second scratch buffer, which a [`super::RowSink`] may use.
///
/// A pass hands rows to its sink from inside [`with_scratch`], so the sink needs a buffer of its own.
pub(crate) fn with_sink_scratch<R>(len: usize, f: impl FnOnce(&mut [F64]) -> R) -> R {
    lend(&SINK_SCRATCH, len, f)
}

fn lend<R>(cell: &'static LocalKey<RefCell<Vec<Line>>>, len: usize, f: impl FnOnce(&mut [F64]) -> R) -> R {
    cell.with_borrow_mut(|lines| {
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
