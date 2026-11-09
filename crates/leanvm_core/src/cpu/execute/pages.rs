//! Cell storage with a lazily faulted Linux mapping and ordinary pages elsewhere.

use super::{Cell, PAGE, PAGE_LOG};

pub(super) struct Pages {
    prefix: Option<Mapping>,
    tail: Vec<Box<[Cell]>>,
    len: usize,
}

impl Pages {
    pub(super) fn new() -> Self {
        Self::with_prefix(Mapping::reserve(super::MAX_CELLS as usize))
    }

    fn with_prefix(prefix: Option<Mapping>) -> Self {
        Self {
            prefix,
            tail: Vec::new(),
            len: 0,
        }
    }

    fn prefix_pages(&self) -> usize {
        self.prefix.as_ref().map_or(0, |m| m.len >> PAGE_LOG)
    }

    pub(super) fn grow_to(&mut self, page: usize) {
        let len = page + 1;
        let tail_len = len.saturating_sub(self.prefix_pages());
        while self.tail.len() < tail_len {
            // SAFETY: zero is a valid Cell: zero field limbs/count and State::Unwritten.
            self.tail
                .push(unsafe { Box::<[Cell]>::new_zeroed_slice(PAGE).assume_init() });
        }
        self.len = self.len.max(len);
    }

    pub(super) fn get(&self, page: usize) -> Option<&[Cell]> {
        if page >= self.len {
            return None;
        }
        let prefix_pages = self.prefix_pages();
        if page < prefix_pages {
            Some(&self.prefix.as_ref().unwrap().cells()[page * PAGE..][..PAGE])
        } else {
            Some(&self.tail[page - prefix_pages])
        }
    }

    pub(super) fn cell_mut(&mut self, cell: usize) -> &mut Cell {
        assert!(cell >> PAGE_LOG < self.len, "cell must be covered before access");
        let prefix_pages = self.prefix_pages();
        if cell >> PAGE_LOG < prefix_pages {
            &mut self.prefix.as_mut().unwrap().cells_mut()[cell]
        } else {
            &mut self.tail[(cell >> PAGE_LOG) - prefix_pages][cell & (PAGE - 1)]
        }
    }

    pub(super) fn prefetch(&self, cell: usize) {
        #[cfg(target_arch = "x86_64")]
        if let Some(cell) = self.prefix.as_ref().and_then(|m| m.cells().get(cell)) {
            // SAFETY: x86-64 has SSE; this is an in-bounds cell of the live mapping, only prefetched.
            unsafe {
                core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_ET0 }>(
                    std::ptr::from_ref(cell).cast(),
                )
            };
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = cell;
    }
}

struct Mapping {
    ptr: std::ptr::NonNull<Cell>,
    len: usize,
}

// SAFETY: the mapping is uniquely owned, and references to its Cells follow normal shared/exclusive borrows.
unsafe impl Send for Mapping {}
// SAFETY: shared access only exposes shared Cell slices; Cell has no interior mutability.
unsafe impl Sync for Mapping {}

impl Mapping {
    fn reserve(len: usize) -> Option<Self> {
        #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
        {
            let layout = std::alloc::Layout::array::<Cell>(len).ok()?;
            if len == 0 || !len.is_multiple_of(PAGE) {
                return None;
            }
            // SAFETY: anonymous private memory, with a nonzero size bounded by isize::MAX through Layout.
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    layout.size(),
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                    -1,
                    0,
                )
            };
            if p == libc::MAP_FAILED {
                return None;
            }
            if p.is_null() || !(p as usize).is_multiple_of(layout.align()) {
                // SAFETY: this is the successful mapping above, not yet exposed or released.
                unsafe { libc::munmap(p, layout.size()) };
                return None;
            }
            // SAFETY: advice covers the owned mapping; failure only leaves ordinary pages.
            unsafe { libc::madvise(p, layout.size(), libc::MADV_HUGEPAGE) };
            Some(Self {
                ptr: std::ptr::NonNull::new(p.cast()).unwrap(),
                len,
            })
        }
        #[cfg(not(all(target_os = "linux", target_pointer_width = "64")))]
        {
            let _ = len;
            None
        }
    }

    fn cells(&self) -> &[Cell] {
        // SAFETY: reserve checks alignment, non-null and slice size. Anonymous pages are zeroed, a valid
        // Cell representation; later writes are Cells. The mapping lives for the returned borrow.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    fn cells_mut(&mut self) -> &mut [Cell] {
        // SAFETY: as in cells(), with exclusive access to the uniquely owned mapping.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
        // SAFETY: the same uniquely owned mapping made by reserve, released once after all borrows end.
        unsafe {
            libc::munmap(self.ptr.as_ptr().cast(), self.len * size_of::<Cell>())
        };
    }
}

#[cfg(test)]
mod tests {
    use super::super::State;
    use super::*;
    use primitives::field::F128;

    fn exercise(mut pages: Pages) {
        for i in [0, 2 * PAGE - 1, 2 * PAGE, usize::MAX] {
            pages.prefetch(i);
        }
        assert!(pages.get(0).is_none());
        for i in [0, PAGE - 1, PAGE, 2 * PAGE - 1, 2 * PAGE, 3 * PAGE + 7] {
            pages.grow_to(i >> PAGE_LOG);
            assert!(pages.get(pages.len).is_none());
            let cell = pages.cell_mut(i);
            assert_eq!(cell.word, F128::ZERO);
            assert_eq!(cell.count, 0);
            assert!(cell.state == State::Unwritten);
            cell.word = F128::new(i as u64 + 1, 0);
            cell.count = i as u32 + 1;
            cell.state = State::Written;
        }
        for i in 0..4 * PAGE {
            let cell = &pages.get(i >> PAGE_LOG).unwrap()[i & (PAGE - 1)];
            let written = [0, PAGE - 1, PAGE, 2 * PAGE - 1, 2 * PAGE, 3 * PAGE + 7].contains(&i);
            assert_eq!(
                cell.word,
                if written {
                    F128::new(i as u64 + 1, 0)
                } else {
                    F128::ZERO
                }
            );
            assert_eq!(cell.count, if written { i as u32 + 1 } else { 0 });
            assert!(cell.state == if written { State::Written } else { State::Unwritten });
        }
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pages.cell_mut(4 * PAGE);
            }))
            .is_err()
        );
    }

    #[test]
    fn ordinary_pages_preserve_bounds_and_zeros() {
        exercise(Pages::with_prefix(None));
    }

    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    #[test]
    fn mapped_prefix_and_paged_tail_preserve_bounds_and_zeros() {
        assert!(Mapping::reserve(0).is_none());
        assert!(Mapping::reserve(PAGE - 1).is_none());
        assert!(Mapping::reserve(usize::MAX).is_none());
        exercise(Pages::with_prefix(Some(
            Mapping::reserve(2 * PAGE).expect("small cell mapping"),
        )));
    }
}
