//! Zeroed values in a Linux anonymous mapping of their own: the bus leaves the GKR recomputes.

use std::ops::{Deref, DerefMut};

/// Plain values, zero until written, in a Linux anonymous mapping of their own.
pub struct Mapping<T> {
    ptr: std::ptr::NonNull<T>,
    len: usize,
    #[cfg(target_os = "linux")]
    bytes: usize,
}

// SAFETY: the mapping is owned like a `Vec<T>`, and shared access only reads.
unsafe impl<T: Send> Send for Mapping<T> {}
// SAFETY: as for `Send`.
unsafe impl<T: Sync> Sync for Mapping<T> {}

impl<T> Mapping<T> {
    /// `len` zeroed values, or `None` where there is no mapping to make.
    ///
    /// The mapping asks for ordinary pages, so that scattered writes fault in only the pages they touch.
    ///
    /// # Safety
    ///
    /// All-zero bytes must be a valid `T`.
    pub(crate) unsafe fn new(len: usize) -> Option<Self> {
        let bytes = std::alloc::Layout::array::<T>(len)
            .expect("mapping size exceeds the addressable range")
            .size();
        #[cfg(target_os = "linux")]
        {
            if bytes == 0 {
                return None;
            }
            // SAFETY: an anonymous private mapping, checked for failure, null and alignment below.
            // The allocation layout bounds its size by isize::MAX, and zero bytes are valid values by the contract.
            unsafe {
                let p = libc::mmap(
                    std::ptr::null_mut(),
                    bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                    -1,
                    0,
                );
                if p == libc::MAP_FAILED {
                    return None;
                }
                let ptr = p.cast::<T>();
                if ptr.is_null() || !ptr.is_aligned() {
                    libc::munmap(p, bytes);
                    return None;
                }
                // Failure only leaves the pages as they are.
                libc::madvise(p, bytes, libc::MADV_NOHUGEPAGE);
                Some(Self {
                    ptr: std::ptr::NonNull::new_unchecked(ptr),
                    len,
                    bytes,
                })
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = bytes;
            None
        }
    }
}

impl<T> Deref for Mapping<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: the mapping holds `len` values, valid by `new`'s contract.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> DerefMut for Mapping<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as in `deref`; the mapping is owned, so the borrow is unique.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> Drop for Mapping<T> {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        // SAFETY: the mapping `new` made, unmapped once.
        unsafe {
            libc::munmap(self.ptr.as_ptr().cast(), self.bytes)
        };
    }
}
