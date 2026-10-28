//! The audited unsafe surface of the AVX-512 code: vector constants, reinterprets and memory moves.
//!
//! A register intrinsic is safe inside a `#[target_feature]` function, so a kernel carries the attribute
//! and moves data through these functions, not through raw pointers. The functions are plain, so any
//! code compiled with `avx512f` calls them.

use core::arch::x86_64::*;

use crate::field::{F64, F192};

/// Plain data: no padding, and every bit pattern is a valid value.
///
/// # Safety
///
/// Implement it only for such types.
pub unsafe trait Plain: Copy {}

// SAFETY: integers, and `repr(transparent)` and `repr(C)` structs of them.
unsafe impl Plain for u8 {}
unsafe impl Plain for u64 {}
unsafe impl Plain for F64 {}
unsafe impl Plain for F192 {}

/// The vector of eight words, the first in the low qword.
pub const fn from_words(w: [u64; 8]) -> __m512i {
    // SAFETY: both types are 64 bytes and every bit pattern is a valid vector.
    unsafe { core::mem::transmute::<[u64; 8], __m512i>(w) }
}

/// The eight words of a vector, the low qword first.
#[inline(always)]
pub fn to_words(v: __m512i) -> [u64; 8] {
    // SAFETY: both types are 64 bytes and every bit pattern is a valid array.
    unsafe { core::mem::transmute::<__m512i, [u64; 8]>(v) }
}

/// Register `i` of `src`: bytes `64 * i..64 * i + 64`, read unaligned.
#[inline(always)]
pub fn load<T: Plain>(src: &[T], i: usize) -> __m512i {
    assert!(64 * (i + 1) <= size_of_val(src));
    // SAFETY: the assert keeps the 64 bytes inside `src`, and any bytes of a `Plain` type are readable.
    unsafe { _mm512_loadu_si512(src.as_ptr().cast::<u8>().add(64 * i).cast()) }
}

/// Write `v` over register `i` of `dst`: bytes `64 * i..64 * i + 64`, unaligned.
#[inline(always)]
pub fn store<T: Plain>(dst: &mut [T], i: usize, v: __m512i) {
    assert!(64 * (i + 1) <= size_of_val(dst));
    // SAFETY: the assert keeps the 64 bytes inside `dst`, and any bytes are a valid `Plain` value.
    unsafe { _mm512_storeu_si512(dst.as_mut_ptr().cast::<u8>().add(64 * i).cast(), v) }
}

/// The bitwise XOR.
#[inline(always)]
pub fn xor(a: __m512i, b: __m512i) -> __m512i {
    // SAFETY: the module is compiled only with avx512f enabled.
    unsafe { _mm512_xor_si512(a, b) }
}
