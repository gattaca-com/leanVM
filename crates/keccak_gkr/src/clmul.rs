//! The carry-less products of the x86 kernels, at the width of the multiplier: a register of
//! 128-bit lanes, one with PCLMULQDQ alone, two with VPCLMULQDQ and four with AVX-512 as well.
//! Every operation is lane by lane, so a kernel written once serves every width.
//!
//! A trait method cannot carry `#[target_feature]`, so each intrinsic call is unsafe. The module is
//! compiled only where its features are, which is the safety of every call below.
//! Memory moves go through [`load_at`] and [`store_at`] of `primitives`, which check their bounds.

use core::arch::x86_64::*;

use primitives::{Plain, load_at, store_at};

cfg_select! {
    all(target_feature = "avx512f", target_feature = "vpclmulqdq") => {
        pub type V = __m512i;
    }
    target_feature = "vpclmulqdq" => {
        pub type V = __m256i;
    }
    _ => {
        pub type V = __m128i;
    }
}

pub trait Vector: Copy {
    /// 128-bit lanes.
    const N: usize;
    /// The `2 * N` words from `src[i]` on.
    fn load<T: Plain>(src: &[T], i: usize) -> Self;
    /// Write the `2 * N` words from `dst[i]` on.
    fn store<T: Plain>(dst: &mut [T], i: usize, v: Self);
    /// `2 * N` packs, a word each, from `src[i]` on.
    fn load_packs(src: &[u32], i: usize) -> Self;
    fn zero() -> Self;
    fn splat(x: u64) -> Self;
    fn xor(self, other: Self) -> Self;
    /// In each lane, the product of the words `IMM` picks, as `_mm_clmulepi64_si128` does.
    fn clmul<const IMM: i32>(self, other: Self) -> Self;
    /// In each lane, the high word moved to the low one.
    fn high(self) -> Self;
    /// In each lane, the low word of each.
    fn unpacklo(self, other: Self) -> Self;
    /// The sum of the lanes.
    fn sum(self) -> u128;
}

impl Vector for __m128i {
    const N: usize = 1;
    #[inline(always)]
    fn load<T: Plain>(src: &[T], i: usize) -> Self {
        load_at(src, i)
    }
    #[inline(always)]
    fn store<T: Plain>(dst: &mut [T], i: usize, v: Self) {
        store_at(dst, i, v)
    }
    #[inline(always)]
    fn load_packs(src: &[u32], i: usize) -> Self {
        unsafe { _mm_cvtepu32_epi64(_mm_cvtsi64_si128(load_at::<u32, u64>(src, i) as i64)) }
    }
    #[inline(always)]
    fn zero() -> Self {
        unsafe { _mm_setzero_si128() }
    }
    #[inline(always)]
    fn splat(x: u64) -> Self {
        unsafe { _mm_set1_epi64x(x as i64) }
    }
    #[inline(always)]
    fn xor(self, other: Self) -> Self {
        unsafe { _mm_xor_si128(self, other) }
    }
    #[inline(always)]
    fn clmul<const IMM: i32>(self, other: Self) -> Self {
        unsafe { _mm_clmulepi64_si128::<IMM>(self, other) }
    }
    #[inline(always)]
    fn high(self) -> Self {
        unsafe { _mm_srli_si128::<8>(self) }
    }
    #[inline(always)]
    fn unpacklo(self, other: Self) -> Self {
        unsafe { _mm_unpacklo_epi64(self, other) }
    }
    #[inline(always)]
    fn sum(self) -> u128 {
        unsafe { _mm_cvtsi128_si64(self) as u64 as u128 | (_mm_extract_epi64::<1>(self) as u64 as u128) << 64 }
    }
}

#[cfg(target_feature = "vpclmulqdq")]
impl Vector for __m256i {
    const N: usize = 2;
    #[inline(always)]
    fn load<T: Plain>(src: &[T], i: usize) -> Self {
        load_at(src, i)
    }
    #[inline(always)]
    fn store<T: Plain>(dst: &mut [T], i: usize, v: Self) {
        store_at(dst, i, v)
    }
    #[inline(always)]
    fn load_packs(src: &[u32], i: usize) -> Self {
        unsafe { _mm256_cvtepu32_epi64(load_at::<u32, __m128i>(src, i)) }
    }
    #[inline(always)]
    fn zero() -> Self {
        unsafe { _mm256_setzero_si256() }
    }
    #[inline(always)]
    fn splat(x: u64) -> Self {
        unsafe { _mm256_set1_epi64x(x as i64) }
    }
    #[inline(always)]
    fn xor(self, other: Self) -> Self {
        unsafe { _mm256_xor_si256(self, other) }
    }
    #[inline(always)]
    fn clmul<const IMM: i32>(self, other: Self) -> Self {
        unsafe { _mm256_clmulepi64_epi128::<IMM>(self, other) }
    }
    #[inline(always)]
    fn high(self) -> Self {
        unsafe { _mm256_bsrli_epi128::<8>(self) }
    }
    #[inline(always)]
    fn unpacklo(self, other: Self) -> Self {
        unsafe { _mm256_unpacklo_epi64(self, other) }
    }
    #[inline(always)]
    fn sum(self) -> u128 {
        unsafe { _mm256_castsi256_si128(self).sum() ^ _mm256_extracti128_si256::<1>(self).sum() }
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
impl Vector for __m512i {
    const N: usize = 4;
    #[inline(always)]
    fn load<T: Plain>(src: &[T], i: usize) -> Self {
        load_at(src, i)
    }
    #[inline(always)]
    fn store<T: Plain>(dst: &mut [T], i: usize, v: Self) {
        store_at(dst, i, v)
    }
    #[inline(always)]
    fn load_packs(src: &[u32], i: usize) -> Self {
        unsafe { _mm512_cvtepu32_epi64(load_at::<u32, __m256i>(src, i)) }
    }
    #[inline(always)]
    fn zero() -> Self {
        unsafe { _mm512_setzero_si512() }
    }
    #[inline(always)]
    fn splat(x: u64) -> Self {
        unsafe { _mm512_set1_epi64(x as i64) }
    }
    #[inline(always)]
    fn xor(self, other: Self) -> Self {
        unsafe { _mm512_xor_si512(self, other) }
    }
    #[inline(always)]
    fn clmul<const IMM: i32>(self, other: Self) -> Self {
        unsafe { _mm512_clmulepi64_epi128::<IMM>(self, other) }
    }
    #[inline(always)]
    fn high(self) -> Self {
        unsafe { _mm512_bsrli_epi128::<8>(self) }
    }
    #[inline(always)]
    fn unpacklo(self, other: Self) -> Self {
        unsafe { _mm512_unpacklo_epi64(self, other) }
    }
    #[inline(always)]
    fn sum(self) -> u128 {
        unsafe { _mm512_castsi512_si256(self).sum() ^ _mm512_extracti64x4_epi64::<1>(self).sum() }
    }
}
