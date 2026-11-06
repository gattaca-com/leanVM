//! The carry-less products of the x86 kernels, at the width of the multiplier: a register of
//! 128-bit lanes, one with PCLMULQDQ alone, two with VPCLMULQDQ and four with AVX-512 as well.
//! Every operation is lane by lane, so a kernel written once serves every width.
//!
//! [`V`] is the register of the width this build has. Its methods carry `#[target_feature]`, so a
//! kernel calls them safely when it carries the same features, which [`clmul_kernel`] gives it.
//! Memory moves go through [`load_at`] and [`store_at`] of `primitives`, which check their bounds.

use core::arch::x86_64::*;

use primitives::{Plain, load_at, store_at};

/// Gives each function in it the features of [`V`], and `#[inline]`: a function that calls
/// methods of [`V`] must carry them to call them safely. Call it from plain code with `unsafe`.
macro_rules! clmul_kernel {
    ($($item:item)*) => {$(
        #[cfg_attr(
            all(target_feature = "avx512f", target_feature = "avx512bw", target_feature = "vpclmulqdq"),
            target_feature(enable = "avx512f,avx512bw,vpclmulqdq")
        )]
        #[cfg_attr(
            all(
                target_feature = "vpclmulqdq",
                not(all(target_feature = "avx512f", target_feature = "avx512bw"))
            ),
            target_feature(enable = "avx2,vpclmulqdq")
        )]
        #[cfg_attr(not(target_feature = "vpclmulqdq"), target_feature(enable = "avx2,pclmulqdq"))]
        #[inline]
        $item
    )*};
}
pub(crate) use clmul_kernel;

cfg_select! {
    all(target_feature = "avx512f", target_feature = "avx512bw", target_feature = "vpclmulqdq") => {
        #[derive(Clone, Copy)]
        #[repr(transparent)]
        pub struct V(__m512i);

        #[target_feature(enable = "avx2")]
        #[inline]
        fn sum256(x: __m256i) -> u128 {
            sum128(_mm256_castsi256_si128(x)) ^ sum128(_mm256_extracti128_si256::<1>(x))
        }

        impl V {
            /// 128-bit lanes.
            pub const N: usize = 4;

            clmul_kernel! {
                /// The `2 * N` words from `src[i]` on.
                pub fn load<T: Plain>(src: &[T], i: usize) -> Self {
                    V(load_at(src, i))
                }
                /// Write the `2 * N` words from `dst[i]` on.
                pub fn store<T: Plain>(dst: &mut [T], i: usize, v: Self) {
                    store_at(dst, i, v.0)
                }
                /// `2 * N` packs, a word each, from `src[i]` on.
                pub fn load_packs(src: &[u32], i: usize) -> Self {
                    V(_mm512_cvtepu32_epi64(load_at::<u32, __m256i>(src, i)))
                }
                pub fn zero() -> Self {
                    V(_mm512_setzero_si512())
                }
                pub fn splat(x: u64) -> Self {
                    V(_mm512_set1_epi64(x as i64))
                }
                pub fn xor(self, other: Self) -> Self {
                    V(_mm512_xor_si512(self.0, other.0))
                }
                /// In each lane, the product of the words `IMM` picks, as `_mm_clmulepi64_si128` does.
                pub fn clmul<const IMM: i32>(self, other: Self) -> Self {
                    V(_mm512_clmulepi64_epi128::<IMM>(self.0, other.0))
                }
                /// In each lane, the high word moved to the low one.
                pub fn high(self) -> Self {
                    V(_mm512_bsrli_epi128::<8>(self.0))
                }
                /// In each lane, the low word of each.
                pub fn unpacklo(self, other: Self) -> Self {
                    V(_mm512_unpacklo_epi64(self.0, other.0))
                }
                /// The sum of the lanes.
                pub fn sum(self) -> u128 {
                    sum256(_mm512_castsi512_si256(self.0)) ^ sum256(_mm512_extracti64x4_epi64::<1>(self.0))
                }
            }
        }
    }
    target_feature = "vpclmulqdq" => {
        #[derive(Clone, Copy)]
        #[repr(transparent)]
        pub struct V(__m256i);

        impl V {
            /// 128-bit lanes.
            pub const N: usize = 2;

            clmul_kernel! {
                /// The `2 * N` words from `src[i]` on.
                pub fn load<T: Plain>(src: &[T], i: usize) -> Self {
                    V(load_at(src, i))
                }
                /// Write the `2 * N` words from `dst[i]` on.
                pub fn store<T: Plain>(dst: &mut [T], i: usize, v: Self) {
                    store_at(dst, i, v.0)
                }
                /// `2 * N` packs, a word each, from `src[i]` on.
                pub fn load_packs(src: &[u32], i: usize) -> Self {
                    V(_mm256_cvtepu32_epi64(load_at::<u32, __m128i>(src, i)))
                }
                pub fn zero() -> Self {
                    V(_mm256_setzero_si256())
                }
                pub fn splat(x: u64) -> Self {
                    V(_mm256_set1_epi64x(x as i64))
                }
                pub fn xor(self, other: Self) -> Self {
                    V(_mm256_xor_si256(self.0, other.0))
                }
                /// In each lane, the product of the words `IMM` picks, as `_mm_clmulepi64_si128` does.
                pub fn clmul<const IMM: i32>(self, other: Self) -> Self {
                    V(_mm256_clmulepi64_epi128::<IMM>(self.0, other.0))
                }
                /// In each lane, the high word moved to the low one.
                pub fn high(self) -> Self {
                    V(_mm256_bsrli_epi128::<8>(self.0))
                }
                /// In each lane, the low word of each.
                pub fn unpacklo(self, other: Self) -> Self {
                    V(_mm256_unpacklo_epi64(self.0, other.0))
                }
                /// The sum of the lanes.
                pub fn sum(self) -> u128 {
                    sum128(_mm256_castsi256_si128(self.0)) ^ sum128(_mm256_extracti128_si256::<1>(self.0))
                }
            }
        }
    }
    _ => {
        #[derive(Clone, Copy)]
        #[repr(transparent)]
        pub struct V(__m128i);

        impl V {
            /// 128-bit lanes.
            pub const N: usize = 1;

            clmul_kernel! {
                /// The `2 * N` words from `src[i]` on.
                pub fn load<T: Plain>(src: &[T], i: usize) -> Self {
                    V(load_at(src, i))
                }
                /// Write the `2 * N` words from `dst[i]` on.
                pub fn store<T: Plain>(dst: &mut [T], i: usize, v: Self) {
                    store_at(dst, i, v.0)
                }
                /// `2 * N` packs, a word each, from `src[i]` on.
                pub fn load_packs(src: &[u32], i: usize) -> Self {
                    V(_mm_cvtepu32_epi64(_mm_cvtsi64_si128(load_at::<u32, u64>(src, i) as i64)))
                }
                pub fn zero() -> Self {
                    V(_mm_setzero_si128())
                }
                pub fn splat(x: u64) -> Self {
                    V(_mm_set1_epi64x(x as i64))
                }
                pub fn xor(self, other: Self) -> Self {
                    V(_mm_xor_si128(self.0, other.0))
                }
                /// In each lane, the product of the words `IMM` picks, as `_mm_clmulepi64_si128` does.
                pub fn clmul<const IMM: i32>(self, other: Self) -> Self {
                    V(_mm_clmulepi64_si128::<IMM>(self.0, other.0))
                }
                /// In each lane, the high word moved to the low one.
                pub fn high(self) -> Self {
                    V(_mm_srli_si128::<8>(self.0))
                }
                /// In each lane, the low word of each.
                pub fn unpacklo(self, other: Self) -> Self {
                    V(_mm_unpacklo_epi64(self.0, other.0))
                }
                /// The sum of the lanes.
                pub fn sum(self) -> u128 {
                    sum128(self.0)
                }
            }
        }
    }
}

/// The two words of a 128-bit lane as one integer.
#[target_feature(enable = "sse4.1")]
#[inline]
fn sum128(x: __m128i) -> u128 {
    _mm_cvtsi128_si64(x) as u64 as u128 | (_mm_extract_epi64::<1>(x) as u64 as u128) << 64
}
