//! Eight-element `F192` kernels that never move a word through an integer register.
//!
//! On AVX-512 the eight elements live in three plane registers, the [`Weights8`] layout:
//!
//! ```text
//!     lo  lane j = [c0, c1] of element 2j        hi  lane j = [c0, c1] of element 2j+1
//!     c2  qword i = c2 of element i
//! ```
//!
//! A kernel loads its operands straight from memory, so it must only read values stored before the
//! current loop iteration: a 512-bit load over a word just stored stalls on the store buffer.
//!
//! Products accumulate unreduced in [`Acc8`], one reduction per sum rather than per term.

use super::gf2_64::F64;
use super::gf2_64x3::{F192, F192Unreduced};
#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
use super::gf2_64x3::{Weights8, dot_base, mul_base8, mul_unreduced4, mul4};

#[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
pub use avx512::Acc8;

/// The unreduced sum of eight-element products, on a target without the wide multiplier.
#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
#[derive(Clone, Copy, Debug)]
pub struct Acc8(F192Unreduced);

#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
impl Acc8 {
    pub const ZERO: Self = Self(F192Unreduced::ZERO);

    #[inline]
    pub fn reduce(self) -> F192 {
        self.0.reduce()
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
impl From<F192Unreduced> for Acc8 {
    #[inline]
    fn from(u: F192Unreduced) -> Self {
        Self(u)
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
impl core::ops::BitXorAssign for Acc8 {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

/// `x0 + r * (x0 + x1)`, elementwise.
#[inline]
pub fn fold_pairs8_ext(x0: &[F192; 8], x1: &[F192; 8], r: F192) -> [F192; 8] {
    #[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
    // SAFETY: the features are enabled at compile time.
    return unsafe { avx512::fold_pairs_ext(x0, x1, r) };
    #[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
    {
        let s: [F192; 8] = core::array::from_fn(|j| x0[j] + x1[j]);
        let lo = mul4([s[0], s[1], s[2], s[3]], [r; 4]);
        let hi = mul4([s[4], s[5], s[6], s[7]], [r; 4]);
        core::array::from_fn(|j| x0[j] + if j < 4 { lo[j] } else { hi[j - 4] })
    }
}

/// `x0 + r * x0`, elementwise.
#[inline]
pub fn fold_lones8_ext(x0: &[F192; 8], r: F192) -> [F192; 8] {
    #[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
    // SAFETY: the features are enabled at compile time.
    return unsafe { avx512::fold_lones_ext(x0, r) };
    #[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
    {
        let lo = mul4([x0[0], x0[1], x0[2], x0[3]], [r; 4]);
        let hi = mul4([x0[4], x0[5], x0[6], x0[7]], [r; 4]);
        core::array::from_fn(|j| x0[j] + if j < 4 { lo[j] } else { hi[j - 4] })
    }
}

/// `x0 + r * (x0 + x1)` with base-field inputs, elementwise.
#[inline]
pub fn fold_pairs8_base(x0: &[F64; 8], x1: &[F64; 8], r: F192) -> [F192; 8] {
    #[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
    // SAFETY: the features are enabled at compile time.
    return unsafe { avx512::fold_pairs_base(x0, x1, r) };
    #[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
    {
        let p = mul_base8(r, core::array::from_fn(|j| x0[j] + x1[j]));
        core::array::from_fn(|j| F192::from(x0[j]) + p[j])
    }
}

/// `x0 + r * x0` with base-field inputs, elementwise.
#[inline]
pub fn fold_lones8_base(x0: &[F64; 8], r: F192) -> [F192; 8] {
    #[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
    // SAFETY: the features are enabled at compile time.
    return unsafe { avx512::fold_lones_base(x0, r) };
    #[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
    {
        let p = mul_base8(r, *x0);
        core::array::from_fn(|j| F192::from(x0[j]) + p[j])
    }
}

/// `(sum x0*y0, sum (x0+x1)*(y0+y1))` over eight terms.
#[inline]
pub fn msg_terms8_ext(x0: &[F192; 8], x1: &[F192; 8], y0: &[F192; 8], y1: &[F192; 8]) -> (Acc8, Acc8) {
    #[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
    // SAFETY: the features are enabled at compile time.
    return unsafe { avx512::msg_terms_ext(x0, x1, y0, y1) };
    #[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
    {
        let xs: [F192; 8] = core::array::from_fn(|j| x0[j] + x1[j]);
        let ys: [F192; 8] = core::array::from_fn(|j| y0[j] + y1[j]);
        (dot_ext(x0, y0), dot_ext(&xs, &ys))
    }
}

/// `sum x0*y0` over eight terms.
#[inline]
pub fn msg_lone8_ext(x0: &[F192; 8], y0: &[F192; 8]) -> Acc8 {
    #[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
    // SAFETY: the features are enabled at compile time.
    return unsafe { avx512::dot_ext(x0, y0) };
    #[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
    dot_ext(x0, y0)
}

/// `sum y_i * k_i` over eight mixed terms.
#[inline]
pub fn dot8(y: &[F192; 8], k: &[F64; 8]) -> Acc8 {
    #[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
    // SAFETY: the features are enabled at compile time.
    return unsafe { avx512::dot_base(y, k) };
    #[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
    Acc8(dot_base(core::slice::from_ref(&Weights8::new(y)), k))
}

/// The eight mixed products `c * k_i`, unreduced.
#[inline]
pub fn mul_base_splat8(c: F192, k: &[F64; 8]) -> Acc8 {
    #[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
    // SAFETY: the features are enabled at compile time.
    return unsafe { avx512::mul_base_splat(c, k) };
    #[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
    Acc8(k.iter().fold(F192Unreduced::ZERO, |acc, &k| acc ^ c.mul_base_unreduced(k)))
}

/// `out[i] += in[i]` over eight elements.
#[inline]
pub fn xor_into8(out: &mut [F192; 8], add: &[F192; 8]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    // SAFETY: the feature is enabled at compile time; both arrays are 24 qwords.
    unsafe {
        use core::arch::x86_64::*;
        let (o, a) = (out.as_mut_ptr().cast::<i64>(), add.as_ptr().cast::<i64>());
        for i in [0, 8, 16] {
            let v = _mm512_xor_si512(_mm512_loadu_si512(o.add(i).cast()), _mm512_loadu_si512(a.add(i).cast()));
            _mm512_storeu_si512(o.add(i).cast(), v);
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    for (o, a) in out.iter_mut().zip(add) {
        *o += *a;
    }
}

impl Acc8 {
    /// The eight lane sums reduced separately, with the sixteenth-lane layout of
    /// [`mul_base_splat8`]: element `i` of the result is the `i`-th product's sum.
    #[inline]
    pub fn reduce_each(self) -> [F192; 8] {
        #[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
        // SAFETY: the features are enabled at compile time.
        return unsafe { avx512::reduce_each(self) };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
        unreachable!("the portable accumulator holds one sum, not eight");
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
#[inline]
fn dot_ext(x: &[F192; 8], y: &[F192; 8]) -> Acc8 {
    let mut acc = F192Unreduced::ZERO;
    for h in [0, 4] {
        for p in mul_unreduced4([x[h], x[h + 1], x[h + 2], x[h + 3]], [y[h], y[h + 1], y[h + 2], y[h + 3]]) {
            acc ^= p;
        }
    }
    Acc8(acc)
}

#[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
mod avx512 {
    use core::arch::x86_64::*;

    use super::{F64, F192, F192Unreduced};
    use crate::field::gf2_64x3::x86_64::{fold, mul_by_pairs, pair, reduce_lanes512};

    /// Eight elements in their three planes.
    #[derive(Clone, Copy)]
    struct Planes {
        lo: __m512i,
        hi: __m512i,
        c2: __m512i,
    }

    /// The unreduced products of eight elements: the 128-bit coefficients of the even elements, lane
    /// by lane, then those of the odd ones.
    #[derive(Clone, Copy, Debug)]
    pub struct Acc8 {
        even: [__m512i; 3],
        odd: [__m512i; 3],
    }

    impl Acc8 {
        // SAFETY: the all-zero bit pattern is a valid vector.
        pub const ZERO: Self = Self {
            even: [unsafe { core::mem::transmute::<[u64; 8], __m512i>([0; 8]) }; 3],
            odd: [unsafe { core::mem::transmute::<[u64; 8], __m512i>([0; 8]) }; 3],
        };

        /// The sum of the sixteen products, reduced.
        #[inline]
        pub fn reduce(self) -> F192 {
            // SAFETY: the module is compiled only with the features enabled.
            unsafe { reduce_sum(self) }
        }
    }

    impl From<F192Unreduced> for Acc8 {
        /// One scalar product, in the first even lane.
        #[inline]
        fn from(u: F192Unreduced) -> Self {
            let [c0, c1, c2] = u.coeffs();
            // SAFETY: the module is compiled only with avx512f enabled.
            let lane = |[lo, hi]: [u64; 2]| unsafe { _mm512_setr_epi64(lo as i64, hi as i64, 0, 0, 0, 0, 0, 0) };
            Self {
                even: [lane(c0), lane(c1), lane(c2)],
                odd: Self::ZERO.odd,
            }
        }
    }

    impl core::ops::BitXorAssign for Acc8 {
        #[inline]
        fn bitxor_assign(&mut self, rhs: Self) {
            // SAFETY: the module is compiled only with avx512f enabled.
            unsafe {
                for i in 0..3 {
                    self.even[i] = _mm512_xor_si512(self.even[i], rhs.even[i]);
                    self.odd[i] = _mm512_xor_si512(self.odd[i], rhs.odd[i]);
                }
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn reduce_sum(acc: Acc8) -> F192 {
        let coeffs = core::array::from_fn(|i| {
            let a = _mm512_xor_si512(acc.even[i], acc.odd[i]);
            let half = _mm256_xor_si256(_mm512_castsi512_si256(a), _mm512_extracti64x4_epi64::<1>(a));
            let q = _mm_xor_si128(_mm256_castsi256_si128(half), _mm256_extracti128_si256::<1>(half));
            // SAFETY: a 128-bit register and two words hold the same bits.
            unsafe { core::mem::transmute::<__m128i, [u64; 2]>(q) }
        });
        F192Unreduced::from_coeffs(coeffs).reduce()
    }

    /// Eight consecutive elements, 24 qwords, into their planes: three loads, six permutes.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn load(e: &[F192; 8]) -> Planes {
        // SAFETY: eight elements are 24 qwords, three registers.
        let [z0, z1, z2] = unsafe {
            let p = e.as_ptr().cast::<i64>();
            [0, 8, 16].map(|o| _mm512_loadu_si512(p.add(o).cast()))
        };
        // Element i is qwords 3i..3i+3; a permute index 8 + j names qword 8 + j, the third register fills the rest.
        let lo = _mm512_permutex2var_epi64(z0, _mm512_setr_epi64(0, 1, 6, 7, 12, 13, 0, 0), z1);
        let lo = _mm512_mask_permutexvar_epi64(lo, 0b1100_0000, _mm512_setr_epi64(0, 0, 0, 0, 0, 0, 2, 3), z2);
        let hi = _mm512_permutex2var_epi64(z0, _mm512_setr_epi64(3, 4, 9, 10, 15, 0, 0, 0), z1);
        let hi = _mm512_mask_permutexvar_epi64(hi, 0b1110_0000, _mm512_setr_epi64(0, 0, 0, 0, 0, 0, 5, 6), z2);
        let c2 = _mm512_permutex2var_epi64(z0, _mm512_setr_epi64(2, 5, 8, 11, 14, 0, 0, 0), z1);
        let c2 = _mm512_mask_permutexvar_epi64(c2, 0b1110_0000, _mm512_setr_epi64(0, 0, 0, 0, 0, 1, 4, 7), z2);
        Planes { lo, hi, c2 }
    }

    /// The inverse of [`load`]: six permutes, three stores.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn store(p: Planes, out: &mut [F192; 8]) {
        let Planes { lo, hi, c2 } = p;
        let v0 = _mm512_permutex2var_epi64(lo, _mm512_setr_epi64(0, 1, 0, 8, 9, 0, 2, 3), hi);
        let v0 = _mm512_mask_permutexvar_epi64(v0, 0b0010_0100, _mm512_setr_epi64(0, 0, 0, 0, 0, 1, 0, 0), c2);
        let v1 = _mm512_permutex2var_epi64(lo, _mm512_setr_epi64(0, 10, 11, 0, 4, 5, 0, 12), hi);
        let v1 = _mm512_mask_permutexvar_epi64(v1, 0b0100_1001, _mm512_setr_epi64(2, 0, 0, 3, 0, 0, 4, 0), c2);
        let v2 = _mm512_permutex2var_epi64(lo, _mm512_setr_epi64(13, 0, 6, 7, 0, 14, 15, 0), hi);
        let v2 = _mm512_mask_permutexvar_epi64(v2, 0b1001_0010, _mm512_setr_epi64(0, 5, 0, 0, 6, 0, 0, 7), c2);
        // SAFETY: the three stores write the 24 qwords of `out`.
        unsafe {
            let p = out.as_mut_ptr().cast::<i64>();
            _mm512_storeu_si512(p.cast(), v0);
            _mm512_storeu_si512(p.add(8).cast(), v1);
            _mm512_storeu_si512(p.add(16).cast(), v2);
        }
    }

    #[inline(always)]
    fn xor(a: Planes, b: Planes) -> Planes {
        // SAFETY: the module is compiled only with avx512f enabled.
        unsafe {
            Planes {
                lo: _mm512_xor_si512(a.lo, b.lo),
                hi: _mm512_xor_si512(a.hi, b.hi),
                c2: _mm512_xor_si512(a.c2, b.c2),
            }
        }
    }

    /// One scalar in every lane of the planes.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn splat(r: F192) -> Planes {
        let lo = _mm512_broadcast_i32x4(pair(r.c0, r.c1));
        Planes {
            lo,
            hi: lo,
            c2: _mm512_set1_epi64(r.c2 as i64),
        }
    }

    /// Eight base-field words as planes: the embedding puts each word in its element's `c0`.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn embed(k: __m512i) -> Planes {
        let zero = _mm512_setzero_si512();
        Planes {
            lo: _mm512_unpacklo_epi64(k, zero),
            hi: _mm512_unpackhi_epi64(k, zero),
            c2: zero,
        }
    }

    /// The Karatsuba operands of one parity: the pair plane, `c0 + c1` in both qwords, and the pair
    /// plane with that parity's `c2` added to both qwords.
    #[inline(always)]
    fn karatsuba_operands(pairs: __m512i, c2_dup: __m512i) -> (__m512i, __m512i, __m512i) {
        // SAFETY: the module is compiled only with avx512f enabled.
        unsafe {
            let swapped = _mm512_shuffle_epi32::<0x4E>(pairs);
            (pairs, _mm512_xor_si512(pairs, swapped), _mm512_xor_si512(pairs, c2_dup))
        }
    }

    /// The six Karatsuba products of one parity, y-folded.
    #[inline(always)]
    fn karatsuba_lanes(
        (ap, a_s, a_t): (__m512i, __m512i, __m512i),
        a2: __m512i,
        (bp, b_s, b_t): (__m512i, __m512i, __m512i),
        b2: __m512i,
    ) -> [__m512i; 3] {
        // SAFETY: the module is compiled only with vpclmulqdq and avx512f enabled.
        unsafe {
            let p0 = _mm512_clmulepi64_epi128::<0x00>(ap, bp);
            let p1 = _mm512_clmulepi64_epi128::<0x11>(ap, bp);
            let p2 = _mm512_clmulepi64_epi128::<0x00>(a2, b2);
            let p01 = _mm512_clmulepi64_epi128::<0x00>(a_s, b_s);
            let p02 = _mm512_clmulepi64_epi128::<0x00>(a_t, b_t);
            let p12 = _mm512_clmulepi64_epi128::<0x11>(a_t, b_t);
            fold(p0, p1, p2, p01, p02, p12, |x, y| _mm512_xor_si512(x, y))
        }
    }

    /// The eight lane-wise products, unreduced: twelve carry-less multiplications.
    #[inline]
    #[target_feature(enable = "vpclmulqdq", enable = "avx512f")]
    unsafe fn mul(a: Planes, b: Planes) -> Acc8 {
        // Each parity's c2 in both qwords of its lane.
        let (a2e, a2o) = (_mm512_shuffle_epi32::<0x44>(a.c2), _mm512_shuffle_epi32::<0xEE>(a.c2));
        let (b2e, b2o) = (_mm512_shuffle_epi32::<0x44>(b.c2), _mm512_shuffle_epi32::<0xEE>(b.c2));
        Acc8 {
            even: karatsuba_lanes(karatsuba_operands(a.lo, a2e), a2e, karatsuba_operands(b.lo, b2e), b2e),
            odd: karatsuba_lanes(karatsuba_operands(a.hi, a2o), a2o, karatsuba_operands(b.hi, b2o), b2o),
        }
    }

    /// The eight mixed products by the words of `k`, unreduced: six carry-less multiplications.
    #[inline]
    #[target_feature(enable = "vpclmulqdq", enable = "avx512f")]
    unsafe fn mul_base(a: Planes, k: __m512i) -> Acc8 {
        // SAFETY: the function carries both features.
        let [e0, e1, o0, o1, e2, o2] = unsafe { mul_by_pairs(a.lo, a.hi, a.c2, k) };
        Acc8 {
            even: [e0, e1, e2],
            odd: [o0, o1, o2],
        }
    }

    /// Reduce every lane back to planes.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn reduce_planes(acc: Acc8) -> Planes {
        let [d0e, d1e, d2e] = acc.even;
        let [d0o, d1o, d2o] = acc.odd;
        // SAFETY: the function carries the feature.
        unsafe {
            let red = |x, y| reduce_lanes512(_mm512_unpacklo_epi64(x, y), _mm512_unpackhi_epi64(x, y));
            Planes {
                lo: red(d0e, d1e),
                hi: red(d0o, d1o),
                c2: red(d2e, d2o),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "vpclmulqdq", enable = "avx512f")]
    pub unsafe fn fold_pairs_ext(x0: &[F192; 8], x1: &[F192; 8], r: F192) -> [F192; 8] {
        // SAFETY: the function carries both features.
        unsafe {
            let (a, b) = (load(x0), load(x1));
            let p = reduce_planes(mul(xor(a, b), splat(r)));
            let mut out = [F192::ZERO; 8];
            store(xor(a, p), &mut out);
            out
        }
    }

    #[inline]
    #[target_feature(enable = "vpclmulqdq", enable = "avx512f")]
    pub unsafe fn fold_lones_ext(x0: &[F192; 8], r: F192) -> [F192; 8] {
        // SAFETY: the function carries both features.
        unsafe {
            let a = load(x0);
            let p = reduce_planes(mul(a, splat(r)));
            let mut out = [F192::ZERO; 8];
            store(xor(a, p), &mut out);
            out
        }
    }

    #[inline]
    #[target_feature(enable = "vpclmulqdq", enable = "avx512f")]
    pub unsafe fn fold_pairs_base(x0: &[F64; 8], x1: &[F64; 8], r: F192) -> [F192; 8] {
        // SAFETY: the function carries both features; eight words are one register.
        unsafe {
            let k0 = _mm512_loadu_si512(x0.as_ptr().cast());
            let k1 = _mm512_loadu_si512(x1.as_ptr().cast());
            let p = reduce_planes(mul_base(splat(r), _mm512_xor_si512(k0, k1)));
            let mut out = [F192::ZERO; 8];
            store(xor(embed(k0), p), &mut out);
            out
        }
    }

    #[inline]
    #[target_feature(enable = "vpclmulqdq", enable = "avx512f")]
    pub unsafe fn fold_lones_base(x0: &[F64; 8], r: F192) -> [F192; 8] {
        // SAFETY: the function carries both features; eight words are one register.
        unsafe {
            let k0 = _mm512_loadu_si512(x0.as_ptr().cast());
            let p = reduce_planes(mul_base(splat(r), k0));
            let mut out = [F192::ZERO; 8];
            store(xor(embed(k0), p), &mut out);
            out
        }
    }

    #[inline]
    #[target_feature(enable = "vpclmulqdq", enable = "avx512f")]
    pub unsafe fn msg_terms_ext(x0: &[F192; 8], x1: &[F192; 8], y0: &[F192; 8], y1: &[F192; 8]) -> (Acc8, Acc8) {
        // SAFETY: the function carries both features.
        unsafe {
            let (a0, a1, b0, b1) = (load(x0), load(x1), load(y0), load(y1));
            (mul(a0, b0), mul(xor(a0, a1), xor(b0, b1)))
        }
    }

    #[inline]
    #[target_feature(enable = "vpclmulqdq", enable = "avx512f")]
    pub unsafe fn dot_ext(x0: &[F192; 8], y0: &[F192; 8]) -> Acc8 {
        // SAFETY: the function carries both features.
        unsafe { mul(load(x0), load(y0)) }
    }

    #[inline]
    #[target_feature(enable = "vpclmulqdq", enable = "avx512f")]
    pub unsafe fn mul_base_splat(c: F192, k: &[F64; 8]) -> Acc8 {
        // SAFETY: the function carries both features; eight words are one register.
        unsafe { mul_base(splat(c), _mm512_loadu_si512(k.as_ptr().cast())) }
    }

    #[inline]
    #[target_feature(enable = "vpclmulqdq", enable = "avx512f")]
    pub unsafe fn reduce_each(acc: Acc8) -> [F192; 8] {
        let mut out = [F192::ZERO; 8];
        // SAFETY: the function carries both features.
        unsafe { store(reduce_planes(acc), &mut out) };
        out
    }

    #[inline]
    #[target_feature(enable = "vpclmulqdq", enable = "avx512f")]
    pub unsafe fn dot_base(y: &[F192; 8], k: &[F64; 8]) -> Acc8 {
        // SAFETY: the function carries both features; eight words are one register.
        unsafe { mul_base(load(y), _mm512_loadu_si512(k.as_ptr().cast())) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_rng::Rng;

    #[test]
    fn kernels_match_the_scalar_forms() {
        let mut rng = Rng::new(0xF192_0008);
        for _ in 0..200 {
            let x0: [F192; 8] = core::array::from_fn(|_| rng.ext());
            let x1: [F192; 8] = core::array::from_fn(|_| rng.ext());
            let y0: [F192; 8] = core::array::from_fn(|_| rng.ext());
            let y1: [F192; 8] = core::array::from_fn(|_| rng.ext());
            let k0: [F64; 8] = core::array::from_fn(|_| F64(rng.next_u64()));
            let k1: [F64; 8] = core::array::from_fn(|_| F64(rng.next_u64()));
            let r = rng.ext();

            assert_eq!(fold_pairs8_ext(&x0, &x1, r), core::array::from_fn(|j| x0[j] + r * (x0[j] + x1[j])));
            assert_eq!(fold_lones8_ext(&x0, r), core::array::from_fn(|j| x0[j] + r * x0[j]));
            assert_eq!(
                fold_pairs8_base(&k0, &k1, r),
                core::array::from_fn(|j| F192::from(k0[j]) + r.mul_base(k0[j] + k1[j]))
            );
            assert_eq!(fold_lones8_base(&k0, r), core::array::from_fn(|j| F192::from(k0[j]) + r.mul_base(k0[j])));

            let dot = |x: &[F192; 8], y: &[F192; 8]| (0..8).fold(F192::ZERO, |acc, j| acc + x[j] * y[j]);
            let (u0, u2) = msg_terms8_ext(&x0, &x1, &y0, &y1);
            assert_eq!(u0.reduce(), dot(&x0, &y0));
            let xs: [F192; 8] = core::array::from_fn(|j| x0[j] + x1[j]);
            let ys: [F192; 8] = core::array::from_fn(|j| y0[j] + y1[j]);
            assert_eq!(u2.reduce(), dot(&xs, &ys));
            assert_eq!(msg_lone8_ext(&x0, &y0).reduce(), dot(&x0, &y0));
            assert_eq!(dot8(&y0, &k0).reduce(), (0..8).fold(F192::ZERO, |acc, j| acc + y0[j].mul_base(k0[j])));

            let mut per: Acc8 = mul_base_splat8(r, &k0);
            per ^= mul_base_splat8(x0[0], &k1);
            assert_eq!(per.reduce_each(), core::array::from_fn(|j| r.mul_base(k0[j]) + x0[0].mul_base(k1[j])));
            let mut sum = x0;
            xor_into8(&mut sum, &x1);
            assert_eq!(sum, core::array::from_fn(|j| x0[j] + x1[j]));

            // Accumulation across calls sums before the one reduction.
            let mut acc = Acc8::ZERO;
            acc ^= msg_lone8_ext(&x0, &y0);
            acc ^= dot8(&y1, &k1);
            assert_eq!(
                acc.reduce(),
                dot(&x0, &y0) + (0..8).fold(F192::ZERO, |a, j| a + y1[j].mul_base(k1[j]))
            );
        }
    }
}
