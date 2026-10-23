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
pub use avx512::{Acc8, F192x8, F64x8};

/// The unreduced products of eight elements, lane by lane: what [`F192x8::mul_unreduced`] returns.
/// A sum of them is an XOR, and [`Prod8::reduce_lanes`] reduces each lane on its own.
#[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
pub type Prod8 = Acc8;

/// Eight elements, on a target without the wide multiplier.
#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
#[derive(Clone, Copy, Debug)]
pub struct F192x8([F192; 8]);

#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
impl F192x8 {
    pub const ZERO: Self = Self([F192::ZERO; 8]);
    pub const ONE: Self = Self([F192::ONE; 8]);

    #[inline]
    pub fn splat(r: F192) -> Self {
        Self([r; 8])
    }

    #[inline]
    pub fn from_array(e: &[F192; 8]) -> Self {
        Self(*e)
    }

    #[inline]
    pub fn to_array(self) -> [F192; 8] {
        self.0
    }

    #[inline]
    pub fn lane(&self, i: usize) -> F192 {
        self.0[i]
    }

    #[inline]
    pub fn set_lane(&mut self, i: usize, e: F192) {
        self.0[i] = e;
    }

    #[inline]
    pub fn mul_unreduced(self, rhs: Self) -> Prod8 {
        let [a, b] = [0, 4].map(|h| {
            mul_unreduced4(
                core::array::from_fn(|i| self.0[h + i]),
                core::array::from_fn(|i| rhs.0[h + i]),
            )
        });
        Prod8(core::array::from_fn(|i| if i < 4 { a[i] } else { b[i - 4] }))
    }

    /// `([a0, a2, a4, a6, b0, b2, b4, b6], [a1, a3, a5, a7, b1, b3, b5, b7])`.
    #[inline]
    pub fn even_odd(a: Self, b: Self) -> (Self, Self) {
        let pick = |parity: usize| {
            Self(core::array::from_fn(|i| {
                if i < 4 {
                    a.0[2 * i + parity]
                } else {
                    b.0[2 * (i - 4) + parity]
                }
            }))
        };
        (pick(0), pick(1))
    }
}

/// The eight lane-wise products.
#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
impl F192x8 {
    /// The 8x8 transpose: lane `j` of `out[p]` is lane `p` of `g[j]`.
    #[inline]
    pub fn transpose(g: [Self; 8]) -> [Self; 8] {
        core::array::from_fn(|p| Self(core::array::from_fn(|j| g[j].0[p])))
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
impl core::ops::Mul for F192x8 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let [a, b] = [0, 4].map(|h| {
            mul4(
                core::array::from_fn(|i| self.0[h + i]),
                core::array::from_fn(|i| rhs.0[h + i]),
            )
        });
        Self(core::array::from_fn(|i| if i < 4 { a[i] } else { b[i - 4] }))
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
impl core::ops::Add for F192x8 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

/// Eight base-field elements, on a target without the wide multiplier.
#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
#[derive(Clone, Copy, Debug)]
pub struct F64x8([F64; 8]);

#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
impl F64x8 {
    pub const ZERO: Self = Self([F64::ZERO; 8]);
    pub const ONE: Self = Self([F64::ONE; 8]);

    #[inline]
    pub fn from_array(k: &[F64; 8]) -> Self {
        Self(*k)
    }

    #[inline]
    pub fn to_array(self) -> [F64; 8] {
        self.0
    }

    #[inline]
    pub fn lane(&self, i: usize) -> F64 {
        self.0[i]
    }

    #[inline]
    pub fn set_lane(&mut self, i: usize, k: F64) {
        self.0[i] = k;
    }

    /// `([a0, a2, a4, a6, b0, b2, b4, b6], [a1, a3, a5, a7, b1, b3, b5, b7])`.
    #[inline]
    pub fn even_odd(a: Self, b: Self) -> (Self, Self) {
        let pick = |parity: usize| {
            Self(core::array::from_fn(|i| {
                if i < 4 { a.0[2 * i + parity] } else { b.0[2 * (i - 4) + parity] }
            }))
        };
        (pick(0), pick(1))
    }

    /// The 8x8 transpose: lane `j` of `out[p]` is lane `p` of `g[j]`.
    #[inline]
    pub fn transpose(g: [Self; 8]) -> [Self; 8] {
        core::array::from_fn(|p| Self(core::array::from_fn(|j| g[j].0[p])))
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
impl core::ops::Add for F64x8 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
impl core::ops::Mul for F64x8 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i] * rhs.0[i]))
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
impl F192x8 {
    /// The eight mixed products `self_i * k_i`, unreduced.
    #[inline]
    pub fn mul_base_unreduced(self, k: F64x8) -> Prod8 {
        Prod8(core::array::from_fn(|i| self.0[i].mul_base_unreduced(k.0[i])))
    }
}

/// The embedding of the base field.
#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
impl From<F64x8> for F192x8 {
    #[inline]
    fn from(k: F64x8) -> Self {
        Self(k.0.map(F192::from))
    }
}

/// The unreduced products of eight elements, lane by lane, on a target without the wide multiplier.
#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
#[derive(Clone, Copy, Debug)]
pub struct Prod8([F192Unreduced; 8]);

#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
impl Prod8 {
    pub const ZERO: Self = Self([F192Unreduced::ZERO; 8]);

    /// The sum of the eight lanes, reduced.
    #[inline]
    pub fn reduce(self) -> F192 {
        self.0
            .into_iter()
            .fold(F192Unreduced::ZERO, |sum, lane| sum ^ lane)
            .reduce()
    }

    /// Every lane reduced on its own.
    #[inline]
    pub fn reduce_lanes(self) -> F192x8 {
        F192x8(self.0.map(F192Unreduced::reduce))
    }

    /// The same sum, held in as few vectors as the target allows.
    #[inline]
    pub fn merge_parities(self) -> Self {
        self
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
impl core::ops::BitXorAssign for Prod8 {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        for (a, b) in self.0.iter_mut().zip(rhs.0) {
            *a ^= b;
        }
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
impl core::ops::BitXor for Prod8 {
    type Output = Self;

    #[inline]
    fn bitxor(mut self, rhs: Self) -> Self {
        self ^= rhs;
        self
    }
}

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

    /// Every lane reduced on its own, as [`Acc8::reduce_each`].
    #[inline]
    pub fn reduce_lanes(self) -> F192x8 {
        unreachable!("the portable accumulator holds one sum, not eight");
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

/// The eight mixed products `c * k_i`, unreduced.
#[inline]
pub fn mul_base_splat8(c: F192, k: &[F64; 8]) -> Acc8 {
    #[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
    // SAFETY: the features are enabled at compile time.
    return unsafe { avx512::mul_base_splat(c, k) };
    #[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
    Acc8(k.iter().fold(F192Unreduced::ZERO, |acc, &k| acc ^ c.mul_base_unreduced(k)))
}

/// The eight products `a_i * b_i`.
#[inline]
pub fn mul_f64x8(a: &[F64; 8], b: &[F64; 8]) -> [F64; 8] {
    #[cfg(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f"))]
    // SAFETY: the features are enabled at compile time.
    return unsafe { avx512::mul_f64x8(a, b) };
    #[cfg(not(all(target_arch = "x86_64", target_feature = "vpclmulqdq", target_feature = "avx512f")))]
    core::array::from_fn(|i| a[i] * b[i])
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
    #[derive(Clone, Copy, Debug)]
    #[repr(C)]
    pub struct F192x8 {
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

    /// The 8x8 transpose of qwords: qword `j` of `out[p]` is qword `p` of `r[j]`.
    #[inline(always)]
    unsafe fn transpose_qwords(r: [__m512i; 8]) -> [__m512i; 8] {
        // SAFETY: the module is compiled only with avx512f enabled.
        unsafe {
            // Pair rows with `unpack`, then pair the pairs by 128-bit lane, twice.
            let t = [0, 2, 4, 6].map(|i| {
                [
                    _mm512_unpacklo_epi64(r[i], r[i + 1]),
                    _mm512_unpackhi_epi64(r[i], r[i + 1]),
                ]
            });
            let u = |a: usize, b: usize| {
                let (x, y) = (t[a], t[b]);
                [
                    _mm512_shuffle_i64x2::<0x88>(x[0], y[0]),
                    _mm512_shuffle_i64x2::<0x88>(x[1], y[1]),
                    _mm512_shuffle_i64x2::<0xDD>(x[0], y[0]),
                    _mm512_shuffle_i64x2::<0xDD>(x[1], y[1]),
                ]
            };
            let (low, high) = (u(0, 1), u(2, 3));
            let mut out = [_mm512_setzero_si512(); 8];
            for q in 0..4 {
                out[q] = _mm512_shuffle_i64x2::<0x88>(low[q], high[q]);
                out[q + 4] = _mm512_shuffle_i64x2::<0xDD>(low[q], high[q]);
            }
            out
        }
    }

    /// Eight base-field elements, one per qword.
    #[derive(Clone, Copy, Debug)]
    #[repr(transparent)]
    pub struct F64x8(__m512i);

    impl F64x8 {
        // SAFETY: the all-zero bit pattern is a valid vector.
        pub const ZERO: Self = Self(unsafe { core::mem::transmute::<[u64; 8], __m512i>([0; 8]) });

        // SAFETY: every bit pattern is a valid vector.
        pub const ONE: Self = Self(unsafe { core::mem::transmute::<[u64; 8], __m512i>([1; 8]) });

        #[inline(always)]
        pub fn from_array(k: &[F64; 8]) -> Self {
            // SAFETY: eight words are one register.
            Self(unsafe { _mm512_loadu_si512(k.as_ptr().cast()) })
        }

        #[inline(always)]
        pub fn to_array(self) -> [F64; 8] {
            // SAFETY: eight words are one register.
            unsafe { core::mem::transmute::<__m512i, [F64; 8]>(self.0) }
        }

        #[inline]
        pub fn lane(&self, i: usize) -> F64 {
            self.to_array()[i]
        }

        #[inline]
        pub fn set_lane(&mut self, i: usize, k: F64) {
            let mut lanes = self.to_array();
            lanes[i] = k;
            *self = Self::from_array(&lanes);
        }

        /// `([a0, a2, a4, a6, b0, b2, b4, b6], [a1, a3, a5, a7, b1, b3, b5, b7])`.
        #[inline(always)]
        pub fn even_odd(a: Self, b: Self) -> (Self, Self) {
            // SAFETY: the module is compiled only with avx512f enabled.
            unsafe {
                let evens = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
                let odds = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
                (
                    Self(_mm512_permutex2var_epi64(a.0, evens, b.0)),
                    Self(_mm512_permutex2var_epi64(a.0, odds, b.0)),
                )
            }
        }

        /// The 8x8 transpose: lane `j` of `out[p]` is lane `p` of `g[j]`.
        #[inline(always)]
        pub fn transpose(g: [Self; 8]) -> [Self; 8] {
            // SAFETY: the module is compiled only with avx512f enabled.
            unsafe { transpose_qwords(g.map(|k| k.0)).map(Self) }
        }
    }

    impl core::ops::Add for F64x8 {
        type Output = Self;

        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            // SAFETY: the module is compiled only with avx512f enabled.
            Self(unsafe { _mm512_xor_si512(self.0, rhs.0) })
        }
    }

    /// The eight lane-wise products: one carry-less multiplication per parity of word, then the
    /// lane-wise reduction of both.
    impl core::ops::Mul for F64x8 {
        type Output = Self;

        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            // SAFETY: the module is compiled only with the features enabled.
            Self(unsafe { mul_words(self.0, rhs.0) })
        }
    }

    impl F192x8 {
        /// The eight mixed products `self_i * k_i`, unreduced: six carry-less multiplications.
        #[inline(always)]
        pub fn mul_base_unreduced(self, k: F64x8) -> Acc8 {
            // SAFETY: the module is compiled only with the features enabled.
            unsafe { mul_base(self, k.0) }
        }
    }

    /// The embedding of the base field.
    impl From<F64x8> for F192x8 {
        #[inline(always)]
        fn from(k: F64x8) -> Self {
            // SAFETY: the module is compiled only with avx512f enabled.
            unsafe { embed(k.0) }
        }
    }

    impl Acc8 {
        // SAFETY: the all-zero bit pattern is a valid vector.
        pub const ZERO: Self = Self {
            even: [unsafe { core::mem::transmute::<[u64; 8], __m512i>([0; 8]) }; 3],
            odd: [unsafe { core::mem::transmute::<[u64; 8], __m512i>([0; 8]) }; 3],
        };

        /// The same sum with the odd elements' products moved onto the even ones: a running sum then
        /// keeps three vectors, not six.
        #[inline(always)]
        pub fn merge_parities(self) -> Self {
            // SAFETY: the module is compiled only with avx512f enabled.
            let even = unsafe { core::array::from_fn(|i| _mm512_xor_si512(self.even[i], self.odd[i])) };
            Self { even, ..Self::ZERO }
        }

        /// The sum of the sixteen products, reduced.
        #[inline]
        pub fn reduce(self) -> F192 {
            // SAFETY: the module is compiled only with the features enabled.
            unsafe { reduce_sum(self) }
        }

        /// Every lane reduced on its own.
        #[inline(always)]
        pub fn reduce_lanes(self) -> F192x8 {
            // SAFETY: the module is compiled only with the features enabled.
            unsafe { reduce_planes(self) }
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
        #[inline(always)]
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

    impl core::ops::BitXor for Acc8 {
        type Output = Self;

        #[inline(always)]
        fn bitxor(mut self, rhs: Self) -> Self {
            self ^= rhs;
            self
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
    unsafe fn load(e: &[F192; 8]) -> F192x8 {
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
        F192x8 { lo, hi, c2 }
    }

    /// The inverse of [`load`]: six permutes, three stores.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn store(p: F192x8, out: &mut [F192; 8]) {
        let F192x8 { lo, hi, c2 } = p;
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

    impl F192x8 {
        // SAFETY: the all-zero bit pattern is a valid vector.
        pub const ZERO: Self = Self {
            lo: unsafe { core::mem::transmute::<[u64; 8], __m512i>([0; 8]) },
            hi: unsafe { core::mem::transmute::<[u64; 8], __m512i>([0; 8]) },
            c2: unsafe { core::mem::transmute::<[u64; 8], __m512i>([0; 8]) },
        };

        // SAFETY: every bit pattern is a valid vector. Element `i` has `c0 = 1` and no other bit.
        pub const ONE: Self = Self {
            lo: unsafe { core::mem::transmute::<[u64; 8], __m512i>([1, 0, 1, 0, 1, 0, 1, 0]) },
            hi: unsafe { core::mem::transmute::<[u64; 8], __m512i>([1, 0, 1, 0, 1, 0, 1, 0]) },
            c2: unsafe { core::mem::transmute::<[u64; 8], __m512i>([0; 8]) },
        };

        /// One scalar in every element.
        #[inline(always)]
        pub fn splat(r: F192) -> Self {
            // SAFETY: the module is compiled only with the features enabled.
            unsafe { splat(r) }
        }

        #[inline(always)]
        pub fn from_array(e: &[F192; 8]) -> Self {
            // SAFETY: the module is compiled only with the features enabled.
            unsafe { load(e) }
        }

        /// Eight elements from their coefficient registers: qword `i` of `c0`, `c1` and `c2` is that coefficient of element `i`.
        #[inline(always)]
        pub fn from_coeffs(c0: __m512i, c1: __m512i, c2: __m512i) -> Self {
            // SAFETY: the module is compiled only with avx512f enabled.
            unsafe {
                Self {
                    lo: _mm512_unpacklo_epi64(c0, c1),
                    hi: _mm512_unpackhi_epi64(c0, c1),
                    c2,
                }
            }
        }

        #[inline(always)]
        pub fn to_array(self) -> [F192; 8] {
            let mut out = [F192::ZERO; 8];
            // SAFETY: the module is compiled only with the features enabled.
            unsafe { store(self, &mut out) };
            out
        }

        /// Element `i`, read from memory with scalar loads.
        #[inline]
        pub fn lane(&self, i: usize) -> F192 {
            assert!(i < 8);
            let (qwords, at) = (core::ptr::from_ref(self).cast::<u64>(), 2 * (i / 2));
            // SAFETY: the struct is three `repr(C)` planes of eight qwords, and `i < 8`.
            unsafe {
                let plane = qwords.add(8 * (i % 2));
                F192::new(*plane.add(at), *plane.add(at + 1), *qwords.add(16 + i))
            }
        }

        /// Overwrite element `i` with scalar stores.
        #[inline]
        pub fn set_lane(&mut self, i: usize, e: F192) {
            assert!(i < 8);
            let (qwords, at) = (core::ptr::from_mut(self).cast::<u64>(), 2 * (i / 2));
            // SAFETY: as in `lane`.
            unsafe {
                let plane = qwords.add(8 * (i % 2));
                *plane.add(at) = e.c0;
                *plane.add(at + 1) = e.c1;
                *qwords.add(16 + i) = e.c2;
            }
        }

        /// The eight lane-wise products, unreduced: twelve carry-less multiplications.
        #[inline(always)]
        pub fn mul_unreduced(self, rhs: Self) -> Acc8 {
            // SAFETY: the module is compiled only with the features enabled.
            unsafe { mul(self, rhs) }
        }

        /// `([a0, a2, a4, a6, b0, b2, b4, b6], [a1, a3, a5, a7, b1, b3, b5, b7])`.
        ///
        /// Even elements are the `lo` plane and odd ones the `hi` plane, so each output plane is one
        /// 128-bit lane shuffle of two inputs, and `c2` is one two-source qword permute.
        #[inline(always)]
        pub fn even_odd(a: Self, b: Self) -> (Self, Self) {
            // SAFETY: the module is compiled only with avx512f enabled.
            unsafe {
                let evens = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
                let odds = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
                (
                    Self {
                        lo: _mm512_shuffle_i64x2::<0x88>(a.lo, b.lo),
                        hi: _mm512_shuffle_i64x2::<0xDD>(a.lo, b.lo),
                        c2: _mm512_permutex2var_epi64(a.c2, evens, b.c2),
                    },
                    Self {
                        lo: _mm512_shuffle_i64x2::<0x88>(a.hi, b.hi),
                        hi: _mm512_shuffle_i64x2::<0xDD>(a.hi, b.hi),
                        c2: _mm512_permutex2var_epi64(a.c2, odds, b.c2),
                    },
                )
            }
        }
    }

    impl F192x8 {
        /// The 8x8 transpose: lane `j` of `out[p]` is lane `p` of `g[j]`.
        ///
        /// Even elements sit in `lo` and odd ones in `hi`, so each source plane splits by the parity
        /// of `j` into a 4x4 transpose of 128-bit lanes, and `c2` is an 8x8 transpose of qwords.
        #[inline(always)]
        pub fn transpose(g: [Self; 8]) -> [Self; 8] {
            // SAFETY: the module is compiled only with avx512f enabled.
            unsafe {
                // The 128-bit lanes of four registers, out[m] lane i = r[i] lane m.
                let lanes = |r: [__m512i; 4]| {
                    let t0 = _mm512_shuffle_i64x2::<0x88>(r[0], r[1]);
                    let t1 = _mm512_shuffle_i64x2::<0xDD>(r[0], r[1]);
                    let t2 = _mm512_shuffle_i64x2::<0x88>(r[2], r[3]);
                    let t3 = _mm512_shuffle_i64x2::<0xDD>(r[2], r[3]);
                    [
                        _mm512_shuffle_i64x2::<0x88>(t0, t2),
                        _mm512_shuffle_i64x2::<0x88>(t1, t3),
                        _mm512_shuffle_i64x2::<0xDD>(t0, t2),
                        _mm512_shuffle_i64x2::<0xDD>(t1, t3),
                    ]
                };
                let mut out = [Self::ZERO; 8];
                // Source plane `lo` feeds the even outputs and `hi` the odd; even `j` feed `lo`.
                let from_lo_even = lanes([g[0].lo, g[2].lo, g[4].lo, g[6].lo]);
                let from_lo_odd = lanes([g[1].lo, g[3].lo, g[5].lo, g[7].lo]);
                let from_hi_even = lanes([g[0].hi, g[2].hi, g[4].hi, g[6].hi]);
                let from_hi_odd = lanes([g[1].hi, g[3].hi, g[5].hi, g[7].hi]);
                for m in 0..4 {
                    out[2 * m].lo = from_lo_even[m];
                    out[2 * m].hi = from_lo_odd[m];
                    out[2 * m + 1].lo = from_hi_even[m];
                    out[2 * m + 1].hi = from_hi_odd[m];
                }
                let c2 = transpose_qwords(g.map(|e| e.c2));
                for (o, c2) in out.iter_mut().zip(c2) {
                    o.c2 = c2;
                }
                out
            }
        }
    }

    impl core::ops::Add for F192x8 {
        type Output = Self;

        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            xor(self, rhs)
        }
    }

    /// The eight lane-wise products.
    impl core::ops::Mul for F192x8 {
        type Output = Self;

        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            // SAFETY: the module is compiled only with the features enabled.
            unsafe { reduce_planes(mul(self, rhs)) }
        }
    }

    impl core::ops::AddAssign for F192x8 {
        #[inline]
        fn add_assign(&mut self, rhs: Self) {
            *self = xor(*self, rhs);
        }
    }

    #[inline(always)]
    fn xor(a: F192x8, b: F192x8) -> F192x8 {
        // SAFETY: the module is compiled only with avx512f enabled.
        unsafe {
            F192x8 {
                lo: _mm512_xor_si512(a.lo, b.lo),
                hi: _mm512_xor_si512(a.hi, b.hi),
                c2: _mm512_xor_si512(a.c2, b.c2),
            }
        }
    }

    /// One scalar in every lane of the planes.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn splat(r: F192) -> F192x8 {
        let lo = _mm512_broadcast_i32x4(pair(r.c0, r.c1));
        F192x8 {
            lo,
            hi: lo,
            c2: _mm512_set1_epi64(r.c2 as i64),
        }
    }

    /// Eight base-field words as planes: the embedding puts each word in its element's `c0`.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn embed(k: __m512i) -> F192x8 {
        let zero = _mm512_setzero_si512();
        F192x8 {
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
    unsafe fn mul(a: F192x8, b: F192x8) -> Acc8 {
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
    unsafe fn mul_base(a: F192x8, k: __m512i) -> Acc8 {
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
    unsafe fn reduce_planes(acc: Acc8) -> F192x8 {
        let [d0e, d1e, d2e] = acc.even;
        let [d0o, d1o, d2o] = acc.odd;
        // SAFETY: the function carries the feature.
        unsafe {
            let red = |x, y| reduce_lanes512(_mm512_unpacklo_epi64(x, y), _mm512_unpackhi_epi64(x, y));
            F192x8 {
                lo: red(d0e, d1e),
                hi: red(d0o, d1o),
                c2: red(d2e, d2o),
            }
        }
    }

    /// Eight base-field products: one carry-less multiplication per parity of word, then the lane-wise
    /// reduction of both.
    #[inline]
    #[target_feature(enable = "vpclmulqdq", enable = "avx512f")]
    unsafe fn mul_words(a: __m512i, b: __m512i) -> __m512i {
        // SAFETY: the function carries both features.
        unsafe {
            let (even, odd) = (_mm512_clmulepi64_epi128::<0x00>(a, b), _mm512_clmulepi64_epi128::<0x11>(a, b));
            reduce_lanes512(_mm512_unpacklo_epi64(even, odd), _mm512_unpackhi_epi64(even, odd))
        }
    }

    #[inline]
    #[target_feature(enable = "vpclmulqdq", enable = "avx512f")]
    pub unsafe fn mul_f64x8(a: &[F64; 8], b: &[F64; 8]) -> [F64; 8] {
        // SAFETY: the function carries both features; eight words are one register.
        unsafe {
            let (a, b) = (_mm512_loadu_si512(a.as_ptr().cast()), _mm512_loadu_si512(b.as_ptr().cast()));
            core::mem::transmute::<__m512i, [F64; 8]>(mul_words(a, b))
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

            assert_eq!(mul_f64x8(&k0, &k1), core::array::from_fn(|j| k0[j] * k1[j]));
            let dot = |x: &[F192; 8], y: &[F192; 8]| (0..8).fold(F192::ZERO, |acc, j| acc + x[j] * y[j]);
            let (u0, u2) = msg_terms8_ext(&x0, &x1, &y0, &y1);
            assert_eq!(u0.reduce(), dot(&x0, &y0));
            let xs: [F192; 8] = core::array::from_fn(|j| x0[j] + x1[j]);
            let ys: [F192; 8] = core::array::from_fn(|j| y0[j] + y1[j]);
            assert_eq!(u2.reduce(), dot(&xs, &ys));
            assert_eq!(msg_lone8_ext(&x0, &y0).reduce(), dot(&x0, &y0));

            let mut per: Acc8 = mul_base_splat8(r, &k0);
            per ^= mul_base_splat8(x0[0], &k1);
            assert_eq!(per.reduce_each(), core::array::from_fn(|j| r.mul_base(k0[j]) + x0[0].mul_base(k1[j])));
            let mut sum = x0;
            xor_into8(&mut sum, &x1);
            assert_eq!(sum, core::array::from_fn(|j| x0[j] + x1[j]));

            let (xa, xb, xc) = (
                F192x8::from_array(&x0),
                F192x8::from_array(&x1),
                F192x8::from_array(&y0),
            );
            assert_eq!(xa.to_array(), x0);
            assert_eq!(F192x8::splat(r).to_array(), [r; 8]);
            assert_eq!(F192x8::ONE.to_array(), [F192::ONE; 8]);
            assert_eq!((xa + xb).to_array(), core::array::from_fn(|j| x0[j] + x1[j]));
            assert_eq!((xa * xb).to_array(), core::array::from_fn(|j| x0[j] * x1[j]));
            let sum = (xa.mul_unreduced(xb) ^ xc.mul_unreduced(xa)).reduce_lanes();
            assert_eq!(sum.to_array(), core::array::from_fn(|j| x0[j] * x1[j] + y0[j] * x0[j]));
            assert_eq!(xa.mul_unreduced(xb).reduce(), dot(&x0, &x1));
            let rows: [F192x8; 8] = core::array::from_fn(|j| {
                let element = |p: usize| F192::new(x0[p].c0 ^ j as u64, x1[j].c1, y0[(j + p) % 8].c2);
                F192x8::from_array(&core::array::from_fn(element))
            });
            let columns = F192x8::transpose(rows);
            for (p, column) in columns.iter().enumerate() {
                for j in 0..8 {
                    assert_eq!(column.lane(j), rows[j].lane(p));
                }
            }
            let (even, odd) = F192x8::even_odd(xa, xb);
            assert_eq!(
                even.to_array(),
                core::array::from_fn(|j| if j < 4 { x0[2 * j] } else { x1[2 * (j - 4)] })
            );
            assert_eq!(
                odd.to_array(),
                core::array::from_fn(|j| if j < 4 { x0[2 * j + 1] } else { x1[2 * (j - 4) + 1] })
            );
            assert_eq!(xa.lane(5), x0[5]);
            let mut moved = F192x8::ZERO;
            for j in 0..8 {
                moved.set_lane(j, y1[j]);
            }
            assert_eq!(moved.to_array(), y1);

            let (ka, kb) = (F64x8::from_array(&k0), F64x8::from_array(&k1));
            assert_eq!(ka.to_array(), k0);
            assert_eq!(F64x8::ONE.to_array(), [F64::ONE; 8]);
            assert_eq!((ka + kb).to_array(), core::array::from_fn(|j| k0[j] + k1[j]));
            assert_eq!((ka * kb).to_array(), core::array::from_fn(|j| k0[j] * k1[j]));
            let (even, odd) = F64x8::even_odd(ka, kb);
            assert_eq!(
                even.to_array(),
                core::array::from_fn(|j| if j < 4 { k0[2 * j] } else { k1[2 * (j - 4)] })
            );
            assert_eq!(
                odd.to_array(),
                core::array::from_fn(|j| if j < 4 { k0[2 * j + 1] } else { k1[2 * (j - 4) + 1] })
            );
            let rows: [F64x8; 8] = core::array::from_fn(|j| {
                F64x8::from_array(&core::array::from_fn(|p| F64(k0[p].0 ^ k1[(j + p) % 8].0.rotate_left(j as u32))))
            });
            for (p, column) in F64x8::transpose(rows).into_iter().enumerate() {
                for j in 0..8 {
                    assert_eq!(column.lane(j), rows[j].lane(p));
                }
            }
            let mut moved = ka;
            moved.set_lane(3, k1[0]);
            assert_eq!(moved.lane(3), k1[0]);
            assert_eq!(moved.lane(2), k0[2]);
            assert_eq!(F192x8::from(ka).to_array(), k0.map(F192::from));
            assert_eq!(
                xa.mul_base_unreduced(ka).reduce_lanes().to_array(),
                core::array::from_fn(|j| x0[j].mul_base(k0[j]))
            );

            // Accumulation across calls sums before the one reduction.
            let mut acc = Acc8::ZERO;
            acc ^= msg_lone8_ext(&x0, &y0);
            acc ^= mul_base_splat8(y1[0], &k1);
            assert_eq!(
                acc.reduce(),
                dot(&x0, &y0) + (0..8).fold(F192::ZERO, |a, j| a + y1[0].mul_base(k1[j]))
            );
        }
    }
}
