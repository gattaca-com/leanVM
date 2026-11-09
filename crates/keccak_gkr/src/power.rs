//! The field in the power basis of a challenge `s`: an element is its 128 coefficients on
//! `1, s, ..., s^127`, and a product is a carry-less one reduced modulo the minimal polynomial
//! of `s`.
//!
//! The rounds after the bit-level ones stay in this basis. Their tables come out of carry-less
//! sums already in it, and moving every entry to the standard basis would cost more than
//! reducing by a dense modulus does.
//!
//! A table row is laid out by word: word `k` of entry `j` at `k * width + j`.

use primitives::field::F128;

use crate::basis::{PolyEval, PowerBasis, add_mul};
use crate::keccak::{BITS, along};

pub type Element = [u64; 2];
pub const FACTOR_ROW: usize = 2 * 64;

/// A row kernel's vector form, where one is compiled and `width` is whole vectors of it, else
/// `portable`. The caller bounds every row.
macro_rules! row_kernel {
    ($width:expr, $kernel:ident($($arg:expr),*), $portable:expr) => {
        cfg_select! {
            all(target_arch = "x86_64", target_feature = "avx2", target_feature = "pclmulqdq") => {
                if $width.is_multiple_of(x86::STEP) {
                    // SAFETY: the features are enabled at compile time, and the caller bounds every row.
                    unsafe { x86::$kernel($($arg),*) }
                } else {
                    $portable
                }
            }
            all(target_arch = "aarch64", target_feature = "aes") => {
                if $width.is_multiple_of(2) {
                    // SAFETY: the feature is enabled at compile time, and the caller bounds every row.
                    unsafe { neon::$kernel($($arg),*) }
                } else {
                    $portable
                }
            }
            _ => $portable,
        }
    };
}

pub struct PowerField {
    pub basis: PowerBasis,
    /// `s^128`, the modulus without its leading term.
    modulus: Element,
    /// `floor(x^256 / modulus)` without its leading term, for Barrett's reduction.
    barrett: Element,
    eval: PolyEval,
}

/// The carry-less product of two elements, four words.
#[inline(always)]
fn mul_wide2(a: &Element, b: &Element) -> [u64; 4] {
    let mut out = [0u64; 4];
    add_mul(&mut out, a, b);
    out
}

impl PowerField {
    /// `None` when `s` lies in a proper subfield.
    pub fn new(s: F128) -> Option<Self> {
        let basis = PowerBasis::new(s)?;
        let mut power = F128::ONE;
        for _ in 0..128 {
            power *= s;
        }
        let modulus = basis.to_poly(power);
        // Long division of `x^256` by `x^128 + modulus`.
        let mut dividend = [0u64; 5];
        dividend[4] = 1;
        let mut barrett = [0u64; 2];
        for bit in (128..=256).rev() {
            if dividend[bit / 64] >> (bit % 64) & 1 == 1 {
                let shift = bit - 128;
                if shift < 128 {
                    barrett[shift / 64] |= 1 << (shift % 64);
                }
                dividend[bit / 64] ^= 1 << (bit % 64);
                for b in (0..128).filter(|b| modulus[b / 64] >> (b % 64) & 1 == 1) {
                    dividend[(b + shift) / 64] ^= 1 << ((b + shift) % 64);
                }
            }
        }
        Some(Self {
            basis,
            modulus,
            barrett,
            eval: PolyEval::new(s, 16),
        })
    }

    pub fn to_poly(&self, a: F128) -> Element {
        self.basis.to_poly(a)
    }

    pub fn to_field(&self, a: &Element) -> F128 {
        self.eval.eval(a)
    }

    /// A polynomial of at most 256 coefficients, modulo the minimal polynomial.
    #[inline(always)]
    pub fn reduce(&self, p: &[u64; 4]) -> Element {
        let high: Element = [p[2], p[3]];
        // The quotient: `high`, plus the top half of `high * barrett`.
        let t = mul_wide2(&high, &self.barrett);
        let q: Element = [high[0] ^ t[2], high[1] ^ t[3]];
        let r = mul_wide2(&q, &self.modulus);
        [p[0] ^ r[0], p[1] ^ r[1]]
    }

    #[inline(always)]
    pub fn mul(&self, a: &Element, b: &Element) -> Element {
        self.reduce(&mul_wide2(a, b))
    }

    /// `sum of a[j] * b[j]` and `sum of (a[j] + a'[j]) * (b[j] + b'[j])` over two pairs of rows,
    /// unreduced.
    pub fn dot_pair(&self, a: [&[u64]; 2], b: [&[u64]; 2], width: usize) -> ([u64; 4], [u64; 4]) {
        assert!(a.iter().chain(&b).all(|row| row.len() >= 2 * width));
        row_kernel!(width, dot_pair(a, b, width), dot_pair_portable(a, b, width))
    }

    /// `sum_j (weights[j] * a[j]) * b[j]`, with the inner product reduced and the sum unreduced.
    pub fn weighted_dot(&self, a: &[u64], b: &[u64], weights: &[u64], width: usize) -> [u64; 4] {
        assert!([a, b, weights].iter().all(|row| row.len() >= 2 * width));
        row_kernel!(
            width,
            weighted_dot(self, a, b, weights, width),
            self.weighted_dot_portable(a, b, weights, width)
        )
    }

    fn weighted_dot_portable(&self, a: &[u64], b: &[u64], weights: &[u64], width: usize) -> [u64; 4] {
        let mut sum = [0; 4];
        for j in 0..width {
            let product = self.mul(&get(a, width, j), &get(weights, width, j));
            add_mul(&mut sum, &product, &get(b, width, j));
        }
        sum
    }

    /// The dot products of `B[j] = weights[j] * (kappa + T[j + 1]) + constant[j]` and `T[j + 2]`,
    /// at zero and infinity over two rows, unreduced or not. Weights and constants are by lane,
    /// then by word.
    pub fn chi_dot_pair(
        &self,
        t: [&[u64]; 2],
        weights: &[u64],
        constant: &[u64],
        kappa: &Element,
        scratch: &mut [u64],
    ) -> ([u64; 4], [u64; 4]) {
        assert!(
            t.into_iter()
                .chain([weights, constant])
                .all(|row| row.len() >= 2 * BITS)
        );
        assert!(scratch.len() >= 4 * FACTOR_ROW);
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "pclmulqdq"))]
        if 64usize.is_multiple_of(x86::STEP) {
            // SAFETY: the rows are bounded above, and a lane contains whole vectors.
            return unsafe { x86::chi_dot_pair(self, t, weights, constant, kappa, scratch) };
        }
        self.chi_dot_pair_by_lane(t, weights, constant, kappa, scratch)
    }

    fn chi_dot_pair_by_lane(
        &self,
        t: [&[u64]; 2],
        weights: &[u64],
        constant: &[u64],
        kappa: &Element,
        scratch: &mut [u64],
    ) -> ([u64; 4], [u64; 4]) {
        let (b, shifted) = scratch.split_at_mut(2 * FACTOR_ROW);
        let (mut zero, mut infinity) = ([0u64; 4], [0u64; 4]);
        for lane in 0..BITS / 64 {
            for (half, t) in t.into_iter().enumerate() {
                let (b, shifted) = (
                    &mut b[half * FACTOR_ROW..][..FACTOR_ROW],
                    &mut shifted[half * FACTOR_ROW..][..FACTOR_ROW],
                );
                let at = along(64 * lane, 1);
                for k in 0..2 {
                    for z in 0..64 {
                        shifted[64 * k + z] = t[k * BITS + at + z] ^ kappa[k];
                    }
                }
                let at = FACTOR_ROW * lane;
                self.mul_row(
                    b,
                    &weights[at..][..FACTOR_ROW],
                    shifted,
                    &constant[at..][..FACTOR_ROW],
                    64,
                );
                let at = along(64 * lane, 2);
                for k in 0..2 {
                    shifted[64 * k..][..64].copy_from_slice(&t[k * BITS + at..][..64]);
                }
            }
            let (z, i) = self.dot_pair(
                [&b[..FACTOR_ROW], &b[FACTOR_ROW..]],
                [&shifted[..FACTOR_ROW], &shifted[FACTOR_ROW..]],
                64,
            );
            for k in 0..4 {
                zero[k] ^= z[k];
                infinity[k] ^= i[k];
            }
        }
        (zero, infinity)
    }

    /// `dst[j] = lo[j] + challenge * (lo[j] + hi[j])`.
    pub fn fold_row(&self, dst: &mut [u64], lo: &[u64], hi: &[u64], challenge: &Element, width: usize) {
        assert!(dst.len() >= 2 * width && lo.len() >= 2 * width && hi.len() >= 2 * width);
        row_kernel!(
            width,
            fold_row(self, dst, lo, hi, challenge, width),
            self.fold_row_portable(dst, lo, hi, challenge, width)
        )
    }

    /// Fold rows indexed by `x + 2y`, with challenges `[r, s, r*s]`, sharing one reduction.
    pub fn fold_quad(&self, dst: &mut [u64], rows: [&[u64]; 4], challenges: &[Element; 3], width: usize) {
        assert!(dst.len() >= 2 * width && rows.iter().all(|row| row.len() >= 2 * width));
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "pclmulqdq"))]
        if width.is_multiple_of(x86::STEP) {
            // SAFETY: the features are enabled at compile time and the rows are bounded above.
            unsafe { x86::fold_quad(self, dst, rows, challenges, width) };
            return;
        }
        for j in 0..width {
            let [a, b, c, d] = rows.map(|row| get(row, width, j));
            let differences: [Element; 3] = [
                std::array::from_fn(|k| a[k] ^ b[k]),
                std::array::from_fn(|k| a[k] ^ c[k]),
                std::array::from_fn(|k| a[k] ^ b[k] ^ c[k] ^ d[k]),
            ];
            let mut wide = [0; 4];
            for (delta, r) in differences.iter().zip(challenges) {
                add_mul(&mut wide, delta, r);
            }
            let folded = self.reduce(&wide);
            set(dst, width, j, &std::array::from_fn(|k| a[k] ^ folded[k]));
        }
    }

    /// Chi at (0,0), (inf,0), (0,1), (inf,1), (1,0), (1,inf), (0,inf), (inf,inf).
    /// Infinity selects the leading coefficient in that variable.
    pub fn chi_dot_quad(
        &self,
        rows: [&[u64]; 4],
        weights: &[u64],
        constant: &[u64],
        kappa: &Element,
        scratch: &mut [u64],
    ) -> [[u64; 4]; 8] {
        assert!(
            rows.into_iter()
                .chain([weights, constant])
                .all(|row| row.len() >= 2 * BITS)
        );
        assert!(scratch.len() >= 12 * FACTOR_ROW);
        let (a, b) = scratch.split_at_mut(6 * FACTOR_ROW);
        let mut sums = [[0; 4]; 8];
        for lane in 0..BITS / 64 {
            for (i, row) in rows.iter().enumerate() {
                let (a, b) = (
                    &mut a[i * FACTOR_ROW..][..FACTOR_ROW],
                    &mut b[i * FACTOR_ROW..][..FACTOR_ROW],
                );
                for k in 0..2 {
                    for z in 0..64 {
                        b[64 * k + z] = row[BITS * k + along(64 * lane, 1) + z] ^ kappa[k];
                    }
                }
                let at = lane * FACTOR_ROW;
                self.mul_row(a, &weights[at..], b, &constant[at..], 64);
                for k in 0..2 {
                    b[64 * k..][..64].copy_from_slice(&row[BITS * k + along(64 * lane, 2)..][..64]);
                }
            }
            for row in [&mut *a, &mut *b] {
                for k in 0..2 * FACTOR_ROW {
                    row[4 * FACTOR_ROW + k] = row[k] ^ row[2 * FACTOR_ROW + k];
                }
            }
            for (pair, [lo, hi]) in [[0, 1], [2, 3], [1, 3], [4, 5]].into_iter().enumerate() {
                let (zero, infinity) = self.dot_pair(
                    [lo, hi].map(|i| &a[i * FACTOR_ROW..][..FACTOR_ROW]),
                    [lo, hi].map(|i| &b[i * FACTOR_ROW..][..FACTOR_ROW]),
                    64,
                );
                for k in 0..4 {
                    sums[2 * pair][k] ^= zero[k];
                    sums[2 * pair + 1][k] ^= infinity[k];
                }
            }
        }
        sums
    }

    /// `dst[j] = wide[j]` reduced, for a row of three-word polynomials laid out by word.
    #[cfg(any(
        test,
        not(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512vbmi",
            target_feature = "gfni"
        ))
    ))]
    pub fn reduce_row(&self, dst: &mut [u64], wide: &[u64], width: usize) {
        assert!(dst.len() >= 2 * width && wide.len() >= 3 * width);
        row_kernel!(
            width,
            reduce_row(self, dst, wide, width),
            self.reduce_row_portable(dst, wide, width)
        )
    }

    #[cfg(any(
        test,
        not(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512vbmi",
            target_feature = "gfni"
        ))
    ))]
    pub fn reduce_row_portable(&self, dst: &mut [u64], wide: &[u64], width: usize) {
        for j in 0..width {
            let p = [wide[j], wide[width + j], wide[2 * width + j], 0];
            set(dst, width, j, &self.reduce(&p));
        }
    }

    /// `dst[j] = a[j] * b[j] + c[j]`.
    pub fn mul_row(&self, dst: &mut [u64], a: &[u64], b: &[u64], c: &[u64], width: usize) {
        assert!(dst.len() >= 2 * width && [a, b, c].iter().all(|row| row.len() >= 2 * width));
        row_kernel!(
            width,
            mul_row(self, dst, a, b, c, width),
            self.mul_row_portable(dst, a, b, c, width)
        )
    }

    pub fn mul_row_portable(&self, dst: &mut [u64], a: &[u64], b: &[u64], c: &[u64], width: usize) {
        for j in 0..width {
            let (p, c) = (self.mul(&get(a, width, j), &get(b, width, j)), get(c, width, j));
            set(dst, width, j, &[p[0] ^ c[0], p[1] ^ c[1]]);
        }
    }

    pub fn fold_row_portable(&self, dst: &mut [u64], lo: &[u64], hi: &[u64], challenge: &Element, width: usize) {
        for j in 0..width {
            let (l, h) = (get(lo, width, j), get(hi, width, j));
            let delta = [l[0] ^ h[0], l[1] ^ h[1]];
            let folded = self.mul(challenge, &delta);
            for k in 0..2 {
                dst[k * width + j] = l[k] ^ folded[k];
            }
        }
    }
}

/// Entry `j` of a row.
#[inline(always)]
pub fn get(row: &[u64], width: usize, j: usize) -> Element {
    [row[j], row[width + j]]
}

#[inline(always)]
pub fn set(row: &mut [u64], width: usize, j: usize, value: &Element) {
    for k in 0..2 {
        row[k * width + j] = value[k];
    }
}

pub fn dot_pair_portable(a: [&[u64]; 2], b: [&[u64]; 2], width: usize) -> ([u64; 4], [u64; 4]) {
    let (mut zero, mut infinity) = ([0u64; 4], [0u64; 4]);
    for j in 0..width {
        let (a0, a1, b0, b1) = (
            get(a[0], width, j),
            get(a[1], width, j),
            get(b[0], width, j),
            get(b[1], width, j),
        );
        let da = [a0[0] ^ a1[0], a0[1] ^ a1[1]];
        let db = [b0[0] ^ b1[0], b0[1] ^ b1[1]];
        for (acc, p) in [(&mut zero, mul_wide2(&a0, &b0)), (&mut infinity, mul_wide2(&da, &db))] {
            for (x, y) in acc.iter_mut().zip(p) {
                *x ^= y;
            }
        }
    }
    (zero, infinity)
}

/// The x86 kernels on a register of [`V`](crate::clmul::V) lanes, two entries to a lane. They carry
/// the features of `V`, which this module's cfg guarantees, so the dispatch calls them with `unsafe`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "pclmulqdq"))]
mod x86 {
    use super::*;
    use crate::clmul::{V, clmul_kernel};

    /// Entries to a register.
    pub const STEP: usize = 2 * V::N;

    clmul_kernel! {
        fn load2(row: &[u64], width: usize, j: usize) -> [V; 2] {
            std::array::from_fn(|k| V::load(row, k * width + j))
        }

        fn store2(row: &mut [u64], width: usize, j: usize, value: [V; 2]) {
            for (k, v) in value.into_iter().enumerate() {
                V::store(row, k * width + j, v);
            }
        }

        fn xor2(a: &[V; 2], b: &[V; 2]) -> [V; 2] {
            std::array::from_fn(|k| a[k].xor(b[k]))
        }

        fn sums(a: &[V; 2]) -> V {
            a[0].xor(a[1])
        }

        /// The three Karatsuba products as sums by word offset, for even and odd entries.
        fn karatsuba(a: &[V; 2], sa: &V, b: &[V; 2], sb: &V) -> [[V; 3]; 2] {
            let product = |x: V, y: V| [x.clmul::<0x00>(y), x.clmul::<0x11>(y)];
            let (p0, p1, middle) = (product(a[0], b[0]), product(a[1], b[1]), product(*sa, *sb));
            std::array::from_fn(|h| [p0[h], middle[h].xor(p0[h]).xor(p1[h]), p1[h]])
        }

        fn partial(a: &[V; 2], b: &[V; 2]) -> [[V; 3]; 2] {
            karatsuba(a, &sums(a), b, &sums(b))
        }

        /// Word `w` of a product from its partial sums by offset, in the low word of each lane.
        fn word(sums: &[V; 3], w: usize) -> V {
            let low = if w < 3 { sums[w] } else { V::zero() };
            let high = if w > 0 { sums[w - 1].high() } else { V::zero() };
            low.xor(high)
        }

        /// Reduce the products given by their words in the low word of each lane.
        fn reduce(field: &PowerField, p: &[V; 4]) -> [V; 2] {
            let barrett: [V; 2] = std::array::from_fn(|k| V::splat(field.barrett[k]));
            let modulus: [V; 2] = std::array::from_fn(|k| V::splat(field.modulus[k]));
            let high = [p[2], p[3]];
            // The top two words of `high * barrett`: sums at offsets 1 and 2.
            let mut t = [V::zero(); 3];
            for (i, k) in [(0, 1), (1, 0), (1, 1)] {
                t[i + k] = t[i + k].xor(high[i].clmul::<0x00>(barrett[k]));
            }
            let q: [V; 2] = std::array::from_fn(|k| high[k].xor(word(&t, 2 + k)));
            // The low two words of `q * modulus`: sums at offsets 0 and 1.
            let mut r = [V::zero(); 3];
            for (i, k) in [(0, 0), (0, 1), (1, 0)] {
                r[i + k] = r[i + k].xor(q[i].clmul::<0x00>(modulus[k]));
            }
            std::array::from_fn(|k| p[k].xor(word(&r, k)))
        }

        /// [`reduce`] of products of three words, whose high half is their word 2 alone: most of
        /// the products vanish.
        #[cfg(any(test, not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni"))))]
        fn reduce3(field: &PowerField, p: &[V; 3]) -> [V; 2] {
            let q = p[2].xor(p[2].clmul::<0x00>(V::splat(field.barrett[1])).high());
            let r: [V; 2] = std::array::from_fn(|k| q.clmul::<0x00>(V::splat(field.modulus[k])));
            std::array::from_fn(|k| word(&[r[0], r[1], V::zero()], k).xor(p[k]))
        }

        /// The even and the odd entries' reductions, interleaved back.
        fn reduce_pair(field: &PowerField, sums: &[[V; 3]; 2]) -> [V; 2] {
            let [even, odd] = [0, 1].map(|half| reduce(field, &std::array::from_fn(|w| word(&sums[half], w))));
            std::array::from_fn(|k| even[k].unpacklo(odd[k]))
        }

        pub fn dot_pair(a: [&[u64]; 2], b: [&[u64]; 2], width: usize) -> ([u64; 4], [u64; 4]) {
            let mut acc = [[V::zero(); 3]; 2];
            for j in (0..width).step_by(STEP) {
                let (a0, a1, b0, b1) = (
                    load2(a[0], width, j),
                    load2(a[1], width, j),
                    load2(b[0], width, j),
                    load2(b[1], width, j),
                );
                for (acc, p) in acc
                    .iter_mut()
                    .zip([partial(&a0, &b0), partial(&xor2(&a0, &a1), &xor2(&b0, &b1))])
                {
                    for s in 0..3 {
                        acc[s] = acc[s].xor(p[0][s].xor(p[1][s]));
                    }
                }
            }
            finish_dot(&acc)
        }

        fn finish_dot(acc: &[[V; 3]; 2]) -> ([u64; 4], [u64; 4]) {
            let finish = |acc: &[V; 3]| {
                let mut out = [0u64; 4];
                for (s, sum) in acc.iter().enumerate() {
                    let sum = sum.sum();
                    out[s] ^= sum as u64;
                    out[s + 1] ^= (sum >> 64) as u64;
                }
                out
            };
            (finish(&acc[0]), finish(&acc[1]))
        }

        pub fn weighted_dot(field: &PowerField, a: &[u64], b: &[u64], weights: &[u64], width: usize) -> [u64; 4] {
            let mut acc = [V::zero(); 3];
            for j in (0..width).step_by(STEP) {
                let weighted = reduce_pair(field, &partial(&load2(a, width, j), &load2(weights, width, j)));
                let p = partial(&weighted, &load2(b, width, j));
                for k in 0..3 {
                    acc[k] = acc[k].xor(p[0][k].xor(p[1][k]));
                }
            }
            finish_dot(&[acc, [V::zero(); 3]]).0
        }

        /// `B` stays unreduced, four words: its two halves each meet `T[j + 2]` in a Karatsuba
        /// product, which costs fewer products than reducing `B` first. The sums are reduced once.
        pub fn chi_dot_pair(
            field: &PowerField,
            t: [&[u64]; 2],
            weights: &[u64],
            constant: &[u64],
            kappa: &Element,
            scratch: &mut [u64],
        ) -> ([u64; 4], [u64; 4]) {
            let kappa: [V; 2] = std::array::from_fn(|k| V::splat(kappa[k]));
            // Sums by word offset: the low half's products at 0, the high half's at 2.
            let mut acc = [[V::zero(); 5]; 2];
            for lane in 0..BITS / 64 {
                let (weights, constant) = (&weights[FACTOR_ROW * lane..], &constant[FACTOR_ROW * lane..]);
                let at = along(64 * lane, 1);
                for (half, t) in t.into_iter().enumerate() {
                    let b = &mut scratch[2 * half * FACTOR_ROW..][..2 * FACTOR_ROW];
                    for z in (0..64).step_by(STEP) {
                        let (w, c, shifted) = (
                            load2(weights, 64, z),
                            load2(constant, 64, z),
                            xor2(&load2(t, BITS, at + z), &kappa),
                        );
                        let sums = partial(&w, &shifted);
                        let [even, odd] = [0, 1].map(|h| -> [V; 4] { std::array::from_fn(|k| word(&sums[h], k)) });
                        let low: [V; 2] = std::array::from_fn(|k| even[k].unpacklo(odd[k]));
                        let high: [V; 2] = std::array::from_fn(|k| even[2 + k].unpacklo(odd[2 + k]));
                        store2(b, 64, z, xor2(&c, &low));
                        store2(&mut b[FACTOR_ROW..], 64, z, high);
                    }
                }
                let at = along(64 * lane, 2);
                for z in (0..64).step_by(STEP) {
                    let b: [[V; 2]; 4] = std::array::from_fn(|i| load2(&scratch[i * FACTOR_ROW..], 64, z));
                    let (t0, t1) = (load2(t[0], BITS, at + z), load2(t[1], BITS, at + z));
                    let (dt, db) = (xor2(&t0, &t1), [xor2(&b[0], &b[2]), xor2(&b[1], &b[3])]);
                    for (acc, (a, t)) in acc.iter_mut().zip([([b[0], b[1]], t0), (db, dt)]) {
                        let (low, high) = (partial(&a[0], &t), partial(&a[1], &t));
                        for s in 0..3 {
                            acc[s] = acc[s].xor(low[0][s].xor(low[1][s]));
                            acc[s + 2] = acc[s + 2].xor(high[0][s].xor(high[1][s]));
                        }
                    }
                }
            }
            let finish = |acc: &[V; 5]| -> [u64; 4] {
                let mut wide = [0u64; 6];
                for (s, sum) in acc.iter().enumerate() {
                    let sum = sum.sum();
                    wide[s] ^= sum as u64;
                    wide[s + 1] ^= (sum >> 64) as u64;
                }
                let high = field.reduce(&wide[2..].try_into().unwrap());
                let r = field.reduce(&[wide[0], wide[1], high[0], high[1]]);
                [r[0], r[1], 0, 0]
            };
            (finish(&acc[0]), finish(&acc[1]))
        }

        #[cfg(any(test, not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni"))))]
        pub fn reduce_row(field: &PowerField, dst: &mut [u64], wide: &[u64], width: usize) {
            for j in (0..width).step_by(STEP) {
                let w: [V; 3] = std::array::from_fn(|k| V::load(wide, k * width + j));
                let [even, odd] = [0, 1].map(|half| {
                    let at = |x: V| if half == 0 { x } else { x.high() };
                    reduce3(field, &w.map(at))
                });
                store2(dst, width, j, std::array::from_fn(|k| even[k].unpacklo(odd[k])));
            }
        }

        pub fn mul_row(field: &PowerField, dst: &mut [u64], a: &[u64], b: &[u64], c: &[u64], width: usize) {
            for j in (0..width).step_by(STEP) {
                let (a, b, c) = (load2(a, width, j), load2(b, width, j), load2(c, width, j));
                let product = reduce_pair(field, &partial(&a, &b));
                store2(dst, width, j, xor2(&c, &product));
            }
        }

        pub fn fold_row(
            field: &PowerField,
            dst: &mut [u64],
            lo: &[u64],
            hi: &[u64],
            challenge: &Element,
            width: usize,
        ) {
            let challenge: [V; 2] = std::array::from_fn(|k| V::splat(challenge[k]));
            let challenge_sums = sums(&challenge);
            for j in (0..width).step_by(STEP) {
                let (l, h) = (load2(lo, width, j), load2(hi, width, j));
                let delta = xor2(&l, &h);
                let folded = reduce_pair(field, &karatsuba(&delta, &sums(&delta), &challenge, &challenge_sums));
                store2(dst, width, j, xor2(&l, &folded));
            }
        }

        pub fn fold_quad(
            field: &PowerField,
            dst: &mut [u64],
            rows: [&[u64]; 4],
            challenges: &[Element; 3],
            width: usize,
        ) {
            let challenges: [[V; 2]; 3] = challenges.map(|r| r.map(|x| V::splat(x)));
            for j in (0..width).step_by(STEP) {
                let [a, b, c, d] = rows.map(|row| load2(row, width, j));
                let dx = xor2(&a, &b);
                let dy = xor2(&a, &c);
                let dxy = xor2(&dx, &xor2(&c, &d));
                let mut wide = [[V::zero(); 3]; 2];
                for (delta, r) in [dx, dy, dxy].iter().zip(&challenges) {
                    let product = partial(delta, r);
                    for half in 0..2 {
                        for k in 0..3 {
                            wide[half][k] = wide[half][k].xor(product[half][k]);
                        }
                    }
                }
                let folded = reduce_pair(field, &wide);
                store2(dst, width, j, xor2(&a, &folded));
            }
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
mod neon {
    use core::arch::aarch64::*;

    use primitives::field::gf2_64::aarch64 as pm;

    // The field helpers are `unsafe fn`s that need `aes`, which this module's cfg enables.
    #[inline(always)]
    fn mul_lo(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        // SAFETY: `aes` is enabled at compile time.
        unsafe { pm::pmull_lo_asm(a, b) }
    }
    #[inline(always)]
    fn mul_hi(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        // SAFETY: `aes` is enabled at compile time.
        unsafe { pm::pmull_hi_asm(a, b) }
    }

    use primitives::{load_at, store_at};

    use super::*;

    /// Entries `j` and `j + 1` of a row, a lane each, by word.
    #[inline(always)]
    fn load2(row: &[u64], width: usize, j: usize) -> [uint64x2_t; 2] {
        std::array::from_fn(|k| load_at(row, k * width + j))
    }

    #[inline(always)]
    unsafe fn xor2(a: &[uint64x2_t; 2], b: &[uint64x2_t; 2]) -> [uint64x2_t; 2] {
        unsafe { std::array::from_fn(|k| veorq_u64(a[k], b[k])) }
    }

    /// The products `a[i] * b[k]` of the listed word pairs, of each lane, as 128-bit sums by
    /// offset `i + k`.
    #[inline(always)]
    unsafe fn partial<const N: usize>(
        a: &[uint64x2_t; 2],
        b: &[uint64x2_t; 2],
        terms: [(usize, usize); N],
    ) -> [[uint64x2_t; 3]; 2] {
        unsafe {
            let mut out = [[vdupq_n_u64(0); 3]; 2];
            for (i, k) in terms {
                out[0][i + k] = veorq_u64(out[0][i + k], mul_lo(a[i], b[k]));
                out[1][i + k] = veorq_u64(out[1][i + k], mul_hi(a[i], b[k]));
            }
            out
        }
    }

    const ALL: [(usize, usize); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];

    /// Word `w` of each lane's product: the low half of its sum `w` and the high half of its
    /// sum `w - 1`.
    #[inline(always)]
    unsafe fn word(sums: &[[uint64x2_t; 3]; 2], w: usize) -> uint64x2_t {
        unsafe {
            let low = if w < 3 {
                vzip1q_u64(sums[0][w], sums[1][w])
            } else {
                vdupq_n_u64(0)
            };
            let high = if w > 0 {
                vzip2q_u64(sums[0][w - 1], sums[1][w - 1])
            } else {
                vdupq_n_u64(0)
            };
            veorq_u64(low, high)
        }
    }

    /// Reduce two products, a lane each, given by their four words.
    #[inline(always)]
    unsafe fn reduce(field: &PowerField, p: &[uint64x2_t; 4]) -> [uint64x2_t; 2] {
        unsafe {
            let barrett: [uint64x2_t; 2] = std::array::from_fn(|k| vdupq_n_u64(field.barrett[k]));
            let modulus: [uint64x2_t; 2] = std::array::from_fn(|k| vdupq_n_u64(field.modulus[k]));
            let high = [p[2], p[3]];
            // The top two words of `high * barrett`, then the low two of `q * modulus`.
            let t = partial(&high, &barrett, [(0, 1), (1, 0), (1, 1)]);
            let q: [uint64x2_t; 2] = std::array::from_fn(|k| veorq_u64(high[k], word(&t, 2 + k)));
            let r = partial(&q, &modulus, [(0, 0), (0, 1), (1, 0)]);
            std::array::from_fn(|k| veorq_u64(p[k], word(&r, k)))
        }
    }

    #[inline(always)]
    fn store2(row: &mut [u64], width: usize, j: usize, value: &[uint64x2_t; 2]) {
        for (k, &v) in value.iter().enumerate() {
            store_at(row, k * width + j, v);
        }
    }

    #[inline]
    pub unsafe fn weighted_dot(field: &PowerField, a: &[u64], b: &[u64], weights: &[u64], width: usize) -> [u64; 4] {
        unsafe {
            let mut acc = [vdupq_n_u64(0); 3];
            for j in (0..width).step_by(2) {
                let p = partial(&load2(a, width, j), &load2(weights, width, j), ALL);
                let weighted = reduce(field, &std::array::from_fn(|w| word(&p, w)));
                let p = partial(&weighted, &load2(b, width, j), ALL);
                for k in 0..3 {
                    acc[k] = veorq_u64(acc[k], veorq_u64(p[0][k], p[1][k]));
                }
            }
            let mut out = [0; 4];
            for (k, &sum) in acc.iter().enumerate() {
                out[k] ^= vgetq_lane_u64::<0>(sum);
                out[k + 1] ^= vgetq_lane_u64::<1>(sum);
            }
            out
        }
    }

    #[inline]
    pub unsafe fn dot_pair(a: [&[u64]; 2], b: [&[u64]; 2], width: usize) -> ([u64; 4], [u64; 4]) {
        unsafe {
            let mut acc = [[vdupq_n_u64(0); 3]; 2];
            for j in (0..width).step_by(2) {
                let (a0, a1, b0, b1) = (
                    load2(a[0], width, j),
                    load2(a[1], width, j),
                    load2(b[0], width, j),
                    load2(b[1], width, j),
                );
                for (acc, p) in acc
                    .iter_mut()
                    .zip([partial(&a0, &b0, ALL), partial(&xor2(&a0, &a1), &xor2(&b0, &b1), ALL)])
                {
                    for s in 0..3 {
                        acc[s] = veorq_u64(acc[s], veorq_u64(p[0][s], p[1][s]));
                    }
                }
            }
            let finish = |acc: &[uint64x2_t; 3]| {
                let mut out = [0u64; 4];
                for (s, &sum) in acc.iter().enumerate() {
                    out[s] ^= vgetq_lane_u64(sum, 0);
                    out[s + 1] ^= vgetq_lane_u64(sum, 1);
                }
                out
            };
            (finish(&acc[0]), finish(&acc[1]))
        }
    }

    #[inline]
    pub unsafe fn reduce_row(field: &PowerField, dst: &mut [u64], wide: &[u64], width: usize) {
        unsafe {
            let zero = vdupq_n_u64(0);
            for j in (0..width).step_by(2) {
                let w: [uint64x2_t; 3] = std::array::from_fn(|k| load_at(wide, k * width + j));
                store2(dst, width, j, &reduce(field, &[w[0], w[1], w[2], zero]));
            }
        }
    }

    #[inline]
    pub unsafe fn mul_row(field: &PowerField, dst: &mut [u64], a: &[u64], b: &[u64], c: &[u64], width: usize) {
        unsafe {
            for j in (0..width).step_by(2) {
                let sums = partial(&load2(a, width, j), &load2(b, width, j), ALL);
                let product = reduce(field, &std::array::from_fn(|w| word(&sums, w)));
                store2(dst, width, j, &xor2(&product, &load2(c, width, j)));
            }
        }
    }

    #[inline]
    pub unsafe fn fold_row(
        field: &PowerField,
        dst: &mut [u64],
        lo: &[u64],
        hi: &[u64],
        challenge: &Element,
        width: usize,
    ) {
        unsafe {
            let challenge: [uint64x2_t; 2] = std::array::from_fn(|k| vdupq_n_u64(challenge[k]));
            for j in (0..width).step_by(2) {
                let l = load2(lo, width, j);
                let sums = partial(&xor2(&l, &load2(hi, width, j)), &challenge, ALL);
                let folded = reduce(field, &std::array::from_fn(|w| word(&sums, w)));
                store2(dst, width, j, &xor2(&l, &folded));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use primitives::test_rng::Rng;

    #[test]
    fn two_round_fold_matches_successive_folds() {
        let mut rng = Rng::new(45);
        let field = PowerField::new(rng.ext()).unwrap();
        for width in [7, 16, BITS] {
            let rows: [Vec<u64>; 4] = std::array::from_fn(|_| (0..2 * width).map(|_| rng.next_u64()).collect());
            let [x, y] = [rng.ext(), rng.ext()].map(|r| field.to_poly(r));
            let mut actual = vec![0; 2 * width];
            let mut expected = actual.clone();
            let mut low = actual.clone();
            let mut high = actual.clone();
            for input in [rows.each_ref().map(Vec::as_slice), [&rows[0][..]; 4]] {
                field.fold_row_portable(&mut low, input[0], input[1], &x, width);
                field.fold_row_portable(&mut high, input[2], input[3], &x, width);
                field.fold_row_portable(&mut expected, &low, &high, &y, width);
                field.fold_quad(&mut actual, input, &[x, y, field.mul(&x, &y)], width);
                assert_eq!(actual, expected, "width {width}");
            }
        }
    }

    #[test]
    fn chi_factors_match_separate_row_operations() {
        let mut rng = Rng::new(43);
        let field = PowerField::new(rng.ext()).unwrap();
        let rows: [Vec<u64>; 4] = std::array::from_fn(|_| (0..2 * BITS).map(|_| rng.next_u64()).collect());
        let kappa = std::array::from_fn(|_| rng.next_u64());
        let mut scratch = vec![0; 4 * FACTOR_ROW];
        for high in [&rows[1], &rows[0]] {
            let t = [rows[0].as_slice(), high.as_slice()];
            let expected = field.chi_dot_pair_by_lane(t, &rows[2], &rows[3], &kappa, &mut scratch);
            let actual = field.chi_dot_pair(t, &rows[2], &rows[3], &kappa, &mut scratch);
            // The kernels may leave the sums reduced to different degrees.
            assert_eq!(field.reduce(&actual.0), field.reduce(&expected.0));
            assert_eq!(field.reduce(&actual.1), field.reduce(&expected.1));
        }
    }

    #[test]
    fn arithmetic_matches_the_standard_basis() {
        let mut rng = Rng::new(41);
        let s = rng.ext();
        let field = PowerField::new(s).unwrap();
        let width = 16;
        let elements = |rng: &mut Rng| -> (Vec<F128>, Vec<u64>) {
            let values = rng.ext_vec(width);
            let mut row = vec![0u64; 2 * width];
            for (j, &v) in values.iter().enumerate() {
                set(&mut row, width, j, &field.to_poly(v));
            }
            (values, row)
        };
        let ((a0, ra0), (a1, ra1), (b0, rb0), (b1, rb1)) = (
            elements(&mut rng),
            elements(&mut rng),
            elements(&mut rng),
            elements(&mut rng),
        );
        assert_eq!(
            field.to_field(&field.mul(&get(&ra0, width, 3), &get(&rb0, width, 5))),
            a0[3] * b0[5]
        );

        let (zero, infinity) = field.dot_pair([&ra0, &ra1], [&rb0, &rb1], width);
        assert_eq!((zero, infinity), dot_pair_portable([&ra0, &ra1], [&rb0, &rb1], width));
        let expected = (0..width).map(|j| a0[j] * b0[j]).sum();
        assert_eq!(field.to_field(&field.reduce(&zero)), expected);
        let expected = (0..width).map(|j| (a0[j] + a1[j]) * (b0[j] + b1[j])).sum();
        assert_eq!(field.to_field(&field.reduce(&infinity)), expected);

        let weighted = field.weighted_dot(&ra0, &rb0, &ra1, width);
        assert_eq!(weighted, field.weighted_dot_portable(&ra0, &rb0, &ra1, width));
        let expected = (0..width).fold(F128::ZERO, |acc, j| acc + a0[j] * b0[j] * a1[j]);
        assert_eq!(field.to_field(&field.reduce(&weighted)), expected);

        let challenge = rng.ext();
        let (mut fast, mut slow) = (vec![0u64; 2 * width], vec![0u64; 2 * width]);
        field.fold_row(&mut fast, &ra0, &ra1, &field.to_poly(challenge), width);
        field.fold_row_portable(&mut slow, &ra0, &ra1, &field.to_poly(challenge), width);
        assert_eq!(fast, slow);
        for j in 0..width {
            assert_eq!(
                field.to_field(&get(&fast, width, j)),
                primitives::multilinear::interp(a0[j], a1[j], challenge)
            );
        }
        let wide: Vec<u64> = (0..3 * width).map(|_| rng.next_u64()).collect();
        field.reduce_row(&mut fast, &wide, width);
        field.reduce_row_portable(&mut slow, &wide, width);
        assert_eq!(fast, slow);
        let eval = PolyEval::new(s, 24);
        for j in 0..width {
            let p = [wide[j], wide[width + j], wide[2 * width + j]];
            assert_eq!(field.to_field(&get(&fast, width, j)), eval.eval(&p));
        }
        field.mul_row(&mut fast, &ra0, &rb0, &rb1, width);
        field.mul_row_portable(&mut slow, &ra0, &rb0, &rb1, width);
        assert_eq!(fast, slow);
        for j in 0..width {
            assert_eq!(field.to_field(&get(&fast, width, j)), a0[j] * b0[j] + b1[j]);
        }
    }
}
