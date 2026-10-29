//! The layer prover.
//!
//! One layer makes four passes over the bits of the states it reduces to:
//!
//! 1. the product polynomial, before its evaluation point `s` is known;
//! 2. three sumcheck rounds on bit-level products, keeping the next variable binary;
//! 3. the next round's quadratic coefficient, from differences of groups;
//! 4. one field element per half block and position, for the rounds that remain.
//!
//! From `s` on, everything stays in the power basis of `s` ([`crate::power`]) until the tables
//! are down to one row.

use fiat_shamir::transcript::ProverState;
use primitives::field::{F192, F192Unreduced};
use primitives::multilinear::{eq_table, poly_eval};

use crate::basis::{PolyEval, add_mul};
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni")))]
use crate::{basis::carry, kernels::{GROUPS, fold_groups}};
use crate::keccak::{BITS, LOCAL_VARS, along, linear_transpose, row_positions};
use crate::kernels::{
    GROUP, PACK_VARS, PACKS, SKIP, SkipBlock, SkipWeights, TERNARY, embed, mixed_points, products,
    slot, used,
};
use crate::open::fold;
use crate::power::{Element, FACTOR_ROW, PowerField, get, set};
use crate::protocol::{Claim, chi_weights, mu_at, sum};
use crate::rmfe::PROD_BITS;
use crate::witness::{Recompute, Shape, Witness, words};
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni")))]
use crate::witness::transpose64;

/// The pattern index of a binary assignment.
fn ternary(b: usize) -> usize {
    (0..SKIP).fold(0, |acc, k| acc + (b >> k & 1) * 3usize.pow(k as u32))
}

/// Digits a weight splits into, and the values of one.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni")))]
const DIGIT_BITS: usize = 8;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni")))]
const DIGITS: usize = 192 / DIGIT_BITS;
/// Values of half a digit.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni")))]
const NIBBLE: usize = 1 << (DIGIT_BITS / 2);
/// The bits of a block position's products, 48 to each pack.
const PAYLOAD_BITS: usize = 48 * PACKS;
/// The packs of a part of a block position, and the words of their products: with AVX-512 the
/// whole position, three registers; with AVX2 half, three registers; elsewhere a quarter, a
/// block's worth of which stays in cache.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
const PART: usize = PACKS;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
const PART: usize = PACKS / 2;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
const PART: usize = 8;
const PART_WORDS: usize = 48 * PART / 64;
// `pack` reads four products to three words and `add` XORs whole vectors, so `PART` divides by four
// and `PART_WORDS` by each arm's lane count. A `PART` that broke either would drop products or store
// past the array rather than fail to build.
const _: () = assert!(PART.is_multiple_of(4));
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
const _: () = assert!(PART_WORDS.is_multiple_of(8));
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
const _: () = assert!(PART_WORDS.is_multiple_of(4));
#[cfg(target_arch = "aarch64")]
const _: () = assert!(PART_WORDS.is_multiple_of(2));

/// The products of a part of a block position's packs, 48 bits each, grouped while every digit
/// of the weights groups the block's positions.
#[derive(Clone, Copy)]
#[cfg_attr(all(target_arch = "x86_64", target_feature = "avx512f"), repr(C, align(64)))]
#[cfg_attr(not(all(target_arch = "x86_64", target_feature = "avx512f")), repr(C, align(16)))]
struct Part([u64; PART_WORDS]);

impl Part {
    const ZERO: Self = Self([0; PART_WORDS]);

    /// Product `q` at bits `48 * q ..`: four products to three words.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi"))]
    #[inline(always)]
    fn pack(products: &[u64; PART]) -> Self {
        // SAFETY: AVX-512 VBMI is enabled at compile time.
        unsafe { pack_vbmi(products) }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi")))]
    #[inline(always)]
    fn pack(products: &[u64; PART]) -> Self {
        let mut out = [0u64; PART_WORDS];
        for (out, p) in out.as_chunks_mut::<3>().0.iter_mut().zip(products.as_chunks::<4>().0) {
            *out = [p[0] | p[1] << 48, p[1] >> 16 | p[2] << 32, p[2] >> 32 | p[3] << 16];
        }
        Self(out)
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni")))]
    #[inline(always)]
    fn add(&mut self, other: &Self) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        for i in 0..PART_WORDS / 8 {
            use primitives::simd::{load, store, xor};
            let x = xor(load(&self.0, i), load(&other.0, i));
            store(&mut self.0, i, x);
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
        // SAFETY: AVX2 is enabled at compile time.
        unsafe {
            add_avx2(&mut self.0, &other.0)
        }
        // Left to itself, the compiler XORs the words in general registers.
        #[cfg(target_arch = "aarch64")]
        // SAFETY: both parts are six words, and aarch64 always has NEON.
        unsafe {
            use core::arch::aarch64::*;
            let (dst, src) = (self.0.as_mut_ptr(), other.0.as_ptr());
            for i in (0..PART_WORDS).step_by(2) {
                vst1q_u64(dst.add(i), veorq_u64(vld1q_u64(dst.add(i)), vld1q_u64(src.add(i))));
            }
        }
        #[cfg(not(any(all(target_arch = "x86_64", target_feature = "avx2"), target_arch = "aarch64")))]
        for (x, y) in self.0.iter_mut().zip(&other.0) {
            *x ^= y;
        }
    }
}

/// [`Part::pack`] with AVX-512 VBMI.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
#[inline]
fn pack_vbmi(products: &[u64; PART]) -> Part {
    use core::arch::x86_64::*;
    use primitives::simd::{load, store};
    /// Byte `j` of output register `k` is byte `j % 6` of product `(64 * k + j) / 6`, which lies
    /// in register `k` or `k + 1` of the products, eight to a register.
    const BYTES: [[u8; 64]; 3] = {
        let mut idx = [[0u8; 64]; 3];
        let mut k = 0;
        while k < 3 {
            let mut j = 0;
            while j < 64 {
                let (q, b) = ((64 * k + j) / 6, (64 * k + j) % 6);
                idx[k][j] = ((q - 8 * k) * 8 + b) as u8;
                j += 1;
            }
            k += 1;
        }
        idx
    };
    // The products are four registers and the parts three.
    let p: [__m512i; 4] = std::array::from_fn(|i| load(&products[..], i));
    let mut out = Part([0; PART_WORDS]);
    for k in 0..3 {
        store(&mut out.0, k, _mm512_permutex2var_epi8(p[k], load(&BYTES[k][..], 0), p[k + 1]));
    }
    out
}

/// [`Part::add`] with AVX2.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
#[target_feature(enable = "avx2")]
#[inline]
fn add_avx2(dst: &mut [u64; PART_WORDS], src: &[u64; PART_WORDS]) {
    use core::arch::x86_64::*;
    use primitives::{load_at, store_at};
    for i in (0..PART_WORDS).step_by(4) {
        let (d, s): (__m256i, __m256i) = (load_at(&dst[..], i), load_at(&src[..], i));
        store_at(&mut dst[..], i, _mm256_xor_si256(d, s));
    }
}

/// Digit `d` of a weight.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni")))]
#[inline(always)]
fn digit(weight: F192, d: usize) -> usize {
    let w = [weight.c0, weight.c1, weight.c2];
    (w[d / 8] >> (DIGIT_BITS * (d % 8))) as usize & ((1 << DIGIT_BITS) - 1)
}

/// The positions grouped by each digit of their weight: those whose digit `d` is `v` are
/// `order[d][starts[d][v]..starts[d][v + 1]]`.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni")))]
struct Groups {
    order: Vec<[u16; BITS]>,
    starts: Vec<[u16; (1 << DIGIT_BITS) + 1]>,
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni")))]
impl Groups {
    fn new(weights: &[F192]) -> Self {
        let mut order = vec![[0u16; BITS]; DIGITS];
        let mut starts = vec![[0u16; (1 << DIGIT_BITS) + 1]; DIGITS];
        for (d, (order, starts)) in order.iter_mut().zip(&mut starts).enumerate() {
            let digits: [usize; BITS] = std::array::from_fn(|j| digit(weights[j], d));
            for &v in &digits {
                starts[v + 1] += 1;
            }
            for v in 0..1 << DIGIT_BITS {
                starts[v + 1] += starts[v];
            }
            let mut next = *starts;
            for (p, &v) in digits.iter().enumerate() {
                order[next[v] as usize] = p as u16;
                next[v] += 1;
            }
        }
        Self { order, starts }
    }

    /// `sum over positions of weight[j] (x) payloads[j]` into `rows`: row `DIGIT_BITS * d + i`
    /// sums the payloads whose weight has that bit set. Each group is summed in registers, so
    /// the payloads are only read.
    fn add_rows(&self, payloads: &[Part; BITS], rows: &mut [Part; 192]) {
        for (rows, (order, starts)) in rows
            .as_chunks_mut::<DIGIT_BITS>()
            .0
            .iter_mut()
            .zip(self.order.iter().zip(&self.starts))
        {
            // The digit is a high and a low nibble. `low[l]` sums the groups whose value ends in `l`,
            // and `high[h]` those that begin with `h`; every row is a sum of one or the other.
            let mut low = [Part::ZERO; NIBBLE];
            let mut high = [Part::ZERO; NIBBLE];
            for (h, high) in high.iter_mut().enumerate() {
                let mut sum = Part::ZERO;
                for (l, low) in low.iter_mut().enumerate() {
                    let v = NIBBLE * h + l;
                    let mut acc = Part::ZERO;
                    for &p in &order[starts[v] as usize..starts[v + 1] as usize] {
                        acc.add(&payloads[p as usize]);
                    }
                    low.add(&acc);
                    sum.add(&acc);
                }
                *high = sum;
            }
            for i in 0..DIGIT_BITS / 2 {
                let (mut by_low, mut by_high) = (Part::ZERO, Part::ZERO);
                for x in (0..NIBBLE).filter(|x| x >> i & 1 == 1) {
                    by_low.add(&low[x]);
                    by_high.add(&high[x]);
                }
                rows[i] = by_low;
                rows[DIGIT_BITS / 2 + i] = by_high;
            }
        }
    }
}

/// A block's payloads summed over the positions, each weighted by its position's weight, then
/// by the block's weight: [`Groups`] of the weights' digits, rows of bits, then field elements.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni")))]
struct Weigh(Groups);

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni")))]
impl Weigh {
    fn new(weights: &[F192]) -> Self {
        Self(Groups::new(weights))
    }

    fn scratch() -> Vec<[Part; 192]> {
        vec![[Part::ZERO; 192]; PACKS / PART]
    }

    fn add(&self, payloads: &[[Part; BITS]], rows: &mut [[Part; 192]], weight: F192, sums: &mut [F192Unreduced]) {
        for (payloads, rows) in payloads.iter().zip(rows.iter_mut()) {
            self.0.add_rows(payloads, rows);
        }
        // Row `r` holds bit `r` of each payload bit's weighted sum: transposed, they are
        // field elements, which the block's weight multiplies.
        let mut columns = [[0u64; 64]; 3];
        for m in 0..PAYLOAD_BITS / 64 {
            let (q, w) = (m / PART_WORDS, m % PART_WORDS);
            for (k, column) in columns.iter_mut().enumerate() {
                for (b, word) in column.iter_mut().enumerate() {
                    *word = rows[q][64 * k + b].0[w];
                }
                transpose64(column);
            }
            for (t, sum) in sums[64 * m..][..64].iter_mut().enumerate() {
                *sum ^= weight.mul_unreduced(F192::new(columns[0][t], columns[1][t], columns[2][t]));
            }
        }
    }
}

/// [`Weigh`] as a product of bit matrices over GF(2), eight positions at a time: the payload bits
/// of eight positions are a byte, which a GF2P8AFFINE matrix of their weights' bits sends to one
/// byte of the weighted sum. The payloads are read once, where digit groups read them per digit.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni"))]
struct Weigh(Vec<[u64; 24]>);

/// Chunks of eight positions, and registers of a payload.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni"))]
const CHUNKS: usize = BITS / 8;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni"))]
impl Weigh {
    /// Matrix `r` of chunk `c` takes bit `k` of an input byte from position `8c + k`, and makes
    /// output bit `t` from bit `8r + t` of each one's weight. Byte `7 - t` is that output's row.
    fn new(weights: &[F192]) -> Self {
        let matrices = weights
            .as_chunks::<8>()
            .0
            .iter()
            .map(|w| std::array::from_fn(|r| affine_matrix(std::array::from_fn(|k| byte(&w[k], r)))))
            .collect();
        Self(matrices)
    }

    /// The payload bits by bit, `PAYLOAD_BITS / 64` registers per chunk: byte `i` of register
    /// `CHUNKS * g + c` holds payload bit `64 g + i` of chunk `c`'s positions.
    fn scratch() -> Vec<[u64; 8]> {
        vec![[0; 8]; PAYLOAD_BITS / 64 * CHUNKS]
    }

    fn add(&self, payloads: &[[Part; BITS]], bits: &mut [[u64; 8]], weight: F192, sums: &mut [F192Unreduced]) {
        assert!(bits.len() == PAYLOAD_BITS / 64 * CHUNKS && sums.len() == PAYLOAD_BITS);
        // SAFETY: the features are enabled at compile time.
        unsafe { weigh_gfni(&self.0, &payloads[0], bits, weight, sums) }
    }
}

/// Byte `r` of an element's 24.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni"))]
pub(crate) fn byte(x: &F192, r: usize) -> u8 {
    ([x.c0, x.c1, x.c2][r / 8] >> (8 * (r % 8))) as u8
}

/// The GF2P8AFFINE matrix whose output bit `t` sums the input bits `k` with bit `t` of
/// `bytes[k]` set: that output's row is matrix byte `7 - t`, the 8 x 8 transpose reversed.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni"))]
pub(crate) fn affine_matrix(bytes: [u8; 8]) -> u64 {
    // Bit `8 r + c` moves to bit `8 c + r` (Hacker's Delight, section 7-3).
    let mut x = u64::from_le_bytes(bytes);
    let t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AA;
    x ^= t ^ (t << 7);
    let t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCC;
    x ^= t ^ (t << 14);
    let t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0;
    x ^= t ^ (t << 28);
    x.swap_bytes()
}

/// The 8 x 8 transpose of eight registers' words: word `k` of output `w` is word `w` of input `k`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni"))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) fn transpose8(a: [core::arch::x86_64::__m512i; 8]) -> [core::arch::x86_64::__m512i; 8] {
    use core::arch::x86_64::*;
    let idx = |v: [i64; 8]| _mm512_setr_epi64(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
    let mut out = a;
    // Swap 4 x 4, then 2 x 2, then 1 x 1 blocks.
    for (step, lo, hi) in [
        (4, [0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 6, 7, 12, 13, 14, 15]),
        (2, [0, 1, 8, 9, 4, 5, 12, 13], [2, 3, 10, 11, 6, 7, 14, 15]),
        (1, [0, 8, 2, 10, 4, 12, 6, 14], [1, 9, 3, 11, 5, 13, 7, 15]),
    ] {
        let prev = out;
        for i in (0..8).filter(|i| i & step == 0) {
            out[i] = _mm512_permutex2var_epi64(prev[i], idx(lo), prev[i + step]);
            out[i + step] = _mm512_permutex2var_epi64(prev[i], idx(hi), prev[i + step]);
        }
    }
    out
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
fn weigh_gfni(
    matrices: &[[u64; 24]],
    payloads: &[Part; BITS],
    bits: &mut [[u64; 8]],
    weight: F192,
    sums: &mut [F192Unreduced],
) {
    use core::arch::x86_64::*;
    use primitives::simd::{load, store};
    const G: usize = PAYLOAD_BITS / 64;
    // Byte `8b + 7 - k` of the output is byte `b` of word `k`, then byte `8b + t` bit `k` is its bit `t`.
    let spread: [u8; 64] = std::array::from_fn(|o| (8 * (7 - o % 8) + o / 8) as u8);
    let spread = load(&spread[..], 0);
    let unit = _mm512_set1_epi64(0x8040201008040201u64 as i64);
    for c in 0..CHUNKS {
        for v in 0..PART_WORDS / 8 {
            let words = transpose8(std::array::from_fn(|k| load(&payloads[8 * c + k].0, v)));
            for (w, x) in words.into_iter().enumerate() {
                let x = _mm512_gf2p8affine_epi64_epi8(unit, _mm512_permutexvar_epi8(spread, x), 0);
                store(&mut bits[CHUNKS * (8 * v + w) + c], 0, x);
            }
        }
    }
    // Byte `8i + r` of the output is byte `8r + i` of the input: an 8 x 8 byte transpose per word.
    let gather: [u8; 64] = std::array::from_fn(|o| (8 * (o % 8) + o / 8) as u8);
    let gather = load(&gather[..], 0);
    let mut limbs = [[0u64; 64]; 3];
    for g in 0..G {
        let mut acc = [_mm512_setzero_si512(); 24];
        let bits = &bits[CHUNKS * g..][..CHUNKS];
        for (pair, m) in bits.as_chunks::<2>().0.iter().zip(matrices.as_chunks::<2>().0) {
            let (x0, x1) = (load(&pair[0], 0), load(&pair[1], 0));
            for (r, acc) in acc.iter_mut().enumerate() {
                let a = _mm512_gf2p8affine_epi64_epi8(x0, _mm512_set1_epi64(m[0][r] as i64), 0);
                let b = _mm512_gf2p8affine_epi64_epi8(x1, _mm512_set1_epi64(m[1][r] as i64), 0);
                *acc = _mm512_ternarylogic_epi64(*acc, a, b, 0x96);
            }
        }
        // Byte `i` of `acc[r]` is byte `r` of payload bit `64 g + i`'s sum.
        for (limb, acc) in limbs.iter_mut().zip(acc.as_chunks::<8>().0) {
            for (m, x) in transpose8(*acc).into_iter().enumerate() {
                store(&mut limb[..], m, _mm512_permutexvar_epi8(gather, x));
            }
        }
        for (t, sum) in sums[64 * g..][..64].iter_mut().enumerate() {
            *sum ^= weight.mul_unreduced(F192::new(limbs[0][t], limbs[1][t], limbs[2][t]));
        }
    }
}

/// The tables' rows of a block: per row and position, the value at `s` of the row's `B / 2`
/// packs, folded. Each row's sums of products, four words, reduced. A row of 16 packs sums two
/// groups' folds, each at its half of the weights.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni")))]
struct Fold<const B: usize>(Vec<[[u64; GROUP]; 3]>);

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni")))]
impl<const B: usize> Fold<B> {
    fn new(_: &PowerField, fold: &[Element]) -> Self {
        assert!(fold.len() == B / 2 && (B / 2).is_multiple_of(GROUP));
        Self(
            fold.chunks_exact(GROUP)
                .map(|fold| std::array::from_fn(|k| std::array::from_fn(|q| fold[q][k])))
                .collect(),
        )
    }

    fn scratch() -> Vec<u64> {
        vec![0u64; 4 * GROUPS * BITS]
    }

    /// The first `rows.len() / ROW` rows.
    fn rows(&self, field: &PowerField, block: &[u64], rows: &mut [u64], wide: &mut [u64]) {
        let n = rows.len() / ROW;
        let parts = self.0.len();
        // The rows' unreduced sums, four words each and laid out by word.
        for j in 0..BITS {
            let e = embed(words(block, j));
            let folds: [[[u128; 3]; GROUPS]; 2] = std::array::from_fn(|t| {
                if t < parts { fold_groups(&e, &self.0[t]) } else { [[0; 3]; GROUPS] }
            });
            for (r, wide) in wide.as_chunks_mut::<{ 4 * BITS }>().0[..n].iter_mut().enumerate() {
                let sum: [u128; 3] =
                    std::array::from_fn(|k| (0..parts).fold(0, |acc, t| acc ^ folds[t][parts * r + t][k]));
                let sum: [u64; 4] = carry(&sum);
                for (k, &word) in sum.iter().enumerate() {
                    wide[k * BITS + j] = word;
                }
            }
        }
        for (t, wide) in rows.as_chunks_mut::<ROW>().0.iter_mut().zip(wide.as_chunks::<{ 4 * BITS }>().0) {
            field.reduce_row(t, wide, BITS);
        }
    }
}

/// [`Fold`] as a GF(2)-linear map of a row's `B` bytes to its value's 24: GF2P8AFFINE matrix
/// `[i][z]` lane `l` sends input byte `i` to output byte `8z + l`, eight positions at a time.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni"))]
struct Fold<const B: usize>([[[u64; 8]; 3]; B]);

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni"))]
impl<const B: usize> Fold<B> {
    fn new(field: &PowerField, fold: &[Element]) -> Self {
        assert!(fold.len() == B / 2);
        Self(std::array::from_fn(|i| {
            // Bit `k` of input byte `i` is bit `8 (i % 2) + k` of pack `i / 2`.
            let values: [F192; 8] = std::array::from_fn(|k| {
                let [c0, c1, c2] = field.mul(&fold[i / 2], &[crate::rmfe::RMFE.emb[8 * (i % 2) + k] as u64, 0, 0]);
                F192::new(c0, c1, c2)
            });
            std::array::from_fn(|z| std::array::from_fn(|l| affine_matrix(values.map(|v| byte(&v, 8 * z + l)))))
        }))
    }

    fn scratch() -> Vec<u64> {
        Vec::new()
    }

    fn rows(&self, _: &PowerField, block: &[u64], rows: &mut [u64], _: &mut [u64]) {
        assert!(block.len() >= BITS * 8 && rows.len().is_multiple_of(ROW) && rows.len() <= 64 / B * ROW);
        // SAFETY: the features are enabled at compile time.
        unsafe { fold_gfni::<B, BITS, false>(&self.0, block, rows) }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
fn fold_gfni<const B: usize, const WIDTH: usize, const DIFFERENCES: bool>(
    maps: &[[[u64; 8]; 3]; B],
    block: &[u64],
    rows: &mut [u64],
) {
    use core::arch::x86_64::*;
    use primitives::simd::{load, store};
    // Byte `8b + m` of the output is byte `8m + b` of the input: an 8 x 8 byte transpose per word.
    let gather: [u8; 64] = std::array::from_fn(|o| (8 * (o % 8) + o / 8) as u8);
    let gather = load(&gather[..], 0);
    let groups = rows.len() / (3 * WIDTH);
    for c in 0..WIDTH / 8 {
        // Word `b` of `bytes[w]` holds byte `b` of word `w` of the eight positions, one to a byte.
        let words = transpose8(std::array::from_fn(|m| load(&block[64 * c..], m)));
        let mut bytes = words.map(|w| _mm512_permutexvar_epi8(gather, w));
        if DIFFERENCES {
            bytes = [
                _mm512_xor_si512(bytes[0], bytes[2]),
                _mm512_xor_si512(bytes[1], bytes[3]),
                _mm512_xor_si512(bytes[4], bytes[6]),
                _mm512_xor_si512(bytes[5], bytes[7]),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
            ];
        }
        for g in 0..groups {
            let mut acc = [_mm512_setzero_si512(); 3];
            for (pair, maps) in maps.as_chunks::<2>().0.iter().enumerate() {
                let x: [__m512i; 2] = std::array::from_fn(|h| {
                    let i = B * g + 2 * pair + h;
                    _mm512_permutexvar_epi64(_mm512_set1_epi64((i % 8) as i64), bytes[i / 8])
                });
                for (z, acc) in acc.iter_mut().enumerate() {
                    let a = _mm512_gf2p8affine_epi64_epi8(x[0], load(&maps[0][z][..], 0), 0);
                    let b = _mm512_gf2p8affine_epi64_epi8(x[1], load(&maps[1][z][..], 0), 0);
                    *acc = _mm512_ternarylogic_epi64(*acc, a, b, 0x96);
                }
            }
            // Lane `l` byte `m` of `acc[z]` is byte `8z + l` of position `m`'s value.
            for (z, acc) in acc.into_iter().enumerate() {
                let at = if DIFFERENCES {
                    (c / 8) * FACTOR_ROW + z * 64 + (8 * c) % 64
                } else {
                    z * WIDTH + 8 * c
                };
                store(
                    &mut rows[g * 3 * WIDTH + at..][..8],
                    0,
                    _mm512_permutexvar_epi8(gather, acc),
                );
            }
        }
    }
}

/// The two differences of a block's adjacent groups, five lanes at a time, laid out by lane.
impl Fold<{ 2 * GROUP }> {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni"
    ))]
    fn difference_scratch() -> Vec<u64> {
        Vec::new()
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni"
    )))]
    fn difference_scratch() -> Vec<u64> {
        vec![0; 2 * 5 * 4 * 64]
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni"
    ))]
    fn differences(&self, _: &PowerField, block: &[u64], rows: &mut [u64], _: &mut [u64]) {
        assert!(block.len() >= 5 * 64 * 8 && rows.len() == 10 * FACTOR_ROW);
        // SAFETY: the features are enabled at compile time and both buffers are bounded above.
        unsafe { fold_gfni::<{ 2 * GROUP }, { 5 * 64 }, true>(&self.0, block, rows) }
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni"
    )))]
    fn differences(&self, field: &PowerField, block: &[u64], rows: &mut [u64], wide: &mut [u64]) {
        for x in 0..5 {
            for z in 0..64 {
                let w = words(block, 64 * x + z);
                let packed = [w[0] ^ w[2], w[1] ^ w[3], w[4] ^ w[6], w[5] ^ w[7], 0, 0, 0, 0];
                let folded = fold_groups(&embed(&packed), &self.0[0]);
                for (half, folded) in folded[..2].iter().enumerate() {
                    let at = (5 * half + x) * 4 * 64;
                    for (k, word) in carry::<3, 4>(folded).into_iter().enumerate() {
                        wide[at + k * 64 + z] = word;
                    }
                }
            }
        }
        for (row, wide) in rows
            .as_chunks_mut::<FACTOR_ROW>()
            .0
            .iter_mut()
            .zip(wide.as_chunks::<{ 4 * 64 }>().0)
        {
            field.reduce_row(row, wide, 64);
        }
    }
}

struct Layer<'a> {
    witness: &'a Witness,
    /// The round whose `chi` each segment's claim is on.
    rounds: Vec<usize>,
}

/// A row of a table in the power basis: three words per position, laid out by word.
const ROW: usize = 3 * BITS;

impl Layer<'_> {
    /// The blocks a segment's tasks walk: those written, then the padding block once, standing
    /// for every block after them.
    fn held(&self) -> usize {
        self.witness.held()
    }

    /// Task `held * segment + g`, as `(segment, g, block)`, recomputed into `scratch` if not stored.
    fn block<'b>(&'b self, scratch: &'b mut Recompute, task: usize) -> (usize, usize, &'b [u64]) {
        let (segment, g) = (task / self.held(), task % self.held());
        (segment, g, self.witness.state(scratch, self.rounds[segment], g))
    }

    fn n_tasks(&self) -> usize {
        self.rounds.len() * self.held()
    }

    /// Each task's weight in `eq`, a table over the blocks then the segment: the padding block's
    /// is the sum over the blocks it stands for.
    fn task_weights(&self, eq: &[F192]) -> Vec<F192> {
        let nb = self.witness.shape.n_blocks();
        (0..self.n_tasks())
            .map(|task| {
                let (segment, g) = (task / self.held(), task % self.held());
                let eq = &eq[nb * segment..][..nb];
                if g < self.witness.written() {
                    eq[g]
                } else {
                    sum(&eq[g..])
                }
            })
            .collect()
    }

    /// Per pack `q` of a block, the coefficients of
    /// `sum over blocks and positions of eq(point[PACK_VARS..], block) * weights[j] * product(q, j)`,
    /// where the product is `ones * u[j] + (ones + u[j + 1]) * u[j + 2]` on embeddings, the
    /// unprojected `chi`.
    ///
    /// Nothing here knows where the polynomials will be evaluated, so each product meets its
    /// weight bit by bit. The position weights are the same in every block, so a block's sum
    /// over positions is a product of bit matrices ([`Weigh`]); only that sum meets the block's
    /// weight, as field elements.
    fn pack_products(&self, claim: &Claim) -> Vec<[F192; PROD_BITS]> {
        let weights = self.task_weights(&eq_table(&claim.point[PACK_VARS..]));
        let weigh = Weigh::new(&claim.weights);
        let sums = parallel::map_reduce_with_state(
            self.n_tasks(),
            || (vec![[Part::ZERO; BITS]; PACKS / PART], Weigh::scratch(), Recompute::new()),
            || vec![F192Unreduced::ZERO; PAYLOAD_BITS],
            |(payloads, scratch, recompute), sums, task| {
                let (_, _, block) = self.block(recompute, task);
                for r in 0..5 {
                    for z in 0..64 {
                        let js = row_positions(r, z);
                        let e: [[u32; PACKS]; 5] = std::array::from_fn(|x| embed(words(block, js[x])));
                        for x in 0..5 {
                            let p = products(&e[x], &e[(x + 1) % 5], &e[(x + 2) % 5]);
                            for (q, part) in payloads.iter_mut().enumerate() {
                                part[js[x]] = Part::pack(p[PART * q..][..PART].try_into().unwrap());
                            }
                        }
                    }
                }
                weigh.add(payloads, scratch, weights[task], sums);
            },
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x ^= y;
                }
                a
            },
        );
        (0..PACKS)
            .map(|q| std::array::from_fn(|c| sums[48 * q + c].reduce()))
            .collect()
    }

    /// The layer's summand at `s`, extended over the packs of a group, keeping variable `SKIP`
    /// binary and summing over the later variables. Its two grids give the first `SKIP` round
    /// messages and the next round's values at zero and one.
    ///
    /// A point at infinity in some variable keeps only the product `u[j + 1] * u[j + 2]`, summed
    /// here. A binary one is a pack, whose sum `binary` already holds. The packs of the padding
    /// block are all equal, so their extension vanishes there.
    fn skip_sums(&self, claim: &Claim, field: &PowerField, s: F192, binary: &[[F192; GROUP]; 2]) -> [Vec<F192>; 2] {
        let weights = SkipWeights::new(claim.weights.iter().map(|&w| field.to_poly(w)).collect());
        let groups: Vec<Element> = eq_table(&claim.point[SKIP + 1..])
            .iter()
            .map(|&e| field.to_poly(e))
            .collect();
        let points: Vec<(usize, usize, usize)> = mixed_points()
            .into_iter()
            .enumerate()
            .filter(|&(_, p)| used(p))
            .map(|(i, p)| (i, p / TERNARY, p % TERNARY))
            .collect();
        let (nb, written) = (self.witness.shape.n_blocks(), self.witness.written());
        let mixed = parallel::map_reduce_with_state(
            self.rounds.len() * written,
            || (SkipBlock::new(), Recompute::new()),
            || std::array::from_fn::<_, 2, _>(|_| vec![[0u64; 7]; TERNARY]),
            |(skip, recompute), total, task| {
                let (segment, g) = (task / written, task % written);
                let (row, block) = (g + nb * segment, self.witness.state(recompute, self.rounds[segment], g));
                skip.run(block, &weights);
                for half in 0..2 {
                    for &(i, group, p) in &points {
                        add_mul(&mut total[group][p], &skip.get(slot(half, i)), &groups[2 * row + half]);
                    }
                }
            },
            |mut a, b| {
                for (a, b) in a.iter_mut().zip(b) {
                    for (x, y) in a.iter_mut().zip(b) {
                        for (x, y) in x.iter_mut().zip(y) {
                            *x ^= y;
                        }
                    }
                }
                a
            },
        );
        let eval = PolyEval::new(s, 56);
        std::array::from_fn(|side| {
            let mut sums = vec![F192::ZERO; TERNARY];
            for &(_, _, p) in &points {
                sums[p] = eval.eval(&mixed[side][p]);
            }
            for (q, &value) in binary[side].iter().enumerate() {
                sums[ternary(q)] = value;
            }
            sums
        })
    }

    /// The fourth round needs only its quadratic coefficient: the binary values rode `Skip`.
    fn group_infinity(&self, round: &Round, challenges: &[F192], point: &[F192]) -> F192 {
        let field = round.field;
        let fold: Vec<Element> = eq_table(challenges).iter().map(|&e| field.to_poly(e)).collect();
        let fold = Fold::<{ 2 * GROUP }>::new(field, &fold);
        let groups: Vec<Element> = eq_table(&point[SKIP + 1..]).iter().map(|&e| field.to_poly(e)).collect();
        let (nb, written) = (self.witness.shape.n_blocks(), self.witness.written());
        let sum = parallel::map_reduce_with_state(
            self.rounds.len() * written,
            || {
                (
                    vec![0u64; 10 * FACTOR_ROW],
                    Fold::<{ 2 * GROUP }>::difference_scratch(),
                    Recompute::new(),
                )
            },
            || [0u64; 9],
            |(rows, wide, recompute), total, task| {
                let (segment, g) = (task / written, task % written);
                let block = self.witness.state(recompute, self.rounds[segment], g);
                let mut sums = [[0u64; 6]; 2];
                for r in 0..5 {
                    fold.differences(field, &block[r * 5 * 64 * 8..], rows, wide);
                    for (half, sum) in sums.iter_mut().enumerate() {
                        let lanes = &rows[5 * half * FACTOR_ROW..][..5 * FACTOR_ROW];
                        for x in 0..5 {
                            let a = &lanes[(x + 1) % 5 * FACTOR_ROW..][..FACTOR_ROW];
                            let b = &lanes[(x + 2) % 5 * FACTOR_ROW..][..FACTOR_ROW];
                            let w = &round.weights[(5 * r + x) * FACTOR_ROW..][..FACTOR_ROW];
                            for (sum, value) in sum.iter_mut().zip(field.weighted_dot(a, b, w, 64)) {
                                *sum ^= value;
                            }
                        }
                    }
                }
                for (half, sum) in sums.iter().enumerate() {
                    add_mul(total, sum, &groups[2 * (nb * segment + g) + half]);
                }
            },
            |mut a, b| {
                for (a, b) in a.iter_mut().zip(b) {
                    *a ^= b;
                }
                a
            },
        );
        round.eval.eval(&sum)
    }

    /// The table the remaining rounds fold, a row per half block: `T[j]` folded over the half's
    /// packs at `challenges`. A block's two halves are the first of those rounds' pairs, so its
    /// round values are summed while the rows are at hand, weighted by `blocks`. The padding
    /// block's halves are one row, its segment's entry of `pads`.
    fn table(
        &self,
        round: &Round,
        challenges: &[F192],
        blocks: &[F192],
        layout: Rows,
        table: &mut [u64],
        pads: &mut [u64],
    ) -> (F192, F192) {
        let field = round.field;
        let fold: Vec<Element> = eq_table(challenges).iter().map(|&e| field.to_poly(e)).collect();
        let fold = Fold::<{ 4 * GROUP }>::new(field, &fold);
        let weights = self.task_weights(blocks);
        // Rows of segments that do not exist are zero, and so is their product.
        table[self.rounds.len() * layout.live * ROW..layout.len() * ROW].fill(0);
        pads[self.rounds.len() * ROW..].fill(0);
        // Task `held * segment + g` owns its rows: a written block's two, or its segment's padding row.
        let (held, written) = (self.held(), self.witness.written());
        let mut parts: Vec<&mut [u64]> = Vec::with_capacity(self.n_tasks());
        let mut rest = &mut table[..self.rounds.len() * layout.live * ROW];
        let mut pad_rows = &mut pads[..];
        for _ in 0..self.rounds.len() {
            let (segment, tail) = rest.split_at_mut(layout.live * ROW);
            let (pad, pad_tail) = pad_rows.split_at_mut(ROW);
            (rest, pad_rows) = (tail, pad_tail);
            parts.extend(segment.chunks_mut(2 * ROW));
            if held > written {
                parts.push(pad);
            }
        }
        let sums = parallel::map_reduce_chunks_mut(
            [&mut parts[..]],
            [1],
            || (vec![0u64; 4 * FACTOR_ROW], Fold::<{ 4 * GROUP }>::scratch(), Recompute::new()),
            RoundSums::default,
            |(scratch, wide, recompute), sums, task, [part]| {
                let (_, _, block) = self.block(recompute, task);
                let rows = &mut *part[0];
                fold.rows(field, block, rows, wide);
                round.add_pair(sums, scratch, &rows[..ROW], &rows[rows.len() - ROW..], weights[task]);
            },
            RoundSums::merge,
        );
        round.finish(&sums)
    }
}

/// The sums a round accumulates, unreduced: its values at `0` and at infinity.
#[derive(Clone, Copy, Default)]
struct RoundSums([u64; 9], [u64; 9]);

impl RoundSums {
    fn merge(mut self, other: Self) -> Self {
        for i in 0..9 {
            self.0[i] ^= other.0[i];
            self.1[i] ^= other.1[i];
        }
        self
    }
}

/// The position weights and constants `kappa * weights[j + 2]`, by lane, then by word.
struct Round<'a> {
    field: &'a PowerField,
    eval: PolyEval,
    kappa: Element,
    weights: Vec<u64>,
    constant: Vec<u64>,
}

impl<'a> Round<'a> {
    fn new(field: &'a PowerField, claim: &Claim, s: F192, kappa: F192) -> Self {
        let (mut weights, mut constant) = (vec![0u64; ROW], vec![0u64; ROW]);
        for j in 0..BITS {
            let at = FACTOR_ROW * (j / 64);
            set(
                &mut weights[at..][..FACTOR_ROW],
                64,
                j % 64,
                &field.to_poly(claim.weights[j]),
            );
            set(
                &mut constant[at..][..FACTOR_ROW],
                64,
                j % 64,
                &field.to_poly(kappa * claim.weights[along(j, 2)]),
            );
        }
        Self {
            field,
            eval: PolyEval::new(s, 72),
            kappa: field.to_poly(kappa),
            weights,
            constant,
        }
    }

    /// Add one pair of rows to a round's sums, weighted by `eq`.
    fn add_pair(&self, sums: &mut RoundSums, scratch: &mut [u64], low: &[u64], high: &[u64], eq: F192) {
        let (zero, infinity) =
            self.field
                .chi_dot_pair([low, high], &self.weights, &self.constant, &self.kappa, scratch);
        let weight = self.field.to_poly(eq);
        add_mul(&mut sums.0, &zero, &weight);
        add_mul(&mut sums.1, &infinity, &weight);
    }

    fn finish(&self, sums: &RoundSums) -> (F192, F192) {
        (self.eval.eval(&sums.0), self.eval.eval(&sums.1))
    }

    /// Fold the rows of `src`, laid out as `from`, in pairs into `dst`, and sum the next round's
    /// values over the pairs of what was just written, weighted by `eq`.
    fn fold(
        &self,
        src: &[u64],
        from: Rows,
        dst: &mut [u64],
        pads: &[u64],
        challenge: &Element,
        eq: &[F192],
    ) -> (F192, F192) {
        let to = from.next();
        let fold = |out: &mut [u64], segment: usize, r: usize| {
            let (low, high) = if from.m == 1 {
                (
                    from.row(src, pads, 2 * segment, 0),
                    from.row(src, pads, 2 * segment + 1, 0),
                )
            } else {
                (
                    from.row(src, pads, segment, 2 * r),
                    from.row(src, pads, segment, 2 * r + 1),
                )
            };
            self.field.fold_row(out, low, high, challenge, BITS);
        };
        if to.m == 1 {
            // A row per segment, few enough to fold in place.
            for (segment, out) in dst[..to.len() * ROW].as_chunks_mut::<ROW>().0.iter_mut().enumerate() {
                fold(out, segment, 0);
            }
            let mut sums = RoundSums::default();
            let mut scratch = vec![0u64; 4 * FACTOR_ROW];
            for (pair, &eq) in dst[..to.len() * ROW].as_chunks::<{ 2 * ROW }>().0.iter().zip(eq) {
                self.add_pair(&mut sums, &mut scratch, &pair[..ROW], &pair[ROW..], eq);
            }
            return self.finish(&sums);
        }
        // A task per pair of rows written, the second of a ragged one being the padding row.
        let (pairs, half) = (to.live.div_ceil(2), to.m / 2);
        // Task `pairs * segment + t` owns rows `2t` and `2t + 1` of its segment, the last one alone if it is ragged.
        // With no block written, no row is live and there are no parts.
        let mut parts: Vec<&mut [u64]> = dst[..to.len() * ROW]
            .chunks_mut(to.live.max(1) * ROW)
            .flat_map(|segment| segment.chunks_mut(2 * ROW))
            .collect();
        let mut sums = parallel::map_reduce_chunks_mut(
            [&mut parts[..]],
            [1],
            || vec![0u64; 4 * FACTOR_ROW],
            RoundSums::default,
            |scratch, sums, task, [part]| {
                let (segment, t) = (task / pairs, task % pairs);
                let rows = &mut *part[0];
                let n = rows.len() / ROW;
                for (i, row) in rows.as_chunks_mut::<ROW>().0.iter_mut().enumerate() {
                    fold(row, segment, 2 * t + i);
                }
                let high = if n == 2 {
                    &rows[ROW..]
                } else {
                    &pads[segment * ROW..][..ROW]
                };
                self.add_pair(sums, scratch, &rows[..ROW], high, eq[half * segment + t]);
            },
            RoundSums::merge,
        );
        self.add_padding(&mut sums, to, pads, eq);
        self.finish(&sums)
    }

    /// Add the pairs of two padding rows of a table laid out as `layout`, weighted by `eq`: each
    /// segment's padding row, once, at the sum of their weights.
    fn add_padding(&self, sums: &mut RoundSums, layout: Rows, pads: &[u64], eq: &[F192]) {
        let (first, half) = (layout.live.div_ceil(2), layout.m / 2);
        if first == half {
            return;
        }
        let mut scratch = vec![0u64; 4 * FACTOR_ROW];
        for (pad, eq) in pads.as_chunks::<ROW>().0.iter().zip(eq.chunks_exact(half)) {
            self.add_pair(sums, &mut scratch, pad, pad, sum(&eq[first..]));
        }
    }
}

/// The rows of the table the tail folds: per segment, the first `live` of `m`, then, if there are
/// fewer, the padding row every later one equals, kept apart. Once `m` is one, a row per segment.
#[derive(Clone, Copy)]
struct Rows {
    m: usize,
    live: usize,
    segments: usize,
}

impl Rows {
    fn len(&self) -> usize {
        self.segments * self.live
    }

    /// The layout once the next variable is bound: the group, the block's, then the segment's.
    fn next(&self) -> Self {
        match self.m {
            1 => Self {
                segments: self.segments / 2,
                ..*self
            },
            // The fold writes every segment's row, a padding one included.
            2 => Self { m: 1, live: 1, ..*self },
            m => Self {
                m: m / 2,
                live: self.live.div_ceil(2),
                ..*self
            },
        }
    }

    fn row<'t>(&self, table: &'t [u64], pads: &'t [u64], segment: usize, r: usize) -> &'t [u64] {
        if r < self.live {
            &table[(self.live * segment + r) * ROW..][..ROW]
        } else {
            &pads[segment * ROW..][..ROW]
        }
    }
}

/// A round of the sumcheck over bit positions, on small tables in the standard basis: the
/// coefficients of `sum_j w (kappa t0 + (kappa + t1) t2)` in the variable being bound.
fn local_round(w: &[F192], t: &[Vec<F192>; 3], kappa: F192) -> [F192; 4] {
    let mut h = [F192::ZERO; 4];
    for i in 0..w.len() / 2 {
        let line = |v: &[F192]| (v[2 * i], v[2 * i] + v[2 * i + 1]);
        let (w, t0, t1, t2) = (line(w), line(&t[0]), line(&t[1]), line(&t[2]));
        let u = (kappa + t1.0, t1.1);
        let a = [
            kappa * t0.0 + u.0 * t2.0,
            kappa * t0.1 + u.0 * t2.1 + u.1 * t2.0,
            u.1 * t2.1,
        ];
        h[0] += w.0 * a[0];
        h[1] += w.0 * a[1] + w.1 * a[0];
        h[2] += w.0 * a[2] + w.1 * a[1];
        h[3] += w.1 * a[2];
    }
    h
}

/// The table the rounds after the bit-level ones fold, and the half they fold it into: reused
/// from layer to layer.
pub struct Scratch {
    table: Vec<u64>,
    /// Each segment's padding row.
    pads: Vec<u64>,
}

impl Scratch {
    pub fn new(shape: Shape) -> Self {
        let rows = 1usize << (shape.y_vars() - PACK_VARS + 1);
        let table = vec![0u64; rows * ROW + rows * ROW / 2];
        let pads = vec![0u64; ROW << shape.seg_log()];
        Self { table, pads }
    }
}

/// Prove one layer, as `crates/verifier`'s `keccak.rs` checks it: a claim on the states `step`
/// layers below the outputs becomes one on the states a layer further down.
pub fn prove_layer(
    ps: &mut ProverState,
    witness: &Witness,
    scratch: &mut Scratch,
    step: usize,
    claim: &Claim,
) -> Claim {
    let shape = witness.shape;
    let layer = Layer {
        witness,
        rounds: (0..shape.segments())
            .map(|segment| shape.round(segment, step))
            .collect(),
    };

    let per_pack = tracing::info_span!("Product").in_scope(|| layer.pack_products(claim));
    let pack_weights = eq_table(&claim.point[..PACK_VARS]);
    let product: [F192; PROD_BITS] =
        std::array::from_fn(|c| (0..PACKS).fold(F192::ZERO, |acc, q| acc + pack_weights[q] * per_pack[q][c]));
    ps.add_scalars(&product);
    let s = ps.sample();
    let kappa = sum(&mu_at(s));
    let mut value = poly_eval(&product, s);
    let field = PowerField::new(s).expect("the challenge lies in no proper subfield");

    // Keep the fourth variable binary, summing over the block's halves.
    let binary: [[F192; GROUP]; 2] = std::array::from_fn(|side| {
        std::array::from_fn(|q| {
            let half = claim.point[SKIP + 1];
            let q = GROUP * side + q;
            (F192::ONE + half) * poly_eval(&per_pack[q], s) + half * poly_eval(&per_pack[2 * GROUP + q], s)
        })
    });
    let mut sums = tracing::info_span!("Skip").in_scope(|| layer.skip_sums(claim, &field, s, &binary));
    let side_weights = [F192::ONE + claim.point[SKIP], claim.point[SKIP]];
    let mut scale = F192::ONE;
    let mut point = Vec::with_capacity(claim.point.len());
    for i in 0..SKIP {
        let later = eq_table(&claim.point[i + 1..SKIP]);
        let mut h = [F192::ZERO; 3];
        for (b, &eq) in later.iter().enumerate() {
            for (d, h) in h.iter_mut().enumerate() {
                *h += eq
                    * (side_weights[0] * sums[0][d + 3 * ternary(b)] + side_weights[1] * sums[1][d + 3 * ternary(b)]);
            }
        }
        let coeffs = [scale * h[0], scale * (h[0] + h[1] + h[2]), scale * h[2]];
        let challenge = bind(ps, &coeffs, claim.point[i], &mut value, &mut scale, &mut point);
        sums = sums.map(|sums| {
            sums.as_chunks::<3>()
                .0
                .iter()
                .map(|v| v[0] + challenge * (v[0] + v[1]) + (challenge.square() + challenge) * v[2])
                .collect()
        });
    }

    let round = Round::new(&field, claim, s, kappa);
    let blocks = eq_table(&claim.point[PACK_VARS..]);
    let infinity = tracing::info_span!("Groups").in_scope(|| layer.group_infinity(&round, &point, &claim.point));
    let coeffs = [
        scale * sums[0][0],
        scale * (sums[0][0] + sums[1][0] + infinity),
        scale * infinity,
    ];
    bind(ps, &coeffs, claim.point[SKIP], &mut value, &mut scale, &mut point);

    // The blocks and segments, on a table in the power basis. Each pass over it writes the rows
    // of one round and sums that round's values over them.
    let mut layout = Rows {
        m: 2 * shape.n_blocks(),
        live: 2 * witness.written(),
        segments: 1 << shape.seg_log(),
    };
    let (mut src, mut dst) = scratch.table.split_at_mut(layout.m * layout.segments * ROW);
    let pads = &mut scratch.pads;
    let mut values = tracing::info_span!("Tables").in_scope(|| layer.table(&round, &point, &blocks, layout, src, pads));
    let tail = tracing::info_span!("Tail").entered();
    for i in PACK_VARS - 1..claim.point.len() {
        let coeffs = round_coefficients(scale, values.0, values.1, value, claim.point[i]);
        let challenge = bind(ps, &coeffs, claim.point[i], &mut value, &mut scale, &mut point);
        let later = eq_table(claim.point.get(i + 2..).unwrap_or(&[]));
        values = round.fold(src, layout, dst, pads, &field.to_poly(challenge), &later);
        layout = layout.next();
        (src, dst) = (dst, src);
    }
    let table = &src[..ROW];
    drop(tail);

    // The bit positions, in the standard basis.
    let cube = 1usize << LOCAL_VARS;
    let standard = |row: &[u64]| -> Vec<F192> {
        let mut out: Vec<F192> = (0..BITS).map(|j| field.to_field(&get(row, BITS, j))).collect();
        out.resize(cube, F192::ZERO);
        out
    };
    let t0 = standard(table);
    let shift = |k: usize| -> Vec<F192> {
        (0..cube)
            .map(|j| if j < BITS { t0[along(j, k)] } else { F192::ZERO })
            .collect()
    };
    let (t1, t2) = (shift(1), shift(2));
    let mut t = [t0, t1, t2];
    let mut w = claim.weights.clone();
    w.resize(cube, F192::ZERO);
    let mut local = Vec::with_capacity(LOCAL_VARS);
    for _ in 0..LOCAL_VARS {
        let coeffs = local_round(&w, &t, kappa).map(|c| scale * c);
        ps.add_round_poly(&coeffs, false);
        let challenge = ps.sample();
        local.push(challenge);
        w = fold(&w, 1, challenge);
        t = t.map(|v| fold(&v, 1, challenge));
    }
    let t = [t[0][0], t[1][0], t[2][0]];
    ps.add_scalars(&t);
    let beta = ps.sample();
    Claim {
        value: t[0] + beta * (t[1] + beta * t[2]),
        mu: mu_at(s),
        point,
        weights: linear_transpose(&chi_weights(&eq_table(&local), beta)),
    }
}

/// Send a round of the sumcheck over the packs and segments and bind its variable, whose point
/// in the claim is `at`: the claim's `eq` factor on it moves into `scale`.
fn bind(
    ps: &mut ProverState,
    coeffs: &[F192; 3],
    at: F192,
    value: &mut F192,
    scale: &mut F192,
    point: &mut Vec<F192>,
) -> F192 {
    ps.add_round_poly(coeffs, true);
    let challenge = ps.sample();
    let eq = F192::ONE + at + challenge;
    *value = eq * poly_eval(coeffs, challenge);
    *scale *= eq;
    point.push(challenge);
    challenge
}

/// The coefficients of a round polynomial `h`, scaled, from its values at `0` and infinity:
/// `(1 + r) h(0) + r h(1) = value` gives the value at one.
fn round_coefficients(scale: F192, zero: F192, infinity: F192, value: F192, r: F192) -> [F192; 3] {
    let (c0, c2) = (scale * zero, scale * infinity);
    // value = c0 + r (c1 + c2).
    let c1 = (value + c0) * r.inv() + c2;
    [c0, c1, c2]
}
