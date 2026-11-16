//! The grouped rounds' sums over a block by VPCLMULQDQ, a register of positions at a time (AVX2).
//!
//! [`skip_row`](crate::kernels::skip_row) takes a position at a time, from its embedded packs
//! extended to every point, in memory. Here a register holds one value of `2 * V::N` positions, so
//! the extension is a sum of registers, and each slot's products are a register's lanes, weighted by
//! those positions' weights at once. The points the fourth variable would take at infinity are
//! left out, which the grouped rounds do not read.

use core::arch::x86_64::*;

use primitives::field::gf2_64::mul_wide;
use primitives::hash::positions::row_positions;
use primitives::load_at;
use verifier::keccak::{BITS, PRODUCT_BITS};

use crate::clmul::{V, Vector};
use crate::kernels::{GROUP, MIXED, NIBBLES, slot};
use crate::witness::BLOCK_WORDS;

/// Positions in a register.
const WIDTH: usize = 2 * V::N;

// `unpacklo` keeps the low word of each product, so a product must fit in one word.
const _: () = assert!(PRODUCT_BITS <= 64);

/// The nibble tables of the embedding, in both 128-bit lanes.
type Tables = [__m256i; 12];

#[inline(always)]
fn tables() -> Tables {
    // SAFETY: AVX2 is enabled at compile time, and each table is 16 bytes.
    std::array::from_fn(|i| unsafe { _mm256_broadcastsi128_si256(load_at::<u8, __m128i>(&NIBBLES[i], 0)) })
}

/// The embedding of each pack of `v`: its low 16 bits, and its third byte in the low byte of a pack.
#[inline(always)]
fn embedding(t: &Tables, v: __m256i) -> (__m256i, __m256i) {
    // SAFETY: AVX2 is enabled at compile time.
    unsafe {
        let mask = _mm256_set1_epi8(0x0f);
        let (low_byte, high_byte) = (_mm256_set1_epi16(0x00ff), _mm256_set1_epi16(0xff00u16 as i16));
        // A pack's low byte holds its nibbles 0 and 1, its high byte nibbles 2 and 3.
        let nibbles = [
            _mm256_and_si256(v, mask),
            _mm256_and_si256(_mm256_srli_epi16::<4>(v), mask),
        ];
        // Byte `k` of the embedding of each byte, read as a pack's low byte or as its high one.
        let look = |k: usize, high: usize| {
            _mm256_xor_si256(
                _mm256_shuffle_epi8(t[4 * k + 2 * high], nibbles[0]),
                _mm256_shuffle_epi8(t[4 * k + 2 * high + 1], nibbles[1]),
            )
        };
        let low = _mm256_or_si256(
            _mm256_and_si256(
                _mm256_xor_si256(look(0, 0), _mm256_srli_epi16::<8>(look(0, 1))),
                low_byte,
            ),
            _mm256_and_si256(
                _mm256_xor_si256(_mm256_slli_epi16::<8>(look(1, 0)), look(1, 1)),
                high_byte,
            ),
        );
        let high = _mm256_and_si256(
            _mm256_xor_si256(look(2, 0), _mm256_srli_epi16::<8>(look(2, 1))),
            low_byte,
        );
        (low, high)
    }
}

/// One half's packs of four consecutive positions, embedded: `out[q]` holds the half's pack `q` of
/// position `i` in word `i`.
#[inline(always)]
fn embed_positions(t: &Tables, positions: &[u64], half: usize, out: &mut [V; GROUP]) {
    // SAFETY: AVX2 is enabled at compile time, and the loads and stores are bounds-checked.
    unsafe {
        // Word `w` of the half, of each of the four positions.
        let r: [__m256i; 4] = std::array::from_fn(|i| load_at(positions, BLOCK_WORDS * i + 4 * half));
        let (a0, a1) = (_mm256_unpacklo_epi64(r[0], r[1]), _mm256_unpackhi_epi64(r[0], r[1]));
        let (a2, a3) = (_mm256_unpacklo_epi64(r[2], r[3]), _mm256_unpackhi_epi64(r[2], r[3]));
        let words = [
            _mm256_permute2x128_si256::<0x20>(a0, a2),
            _mm256_permute2x128_si256::<0x20>(a1, a3),
            _mm256_permute2x128_si256::<0x31>(a0, a2),
            _mm256_permute2x128_si256::<0x31>(a1, a3),
        ];
        let zero = _mm256_setzero_si256();
        for (w, &v) in words.iter().enumerate() {
            let (low, high) = embedding(t, v);
            // Per lane, the four packs of its even word, then of its odd one.
            let (even, odd) = (_mm256_unpacklo_epi16(low, high), _mm256_unpackhi_epi16(low, high));
            let (first, second) = (_mm256_unpacklo_epi32(even, odd), _mm256_unpackhi_epi32(even, odd));
            out[4 * w] = _mm256_unpacklo_epi32(first, zero);
            out[4 * w + 1] = _mm256_unpackhi_epi32(first, zero);
            out[4 * w + 2] = _mm256_unpacklo_epi32(second, zero);
            out[4 * w + 3] = _mm256_unpackhi_epi32(second, zero);
        }
    }
}

/// One high-variable pair of [`crate::kernels::extend_group`], leaving the low pair binary.
#[inline(always)]
fn extend_high(v: &[V], high: usize) -> [V; 4] {
    let x = |a: [V; 4], b: [V; 4]| -> [V; 4] { std::array::from_fn(|l| a[l].xor(b[l])) };
    let q = |k: usize| -> [V; 4] { std::array::from_fn(|l| v[4 * k + l]) };
    match high {
        0..=3 => q(high),
        4 => x(q(0), q(1)),
        5 => x(q(2), q(3)),
        6 => x(q(0), q(2)),
        7 => x(q(1), q(3)),
        8 => x(x(q(0), q(1)), x(q(2), q(3))),
        _ => unreachable!(),
    }
}

/// The sums of a block at each slot, the position weights as elements.
///
/// A lane holds two positions, `2i` and `2i + 1`, with products `d0, d1` and weight words `w0, w1`.
/// Their weighted sum `w0 d0 + w1 d1` is `(w0 + d1) (w1 + d0) + w0 w1 + d0 d1`.
/// That is one product per weight word, and `d0 d1` once for both words.
/// `w0 w1` is the layer's, summed in `pairs`. It makes three products a pair of positions, where four were.
pub(crate) struct SkipBlock {
    /// Per slot, the sums of `(w0 + d1) (w1 + d0)` for each weight word, then of `d0 d1`.
    acc: Vec<[V; 3]>,
    packs: [[V; GROUP]; 5],
    tile: [[V; 4]; 5],
    /// The position weights by word, for vector loads: word `k` of position `j` at `k * BITS + j`.
    weights: Vec<u64>,
    /// Per word `k`, the sum over the pairs of positions of the product of their words `k`.
    pairs: [u128; 2],
}

impl SkipBlock {
    /// For blocks summed at the position weights `weights`, as polynomials.
    pub fn new(weights: &[[u64; 2]]) -> Box<Self> {
        assert_eq!(weights.len(), BITS);
        let mut words = vec![0; 2 * BITS];
        for (j, w) in weights.iter().enumerate() {
            for (k, &w) in w.iter().enumerate() {
                words[k * BITS + j] = w;
            }
        }
        let pairs = std::array::from_fn(|k| {
            weights
                .as_chunks::<2>()
                .0
                .iter()
                .fold(0, |acc, pair| acc ^ mul_wide(pair[0][k], pair[1][k]))
        });
        let zero = V::zero();
        Box::new(Self {
            acc: vec![[zero; 3]; 2 * MIXED],
            packs: [[zero; GROUP]; 5],
            tile: [[zero; 4]; 5],
            weights: words,
            pairs,
        })
    }

    /// Sum a block.
    pub fn run(&mut self, block: &[u64]) {
        assert!(block.len() >= BITS * BLOCK_WORDS);
        self.acc.fill([V::zero(); 3]);
        let t = tables();
        for half in 0..2 {
            for r in 0..5 {
                for z in (0..64).step_by(WIDTH) {
                    let js = row_positions(r, z);
                    for (packs, &j) in self.packs.iter_mut().zip(&js) {
                        embed_positions(&t, &block[BLOCK_WORDS * j..][..WIDTH * BLOCK_WORDS], half, packs);
                    }
                    // SAFETY: a position's four words from `js[x]` lie in the weights, which hold `BITS` of each.
                    let w: [[V; 2]; 5] = std::array::from_fn(|x| {
                        std::array::from_fn(|k| unsafe { V::load(self.weights[k * BITS + js[x]..].as_ptr()) })
                    });
                    // `HIGH[6..]` have the group at infinity, which the rounds do not use.
                    for high in 0..6 {
                        for (tile, packs) in self.tile.iter_mut().zip(&self.packs) {
                            *tile = extend_high(packs, high);
                        }
                        for i in 0..if high < 4 { 5 } else { 9 } {
                            let e: [V; 5] = std::array::from_fn(|x| {
                                let v = &self.tile[x];
                                match i {
                                    0 => v[0].xor(v[1]),
                                    1 => v[2].xor(v[3]),
                                    2 => v[0].xor(v[2]),
                                    3 => v[1].xor(v[3]),
                                    4 => v[0].xor(v[1]).xor(v[2].xor(v[3])),
                                    _ => v[i - 5],
                                }
                            });
                            let mut sums = [V::zero(); 3];
                            for x in 0..5 {
                                let (b, c) = (e[(x + 1) % 5], e[(x + 2) % 5]);
                                let (even, odd) = (b.clmul::<0x00>(c), b.clmul::<0x11>(c));
                                // The odd position's product beside the even one's weight, and the reverse.
                                let crossed = odd.unpacklo(even);
                                for k in 0..2 {
                                    let y = w[x][k].xor(crossed);
                                    sums[k] = sums[k].xor(y.clmul::<0x01>(y));
                                }
                                sums[2] = sums[2].xor(even.clmul::<0x00>(odd));
                            }
                            let point = match i {
                                0..=3 => 20 + 4 * high + i,
                                4 => 56 + high,
                                _ => 4 * (high - 4) + i - 5,
                            };
                            let acc = &mut self.acc[slot(half, point)];
                            for (acc, sum) in acc.iter_mut().zip(sums) {
                                *acc = acc.xor(sum);
                            }
                        }
                    }
                }
            }
        }
    }

    /// The sum for slot `i`, as three polynomial words.
    ///
    /// A slot the kernel never accumulates, points 8..19, 44..55 and 62..64, returns the layer's pair sums.
    /// The grouped rounds read only the slots they use.
    pub fn get(&self, i: usize) -> [u64; 3] {
        let [first, second, products]: [u128; 3] = std::array::from_fn(|k| self.acc[i][k].sum());
        // Each word's weighted sum, below bit 110: the pair products cancel above it.
        let low = first ^ products ^ self.pairs[0];
        let high = second ^ products ^ self.pairs[1];
        [low as u64, (low >> 64) as u64 ^ high as u64, (high >> 64) as u64]
    }
}
