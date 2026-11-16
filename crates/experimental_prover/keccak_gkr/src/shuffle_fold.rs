//! The fold of a block position's packs by nibble tables (AVX2 without AVX-512).
//!
//! A fold is `sum over packs q of weight[q] * embed(pack q)`, reduced.
//! It is a GF(2)-linear map of the packs' bits to the 16 bytes of a field element.
//! The weights are the same at every position of every block of a layer, so the map is tabulated once.
//! Each nibble of the input has one byte table per output byte, holding the image of its sixteen values.
//! `VPSHUFB` then looks one nibble of sixteen positions up at once.
//!
//! On Zen 5 a `VPCLMULQDQ` takes two cycles at any width and a `VPSHUFB` half a cycle.
//! The carry-less fold costs two products a pack, the table fold one shuffle a pack.
//! Memory moves go through `load_at` and `store_at` of `primitives`, which check their bounds.

use core::arch::x86_64::*;

use primitives::{load_at, store_at};

use crate::power::{Element, PowerField};
use crate::prover::LANE_ROW as FACTOR_ROW;
use crate::rmfe::RMFE;
use crate::witness::BLOCK_WORDS;
use verifier::keccak::BITS;

/// Inputs a step folds: a 16 by 16 byte transpose brings one byte of each into a lane.
const STEP: usize = 16;
/// Bytes of an input a lane holds, and of a field element.
const LANE: usize = 16;

/// The map of 32 input bytes, 16 to a lane, to one field element per lane.
pub struct ShuffleFold {
    /// Table `(2 * byte + nibble) * 16 + b`: output byte `b` of the image of each value of that nibble.
    /// The low lane is input byte `byte`'s, the high lane input byte `16 + byte`'s.
    tables: Box<[[u8; 32]; 2 * LANE * LANE]>,
}

impl ShuffleFold {
    /// The fold in which input byte `byte` is a byte of a pack weighing `weight(byte)`.
    /// It is byte `byte % 2` of that pack, as a block lays its packs out.
    pub fn new(field: &PowerField, weight: impl Fn(usize) -> Element) -> Self {
        let mut tables = vec![[0u8; 32]; 2 * LANE * LANE];
        for byte in 0..2 * LANE {
            let weight = weight(byte);
            for nibble in 0..2 {
                let bits: [Element; 4] =
                    std::array::from_fn(|t| field.mul(&weight, &[RMFE.emb[8 * (byte % 2) + 4 * nibble + t] as u64, 0]));
                for value in 0..16 {
                    let mut image = [0u64; 2];
                    for (t, bit) in bits.iter().enumerate() {
                        if value >> t & 1 == 1 {
                            image[0] ^= bit[0];
                            image[1] ^= bit[1];
                        }
                    }
                    for b in 0..LANE {
                        let table = &mut tables[(2 * (byte % LANE) + nibble) * LANE + b];
                        table[LANE * (byte / LANE) + value] = (image[b / 8] >> (8 * (b % 8))) as u8;
                    }
                }
            }
        }
        let tables = tables
            .into_boxed_slice()
            .try_into()
            .expect("a table per nibble and output byte");
        Self { tables }
    }

    /// The images of sixteen inputs: lane `l` of `out[p]` is the element lane `l` of `x[p]` maps to.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn images(&self, mut x: [__m256i; STEP]) -> [__m256i; STEP] {
        // Register `c` now holds byte `c` of every input, a lane apart.
        primitives::bits::transpose_bytes16x16_per_lane256(&mut x);
        let mask = _mm256_set1_epi8(0x0f);
        let mut nibbles = [_mm256_setzero_si256(); 2 * LANE];
        for (c, &column) in x.iter().enumerate() {
            nibbles[2 * c] = _mm256_and_si256(column, mask);
            nibbles[2 * c + 1] = _mm256_and_si256(_mm256_srli_epi16::<4>(column), mask);
        }
        // Eight output bytes at a time, so their sums stay in registers.
        let mut out = [_mm256_setzero_si256(); STEP];
        for (half, out) in out.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let mut acc = [_mm256_setzero_si256(); 8];
            for (n, &index) in nibbles.iter().enumerate() {
                let tables = &self.tables[LANE * n + 8 * half..][..8];
                for (acc, table) in acc.iter_mut().zip(tables) {
                    let table: __m256i = load_at(table, 0);
                    *acc = _mm256_xor_si256(*acc, _mm256_shuffle_epi8(table, index));
                }
            }
            *out = acc;
        }
        // Register `b` holds byte `b` of every image: back to an image per register.
        primitives::bits::transpose_bytes16x16_per_lane256(&mut out);
        out
    }

    /// The two rows of a block's table, a row per half block.
    /// Entry `rows[h * 2 * BITS + k * BITS + j]` is word `k` of the fold of half `h` of position `j`.
    /// A half is four words, both lanes.
    pub fn rows(&self, block: &[u64], rows: &mut [u64]) {
        const { assert!(BITS.is_multiple_of(STEP)) };
        // SAFETY: AVX2 is enabled at compile time (the module's cfg). Memory moves check their bounds.
        unsafe {
            for half in 0..2 {
                for j in (0..BITS).step_by(STEP) {
                    let x = std::array::from_fn(|p| load_at(block, BLOCK_WORDS * (j + p) + 4 * half));
                    let images = self.images(x);
                    let row = 2 * BITS * half + j;
                    for (p, pair) in images.as_chunks::<2>().0.iter().enumerate() {
                        // The two lanes are the two halves of one sum.
                        let [a, b] =
                            pair.map(|v| _mm_xor_si128(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v)));
                        store_at(rows, row + 2 * p, _mm_unpacklo_epi64(a, b));
                        store_at(rows, row + BITS + 2 * p, _mm_unpackhi_epi64(a, b));
                    }
                }
            }
        }
    }

    /// The folds of the differences of a block's adjacent groups, five lanes of 64 positions.
    /// Row `5 * half + x` of `rows` is laid out by word, 64 positions wide.
    /// A lane is a half's difference, two words.
    pub fn differences(&self, block: &[u64], rows: &mut [u64]) {
        // SAFETY: AVX2 is enabled at compile time (the module's cfg). Memory moves check their bounds.
        unsafe {
            for lane in 0..5 {
                for z in (0..64).step_by(STEP) {
                    let x = std::array::from_fn(|p| {
                        let at = BLOCK_WORDS * (64 * lane + z + p);
                        let (low, high): (__m256i, __m256i) = (load_at(block, at), load_at(block, at + 4));
                        // Words `0 ^ 2, 1 ^ 3` in the low lane and `4 ^ 6, 5 ^ 7` in the high one.
                        _mm256_xor_si256(
                            _mm256_permute2x128_si256::<0x20>(low, high),
                            _mm256_permute2x128_si256::<0x31>(low, high),
                        )
                    });
                    let images = self.images(x);
                    for (p, pair) in images.as_chunks::<2>().0.iter().enumerate() {
                        let words = [
                            _mm256_unpacklo_epi64(pair[0], pair[1]),
                            _mm256_unpackhi_epi64(pair[0], pair[1]),
                        ];
                        for (k, word) in words.into_iter().enumerate() {
                            let at = k * 64 + z + 2 * p;
                            let halves = [_mm256_castsi256_si128(word), _mm256_extracti128_si256::<1>(word)];
                            for (half, v) in halves.into_iter().enumerate() {
                                store_at(rows, (5 * half + lane) * FACTOR_ROW + at, v);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use primitives::test_rng::Rng;

    /// Pack `q` of a run of words.
    fn pack(words: &[u64], q: usize) -> u16 {
        (words[q / 4] >> (16 * (q % 4))) as u16
    }

    /// `sum over q of fold[q] * embed(pack q)`, reduced.
    fn folded(field: &PowerField, fold: &[Element], words: &[u64]) -> Element {
        let mut sum = [0u64; 2];
        for (q, weight) in fold.iter().enumerate() {
            let product = field.mul(weight, &[RMFE.embed(pack(words, q)) as u64, 0]);
            sum[0] ^= product[0];
            sum[1] ^= product[1];
        }
        sum
    }

    #[test]
    fn tables_give_the_products() {
        let mut rng = Rng::new(51);
        let field = PowerField::new(rng.ext()).unwrap();
        let block: Vec<u64> = (0..BITS * BLOCK_WORDS).map(|_| rng.next_u64()).collect();

        let fold: Vec<Element> = (0..16).map(|_| field.to_poly(rng.ext())).collect();
        let shuffle = ShuffleFold::new(&field, |byte| fold[byte / 2]);
        let mut rows = vec![0u64; 4 * BITS];
        shuffle.rows(&block, &mut rows);
        for j in 0..BITS {
            for half in 0..2 {
                let expected = folded(&field, &fold, &block[BLOCK_WORDS * j + 4 * half..][..4]);
                let row = &rows[2 * BITS * half..];
                assert_eq!([row[j], row[BITS + j]], expected, "position {j} half {half}");
            }
        }

        let fold: Vec<Element> = (0..8).map(|_| field.to_poly(rng.ext())).collect();
        let shuffle = ShuffleFold::new(&field, |byte| fold[byte / 2 % 8]);
        let mut rows = vec![0u64; 10 * FACTOR_ROW];
        shuffle.differences(&block, &mut rows);
        for lane in 0..5 {
            for z in 0..64 {
                let w = &block[BLOCK_WORDS * (64 * lane + z)..][..BLOCK_WORDS];
                let packed = [w[0] ^ w[2], w[1] ^ w[3], w[4] ^ w[6], w[5] ^ w[7]];
                for half in 0..2 {
                    let expected = folded(&field, &fold, &packed[2 * half..][..2]);
                    let row = &rows[(5 * half + lane) * FACTOR_ROW..];
                    assert_eq!([row[z], row[64 + z]], expected, "lane {lane} slice {z} half {half}");
                }
            }
        }
    }
}
