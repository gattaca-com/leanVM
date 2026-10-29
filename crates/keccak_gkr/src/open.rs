//! From the two claims the layers leave to openings of the commitment.
//!
//! A claim weighs every bit of the committed states it speaks of. The commitment holds those
//! bits as lanes, 64 positions to a word, so an opening is the 64 partial evaluations along `z`
//! at one point of the word index, an instance then a slot. A short sumcheck over the slot (and,
//! for a claim whose instance weights are not an `eq`, over the instances of a pack) gets there.
//! The first claim weighs the bits of a word as the word itself does, so its 64 evaluations
//! are only ever needed as the one they add up to: the committed polynomial at that point.

use fiat_shamir::transcript::ProverState;
use pcs::stack_open::{RingSwitchClaim, RingSwitchOpen, StackClaim};
use primitives::field::{F64, F192};
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni")))]
use primitives::field::F192Unreduced;
use primitives::multilinear::{eq_table, mle_eval, poly_eval};

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni")))]
use crate::basis::byte_sums;
use crate::keccak::{BITS, LANES, along, iota_bit};
use crate::protocol::{Claim, dot, sum};
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni"))]
use crate::prover::{affine_matrix, byte};
use crate::witness::{BLOCK_WORDS, Shape, Witness, words};

/// The 64 slices of the committed stack at `point`: bit `z` of every word, as a multilinear.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Opening {
    pub point: Vec<F192>,
    pub slices: Vec<F192>,
}

/// The committed stack at `point`, its words as field elements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WordOpening {
    pub point: Vec<F192>,
    pub value: F192,
}

/// What the reduction leaves for the commitment to answer: the committed segment outputs as
/// words, if a segment's output is committed at all, and the inputs as slices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Openings {
    pub outputs: Option<WordOpening>,
    pub inputs: Opening,
}

impl Openings {
    /// The openings as claims on the committed states, placed at `offset` of the stack: the
    /// inputs' ring-switched, the outputs' a point claim to add to the batch.
    pub fn into_pcs_claims(self, offset: usize) -> (RingSwitchOpen, Option<StackClaim>) {
        let ring = RingSwitchOpen {
            offset,
            claims: vec![RingSwitchClaim {
                suffix_point: self.inputs.point,
                s_hat_v: Some(self.inputs.slices),
            }],
        };
        let outputs = self.outputs.map(|outputs| StackClaim::Point {
            offset,
            low_point: outputs.point,
            value: outputs.value,
        });
        (ring, outputs)
    }
}

/// The weight a claim puts on each `(slot, z)`: its position weights, on the slots of the states
/// it speaks of, scaled by the segment. `first` is the state segment 0 claims, and only the first
/// `count` segments' states are committed.
pub fn slot_weights(shape: Shape, claim: &Claim, first: usize, count: usize) -> Vec<F192> {
    let segments = eq_table(&claim.point[shape.pack_vars()..]);
    let mut out = vec![F192::ZERO; 64 << shape.slot_log()];
    for segment in 0..count {
        let base = 64 * LANES * (first + segment);
        for j in 0..BITS {
            out[base + j] = segments[segment] * claim.weights[j];
        }
    }
    out
}

/// One word of the sliced state after a round, from the state the round applies `chi` to.
#[inline(always)]
fn chi_word(block: &[u64], round: usize, j: usize, w: usize) -> u64 {
    let chi =
        block[j * BLOCK_WORDS + w] ^ (!block[along(j, 1) * BLOCK_WORDS + w] & block[along(j, 2) * BLOCK_WORDS + w]);
    if j < 64 && iota_bit(round, j) { !chi } else { chi }
}

/// The eight words of position `j` in block `g` of the state after `round` rounds.
#[inline(always)]
pub(crate) fn state_words(witness: &Witness, round: usize, g: usize, j: usize) -> [u64; BLOCK_WORDS] {
    let block = witness.layer(round, g);
    if round == 0 {
        return *words(block, j);
    }
    std::array::from_fn(|w| chi_word(block, round, j, w))
}

/// Every position of every committed state, at a pack `point`, keeping the sixteen instances
/// of a pack apart: entry `(i + 16 * slot) * 64 + z`.
///
/// The weight of a pack is a product over its index in a run of blocks and the run's own
/// index. The first factor meets the bits as subset sums, a table per pack of a word and block
/// of the run, so only a run's sum is multiplied. Every bit of the padding block is that of
/// its first word, so the blocks it stands for weigh it together.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni")))]
fn fold_packs(witness: &Witness, point: &[F192]) -> Vec<F192> {
    let shape = witness.shape;
    let written = witness.written();
    let padding = sum(&eq_table(&point[5..])[written..]);
    let run_vars = (point.len() - 5).min(3);
    let (low, runs) = (eq_table(&point[..5 + run_vars]), eq_table(&point[5 + run_vars..]));
    // Table `4 * block + p`: pack `p + 4 * w` of the run's block, over the eight words `w`.
    let tables: Vec<[F192; 256]> = (0..4 << run_vars)
        .map(|t| byte_sums(&std::array::from_fn(|w| low[t % 4 + 4 * w + 32 * (t / 4)])))
        .collect();
    // A task per position of a committed lane: a task per lane would leave threads idle.
    let positions: Vec<[F192; 16]> = parallel::map_collect(64 * shape.slots(), |task| {
        let (lane, z) = (task / 64, task % 64);
        let round = shape.depth * (lane / LANES);
        let j = 64 * (lane % LANES) + z;
        let mut acc = [F192Unreduced::ZERO; 16];
        for (run, &scale) in runs.iter().enumerate().take(written.div_ceil(1 << run_vars)) {
            let mut sums = [F192::ZERO; 16];
            for block in 0..(1 << run_vars).min(written - (run << run_vars)) {
                let words = state_words(witness, round, (run << run_vars) + block, j);
                // Byte `8 * b + t` gathers bit `t` of byte `b` of each word: instance
                // `8 * (b % 2) + t` of pack `b / 2`, across the eight words.
                let mut gathered = [0u8; 64];
                primitives::bits::bit_transpose_64bytes(&le_bytes(&words), &mut gathered);
                for (i, sum) in sums.iter_mut().enumerate() {
                    for p in 0..4 {
                        *sum += tables[4 * block + p][gathered[8 * (2 * p + i / 8) + i % 8] as usize];
                    }
                }
            }
            for (acc, sum) in acc.iter_mut().zip(sums) {
                *acc ^= scale.mul_unreduced(sum);
            }
        }
        let pad = written < shape.n_blocks() && state_words(witness, round, written, j)[0] & 1 == 1;
        acc.map(|acc| if pad { acc.reduce() + padding } else { acc.reduce() })
    });
    let mut out = vec![F192::ZERO; (16 * 64) << shape.slot_log()];
    for (task, sums) in positions.iter().enumerate() {
        let (lane, z) = (task / 64, task % 64);
        for (i, &sum) in sums.iter().enumerate() {
            out[(i + 16 * lane) * 64 + z] = sum;
        }
    }
    out
}

/// [`fold_packs`] as a GF(2) matrix product, four positions and their sixteen instances at a
/// time. A position's bits, transposed, are bytes of eight packs `4w + r` of one instance; a
/// GF2P8AFFINE matrix of those packs' weights sends such a byte to one byte of the sum.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni"))]
fn fold_packs(witness: &Witness, point: &[F192]) -> Vec<F192> {
    let shape = witness.shape;
    let written = witness.written();
    let eq = eq_table(point);
    let padding = sum(&eq[32 * written..]);
    // Matrix `[g][r][o]`: input bit `w` is pack `32 g + 4 w + r`, output bit `t` is bit `8 o + t`
    // of its weight, read off matrix byte `7 - t`.
    let matrices: Vec<[[u64; 24]; 4]> = eq[..32 * written]
        .as_chunks::<32>()
        .0
        .iter()
        .map(|e| std::array::from_fn(|r| std::array::from_fn(|o| affine_matrix(std::array::from_fn(|w| byte(&e[4 * w + r], o))))))
        .collect();
    // A task per four positions of a committed lane.
    let positions: Vec<[[F192; 16]; 4]> = parallel::map_collect(16 * shape.slots(), |task| {
        let (lane, z) = (task / 16, 4 * (task % 16));
        let round = shape.depth * (lane / LANES);
        let j = 64 * (lane % LANES) + z;
        // SAFETY: the features are enabled at compile time.
        let mut sums = unsafe { fold_gfni(witness, round, j, &matrices) };
        for (p, sums) in sums.iter_mut().enumerate() {
            if written < shape.n_blocks() && state_words(witness, round, written, j + p)[0] & 1 == 1 {
                for sum in sums.iter_mut() {
                    *sum += padding;
                }
            }
        }
        sums
    });
    let mut out = vec![F192::ZERO; (16 * 64) << shape.slot_log()];
    for (task, sums) in positions.iter().enumerate() {
        let (lane, z) = (task / 16, 4 * (task % 16));
        for (p, sums) in sums.iter().enumerate() {
            for (i, &sum) in sums.iter().enumerate() {
                out[(i + 16 * lane) * 64 + z + p] = sum;
            }
        }
    }
    out
}

/// The written blocks' sums for positions `j..j + 4` of the state after `round` rounds.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
fn fold_gfni(witness: &Witness, round: usize, j: usize, matrices: &[[[u64; 24]; 4]]) -> [[F192; 16]; 4] {
    use core::arch::x86_64::*;
    use primitives::bits::bit_transpose_64bytes_m512;
    use primitives::simd::{load, store};
    let mut acc = [_mm512_setzero_si512(); 24];
    for (g, m) in matrices.iter().enumerate() {
        // Byte `16 r + i` of position `p`: bit `w` is instance `i` of pack `4 w + r`.
        let block = witness.layer(round, g);
        let bits: [__m512i; 4] = std::array::from_fn(|p| {
            let j = j + p;
            let state = if round == 0 {
                load(&block[BLOCK_WORDS * j..][..BLOCK_WORDS], 0)
            } else {
                // `chi`, then `iota` on lane 0.
                let at = |j: usize| load(&block[BLOCK_WORDS * j..][..BLOCK_WORDS], 0);
                let chi = _mm512_ternarylogic_epi64(at(j), at(along(j, 1)), at(along(j, 2)), 0xD2);
                if j < 64 && iota_bit(round, j) { _mm512_ternarylogic_epi64(chi, chi, chi, 0x0F) } else { chi }
            };
            bit_transpose_64bytes_m512(state)
        });
        // Lane `p` of `x[r]` is lane `r` of position `p`.
        let (low, high) = (
            |a, b| _mm512_shuffle_i64x2::<0b01_00_01_00>(a, b),
            |a, b| _mm512_shuffle_i64x2::<0b11_10_11_10>(a, b),
        );
        let t = [low(bits[0], bits[1]), low(bits[2], bits[3]), high(bits[0], bits[1]), high(bits[2], bits[3])];
        let x = [
            _mm512_shuffle_i64x2::<0b10_00_10_00>(t[0], t[1]),
            _mm512_shuffle_i64x2::<0b11_01_11_01>(t[0], t[1]),
            _mm512_shuffle_i64x2::<0b10_00_10_00>(t[2], t[3]),
            _mm512_shuffle_i64x2::<0b11_01_11_01>(t[2], t[3]),
        ];
        for (r, x) in x.as_chunks::<2>().0.iter().enumerate() {
            for (o, acc) in acc.iter_mut().enumerate() {
                let a = _mm512_gf2p8affine_epi64_epi8(x[0], _mm512_set1_epi64(m[2 * r][o] as i64), 0);
                let b = _mm512_gf2p8affine_epi64_epi8(x[1], _mm512_set1_epi64(m[2 * r + 1][o] as i64), 0);
                *acc = _mm512_ternarylogic_epi64(*acc, a, b, 0x96);
            }
        }
    }
    // Byte `m` of `acc[o]` is byte `o` of instance `m % 16` of position `m / 16`.
    let gather: [u8; 64] = std::array::from_fn(|o| (8 * (o % 8) + o / 8) as u8);
    let gather = load(&gather[..], 0);
    let mut limbs = [[0u64; 64]; 3];
    for (limb, acc) in limbs.iter_mut().zip(acc.as_chunks::<8>().0) {
        for (m, x) in crate::prover::transpose8(*acc).into_iter().enumerate() {
            store(&mut limb[..], m, _mm512_permutexvar_epi8(gather, x));
        }
    }
    std::array::from_fn(|p| std::array::from_fn(|i| F192::new(limbs[0][16 * p + i], limbs[1][16 * p + i], limbs[2][16 * p + i])))
}

/// Byte `b` of word `x` at `8 * x + b`.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni")))]
fn le_bytes(words: &[u64; BLOCK_WORDS]) -> [u8; 64] {
    let mut bytes = [0u8; 64];
    for (bytes, word) in bytes.as_chunks_mut::<8>().0.iter_mut().zip(words) {
        *bytes = word.to_le_bytes();
    }
    bytes
}

/// The sum over rows of the two tables' dot product: its round values at `0` and infinity.
fn round_values(a: &[F192], b: &[F192], width: usize) -> (F192, F192) {
    let mut zero = F192::ZERO;
    let mut infinity = F192::ZERO;
    for (a, b) in a.chunks_exact(2 * width).zip(b.chunks_exact(2 * width)) {
        for j in 0..width {
            zero += a[j] * b[j];
            infinity += (a[j] + a[width + j]) * (b[j] + b[width + j]);
        }
    }
    (zero, infinity)
}

pub(crate) fn fold(table: &[F192], width: usize, challenge: F192) -> Vec<F192> {
    table
        .chunks_exact(2 * width)
        .flat_map(|pair| (0..width).map(move |j| pair[j] + challenge * (pair[j] + pair[width + j])))
        .collect()
}

/// Bind the leading variable of `scale * sum of a * b` in every round, down to one row of `a`.
fn prove_rounds(
    ps: &mut ProverState,
    mut a: Vec<F192>,
    mut b: Vec<F192>,
    width: usize,
    scale: F192,
    value: &mut F192,
) -> (Vec<F192>, Vec<F192>, Vec<F192>) {
    let mut point = Vec::new();
    while a.len() > width {
        let (zero, infinity) = round_values(&a, &b, width);
        let (zero, infinity) = (scale * zero, scale * infinity);
        let coeffs = [zero, *value + infinity, infinity];
        ps.add_round_poly(&coeffs, false);
        let challenge = ps.sample();
        *value = poly_eval(&coeffs, challenge);
        point.push(challenge);
        a = fold(&a, width, challenge);
        b = fold(&b, width, challenge);
    }
    (a, b, point)
}

/// The weight a word-level claim puts on each slot of the committed segment outputs, every
/// segment's but the last: what [`slot_weights`] gives its bit 0.
fn word_weights(shape: Shape, claim: &Claim) -> Vec<F192> {
    slot_weights(shape, claim, 1, shape.segments() - 1)
        .into_iter()
        .step_by(64)
        .collect()
}

/// Every slot of the committed block as a word, at an instance point, and the part of the
/// word-level `claim` on the committed segment outputs among them. An opening is of the whole
/// block, so the slots the claim puts no weight on are evaluated too: the spare ones past
/// [`Shape::slots`] carry other columns (§cpu), and the fold that follows must see them.
pub fn output_claim(shape: Shape, committed: &[F64], instance: &[F192], claim: &Claim) -> (Vec<F192>, F192) {
    let n = shape.n();
    let words = parallel::map_collect(1 << shape.slot_log(), |slot| {
        mle_eval(&committed[slot * n..][..n], instance)
    });
    let value = dot(&words, &word_weights(shape, claim));
    (words, value)
}

/// Open `value`, the part of the first claim on the committed segment outputs, whose instance
/// weights are `eq(instance, .)` and whose position weights are those of a word.
pub fn open_outputs(
    ps: &mut ProverState,
    shape: Shape,
    instance: &[F192],
    claim: &Claim,
    mut value: F192,
    words: Vec<F192>,
) -> WordOpening {
    let (word, _, slot) = prove_rounds(ps, words, word_weights(shape, claim), 1, F192::ONE, &mut value);
    let word = word[0];
    ps.add_scalar(word);
    WordOpening {
        point: [instance, &slot].concat(),
        value: word,
    }
}

/// The bit slices, at the instance point, of the committed slots the claim weighs nothing on:
/// the spare ones past [`Shape::slots`], where other columns ride (§cpu). The reduction says
/// nothing about them, but the claim it leaves is on the whole committed block, so the slot
/// fold below has to carry their bits.
///
/// Slice `z` of slot `s` is `Σ_step eq(instance, step) · bit z of committed[s·n + step]`: a
/// subset sum of the `eq` table over the steps whose bit is set, read eight steps at a time off
/// their transposed bytes.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni")))]
fn spare_slices(shape: Shape, committed: &[F64], instance: &[F192]) -> Vec<[F192; 64]> {
    let (n, spare) = (shape.n(), (1 << shape.slot_log()) - shape.slots());
    let eq = eq_table(instance);
    let (groups, slots) = (eq.as_chunks::<8>().0, &committed[shape.slots() * n..]);
    // A task per group of eight steps rather than per slot: the subset sums are the work, and one
    // group's table serves every slot, where a task per slot would rebuild it and, there being as
    // few slots as there are, leave most of the pool idle.
    parallel::fold_reduce(
        n / 8,
        || vec![[F192::ZERO; 64]; spare],
        |acc, group| {
            let table = byte_sums(&groups[group]);
            for (slot, acc) in acc.iter_mut().enumerate() {
                let words = std::array::from_fn(|w| slots[slot * n + 8 * group + w].0);
                let mut gathered = [0u8; 64];
                primitives::bits::bit_transpose_64bytes(&le_bytes(&words), &mut gathered);
                for (acc, &byte) in acc.iter_mut().zip(&gathered) {
                    *acc += table[byte as usize];
                }
            }
        },
        |mut a, b| {
            for (a, b) in a.iter_mut().flatten().zip(b.iter().flatten()) {
                *a += *b;
            }
            a
        },
    )
}

/// [`spare_slices`] as a GF(2) matrix product: eight steps' words, transposed, are a byte per
/// bit position, which a GF2P8AFFINE matrix of the steps' weights sends to one byte of the slice.
/// A task per run of steps, whose matrices serve every spare slot.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni"))]
fn spare_slices(shape: Shape, committed: &[F64], instance: &[F192]) -> Vec<[F192; 64]> {
    const RUN: usize = 128;
    let (n, spare) = (shape.n(), (1 << shape.slot_log()) - shape.slots());
    let eq = eq_table(instance);
    let slots = &committed[shape.slots() * n..];
    let chunks = eq.as_chunks::<8>().0;
    parallel::fold_reduce(
        chunks.len().div_ceil(RUN),
        || vec![[F192::ZERO; 64]; spare],
        |acc, task| {
            let first = RUN * task;
            let matrices: Vec<[u64; 24]> = chunks[first..(first + RUN).min(chunks.len())]
                .iter()
                .map(|e| std::array::from_fn(|o| affine_matrix(std::array::from_fn(|w| byte(&e[w], o)))))
                .collect();
            for (slot, acc) in acc.iter_mut().enumerate() {
                let words = &slots[slot * n + 8 * first..][..8 * matrices.len()];
                // SAFETY: the features are enabled at compile time.
                let sums = unsafe { slices_gfni(words, &matrices) };
                for (acc, sum) in acc.iter_mut().zip(sums) {
                    *acc += sum;
                }
            }
        },
        |mut a, b| {
            for (a, b) in a.iter_mut().flatten().zip(b.iter().flatten()) {
                *a += *b;
            }
            a
        },
    )
}

/// Per bit position, the sum of the weights of the steps whose word has it set: matrix `[c][o]`
/// makes byte `o` of the sum from steps `8c..8c + 8`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
fn slices_gfni(words: &[F64], matrices: &[[u64; 24]]) -> [F192; 64] {
    use core::arch::x86_64::*;
    use primitives::bits::bit_transpose_64bytes_m512;
    use primitives::simd::{load, store};
    assert!(words.len() == 8 * matrices.len() && matrices.len().is_multiple_of(2));
    let mut acc = [_mm512_setzero_si512(); 24];
    for (c, m) in matrices.as_chunks::<2>().0.iter().enumerate() {
        // Byte `z` of `x[h]`: bit `w` is bit `z` of step `8 (2c + h) + w`.
        let x = [0, 1].map(|h| bit_transpose_64bytes_m512(load(&words[8 * (2 * c + h)..][..8], 0)));
        for (o, acc) in acc.iter_mut().enumerate() {
            let a = _mm512_gf2p8affine_epi64_epi8(x[0], _mm512_set1_epi64(m[0][o] as i64), 0);
            let b = _mm512_gf2p8affine_epi64_epi8(x[1], _mm512_set1_epi64(m[1][o] as i64), 0);
            *acc = _mm512_ternarylogic_epi64(*acc, a, b, 0x96);
        }
    }
    // Byte `z` of `acc[o]` is byte `o` of position `z`'s sum.
    let gather: [u8; 64] = std::array::from_fn(|o| (8 * (o % 8) + o / 8) as u8);
    let gather = load(&gather[..], 0);
    let mut limbs = [[0u64; 64]; 3];
    for (limb, acc) in limbs.iter_mut().zip(acc.as_chunks::<8>().0) {
        for (m, x) in crate::prover::transpose8(*acc).into_iter().enumerate() {
            store(&mut limb[..], m, _mm512_permutexvar_epi8(gather, x));
        }
    }
    std::array::from_fn(|z| F192::new(limbs[0][z], limbs[1][z], limbs[2][z]))
}

/// Open the claim on the inputs: the instances of a pack first, then the slot.
pub fn open_inputs(ps: &mut ProverState, witness: &Witness, committed: &[F64], claim: &Claim) -> Opening {
    let shape = witness.shape;
    let pack = &claim.point[..shape.pack_vars()];
    let folded = fold_packs(witness, pack);
    let weights = slot_weights(shape, claim, 0, shape.segments());
    // Per instance of a pack, the weighted sum over slots and positions.
    let per_instance: Vec<F192> = (0..16)
        .map(|i| {
            (0..1usize << shape.slot_log()).fold(F192::ZERO, |acc, slot| {
                acc + dot(&folded[(i + 16 * slot) * 64..][..64], &weights[slot * 64..][..64])
            })
        })
        .collect();
    let mut value = claim.value;
    let (_, mu, low) = prove_rounds(ps, per_instance, claim.mu.to_vec(), 1, F192::ONE, &mut value);
    // The slots of the stack at that instance: entry `(i + 16 * slot) * 64 + z` folds over `i`.
    let mut at_instance = vec![F192::ZERO; 64 << shape.slot_log()];
    let low_eq = eq_table(&low);
    for (slot, out) in at_instance.as_chunks_mut::<64>().0.iter_mut().enumerate() {
        for i in 0..16 {
            for z in 0..64 {
                out[z] += low_eq[i] * folded[(i + 16 * slot) * 64 + z];
            }
        }
    }
    for (slot, slices) in spare_slices(shape, committed, &[&low, pack].concat())
        .into_iter()
        .enumerate()
    {
        at_instance[(shape.slots() + slot) * 64..][..64].copy_from_slice(&slices);
    }
    let (slices, _, slot) = prove_rounds(ps, at_instance, weights, 64, mu[0], &mut value);
    ps.add_scalars(&slices);
    Opening {
        point: [&low, pack, &slot].concat(),
        slices,
    }
}
