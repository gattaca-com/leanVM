//! The layer prover.
//!
//! One layer makes three passes over the bits of the states it reduces to:
//!
//! 1. the product polynomial, before its evaluation point `s` is known;
//! 2. the first sumcheck rounds, over the packs of half a block, as sums of bit-level products;
//! 3. one field element per half block and position, for the rounds that remain.
//!
//! From `s` on, everything stays in the power basis of `s` ([`crate::power`]) until the tables
//! are down to one row.

use fiat_shamir::transcript::ProverState;
use primitives::field::{F192, F192Unreduced};
use primitives::multilinear::{eq_table, poly_eval};

use crate::basis::{PolyEval, add_mul, carry};
use crate::keccak::{BITS, LOCAL_VARS, along, linear_transpose, row_positions};
use crate::kernels::{
    GROUP, MIXED, PACKS, SKIP, SkipBlock, SkipWeights, TERNARY, embed, fold_groups, mixed_points, products, slot,
};
use crate::open::fold;
use crate::power::{Element, FACTOR_ROW, PowerField, get, set};
use crate::protocol::{Claim, chi_weights, mu_at, sum};
use crate::rmfe::PROD_BITS;
use crate::witness::{Recompute, Shape, Witness, transpose64, words};

/// The pattern index of a binary assignment.
fn ternary(b: usize) -> usize {
    (0..SKIP).fold(0, |acc, k| acc + (b >> k & 1) * 3usize.pow(k as u32))
}

/// Digits a weight splits into, and the values of one.
const DIGIT_BITS: usize = 8;
const DIGITS: usize = 192 / DIGIT_BITS;
/// Values of half a digit.
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
#[inline(always)]
fn digit(weight: F192, d: usize) -> usize {
    let w = [weight.c0, weight.c1, weight.c2];
    (w[d / 8] >> (DIGIT_BITS * (d % 8))) as usize & ((1 << DIGIT_BITS) - 1)
}

/// The positions grouped by each digit of their weight: those whose digit `d` is `v` are
/// `order[d][starts[d][v]..starts[d][v + 1]]`.
struct Groups {
    order: Vec<[u16; BITS]>,
    starts: Vec<[u16; (1 << DIGIT_BITS) + 1]>,
}

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
    /// `sum over blocks and positions of eq(point[5..], block) * weights[j] * product(q, j)`,
    /// where the product is `ones * u[j] + (ones + u[j + 1]) * u[j + 2]` on embeddings, the
    /// unprojected `chi`.
    ///
    /// Nothing here knows where the polynomials will be evaluated, so each product meets its
    /// weight bit by bit. The position weights are the same in every block, so a block's sum
    /// over positions is grouped by their digits ([`Groups`]), a part of the packs at a time;
    /// only that sum meets the block's weight, as field elements.
    fn pack_products(&self, claim: &Claim) -> Vec<[F192; PROD_BITS]> {
        let weights = self.task_weights(&eq_table(&claim.point[SKIP + 1..]));
        let groups = Groups::new(&claim.weights);
        let sums = parallel::map_reduce_with_state(
            self.n_tasks(),
            || {
                (
                    vec![[Part::ZERO; BITS]; PACKS / PART],
                    vec![[Part::ZERO; 192]; PACKS / PART],
                    Recompute::new(),
                )
            },
            || vec![F192Unreduced::ZERO; PAYLOAD_BITS],
            |(payloads, rows, recompute), sums, task| {
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
                for (payloads, rows) in payloads.iter().zip(rows.iter_mut()) {
                    groups.add_rows(payloads, rows);
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
                        *sum ^= weights[task].mul_unreduced(F192::new(columns[0][t], columns[1][t], columns[2][t]));
                    }
                }
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

    /// The layer's summand at `s`, extended over the packs of a group and summed over the groups:
    /// for every point `p` of `{0, 1, infinity}^SKIP`,
    /// `sum over groups of eq(point[SKIP..], group) * Q(p, group)`.
    ///
    /// A point at infinity in some variable keeps only the product `u[j + 1] * u[j + 2]`, summed
    /// here. A binary one is a pack, whose sum `binary` already holds. The packs of the padding
    /// block are all equal, so their extension vanishes there.
    fn skip_sums(&self, claim: &Claim, field: &PowerField, s: F192, binary: &[F192; GROUP]) -> Vec<F192> {
        let weights = SkipWeights::new(claim.weights.iter().map(|&w| field.to_poly(w)).collect());
        let groups: Vec<Element> = eq_table(&claim.point[SKIP..])
            .iter()
            .map(|&e| field.to_poly(e))
            .collect();
        let (nb, written) = (self.witness.shape.n_blocks(), self.witness.written());
        let mixed = parallel::map_reduce_with_state(
            self.rounds.len() * written,
            || (SkipBlock::new(), Recompute::new()),
            || vec![[0u64; 7]; MIXED],
            |(skip, recompute), total, task| {
                let (segment, g) = (task / written, task % written);
                let (row, block) = (g + nb * segment, self.witness.state(recompute, self.rounds[segment], g));
                skip.run(block, &weights);
                for half in 0..2 {
                    for (i, total) in total.iter_mut().enumerate() {
                        add_mul(total, &skip.get(slot(half, i)), &groups[2 * row + half]);
                    }
                }
            },
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    for (x, y) in x.iter_mut().zip(y) {
                        *x ^= y;
                    }
                }
                a
            },
        );
        let eval = PolyEval::new(s, 56);
        let mut sums = vec![F192::ZERO; TERNARY];
        for (slot, p) in mixed_points().into_iter().enumerate() {
            sums[p] = eval.eval(&mixed[slot]);
        }
        for (q, &value) in binary.iter().enumerate() {
            sums[ternary(q)] = value;
        }
        sums
    }

    /// The table the remaining rounds fold, a row per group: `T[j]`, the value at `s` of position
    /// `j`'s embeddings, folded over the group's packs at `challenges`. A block's two groups are
    /// the first of those rounds' pairs, so its round values are summed while the rows are at hand.
    /// The padding block's groups are one row, its segment's entry of `pads`.
    fn table(
        &self,
        round: &Round,
        challenges: &[F192],
        eq: &[F192],
        layout: Rows,
        table: &mut [u64],
        pads: &mut [u64],
    ) -> (F192, F192) {
        let field = round.field;
        let fold: Vec<Element> = eq_table(challenges).iter().map(|&e| field.to_poly(e)).collect();
        let fold: [[u64; GROUP]; 3] = std::array::from_fn(|k| std::array::from_fn(|q| fold[q][k]));
        let weights = self.task_weights(eq);
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
            || (vec![0u64; 4 * FACTOR_ROW], vec![0u64; 8 * BITS], Recompute::new()),
            RoundSums::default,
            |(scratch, wide, recompute), sums, task, [part]| {
                let (_, _, block) = self.block(recompute, task);
                let rows = &mut *part[0];
                let halves = rows.len() / ROW;
                // Both groups' unreduced sums, four words each and laid out by word.
                for j in 0..BITS {
                    let sums = fold_groups(&embed(words(block, j)), &fold);
                    for (wide, sum) in wide.as_chunks_mut::<{ 4 * BITS }>().0[..halves].iter_mut().zip(&sums) {
                        let sum: [u64; 4] = carry(sum);
                        for (k, &word) in sum.iter().enumerate() {
                            wide[k * BITS + j] = word;
                        }
                    }
                }
                for (t, wide) in rows
                    .as_chunks_mut::<ROW>()
                    .0
                    .iter_mut()
                    .zip(wide.as_chunks::<{ 4 * BITS }>().0)
                {
                    field.reduce_row(t, wide, BITS);
                }
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
        let mut parts: Vec<&mut [u64]> = dst[..to.len() * ROW]
            .chunks_mut(to.live * ROW)
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
        let rows = 1usize << (shape.y_vars() - SKIP);
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
    let pack_weights = eq_table(&claim.point[..SKIP + 1]);
    let product: [F192; PROD_BITS] =
        std::array::from_fn(|c| (0..PACKS).fold(F192::ZERO, |acc, q| acc + pack_weights[q] * per_pack[q][c]));
    ps.add_scalars(&product);
    let s = ps.sample();
    let kappa = sum(&mu_at(s));
    let mut value = poly_eval(&product, s);
    let field = PowerField::new(s).expect("the challenge lies in no proper subfield");

    // The packs of a group, from sums of bit-level products. A pack's own sum is its product
    // polynomial at `s`, over both groups of the block.
    let binary: [F192; GROUP] = std::array::from_fn(|q| {
        let half = claim.point[SKIP];
        (F192::ONE + half) * poly_eval(&per_pack[q], s) + half * poly_eval(&per_pack[GROUP + q], s)
    });
    let mut sums = tracing::info_span!("Skip").in_scope(|| layer.skip_sums(claim, &field, s, &binary));
    let mut scale = F192::ONE;
    let mut point = Vec::with_capacity(claim.point.len());
    for i in 0..SKIP {
        let later = eq_table(&claim.point[i + 1..SKIP]);
        let mut h = [F192::ZERO; 3];
        for (b, &eq) in later.iter().enumerate() {
            for (d, h) in h.iter_mut().enumerate() {
                *h += eq * sums[d + 3 * ternary(b)];
            }
        }
        let coeffs = [scale * h[0], scale * (h[0] + h[1] + h[2]), scale * h[2]];
        let challenge = bind(ps, &coeffs, claim.point[i], &mut value, &mut scale, &mut point);
        sums = sums
            .as_chunks::<3>()
            .0
            .iter()
            .map(|v| v[0] + challenge * (v[0] + v[1]) + (challenge.square() + challenge) * v[2])
            .collect();
    }

    // The groups, blocks and segments, on a table in the power basis. Each pass over it writes
    // the rows of one round and sums that round's values over them.
    let round = Round::new(&field, claim, s, kappa);
    let mut layout = Rows {
        m: 2 * shape.n_blocks(),
        live: 2 * witness.written(),
        segments: 1 << shape.seg_log(),
    };
    let (mut src, mut dst) = scratch.table.split_at_mut(layout.m * layout.segments * ROW);
    let pads = &mut scratch.pads;
    let mut values = tracing::info_span!("Tables").in_scope(|| {
        let eq = eq_table(&claim.point[SKIP + 1..]);
        layer.table(&round, &point, &eq, layout, src, pads)
    });
    let tail = tracing::info_span!("Tail").entered();
    for i in SKIP..claim.point.len() {
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
