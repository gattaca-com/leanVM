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

use primitives::field::{F128, F128Unreduced};
use primitives::hash::positions::row_positions;
use primitives::multilinear::{eq_table, interp, poly_eval};
use primitives::transcript::ProverState;
use verifier::keccak::{BITS, LOCAL_BITS, along, linear_transpose};

use crate::basis::{PolyEval, add_mul, carry};
use crate::kernels::{
    GROUP, GROUPED, MIXED, PACKS, SKIP, SkipBlock, TERNARY, embed, fold_groups, mixed_points, products, slot, used,
};
use crate::power::{Element, PowerField, get, set};
use crate::protocol::{Claim, chi_weights, mu_at};
use crate::witness::{BLOCK_WORDS, Recompute, Shape, TableWitness, Witness, transpose64, words};
use verifier::keccak::PRODUCT_BITS;

/// The pattern index of a binary assignment.
fn ternary(b: usize) -> usize {
    (0..SKIP).fold(0, |acc, k| acc + (b >> k & 1) * 3usize.pow(k as u32))
}

/// Digits a weight splits into, and the values of one.
const DIGIT_BITS: usize = 8;
const DIGITS: usize = 128 / DIGIT_BITS;
/// The bits of a block position's products, 48 to each pack.
pub(crate) const PAYLOAD_BITS: usize = 48 * PACKS;
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
        // SAFETY: both parts are three registers, aligned to 64, and AVX-512 is enabled at compile time.
        unsafe {
            use core::arch::x86_64::*;
            let (dst, src) = (
                self.0.as_mut_ptr().cast::<__m512i>(),
                other.0.as_ptr().cast::<__m512i>(),
            );
            for i in 0..PART_WORDS / 8 {
                let x = _mm512_xor_si512(
                    _mm512_load_si512(dst.add(i).cast()),
                    _mm512_load_si512(src.add(i).cast()),
                );
                _mm512_store_si512(dst.add(i).cast(), x);
            }
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
        // SAFETY: both parts are three registers, and AVX2 is enabled at compile time.
        unsafe {
            use core::arch::x86_64::*;
            let (dst, src) = (self.0.as_mut_ptr(), other.0.as_ptr());
            for i in (0..PART_WORDS).step_by(4) {
                let (d, s) = (dst.add(i).cast::<__m256i>(), src.add(i).cast::<__m256i>());
                _mm256_storeu_si256(d, _mm256_xor_si256(_mm256_loadu_si256(d), _mm256_loadu_si256(s)));
            }
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

/// Digit `d` of a weight.
#[inline(always)]
fn digit(weight: F128, d: usize) -> usize {
    let w = weight.to_limbs();
    (w[d / 8] >> (DIGIT_BITS * (d % 8))) as usize & ((1 << DIGIT_BITS) - 1)
}

/// The positions grouped by each digit of their weight: those whose digit `d` is `v` are
/// `order[d][starts[d][v]..starts[d][v + 1]]`.
struct Groups {
    order: Vec<[u16; BITS]>,
    starts: Vec<[u16; (1 << DIGIT_BITS) + 1]>,
}

impl Groups {
    fn new(weights: &[F128]) -> Self {
        let mut order = vec![[0u16; BITS]; DIGITS];
        let mut starts = vec![[0u16; (1 << DIGIT_BITS) + 1]; DIGITS];
        for (d, (order, starts)) in order.iter_mut().zip(&mut starts).enumerate() {
            let digits: Vec<usize> = weights[..BITS].iter().map(|&w| digit(w, d)).collect();
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
    fn add_rows(&self, payloads: &[Part; BITS], rows: &mut [Part; 128], sums: &mut [Part; 1 << DIGIT_BITS]) {
        for (rows, (order, starts)) in rows
            .as_chunks_mut::<DIGIT_BITS>()
            .0
            .iter_mut()
            .zip(self.order.iter().zip(&self.starts))
        {
            for (v, sum) in sums.iter_mut().enumerate() {
                let mut acc = Part::ZERO;
                for &p in &order[starts[v] as usize..starts[v + 1] as usize] {
                    // SAFETY: `order` holds positions.
                    acc.add(unsafe { payloads.get_unchecked(p as usize) });
                }
                *sum = acc;
            }
            // Bit `i` of the digit: the sums whose value has it, then fold it away.
            for (i, row) in rows.iter_mut().enumerate().rev() {
                let (low, high) = sums[..2 << i].split_at_mut(1 << i);
                let mut acc = Part::ZERO;
                for (low, high) in low.iter_mut().zip(&*high) {
                    acc.add(high);
                    low.add(high);
                }
                *row = acc;
            }
        }
    }
}

struct Layer<'a> {
    witness: &'a Witness,
    /// The round whose `chi` each segment's claim is on.
    rounds: Vec<usize>,
    /// The tasks of segment `s` are `held[s]..held[s + 1]`: the blocks of its table written, then the
    /// padding block once, standing for every block after them. Tables are written to different
    /// heights, so the segments differ in length.
    held: Vec<usize>,
    /// The tasks of segment `s` that walk written blocks only: `written[s]..written[s + 1]`.
    written: Vec<usize>,
}

/// A row of a table in the power basis: two words per position, laid out by word.
const ROW: usize = 2 * BITS;

/// The most segments a layer walks: four tables of the runs of a permutation, at depth 6.
const MAX_SEGMENTS: usize = 16;

impl<'a> Layer<'a> {
    fn new(witness: &'a Witness, rounds: Vec<usize>) -> Self {
        let shape = witness.shape;
        let tables: Vec<_> = (0..rounds.len()).map(|s| witness.table(shape.table_of(s))).collect();
        let prefix = |len: &dyn Fn(&TableWitness) -> usize| {
            let mut sums = vec![0];
            for table in &tables {
                sums.push(sums.last().unwrap() + len(table));
            }
            sums
        };
        Self {
            witness,
            rounds,
            held: prefix(&|t| t.held()),
            written: prefix(&|t| t.written()),
        }
    }

    /// The witness of the table a segment belongs to.
    fn table_of(&self, segment: usize) -> &TableWitness {
        self.witness.table(self.witness.shape.table_of(segment))
    }

    /// The blocks written of segment `segment`'s table.
    fn written_of(&self, segment: usize) -> usize {
        self.written[segment + 1] - self.written[segment]
    }

    /// The blocks segment `segment` walks.
    fn held_of(&self, segment: usize) -> usize {
        self.held[segment + 1] - self.held[segment]
    }

    /// Task `held[segment] + g`, as `(segment, g, block)`: stored, or recomputed into `scratch`.
    fn block<'b>(&'b self, scratch: &'b mut Recompute, task: usize) -> (usize, usize, &'b [u64]) {
        let segment = self.held.partition_point(|&start| start <= task) - 1;
        let g = task - self.held[segment];
        (
            segment,
            g,
            self.table_of(segment).state(scratch, self.rounds[segment], g),
        )
    }

    fn n_tasks(&self) -> usize {
        *self.held.last().unwrap()
    }

    /// The tasks that walk written blocks only, `(segment, g)` of task `task`.
    fn written_task(&self, task: usize) -> (usize, usize) {
        let segment = self.written.partition_point(|&start| start <= task) - 1;
        (segment, task - self.written[segment])
    }

    fn n_written_tasks(&self) -> usize {
        *self.written.last().unwrap()
    }

    /// Each task's weight in `eq`, a table over the blocks then the segment: the padding block's
    /// is the sum over the blocks it stands for.
    fn task_weights(&self, eq: &[F128]) -> Vec<F128> {
        let nb = self.witness.shape.n_blocks();
        let mut weights = Vec::with_capacity(self.n_tasks());
        for segment in 0..self.rounds.len() {
            let eq = &eq[nb * segment..][..nb];
            let written = self.written_of(segment);
            weights.extend((0..self.held_of(segment)).map(|g| if g < written { eq[g] } else { eq[g..].iter().sum() }));
        }
        weights
    }

    /// Per pack `q` of a block, the coefficients of
    /// `sum over blocks and positions of eq(point[5..], block) * weights[j] * product(q, j)`,
    /// where the product is `ones * u[j] + (ones + u[j + 1]) * u[j + 2]` on embeddings, the
    /// unprojected `chi`.
    ///
    /// Nothing here knows where the polynomials will be evaluated, so each product meets its
    /// weight bit by bit. The position weights are the same in every block, so a block's sum
    /// over positions is taken bit by bit, by GFNI outer products where they are compiled
    /// (`products_gfni`) and by digit buckets elsewhere ([`Groups`]); only that sum meets
    /// the block's weight, as field elements.
    fn pack_products(&self, claim: &Claim) -> Vec<[F128; PRODUCT_BITS]> {
        cfg_select! {
            all(target_arch = "x86_64", target_feature = "avx2", target_feature = "gfni") => {
                self.pack_products_gfni(claim)
            }
            _ => { self.pack_products_buckets(claim) }
        }
    }

    /// [`Self::pack_products`] by GFNI outer products (`products_gfni`): the same sums.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "gfni"))]
    fn pack_products_gfni(&self, claim: &Claim) -> Vec<[F128; PRODUCT_BITS]> {
        use crate::products_gfni::ProductSums;
        let weights = self.task_weights(&eq_table(&claim.point[SKIP + 1..]));
        let kernel = ProductSums::new(&claim.weights);
        let sums = parallel::map_reduce_with_state(
            self.n_tasks(),
            || (kernel.scratch(), Recompute::new()),
            || vec![F128Unreduced::ZERO; PAYLOAD_BITS],
            |(scratch, recompute), sums, task| {
                let (_, _, block) = self.block(recompute, task);
                kernel.block_sums(scratch, block);
                for (sum, v) in sums.iter_mut().zip(scratch.out.iter()) {
                    *sum ^= weights[task].mul_unreduced(F128::from_limbs(*v));
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

    /// [`Self::pack_products`] by digit buckets: the block's positions grouped by each digit of
    /// their weight ([`Groups`]), a part of the packs at a time. The GFNI form's test oracle.
    #[cfg_attr(
        all(target_arch = "x86_64", target_feature = "avx2", target_feature = "gfni"),
        allow(dead_code)
    )]
    fn pack_products_buckets(&self, claim: &Claim) -> Vec<[F128; PRODUCT_BITS]> {
        let weights = self.task_weights(&eq_table(&claim.point[SKIP + 1..]));
        let groups = Groups::new(&claim.weights);
        let sums = parallel::map_reduce_with_state(
            self.n_tasks(),
            || {
                (
                    vec![[Part::ZERO; BITS]; PACKS / PART],
                    vec![[Part::ZERO; 128]; PACKS / PART],
                    Box::new([Part::ZERO; 1 << DIGIT_BITS]),
                    Recompute::new(),
                )
            },
            || vec![F128Unreduced::ZERO; PAYLOAD_BITS],
            |(payloads, rows, digit_sums, recompute), sums, task| {
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
                    groups.add_rows(payloads, rows, digit_sums);
                }
                // Row `r` holds bit `r` of each payload bit's weighted sum: transposed, they are
                // field elements, which the block's weight multiplies.
                let mut columns = [[0u64; 64]; 2];
                for m in 0..PAYLOAD_BITS / 64 {
                    let (q, w) = (m / PART_WORDS, m % PART_WORDS);
                    for (k, column) in columns.iter_mut().enumerate() {
                        for (b, word) in column.iter_mut().enumerate() {
                            *word = rows[q][64 * k + b].0[w];
                        }
                        transpose64(column);
                    }
                    for (t, sum) in sums[64 * m..][..64].iter_mut().enumerate() {
                        *sum ^= weights[task].mul_unreduced(F128::new(columns[0][t], columns[1][t]));
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
    fn skip_sums(
        &self,
        claim: &Claim,
        weights: &[Element],
        field: &PowerField,
        s: F128,
        binary: &[F128; GROUP],
    ) -> Vec<F128> {
        let groups: Vec<Element> = eq_table(&claim.point[SKIP..])
            .iter()
            .map(|&e| field.to_poly(e))
            .collect();
        let nb = self.witness.shape.n_blocks();
        let mixed = parallel::map_reduce_with_state(
            self.n_written_tasks(),
            || (SkipBlock::new(weights), Recompute::new()),
            || vec![[0u64; 5]; MIXED],
            |(skip, recompute), total, task| {
                let (segment, g) = self.written_task(task);
                let (row, block) = (
                    g + nb * segment,
                    self.table_of(segment).state(recompute, self.rounds[segment], g),
                );
                skip.run(block);
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
        let eval = PolyEval::new(s, 40);
        let mut sums = vec![F128::ZERO; TERNARY];
        for (slot, p) in mixed_points().into_iter().enumerate() {
            sums[p] = eval.eval(&mixed[slot]);
        }
        for (q, &value) in binary.iter().enumerate() {
            sums[ternary(q)] = value;
        }
        sums
    }

    /// [`Self::skip_sums`] of the grouped rounds: the sums over the points of `{0, 1, infinity}^(SKIP - 1)`
    /// for each of the two groups of a half, the fourth variable binary. `binary[side][q]` is the sum of
    /// pack `q` of group `side`. The points the fourth variable would take at infinity are not summed.
    fn skip_sums_grouped(
        &self,
        claim: &Claim,
        weights: &[Element],
        field: &PowerField,
        s: F128,
        binary: &[[F128; GROUP / 2]; 2],
    ) -> [Vec<F128>; 2] {
        const POINTS: usize = TERNARY / 3;
        let groups: Vec<Element> = eq_table(&claim.point[SKIP..])
            .iter()
            .map(|&e| field.to_poly(e))
            .collect();
        let points: Vec<(usize, usize, usize)> = mixed_points()
            .into_iter()
            .enumerate()
            .filter(|&(_, p)| used(p))
            .map(|(i, p)| (i, p / POINTS, p % POINTS))
            .collect();
        let nb = self.witness.shape.n_blocks();
        let mixed = parallel::map_reduce_with_state(
            self.n_written_tasks(),
            || (SkipBlock::new(weights), Recompute::new()),
            || std::array::from_fn::<_, 2, _>(|_| vec![[0u64; 5]; POINTS]),
            |(skip, recompute), total, task| {
                let (segment, g) = self.written_task(task);
                let (row, block) = (
                    g + nb * segment,
                    self.table_of(segment).state(recompute, self.rounds[segment], g),
                );
                skip.run(block);
                for half in 0..2 {
                    for &(i, side, p) in &points {
                        add_mul(&mut total[side][p], &skip.get(slot(half, i)), &groups[2 * row + half]);
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
        let eval = PolyEval::new(s, 40);
        std::array::from_fn(|side| {
            let mut sums = vec![F128::ZERO; POINTS];
            for &(_, _, p) in points.iter().filter(|&&(_, group, _)| group == side) {
                sums[p] = eval.eval(&mixed[side][p]);
            }
            for (q, &value) in binary[side].iter().enumerate() {
                sums[ternary(q)] = value;
            }
            sums
        })
    }

    /// The fourth round's quadratic coefficient of the grouped rounds, its values at zero and one having
    /// ridden [`Self::skip_sums_grouped`]: the sum over blocks, halves and positions of
    /// `weights[j] * D[j + 1] * D[j + 2]`, with `D` the difference of the two groups' folds at `challenges`.
    /// A padding block's groups are equal, so it adds nothing.
    fn group_infinity(&self, round: &Round, challenges: &[F128], point: &[F128]) -> F128 {
        let field = round.field;
        let differences = Differences::new(field, challenges);
        let groups: Vec<Element> = eq_table(&point[SKIP..]).iter().map(|&e| field.to_poly(e)).collect();
        // The weights by lane, then by word.
        let mut lane_weights = vec![0u64; BITS / 64 * LANE_ROW];
        for (j, w) in lane_weights.as_chunks_mut::<LANE_ROW>().0.iter_mut().enumerate() {
            for k in 0..2 {
                w[64 * k..][..64].copy_from_slice(&round.weights[k * BITS + 64 * j..][..64]);
            }
        }
        let nb = self.witness.shape.n_blocks();
        let sum = parallel::map_reduce_with_state(
            self.n_written_tasks(),
            || (vec![0u64; 10 * LANE_ROW], vec![0u64; 10 * 3 * 64], Recompute::new()),
            || [0u64; 6],
            |(rows, wide, recompute), total, task| {
                let (segment, g) = self.written_task(task);
                let block = self.table_of(segment).state(recompute, self.rounds[segment], g);
                let mut sums = [[0u64; 4]; 2];
                for r in 0..5 {
                    differences.rows(field, &block[r * 5 * 64 * BLOCK_WORDS..], rows, wide);
                    for (half, sum) in sums.iter_mut().enumerate() {
                        let lanes = &rows[5 * half * LANE_ROW..][..5 * LANE_ROW];
                        for x in 0..5 {
                            let a = &lanes[(x + 1) % 5 * LANE_ROW..][..LANE_ROW];
                            let b = &lanes[(x + 2) % 5 * LANE_ROW..][..LANE_ROW];
                            let w = &lane_weights[(5 * r + x) * LANE_ROW..][..LANE_ROW];
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

    /// The table the remaining rounds fold, a row per group: `T[j]`, the value at `s` of position
    /// `j`'s embeddings, folded over the group's packs at `challenges`. A block's two groups are
    /// the first of those rounds' pairs, so its round values are summed while the rows are at hand.
    /// The padding block's groups are one row, its segment's entry of `pads`.
    fn table(
        &self,
        round: &Round,
        challenges: &[F128],
        eq: &[F128],
        layout: Rows,
        table: &mut [u64],
        pads: &mut [u64],
    ) -> (F128, F128) {
        let builder = RowBuilder::new(round.field, challenges);
        let weights = self.task_weights(eq);
        // Segments that do not exist have no live row, and their padding row is zero, and so is their product.
        pads[self.rounds.len() * ROW..].fill(0);
        let (ptr, pads) = (
            parallel::SendPtr(table.as_mut_ptr()),
            parallel::SendPtr(pads.as_mut_ptr()),
        );
        let sums = parallel::map_reduce_with_state(
            self.n_tasks(),
            || (builder.scratch(), vec![0u64; 2 * ROW], Recompute::new()),
            RoundSums::default,
            |(wide, rows, recompute), sums, task| {
                let (segment, g, block) = self.block(recompute, task);
                // SAFETY: each task owns its rows: a written block's two, or its segment's padding row.
                let out = unsafe {
                    if 2 * g < layout.live[segment] {
                        ptr.slice((layout.start(segment) + 2 * g) * ROW, 2 * ROW)
                    } else {
                        pads.slice(segment * ROW, ROW)
                    }
                };
                let rows = &mut rows[..out.len()];
                builder.rows(wide, block, rows);
                round.add_pair(sums, &rows[..ROW], &rows[rows.len() - ROW..], weights[task]);
                // Built in cache for the round values, then published: the next round reads them.
                // A normal copy, not a non-temporal one: the next pass reads these rows from cache.
                out.copy_from_slice(rows);
            },
            RoundSums::merge,
        );
        round.finish(&sums)
    }
}

/// Words of one lane's row in the power basis: 64 positions, two words each, laid out by word.
pub(crate) const LANE_ROW: usize = 2 * 64;

/// The folds of the differences of a block's two groups, in each half, at the challenges of the
/// grouped rounds: the first `SKIP - 1` variables' fold of the packs of a group.
struct Differences {
    /// The fold's weights, a half's first eight packs, the other eight weighing zero.
    weights: [[u64; GROUP]; 2],
    /// The same fold as nibble tables, which AVX2 looks up faster than it multiplies.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
    shuffle: crate::shuffle_fold::ShuffleFold,
}

impl Differences {
    fn new(field: &PowerField, challenges: &[F128]) -> Self {
        assert_eq!(challenges.len(), SKIP - 1);
        let fold: Vec<Element> = eq_table(challenges).iter().map(|&e| field.to_poly(e)).collect();
        Self {
            weights: std::array::from_fn(|k| std::array::from_fn(|q| fold.get(q).map_or(0, |e| e[k]))),
            // Input byte `i` of a lane is a byte of pack `i / 2` of the group.
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
            shuffle: crate::shuffle_fold::ShuffleFold::new(field, |byte| fold[byte / 2 % fold.len()]),
        }
    }

    /// The differences of the five lanes of one row of `block` (its first `5 * 64` positions): ten
    /// rows in `rows`, those of lane `x` of half `h` at `5 * h + x`. `wide` is their unreduced sums.
    fn rows(&self, field: &PowerField, block: &[u64], rows: &mut [u64], wide: &mut [u64]) {
        cfg_select! {
            all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")) => {
                let _ = (field, wide);
                self.shuffle.differences(block, rows);
            }
            _ => { self.rows_clmul(field, block, rows, wide) }
        }
    }

    /// [`Self::rows`] by carry-less products, which the nibble tables replace on AVX2.
    #[cfg_attr(
        all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")),
        allow(dead_code)
    )]
    fn rows_clmul(&self, field: &PowerField, block: &[u64], rows: &mut [u64], wide: &mut [u64]) {
        for x in 0..5 {
            for z in 0..64 {
                let w = words(block, 64 * x + z);
                // Words 0 and 1 of a half are its first group's packs, 2 and 3 its second's.
                let packed = [w[0] ^ w[2], w[1] ^ w[3], 0, 0, w[4] ^ w[6], w[5] ^ w[7], 0, 0];
                for (half, folded) in fold_groups(&embed(&packed), &self.weights).iter().enumerate() {
                    for (k, word) in carry::<2, 3>(folded).into_iter().enumerate() {
                        wide[(5 * half + x) * 3 * 64 + k * 64 + z] = word;
                    }
                }
            }
        }
        for (row, wide) in rows
            .as_chunks_mut::<LANE_ROW>()
            .0
            .iter_mut()
            .zip(wide.as_chunks::<{ 3 * 64 }>().0)
        {
            field.reduce_row(row, wide, 64);
        }
    }
}

/// The rows of one block's groups: `T[j]`, the value at `s` of position `j`'s embeddings, folded
/// over a group's packs at the bit-level rounds' challenges.
struct RowBuilder<'a> {
    // The GFNI rows reduce nothing.
    #[cfg_attr(
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "gfni"
        ),
        allow(dead_code)
    )]
    field: &'a PowerField,
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "gfni"
    )))]
    fold: [[u64; GROUP]; 2],
    /// The same fold as nibble tables: AVX2 looks a pack up faster than it multiplies it.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
    shuffle: crate::shuffle_fold::ShuffleFold,
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "gfni"
    ))]
    row_maps: crate::table_gfni::RowMaps,
}

impl<'a> RowBuilder<'a> {
    fn new(field: &'a PowerField, challenges: &[F128]) -> Self {
        let weights: Vec<Element> = eq_table(challenges).iter().map(|&e| field.to_poly(e)).collect();
        let fold: [[u64; GROUP]; 2] = std::array::from_fn(|k| std::array::from_fn(|q| weights[q][k]));
        cfg_select! {
            all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw", target_feature = "gfni") => {
                Self { field, row_maps: crate::table_gfni::RowMaps::new(field, &fold) }
            }
            all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")) => {
                // Input byte `i` of a lane is a byte of pack `i / 2` of the group.
                Self { field, fold, shuffle: crate::shuffle_fold::ShuffleFold::new(field, |byte| weights[byte / 2]) }
            }
            _ => { Self { field, fold } }
        }
    }

    /// A worker's wide scratch: the GFNI rows need none.
    fn scratch(&self) -> Vec<u64> {
        let gfni = cfg!(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "gfni"
        ));
        vec![0u64; if gfni { 0 } else { 6 * BITS }]
    }

    /// The rows of `block`: one per half in `rows`, which holds a row or two.
    fn rows(&self, wide: &mut [u64], block: &[u64], rows: &mut [u64]) {
        let halves = rows.len() / ROW;
        cfg_select! {
            all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw", target_feature = "gfni") => {
                // Each group's value by affine byte maps (`table_gfni`).
                let _ = wide;
                self.row_maps.rows(block, rows, halves);
            }
            _ => {
                // A written block's two rows by nibble tables, where there are any.
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
                if halves == 2 {
                    self.shuffle.rows(block, rows);
                    return;
                }
                // Both groups' unreduced sums, three words each and laid out by word, then reduced.
                for j in 0..BITS {
                    let sums = crate::kernels::fold_groups(&embed(words(block, j)), &self.fold);
                    for (wide, sum) in wide.as_chunks_mut::<{ 3 * BITS }>().0[..halves].iter_mut().zip(&sums) {
                        let sum: [u64; 3] = crate::basis::carry(sum);
                        for (k, &word) in sum.iter().enumerate() {
                            wide[k * BITS + j] = word;
                        }
                    }
                }
                for (t, wide) in rows.as_chunks_mut::<ROW>().0.iter_mut().zip(wide.as_chunks::<{ 3 * BITS }>().0) {
                    self.field.reduce_row(t, wide, BITS);
                }
            }
        }
    }
}

/// The sums a round accumulates, unreduced: its values at `0` and at infinity.
#[derive(Clone, Copy, Default)]
struct RoundSums([u64; 6], [u64; 6]);

impl RoundSums {
    fn merge(mut self, other: Self) -> Self {
        for i in 0..6 {
            self.0[i] ^= other.0[i];
            self.1[i] ^= other.1[i];
        }
        self
    }
}

/// What a round needs of the claim, by word as a table row is: the position weights, and each
/// position's constant `kappa * (weights[j] + weights[j + 2])`.
struct Round<'a> {
    field: &'a PowerField,
    eval: PolyEval,
    weights: Vec<u64>,
    constant: Vec<u64>,
    /// The constant of a difference of rows, which is linear in them: none.
    linear: Vec<u64>,
}

impl<'a> Round<'a> {
    /// `polys` are the claim's weights in the power basis.
    fn new(field: &'a PowerField, claim: &Claim, polys: &[Element], s: F128, kappa: F128) -> Self {
        let w = &claim.weights;
        let (mut weights, mut constant) = (vec![0u64; ROW], vec![0u64; ROW]);
        for j in 0..BITS {
            set(&mut weights, BITS, j, &polys[j]);
            set(&mut constant, BITS, j, &field.to_poly(kappa * (w[j] + w[along(j, 2)])));
        }
        Self {
            field,
            eval: PolyEval::new(s, 48),
            weights,
            constant,
            linear: vec![0u64; ROW],
        }
    }

    /// The values at `0` and at infinity of `sum over positions of B[j] * T[j + 2]`, unreduced,
    /// over a pair of rows, where `B[j] = weights[j] * T[j + 1] + constant[j]`.
    fn pair_sums(&self, constant: &[u64], rows: [&[u64]; 2]) -> ([u64; 4], [u64; 4]) {
        // A lane is 64 positions, five lanes to a row.
        let shifted: [_; BITS / 64] = std::array::from_fn(|i| (along(64 * i, 1), along(64 * i, 2)));
        self.field.pair_sums(&self.weights, constant, rows, BITS, 64, &shifted)
    }

    /// Add one pair of rows to a round's sums: the values at `0` and at infinity of
    /// `sum over positions of B[j] * T[j + 2]`, weighted by `eq`, where
    /// `B[j] = weights[j] * (kappa + T[j + 1]) + kappa * weights[j + 2]`, which is
    /// `weights[j] * T[j + 1] + constant[j]`.
    fn add_pair(&self, sums: &mut RoundSums, low: &[u64], high: &[u64], eq: F128) {
        let (zero, infinity) = self.pair_sums(&self.constant, [low, high]);
        let weight = self.field.to_poly(eq);
        add_mul(&mut sums.0, &zero, &weight);
        add_mul(&mut sums.1, &infinity, &weight);
    }

    fn finish(&self, sums: &RoundSums) -> (F128, F128) {
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
        eq: &[F128],
    ) -> (F128, F128) {
        let to = from.next();
        let ptr = parallel::SendPtr(dst.as_mut_ptr());
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
            for (pair, &eq) in dst[..to.len() * ROW].as_chunks::<{ 2 * ROW }>().0.iter().zip(eq) {
                self.add_pair(&mut sums, &pair[..ROW], &pair[ROW..], eq);
            }
            return self.finish(&sums);
        }
        // A task per pair of rows written, the second of a ragged one being the padding row.
        // Task `pair_starts[segment] + t` owns rows `2t` and `2t + 1` of its segment. A segment with no
        // block written has no live row and no task.
        let half = to.m / 2;
        let mut pair_starts = vec![0];
        for segment in 0..to.segments {
            pair_starts.push(pair_starts[segment] + to.live[segment].div_ceil(2));
        }
        let mut sums = parallel::map_reduce_with_state(
            pair_starts[to.segments],
            || vec![0u64; 2 * ROW],
            RoundSums::default,
            |rows, sums, task| {
                let segment = pair_starts.partition_point(|&start| start <= task) - 1;
                let t = task - pair_starts[segment];
                let n = (to.live[segment] - 2 * t).min(2);
                let rows = &mut rows[..n * ROW];
                for (i, row) in rows.as_chunks_mut::<ROW>().0.iter_mut().enumerate() {
                    fold(row, segment, 2 * t + i);
                }
                let high = if n == 2 {
                    &rows[ROW..]
                } else {
                    &pads[segment * ROW..][..ROW]
                };
                self.add_pair(sums, &rows[..ROW], high, eq[half * segment + t]);
                // SAFETY: each task owns its rows of `dst`.
                let out = unsafe { ptr.slice((to.start(segment) + 2 * t) * ROW, n * ROW) };
                // Built in cache for the round values, then published: the next round reads them.
                // A normal copy, not a non-temporal one: the next pass reads these rows from cache.
                out.copy_from_slice(rows);
            },
            RoundSums::merge,
        );
        self.add_padding(&mut sums, to, pads, eq);
        self.finish(&sums)
    }

    /// Add the pairs of two padding rows of a table laid out as `layout`, weighted by `eq`: each
    /// segment's padding row, once, at the sum of their weights.
    fn add_padding(&self, sums: &mut RoundSums, layout: Rows, pads: &[u64], eq: &[F128]) {
        let half = layout.m / 2;
        for (segment, (pad, eq)) in pads.as_chunks::<ROW>().0.iter().zip(eq.chunks_exact(half)).enumerate() {
            let first = layout.live[segment].div_ceil(2);
            if first != half {
                self.add_pair(sums, pad, pad, eq[first..].iter().sum());
            }
        }
    }
}

/// The rows of the table the tail folds: per segment, the first `live[segment]` of `m`, then, if there
/// are fewer, the padding row every later one equals, kept apart. A segment's rows follow the previous
/// segment's, whatever their number: the tables are written to different heights. Once `m` is one, a
/// row per segment.
#[derive(Clone, Copy)]
struct Rows {
    m: usize,
    live: [usize; MAX_SEGMENTS],
    segments: usize,
}

impl Rows {
    fn len(&self) -> usize {
        self.live[..self.segments].iter().sum()
    }

    /// The first row of `segment`.
    fn start(&self, segment: usize) -> usize {
        self.live[..segment].iter().sum()
    }

    /// The layout once the next variable is bound: the group, the block's, then the segment's.
    fn next(&self) -> Self {
        match self.m {
            1 => Self {
                segments: self.segments / 2,
                ..*self
            },
            // The fold writes every segment's row, a padding one included.
            2 => Self {
                m: 1,
                live: [1; MAX_SEGMENTS],
                ..*self
            },
            m => Self {
                m: m / 2,
                live: self.live.map(|live| live.div_ceil(2)),
                ..*self
            },
        }
    }

    fn row<'t>(&self, table: &'t [u64], pads: &'t [u64], segment: usize, r: usize) -> &'t [u64] {
        if r < self.live[segment] {
            &table[(self.start(segment) + r) * ROW..][..ROW]
        } else {
            &pads[segment * ROW..][..ROW]
        }
    }
}

// The paired tail: a pass over the table binds two variables, which reads it half as often. A
// pass folds four rows to one with one reduction, and its messages are the same as two single
// rounds'. It takes a third more row sums than single rounds do: where those are cheap, with
// AVX-512, it wins; with AVX2 it did not.

const PAIRED_TAIL: bool = cfg!(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
));

impl Rows {
    fn paired(&self) -> bool {
        PAIRED_TAIL && self.m >= 4
    }
}

#[derive(Clone, Copy, Default)]
struct PairSums([[u64; 6]; 8]);

impl PairSums {
    fn merge(mut self, other: Self) -> Self {
        for (a, b) in self.0.iter_mut().zip(other.0) {
            for (a, b) in a.iter_mut().zip(b) {
                *a ^= b;
            }
        }
        self
    }
}

/// The eight bivariate values needed to send two ordinary round messages: chi at
/// `(0, 0), (inf, 0), (0, 1), (inf, 1), (1, 0), (1, inf), (0, inf), (inf, inf)`, infinity
/// selecting the leading coefficient in that variable.
struct RoundPair([F128; 8]);

impl RoundPair {
    /// The first message's values at `0` and infinity, the second variable summed at `eta`.
    fn first(&self, eta: F128) -> (F128, F128) {
        let q = &self.0;
        (q[0] + eta * (q[0] + q[2]), q[1] + eta * (q[1] + q[3]))
    }

    /// The second message's values, the first variable bound to `r`.
    fn second(&self, r: F128) -> (F128, F128) {
        let q = &self.0;
        let r2 = r.square() + r;
        (
            q[0] + r * (q[0] + q[4]) + r2 * q[1],
            q[6] + r * (q[6] + q[5]) + r2 * q[7],
        )
    }
}

enum TailMessage {
    Single(F128, F128),
    Pair(RoundPair),
}

impl TailMessage {
    fn first(&self, eta: F128) -> (F128, F128) {
        match self {
            Self::Single(zero, infinity) => (*zero, *infinity),
            Self::Pair(pair) => pair.first(eta),
        }
    }
}

impl Layer<'_> {
    /// [`Layer::table`], summing the values of the first two rounds over each four rows.
    fn table_pair(
        &self,
        round: &Round,
        challenges: &[F128],
        eq: &[F128],
        layout: Rows,
        table: &mut [u64],
        pads: &mut [u64],
    ) -> RoundPair {
        let builder = RowBuilder::new(round.field, challenges);
        // Segments that do not exist have no live row, and their padding row is zero, and so is their product.
        pads[self.rounds.len() * ROW..].fill(0);
        {
            let (mut recompute, mut wide) = (Recompute::new(), builder.scratch());
            for (segment, &round) in self.rounds.iter().enumerate() {
                let written = self.written_of(segment);
                if written < self.witness.shape.n_blocks() {
                    let block = self.table_of(segment).state(&mut recompute, round, written);
                    builder.rows(&mut wide, block, &mut pads[segment * ROW..][..ROW]);
                }
            }
        }
        let pads = &*pads;
        let ptr = parallel::SendPtr(table.as_mut_ptr());
        // Task `quad_starts[segment] + q` owns rows `4q ..` of its segment, as in `Round::fold`.
        let mut quad_starts = vec![0];
        for segment in 0..self.rounds.len() {
            quad_starts.push(quad_starts[segment] + layout.live[segment].div_ceil(4));
        }
        let mut sums = parallel::map_reduce_with_state(
            quad_starts[self.rounds.len()],
            || {
                (
                    builder.scratch(),
                    vec![0u64; 4 * ROW],
                    vec![0u64; 2 * ROW],
                    Recompute::new(),
                )
            },
            PairSums::default,
            |(wide, rows, diffs, recompute), sums, task| {
                let segment = quad_starts.partition_point(|&start| start <= task) - 1;
                let q = task - quad_starts[segment];
                // A ragged last quad has two rows written, the others are the padding row.
                let n = (layout.live[segment] - 4 * q).min(4);
                let rows = &mut rows[..n * ROW];
                for (i, rows) in rows.chunks_mut(2 * ROW).enumerate() {
                    let block = self.table_of(segment).state(recompute, self.rounds[segment], 2 * q + i);
                    builder.rows(wide, block, rows);
                }
                round.add_quad(
                    sums,
                    diffs,
                    std::array::from_fn(|i| {
                        if i < n {
                            &rows[i * ROW..][..ROW]
                        } else {
                            &pads[segment * ROW..][..ROW]
                        }
                    }),
                    eq[layout.m / 4 * segment + q],
                );
                // SAFETY: each task owns its rows.
                let out = unsafe { ptr.slice((layout.start(segment) + 4 * q) * ROW, n * ROW) };
                // Built in cache for the round values, then published: the next round reads them.
                // A normal copy, not a non-temporal one: the next pass reads these rows from cache.
                out.copy_from_slice(rows);
            },
            PairSums::merge,
        );
        round.add_padding_quad(&mut sums, layout, pads, eq);
        round.finish_pair(&sums)
    }
}

impl Round<'_> {
    /// Add four rows, indexed `x + 2y`, to a pair round's sums, weighted by `eq`: each of the eight
    /// values of [`RoundPair`] is a dot product of [`Self::pair_sums`]'s kind, the last two over
    /// the sums of rows `0` and `2`, and `1` and `3`.
    fn add_quad(&self, sums: &mut PairSums, diffs: &mut [u64], rows: [&[u64]; 4], eq: F128) {
        let (d0, d1) = diffs.split_at_mut(ROW);
        for (k, d) in d0.iter_mut().enumerate() {
            *d = rows[0][k] ^ rows[2][k];
        }
        for (k, d) in d1[..ROW].iter_mut().enumerate() {
            *d = rows[1][k] ^ rows[3][k];
        }
        let values = [
            self.pair_sums(&self.constant, [rows[0], rows[1]]),
            self.pair_sums(&self.constant, [rows[2], rows[3]]),
            self.pair_sums(&self.constant, [rows[1], rows[3]]),
            self.pair_sums(&self.linear, [d0, &d1[..ROW]]),
        ];
        let weight = self.field.to_poly(eq);
        for (sum, (zero, infinity)) in sums.0.chunks_mut(2).zip(values) {
            add_mul(&mut sum[0], &zero, &weight);
            add_mul(&mut sum[1], &infinity, &weight);
        }
    }

    fn finish_pair(&self, sums: &PairSums) -> RoundPair {
        RoundPair(sums.0.map(|sum| self.eval.eval(&sum)))
    }

    /// [`Round::fold`] of two variables at once.
    fn fold_pair(
        &self,
        src: &[u64],
        from: Rows,
        dst: &mut [u64],
        pads: &[u64],
        challenges: &[Element; 2],
        eq: &[F128],
    ) -> TailMessage {
        let to = from.next().next();
        let challenges = [
            challenges[0],
            challenges[1],
            self.field.mul(&challenges[0], &challenges[1]),
        ];
        let fold = |out: &mut [u64], segment: usize, r: usize| {
            self.field.fold_quad(
                out,
                std::array::from_fn(|i| from.row(src, pads, segment, 4 * r + i)),
                &challenges,
                BITS,
            );
        };
        if to.m < 4 {
            // Few enough rows to fold in place, the padding rows being their own folds.
            for segment in 0..to.segments {
                for r in 0..to.live[segment] {
                    fold(&mut dst[(to.start(segment) + r) * ROW..][..ROW], segment, r);
                }
            }
            let mut sums = RoundSums::default();
            if to.m == 1 {
                for (pair, &eq) in dst[..to.len() * ROW].as_chunks::<{ 2 * ROW }>().0.iter().zip(eq) {
                    self.add_pair(&mut sums, &pair[..ROW], &pair[ROW..], eq);
                }
            } else {
                for (segment, &eq) in eq.iter().enumerate() {
                    self.add_pair(
                        &mut sums,
                        to.row(dst, pads, segment, 0),
                        to.row(dst, pads, segment, 1),
                        eq,
                    );
                }
            }
            let (zero, infinity) = self.finish(&sums);
            return TailMessage::Single(zero, infinity);
        }
        // A task per quad of rows written, the last of a segment possibly ragged. Task
        // `quad_starts[segment] + q` owns rows `4q ..` of its segment, which may have none.
        let half = to.m / 4;
        let mut quad_starts = vec![0];
        for segment in 0..to.segments {
            quad_starts.push(quad_starts[segment] + to.live[segment].div_ceil(4));
        }
        let ptr = parallel::SendPtr(dst.as_mut_ptr());
        let mut sums = parallel::map_reduce_with_state(
            quad_starts[to.segments],
            || (vec![0u64; 4 * ROW], vec![0u64; 2 * ROW]),
            PairSums::default,
            |(rows, diffs), sums, task| {
                let segment = quad_starts.partition_point(|&start| start <= task) - 1;
                let q = task - quad_starts[segment];
                let n = (to.live[segment] - 4 * q).min(4);
                let rows = &mut rows[..n * ROW];
                for (i, row) in rows.as_chunks_mut::<ROW>().0.iter_mut().enumerate() {
                    fold(row, segment, 4 * q + i);
                }
                self.add_quad(
                    sums,
                    diffs,
                    std::array::from_fn(|i| {
                        if i < n {
                            &rows[i * ROW..][..ROW]
                        } else {
                            &pads[segment * ROW..][..ROW]
                        }
                    }),
                    eq[half * segment + q],
                );
                // SAFETY: each task owns its rows of `dst`.
                let out = unsafe { ptr.slice((to.start(segment) + 4 * q) * ROW, n * ROW) };
                // Built in cache for the round values, then published: the next round reads them.
                // A normal copy, not a non-temporal one: the next pass reads these rows from cache.
                out.copy_from_slice(rows);
            },
            PairSums::merge,
        );
        self.add_padding_quad(&mut sums, to, pads, eq);
        TailMessage::Pair(self.finish_pair(&sums))
    }

    /// Add the quads of four padding rows of a table laid out as `layout`, weighted by `eq`: each
    /// segment's padding row, once, at the sum of their weights.
    fn add_padding_quad(&self, sums: &mut PairSums, layout: Rows, pads: &[u64], eq: &[F128]) {
        let quads = layout.m / 4;
        let mut diffs = vec![0u64; 2 * ROW];
        for (segment, (pad, eq)) in pads.as_chunks::<ROW>().0.iter().zip(eq.chunks_exact(quads)).enumerate() {
            let first = layout.live[segment].div_ceil(4);
            if first != quads {
                self.add_quad(sums, &mut diffs, [pad; 4], eq[first..].iter().sum());
            }
        }
    }
}

/// A round of the sumcheck over bit positions, on small tables in the standard basis, entry `j`
/// holding `w, t0, t1, t2` there: the coefficients of `sum_j w (kappa t0 + (kappa + t1) t2)` in
/// the variable being bound.
fn local_round(tables: &[[F128; 4]], kappa: F128) -> [F128; 4] {
    let mut h = [F128::ZERO; 4];
    for pair in tables.as_chunks::<2>().0 {
        let [w, t0, t1, t2] = std::array::from_fn(|k| (pair[0][k], pair[0][k] + pair[1][k]));
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
        // SAFETY: every pass writes the rows it reads.
        let table = unsafe { primitives::uninit_vec(rows * ROW + rows * ROW / 2) };
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
    prove_layer_with(ps, witness, scratch, step, claim, GROUPED)
}

/// [`prove_layer`] with the bit-level rounds [`GROUPED`] or not: the messages are the same.
/// A grouped layer needs a Skip kernel that sums the points the fourth variable takes binary.
fn prove_layer_with(
    ps: &mut ProverState,
    witness: &Witness,
    scratch: &mut Scratch,
    step: usize,
    claim: &Claim,
    grouped: bool,
) -> Claim {
    let shape = witness.shape;
    let layer = Layer::new(
        witness,
        (0..shape.segments())
            .map(|segment| shape.round(segment, step))
            .collect(),
    );

    let per_pack = tracing::info_span!("Product").in_scope(|| layer.pack_products(claim));
    let pack_weights = eq_table(&claim.point[..SKIP + 1]);
    let product: [F128; PRODUCT_BITS] =
        std::array::from_fn(|c| (0..PACKS).map(|q| pack_weights[q] * per_pack[q][c]).sum());
    ps.add_scalars(&product);
    // `s`: the product polynomial has degree PRODUCT_BITS - 1.
    // Bad event, probability `2^-64`: `s` in `K` (which holds every proper subfield) has no power basis. A verifier
    // accepts any passing nonce, so the prover grinds on.
    let bits = fiat_shamir::grinding_bits(PRODUCT_BITS as u128 - 1);
    let s = ps.grind_sample(bits, |s| s.to_limbs()[1] != 0);
    let kappa = mu_at(s).iter().sum();
    let mut value = poly_eval(&product, s);
    let field = PowerField::new(s).expect("an `s` outside `K` has a power basis");

    // The claim's weights in the power basis, which Skip and the later rounds share.
    let polys: Vec<Element> = claim.weights.iter().map(|&w| field.to_poly(w)).collect();
    let round = Round::new(&field, claim, &polys, s, kappa);
    let mut point = Vec::with_capacity(claim.point.len());
    if grouped {
        // The first SKIP - 1 rounds, the last variable of a group staying binary, then its round.
        // A pack's own sum is its product polynomial at `s`, over both halves of the block.
        let binary: [[F128; GROUP / 2]; 2] = std::array::from_fn(|side| {
            std::array::from_fn(|q| {
                let half = claim.point[SKIP];
                let q = GROUP / 2 * side + q;
                (F128::ONE + half) * poly_eval(&per_pack[q], s) + half * poly_eval(&per_pack[GROUP + q], s)
            })
        });
        let mut sums =
            tracing::info_span!("Skip").in_scope(|| layer.skip_sums_grouped(claim, &polys, &field, s, &binary));
        let side_weights = [F128::ONE + claim.point[SKIP - 1], claim.point[SKIP - 1]];
        for i in 0..SKIP - 1 {
            let later = eq_table(&claim.point[i + 1..SKIP - 1]);
            let mut h = [F128::ZERO; 3];
            for (b, &eq) in later.iter().enumerate() {
                for (d, h) in h.iter_mut().enumerate() {
                    *h += eq
                        * (side_weights[0] * sums[0][d + 3 * ternary(b)]
                            + side_weights[1] * sums[1][d + 3 * ternary(b)]);
                }
            }
            let coeffs = [h[0], h[0] + h[1] + h[2], h[2]];
            let challenge = bind(ps, &coeffs, &mut value, &mut point);
            sums = sums.map(|sums| {
                sums.as_chunks::<3>()
                    .0
                    .iter()
                    .map(|v| v[0] + challenge * (v[0] + v[1]) + (challenge.square() + challenge) * v[2])
                    .collect()
            });
        }
        let infinity = tracing::info_span!("Groups").in_scope(|| layer.group_infinity(&round, &point, &claim.point));
        let coeffs = [sums[0][0], sums[0][0] + sums[1][0] + infinity, infinity];
        bind(ps, &coeffs, &mut value, &mut point);
    } else {
        // The packs of a group, from sums of bit-level products. A pack's own sum is its product
        // polynomial at `s`, over both groups of the block.
        let binary: [F128; GROUP] = std::array::from_fn(|q| {
            let half = claim.point[SKIP];
            (F128::ONE + half) * poly_eval(&per_pack[q], s) + half * poly_eval(&per_pack[GROUP + q], s)
        });
        let mut sums = tracing::info_span!("Skip").in_scope(|| layer.skip_sums(claim, &polys, &field, s, &binary));
        for i in 0..SKIP {
            let later = eq_table(&claim.point[i + 1..SKIP]);
            let mut h = [F128::ZERO; 3];
            for (b, &eq) in later.iter().enumerate() {
                for (d, h) in h.iter_mut().enumerate() {
                    *h += eq * sums[d + 3 * ternary(b)];
                }
            }
            let coeffs = [h[0], h[0] + h[1] + h[2], h[2]];
            let challenge = bind(ps, &coeffs, &mut value, &mut point);
            sums = sums
                .as_chunks::<3>()
                .0
                .iter()
                .map(|v| v[0] + challenge * (v[0] + v[1]) + (challenge.square() + challenge) * v[2])
                .collect();
        }
    }

    // The groups, blocks and segments, on a table in the power basis. Each pass over it writes
    // the rows of one round and sums that round's values over them.
    let mut layout = Rows {
        m: 2 * shape.n_blocks(),
        live: std::array::from_fn(|segment| {
            if segment < shape.segments() {
                2 * layer.written_of(segment)
            } else {
                0
            }
        }),
        segments: 1 << shape.seg_log(),
    };
    assert!(layout.segments <= MAX_SEGMENTS);
    let (mut src, mut dst) = scratch.table.split_at_mut(layout.m * layout.segments * ROW);
    let pads = &mut scratch.pads;
    let mut message = tracing::info_span!("Tables").in_scope(|| {
        if layout.paired() {
            let eq = eq_table(&claim.point[SKIP + 2..]);
            TailMessage::Pair(layer.table_pair(&round, &point, &eq, layout, src, pads))
        } else {
            let eq = eq_table(&claim.point[SKIP + 1..]);
            let (zero, infinity) = layer.table(&round, &point, &eq, layout, src, pads);
            TailMessage::Single(zero, infinity)
        }
    });
    let tail = tracing::info_span!("Tail").entered();
    let mut i = SKIP;
    while i < claim.point.len() {
        let values = message.first(claim.point.get(i + 1).copied().unwrap_or(F128::ZERO));
        let coeffs = round_coefficients(values.0, values.1, value, claim.point[i]);
        let challenge = bind(ps, &coeffs, &mut value, &mut point);
        match message {
            TailMessage::Pair(pair) => {
                let values = pair.second(challenge);
                let coeffs = round_coefficients(values.0, values.1, value, claim.point[i + 1]);
                let second = bind(ps, &coeffs, &mut value, &mut point);
                let to = layout.next().next();
                i += 2;
                let skipped = if to.paired() { 2 } else { 1 };
                let later = eq_table(claim.point.get(i + skipped..).unwrap_or(&[]));
                message = round.fold_pair(
                    src,
                    layout,
                    dst,
                    pads,
                    &[field.to_poly(challenge), field.to_poly(second)],
                    &later,
                );
                layout = to;
            }
            TailMessage::Single(..) => {
                let later = eq_table(claim.point.get(i + 2..).unwrap_or(&[]));
                let (zero, infinity) = round.fold(src, layout, dst, pads, &field.to_poly(challenge), &later);
                message = TailMessage::Single(zero, infinity);
                layout = layout.next();
                i += 1;
            }
        }
        (src, dst) = (dst, src);
    }
    let table = &src[..ROW];
    drop(tail);

    // The bit positions, in the standard basis. The cube's entries past the positions are zero,
    // so the tables hold only the positions, and a zero where a round's last pair needs one.
    let t0: Vec<F128> = (0..BITS).map(|j| field.to_field(&get(table, BITS, j))).collect();
    let mut tables: Vec<[F128; 4]> = (0..BITS)
        .map(|j| [claim.weights[j], t0[j], t0[along(j, 1)], t0[along(j, 2)]])
        .collect();
    let mut local = Vec::with_capacity(LOCAL_BITS);
    for _ in 0..LOCAL_BITS {
        if tables.len() % 2 == 1 {
            tables.push([F128::ZERO; 4]);
        }
        let coeffs = local_round(&tables, kappa);
        ps.add_round_poly(&coeffs, false);
        let challenge = ps.sample();
        local.push(challenge);
        tables = tables
            .as_chunks::<2>()
            .0
            .iter()
            .map(|[lo, hi]| std::array::from_fn(|k| interp(lo[k], hi[k], challenge)))
            .collect();
    }
    let t = [1, 2, 3].map(|k| tables[0][k]);
    ps.add_scalars(&t);
    // `beta`: the next claim is quadratic in it.
    ps.grind(fiat_shamir::grinding_bits(2));
    let beta = ps.sample();
    Claim {
        value: t[0] + beta * (t[1] + beta * t[2]),
        mu: mu_at(s),
        point,
        weights: linear_transpose(&chi_weights(&eq_table(&local), beta)),
    }
}

/// Send a normalized round of the sumcheck over the packs and segments and bind its variable.
fn bind(ps: &mut ProverState, coeffs: &[F128; 3], value: &mut F128, point: &mut Vec<F128>) -> F128 {
    ps.add_round_poly(coeffs, true);
    let challenge = ps.sample();
    *value = poly_eval(coeffs, challenge);
    point.push(challenge);
    challenge
}

/// The coefficients of a round polynomial `h` from its values at `0` and infinity:
/// `(1 + r) h(0) + r h(1) = value` gives the value at one.
fn round_coefficients(zero: F128, infinity: F128, value: F128, r: F128) -> [F128; 3] {
    let (c0, c2) = (zero, infinity);
    // value = c0 + r (c1 + c2).
    let c1 = (value + c0) * r.inv() + c2;
    [c0, c1, c2]
}

#[cfg(test)]
mod tests {
    use primitives::test_rng::Rng;
    use verifier::keccak::PACK;

    use super::*;
    use crate::protocol::first_claim;

    /// The fold of the differences of a block's groups is the same by the nibble tables as by
    /// carry-less products, and a quad of those is what the single rows give.
    #[test]
    fn group_differences_agree_between_kernels() {
        let mut rng = Rng::new(50);
        let field = PowerField::new(rng.ext()).unwrap();
        let block: Vec<u64> = (0..BITS * BLOCK_WORDS).map(|_| rng.next_u64()).collect();
        let differences = Differences::new(&field, &rng.ext_vec(SKIP - 1));
        let (mut fast, mut slow) = (vec![0u64; 10 * LANE_ROW], vec![0u64; 10 * LANE_ROW]);
        let mut wide = vec![0u64; 10 * 3 * 64];
        for r in 0..5 {
            differences.rows(&field, &block[r * 5 * 64 * BLOCK_WORDS..], &mut fast, &mut wide);
            differences.rows_clmul(&field, &block[r * 5 * 64 * BLOCK_WORDS..], &mut slow, &mut wide);
            assert!(fast == slow, "row {r}");
        }
    }

    /// The grouped rounds send what the four skipped rounds send, and leave the same claim and
    /// transcript, with a padding block and at several steps, for one table and for several of
    /// different heights and paddings.
    #[test]
    fn grouped_rounds_equal_the_skipped_ones() {
        if GROUPED {
            // Its kernels leave out what the skipped rounds read.
            return;
        }
        let mut rng = Rng::new(49);
        let cases: [(&[usize], usize); 3] = [(&[1100], 12), (&[1100, 2600], 12), (&[700, 0, 3000, 40], 6)];
        for (rows, depth) in cases {
            let inputs: Vec<Vec<[u64; primitives::hash::STATE_LANES]>> = rows
                .iter()
                .map(|&n| (0..n).map(|_| std::array::from_fn(|_| rng.next_u64())).collect())
                .collect();
            let tables: Vec<crate::witness::TableInputs> = inputs
                .iter()
                .enumerate()
                .map(|(t, inputs)| crate::witness::TableInputs {
                    inputs,
                    pad: std::array::from_fn(|l| if t == 0 { 0 } else { 0x0101 * (l as u64 + t as u64) }),
                })
                .collect();
            let shape = Shape::with_tables(*rows.iter().max().unwrap(), depth, rows.len().ilog2() as usize);
            let mut committed = vec![primitives::field::F64(0); 1 << shape.stack_log()];
            let witness = Witness::generate_tables(shape, &tables, &mut committed);
            assert!(
                (0..shape.tables()).any(|t| witness.table(t).written() < shape.n_blocks()),
                "a padding block"
            );
            for step in [0, 1, depth - 1] {
                let claim = first_claim(shape, &rng.ext_vec(4), rng.ext_vec(shape.y_vars()), &rng.ext_vec(5));
                let run = |grouped| {
                    let mut ps = ProverState::from_label(b"keccak-gkr-grouped");
                    let mut scratch = Scratch::new(shape);
                    let next = prove_layer_with(&mut ps, &witness, &mut scratch, step, &claim, grouped);
                    (next, ps.into_proof())
                };
                assert!(run(true) == run(false), "{rows:?} step {step}");
            }
        }
    }

    /// A pair round's two messages are those of two single rounds, whether its four rows differ,
    /// repeat, or are all one padding row.
    #[test]
    fn paired_messages_match_successive_rounds() {
        let mut rng = Rng::new(47);
        let s = rng.ext();
        let field = PowerField::new(s).unwrap();
        let weights = rng.ext_vec(BITS);
        let polys: Vec<Element> = weights.iter().map(|&w| field.to_poly(w)).collect();
        let claim = Claim {
            value: F128::ZERO,
            mu: [F128::ZERO; PACK],
            point: Vec::new(),
            weights,
        };
        let round = Round::new(&field, &claim, &polys, s, rng.ext());
        let rows: [Vec<u64>; 4] = std::array::from_fn(|_| (0..ROW).map(|_| rng.next_u64()).collect());
        let mut diffs = vec![0u64; 2 * ROW];
        for input in [
            rows.each_ref().map(Vec::as_slice),
            [&rows[0][..], &rows[1], &rows[2], &rows[2]],
            [&rows[0][..]; 4],
        ] {
            let (eq, eta, r) = (rng.ext(), rng.ext(), rng.ext());
            let mut sums = PairSums::default();
            round.add_quad(&mut sums, &mut diffs, input, eq);
            let pair = round.finish_pair(&sums);
            let mut first = RoundSums::default();
            round.add_pair(&mut first, input[0], input[1], eq * (F128::ONE + eta));
            round.add_pair(&mut first, input[2], input[3], eq * eta);
            assert_eq!(pair.first(eta), round.finish(&first));

            let (mut low, mut high) = (vec![0; ROW], vec![0; ROW]);
            field.fold_row_portable(&mut low, input[0], input[1], &field.to_poly(r), BITS);
            field.fold_row_portable(&mut high, input[2], input[3], &field.to_poly(r), BITS);
            let mut second = RoundSums::default();
            round.add_pair(&mut second, &low, &high, eq);
            assert_eq!(pair.second(r), round.finish(&second));
        }
    }

    /// One fold of four rows is two folds in a row.
    #[test]
    fn a_quad_fold_equals_two_single_folds() {
        let mut rng = Rng::new(48);
        let field = PowerField::new(rng.ext()).unwrap();
        let rows: [Vec<u64>; 4] = std::array::from_fn(|_| (0..ROW).map(|_| rng.next_u64()).collect());
        let (r, s) = (field.to_poly(rng.ext()), field.to_poly(rng.ext()));
        let (mut low, mut high, mut want) = (vec![0; ROW], vec![0; ROW], vec![0; ROW]);
        field.fold_row_portable(&mut low, &rows[0], &rows[1], &r, BITS);
        field.fold_row_portable(&mut high, &rows[2], &rows[3], &r, BITS);
        field.fold_row_portable(&mut want, &low, &high, &s, BITS);
        let mut got = vec![0; ROW];
        let rows = rows.each_ref().map(Vec::as_slice);
        field.fold_quad(&mut got, rows, &[r, s, field.mul(&r, &s)], BITS);
        assert!(got == want);
        field.fold_quad_portable(&mut got, rows, &[r, s, field.mul(&r, &s)], BITS);
        assert!(got == want);
    }
}

#[cfg(all(test, target_arch = "x86_64", target_feature = "avx2", target_feature = "gfni"))]
mod gfni_tests {
    use primitives::test_rng::Rng;

    use super::*;
    use crate::protocol::first_claim;
    use primitives::hash::STATE_LANES;

    /// The GFNI outer products give the digit buckets' sums, on dense, sparse and zero weights,
    /// with and without a padding block.
    #[test]
    fn gfni_pack_products_equal_the_buckets() {
        // 1100 instances: three blocks written, a padding block for the fourth.
        {
            let (n, depth, seed) = (1100, 12, 2);
            let mut rng = Rng::new(seed);
            let inputs: Vec<[u64; STATE_LANES]> = (0..n).map(|_| std::array::from_fn(|_| rng.next_u64())).collect();
            let shape = Shape::new(n, depth);
            let mut committed = vec![primitives::field::F64(0); 1 << shape.stack_log()];
            let witness = Witness::generate(shape, &inputs, &mut committed);
            assert!(witness.table(0).written() < shape.n_blocks(), "a padding block");
            for (step, mode) in [(0, "dense"), (depth - 1, "sparse"), (depth / 2, "lanes"), (1, "zero")] {
                let layer = Layer::new(&witness, (0..shape.segments()).map(|m| shape.round(m, step)).collect());
                let mut claim = first_claim(shape, &rng.ext_vec(4), rng.ext_vec(shape.y_vars()), &rng.ext_vec(5));
                claim.weights = (0..BITS)
                    .map(|j| {
                        // Sparse: whole lane bytes and some single positions are zero. Lanes: only
                        // lanes 0..3 weigh, so rows with no active lane at all are skipped.
                        if mode == "zero"
                            || (mode == "sparse" && ((j / 8) % 3 == 1 || j % 7 == 0))
                            || (mode == "lanes" && j >= 4 * 64)
                        {
                            F128::ZERO
                        } else {
                            rng.ext()
                        }
                    })
                    .collect();
                assert!(
                    layer.pack_products_gfni(&claim) == layer.pack_products_buckets(&claim),
                    "n {n} depth {depth} step {step} {mode}"
                );
            }
        }
    }
}
