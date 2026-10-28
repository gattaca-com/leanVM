// CREDIT: https://github.com/succinctlabs/flock (flock-core), MIT OR Apache-2.0.
//! Multilinear sumcheck: rounds 2..(m − k_skip + 1) of the zerocheck protocol.
//!
//! After the round-1 URM and the verifier's univariate-skip fold-point `z`, the
//! protocol enters a standard multilinear sumcheck over `n = m − k_skip`
//! variables, on the whole R1CS polynomial:
//!
//!   `Σ_x eq(r_rest, x) · (a_mlv(x) · b_mlv(x) + c_mlv(x))`
//!
//! with claim `P(z)` from round 1. The quadratic AB part and the linear C part
//! ride the same rounds, so all three claims land at one point; each round
//! sends `(P_r(1), P_r(∞))` via the Karatsuba ∞-trick, with C contributing to
//! `P_r(1)` only.
//!
//! The rounds run in two regimes, cross-checked in tests against the naive fold-then-sum references:
//!
//! - **Bit rounds.** While a folded F192 table would outweigh the packed bits, each pass re-reads the bits.
//!   A pass folds them on the fly and sends two rounds, the second as a quadratic in the first's challenge.
//! - **Table rounds.** The last bit pass stores the folded tables, and each later round folds and sums them.
//!
//! **Index convention** (matches the C++ extract_c pipeline's `sumcheck_round_pair`
//! and the NEON `fold_in_place_pair`): the **low bit** of the multilinear index
//! is bound first. So `a_mlv[2k]` is the X=0 value and `a_mlv[2k+1]` is the X=1
//! value, paired by the round message and the fold.
//!
//! For `[r_0, …, r_{n-1}]` (one eq challenge per multilinear variable, built so
//! `build_eq` places `r_i` at bit i), **round r=2 binds the variable of `r_0`**
//! and takes eq over `r_1..` for the remaining variables. Subsequent rounds peel
//! off one more.
//!
//! **Round message format**: the kernels return `(G(1), G(∞))`, which is what
//! goes on the wire. The protocol polynomial is `Π(X) = eq(r_now, X) · G(X)` of
//! degree 3, for `r_now` the challenge of the variable bound this round; the
//! verifier reconstructs `G(0)` from the running claim via
//! `current_claim = (1+r_now)·G(0) + r_now·G(1)`.

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq"
))]
mod lanes;

use crate::zerocheck::PaddingSpec;
use crate::zerocheck::bit_fold::{BLOCK, BitFold};
use crate::zerocheck::bits::{Bits, Scratch};
#[cfg(test)]
use crate::zerocheck::univariate_skip::pack_bits;
use crate::zerocheck::univariate_skip::{SharedPadding, build_eq, eq_mass_below};
use primitives::field::{F192, F192x8, PHI_8_TABLE_192 as PHI_8_TABLE, Prod8};
use primitives::stream::Stream;

// ---------------------------------------------------------------------------
// Lagrange weights for the univariate-skip fold at z.
// ---------------------------------------------------------------------------

#[cfg(test)]
use primitives::multilinear::lagrange_weights_naive;

/// Interpolate a degree-`< 2·2^k_skip` polynomial at z, given its `2^k_skip`
/// evaluations on Λ and the assumption that it equals **zero on S**.
///
/// This is the verifier's round-1 reconstruction trick: for an honest prover
/// the combined polynomial `P = P^{AB} + P^C` satisfies `P(λ) = 0` for every
/// `λ ∈ S` (the zerocheck identity at S). Together with the `2^k_skip`
/// evaluations on Λ that the prover sends, that's `2·2^k_skip` evaluations -
/// enough to interpolate the degree-`< 2·2^k_skip` polynomial uniquely.
///
pub fn interpolate_at_z_combined(values_on_lambda: &[F192], k_skip: usize, z: F192) -> F192 {
    let ell = 1usize << k_skip;
    assert_eq!(values_on_lambda.len(), ell);
    assert!(2 * ell <= 256, "Λ ∪ S must fit in F_8 (need k_skip ≤ 7)");
    // The first `ell` nodes are S, where the polynomial is zero by assumption;
    // the Λ evaluations follow.
    let mut values = vec![F192::ZERO; 2 * ell];
    values[ell..].copy_from_slice(values_on_lambda);
    primitives::multilinear::lagrange_eval(&PHI_8_TABLE[..2 * ell], &values, z)
}

// ---------------------------------------------------------------------------
// Fold a Boolean witness at z.
// ---------------------------------------------------------------------------

/// Evaluate the univariate-skip polynomial at the fold point `z`, given the
/// precomputed Lagrange `weights`. Returns the multilinear extension table
/// `a_mlv` of length `2^(m − k_skip)` over F_{2^192}.
///
///   `a_mlv[x_rest] = Σ_s a(s, x_rest) · L_s(z)`
///
/// `a(s, x_rest)` is the witness bit at index `x_rest * 2^k_skip + s` (low
/// bits = skip variable, high bits = rest variables).
#[cfg(test)]
fn fold_at_z_naive(witness: &[bool], m: usize, k_skip: usize, weights: &[F192]) -> Vec<F192> {
    assert!(k_skip <= m);
    let ell = 1usize << k_skip;
    let n_rest = 1usize << (m - k_skip);
    assert_eq!(witness.len(), 1usize << m);
    assert_eq!(weights.len(), ell);

    (0..n_rest)
        .map(|x_rest| {
            let base = x_rest * ell;
            let mut acc = F192::ZERO;
            for s in 0..ell {
                if witness[base + s] {
                    acc += weights[s];
                }
            }
            acc
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Naive round-2 prover message (AB-pair multilinear sumcheck).
// ---------------------------------------------------------------------------

/// Single-table sibling of [`round_pair_naive`], for the linear `c` term:
/// `G_c(1) = Σ_{x'} eq(r_eq, x') · c_mlv(1, x')`. Linear, so no `G(∞)`.
pub fn round_single_naive(c_mlv: &[F192], r_eq: &[F192]) -> F192 {
    let n = c_mlv.len();
    assert!(n.is_power_of_two() && n >= 2);
    assert_eq!(r_eq.len(), n.trailing_zeros() as usize - 1);
    let eq_remaining = build_eq(r_eq);
    let mut g_one = F192::ZERO;
    for (x_prime, &eq_x) in eq_remaining.iter().enumerate() {
        g_one += eq_x * c_mlv[2 * x_prime + 1];
    }
    g_one
}

/// Round-2 (and any subsequent round) prover message for the AB-pair
/// multilinear sumcheck.
///
/// Inputs:
/// - `a_mlv`, `b_mlv`: F192 vectors of length `2^n` for some `n ≥ 1`.
/// - `r_eq`: the eq challenges of the `n − 1` variables NOT bound this round.
///
/// Output: `(G(1), G(∞))` for the round polynomial `G(X) = Σ_{x'} eq(r_eq, x')
/// · a_mlv(X, x') · b_mlv(X, x')`, where `a_mlv(0, x') = a_mlv[2x']` and
/// `a_mlv(1, x') = a_mlv[2x' + 1]` (low bit bound).
pub fn round_pair_naive(a_mlv: &[F192], b_mlv: &[F192], r_eq: &[F192]) -> (F192, F192) {
    let n = a_mlv.len();
    assert_eq!(b_mlv.len(), n);
    assert!(n.is_power_of_two() && n >= 2);
    let half = n / 2;
    assert_eq!(r_eq.len(), n.trailing_zeros() as usize - 1);

    let eq_remaining = build_eq(r_eq);
    assert_eq!(eq_remaining.len(), half);

    let mut g_one = F192::ZERO;
    let mut g_inf = F192::ZERO;
    for x_prime in 0..half {
        let a0 = a_mlv[2 * x_prime];
        let a1 = a_mlv[2 * x_prime + 1];
        let b0 = b_mlv[2 * x_prime];
        let b1 = b_mlv[2 * x_prime + 1];
        let eq_x = eq_remaining[x_prime];
        g_one += eq_x * a1 * b1;
        // Char-2: (a_1 − a_0)(b_1 − b_0) = (a_0 + a_1)(b_0 + b_1).
        g_inf += eq_x * (a0 + a1) * (b0 + b1);
    }
    (g_one, g_inf)
}

// ---------------------------------------------------------------------------
// Bit-resident rounds: fold and message straight from the packed witness.
// ---------------------------------------------------------------------------

/// Returns `(pair_in_block_mask, live_pairs)` for a round whose positions each cover `2^position_log` witness bits.
///
/// Pair `k` (positions `2k`, `2k+1`) lies wholly in a block's zero padding iff `(k & pair_in_block_mask) >= live_pairs`.
///
/// - Such a pair folds to zero, so it adds nothing to the message.
/// - A pair straddling the boundary counts as live: its padding half is honestly zero.
/// - With no whole padding pair, the mask is zero and every pair is live.
fn padding_pairs(padding: &PaddingSpec, position_log: usize) -> (usize, usize) {
    if padding.k_log <= position_log + 1 {
        return (0, usize::MAX);
    }
    let pairs_per_block = 1usize << (padding.k_log - position_log - 1);
    let live_pairs = padding.useful_bits_per_block.div_ceil(2 << position_log);
    if live_pairs >= pairs_per_block {
        return (0, usize::MAX);
    }
    (pairs_per_block - 1, live_pairs)
}

/// Eq variables in the per-task half of the split eq table.
///
/// - 2^10 entries are 24 KiB, so the table stays in L1 beside the fold's matrices.
/// - The remaining variables index the tasks, one reduced product each.
const EQ_LO_VARS: usize = 10;

/// An eq table packed eight to a vector, zero-padded to a multiple of four vectors.
fn pack_eq(eq: &[F192]) -> Vec<F192x8> {
    let mut packed: Vec<F192x8> = eq
        .chunks(8)
        .map(|c| F192x8::from_array(&std::array::from_fn(|i| c.get(i).copied().unwrap_or(F192::ZERO))))
        .collect();
    packed.resize(packed.len().next_multiple_of(4), F192x8::ZERO);
    packed
}

/// The split eq table over `r`: `eq(r, k) = hi[k >> n_lo] * lo[k & (2^n_lo - 1)]`.
///
/// Each task sums its `2^n_lo` terms unreduced, then pays one reduction and one product.
///
/// Returns `lo` packed, `hi`, and `2^n_lo`.
fn split_eq(r: &[F192]) -> (Vec<F192x8>, Vec<F192>, usize) {
    let n_lo = r.len().min(EQ_LO_VARS);
    (pack_eq(&build_eq(&r[..n_lo])), build_eq(&r[n_lo..]), 1 << n_lo)
}

/// The packed `a`, `b`, `c` witnesses in memory, 64 skip bits per row.
///
/// `c` spans every position. `a` and `b` may stop early, as long as the kernels read no row past their end:
/// a padding tail is read up to its template block only.
#[derive(Clone, Copy, Debug)]
pub struct PackedWitness<'a> {
    /// The `A z` bits.
    pub a: &'a [u8],
    /// The `B z` bits.
    pub b: &'a [u8],
    /// The `C z` bits.
    pub c: &'a [u8],
}

/// The folded `a`, `b`, `c` values of up to 64 consecutive positions, as eight vectors each.
type Folded = [[F192x8; 8]; 3];

/// Fold positions `first..first + len` of each witness, in the vector order of [`BitFold::fold_block_x8`].
///
/// A rebuilt witness is rebuilt into `scratch` as far as these positions need.
#[inline(always)]
fn fold_x8<const CHUNKS: usize, const GROUP_LOG: usize>(
    fold: &BitFold,
    bits: &Bits<'_>,
    scratch: &mut Option<Scratch>,
    first: usize,
    len: usize,
) -> Folded {
    let window = bits.window(first * CHUNKS..(first + len) * CHUNKS, scratch);
    window
        .rows::<CHUNKS>(first, len)
        .map(|r| fold.fold_block_x8::<CHUNKS, GROUP_LOG>(r))
}

/// Two consecutive multilinear rounds from one pass over the packed bits.
///
/// Round `t + 1` binds its variable after the verifier samples `rho`, the challenge of round `t`.
///
/// Its polynomial is quadratic in that `rho`, so one pass stores its three coefficients per evaluation point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoundPair {
    /// Round `t`'s `(G(1), G(inf))`.
    pub first: (F192, F192),
    /// Round `t + 1` at `Y = 1` and `Y = inf`, each as `[S_0, S_1, S_2]`.
    ///
    /// ```text
    ///     G(Y) = (1 + rho) S_0 + rho S_1 + rho (1 + rho) S_2
    /// ```
    second: [[F192; 3]; 2],
}

impl RoundPair {
    /// Round `t + 1`'s `(G(1), G(inf))`, once round `t`'s challenge `rho` is known.
    pub fn second(&self, rho: F192) -> (F192, F192) {
        let [one, inf] = self
            .second
            .map(|[s0, s1, s2]| s0 + rho * (s0 + s1) + rho * (F192::ONE + rho) * s2);
        (one, inf)
    }
}

/// Rounds `t` and `t + 1` straight from the packed bits, the folded tables never stored.
///
/// With `rho_1..rho_t` bound, `fold` weights each position's `2^t` rows (see its level constructor).
///
/// Each round's polynomial, with `r_eq` the eq challenges of the variables round `t` does not bind:
///
/// ```text
///     G(X) = sum_x' eq(r_eq, x') * (a(X, x') * b(X, x') + c(X, x'))
/// ```
///
/// The linear `c` term reaches `G(1)` only.
pub fn bit_round_pair<'a>(
    bits: impl Into<Bits<'a>>,
    fold: &BitFold,
    r_eq: &[F192],
    padding: &PaddingSpec,
) -> RoundPair {
    let bits = bits.into();
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq"
    ))]
    {
        let by_instance = if fold.n_chunks() == 8 {
            lanes::pair(bits, fold, r_eq, padding)
        } else {
            None
        };
        if let Some(pair) = by_instance {
            return pair;
        }
    }
    match fold.n_chunks() {
        8 => bit_round_pair_kernel::<8>(bits, fold, r_eq, padding),
        16 => bit_round_pair_kernel::<16>(bits, fold, r_eq, padding),
        32 => bit_round_pair_kernel::<32>(bits, fold, r_eq, padding),
        64 => bit_round_pair_kernel::<64>(bits, fold, r_eq, padding),
        128 => bit_round_pair_kernel::<128>(bits, fold, r_eq, padding),
        n => panic!("no bit-round kernel for {n}-byte rows"),
    }
}

/// One round straight from the packed bits, storing the folded `(a, b, c)` tables for the rounds that follow.
///
/// Returns the round's `(G(1), G(inf))`, then the three tables.
pub fn bit_round_materialize<'a>(
    bits: impl Into<Bits<'a>>,
    fold: &BitFold,
    r_eq: &[F192],
    padding: &PaddingSpec,
) -> ((F192, F192), [Vec<F192>; 3]) {
    let bits = bits.into();
    match fold.n_chunks() {
        8 => bit_round_store_kernel::<8>(bits, fold, r_eq, padding),
        16 => bit_round_store_kernel::<16>(bits, fold, r_eq, padding),
        32 => bit_round_store_kernel::<32>(bits, fold, r_eq, padding),
        64 => bit_round_store_kernel::<64>(bits, fold, r_eq, padding),
        128 => bit_round_store_kernel::<128>(bits, fold, r_eq, padding),
        n => panic!("no bit-round kernel for {n}-byte rows"),
    }
}

/// The two-round pass, for rows of `CHUNKS` bytes.
///
/// Positions group in quads `4k + u + 2v`: `u` is round `t`'s variable and `v` round `t + 1`'s.
///
/// ```text
///     position   4k     4k+1   4k+2   4k+3
///     (u, v)     (0,0)  (1,0)  (0,1)  (1,1)
/// ```
///
/// Round `t + 1` folds `u` at `rho` first, so each of its values is `f(rho, Y) = f(0, Y) + rho * (f(0, Y) + f(1, Y))`.
///
/// Expanding the product in `rho` gives the three sums of the second round:
///
/// ```text
///     S_0 = sum eq * a(0, Y) b(0, Y)      S_1 = sum eq * a(1, Y) b(1, Y)
///     S_2 = sum eq * (a(0, Y) + a(1, Y)) (b(0, Y) + b(1, Y))
/// ```
///
/// Round `t` needs its sums split by `v`, and two of them coincide with round `t + 1`'s.
///
/// So eight products per quad cover both rounds, the same count as two passes.
fn bit_round_pair_kernel<const CHUNKS: usize>(
    bits: Bits<'_>,
    fold: &BitFold,
    r_eq: &[F192],
    padding: &PaddingSpec,
) -> RoundPair {
    let n_quads = bits.positions::<CHUNKS>() / 4;
    assert!(n_quads >= 1, "two rounds need four positions");
    assert_eq!(r_eq.len(), n_quads.trailing_zeros() as usize + 1);

    // `r_eq[0]` weights round `t`'s split by `v`; the rest weight the quads.
    let (r_v, r_quad) = (r_eq[0], &r_eq[1..]);
    let (eq_lo, eq_hi, lo_size) = split_eq(r_quad);

    // A quad covers 2^6 skip bits times its 4 * 2^t bound rows.
    let quad_log = (32 * CHUNKS).trailing_zeros() as usize;
    let (quad_in_block_mask, live_quads) = padding_pairs(padding, quad_log - 1);
    let live = |quad: usize| (quad & quad_in_block_mask) < live_quads;

    // The tasks cover the quads before the padding tail, then one task folds the tail's first block.
    let shared = SharedPadding::new(padding, quad_log, r_quad);
    let shared_eq = shared.as_ref().map(|s| pack_eq(&s.eq));
    let n_live_tasks = shared.as_ref().map_or(eq_hi.len(), |s| s.live.div_ceil(lo_size));

    let sums = parallel::map_reduce_with_state(
        n_live_tasks + shared.is_some() as usize,
        || bits.scratch(BLOCK * CHUNKS),
        || [F192::ZERO; 8],
        |scratch, total, hi| {
            let (first, count, eq_lo, weight) = match &shared {
                Some(s) if hi == n_live_tasks => {
                    (s.live, s.eq.len(), shared_eq.as_deref().unwrap_or_default(), s.weight)
                }
                Some(s) => (
                    hi * lo_size,
                    (s.live - hi * lo_size).min(lo_size),
                    &eq_lo[..],
                    eq_hi[hi],
                ),
                None => (hi * lo_size, lo_size, &eq_lo[..], eq_hi[hi]),
            };
            let mut acc = [Prod8::ZERO; 8];
            // Sixteen quads per folded block.
            for lo_first in (0..count).step_by(BLOCK / 4) {
                let n = (count - lo_first).min(BLOCK / 4);
                let quad_first = first + lo_first;
                // A block wholly in padding folds to zero.
                if !(quad_first..quad_first + n).any(live) {
                    continue;
                }
                let f = fold_x8::<CHUNKS, 2>(fold, &bits, scratch, 4 * quad_first, 4 * n);
                quad_sums(
                    &f,
                    eq_lo[lo_first / 8..][..2].try_into().expect("two vectors"),
                    &mut acc,
                );
            }
            *total = std::array::from_fn(|i| total[i] + weight * acc[i].reduce());
        },
        |x, y| std::array::from_fn(|i| x[i] + y[i]),
    );

    // Slots 0, 1 hold round t's G(1) at v = 0, 1, and slots 2, 3 its G(inf).
    // Round t + 1 reads slots 4, 1, 3 at Y = 1 and 5, 6, 7 at Y = inf.
    let split_v = |v0: F192, v1: F192| v0 + r_v * (v0 + v1);
    RoundPair {
        first: (split_v(sums[0], sums[1]), split_v(sums[2], sums[3])),
        second: [[sums[4], sums[1], sums[3]], [sums[5], sums[6], sums[7]]],
    }
}

/// The eight sums of a block's sixteen quads, `eq`-weighted and accumulated unreduced, eight quads to a half.
///
/// The block is in the order of group size 4: vector `j + 4h` holds position `j` of the quads `8h..8h+8`, one per lane.
/// So every product is lane-wise, and the slots are those of [`bit_round_pair_kernel`]'s message.
#[inline(always)]
fn quad_sums(f: &Folded, eq: &[F192x8; 2], acc: &mut [Prod8; 8]) {
    let [a, b, c] = f;
    for (h, eq) in eq.iter().enumerate() {
        let quad = |t: &[F192x8; 8]| -> [F192x8; 4] { t[4 * h..4 * h + 4].try_into().expect("a quad") };
        let ([a0, a1, a2, a3], [b0, b1, b2, b3], [_, c1, c2, c3]) = (quad(a), quad(b), quad(c));

        // Weighting `a` first shares the eq product: every sum below is linear in `a`, and only four are reduced.
        let [e0, e1, e2, e3] = [a0, a1, a2, a3].map(|a| *eq * a);

        // Leading coefficients along `u` (positions 0,1 and 2,3) and along `v` (0,2 and 1,3).
        let (du0, du1, dv0, dv1) = (e0 + e1, e2 + e3, e0 + e2, e1 + e3);
        let (eu0, eu1, ev0, ev1) = (b0 + b1, b2 + b3, b0 + b2, b1 + b3);
        acc[0] ^= e1.mul_unreduced(b1) ^ eq.mul_unreduced(c1);
        acc[1] ^= e3.mul_unreduced(b3) ^ eq.mul_unreduced(c3);
        acc[2] ^= du0.mul_unreduced(eu0);
        acc[3] ^= du1.mul_unreduced(eu1);
        acc[4] ^= e2.mul_unreduced(b2) ^ eq.mul_unreduced(c2);
        acc[5] ^= dv0.mul_unreduced(ev0);
        acc[6] ^= dv1.mul_unreduced(ev1);
        acc[7] ^= (du0 + du1).mul_unreduced(eu0 + eu1);
    }
}

/// The `G(1)` and `G(inf)` sums of a block's thirty-two pairs, `eq`-weighted and accumulated unreduced.
///
/// The block is in natural order: vector `k` holds the positions `8k..8k+8`, so `even_odd` of vectors `2m` and
/// `2m + 1` splits the pairs `8m..8m+8` into their two positions.
#[inline(always)]
fn pair_sums(f: &Folded, eq: &[F192x8; 4], acc: &mut [Prod8; 2]) {
    let [a, b, c] = f;
    for (m, eq) in eq.iter().enumerate() {
        let (a0, a1) = F192x8::even_odd(a[2 * m], a[2 * m + 1]);
        let (b0, b1) = F192x8::even_odd(b[2 * m], b[2 * m + 1]);
        let (_, c1) = F192x8::even_odd(c[2 * m], c[2 * m + 1]);
        // G(1) takes a_1 b_1 + c_1.
        // G(inf) takes (a_0 + a_1)(b_0 + b_1), the leading coefficient in characteristic 2.
        acc[0] ^= eq.mul_unreduced(a1 * b1 + c1);
        acc[1] ^= eq.mul_unreduced((a0 + a1) * (b0 + b1));
    }
}

/// The storing single-round pass, for rows of `CHUNKS` bytes.
///
/// Positions pair up as `(2k, 2k + 1)`, the low index bit being the variable this round binds.
fn bit_round_store_kernel<const CHUNKS: usize>(
    bits: Bits<'_>,
    fold: &BitFold,
    r_eq: &[F192],
    padding: &PaddingSpec,
) -> ((F192, F192), [Vec<F192>; 3]) {
    let n_pos = bits.positions::<CHUNKS>();
    assert!(n_pos >= 2, "a round needs two positions");
    assert_eq!(r_eq.len(), n_pos.trailing_zeros() as usize - 1);

    let (eq_lo, eq_hi, lo_size) = split_eq(r_eq);

    // A position covers 2^6 skip bits times its 2^t bound rows.
    let position_log = (8 * CHUNKS).trailing_zeros() as usize;
    let (pair_in_block_mask, live_pairs) = padding_pairs(padding, position_log);
    let live = |pair: usize| (pair & pair_in_block_mask) < live_pairs;

    // Folds `count` pairs from pair `first` into `outs`, each pair weighted by `eq_lo`: the unreduced `(G(1), G(inf))`.
    let fold_run = |scratch: &mut Option<Scratch>,
                    first: usize,
                    count: usize,
                    eq_lo: &[F192x8],
                    [oa, ob, oc]: [&mut [F192]; 3]| {
        let stream = Stream::new();
        let mut acc = [Prod8::ZERO; 2];
        // Thirty-two pairs per folded block.
        for lo_first in (0..count).step_by(BLOCK / 2) {
            let n = (count - lo_first).min(BLOCK / 2);
            let pair_first = first + lo_first;
            let (o_first, o_len) = (2 * lo_first, 2 * n);
            // A block wholly in padding folds to zero.
            if !(pair_first..pair_first + n).any(live) {
                for o in [&mut *oa, &mut *ob, &mut *oc] {
                    o[o_first..o_first + o_len].fill(F192::ZERO);
                }
                continue;
            }
            let f = fold_x8::<CHUNKS, 0>(fold, &bits, scratch, 2 * pair_first, o_len);
            pair_sums(
                &f,
                eq_lo[lo_first / 8..][..4].try_into().expect("four vectors"),
                &mut acc,
            );

            // Publish the block without a read: nothing touches these tables before the next round.
            for (o, t) in [(&mut *oa, &f[0]), (&mut *ob, &f[1]), (&mut *oc, &f[2])] {
                for (k, v) in t.iter().take(o_len.div_ceil(8)).enumerate() {
                    let dst = &mut o[o_first + 8 * k..o_first + o_len.min(8 * k + 8)];
                    stream.copy(dst, &v.to_array()[..dst.len()]);
                }
            }
        }
        let [g1, ginf] = acc;
        (g1, ginf)
    };

    // The padding tail repeats its first block, so that block is folded once and copied over the tail.
    let shared = SharedPadding::new(padding, position_log + 1, r_eq);
    let n_live_pairs = shared.as_ref().map_or(n_pos / 2, |s| s.live);
    let template = shared.map(|s| {
        let mut t: [Vec<F192>; 3] = std::array::from_fn(|_| vec![F192::ZERO; 2 * s.eq.len()]);
        let [ta, tb, tc] = &mut t;
        let (g1, ginf) = fold_run(
            &mut bits.scratch(BLOCK * CHUNKS),
            s.live,
            s.eq.len(),
            &pack_eq(&s.eq),
            [ta, tb, tc],
        );
        (t, (s.weight * g1.reduce(), s.weight * ginf.reduce()))
    });

    // `Tables` cuts a tail of blocks of at most 32 positions to its pattern, so it reads `TAIL_KEPT` positions past the live ones.
    let short_tail = padding.k_log.saturating_sub(position_log) <= 5;
    let n_out = if short_tail {
        (2 * n_live_pairs).saturating_add(TAIL_KEPT).min(n_pos)
    } else {
        n_pos
    };
    // Every slot is written below, padding included.
    let (mut g1, mut ginf) = (F192::ZERO, F192::ZERO);
    let out = primitives::filled_vecs([n_out; 3], |outs| {
        (g1, ginf) = parallel::map_reduce_chunks_mut(
            outs,
            [2 * lo_size; 3],
            || bits.scratch(BLOCK * CHUNKS),
            || (F192::ZERO, F192::ZERO),
            |scratch, (total1, total_inf), hi, outs| {
                let first = hi * lo_size;
                let n_live = n_live_pairs.saturating_sub(first).min(lo_size);
                let mut halves = outs.map(|o| o.split_at_mut(2 * n_live));
                let (g1, ginf) = fold_run(scratch, first, n_live, &eq_lo, halves.each_mut().map(|h| &mut *h.0));
                if let Some((t, _)) = &template {
                    for ((_, rest), t) in halves.iter_mut().zip(t) {
                        let at = 2 * (first + n_live);
                        for (k, d) in rest.iter_mut().enumerate() {
                            *d = t[(at + k) & (t.len() - 1)];
                        }
                    }
                }
                *total1 += eq_hi[hi] * g1.reduce();
                *total_inf += eq_hi[hi] * ginf.reduce();
            },
            |(s1, si), (t1, ti)| (s1 + t1, si + ti),
        );
    });
    let (t1, ti) = template.map_or((F192::ZERO, F192::ZERO), |(_, w)| w);
    ((g1 + t1, ginf + ti), out)
}

// ---------------------------------------------------------------------------
// Subsequent multilinear rounds (3..(m−k_skip+1)): fold + next message.
// ---------------------------------------------------------------------------

/// In-place fold of a single multilinear polynomial table at `challenge`.
/// Pairs `(a[2x], a[2x+1])` collapse to `a[x] = a[2x] + challenge · (a[2x+1] + a[2x])`.
/// After the call, `a.len()` is halved.
pub fn fold_in_place_single(a: &mut Vec<F192>, challenge: F192) {
    let n = a.len();
    assert!(n.is_power_of_two() && n >= 2);
    let half = n / 2;
    for x in 0..half {
        let a0 = a[2 * x];
        let a1 = a[2 * x + 1];
        a[x] = a0 + challenge * (a1 + a0);
    }
    a.truncate(half);
}

/// In-place fold of a pair `(a, b)` of multilinear polynomial tables at
/// `challenge`. Binds the lowest bit of the index: pairs `(a[2x], a[2x+1])`
/// collapse to `a[x] = a[2x] + challenge · (a[2x+1] + a[2x])` (and same for b).
/// After the call, `a.len()` and `b.len()` are halved.
///
/// Used at the tail of the multilinear-round sequence where the polynomial is
/// small enough that parallel/fusion overhead outweighs benefit.
pub fn fold_in_place_pair(a: &mut Vec<F192>, b: &mut Vec<F192>, challenge: F192) {
    let n = a.len();
    assert_eq!(b.len(), n);
    assert!(n.is_power_of_two() && n >= 2);
    let half = n / 2;
    for x in 0..half {
        let a0 = a[2 * x];
        let a1 = a[2 * x + 1];
        let b0 = b[2 * x];
        let b1 = b[2 * x + 1];
        a[x] = a0 + challenge * (a1 + a0);
        b[x] = b0 + challenge * (b1 + b0);
    }
    a.truncate(half);
    b.truncate(half);
}

/// A fold whose output is at least this large streams it out: below, the next round finds it in cache.
const STREAM_OUT_BYTES: usize = 1 << 24;

/// Steps ahead that the fold prefetches its inputs: the hardware prefetcher alone left the first round slow and erratic.
const PREFETCH_STEPS: usize = 8;

/// Prefetch `lines` cache lines from `p`. Nothing is read, so any address will do.
#[inline(always)]
fn prefetch(p: *const F192, lines: usize) {
    #[cfg(target_arch = "x86_64")]
    for line in 0..lines {
        // SAFETY: a prefetch of any address is harmless.
        unsafe {
            core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
                p.wrapping_byte_add(64 * line).cast(),
            )
        };
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (p, lines);
}

/// Fused: bind the low variable of the `ins` tables at `r_fold`, write the folded tables to `outs`,
/// and return the next round's message `(G(1), G(inf))` over the pairs of the folded positions.
///
/// - Each task takes up to `2^n_lo` pairs, eight at a time, and pays one reduced product by its hi weight.
/// - The outputs are stored plainly: the next round reads them, and streaming them out lost to the cache.
///
/// `outs` hold `m` positions, a multiple of 32, and `ins` at least `2 m`. `r_eq` has a variable per pair index bit.
fn fold_and_compute_round_into(ins: [&[F192]; 3], outs: [&mut [F192]; 3], r_fold: F192, r_eq: &[F192]) -> (F192, F192) {
    let (eq_lo, eq_hi, lo_size) = split_eq(r_eq);
    let r = F192x8::splat(r_fold);
    let stream_out = 3 * outs[0].len().div_ceil(2 * lo_size) * 2 * lo_size * size_of::<F192>() >= STREAM_OUT_BYTES;

    parallel::map_reduce_chunks_mut(
        outs,
        [2 * lo_size; 3],
        || (),
        || (F192::ZERO, F192::ZERO),
        |(), (total1, total_inf), hi, outs| {
            let steps = outs[0].len() / 16;
            let mut acc = [Prod8::ZERO; 2];
            // Eight pairs of folded positions: thirty-two input positions and sixteen output ones per table.
            let [mut sa, mut sb, mut sc] = ins.map(|t| t[hi * 4 * lo_size..][..32 * steps].chunks_exact(32));
            let [mut oa, mut ob, mut oc] = outs.map(|o| o.chunks_exact_mut(16));
            let stream = Stream::new();
            let heads = ins.map(|t| t[hi * 4 * lo_size..].as_ptr());
            for (step, eq) in eq_lo[..steps].iter().enumerate() {
                for h in heads {
                    prefetch(h.wrapping_add(32 * (step + PREFETCH_STEPS)), 12);
                }
                let fold = |x: &[F192], out: &mut [F192]| {
                    let v = |k: usize| F192x8::from_array(x[8 * k..8 * k + 8].try_into().expect("eight elements"));
                    let ((e0, o0), (e1, o1)) = (F192x8::even_odd(v(0), v(1)), F192x8::even_odd(v(2), v(3)));
                    let f = [e0 + (e0 + o0) * r, e1 + (e1 + o1) * r];
                    for (f, d) in f.iter().zip(out.chunks_exact_mut(8)) {
                        if stream_out {
                            stream.copy(d, &f.to_array());
                        } else {
                            d.copy_from_slice(&f.to_array());
                        }
                    }
                    F192x8::even_odd(f[0], f[1])
                };
                let (a0, a1) = fold(sa.next().expect("a step"), oa.next().expect("a out"));
                let (b0, b1) = fold(sb.next().expect("b step"), ob.next().expect("b out"));
                let (_, c1) = fold(sc.next().expect("c step"), oc.next().expect("c out"));
                // G(1) takes a_1 b_1 + c_1, and G(inf) the leading coefficient (a_0 + a_1)(b_0 + b_1).
                acc[0] ^= eq.mul_unreduced(a1 * b1 + c1);
                acc[1] ^= eq.mul_unreduced((a0 + a1) * (b0 + b1));
            }
            let [g1, ginf] = acc;
            *total1 += eq_hi[hi] * g1.reduce();
            *total_inf += eq_hi[hi] * ginf.reduce();
        },
        |(s1, si), (t1, ti)| (s1 + t1, si + ti),
    )
}

/// Positions kept past the live ones: the kernel reads up to 62 beyond, so they carry the tail's pattern.
const EXTENSION: usize = 64;

/// Positions `bit_round_materialize` stores past the live ones: the rounding of the live count up to 32, then the extension.
const TAIL_KEPT: usize = 32 + EXTENSION;

/// The stored `a`, `b`, `c` tables during the table rounds, a tail of identical padding blocks cut to its template.
///
/// Position `p >= live` holds `template[p % period]`, `period` being a power of two of at least two.
/// A tail pair adds `weight * (the template's pair sum)` to a round message, as in the bit rounds.
///
/// - `live` is a multiple of 32, and a round keeps it so, rounding up with positions of the pattern.
/// - Each table holds `live + EXTENSION` positions: the live ones, then the pattern again.
pub struct Tables {
    cols: [Vec<F192>; 3],
    scratch: [Vec<F192>; 3],
    template: [Vec<F192>; 3],
    live: usize,
    log_n: usize,
}

impl Tables {
    /// The tables of `2^log_n` positions that `bit_round_materialize` stored for a witness of `2^m` bits.
    ///
    /// A table may stop after its first template block, or hold the whole tail.
    pub fn new(mut cols: [Vec<F192>; 3], log_n: usize, m: usize, padding: &PaddingSpec) -> Self {
        let n = 1usize << log_n;
        // A block holds `n >> (m - k_log)` positions. Only a block of at most 32 fits the 32-position granularity.
        let tail = m.checked_sub(padding.k_log).and_then(|n_blocks_log| {
            let period = n >> n_blocks_log;
            let first = padding.live_blocks.checked_mul(period)?;
            ((2..=32).contains(&period) && first < n).then_some((first, period))
        });
        let (first, period) = tail.unwrap_or((n, 2));
        let live = first.next_multiple_of(32).min(n);
        let template = cols.each_mut().map(|c| {
            let t = if first < n {
                c[first..first + period].to_vec()
            } else {
                vec![F192::ZERO; 2]
            };
            c.truncate(live + EXTENSION);
            let stored = c.len();
            c.extend((stored..live + EXTENSION).map(|p| t[(p - first) % t.len()]));
            t
        });
        // SAFETY (x3): a round writes every position of the prefix it then reads.
        let scratch_len = (live / 2).next_multiple_of(32) + EXTENSION;
        let scratch = std::array::from_fn(|_| unsafe { primitives::uninit_vec(scratch_len) });
        Self {
            cols,
            scratch,
            template,
            live,
            log_n,
        }
    }

    /// Whether another fused round fits: the folded table must hold a whole 32-position group.
    pub fn can_fold(&self) -> bool {
        self.log_n > 5
    }

    /// The whole tables, the tail written out.
    pub fn dense(&self) -> [Vec<F192>; 3] {
        let n = 1usize << self.log_n;
        std::array::from_fn(|j| {
            let (c, t) = (&self.cols[j], &self.template[j]);
            (0..n)
                .map(|p| if p < self.live { c[p] } else { t[p % t.len()] })
                .collect()
        })
    }

    /// Bind the low variable at `chi`, and return the next round's `(G(1), G(inf))`.
    ///
    /// `r_eq` holds the eq challenges of the variables that round does not bind.
    pub fn fold_round(&mut self, chi: F192, r_eq: &[F192]) -> (F192, F192) {
        assert!(self.can_fold(), "the folded table is under 32 positions");
        assert_eq!(r_eq.len(), self.log_n - 2);
        let live = (self.live / 2).next_multiple_of(32);

        let [sa, sb, sc] = &mut self.scratch;
        let outs = [sa, sb, sc].map(|s| &mut s[..live]);
        let ins = self.cols.each_ref().map(|c| &c[..2 * live]);
        let (mut g1, mut ginf) = fold_and_compute_round_into(ins, outs, chi, r_eq);

        // The fold keeps the tail's blocks whole, halved: the template folds like the tables.
        for t in &mut self.template {
            let folded: Vec<F192> = t.chunks_exact(2).map(|p| p[0] + chi * (p[0] + p[1])).collect();
            *t = if folded.len() == 1 { vec![folded[0]; 2] } else { folded };
        }
        for (s, t) in self.scratch.iter_mut().zip(&self.template) {
            for (p, d) in s[live..live + EXTENSION].iter_mut().enumerate() {
                *d = t[p % t.len()];
            }
        }
        std::mem::swap(&mut self.cols, &mut self.scratch);
        self.live = live;
        self.log_n -= 1;

        // Each block of the tail holds `q` pairs, so its sum takes the eq of the pair's place in the block,
        // and the eq mass of the blocks past `live` weighs it.
        let q = self.template[0].len() / 2;
        let (r_in, r_blk) = r_eq.split_at(q.trailing_zeros() as usize);
        let first_block = live / 2 / q;
        if first_block >> r_blk.len() == 0 {
            let weight = F192::ONE + eq_mass_below(r_blk, first_block);
            let [a, b, c] = &self.template;
            let (mut t1, mut ti) = (F192::ZERO, F192::ZERO);
            for (j, &eq) in build_eq(r_in).iter().enumerate() {
                let ((a0, a1), (b0, b1), c1) = ((a[2 * j], a[2 * j + 1]), (b[2 * j], b[2 * j + 1]), c[2 * j + 1]);
                t1 += eq * (a1 * b1 + c1);
                ti += eq * (a0 + a1) * (b0 + b1);
            }
            g1 += weight * t1;
            ginf += weight * ti;
        }
        (g1, ginf)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use primitives::test_rng::Rng;

    /// Interpolate a degree-`< 2^k_skip` polynomial at z, given its `2^k_skip`
    /// evaluations on the **extension domain** `Λ = {2^k_skip, …, 2^(k_skip+1) − 1}`
    /// embedded via `φ_8` (offset by `2^k_skip` from the S-domain nodes).
    ///
    /// `P^C` no longer travels on its own (it rides the sum the prover sends), so
    /// this is only the cross-check handle: it turns the URM kernel's C Λ-vector
    /// into `P^C(z) = ĉ(z, r_rest)`, which a direct fold of the witness must match.
    fn interpolate_at_z_on_lambda(values: &[F192], k_skip: usize, z: F192) -> F192 {
        let ell = 1usize << k_skip;
        assert_eq!(values.len(), ell);
        assert!(2 * ell <= 256, "Λ ∪ S must fit in F_8 (need k_skip ≤ 7)");
        primitives::multilinear::lagrange_eval(&PHI_8_TABLE[ell..2 * ell], values, z)
    }

    /// `fold_in_place_pair` correctness: post-fold a[x] = a[2x] + X·(a[2x+1]+a[2x]).
    #[test]
    fn fold_in_place_pair_matches_formula() {
        let mut rng = Rng::new(300);
        for &log_n in &[1usize, 2, 3, 4, 6] {
            let n = 1usize << log_n;
            let a_orig: Vec<F192> = (0..n).map(|_| rng.ext()).collect();
            let b_orig: Vec<F192> = (0..n).map(|_| rng.ext()).collect();
            let challenge = rng.ext();

            let mut a = a_orig.to_vec();
            let mut b = b_orig.to_vec();
            fold_in_place_pair(&mut a, &mut b, challenge);

            assert_eq!(a.len(), n / 2);
            assert_eq!(b.len(), n / 2);
            for x in 0..(n / 2) {
                let a0 = a_orig[2 * x];
                let a1 = a_orig[2 * x + 1];
                let b0 = b_orig[2 * x];
                let b1 = b_orig[2 * x + 1];
                assert_eq!(a[x], a0 + challenge * (a1 + a0), "log_n={log_n}, x={x}");
                assert_eq!(b[x], b0 + challenge * (b1 + b0), "log_n={log_n}, x={x}");
            }
        }
    }

    /// **The URM kernel's C side**: `C_s · interpolate(round1_c, k_skip, z)`
    /// equals `ĉ(z, r_rest)` computed by direct folding (Lagrange at z, then
    /// bind each `r_rest` value). The protocol sends `round1_c` only inside its
    /// sum with the AB half, so this is what pins that half on its own.
    #[test]
    fn c_eval_from_round1_c_matches_direct_fold() {
        use crate::zerocheck::univariate_skip_optimized::{
            c_s, medium_challenges, round1_shift_reduce_extract_c_packed_padded, small_challenges,
        };
        use pcs::ntt::{AdditiveNttGf8, InvNttTableByteSingleGf8};
        use primitives::field::F8;

        const K_SKIP: usize = 6;
        const N_INNER: usize = 7;

        for &m in &[14usize, 15, 16] {
            let mut rng = Rng::new(500 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let c = rng.bits(1 << m);

            // Build the equality tail with the seven protocol-fixed constants,
            // matching how `prove` constructs it.
            let mut r = vec![F192::ZERO; m - K_SKIP];
            for (i, v) in small_challenges().iter().enumerate() {
                r[i] = *v;
            }
            for (i, v) in medium_challenges().iter().enumerate() {
                r[3 + i] = *v;
            }
            for slot in r[N_INNER..].iter_mut() {
                *slot = rng.ext();
            }
            let z = rng.ext();

            let a_packed = pack_bits(&a);
            let b_packed = pack_bits(&b);
            let c_packed = pack_bits(&c);

            let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
            let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
            let inv_table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
            let (_round1_ab, round1_c) = round1_shift_reduce_extract_c_packed_padded(
                &a_packed,
                &b_packed,
                &c_packed,
                m,
                K_SKIP,
                &r,
                &inv_table,
                &PaddingSpec::dense(m),
            );

            // Path A: interpolate round1_c at z, scale by C_s.
            let c_eval_via_interpolation = c_s() * interpolate_at_z_on_lambda(&round1_c, K_SKIP, z);

            // Path B: direct fold of c at z (Lagrange) then bind each
            // r_rest element with fold_in_place_single.
            let weights = lagrange_weights_naive(K_SKIP, z);
            let mut c_mlv = fold_at_z_naive(&c, m, K_SKIP, &weights);
            for &r_val in &r {
                fold_in_place_single(&mut c_mlv, r_val);
            }
            assert_eq!(c_mlv.len(), 1);
            let c_eval_via_fold = c_mlv[0];

            assert_eq!(
                c_eval_via_interpolation, c_eval_via_fold,
                "c-claim identity broken at m={m}"
            );
        }
    }

    /// The table rounds, with the padding tail cut to its template or not, against folding each table
    /// (`fold_in_place_*`) and summing naively (`naive_message`), down to the fused rounds' size gate.
    ///
    /// Shapes: `(log_n, positions per block, live blocks)`. The last of each repeats from block `live_blocks`.
    #[test]
    fn table_rounds_match_naive() {
        let mut rng = Rng::new(310);
        for (log_n, period, live_blocks) in [
            (12usize, 16usize, 100usize),
            (7, 4, 3),
            (6, 2, 5),
            (8, 16, 2),
            (12, 16, 255),
            (12, 16, 256),
            (12, 8, 7),
            (12, 2, 3),
            (12, 32, 1),
            (11, 4, 77),
            (14, 16, 700),
            (14, 8, 1000),
        ] {
            let n = 1usize << log_n;
            let n_blocks_log = log_n - period.trailing_zeros() as usize;
            let m = log_n + 6;
            let padding = PaddingSpec {
                k_log: m - n_blocks_log,
                useful_bits_per_block: 1 << (m - n_blocks_log),
                live_blocks,
            };
            let mut dense: [Vec<F192>; 3] = std::array::from_fn(|_| (0..n).map(|_| rng.ext()).collect());
            for t in &mut dense {
                for p in (live_blocks + 1) * period..n {
                    t[p] = t[live_blocks * period + p % period];
                }
            }
            // The stored tables may stop after the template block.
            let stored_len = (live_blocks * period + period).min(n);
            let cut = dense.each_ref().map(|t| t[..stored_len].to_vec());
            let mut tables = Tables::new(dense.clone(), log_n, m, &padding);
            assert_eq!(tables.dense(), dense, "tables, log_n={log_n}, live={live_blocks}");
            assert_eq!(
                Tables::new(cut, log_n, m, &padding).dense(),
                dense,
                "cut, log_n={log_n}, live={live_blocks}"
            );
            for log in (6..=log_n).rev() {
                assert!(tables.can_fold());
                let (chi, r_eq) = (rng.ext(), rng.ext_vec(log - 2));
                let message = tables.fold_round(chi, &r_eq);
                let [a, b, c] = &mut dense;
                fold_in_place_pair(a, b, chi);
                fold_in_place_single(c, chi);
                assert_eq!(
                    message,
                    naive_message(&dense, &r_eq),
                    "message, log_n={log_n}, live={live_blocks}"
                );
                assert_eq!(tables.dense(), dense, "tables, log_n={log_n}, live={live_blocks}");
            }
        }
    }

    /// A random `a, b, c` witness over `2^m` slots, bits and packed.
    fn random_witness(rng: &mut Rng, m: usize) -> ([Vec<bool>; 3], [Vec<u8>; 3]) {
        let bits = [rng.bits(1 << m), rng.bits(1 << m), rng.bits(1 << m)];
        let packed = [0, 1, 2].map(|i| pack_bits(&bits[i]));
        (bits, packed)
    }

    fn packed(p: &[Vec<u8>; 3]) -> PackedWitness<'_> {
        PackedWitness {
            a: &p[0],
            b: &p[1],
            c: &p[2],
        }
    }

    /// The naive message of one round on stored tables: `(G(1), G(inf))` with `c` added to `G(1)`.
    fn naive_message(t: &[Vec<F192>; 3], r_eq: &[F192]) -> (F192, F192) {
        let (g1, g_inf) = round_pair_naive(&t[0], &t[1], r_eq);
        (g1 + round_single_naive(&t[2], r_eq), g_inf)
    }

    /// Every bit-round kernel at every level against the naive route: fold at z, bind each rho, then sum.
    #[test]
    fn bit_rounds_match_naive() {
        const K_SKIP: usize = 6;
        for m in [13usize, 14, 15] {
            let mut rng = Rng::new(0xB17_0000 + m as u64);
            let (bits, packed_bits) = random_witness(&mut rng, m);
            let n_mlv = m - K_SKIP;
            let z = rng.ext();
            let r_rest = rng.ext_vec(n_mlv);
            let rho = rng.ext_vec(n_mlv);
            let lagrange = lagrange_weights_naive(K_SKIP, z);
            let dense = PaddingSpec::dense(m);

            // Level 0 of the naive route: the univariate-skip fold at z.
            let mut tables = [0, 1, 2].map(|i| fold_at_z_naive(&bits[i], m, K_SKIP, &lagrange));
            for t in 0..=4 {
                let fold = BitFold::at_level(&lagrange, &rho[..t]);
                let r_eq = &r_rest[t + 1..];
                let expected = naive_message(&tables, r_eq);

                // The storing kernel: this level's message and tables.
                let (message, stored) = bit_round_materialize(packed(&packed_bits), &fold, r_eq, &dense);
                assert_eq!(message, expected, "store message, m={m}, t={t}");
                for (got, want) in stored.iter().zip(&tables) {
                    assert_eq!(&got[..], &want[..], "stored table, m={m}, t={t}");
                }

                // Bind rho_{t+1} on the naive side; the pair kernel's second round must land on it.
                let [ta, tb, tc] = &mut tables;
                fold_in_place_pair(ta, tb, rho[t]);
                fold_in_place_single(tc, rho[t]);
                let pair = bit_round_pair(packed(&packed_bits), &fold, r_eq, &dense);
                assert_eq!(pair.first, expected, "pair first round, m={m}, t={t}");
                assert_eq!(
                    pair.second(rho[t]),
                    naive_message(&tables, &r_rest[t + 2..]),
                    "pair second round, m={m}, t={t}"
                );
            }
        }
    }

    /// Skipping whole padding pairs and quads, and folding a tail of identical blocks once, changes nothing.
    ///
    /// Shapes: BLAKE2s (k_log = 14, 16,000 bits), an odd boundary, and a two-block-per-row stride.
    /// The last field is the first of the identical blocks, or all of them when none repeat.
    /// The tail runs read `a` and `b` cut after that block.
    #[test]
    fn bit_rounds_skip_padding_exactly() {
        const K_SKIP: usize = 6;
        for (m, k_log, useful, live_blocks) in [
            (17usize, 14usize, 16_000usize, 8usize),
            (17, 14, 15_409, 8),
            (18, 15, 31_401, 8),
            (17, 14, 16_000, 5),
            (17, 14, 15_409, 3),
            (17, 14, 16_000, 1),
            (18, 15, 31_401, 2),
            (18, 15, 31_401, 0),
        ] {
            let mut rng = Rng::new(0xFADE_F00D + (k_log * 31 + useful) as u64);
            let ([mut a, mut b, mut c], _) = random_witness(&mut rng, m);
            // Zero bits [useful, 2^k_log) of every block, as the hash witness does, and repeat the first tail block.
            for x in [&mut a, &mut b, &mut c] {
                for block in x.chunks_mut(1 << k_log) {
                    block[useful..].fill(false);
                }
                for j in live_blocks + 1..x.len() >> k_log {
                    x.copy_within(live_blocks << k_log..(live_blocks + 1) << k_log, j << k_log);
                }
            }
            let packed_bits = [pack_bits(&a), pack_bits(&b), pack_bits(&c)];
            let padding = PaddingSpec {
                k_log,
                useful_bits_per_block: useful,
                live_blocks,
            };
            let lagrange = lagrange_weights_naive(K_SKIP, rng.ext());
            let r_rest = rng.ext_vec(m - K_SKIP);
            let rho = rng.ext_vec(4);
            for t in 0..=4 {
                let fold = BitFold::at_level(&lagrange, &rho[..t]);
                let r_eq = &r_rest[t + 1..];
                // With a tail, `a` and `b` stop after the template block.
                let run = |p: &PaddingSpec| {
                    let ab_len = ((p.live_blocks + 1) << (p.k_log - 3)).min(packed_bits[2].len());
                    let bits = PackedWitness {
                        a: &packed_bits[0][..ab_len],
                        b: &packed_bits[1][..ab_len],
                        c: &packed_bits[2],
                    };
                    (
                        bit_round_pair(bits, &fold, r_eq, p),
                        bit_round_materialize(bits, &fold, r_eq, p),
                    )
                };
                let ((pair_d, (msg_d, tab_d)), (pair_p, (msg_p, tab_p))) = (run(&PaddingSpec::dense(m)), run(&padding));
                assert_eq!(pair_d, pair_p, "pair, m={m}, useful={useful}, t={t}");
                assert_eq!(msg_d, msg_p, "store message, m={m}, useful={useful}, t={t}");
                for (d, p) in tab_d.iter().zip(&tab_p) {
                    assert_eq!(&d[..p.len()], &p[..], "stored table, m={m}, useful={useful}, t={t}");
                }
            }
        }
    }

    /// Strong cross-check: compute G(0), G(1), G(∞) by direct sum (using the
    /// LSB-first index convention `a_mlv(0, x') = a[2x']`, `a_mlv(1, x') = a[2x'+1]`),
    /// then verify that G interpolated through those three values agrees with
    /// the direct multilinear evaluation at a fresh random X: confirming G
    /// genuinely has degree ≤ 2.
    ///
    /// Also verifies `round_pair_naive` returns `(r[0] · G(1), G(∞))`.
    #[test]
    fn round_pair_message_has_degree_two() {
        let m = 6;
        let k_skip = 3;
        let mut rng = Rng::new(55);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let z = rng.ext();
        let r_eq = rng.ext_vec(m - k_skip - 1);

        let weights = lagrange_weights_naive(k_skip, z);
        let a_mlv = fold_at_z_naive(&a, m, k_skip, &weights);
        let b_mlv = fold_at_z_naive(&b, m, k_skip, &weights);

        let n = a_mlv.len();
        let half = n / 2;
        let eq_remaining = build_eq(&r_eq);

        // G(0), G(1), G(∞) by direct definition.
        let mut g0 = F192::ZERO;
        let mut g1 = F192::ZERO;
        let mut g_inf = F192::ZERO;
        for x_prime in 0..half {
            let a0 = a_mlv[2 * x_prime];
            let a1 = a_mlv[2 * x_prime + 1];
            let b0 = b_mlv[2 * x_prime];
            let b1 = b_mlv[2 * x_prime + 1];
            let eq_x = eq_remaining[x_prime];
            g0 += eq_x * a0 * b0;
            g1 += eq_x * a1 * b1;
            g_inf += eq_x * (a0 + a1) * (b0 + b1);
        }

        let (msg_1, msg_inf) = round_pair_naive(&a_mlv, &b_mlv, &r_eq);
        assert_eq!(msg_1, g1);
        assert_eq!(msg_inf, g_inf);

        // Degree-2 check: G(X) reconstructed through (G(0), G(1), G(∞)) must
        // agree with the direct multilinear evaluation at a fresh point X.
        // Char-2 interpolation: G(X) = G(0) + X·(G(0)+G(1)) + X·(X+1)·G(∞).
        let x = rng.ext();
        let g_via_poly = g0 + x * (g0 + g1) + x * (x + F192::ONE) * g_inf;
        let mut g_via_sum = F192::ZERO;
        for x_prime in 0..half {
            let a0 = a_mlv[2 * x_prime];
            let a1 = a_mlv[2 * x_prime + 1];
            let b0 = b_mlv[2 * x_prime];
            let b1 = b_mlv[2 * x_prime + 1];
            let a_x = a0 + x * (a0 + a1);
            let b_x = b0 + x * (b0 + b1);
            g_via_sum += eq_remaining[x_prime] * a_x * b_x;
        }
        assert_eq!(g_via_poly, g_via_sum);
    }
}
