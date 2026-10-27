//! The two-round pass with GFNI lanes on instances.
//!
//! The witness is `2^n` instances (blocks), `2^k_log` bits each. The fold's 64 lanes are 64 instances at one
//! position inside the instance, so the lanes share the in-instance part of the eq weight:
//!
//! ```text
//!     eq(r_quad, q) = eq(r_in, qi) * eq(r_blk, instance)        q = instance * 2^in_log + qi
//! ```
//!
//! - `eq(r_in, qi)` is in the fold matrices of `a` and `c`, one matrix set per in-instance quad `qi`.
//! - So a quad costs eight unreduced products per eight instances, and no reduction or `eq * c` product.
//! - `c` is linear: its folds add up per quad class over the whole group, and join the sums once per group.
//! - A group's sums are per lane. One product per lane by `eq(r_blk, instance)` ends the group.
//!
//! The input is a 64-byte line from each of the group's instances, 64 instances a block apart. At 8 bytes a position,
//! a line holds eight positions, two quads.

use super::{RoundPair, padding_pairs};
use crate::zerocheck::PaddingSpec;
use crate::zerocheck::bit_fold::{
    BitFold,
    gfni::{ByteAcc, LineRegs, byte_matrices},
};
use crate::zerocheck::bits::Bits;
use crate::zerocheck::univariate_skip::{build_eq, eq_mass_below};
use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
use primitives::field::{F192, F192x8, Prod8};

/// Steps ahead that the lines are prefetched.
const PREFETCH_STEPS: usize = 2;

/// Instances per group: one per GFNI lane.
const LANES: usize = 64;

/// The eight sums of a quad, per lane, accumulated unreduced: `[slot][vector of eight lanes]`.
type QuadSums = [[Prod8; 8]; 8];

/// Bytes in a position: one 64-bit row, the first pass of the pair rounds.
const CHUNKS: usize = 8;

/// Positions in a 64-byte line, and in a step: a step is one line, two quads.
const STEP: usize = 64 / CHUNKS;

/// The pair pass for rows of `CHUNKS` bytes, or `None` when an instance holds under two quads.
///
/// Same message as [`super::bit_round_pair_kernel`], on a witness of whole instances.
pub(super) fn pair(bits: Bits<'_>, fold: &BitFold, r_eq: &[F192], padding: &PaddingSpec) -> Option<RoundPair> {
    assert_eq!(fold.n_chunks(), CHUNKS);
    let m = (8 * bits.len_bytes()).trailing_zeros() as usize;
    // A quad covers 4 positions of 2^6 skip bits times 2^t bound rows.
    let quad_log = (32 * CHUNKS).trailing_zeros() as usize;
    let in_log = padding.k_log.checked_sub(quad_log).filter(|&l| l >= 1)?;
    assert_eq!(r_eq.len(), m - quad_log + 1);
    let (r_v, r_quad) = (r_eq[0], &r_eq[1..]);
    let (r_in, r_blk) = r_quad.split_at(in_log);

    // Instances 0..=last are read: all, or those up to the template of the identical padding tail.
    let n_inst = 1usize << r_blk.len();
    let tail = padding.live_blocks < n_inst;
    let last = if tail { padding.live_blocks } else { n_inst - 1 };
    let inst_bytes = 1usize << (padding.k_log - 3);
    if let Bits::Given(p) = bits {
        assert!(p.a.len() >= (last + 1) * inst_bytes && p.b.len() == p.a.len());
    }

    // Quads wholly in an instance's zero padding fold to zero.
    let (_, live) = padding_pairs(padding, quad_log - 1);
    let n_lines = (4 * live.min(1 << in_log)).div_ceil(STEP);
    let n_quads = n_lines * STEP / 4;

    // The fold matrices: `a` and `c` carry the in-instance eq, `b` does not.
    let weights = fold.weights();
    let eq_in = build_eq(r_in);
    let ma: Vec<_> = eq_in[..n_quads]
        .iter()
        .flat_map(|&e| {
            let e = F192x8::splat(e);
            let scaled = weights
                .as_chunks::<8>()
                .0
                .iter()
                .map(|w| (F192x8::from_array(w) * e).to_array())
                .collect::<Vec<_>>();
            byte_matrices(scaled.as_flattened())
        })
        .collect();
    let mb = byte_matrices(weights);

    // The lane weights `eq(r_blk, instance)` split as `lo[lane] * hi[group]`.
    let n_lo = r_blk.len().min(LANES.trailing_zeros() as usize);
    let hi = build_eq(&r_blk[n_lo..]);
    let mut lo = build_eq(&r_blk[..n_lo]);
    lo.resize(LANES, F192::ZERO);
    let pack = |w: &[F192]| -> [F192x8; 8] {
        std::array::from_fn(|v| F192x8::from_array(&std::array::from_fn(|l| w[ByteAcc::x8_lane(v, l)])))
    };
    let lo_packed = pack(&lo);
    // The template's group weighs it by the eq mass of the whole tail, and the lanes after it by nothing.
    let (template_group, template_lane) = (padding.live_blocks / LANES, padding.live_blocks % LANES);
    let template_weights = tail.then(|| {
        let mut w: Vec<F192> = lo.iter().map(|&l| l * hi[template_group]).collect();
        w[template_lane] = F192::ONE + eq_mass_below(r_blk, padding.live_blocks);
        w[template_lane + 1..].fill(F192::ZERO);
        pack(&w)
    });

    let n_groups = (last + 1).div_ceil(LANES);
    let sums = parallel::map_reduce_with_state(
        n_groups,
        || bits.scratch(LANES * inst_bytes),
        || [F192::ZERO; 8],
        |scratch, total, g| {
            let (weights, scale) = match &template_weights {
                Some(w) if g == template_group => (w, F192::ONE),
                _ => (&lo_packed, hi[g]),
            };
            // Each lane's instance, the template's again past the last: those lanes weigh zero.
            let window = bits.window(
                g * LANES * inst_bytes..((g + 1) * LANES).min(last + 1) * inst_bytes,
                scratch,
            );
            let tables = window
                .tables
                .map(|t| lanes_of(t, window.first, (g * LANES, last), inst_bytes));
            // Rebuilt tables are in cache already, so only given ones prefetch the next group.
            let next = match bits {
                Bits::Given(p) if g + 1 < n_groups => {
                    Some([p.a, p.b, p.c].map(|t| lanes_of(t, 0, ((g + 1) * LANES, last), inst_bytes)))
                }
                _ => None,
            };
            let sums = group(&tables, next.as_ref(), n_lines, &ma, &mb, weights);
            *total = std::array::from_fn(|i| total[i] + sums[i] * scale);
        },
        |x, y| std::array::from_fn(|i| x[i] + y[i]),
    );

    // Slots 0, 1 hold round t's G(1) at v = 0, 1, and slots 2, 3 its G(inf), as in the position-lane kernel.
    let split_v = |v0: F192, v1: F192| v0 + r_v * (v0 + v1);
    Some(RoundPair {
        first: (split_v(sums[0], sums[1]), split_v(sums[2], sums[3])),
        second: [[sums[4], sums[1], sums[3]], [sums[5], sums[6], sums[7]]],
    })
}

/// The 64-byte lines of instances `first..first + 64` of `table`, the instance `last` again past it.
///
/// `table` starts at byte `at` of the whole.
fn lanes_of(table: &[u8], at: usize, (first, last): (usize, usize), inst_bytes: usize) -> [&[[u8; 64]]; LANES] {
    std::array::from_fn(|l| {
        let inst = (first + l).min(last);
        table[inst * inst_bytes - at..][..inst_bytes].as_chunks().0
    })
}

/// The eight sums of a quad's folded `a` and `b`, per lane, added to `acc`: `a` is weighted by its `eq`, so the products are too.
///
/// The four positions of the quad are `[j][vector]`. The slots are those of [`super::bit_round_pair_kernel`]
/// without the `c` terms.
#[inline(always)]
fn quad_sums(a: &[[F192x8; 8]; 4], b: &[[F192x8; 8]; 4], acc: &mut QuadSums) {
    for v in 0..8 {
        let [e0, e1, e2, e3] = std::array::from_fn(|j| a[j][v]);
        let [b0, b1, b2, b3] = std::array::from_fn(|j| b[j][v]);
        let (du0, du1, dv0, dv1) = (e0 + e1, e2 + e3, e0 + e2, e1 + e3);
        let (eu0, eu1, ev0, ev1) = (b0 + b1, b2 + b3, b0 + b2, b1 + b3);
        acc[0][v] ^= e1.mul_unreduced(b1);
        acc[1][v] ^= e3.mul_unreduced(b3);
        acc[2][v] ^= du0.mul_unreduced(eu0);
        acc[3][v] ^= du1.mul_unreduced(eu1);
        acc[4][v] ^= e2.mul_unreduced(b2);
        acc[5][v] ^= dv0.mul_unreduced(ev0);
        acc[6][v] ^= dv1.mul_unreduced(ev1);
        acc[7][v] ^= (du0 + du1).mul_unreduced(eu0 + eu1);
    }
}

/// The eight sums of one group of instances, each lane weighted by `weights` and the lanes added.
///
/// `tables` holds each lane's lines of `a`, `b` and `c`, and `next` the next group's. `ma` and `mb` are the fold
/// matrices of `a` and `c` (per quad) and `b`, one per byte of a position.
fn group(
    tables: &[[&[[u8; 64]]; LANES]; 3],
    next: Option<&[[&[[u8; 64]]; LANES]; 3]>,
    n_lines: usize,
    ma: &[[u64; 24]],
    mb: &[[u64; 24]],
    weights: &[F192x8; 8],
) -> [F192; 8] {
    let mb: &[[u64; 24]; CHUNKS] = mb.try_into().expect("a matrix per byte");
    let mut regs: [LineRegs; 3] = std::array::from_fn(|_| LineRegs::new());
    let mut acc: QuadSums = [[Prod8::ZERO; 8]; 8];
    // The `c` folds of the quad classes 1, 2, 3, added over the quads.
    let mut c_fold = [ByteAcc::zero(); 3];
    // The folded `a` and `b` of the quad.
    let mut folded = [[[F192x8::ZERO; 8]; 4]; 2];
    for step in 0..n_lines {
        for (r, lines) in regs.iter_mut().zip(tables) {
            r.load(std::array::from_fn(|lane| &lines[lane][step]));
        }
        let [ra, rb, rc] = &regs;
        for i in 0..STEP {
            // Position `i` of the step is position `i % 4` of quad `2 step + i / 4`.
            let (q, j) = (i / 4, i % 4);
            let m: &[[u64; 24]; CHUNKS] = ma[(2 * step + q) * CHUNKS..][..CHUNKS]
                .try_into()
                .expect("a matrix per byte");
            // The lines of `PREFETCH_STEPS` ahead: the next group's past the last, spread over the step.
            let prefetch = |sub: usize| {
                let (target, ahead) = match step + PREFETCH_STEPS {
                    s if s < n_lines => (Some(tables), s),
                    s => (next, s - n_lines),
                };
                let slot = i * 3 + sub;
                for n in slot * (3 * LANES) / (3 * STEP)..(slot + 1) * (3 * LANES) / (3 * STEP) {
                    if let Some(target) = target {
                        let line = target[n % 3][n / 3].as_ptr().wrapping_add(ahead);
                        // SAFETY: a prefetch of any address is harmless.
                        unsafe { _mm_prefetch::<_MM_HINT_T0>(line.cast()) };
                    }
                }
            };
            prefetch(0);
            let mut fold = ByteAcc::zero();
            fold.add_fold(ra, i * CHUNKS, m);
            folded[0][j] = fold.to_x8();
            prefetch(1);
            let mut fold = ByteAcc::zero();
            fold.add_fold(rb, i * CHUNKS, mb);
            folded[1][j] = fold.to_x8();
            prefetch(2);
            if j > 0 {
                c_fold[j - 1].add_fold(rc, i * CHUNKS, m);
            }
            if j == 3 {
                quad_sums(&folded[0], &folded[1], &mut acc);
            }
        }
    }

    // Per lane: the sums, plus `c` for the slots that carry it (`c1`, `c3`, `c2`), times the lane's weight.
    let c = c_fold.map(|c| c.to_x8());
    std::array::from_fn(|k| {
        let c_class = match k {
            0 => Some(0),
            1 => Some(2),
            4 => Some(1),
            _ => None,
        };
        let mut sum = Prod8::ZERO;
        for v in 0..8 {
            let lanes = acc[k][v].reduce_lanes();
            sum ^= c_class.map_or(lanes, |j| lanes + c[j][v]).mul_unreduced(weights[v]);
        }
        sum.reduce()
    })
}

#[cfg(test)]
mod tests {
    use super::super::{PackedWitness, bit_round_pair_kernel};
    use super::*;
    use primitives::multilinear::lagrange_weights_naive;
    use primitives::test_rng::Rng;

    /// The instance-lane pass against the position-lane kernel.
    ///
    /// Shapes: `(k_log, useful bits, log2 instances, live blocks)`, with tails starting in, before and after a group.
    #[test]
    fn matches_the_position_lane_kernel() {
        let mut rng = Rng::new(0x1A_2E5);
        for (k_log, useful, n_log, live_blocks) in [
            (14usize, 16_000usize, 7usize, 1000usize),
            (14, 16_000, 7, 127),
            (14, 16_000, 7, 100),
            (14, 16_000, 7, 64),
            (14, 16_000, 7, 63),
            (14, 16_000, 7, 1),
            (14, 16_000, 7, 0),
            (14, 15_409, 6, 3),
            (14, 15_409, 3, 5),
            (14, 15_409, 3, 8),
            (15, 31_401, 2, 2),
            (13, 7_000, 6, 40),
            (13, 8_192, 5, 100),
        ] {
            let m = k_log + n_log;
            let inst_bytes = 1usize << (k_log - 3);
            let n_inst = 1usize << n_log;
            let [a, b, c] = std::array::from_fn(|_| {
                let mut x: Vec<u8> = (0..n_inst * inst_bytes).map(|_| rng.next_u64() as u8).collect();
                for block in x.chunks_mut(inst_bytes) {
                    if useful < 1 << k_log {
                        block[useful / 8] &= (1u8 << (useful % 8)) - 1;
                        block[useful.div_ceil(8)..].fill(0);
                    }
                }
                for j in live_blocks + 1..n_inst {
                    x.copy_within(live_blocks * inst_bytes..(live_blocks + 1) * inst_bytes, j * inst_bytes);
                }
                x
            });
            let padding = PaddingSpec {
                k_log,
                useful_bits_per_block: useful,
                live_blocks,
            };
            let ab_len = ((live_blocks + 1) * inst_bytes).min(a.len());
            let bits = PackedWitness {
                a: &a[..ab_len],
                b: &b[..ab_len],
                c: &c,
            };
            let lagrange = lagrange_weights_naive(6, rng.ext());
            let r_rest = rng.ext_vec(m - 6);
            let fold = BitFold::at_level(&lagrange, &[]);
            let r_eq = &r_rest[1..];
            assert_eq!(
                pair(bits.into(), &fold, r_eq, &padding),
                Some(bit_round_pair_kernel::<CHUNKS>(bits.into(), &fold, r_eq, &padding)),
                "k_log={k_log}, useful={useful}, n_log={n_log}, live={live_blocks}"
            );
        }
    }
}
