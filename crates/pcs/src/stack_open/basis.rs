//! The opening's weight as terms, and the projection of the stack onto them.
//!
//! A claim's weight is a product over the bits of the stack index `y = l * 2^lane_vars + x`, each bit
//! carrying a [`Factor`]. Splitting `x` into a low part `x_lo` of `m = min(10, lane_vars)` bits and a
//! block index `h` gives
//!
//! ```text
//!     claim:  lambda * L(l) * gamma(h) * pi(x_lo)
//!     ring:   Phi(e(x) * s(l)),   e(x) = e_lo(x_lo) * e_hi(h)
//! ```
//!
//! `Phi` is F2-linear, so a ring claim is `sum_w bit_w(e(x)) * Phi(basis_w * s(l))`. The weight is then
//! `b(l, x) = sum_t A(l, t) * K_t(x)` over one term per claim and 192 per ring claim, and the table
//! `P(l, t) = sum_x stack(l, x) * K_t(x)` takes the stack's place in the lane rounds.

use primitives::bit_fold::BitFold;
use primitives::bits::bit_transpose_64bytes;
use primitives::field::{
    Acc8, F64, F192, F192Unreduced, Weights8, dot_base, fold_lones8_ext, msg_lone8_ext, xor_into8,
};

use super::{RingSwitchOpen, StackClaim};
use crate::ring_switch::{RingWeight, coordinate_basis};
use crate::tensor_algebra::{DEGREE_E, transpose_s_hat};
use crate::whir::{InitialBasis, LaneLift};

/// Bits of a block's low part.
const LOW_VARS: usize = 10;
/// Packing width: the bit rows of a block of words.
const ROWS: usize = 64;

/// `g * t`, elementwise: the lone fold `t + r * t` at `r = g + 1`.
fn scale8(t: &[F192; 8], g: F192) -> [F192; 8] {
    fold_lones8_ext(t, g + F192::ONE)
}

/// What a weight does to one bit of the stack index.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Factor {
    /// The bit must take this value.
    Fixed(bool),
    /// `eq(p, bit)`.
    Eq(F192),
}

/// `prod_i factors[i](bit i of x)` for every `x`.
fn table(factors: &[Factor]) -> Vec<F192> {
    let mut out = Vec::with_capacity(1 << factors.len());
    out.push(F192::ONE);
    for &factor in factors {
        let len = out.len();
        out.resize(2 * len, F192::ZERO);
        for i in 0..len {
            let v = out[i];
            match factor {
                Factor::Eq(p) => {
                    let high = v * p;
                    out[i + len] = high;
                    out[i] = v + high;
                }
                Factor::Fixed(true) => {
                    out[i + len] = v;
                    out[i] = F192::ZERO;
                }
                Factor::Fixed(false) => {}
            }
        }
    }
    out
}

/// A claim's factors on the low `bits` bits of the stack index.
fn claim_factors(claim: &StackClaim, bits: usize) -> Vec<Factor> {
    let (offset, slot, stride_log, point) = match claim {
        StackClaim::Point { offset, low_point, .. } => (*offset, 0, 0, low_point),
        StackClaim::Strided {
            offset,
            slot,
            stride_log,
            point,
            ..
        } => (*offset, *slot, *stride_log, point),
    };
    (0..bits)
        .map(|b| {
            if b < stride_log {
                Factor::Fixed((slot >> b) & 1 == 1)
            } else if b < stride_log + point.len() {
                Factor::Eq(point[b - stride_log])
            } else {
                Factor::Fixed((offset >> b) & 1 == 1)
            }
        })
        .collect()
}

/// A ring claim's factors on the low `bits` bits of the stack index.
fn ring_factors(point: &[F192], offset: usize, bits: usize) -> Vec<Factor> {
    (0..bits)
        .map(|b| match point.get(b) {
            Some(&p) => Factor::Eq(p),
            None => Factor::Fixed((offset >> b) & 1 == 1),
        })
        .collect()
}

/// The claims that share one low factor list `pi`, and `pi`'s form.
struct Pattern {
    factors: Vec<Factor>,
    claims: Vec<usize>,
    low: Low,
    /// The blocks where some claim of the pattern is nonzero.
    active: Vec<bool>,
}

enum Low {
    /// Every position: the table for `add`, packed for the dot against a block.
    Dense { table: Vec<F192>, packed: Vec<Weights8> },
    /// Few positions: the support of `pi`.
    Sparse(Vec<(usize, F192)>),
}

impl Low {
    fn new(factors: &[Factor]) -> Self {
        let table = table(factors);
        let eqs = factors.iter().filter(|f| matches!(f, Factor::Eq(_))).count();
        if eqs + 4 <= factors.len() {
            let support = table.iter().enumerate().filter(|(_, c)| **c != F192::ZERO);
            Low::Sparse(support.map(|(x, &c)| (x, c)).collect())
        } else {
            let packed = table.as_chunks::<8>().0.iter().map(Weights8::new).collect();
            Low::Dense { table, packed }
        }
    }

    /// `sum_x block[x] * pi(x)`.
    fn dot(&self, block: &[F64]) -> F192 {
        match self {
            Low::Dense { packed, .. } => dot_base(packed, block).reduce(),
            Low::Sparse(support) => support
                .iter()
                .fold(F192Unreduced::ZERO, |acc, &(x, c)| acc ^ c.mul_base_unreduced(block[x]))
                .reduce(),
        }
    }

    /// `out[x] += g * pi(x)`.
    fn add(&self, g: F192, out: &mut [F192]) {
        match self {
            Low::Dense { table, .. } => {
                for (o, t) in out.as_chunks_mut::<8>().0.iter_mut().zip(table.as_chunks::<8>().0) {
                    xor_into8(o, &scale8(t, g));
                }
            }
            Low::Sparse(support) => {
                for &(x, c) in support {
                    out[x] += g * c;
                }
            }
        }
    }
}

/// A ring claim's block structure and where its terms sit.
struct Ring {
    /// Folds a bit row of a block against `e_lo`.
    fold: BitFold,
    /// `e_hi(h)` for every block.
    e_hi: Vec<F192>,
    point: Vec<F192>,
    /// Where `e` starts inside a lane block.
    x_offset: usize,
    /// The first of its 192 terms.
    first: usize,
}

/// The factored terms of the opening's weight.
struct Terms {
    /// Bits of a lane block.
    lane_vars: usize,
    /// Bits of a block's low part.
    low_vars: usize,
    /// `gamma(h)` of every claim.
    gamma: Vec<Vec<F192>>,
    /// The blocks a claim's `gamma` is nonzero on, widened to whole vectors.
    ranges: Vec<std::ops::Range<usize>>,
    patterns: Vec<Pattern>,
    rings: Vec<Ring>,
}

impl Terms {
    fn n_claims(&self) -> usize {
        self.gamma.len()
    }

    fn n_terms(&self) -> usize {
        self.n_claims() + DEGREE_E * self.rings.len()
    }

    /// `P(l, .)` for the lane's words.
    fn project_lane<const CHUNKS: usize>(&self, lane: &[F64], row: &mut [F192]) {
        let n_blocks = lane.len() >> self.low_vars;
        let mut q = vec![F192::ZERO; n_blocks * self.patterns.len()];
        let mut sigmas = vec![F192::ZERO; ROWS * self.rings.len()];
        let mut cols = [[0u8; ROWS]; CHUNKS];
        let mut folded = [F192::ZERO; ROWS];
        for (h, block) in lane.chunks_exact(1 << self.low_vars).enumerate() {
            for (pattern, q) in self.patterns.iter().zip(q.chunks_exact_mut(n_blocks)) {
                if pattern.active[h] {
                    q[h] = pattern.low.dot(block);
                }
            }
            let mut transposed = false;
            for (ring, sigma) in self.rings.iter().zip(sigmas.chunks_exact_mut(ROWS)) {
                let e_hi = ring.e_hi[h];
                if e_hi == F192::ZERO {
                    continue;
                }
                if !transposed {
                    for (col, group) in cols.iter_mut().zip(block.as_chunks::<8>().0) {
                        // SAFETY: `F64` is a `u64`, so eight of them are 64 bytes.
                        bit_transpose_64bytes(unsafe { &*group.as_ptr().cast() }, col);
                    }
                    transposed = true;
                }
                ring.fold.fold_columns(&cols, &mut folded);
                for (s, v) in sigma.as_chunks_mut::<8>().0.iter_mut().zip(folded.as_chunks::<8>().0) {
                    xor_into8(s, &scale8(v, e_hi));
                }
            }
        }
        // `P(l, c) = sum_h gamma_c(h) * Q(h)` over the claim's blocks.
        for (p, pattern) in self.patterns.iter().enumerate() {
            let q = &q[p * n_blocks..(p + 1) * n_blocks];
            for &c in &pattern.claims {
                let range = self.ranges[c].clone();
                let (gamma, q) = (&self.gamma[c][range.clone()], &q[range]);
                let ((g8, g_r), (q8, q_r)) = (gamma.as_chunks::<8>(), q.as_chunks::<8>());
                let sum = g8.iter().zip(q8).fold(Acc8::ZERO, |mut acc, (g, q)| {
                    acc ^= msg_lone8_ext(g, q);
                    acc
                });
                row[c] = g_r.iter().zip(q_r).fold(sum.reduce(), |sum, (&g, &q)| sum + g * q);
            }
        }
        for (ring, sigma) in self.rings.iter().zip(sigmas.chunks_exact(ROWS)) {
            for (slot, k) in row[ring.first..].iter_mut().zip(transpose_s_hat(sigma)) {
                *slot = F192::from(k);
            }
        }
    }

    fn project(&self, stack: &[F64]) -> Vec<F192> {
        let (block, n_terms) = (1usize << self.lane_vars, self.n_terms());
        let mut p = vec![F192::ZERO; stack.len() / block * n_terms];
        parallel::chunks_mut(&mut p, n_terms, |l, row| {
            let lane = &stack[l * block..(l + 1) * block];
            match self.low_vars {
                6 => self.project_lane::<8>(lane, row),
                7 => self.project_lane::<16>(lane, row),
                8 => self.project_lane::<32>(lane, row),
                9 => self.project_lane::<64>(lane, row),
                _ => self.project_lane::<128>(lane, row),
            }
        });
        p
    }
}

impl LaneLift for Terms {
    /// `b[h * 2^m + x] = sum_claims A[c] * gamma_c(h) * pi_c(x) + sum_rings Phi'(e(x))`, where
    /// `Phi'` weighs the ring's coordinates by its 192 terms of `a`.
    fn lift(&self, a: &[F192]) -> Vec<F192> {
        let mut b = vec![F192::ZERO; 1 << self.lane_vars];
        parallel::chunks_mut(&mut b, 1 << self.low_vars, |h, block| {
            for pattern in &self.patterns {
                let g = pattern
                    .claims
                    .iter()
                    .fold(F192::ZERO, |sum, &c| sum + a[c] * self.gamma[c][h]);
                if g != F192::ZERO {
                    pattern.low.add(g, block);
                }
            }
        });
        const CHUNK: usize = 1 << 12;
        for ring in &self.rings {
            let vars = ring.point.len().min(self.lane_vars);
            let weight = RingWeight::from_point(
                a[ring.first..ring.first + DEGREE_E].to_vec(),
                ring.point[..vars].to_vec(),
            );
            let dst = &mut b[ring.x_offset..ring.x_offset + (1 << vars)];
            parallel::chunks_mut(dst, CHUNK, |i, out| weight.add_chunk(i * CHUNK, out));
        }
        b
    }
}

/// The opening's weight as terms: its table `P` against the stack, its table `A`, and the lift of `A`'s last row.
///
/// Every claim must lie inside the stack, and `lane_vars` must be at least 6.
pub(super) fn project(
    stack: &[F64],
    lane_vars: usize,
    claims: &[StackClaim],
    lambdas: &[F192],
    ring: &RingSwitchOpen,
    rings: &[RingWeight],
) -> InitialBasis {
    assert_eq!(claims.len(), lambdas.len());
    assert_eq!(rings.len(), ring.claims.len());
    assert!(
        lane_vars >= ROWS.ilog2() as usize,
        "a lane block holds at least one row of bits"
    );
    let block = 1usize << lane_vars;
    let n_lanes = stack.len() / block;
    assert_eq!(stack.len(), n_lanes * block, "the stack is whole lane blocks");
    let (low_vars, lane_bits) = (
        lane_vars.min(LOW_VARS),
        (usize::BITS - (n_lanes - 1).leading_zeros()) as usize,
    );
    let bits = lane_vars + lane_bits;

    // Claim terms: A(l, c) = lambda_c * L_c(l), and the pattern each claim's low factors fall in.
    let mut terms = Terms {
        lane_vars,
        low_vars,
        gamma: Vec::new(),
        ranges: Vec::new(),
        patterns: Vec::new(),
        rings: Vec::new(),
    };
    let mut lane_weights: Vec<Vec<F192>> = Vec::new();
    for (c, claim) in claims.iter().enumerate() {
        let factors = claim_factors(claim, bits);
        let (low, rest) = factors.split_at(low_vars);
        let (mid, lane) = rest.split_at(lane_vars - low_vars);
        terms.gamma.push(table(mid));
        lane_weights.push(table(lane).into_iter().map(|l| lambdas[c] * l).collect());
        match terms.patterns.iter_mut().find(|p| p.factors == low) {
            Some(pattern) => pattern.claims.push(c),
            None => terms.patterns.push(Pattern {
                low: Low::new(low),
                factors: low.to_vec(),
                claims: vec![c],
                active: Vec::new(),
            }),
        }
    }
    let n_blocks = 1usize << (lane_vars - low_vars);
    for gamma in &terms.gamma {
        let nonzero = |h: &usize| gamma[*h] != F192::ZERO;
        let first = (0..n_blocks).find(nonzero).unwrap_or(0);
        let end = (0..n_blocks).rfind(nonzero).map_or(0, |h| h + 1);
        terms.ranges.push(first / 8 * 8..end.next_multiple_of(8).min(n_blocks));
    }
    for pattern in &mut terms.patterns {
        pattern.active = (0..n_blocks)
            .map(|h| pattern.claims.iter().any(|&c| terms.gamma[c][h] != F192::ZERO))
            .collect();
    }
    let mut ring_lanes: Vec<Vec<F192>> = Vec::new();
    for (i, weight) in rings.iter().enumerate() {
        let factors = ring_factors(weight.point(), ring.offset, bits);
        let (low, rest) = factors.split_at(low_vars);
        let (mid, lane) = rest.split_at(lane_vars - low_vars);
        terms.rings.push(Ring {
            fold: BitFold::new(&table(low)),
            e_hi: table(mid),
            point: weight.point().to_vec(),
            x_offset: ring.offset & (block - 1),
            first: claims.len() + DEGREE_E * i,
        });
        ring_lanes.push(table(lane));
    }

    let n_terms = terms.n_terms();
    let mut a = vec![F192::ZERO; n_lanes * n_terms];
    for (l, row) in a.chunks_exact_mut(n_terms).enumerate() {
        for (slot, lane) in row.iter_mut().zip(&lane_weights) {
            *slot = lane[l];
        }
        for ((weight, lane), ring) in rings.iter().zip(&ring_lanes).zip(&terms.rings) {
            let s = lane[l];
            if s != F192::ZERO {
                for (w, slot) in row[ring.first..ring.first + DEGREE_E].iter_mut().enumerate() {
                    *slot = weight.apply(coordinate_basis(w) * s);
                }
            }
        }
    }
    let p = terms.project(stack);
    InitialBasis {
        p,
        a,
        block: n_terms,
        lift: Box::new(terms),
    }
}
