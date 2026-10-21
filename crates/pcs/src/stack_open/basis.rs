use std::mem::MaybeUninit;

#[cfg(test)]
use primitives::field::F64;
use primitives::field::F192;
use primitives::multilinear::fill_eq_table_uninit;

use super::{RingSwitchOpen, StackClaim};
use crate::ring_switch::RingWeight;
#[cfg(test)]
use crate::ring_switch::DeferredRingSwitchOutput;
use crate::whir::{INITIAL_BASIS_CHUNK, LaneWeight};
#[cfg(test)]
use crate::whir::{SumcheckMessage, build_initial_basis};

/// `lambda·eq(point, ·)` on the positions `offset + slot + j·2^stride_log`.
#[derive(Clone)]
struct PointClaim {
    offset: usize,
    slot: usize,
    stride_log: usize,
    point: Vec<F192>,
    scale: F192,
}

/// `x` with bit `bit` taken out, the higher bits shifting down into its place.
fn remove_bit(x: usize, bit: usize) -> usize {
    (x >> (bit + 1) << bit) | (x & ((1 << bit) - 1))
}

/// What a fold with challenge `r` multiplies a weight by whose coordinate is the fixed `bit`.
fn bound(bit: bool, r: F192) -> F192 {
    if bit { r } else { F192::ONE + r }
}

impl PointClaim {
    fn new(claim: &StackClaim, lambda: F192) -> Self {
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
        Self {
            offset,
            slot,
            stride_log,
            point: point.clone(),
            scale: lambda,
        }
    }

    /// The claim with stack index bit `bit` bound to `r`: `(1 + r)·w(.., 0, ..) + r·w(.., 1, ..)`.
    fn fold_bit(&self, bit: usize, r: F192) -> Self {
        let mut next = self.clone();
        next.offset = remove_bit(self.offset, bit);
        if bit < self.stride_log {
            next.scale *= bound((self.slot >> bit) & 1 == 1, r);
            next.slot = remove_bit(self.slot, bit);
            next.stride_log -= 1;
        } else if bit < self.stride_log + self.point.len() {
            let p = next.point.remove(bit - self.stride_log);
            next.scale *= F192::ONE + r + p;
        } else {
            next.scale *= bound((self.offset >> bit) & 1 == 1, r);
        }
        next
    }
}

struct PointWeight {
    claim: PointClaim,
    end: usize,
    stride: usize,
    low_vars: usize,
    high: Vec<F192>,
}

impl PointWeight {
    fn new(claim: PointClaim, chunk_log: usize) -> Self {
        let len = 1usize << (claim.stride_log + claim.point.len());
        let stride = 1usize << claim.stride_log;
        assert!(claim.offset.is_multiple_of(len), "claim must be aligned to its support");
        assert!(claim.slot < stride, "claim slot must fit the stride");
        let low_vars = claim.point.len().min(chunk_log.saturating_sub(claim.stride_log));
        let high_point = &claim.point[low_vars..];
        let mut high = Box::new_uninit_slice(1 << high_point.len()).into_vec();
        fill_eq_table_uninit(high_point, claim.scale, &mut high);
        // SAFETY: the seeded equality build initializes the whole table.
        let high = unsafe { high.into_boxed_slice().assume_init() }.into_vec();
        Self {
            end: claim.offset + len,
            stride,
            low_vars,
            high,
            claim,
        }
    }

    fn add(&self, start: usize, dst: &mut [F192], scratch: &mut [MaybeUninit<F192>]) {
        let base = self.claim.offset + self.claim.slot;
        let lo = start.max(base);
        let hi = (start + dst.len()).min(self.end);
        if lo >= hi {
            return;
        }
        let first = (lo - base).div_ceil(self.stride);
        let end = (hi - base).div_ceil(self.stride);
        if first == end {
            return;
        }
        let len = 1usize << self.low_vars;
        assert!(first.is_multiple_of(len) && end - first == len);
        fill_eq_table_uninit(&self.claim.point[..self.low_vars], self.high[first / len], &mut scratch[..len]);
        // SAFETY: the build above initializes this prefix before the scatter reads it.
        let eq = unsafe { std::slice::from_raw_parts(scratch.as_ptr().cast::<F192>(), len) };
        let dst_offset = base + first * self.stride - start;
        for (i, &value) in eq.iter().enumerate() {
            dst[dst_offset + i * self.stride] += value;
        }
    }
}

/// The batched opening weight over the stack, one [`INITIAL_BASIS_CHUNK`] at a time.
pub(super) struct StackWeight {
    len: usize,
    lane_block: usize,
    points: Vec<PointWeight>,
    points_by_lane: Vec<Vec<usize>>,
    ring_start: usize,
    rings: Vec<RingWeight>,
}

impl StackWeight {
    pub(super) fn new(
        stack_len: usize,
        lane_block: usize,
        claims: &[StackClaim],
        lambdas: &[F192],
        ring: &RingSwitchOpen,
        rings: Vec<RingWeight>,
    ) -> Self {
        assert_eq!(claims.len(), lambdas.len());
        let points = claims
            .iter()
            .zip(lambdas)
            .map(|(claim, &lambda)| PointClaim::new(claim, lambda))
            .collect();
        Self::from_parts(stack_len, lane_block, points, ring.offset, rings)
    }

    fn from_parts(
        len: usize,
        lane_block: usize,
        claims: Vec<PointClaim>,
        ring_start: usize,
        rings: Vec<RingWeight>,
    ) -> Self {
        let chunk_log = lane_block.min(INITIAL_BASIS_CHUNK).ilog2() as usize;
        let points: Vec<_> = claims
            .into_iter()
            .map(|claim| PointWeight::new(claim, chunk_log))
            .collect();
        let mut points_by_lane = vec![Vec::new(); len.div_ceil(lane_block)];
        for (index, weight) in points.iter().enumerate() {
            for lane in &mut points_by_lane[weight.claim.offset / lane_block..weight.end.div_ceil(lane_block)] {
                lane.push(index);
            }
        }
        Self {
            len,
            lane_block,
            points,
            points_by_lane,
            ring_start,
            rings,
        }
    }

    fn ring_end(&self) -> usize {
        self.rings.first().map_or(self.ring_start, |ring| self.ring_start + (1 << ring.vars()))
    }
}

impl LaneWeight for StackWeight {
    fn len(&self) -> usize {
        self.len
    }

    fn fill(&self, start: usize, dst: &mut [F192]) {
        dst.fill(F192::ZERO);
        let lo = start.max(self.ring_start);
        let hi = (start + dst.len()).min(self.ring_end());
        if lo < hi {
            for ring in &self.rings {
                ring.add_chunk(lo - self.ring_start, &mut dst[lo - start..hi - start]);
            }
        }
        let mut scratch = [MaybeUninit::uninit(); INITIAL_BASIS_CHUNK];
        for &index in &self.points_by_lane[start / self.lane_block] {
            self.points[index].add(start, dst, &mut scratch);
        }
    }

    fn fold_lane(&self, r: F192) -> Box<dyn LaneWeight> {
        let bit = self.lane_block.ilog2() as usize;
        let len = self.len.div_ceil(2 * self.lane_block) * self.lane_block;
        let points = self.points.iter().map(|p| p.claim.fold_bit(bit, r)).collect();
        let ring_vars = self.rings.first().map_or(0, RingWeight::vars);
        let rings = if bit < ring_vars {
            self.rings.iter().map(|ring| ring.fold_coordinate(bit, r)).collect()
        } else {
            let scalar = bound((self.ring_start >> bit) & 1 == 1, r);
            self.rings.iter().map(|ring| ring.scaled(scalar)).collect()
        };
        let ring_start = remove_bit(self.ring_start, bit);
        Box::new(Self::from_parts(len, self.lane_block, points, ring_start, rings))
    }
}

#[cfg(test)]
pub(super) fn build(
    stack: &[F64],
    lane_block: usize,
    claims: &[StackClaim],
    lambdas: &[F192],
    ring: &RingSwitchOpen,
    rs_outputs: &[DeferredRingSwitchOutput],
) -> (Vec<F192>, SumcheckMessage) {
    let rings = rs_outputs.iter().map(|output| output.weight.clone()).collect();
    let weight = StackWeight::new(stack.len(), lane_block, claims, lambdas, ring, rings);
    build_initial_basis(stack, lane_block, |start, dst| weight.fill(start, dst))
}
