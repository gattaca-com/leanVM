//! Multilinear and stacking helpers.

use crate::field::{E, GEN, ONE, Scaled, ZERO};
use crate::{VerifierError, require};

/// `eq(i, point)` for every index `i`, LSB-first, at one product per entry: `v * (1 + r) = v + v * r`.
pub fn eq_kernel(point: &[E]) -> Vec<E> {
    let mut out = vec![ZERO; 1 << point.len()];
    out[0] = ONE;
    for (k, &r) in point.iter().enumerate() {
        let (low, high) = out[..2 << k].split_at_mut(1 << k);
        let scale = Scaler::new(r, low.len());
        for (v, h) in low.iter_mut().zip(high) {
            *h = scale.mul(*v);
            *v += *h;
        }
    }
    out
}

/// Multiplication by `r`, through [`Scaled`] once it multiplies enough values to repay the table.
enum Scaler {
    Table(Scaled),
    Plain(E),
}

impl Scaler {
    fn new(r: E, uses: usize) -> Self {
        if uses >= 256 {
            Self::Table(Scaled::new(r))
        } else {
            Self::Plain(r)
        }
    }

    #[inline]
    fn mul(&self, value: E) -> E {
        match self {
            Self::Table(table) => table.mul(value),
            Self::Plain(r) => value * *r,
        }
    }
}

pub fn multilinear_eval<T: Copy + Into<E>>(mle: &[T], point: &[E]) -> Result<E, VerifierError> {
    require(
        point.len() < 64 && mle.len() == 1 << point.len(),
        VerifierError::MultilinearSize,
    )?;
    let mut cur: Vec<E> = mle.iter().map(|&value| value.into()).collect();
    for &r in point {
        let scale = Scaler::new(r, cur.len() / 2);
        for i in 0..cur.len() / 2 {
            cur[i] = cur[2 * i] + scale.mul(cur[2 * i] + cur[2 * i + 1]);
        }
        cur.truncate(cur.len() / 2);
    }
    Ok(cur[0])
}

pub const fn log2_ceil(value: u64) -> usize {
    (64 - value.saturating_sub(1).leading_zeros()) as usize
}

pub fn log2_strict(value: u64) -> Result<usize, VerifierError> {
    require(value.is_power_of_two(), VerifierError::NotPowerOfTwo)?;
    Ok(value.trailing_zeros() as usize)
}

pub fn eq_eval(left: &[E], right: &[E]) -> E {
    assert_eq!(left.len(), right.len());
    let mut result = ONE;
    for (&x, &y) in left.iter().zip(right) {
        result *= ONE + x + y;
    }
    result
}

pub fn dot(left: &[E], right: &[E]) -> E {
    assert_eq!(left.len(), right.len());
    let mut result = ZERO;
    for (&x, &y) in left.iter().zip(right) {
        result += x * y;
    }
    result
}

/// `pad_n(point)`, the multilinear extension of `[z >= n]`, `n <= 2^len(point)` (lemma lem:suffix).
pub fn padding_weight(point: &[E], n: u64) -> E {
    let bit = |k: usize| E::new(n >> k & 1, 0, 0);
    let low = point
        .iter()
        .enumerate()
        .fold(ONE, |pad, (k, &r)| (ONE + bit(k)) * r + (ONE + bit(k) + r) * pad);
    (ONE + bit(point.len())) * low
}

/// MLE of `[1, g, g^2, ...]` at an LSB-first point.
pub fn index_mle(point: &[E]) -> E {
    let mut result = ONE;
    let mut generator_power = GEN;
    for &challenge in point {
        result *= ONE + challenge * (ONE + generator_power);
        generator_power = generator_power.square();
    }
    result
}

/// A polynomial at `point`, by Horner over its coefficients, constant first.
pub fn poly_eval(coefficients: &[E], point: E) -> E {
    coefficients.iter().rev().fold(ZERO, |acc, &c| acc * point + c)
}

/// Where something sits in the stacked cube: a claim's point fills the low `variables` coordinates, the bits of `index` fixing the rest.
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    pub variables: usize,
    pub index: u64,
}

impl Placement {
    pub fn stack_point(&self, point: &[E], stack_log: usize) -> Vec<E> {
        let above = selector_point(self.index >> self.variables, stack_log - self.variables);
        [point, &above].concat()
    }

    /// eq weight of the coordinates above the window.
    pub fn eq_above(&self, point: &[E]) -> E {
        let bits = selector_point(self.index >> self.variables, point.len() - self.variables);
        eq_eval(&bits, &point[self.variables..])
    }
}

/// Rows of a jagged chunk: a column's run inside one chunk of its group.
pub const CHUNK_LOG: usize = 10;

/// `eq(z, b)` for a field point and a bit.
fn eq_bit(z: E, bit: u64) -> E {
    if bit == 1 { z } else { ONE + z }
}

/// The jagged indicator `G(z, y; a, n) = Σ_{B < n} eq(z, B)·eq(y, a + B)`, both points zero-extended.
///
/// A read-once branching program over the bits, low first. Its state is the carry of `a + B` and whether
/// `B < n` on the bits read so far. A sum `a + B` past `y`'s cube meets a zero coordinate and weighs nothing.
pub fn jagged_indicator(z: &[E], y: &[E], a: u64, n: u64) -> E {
    let bits = z.len().max(y.len()).max((u64::BITS - (a | n).leading_zeros()) as usize);
    let at = |v: &[E], k: usize| v.get(k).copied().unwrap_or(ZERO);
    let bit = |v: u64, k: usize| v.checked_shr(k as u32).unwrap_or(0) & 1;
    // state[carry][below]
    let mut state = [[ONE, ZERO], [ZERO, ZERO]];
    for k in 0..bits {
        let (zk, yk, ak, nk) = (at(z, k), at(y, k), bit(a, k), bit(n, k));
        let mut next = [[ZERO; 2]; 2];
        for (carry, row) in state.iter().enumerate() {
            for (below, &mass) in row.iter().enumerate() {
                for b in 0..2 {
                    let sum = ak + b + carry as u64;
                    let below = if b == nk { below } else { usize::from(b < nk) };
                    next[(sum >> 1) as usize][below] += mass * eq_bit(zk, b) * eq_bit(yk, sum & 1);
                }
            }
        }
        state = next;
    }
    state[0][1]
}

/// Where a column of a jagged group sits: row `i` at `offset + (i >> CHUNK_LOG)·2^(CHUNK_LOG + log_width) + column·2^CHUNK_LOG + (i mod 2^CHUNK_LOG)`.
#[derive(Clone, Copy, Debug)]
pub struct JaggedSlot {
    pub offset: u64,
    pub log_width: usize,
    pub column: usize,
    /// The group's chunks, `⌈rows / 2^CHUNK_LOG⌉`: every row below `chunks << CHUNK_LOG` is committed.
    pub chunks: u64,
}

impl JaggedSlot {
    /// The stack weight of a claim at `point` on this column, at the stack point `x`.
    pub fn weight(&self, point: &[E], x: &[E]) -> E {
        let at = |v: &[E], k: usize| v.get(k).copied().unwrap_or(ZERO);
        let low = (0..CHUNK_LOG).fold(ONE, |acc, k| acc * (ONE + at(point, k) + at(x, k)));
        let column = (0..self.log_width).fold(ONE, |acc, k| {
            acc * eq_bit(at(x, CHUNK_LOG + k), (self.column >> k) as u64 & 1)
        });
        let high = x.get(CHUNK_LOG + self.log_width..).unwrap_or(&[]);
        let base = self.offset >> (CHUNK_LOG + self.log_width);
        low * column * jagged_indicator(point.get(CHUNK_LOG..).unwrap_or(&[]), high, base, self.chunks)
    }
}

pub fn stack_offsets(sizes: &[usize]) -> (Vec<u64>, usize) {
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by_key(|&index| (std::cmp::Reverse(sizes[index]), index));
    let mut offsets = vec![0; sizes.len()];
    let mut total = 0;
    for index in order {
        offsets[index] = total;
        total += 1 << sizes[index];
    }
    (offsets, log2_ceil(total))
}

fn selector_point(selector: u64, length: usize) -> Vec<E> {
    (0..length).map(|bit| E::new(selector >> bit & 1, 0, 0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The branching program has to equal the sum it stands for, the weight of every committed row written out:
    /// across unaligned bases, counts that are and are not powers of two, a point shorter or longer than the
    /// cube, and sums that run past the cube.
    #[test]
    fn jagged_indicator_is_the_written_out_sum() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut random = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut points = |n: usize| -> Vec<E> { (0..n).map(|_| E::new(random(), random(), random())).collect() };
        let eq_index = |point: &[E], index: u64| -> E {
            if index >> point.len() != 0 {
                return ZERO;
            }
            (0..point.len()).fold(ONE, |acc, k| acc * eq_bit(point[k], index >> k & 1))
        };
        for (z_len, y_len, a, n) in [
            (3, 5, 0, 8),
            (3, 5, 5, 7),
            (4, 5, 13, 11),
            (2, 6, 33, 3),
            (5, 4, 3, 13),
            (4, 4, 9, 9),
            (0, 3, 6, 1),
            (3, 3, 0, 0),
        ] {
            let z = points(z_len);
            let y = points(y_len);
            let sum = E::sum((0..n).map(|b| eq_index(&z, b) * eq_index(&y, a + b)));
            assert_eq!(
                jagged_indicator(&z, &y, a, n),
                sum,
                "z {z_len}, y {y_len}, a {a}, n {n}"
            );
        }
    }
}
