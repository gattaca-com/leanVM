//! A field element as a binary polynomial in a challenge `s`, and back.
//!
//! In the power basis of `s`, a sum of bits weighted by powers of `s` is the bits themselves, and
//! multiplying it by a weight is a carry-less product. So the prover moves its weights into that
//! basis once, accumulates unreduced polynomials in `s`, and evaluates each total at `s` at the end.

use primitives::field::gf2_64::mul_wide;
use primitives::field::{F192, powers};

fn bits(a: F192) -> [u64; 3] {
    [a.c0, a.c1, a.c2]
}

fn xor3(a: &mut [u64; 3], b: &[u64; 3]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x ^= y;
    }
}

/// Transpose a 192 by 192 bit matrix, a row in three words: bit `c` of row `r` becomes bit `r`
/// of row `c`. Nine 64 by 64 blocks.
fn transpose192(a: &[[u64; 3]]) -> Vec<[u64; 3]> {
    assert_eq!(a.len(), 192);
    let mut out = vec![[0u64; 3]; 192];
    for i in 0..3 {
        for j in 0..3 {
            let mut block: [u64; 64] = std::array::from_fn(|r| a[64 * i + r][j]);
            crate::witness::transpose64(&mut block);
            for (c, &word) in block.iter().enumerate() {
                out[64 * j + c][i] = word;
            }
        }
    }
    out
}

/// The map from a field element to its coefficients on `1, s, ..., s^191`.
pub struct PowerBasis {
    tables: Vec<[[u64; 3]; 256]>,
}

impl PowerBasis {
    /// `None` when `s` lies in a proper subfield, where its powers do not span the field.
    pub fn new(s: F192) -> Option<Self> {
        // Row `r` holds bit `r` of every power, beside the identity.
        let mut power = F192::ONE;
        let powers: Vec<[u64; 3]> = (0..192)
            .map(|_| {
                let p = bits(power);
                power *= s;
                p
            })
            .collect();
        let mut rows: Vec<([u64; 3], [u64; 3])> = transpose192(&powers).into_iter().map(|r| (r, [0; 3])).collect();
        for (r, row) in rows.iter_mut().enumerate() {
            row.1[r / 64] = 1 << (r % 64);
        }
        for b in 0..192 {
            let pivot = (b..192).find(|&r| rows[r].0[b / 64] >> (b % 64) & 1 == 1)?;
            rows.swap(b, pivot);
            let (pa, pi) = rows[b];
            // Without a branch: whether a row has the bit is a coin toss.
            for (r, row) in rows.iter_mut().enumerate() {
                let mask = 0u64.wrapping_sub(row.0[b / 64] >> (b % 64) & 1 & u64::from(r != b));
                for k in 0..3 {
                    row.0[k] ^= pa[k] & mask;
                    row.1[k] ^= pi[k] & mask;
                }
            }
        }
        // Row `b` of the inverse is coefficient `b` as a function of the element's bits.
        let columns = transpose192(&rows.iter().map(|row| row.1).collect::<Vec<_>>());
        let tables = (0..24)
            .map(|k| {
                let mut t = [[0u64; 3]; 256];
                for byte in 1..256usize {
                    let low = byte.trailing_zeros() as usize;
                    t[byte] = t[byte & (byte - 1)];
                    xor3(&mut t[byte], &columns[8 * k + low]);
                }
                t
            })
            .collect();
        Some(Self { tables })
    }

    pub fn to_poly(&self, a: F192) -> [u64; 3] {
        let a = bits(a);
        let mut p = [0u64; 3];
        for (k, table) in self.tables.iter().enumerate() {
            xor3(&mut p, &table[(a[k / 8] >> (8 * (k % 8))) as usize & 0xff]);
        }
        p
    }
}

/// The sum of every subset of eight weights, indexed by the byte whose bits pick it.
pub fn byte_sums(weights: &[F192; 8]) -> [F192; 256] {
    let mut t = [F192::ZERO; 256];
    for (bit, &w) in weights.iter().enumerate() {
        for byte in 1usize << bit..2 << bit {
            t[byte] = t[byte - (1 << bit)] + w;
        }
    }
    t
}

/// Evaluation at `s` of binary polynomials of at most `8 * n_bytes` coefficients.
pub struct PolyEval {
    tables: Vec<[F192; 256]>,
}

impl PolyEval {
    pub fn new(s: F192, n_bytes: usize) -> Self {
        Self {
            tables: powers(s, 8 * n_bytes)
                .as_chunks::<8>()
                .0
                .iter()
                .map(byte_sums)
                .collect(),
        }
    }

    #[inline]
    pub fn eval(&self, words: &[u64]) -> F192 {
        let mut acc = F192::ZERO;
        for (k, table) in self.tables.iter().enumerate() {
            acc += table[(words[k / 8] >> (8 * (k % 8))) as usize & 0xff];
        }
        acc
    }
}

/// The polynomial whose word `i` starts `acc[i]`.
#[inline(always)]
#[cfg(any(test, not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni"))))]
pub fn carry<const N: usize, const M: usize>(acc: &[u128; N]) -> [u64; M] {
    let mut out = [0u64; M];
    for (i, &x) in acc.iter().enumerate() {
        out[i] ^= x as u64;
        out[i + 1] ^= (x >> 64) as u64;
    }
    out
}

/// `acc += a * b`, on binary polynomials.
#[inline(always)]
pub fn add_mul<const A: usize, const B: usize, const N: usize>(acc: &mut [u64; N], a: &[u64; A], b: &[u64; B]) {
    for (i, &a) in a.iter().enumerate() {
        for (k, &b) in b.iter().enumerate() {
            let p = mul_wide(a, b);
            acc[i + k] ^= p as u64;
            acc[i + k + 1] ^= (p >> 64) as u64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use primitives::test_rng::Rng;

    #[test]
    fn power_basis_round_trips_and_products_are_carryless() {
        let mut rng = Rng::new(11);
        let s = rng.ext();
        let basis = PowerBasis::new(s).expect("a generic element");
        let eval = PolyEval::new(s, 40);
        for _ in 0..50 {
            let (a, b) = (rng.ext(), rng.next_u64());
            let pa = basis.to_poly(a);
            assert_eq!(eval.eval(&[pa[0], pa[1], pa[2], 0, 0]), a);
            let acc: [u128; 3] = std::array::from_fn(|k| mul_wide(pa[k], b));
            let product: [u64; 4] = carry(&acc);
            let pb = eval.eval(&[b, 0, 0, 0, 0]);
            assert_eq!(eval.eval(&[product[0], product[1], product[2], product[3], 0]), a * pb);
        }
        assert!(PowerBasis::new(F192::ONE).is_none());
    }
}
