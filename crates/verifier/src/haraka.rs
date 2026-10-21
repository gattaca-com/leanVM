//! Flock reduction for the Haraka-512 circuit (`flock::haraka`), then ring switching.
//!
//! The block is rectangular: `2^11` rows of GF(2^8)-coefficient constraints against `2^13` committed bits per
//! call. The zerocheck is the BLAKE2s one; the lincheck binds the ten high column bits and closes on the
//! terminal identity `e_row^T (A + α B + α² C) w_col + α³ w_col[const]`, the three row values coming from one
//! walk of Haraka's linear layers on bit-planes of the column weights.

use crate::field::{E, ZERO};
use crate::flock::{ZerocheckResult, lagrange_weights, phi, verify_flock_zerocheck};
use crate::layout::K_BITS;
use crate::multilinear::eq_kernel;
use crate::transcript::{Transcript, sumcheck};
use crate::{VerifierError, require};

pub const HARAKA_ROWS_LOG: usize = 11;
pub const HARAKA_K_LOG: usize = 13;
pub const HARAKA_ROW_INNER: usize = HARAKA_ROWS_LOG - 6;
pub const HARAKA_LINCHECK_ROUNDS: usize = HARAKA_K_LOG - 6;
const CALLS_PER_SLOT: usize = 1;
const IN_LEN: usize = 64;
const OUT_LEN: usize = 32;
const SBOXES: usize = 640;
const CALL_BITS: usize = 8 * (IN_LEN + OUT_LEN + SBOXES);
const CALL_ROWS: usize = 2 * SBOXES + 8 * OUT_LEN;
const OUT_BASE: usize = 8 * IN_LEN;
const U_BASE: usize = 8 * (IN_LEN + OUT_LEN);
pub const HARAKA_CONSTANT_COLUMN: usize = CALLS_PER_SLOT * CALL_BITS;
const MIX_COLUMNS: [[u8; 4]; 4] = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]];
const MIX: [[(usize, usize); 4]; 4] = [
    [(0, 3), (2, 3), (1, 3), (3, 3)],
    [(2, 0), (0, 0), (3, 0), (1, 0)],
    [(2, 1), (0, 1), (3, 1), (1, 1)],
    [(0, 2), (2, 2), (1, 2), (3, 2)],
];
const RC32: [[u32; 4]; 40] = [
    [0x0684704c, 0xe620c00a, 0xb2c5fef0, 0x75817b9d],
    [0x8b66b4e1, 0x88f3a06b, 0x640f6ba4, 0x2f08f717],
    [0x3402de2d, 0x53f28498, 0xcf029d60, 0x9f029114],
    [0x0ed6eae6, 0x2e7b4f08, 0xbbf3bcaf, 0xfd5b4f79],
    [0xcbcfb0cb, 0x4872448b, 0x79eecd1c, 0xbe397044],
    [0x7eeacdee, 0x6e9032b7, 0x8d5335ed, 0x2b8a057b],
    [0x67c28f43, 0x5e2e7cd0, 0xe2412761, 0xda4fef1b],
    [0x2924d9b0, 0xafcacc07, 0x675ffde2, 0x1fc70b3b],
    [0xab4d63f1, 0xe6867fe9, 0xecdb8fca, 0xb9d465ee],
    [0x1c30bf84, 0xd4b7cd64, 0x5b2a404f, 0xad037e33],
    [0xb2cc0bb9, 0x941723bf, 0x69028b2e, 0x8df69800],
    [0xfa0478a6, 0xde6f5572, 0x4aaa9ec8, 0x5c9d2d8a],
    [0xdfb49f2b, 0x6b772a12, 0x0efa4f2e, 0x29129fd4],
    [0x1ea10344, 0xf449a236, 0x32d611ae, 0xbb6a12ee],
    [0xaf044988, 0x4b050084, 0x5f9600c9, 0x9ca8eca6],
    [0x21025ed8, 0x9d199c4f, 0x78a2c7e3, 0x27e593ec],
    [0xbf3aaaf8, 0xa759c9b7, 0xb9282ecd, 0x82d40173],
    [0x6260700d, 0x6186b017, 0x37f2efd9, 0x10307d6b],
    [0x5aca45c2, 0x21300443, 0x81c29153, 0xf6fc9ac6],
    [0x9223973c, 0x226b68bb, 0x2caf92e8, 0x36d1943a],
    [0xd3bf9238, 0x225886eb, 0x6cbab958, 0xe51071b4],
    [0xdb863ce5, 0xaef0c677, 0x933dfddd, 0x24e1128d],
    [0xbb606268, 0xffeba09c, 0x83e48de3, 0xcb2212b1],
    [0x734bd3dc, 0xe2e4d19c, 0x2db91a4e, 0xc72bf77d],
    [0x43bb47c3, 0x61301b43, 0x4b1415c4, 0x2cb3924e],
    [0xdba775a8, 0xe707eff6, 0x03b231dd, 0x16eb6899],
    [0x6df3614b, 0x3c755977, 0x8e5e2302, 0x7eca472c],
    [0xcda75a17, 0xd6de7d77, 0x6d1be5b9, 0xb88617f9],
    [0xec6b43f0, 0x6ba8e9aa, 0x9d6c069d, 0xa946ee5d],
    [0xcb1e6950, 0xf957332b, 0xa2531159, 0x3bf327c1],
    [0x2cee0c75, 0x00da619c, 0xe4ed0353, 0x600ed0d9],
    [0xf0b1a5a1, 0x96e90cab, 0x80bbbabc, 0x63a4a350],
    [0xae3db102, 0x5e962988, 0xab0dde30, 0x938dca39],
    [0x17bb8f38, 0xd554a40b, 0x8814f3a8, 0x2e75b442],
    [0x34bb8a5b, 0x5f427fd7, 0xaeb6b779, 0x360a16f6],
    [0x26f65241, 0xcbe55438, 0x43ce5918, 0xffbaafde],
    [0x4ce99a54, 0xb9f3026a, 0xa2ca9cf7, 0x839ec978],
    [0xae51a51a, 0x1bdff7be, 0x40c06e28, 0x22901235],
    [0xa0c1613c, 0xba7ed22b, 0xc173bc0f, 0x48a659cf],
    [0x756acc03, 0x02288288, 0x4ad6bdfd, 0xe9c59da1],
];

fn rc_byte(i: usize, b: usize) -> u8 {
    (RC32[i][3 - b / 4] >> (8 * (b % 4))) as u8
}

/// A byte-valued affine form by its eight bit-planes under the column weights.
type Byte = [E; 8];

fn xor(x: &Byte, y: &Byte) -> Byte {
    std::array::from_fn(|j| x[j] + y[j])
}

/// Multiplication by `2` in GF(2^8) on bit-planes: shift, then `0x1b` where bit 7 overflows.
fn xtime(x: &Byte) -> Byte {
    let mut o = [ZERO; 8];
    o[0] = x[7];
    o[1..8].copy_from_slice(&x[..7]);
    for j in [1, 3, 4] {
        o[j] += x[7];
    }
    o
}

fn scale(x: &Byte, c: u8) -> Byte {
    match c {
        1 => *x,
        2 => xtime(x),
        3 => xor(&xtime(x), x),
        _ => unreachable!("MixColumns coefficients are 1, 2 or 3"),
    }
}

fn literal(value: u8, constant: E) -> Byte {
    std::array::from_fn(|j| if value >> j & 1 == 1 { constant } else { ZERO })
}

/// `Σ_j φ(2^j) p_j`: the byte's value.
fn value(p: &Byte) -> E {
    (0..8).fold(ZERO, |acc, j| acc + phi(1 << j) * p[j])
}

fn squared_basis(j: usize) -> E {
    // 2^(2j) in GF(2^8): x^8 = x^4 + x^3 + x + 1.
    let mut v: u16 = 1 << (2 * j);
    for bit in (8..16).rev() {
        if v >> bit & 1 == 1 {
            v ^= 0x11b << (bit - 8);
        }
    }
    phi(v as usize)
}

fn value_sq(p: &Byte) -> E {
    (0..8).fold(ZERO, |acc, j| acc + squared_basis(j) * p[j])
}

/// `A w`, `B w` and `C w` over one slot's rows by one forward walk of the ten calls.
pub fn haraka_row_values(w: &[E]) -> [Vec<E>; 3] {
    let constant = w[HARAKA_CONSTANT_COLUMN];
    let n_rows = CALLS_PER_SLOT * CALL_ROWS;
    let mut a = vec![ZERO; n_rows];
    let mut b = vec![ZERO; n_rows];
    let mut c = vec![ZERO; n_rows];
    let bits = |base: usize| -> Byte { std::array::from_fn(|j| w[base + j]) };
    for call in 0..CALLS_PER_SLOT {
        let zb = call * CALL_BITS;
        let rb = call * CALL_ROWS;
        let mut st: Vec<Byte> = (0..IN_LEN).map(|i| bits(zb + 8 * i)).collect();
        for r in 0..5 {
            for sub in 0..2 {
                for blk in 0..4 {
                    let mut y: Vec<Byte> = Vec::with_capacity(16);
                    for byte in 0..16 {
                        let k = ((r * 2 + sub) * 4 + blk) * 16 + byte;
                        let x = st[blk * 16 + byte];
                        let u = bits(zb + U_BASE + 8 * k);
                        let (xv, uv) = (value(&x), value(&u));
                        a[rb + 2 * k] = value_sq(&x);
                        b[rb + 2 * k] = uv;
                        c[rb + 2 * k] = xv;
                        a[rb + 2 * k + 1] = value_sq(&u);
                        b[rb + 2 * k + 1] = xv;
                        c[rb + 2 * k + 1] = uv;
                        // y = u + rotl(u,1) + rotl(u,2) + rotl(u,3) + rotl(u,4) + 0x63
                        let mut yb = literal(0x63, constant);
                        for j in 0..8 {
                            for s in 0..5 {
                                yb[(j + s) % 8] += u[j];
                            }
                        }
                        y.push(yb);
                    }
                    let sr: Vec<Byte> = (0..16).map(|i| y[4 * ((i / 4 + i % 4) % 4) + i % 4]).collect();
                    let rc = 4 * (2 * r + sub) + blk;
                    for col in 0..4 {
                        for row in 0..4 {
                            let mut e = literal(rc_byte(rc, 4 * col + row), constant);
                            for t in 0..4 {
                                e = xor(&e, &scale(&sr[4 * col + t], MIX_COLUMNS[row][t]));
                            }
                            st[blk * 16 + 4 * col + row] = e;
                        }
                    }
                }
            }
            let old = st.clone();
            for (dst, row) in MIX.iter().enumerate() {
                for (d, &(src_blk, src_d)) in row.iter().enumerate() {
                    for t in 0..4 {
                        st[dst * 16 + 4 * d + t] = old[src_blk * 16 + 4 * src_d + t];
                    }
                }
            }
        }
        for i in 0..OUT_LEN {
            let (blk, byte) = match i / 8 {
                0 => (0, 8 + i % 8),
                1 => (1, 8 + i % 8),
                2 => (2, i % 8),
                _ => (3, i % 8),
            };
            let o = xor(&st[blk * 16 + byte], &bits(zb + 8 * (16 * blk + byte)));
            for j in 0..8 {
                let row = rb + 2 * SBOXES + 8 * i + j;
                a[row] = o[j];
                b[row] = constant;
                c[row] = w[zb + OUT_BASE + 8 * i + j];
            }
        }
    }
    [a, b, c]
}

/// What the reduction leaves, as [`crate::flock::FlockClaims`] but at the rectangular block's point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HarakaClaims {
    pub z_skip: E,
    pub chi: Vec<E>,
    pub v_a: E,
    pub v_b: E,
    pub v_c: E,
    pub alpha: E,
    pub round_challenges: Vec<E>,
    pub s: Vec<E>,
}

impl HarakaClaims {
    /// The slices' point: the lincheck's ten column challenges in coordinate order, then the slot coordinates.
    pub(crate) fn point(&self) -> Vec<E> {
        let cols = self.round_challenges.iter().rev().copied();
        let outer = self.chi[HARAKA_ROW_INNER..].iter().copied();
        cols.chain(outer).collect()
    }
}

fn verify_lincheck(zc: ZerocheckResult, transcript: &mut Transcript) -> Result<HarakaClaims, VerifierError> {
    let alpha = transcript.sample();
    let claim = zc.v_a + alpha * zc.v_b + alpha.pow(2) * zc.v_c + alpha.pow(3);
    let (round_challenges, r_lc) = sumcheck(transcript, claim, 3, &[None; HARAKA_LINCHECK_ROUNDS])?;
    let s = transcript.next_scalars(K_BITS)?;
    if transcript.checked() {
        let skip_weights = lagrange_weights(K_BITS, zc.z_skip);
        let row_high = eq_kernel(&zc.chi[..HARAKA_ROW_INNER]);
        let chi_cols: Vec<E> = round_challenges.iter().rev().copied().collect();
        let col_high = eq_kernel(&chi_cols);
        let mut w_col = vec![ZERO; col_high.len() * K_BITS];
        for (h, &weight) in col_high.iter().enumerate() {
            for (l, &slice) in s.iter().enumerate() {
                w_col[h * K_BITS + l] = weight * slice;
            }
        }
        let [ra, rb, rc] = haraka_row_values(&w_col);
        let mut terminal = ZERO;
        for (row, ((&va, &vb), &vc)) in ra.iter().zip(&rb).zip(&rc).enumerate() {
            let e_row = row_high[row / K_BITS] * skip_weights[row % K_BITS];
            terminal += e_row * (va + alpha * vb + alpha.pow(2) * vc);
        }
        terminal += alpha.pow(3) * w_col[HARAKA_CONSTANT_COLUMN];
        require(terminal == r_lc, VerifierError::FlockLincheckTerminal)?;
    }
    Ok(HarakaClaims {
        z_skip: zc.z_skip,
        chi: zc.chi,
        v_a: zc.v_a,
        v_b: zc.v_b,
        v_c: zc.v_c,
        alpha,
        round_challenges,
        s,
    })
}

/// The reduction: `log_rows = 14 + n_log` is the zerocheck's dimension.
pub fn verify_flock_haraka(log_rows: usize, transcript: &mut Transcript) -> Result<HarakaClaims, VerifierError> {
    require(log_rows >= HARAKA_ROWS_LOG + 3, VerifierError::CommittedSize)?;
    let zc = verify_flock_zerocheck(log_rows, transcript)?;
    verify_lincheck(zc, transcript)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squared_basis_matches_phi_of_squares() {
        for j in 0..8usize {
            let mut v: u16 = 1 << (2 * j);
            for bit in (8..16).rev() {
                if v >> bit & 1 == 1 {
                    v ^= 0x11b << (bit - 8);
                }
            }
            assert_eq!(squared_basis(j), phi(v as usize));
        }
        assert_eq!(value(&[phi(1); 8]) + value(&[phi(1); 8]), ZERO);
    }
}
