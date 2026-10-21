//! The Haraka-512 R1CS with GF(2^8) coefficients.
//!
//! Every S-box input `x` is an affine form of the committed bits with
//! coefficients in GF(2^8) (bit `j` of a byte is the field element `2^j`), so
//! the inverse `u = x^-1` costs two degree-2 rows and eight committed bits:
//!
//! ```text
//!   x² · u = x        (x ≠ 0 ⇒ u = 1/x)
//!   u² · x = u        (x = 0 ⇒ u = 0)
//! ```
//!
//! Squaring is additive, so `x²` is again an affine form (coefficients squared).
//! ShiftRows, MixColumns, the S-box's affine part and the round constants are
//! all GF(2)-linear and stay symbolic. The truncated output is materialized by
//! 256 linear rows (`out_bit · 1 = z[out]`) so an embedding can open it.
//!
//! ## Per-call layout (`CALL_BITS` z bits, `CALL_ROWS` rows)
//!
//! ```text
//!   z[0     ..   512)   input, 64 bytes (byte b bit j at 8b + j)
//!   z[512   ..   768)   output, 32 bytes
//!   z[768   .. 5,888)   u of each S-box, 640 bytes
//!   rows [0, 1,280)     S-box k: row 2k is x²·u = x, row 2k+1 is u²·x = u
//!   rows [1,280, 1,536) output bit i: out_i · 1 = z[512 + i]
//! ```
//!
//! A slot holds [`CALLS_PER_SLOT`] calls then the constant wire at [`CONST_COL`].

use super::primitive::*;
use primitives::field::F8;

pub const CALLS_PER_SLOT: usize = 10;
pub const K_LOG: usize = 16;
pub const ROWS_LOG: usize = 14;
pub const K: usize = 1 << K_LOG;
pub const ROWS_PER_SLOT: usize = 1 << ROWS_LOG;
pub const IN_BITS: usize = 8 * IN_LEN;
pub const OUT_BITS: usize = 8 * OUT_LEN;
pub const IN_BASE: usize = 0;
pub const OUT_BASE: usize = IN_BITS;
pub const U_BASE: usize = IN_BITS + OUT_BITS;
pub const CALL_BITS: usize = U_BASE + 8 * SBOX_PER_CALL;
pub const SBOX_ROWS: usize = 2 * SBOX_PER_CALL;
pub const CALL_ROWS: usize = SBOX_ROWS + OUT_BITS;
pub const CONST_COL: usize = CALLS_PER_SLOT * CALL_BITS;
pub const USEFUL_BITS: usize = CONST_COL + 1;
pub const USED_ROWS: usize = CALLS_PER_SLOT * CALL_ROWS;
/// Rows per zerocheck column (the univariate-skip block).
pub const COL_ROWS: usize = 64;
pub const COLS_PER_CALL: usize = CALL_ROWS / COL_ROWS;
pub const COLS_PER_SLOT: usize = ROWS_PER_SLOT / COL_ROWS;
pub const USED_COLS: usize = USED_ROWS / COL_ROWS;
/// Columns of a call whose rows are the linear output rows (`b = 1` there).
pub const LIN_COL_START: usize = SBOX_ROWS / COL_ROWS;

const _: () = assert!(USED_ROWS <= ROWS_PER_SLOT && USEFUL_BITS <= K);
const _: () =
    assert!(CALL_BITS.is_multiple_of(128) && CALL_ROWS.is_multiple_of(COL_ROWS) && SBOX_ROWS.is_multiple_of(COL_ROWS));

/// Whether zerocheck column `col` of a slot holds linear output rows.
pub const fn is_lin_col(col: usize) -> bool {
    col < USED_COLS && col % COLS_PER_CALL >= LIN_COL_START
}

/// An affine form `Σ coef · z[col] + cst` with GF(2^8) coefficients.
#[derive(Clone, Default, Debug)]
pub struct Expr {
    pub terms: Vec<(u32, u8)>,
    pub cst: u8,
}

fn mul(a: u8, b: u8) -> u8 {
    (F8(a) * F8(b)).0
}

impl Expr {
    fn bits(col0: usize) -> Self {
        Expr {
            terms: (0..8).map(|i| ((col0 + i) as u32, 1u8 << i)).collect(),
            cst: 0,
        }
    }
    fn scaled_add(&mut self, o: &Expr, c: u8) {
        for &(col, g) in &o.terms {
            self.terms.push((col, mul(g, c)));
        }
        self.cst ^= mul(o.cst, c);
    }
    fn normalize(&mut self) {
        self.terms.sort_by_key(|t| t.0);
        let mut out: Vec<(u32, u8)> = Vec::with_capacity(self.terms.len());
        for &(col, g) in &self.terms {
            match out.last_mut() {
                Some(last) if last.0 == col => last.1 ^= g,
                _ => out.push((col, g)),
            }
        }
        out.retain(|t| t.1 != 0);
        self.terms = out;
    }
    pub fn square(&self) -> Expr {
        Expr {
            terms: self.terms.iter().map(|&(c, g)| (c, mul(g, g))).collect(),
            cst: mul(self.cst, self.cst),
        }
    }
    /// Bit `j` of the byte-valued form, itself an affine form (coefficient `1` on each bit that feeds it).
    fn bit(&self, j: u32) -> Expr {
        Expr {
            terms: self
                .terms
                .iter()
                .filter(|&&(_, g)| g >> j & 1 == 1)
                .map(|&(c, _)| (c, 1u8))
                .collect(),
            cst: self.cst >> j & 1,
        }
    }
    pub fn eval(&self, z: impl Fn(usize) -> bool) -> u8 {
        self.terms
            .iter()
            .fold(self.cst, |v, &(c, g)| if z(c as usize) { v ^ g } else { v })
    }
}

pub struct Row {
    pub a: Expr,
    pub b: Expr,
    pub c: Expr,
}

/// The rows of one call, with z columns relative to the call's base.
pub fn build_call() -> Vec<Row> {
    let mut rows = Vec::with_capacity(CALL_ROWS);
    let mut st: Vec<Expr> = (0..IN_LEN).map(|b| Expr::bits(IN_BASE + 8 * b)).collect();
    for r in 0..ROUNDS {
        for j in 0..AES_PER_ROUND {
            for blk in 0..STATES {
                let mut y: Vec<Expr> = Vec::with_capacity(16);
                for b in 0..16 {
                    let k = sbox_index(r, j, blk, b);
                    let x = st[blk * 16 + b].clone();
                    let u = Expr::bits(U_BASE + 8 * k);
                    rows.push(Row {
                        a: x.square(),
                        b: u.clone(),
                        c: x.clone(),
                    });
                    rows.push(Row {
                        a: u.square(),
                        b: x,
                        c: u.clone(),
                    });
                    let mut yb = Expr {
                        terms: Vec::with_capacity(8),
                        cst: 0x63,
                    };
                    for (i, &(col, _)) in u.terms.iter().enumerate() {
                        yb.terms.push((col, affine_column(i as u32)));
                    }
                    y.push(yb);
                }
                let sr: Vec<Expr> = (0..16).map(|i| y[4 * ((i / 4 + i % 4) % 4) + i % 4].clone()).collect();
                let rc = rc_bytes(rc_index(r, j, blk));
                for col in 0..4 {
                    for row in 0..4 {
                        let mut e = Expr::default();
                        for t in 0..4 {
                            e.scaled_add(&sr[4 * col + t], MIX_COLUMNS[row][t]);
                        }
                        e.cst ^= rc[4 * col + row];
                        e.normalize();
                        st[blk * 16 + 4 * col + row] = e;
                    }
                }
            }
        }
        let old = st.clone();
        for (dst, row) in MIX.iter().enumerate() {
            for (d, &(src_blk, src_d)) in row.iter().enumerate() {
                for t in 0..4 {
                    st[dst * 16 + 4 * d + t] = old[src_blk * 16 + 4 * src_d + t].clone();
                }
            }
        }
    }
    let one = Expr {
        terms: Vec::new(),
        cst: 1,
    };
    for i in 0..OUT_LEN {
        let (blk, b) = out_source(i);
        let mut o = st[blk * 16 + b].clone();
        o.scaled_add(&Expr::bits(IN_BASE + 8 * (16 * blk + b)), 1);
        o.normalize();
        for j in 0..8 {
            rows.push(Row {
                a: o.bit(j),
                b: one.clone(),
                c: Expr {
                    terms: vec![((OUT_BASE + 8 * i + j as usize) as u32, 1)],
                    cst: 0,
                },
            });
        }
    }
    debug_assert_eq!(rows.len(), CALL_ROWS);
    rows
}

/// One slot's three matrices, row-major sparse, with constants on [`CONST_COL`].
pub struct SlotMatrices {
    pub a: Vec<Vec<(u32, u8)>>,
    pub b: Vec<Vec<(u32, u8)>>,
    pub c: Vec<Vec<(u32, u8)>>,
}

impl SlotMatrices {
    pub fn build() -> Self {
        let call = build_call();
        let place = |e: &Expr, base: usize| -> Vec<(u32, u8)> {
            let mut v: Vec<(u32, u8)> = e.terms.iter().map(|&(c, g)| ((c as usize + base) as u32, g)).collect();
            if e.cst != 0 {
                v.push((CONST_COL as u32, e.cst));
            }
            v
        };
        let mut m = SlotMatrices {
            a: Vec::with_capacity(USED_ROWS),
            b: Vec::with_capacity(USED_ROWS),
            c: Vec::with_capacity(USED_ROWS),
        };
        for call_idx in 0..CALLS_PER_SLOT {
            let base = call_idx * CALL_BITS;
            for r in &call {
                m.a.push(place(&r.a, base));
                m.b.push(place(&r.b, base));
                m.c.push(place(&r.c, base));
            }
        }
        m
    }
}
