//! The byte-valued kernels: F192 in GF(2^8) coordinates, the 64-point S→Λ
//! extension, the round-1 accumulation and the fold at `z`. Each has a portable
//! form; x86 with AVX-512 + GFNI gets a vector form doing the same arithmetic.
//!
//! `F192` is a 24-dimensional GF(2^8)-space with basis `x^i y^j` (`i < 8`,
//! `j < 3`), so a product `φ8(v) · w` is the bytewise product of `v` with the
//! coordinates of `w`: a round's eq weight times a GF(2^8) row value is one
//! GFNI multiply per coordinate.

use super::circuit::{COL_ROWS, is_lin_col};
use pcs::ntt::AdditiveNttGf8;
use primitives::field::{F8, F192, phi8_192};

pub const GFNI: bool = cfg!(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
));

pub const COORDS: usize = 24;

// ---------------------------------------------------------------- basis

/// The two 192×192 GF(2) maps between the polynomial basis and GF(2^8)
/// coordinates, as 24×24 blocks of 8×8 in the GFNI qword encoding
/// (`blocks[i][j]` maps input byte `i` to output byte `j`).
pub struct Basis {
    pub to_coords: Vec<[u64; COORDS]>,
    pub from_coords: Vec<[u64; COORDS]>,
}

fn bits_of(w: F192) -> [u64; 3] {
    [w.c0, w.c1, w.c2]
}

fn bit(v: &[u64; 3], i: usize) -> bool {
    v[i / 64] >> (i % 64) & 1 == 1
}

fn matrix_blocks(rows: &[[u64; 3]; 192]) -> Vec<[u64; COORDS]> {
    (0..COORDS)
        .map(|i| {
            std::array::from_fn(|j| {
                let mut q = 0u64;
                for t in 0..8 {
                    let row = rows[8 * j + t];
                    let slice = (row[i / 8] >> (8 * (i % 8))) & 0xff;
                    q |= slice << (8 * (7 - t));
                }
                q
            })
        })
        .collect()
}

impl Basis {
    pub fn new() -> Self {
        let x = F192::new(2, 0, 0);
        let mut cols = [[0u64; 3]; 192];
        for j in 0..3 {
            let mut e = F192::ONE;
            for _ in 0..j {
                e *= F192::Y;
            }
            for i in 0..8 {
                for t in 0..8 {
                    cols[8 * (i + 8 * j) + t] = bits_of(phi8_192(F8(1 << t)) * e);
                }
                e *= x;
            }
        }
        let mut rows = [[0u64; 3]; 192];
        for (c, col) in cols.iter().enumerate() {
            for r in 0..192 {
                if bit(col, r) {
                    rows[r][c / 64] |= 1u64 << (c % 64);
                }
            }
        }
        let mut a = rows;
        let mut inv = [[0u64; 3]; 192];
        for (i, row) in inv.iter_mut().enumerate() {
            row[i / 64] = 1u64 << (i % 64);
        }
        for c in 0..192 {
            let p = (c..192).find(|&r| bit(&a[r], c)).expect("basis matrix is invertible");
            a.swap(c, p);
            inv.swap(c, p);
            for r in 0..192 {
                if r != c && bit(&a[r], c) {
                    for w in 0..3 {
                        a[r][w] ^= a[c][w];
                        inv[r][w] ^= inv[c][w];
                    }
                }
            }
        }
        Basis {
            to_coords: matrix_blocks(&inv),
            from_coords: matrix_blocks(&rows),
        }
    }

    pub fn to_coords(&self, w: F192) -> [u8; COORDS] {
        let b = bits_of(w);
        let mut inp = [0u8; COORDS];
        for (i, v) in inp.iter_mut().enumerate() {
            *v = (b[i / 8] >> (8 * (i % 8))) as u8;
        }
        apply_scalar(&self.to_coords, &inp)
    }

    pub fn from_coords(&self, v: &[u8; COORDS]) -> F192 {
        let o = apply_scalar(&self.from_coords, v);
        let word = |q: usize| u64::from_le_bytes(o[8 * q..8 * q + 8].try_into().unwrap());
        F192::new(word(0), word(1), word(2))
    }
}

impl Default for Basis {
    fn default() -> Self {
        Self::new()
    }
}

fn apply_scalar(blocks: &[[u64; COORDS]], inp: &[u8; COORDS]) -> [u8; COORDS] {
    let mut out = [0u8; COORDS];
    for i in 0..COORDS {
        for j in 0..COORDS {
            let q = blocks[i][j];
            let mut b = 0u8;
            for t in 0..8 {
                let row = (q >> (8 * (7 - t))) as u8;
                b |= (((row & inp[i]).count_ones() & 1) as u8) << t;
            }
            out[j] ^= b;
        }
    }
    out
}

/// Transpose an 8×8 byte matrix held as 8 rows of `u64`.
#[inline]
pub fn transpose8x8(m: &mut [u64; 8]) {
    for i in (0..8).step_by(2) {
        let (a, b) = (m[i], m[i + 1]);
        let t = ((a >> 8) ^ b) & 0x00FF00FF00FF00FF;
        m[i] = a ^ (t << 8);
        m[i + 1] = b ^ t;
    }
    for i in [0, 1, 4, 5] {
        let (a, b) = (m[i], m[i + 2]);
        let t = ((a >> 16) ^ b) & 0x0000FFFF0000FFFF;
        m[i] = a ^ (t << 16);
        m[i + 2] = b ^ t;
    }
    for i in 0..4 {
        let (a, b) = (m[i], m[i + 4]);
        let t = ((a >> 32) ^ b) & 0x00000000FFFFFFFF;
        m[i] = a ^ (t << 32);
        m[i + 4] = b ^ t;
    }
}

// ---------------------------------------------------------------- S → Λ extension

/// The 64×64 GF(2^8) matrix of the extension (inverse NTT on S, forward on Λ): `m[s]` is column `s`.
pub fn extension_matrix() -> Vec<[u8; 64]> {
    let ntt_s = AdditiveNttGf8::new(6, F8::ZERO);
    let ntt_l = AdditiveNttGf8::new(6, F8(64));
    (0..64)
        .map(|s| {
            let mut v = vec![F8::ZERO; 64];
            v[s] = F8::ONE;
            ntt_s.inverse(&mut v);
            ntt_l.forward(&mut v);
            std::array::from_fn(|i| v[i].0)
        })
        .collect()
}

pub fn extend_scalar(m: &[[u8; 64]], col: &[u8; 64]) -> [u8; 64] {
    let mut out = [0u8; 64];
    for s in 0..64 {
        if col[s] != 0 {
            for (o, &e) in out.iter_mut().zip(&m[s]) {
                *o ^= (F8(e) * F8(col[s])).0;
            }
        }
    }
    out
}

pub fn square_byte(v: u8) -> u8 {
    (F8(v) * F8(v)).0
}

/// A zerocheck column's three row vectors from its stored `c` bytes:
/// `a = c²`, `b = c` of the paired row, or `1` on the linear output rows.
pub fn abc_column(c: &[u8; 64], lin: bool) -> ([u8; 64], [u8; 64]) {
    let a = std::array::from_fn(|s| square_byte(c[s]));
    let b = std::array::from_fn(|s| if lin { 1 } else { c[s ^ 1] });
    (a, b)
}

// ---------------------------------------------------------------- round 1, portable

/// Portable round 1 over `slots`: `P^AB(λ) = Σ_x eq(x) φ8(â b̂(λ, x))` and
/// `D[s] = Σ_x eq(x) φ8(c(s, x))`, bucketed by byte value.
pub fn round1_scalar(
    c: &[u8],
    eq: &[F192],
    slots: std::ops::Range<usize>,
    used_cols: usize,
    cols_per_slot: usize,
    m: &[[u8; 64]],
) -> (Vec<F192>, Vec<F192>) {
    let mut acc_ab = vec![F192::ZERO; 64 * 256];
    let mut acc_c = vec![F192::ZERO; 64 * 256];
    for slot in slots {
        for col in 0..used_cols {
            let x = slot * cols_per_slot + col;
            let cv: &[u8; 64] = c[x * COL_ROWS..x * COL_ROWS + COL_ROWS].try_into().unwrap();
            let (a, b) = abc_column(cv, is_lin_col(col));
            let al = extend_scalar(m, &a);
            let bl = extend_scalar(m, &b);
            let e = eq[x];
            for lam in 0..64 {
                acc_ab[lam * 256 + (F8(al[lam]) * F8(bl[lam])).0 as usize] += e;
                acc_c[lam * 256 + cv[lam] as usize] += e;
            }
        }
    }
    let fin = |acc: &[F192]| -> Vec<F192> {
        (0..64)
            .map(|lam| (1..256).fold(F192::ZERO, |s, v| s + phi8_192(F8(v as u8)) * acc[lam * 256 + v]))
            .collect()
    };
    (fin(&acc_ab), fin(&acc_c))
}

/// Portable fold at `z`: `out_a[x] = Σ_s w_s φ8(a(s,x))`, `out_b[x] = Σ_s w_s φ8(b(s,x))`.
pub fn fold_scalar(
    c: &[u8],
    w: &[F192],
    slot: usize,
    used_cols: usize,
    cols_per_slot: usize,
    out_a: &mut [F192],
    out_b: &mut [F192],
) {
    let mut ta = vec![F192::ZERO; 64 * 256];
    let mut tb = vec![F192::ZERO; 64 * 256];
    for s in 0..64 {
        for v in 0..256 {
            ta[s * 256 + v] = w[s] * phi8_192(F8(square_byte(v as u8)));
            tb[s * 256 + v] = w[s] * phi8_192(F8(v as u8));
        }
    }
    for col in 0..cols_per_slot {
        if col >= used_cols {
            out_a[col] = F192::ZERO;
            out_b[col] = F192::ZERO;
            continue;
        }
        let x = slot * cols_per_slot + col;
        let cv: &[u8; 64] = c[x * COL_ROWS..x * COL_ROWS + COL_ROWS].try_into().unwrap();
        let (_, b) = abc_column(cv, is_lin_col(col));
        let mut acc_a = F192::ZERO;
        let mut acc_b = F192::ZERO;
        for s in 0..64 {
            acc_a += ta[s * 256 + cv[s] as usize];
            acc_b += tb[s * 256 + b[s] as usize];
        }
        out_a[col] = acc_a;
        out_b[col] = acc_b;
    }
}

// ---------------------------------------------------------------- x86 GFNI

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
pub mod gfni {
    use super::*;
    use std::arch::x86_64::*;

    pub struct Ntt64 {
        perm: [__m512i; 6],
        lo: [u64; 6],
        tw_s: [__m512i; 6],
        tw_l: [__m512i; 6],
    }

    impl Ntt64 {
        pub fn new() -> Self {
            let ntt_s = AdditiveNttGf8::new(6, F8::ZERO);
            let ntt_l = AdditiveNttGf8::new(6, F8(64));
            let mut perm = [[0u8; 64]; 6];
            let mut lo = [0u64; 6];
            let mut tw_s = [[0u8; 64]; 6];
            let mut tw_l = [[0u8; 64]; 6];
            for d in 0..6 {
                let b = 64 >> d;
                let h = b / 2;
                for p in 0..64 {
                    perm[d][p] = (p ^ h) as u8;
                    if p & h == 0 {
                        lo[d] |= 1u64 << p;
                        tw_s[d][p] = ntt_s.twiddles()[(1 << d) - 1 + p / b].0;
                        tw_l[d][p] = ntt_l.twiddles()[(1 << d) - 1 + p / b].0;
                    }
                }
            }
            // SAFETY: the target features are enabled at compile time.
            unsafe {
                let ld = |a: &[u8; 64]| _mm512_loadu_si512(a.as_ptr() as *const _);
                Ntt64 {
                    perm: std::array::from_fn(|d| ld(&perm[d])),
                    lo,
                    tw_s: std::array::from_fn(|d| ld(&tw_s[d])),
                    tw_l: std::array::from_fn(|d| ld(&tw_l[d])),
                }
            }
        }

        /// Two columns extended together: twelve butterfly layers each, interleaved.
        ///
        /// # Safety
        ///
        /// The AVX-512 and GFNI target features must be enabled, as they are when this module compiles.
        #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
        #[inline]
        pub unsafe fn extend2(&self, mut v: __m512i, mut w: __m512i) -> (__m512i, __m512i) {
            for d in (0..6).rev() {
                v = _mm512_xor_si512(v, _mm512_maskz_permutexvar_epi8(!self.lo[d], self.perm[d], v));
                w = _mm512_xor_si512(w, _mm512_maskz_permutexvar_epi8(!self.lo[d], self.perm[d], w));
                v = _mm512_xor_si512(
                    v,
                    _mm512_gf2p8mul_epi8(self.tw_s[d], _mm512_permutexvar_epi8(self.perm[d], v)),
                );
                w = _mm512_xor_si512(
                    w,
                    _mm512_gf2p8mul_epi8(self.tw_s[d], _mm512_permutexvar_epi8(self.perm[d], w)),
                );
            }
            for d in 0..6 {
                v = _mm512_xor_si512(
                    v,
                    _mm512_gf2p8mul_epi8(self.tw_l[d], _mm512_permutexvar_epi8(self.perm[d], v)),
                );
                w = _mm512_xor_si512(
                    w,
                    _mm512_gf2p8mul_epi8(self.tw_l[d], _mm512_permutexvar_epi8(self.perm[d], w)),
                );
                v = _mm512_xor_si512(v, _mm512_maskz_permutexvar_epi8(!self.lo[d], self.perm[d], v));
                w = _mm512_xor_si512(w, _mm512_maskz_permutexvar_epi8(!self.lo[d], self.perm[d], w));
            }
            (v, w)
        }
    }

    impl Default for Ntt64 {
        fn default() -> Self {
            Self::new()
        }
    }

    pub fn frobenius_matrix() -> i64 {
        let mut m = 0u64;
        for i in 0..8u32 {
            let sq = square_byte(1 << i);
            for o in 0..8u32 {
                if sq >> o & 1 == 1 {
                    m |= 1u64 << (8 * (7 - o) + i);
                }
            }
        }
        m as i64
    }

    pub struct Consts {
        pub ntt: Ntt64,
        pub basis: Basis,
        pub frob: __m512i,
        pub swap: __m512i,
        pub ones: __m512i,
        /// Stage `s` of the 64×64 byte transpose swaps row bit `s` with column bit `s`: the two
        /// two-source byte permutes of a vector pair.
        pub tidx: [[__m512i; 2]; 6],
    }

    impl Consts {
        pub fn new() -> Self {
            // SAFETY: the target features are enabled at compile time.
            unsafe {
                let swap: [u8; 64] = std::array::from_fn(|i| (i ^ 1) as u8);
                let tidx: [[__m512i; 2]; 6] = std::array::from_fn(|st| {
                    let bit = 1usize << st;
                    let lo: [u8; 64] = std::array::from_fn(|x| if x & bit == 0 { x } else { 64 + (x ^ bit) } as u8);
                    let hi: [u8; 64] = std::array::from_fn(|x| if x & bit == 0 { x | bit } else { 64 + x } as u8);
                    [
                        _mm512_loadu_si512(lo.as_ptr() as *const _),
                        _mm512_loadu_si512(hi.as_ptr() as *const _),
                    ]
                });
                Consts {
                    ntt: Ntt64::new(),
                    basis: Basis::new(),
                    frob: _mm512_set1_epi64(frobenius_matrix()),
                    swap: _mm512_loadu_si512(swap.as_ptr() as *const _),
                    ones: _mm512_set1_epi8(1),
                    tidx,
                }
            }
        }
    }

    impl Default for Consts {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Apply a 192×192 block matrix to 8 elements given coordinate-major (`t[i]` = byte `i` of the 8 elements);
    /// returns them element-major, 24 bytes each.
    ///
    /// # Safety
    ///
    /// The AVX-512 and GFNI target features must be enabled, as they are when this module compiles.
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
    #[inline]
    pub unsafe fn apply8_from_qwords(blocks: &[[u64; COORDS]], t: &[u64; COORDS], out: &mut [u8; 8 * COORDS]) {
        unsafe {
            let mut o = [_mm512_setzero_si512(); 3];
            for i in 0..COORDS {
                let q = _mm512_set1_epi64(t[i] as i64);
                for (k, acc) in o.iter_mut().enumerate() {
                    let m = _mm512_loadu_si512(blocks[i].as_ptr().add(8 * k) as *const _);
                    *acc = _mm512_xor_si512(*acc, _mm512_gf2p8affine_epi64_epi8::<0>(q, m));
                }
            }
            for (k, acc) in o.iter().enumerate() {
                let mut rows = [0u64; 8];
                _mm512_storeu_si512(rows.as_mut_ptr() as *mut _, *acc);
                transpose8x8(&mut rows);
                for (e, r) in rows.iter().enumerate() {
                    out[COORDS * e + 8 * k..COORDS * e + 8 * k + 8].copy_from_slice(&r.to_le_bytes());
                }
            }
        }
    }

    /// [`apply8_from_qwords`] writing the 8 elements straight to `out` (8 consecutive `F192`s).
    ///
    /// # Safety
    ///
    /// `out` must be valid for 8 `F192` writes, with the target features enabled.
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
    #[inline]
    pub unsafe fn apply8_into(blocks: &[[u64; COORDS]], t: &[u64; COORDS], out: *mut F192) {
        unsafe {
            let mut o = [_mm512_setzero_si512(); 3];
            for i in 0..COORDS {
                let q = _mm512_set1_epi64(t[i] as i64);
                for (k, acc) in o.iter_mut().enumerate() {
                    let m = _mm512_loadu_si512(blocks[i].as_ptr().add(8 * k) as *const _);
                    *acc = _mm512_xor_si512(*acc, _mm512_gf2p8affine_epi64_epi8::<0>(q, m));
                }
            }
            let out = out as *mut u64;
            for (k, acc) in o.iter().enumerate() {
                let mut rows = [0u64; 8];
                _mm512_storeu_si512(rows.as_mut_ptr() as *mut _, *acc);
                transpose8x8(&mut rows);
                for (e, r) in rows.iter().enumerate() {
                    out.add(3 * e + k).write_unaligned(*r);
                }
            }
        }
    }

    /// Convert 8 element-major `F192`s (24 bytes each) with a block matrix.
    ///
    /// # Safety
    ///
    /// The AVX-512 and GFNI target features must be enabled, as they are when this module compiles.
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
    #[inline]
    pub unsafe fn convert8(blocks: &[[u64; COORDS]], inp: &[u8; 8 * COORDS], out: &mut [u8; 8 * COORDS]) {
        unsafe {
            let mut t = [0u64; COORDS];
            for k in 0..3 {
                let mut rows: [u64; 8] = std::array::from_fn(|e| {
                    u64::from_le_bytes(inp[COORDS * e + 8 * k..COORDS * e + 8 * k + 8].try_into().unwrap())
                });
                transpose8x8(&mut rows);
                t[8 * k..8 * k + 8].copy_from_slice(&rows);
            }
            apply8_from_qwords(blocks, &t, out)
        }
    }

    /// Coordinate-major accumulators: `acc[i]` byte `λ` is coordinate `i` of index `λ`.
    pub struct Acc24(pub [__m512i; COORDS]);

    impl Acc24 {
        pub fn zero() -> Self {
            // SAFETY: the target features are enabled at compile time.
            Acc24([unsafe { _mm512_setzero_si512() }; COORDS])
        }
        pub fn finish(&self, b: &Basis) -> Vec<F192> {
            let mut raw = [[0u8; 64]; COORDS];
            // SAFETY: the target features are enabled at compile time.
            unsafe {
                for i in 0..COORDS {
                    _mm512_storeu_si512(raw[i].as_mut_ptr() as *mut _, self.0[i]);
                }
            }
            (0..64)
                .map(|lam| {
                    let c: [u8; COORDS] = std::array::from_fn(|i| raw[i][lam]);
                    b.from_coords(&c)
                })
                .collect()
        }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
    #[inline]
    unsafe fn accumulate(acc: &mut Acc24, v: __m512i, eq8: *const u8) {
        unsafe {
            for i in 0..COORDS {
                acc.0[i] = _mm512_xor_si512(acc.0[i], _mm512_gf2p8mul_epi8(v, _mm512_set1_epi8(*eq8.add(i) as i8)));
            }
        }
    }

    /// Round 1 over `n` columns of `c` (64 bytes each) whose eq weights are `eq8` (24 coordinate bytes each).
    /// `lin(col)` says whether a column holds linear output rows.
    ///
    /// # Safety
    ///
    /// `c` must point at `64 * n` readable bytes and `eq8` at `24 * n`, with the target features enabled.
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
    pub unsafe fn round1_tile(
        k: &Consts,
        c: *const u8,
        eq8: *const u8,
        n: usize,
        lin: impl Fn(usize) -> bool,
        acc_ab: &mut Acc24,
        acc_c: &mut Acc24,
    ) {
        unsafe {
            let bvec = |cv: __m512i, x: usize| {
                if lin(x) {
                    k.ones
                } else {
                    _mm512_permutexvar_epi8(k.swap, cv)
                }
            };
            let mut x = 0;
            while x + 2 <= n {
                let cv0 = _mm512_loadu_si512(c.add(64 * x) as *const _);
                let cv1 = _mm512_loadu_si512(c.add(64 * x + 64) as *const _);
                let a0 = _mm512_gf2p8affine_epi64_epi8::<0>(cv0, k.frob);
                let a1 = _mm512_gf2p8affine_epi64_epi8::<0>(cv1, k.frob);
                let (al0, bl0) = k.ntt.extend2(a0, bvec(cv0, x));
                let (al1, bl1) = k.ntt.extend2(a1, bvec(cv1, x + 1));
                accumulate(acc_ab, _mm512_gf2p8mul_epi8(al0, bl0), eq8.add(COORDS * x));
                accumulate(acc_ab, _mm512_gf2p8mul_epi8(al1, bl1), eq8.add(COORDS * (x + 1)));
                accumulate(acc_c, cv0, eq8.add(COORDS * x));
                accumulate(acc_c, cv1, eq8.add(COORDS * (x + 1)));
                x += 2;
            }
            if x < n {
                let cv = _mm512_loadu_si512(c.add(64 * x) as *const _);
                let a = _mm512_gf2p8affine_epi64_epi8::<0>(cv, k.frob);
                let (al, bl) = k.ntt.extend2(a, bvec(cv, x));
                accumulate(acc_ab, _mm512_gf2p8mul_epi8(al, bl), eq8.add(COORDS * x));
                accumulate(acc_c, cv, eq8.add(COORDS * x));
            }
        }
    }

    /// Transpose a 64×64 byte tile in place (`t[r]` = row `r`), via 8×8 blocks.
    pub fn transpose64(t: &mut [[u8; 64]; 64]) {
        let mut out = [[0u8; 64]; 64];
        for rb in (0..64).step_by(8) {
            for cb in (0..64).step_by(8) {
                let mut m: [u64; 8] =
                    std::array::from_fn(|i| u64::from_le_bytes(t[rb + i][cb..cb + 8].try_into().unwrap()));
                transpose8x8(&mut m);
                for (i, row) in m.iter().enumerate() {
                    out[cb + i][rb..rb + 8].copy_from_slice(&row.to_le_bytes());
                }
            }
        }
        *t = out;
    }

    /// Transpose 64 rows of 64 bytes in registers: six stages, each exchanging one row-index bit with the
    /// same column-index bit through a pair of two-source byte permutes.
    ///
    /// # Safety
    ///
    /// The AVX-512 and GFNI target features must be enabled, as they are when this module compiles.
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
    pub unsafe fn transpose64_simd(k: &Consts, v: &mut [__m512i; 64]) {
        for st in 0..6 {
            let bit = 1usize << st;
            let [lo, hi] = k.tidx[st];
            for i0 in (0..64).filter(|i| i & bit == 0) {
                let i1 = i0 | bit;
                let (a, b) = (v[i0], v[i1]);
                v[i0] = _mm512_permutex2var_epi8(a, lo, b);
                v[i1] = _mm512_permutex2var_epi8(a, hi, b);
            }
        }
    }

    /// Fold a tile of 64 columns at `z`: weights `wa[s][i]`, `wb[s][i]` are the coordinates of the
    /// `a` and `b` weights (`wb[s] = w[s ^ 1]`), `wsum` the coordinates of `Σ_s w_s` for the linear columns.
    /// Writes 64 `F192`s to each of `out_a`, `out_b`, `out_c` (`out_c[x] = Σ_s w_s φ8(c(s, x))`).
    ///
    /// # Safety
    ///
    /// `c` must point at 4,096 readable bytes and `out_a`, `out_b`, `out_c` at 64 writable `F192`s each, with
    /// the target features enabled.
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
    pub unsafe fn fold_tile(
        k: &Consts,
        c: *const u8,
        lin_mask: u64,
        wa: &[[u8; COORDS]; 64],
        wb: &[[u8; COORDS]; 64],
        wsum: &[u8; COORDS],
        out_a: *mut F192,
        out_b: *mut F192,
        out_c: *mut F192,
    ) {
        unsafe {
            let mut rows: [__m512i; 64] = std::array::from_fn(|s| _mm512_loadu_si512(c.add(64 * s) as *const _));
            transpose64_simd(k, &mut rows);
            let a_rows: [__m512i; 64] = std::array::from_fn(|s| _mm512_gf2p8affine_epi64_epi8::<0>(rows[s], k.frob));
            let mut raw = [[[0u64; 8]; COORDS]; 3];
            for (t, raw_t) in raw.iter_mut().enumerate() {
                let (src, w): (&[__m512i; 64], &[[u8; COORDS]; 64]) = match t {
                    0 => (&a_rows, wa),
                    1 => (&rows, wb),
                    _ => (&rows, wa),
                };
                for g in 0..COORDS / 8 {
                    let mut acc = [_mm512_setzero_si512(); 8];
                    for s in 0..64 {
                        let v = src[s];
                        let ws = &w[s];
                        for (j, a) in acc.iter_mut().enumerate() {
                            *a = _mm512_xor_si512(*a, _mm512_gf2p8mul_epi8(v, _mm512_set1_epi8(ws[8 * g + j] as i8)));
                        }
                    }
                    for (j, a) in acc.iter().enumerate() {
                        let i = 8 * g + j;
                        let a = if t == 1 {
                            _mm512_mask_mov_epi8(*a, lin_mask, _mm512_set1_epi8(wsum[i] as i8))
                        } else {
                            *a
                        };
                        _mm512_storeu_si512(raw_t[i].as_mut_ptr() as *mut _, a);
                    }
                }
            }
            for (t, out) in [out_a, out_b, out_c].into_iter().enumerate() {
                for g in 0..8 {
                    let q: [u64; COORDS] = std::array::from_fn(|i| raw[t][i][g]);
                    apply8_into(&k.basis.from_coords, &q, out.add(8 * g));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use primitives::test_rng::Rng;

    #[test]
    fn basis_roundtrip_and_products() {
        let b = Basis::new();
        let mut rng = Rng::new(7);
        for _ in 0..32 {
            let w = F192::new(rng.next_u64(), rng.next_u64(), rng.next_u64());
            let v = F8((rng.next_u64() & 0xff) as u8);
            let c = b.to_coords(w);
            assert_eq!(b.from_coords(&c), w);
            let cv: [u8; COORDS] = std::array::from_fn(|i| (F8(c[i]) * v).0);
            assert_eq!(b.from_coords(&cv), w * phi8_192(v));
        }
    }

    #[test]
    fn transpose8x8_is_a_transpose() {
        let mut m: [u64; 8] = std::array::from_fn(|r| u64::from_le_bytes(std::array::from_fn(|c| (8 * r + c) as u8)));
        transpose8x8(&mut m);
        for (k, row) in m.iter().enumerate() {
            let bytes = row.to_le_bytes();
            for (e, &v) in bytes.iter().enumerate() {
                assert_eq!(v as usize, 8 * e + k);
            }
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni"
    ))]
    #[test]
    fn gfni_matches_scalar() {
        use std::arch::x86_64::*;
        let k = gfni::Consts::new();
        let m = extension_matrix();
        let mut rng = Rng::new(11);
        for _ in 0..8 {
            let col: [u8; 64] = std::array::from_fn(|_| (rng.next_u64() & 0xff) as u8);
            let want = extend_scalar(&m, &col);
            let mut got = [0u8; 64];
            // SAFETY: the target features are enabled at compile time.
            unsafe {
                let v = _mm512_loadu_si512(col.as_ptr() as *const _);
                let (r, _) = k.ntt.extend2(v, v);
                _mm512_storeu_si512(got.as_mut_ptr() as *mut _, r);
            }
            assert_eq!(got, want);
        }
        let ws: Vec<F192> = (0..8)
            .map(|_| F192::new(rng.next_u64(), rng.next_u64(), rng.next_u64()))
            .collect();
        let mut inp = [0u8; 8 * COORDS];
        for (e, w) in ws.iter().enumerate() {
            inp[COORDS * e..COORDS * e + 8].copy_from_slice(&w.c0.to_le_bytes());
            inp[COORDS * e + 8..COORDS * e + 16].copy_from_slice(&w.c1.to_le_bytes());
            inp[COORDS * e + 16..COORDS * e + 24].copy_from_slice(&w.c2.to_le_bytes());
        }
        let mut out = [0u8; 8 * COORDS];
        // SAFETY: the target features are enabled at compile time.
        unsafe { gfni::convert8(&k.basis.to_coords, &inp, &mut out) };
        for (e, w) in ws.iter().enumerate() {
            assert_eq!(&out[COORDS * e..COORDS * e + COORDS], &k.basis.to_coords(*w));
        }
    }
}
