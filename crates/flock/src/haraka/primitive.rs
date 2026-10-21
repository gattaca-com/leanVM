//! Haraka-512 v2 (Kölbl, Lauridsen, Mendel, Rechberger): ten AES rounds on four
//! 16-byte states with a dword mix after every second round, feed-forward, and a
//! 32-byte truncation. The prover needs every S-box input `x` and its field
//! inverse `u` (`y = Aff(u) + 0x63` is the S-box output), which the AES-NI path
//! reads off the state with GFNI and the portable path computes bytewise.

pub const ROUNDS: usize = 5;
pub const AES_PER_ROUND: usize = 2;
pub const STATES: usize = 4;
pub const AES_ROUNDS: usize = ROUNDS * AES_PER_ROUND;
pub const SBOX_PER_CALL: usize = AES_ROUNDS * STATES * 16;
pub const IN_LEN: usize = 64;
pub const OUT_LEN: usize = 32;

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

/// The 16 bytes of round constant `i`, in state byte order (`_mm_set_epi32` is big-endian in dwords).
pub fn rc_bytes(i: usize) -> [u8; 16] {
    let e = RC32[i];
    std::array::from_fn(|b| (e[3 - b / 4] >> (8 * (b % 4))) as u8)
}

/// Round constant index of AES round `sub` of mix round `round` on state `blk`.
pub const fn rc_index(round: usize, sub: usize, blk: usize) -> usize {
    STATES * (AES_PER_ROUND * round + sub) + blk
}

/// Position of an S-box among the call's `SBOX_PER_CALL`.
pub const fn sbox_index(round: usize, sub: usize, blk: usize, byte: usize) -> usize {
    ((round * AES_PER_ROUND + sub) * STATES + blk) * 16 + byte
}

/// `MIX[dst][d] = (src_state, src_dword)` of the dword mix between rounds.
pub const MIX: [[(usize, usize); 4]; 4] = [
    [(0, 3), (2, 3), (1, 3), (3, 3)],
    [(2, 0), (0, 0), (3, 0), (1, 0)],
    [(2, 1), (0, 1), (3, 1), (1, 1)],
    [(0, 2), (2, 2), (1, 2), (3, 2)],
];

/// Byte `i` of the truncated output comes from `(state, byte)` of the fed-forward state.
pub const fn out_source(i: usize) -> (usize, usize) {
    match i / 8 {
        0 => (0, 8 + i % 8),
        1 => (1, 8 + i % 8),
        2 => (2, i % 8),
        _ => (3, i % 8),
    }
}

/// AES S-box: `aff(x^-1)`.
pub fn sbox(x: u8) -> u8 {
    let u = gf_inv(x);
    affine(u)
}

/// The S-box's affine part on `u`.
pub fn affine(u: u8) -> u8 {
    u ^ u.rotate_left(1) ^ u.rotate_left(2) ^ u.rotate_left(3) ^ u.rotate_left(4) ^ 0x63
}

/// Column `j` of the affine map as a byte: which output bits input bit `j` feeds.
pub const fn affine_column(j: u32) -> u8 {
    0x1fu8.rotate_left(j)
}

pub fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut p = 0u8;
    for _ in 0..8 {
        if b & 1 == 1 {
            p ^= a;
        }
        let carry = a & 0x80;
        a <<= 1;
        if carry != 0 {
            a ^= 0x1b;
        }
        b >>= 1;
    }
    p
}

pub fn gf_inv(x: u8) -> u8 {
    if x == 0 {
        return 0;
    }
    let mut r = x;
    for _ in 0..6 {
        r = gf_mul(gf_mul(r, r), x);
    }
    gf_mul(r, r)
}

/// ShiftRows on the 16 bytes of one state (column-major layout: byte `4c + r`).
pub fn shift_rows<T: Copy>(s: &[T; 16]) -> [T; 16] {
    std::array::from_fn(|i| s[4 * ((i / 4 + i % 4) % 4) + i % 4])
}

/// MixColumns coefficients: output row `r` of a column is `Σ_t MIX_COLUMNS[r][t] · input[t]`.
pub const MIX_COLUMNS: [[u8; 4]; 4] = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]];

fn mix_columns(s: &[u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    for c in 0..4 {
        for r in 0..4 {
            o[4 * c + r] = (0..4).fold(0, |acc, t| acc ^ gf_mul(MIX_COLUMNS[r][t], s[4 * c + t]));
        }
    }
    o
}

/// Portable Haraka-512 recording each S-box input `x` and inverse `u`.
pub fn haraka512_witness_scalar(
    msg: &[u8; IN_LEN],
    xs: &mut [u8; SBOX_PER_CALL],
    us: &mut [u8; SBOX_PER_CALL],
) -> [u8; OUT_LEN] {
    let mut st: [[u8; 16]; STATES] = std::array::from_fn(|b| msg[16 * b..16 * b + 16].try_into().unwrap());
    for r in 0..ROUNDS {
        for j in 0..AES_PER_ROUND {
            for blk in 0..STATES {
                let mut y = [0u8; 16];
                for b in 0..16 {
                    let k = sbox_index(r, j, blk, b);
                    let x = st[blk][b];
                    let u = gf_inv(x);
                    xs[k] = x;
                    us[k] = u;
                    y[b] = affine(u);
                }
                let mut nst = mix_columns(&shift_rows(&y));
                let rc = rc_bytes(rc_index(r, j, blk));
                for b in 0..16 {
                    nst[b] ^= rc[b];
                }
                st[blk] = nst;
            }
        }
        let old = st;
        for (dst, row) in MIX.iter().enumerate() {
            for (d, &(src_blk, src_d)) in row.iter().enumerate() {
                st[dst][4 * d..4 * d + 4].copy_from_slice(&old[src_blk][4 * src_d..4 * src_d + 4]);
            }
        }
    }
    let mut out = [0u8; OUT_LEN];
    for (i, o) in out.iter_mut().enumerate() {
        let (blk, b) = out_source(i);
        *o = st[blk][b] ^ msg[16 * blk + b];
    }
    out
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "aes",
    target_feature = "gfni",
    target_feature = "sse2"
))]
pub mod aesni {
    use super::*;
    use std::arch::x86_64::*;

    const GFNI_ID: i64 = 0x0102040810204080;

    #[inline]
    unsafe fn ld(p: &[u8]) -> __m128i {
        unsafe { _mm_loadu_si128(p.as_ptr() as *const __m128i) }
    }
    #[inline]
    unsafe fn st(p: &mut [u8], v: __m128i) {
        unsafe { _mm_storeu_si128(p.as_mut_ptr() as *mut __m128i, v) }
    }

    /// Same contract as [`haraka512_witness_scalar`], on AES-NI with GFNI for the inverses.
    pub fn haraka512_witness(
        msg: &[u8; IN_LEN],
        xs: &mut [u8; SBOX_PER_CALL],
        us: &mut [u8; SBOX_PER_CALL],
    ) -> [u8; OUT_LEN] {
        // SAFETY: the target features are enabled at compile time.
        unsafe {
            let id = _mm_set1_epi64x(GFNI_ID);
            let mut s: [__m128i; STATES] = std::array::from_fn(|b| ld(&msg[16 * b..16 * b + 16]));
            for r in 0..ROUNDS {
                for j in 0..AES_PER_ROUND {
                    for blk in 0..STATES {
                        let k = sbox_index(r, j, blk, 0);
                        st(&mut xs[k..k + 16], s[blk]);
                        st(&mut us[k..k + 16], _mm_gf2p8affineinv_epi64_epi8::<0>(s[blk], id));
                        s[blk] = _mm_aesenc_si128(s[blk], ld(&rc_bytes(rc_index(r, j, blk))));
                    }
                }
                let tmp = _mm_unpacklo_epi32(s[0], s[1]);
                s[0] = _mm_unpackhi_epi32(s[0], s[1]);
                s[1] = _mm_unpacklo_epi32(s[2], s[3]);
                s[2] = _mm_unpackhi_epi32(s[2], s[3]);
                s[3] = _mm_unpacklo_epi32(s[0], s[2]);
                s[0] = _mm_unpackhi_epi32(s[0], s[2]);
                s[2] = _mm_unpackhi_epi32(s[1], tmp);
                s[1] = _mm_unpacklo_epi32(s[1], tmp);
            }
            let mut full = [0u8; IN_LEN];
            for b in 0..STATES {
                st(
                    &mut full[16 * b..16 * b + 16],
                    _mm_xor_si128(s[b], ld(&msg[16 * b..16 * b + 16])),
                );
            }
            let mut out = [0u8; OUT_LEN];
            for (i, o) in out.iter_mut().enumerate() {
                let (blk, b) = out_source(i);
                *o = full[16 * blk + b];
            }
            out
        }
    }
}

/// The fastest available witness-recording Haraka-512.
pub fn haraka512_witness(
    msg: &[u8; IN_LEN],
    xs: &mut [u8; SBOX_PER_CALL],
    us: &mut [u8; SBOX_PER_CALL],
) -> [u8; OUT_LEN] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "aes",
        target_feature = "gfni",
        target_feature = "sse2"
    ))]
    {
        aesni::haraka512_witness(msg, xs, us)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "aes",
        target_feature = "gfni",
        target_feature = "sse2"
    )))]
    {
        haraka512_witness_scalar(msg, xs, us)
    }
}

pub fn haraka512(msg: &[u8; IN_LEN]) -> [u8; OUT_LEN] {
    let mut xs = [0u8; SBOX_PER_CALL];
    let mut us = [0u8; SBOX_PER_CALL];
    haraka512_witness(msg, &mut xs, &mut us)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector() {
        let msg: [u8; 64] = std::array::from_fn(|i| i as u8);
        let hex: String = haraka512(&msg).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "be7f723b4e80a99813b292287f306f625a6d57331cae5f34dd9277b0945be2aa");
    }

    #[test]
    fn scalar_matches_fast_path() {
        let mut msg = [0u8; 64];
        for (i, m) in msg.iter_mut().enumerate() {
            *m = (i as u8).wrapping_mul(37).wrapping_add(11);
        }
        let (mut xs, mut us) = ([0u8; SBOX_PER_CALL], [0u8; SBOX_PER_CALL]);
        let (mut xs2, mut us2) = ([0u8; SBOX_PER_CALL], [0u8; SBOX_PER_CALL]);
        let a = haraka512_witness_scalar(&msg, &mut xs, &mut us);
        let b = haraka512_witness(&msg, &mut xs2, &mut us2);
        assert_eq!(a, b);
        assert_eq!(xs, xs2);
        assert_eq!(us, us2);
        for x in 0..=255u8 {
            assert_eq!(gf_mul(x, gf_inv(x)), if x == 0 { 0 } else { 1 });
            let u = gf_inv(x);
            let aff = (0..8).fold(0u8, |acc, j| if u >> j & 1 == 1 { acc ^ affine_column(j) } else { acc }) ^ 0x63;
            assert_eq!(aff, sbox(x));
        }
    }
}
