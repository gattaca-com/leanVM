//! Haraka-512 and digests.

use crate::field::E;
use crate::{VerifierError, require};

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

/// Byte `b` of round constant `i`, in state byte order.
pub(crate) fn rc_byte(i: usize, b: usize) -> u8 {
    (RC32[i][3 - b / 4] >> (8 * (b % 4))) as u8
}

/// `MIX[dst][d] = (src_state, src_dword)` of the dword mix between rounds.
pub(crate) const MIX: [[(usize, usize); 4]; 4] = [
    [(0, 3), (2, 3), (1, 3), (3, 3)],
    [(2, 0), (0, 0), (3, 0), (1, 0)],
    [(2, 1), (0, 1), (3, 1), (1, 1)],
    [(0, 2), (2, 2), (1, 2), (3, 2)],
];

pub(crate) const MIX_COLUMNS: [[u8; 4]; 4] = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]];

/// Byte `i` of the truncated output comes from `(state, byte)` of the fed-forward state.
pub(crate) const fn out_source(i: usize) -> (usize, usize) {
    match i / 8 {
        0 => (0, 8 + i % 8),
        1 => (1, 8 + i % 8),
        2 => (2, i % 8),
        _ => (3, i % 8),
    }
}

const fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut p = 0u8;
    let mut i = 0;
    while i < 8 {
        if b & 1 == 1 {
            p ^= a;
        }
        let carry = a & 0x80;
        a <<= 1;
        if carry != 0 {
            a ^= 0x1b;
        }
        b >>= 1;
        i += 1;
    }
    p
}

const fn sbox_of(x: u8) -> u8 {
    let mut r = x;
    if x != 0 {
        let mut i = 0;
        while i < 6 {
            r = gf_mul(gf_mul(r, r), x);
            i += 1;
        }
        r = gf_mul(r, r);
    }
    r ^ r.rotate_left(1) ^ r.rotate_left(2) ^ r.rotate_left(3) ^ r.rotate_left(4) ^ 0x63
}

/// The AES S-box, `aff(x^-1)`, tabulated at compile time.
const SBOX: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut x = 0;
    while x < 256 {
        table[x] = sbox_of(x as u8);
        x += 1;
    }
    table
};

/// `2 · x` in GF(2^8), tabulated.
const XTIME: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut x = 0;
    while x < 256 {
        table[x] = gf_mul(x as u8, 2);
        x += 1;
    }
    table
};

/// One AES round on a column-major state: SubBytes, ShiftRows, MixColumns, then the round constant.
fn aes_round(s: &[u8; 16], rc: usize) -> [u8; 16] {
    let y: [u8; 16] = std::array::from_fn(|i| SBOX[s[4 * ((i / 4 + i % 4) % 4) + i % 4] as usize]);
    let scale = |c: u8, v: u8| match c {
        1 => v,
        2 => XTIME[v as usize],
        _ => XTIME[v as usize] ^ v,
    };
    std::array::from_fn(|i| {
        let (c, r) = (i / 4, i % 4);
        (0..4).fold(rc_byte(rc, i), |acc, t| acc ^ scale(MIX_COLUMNS[r][t], y[4 * c + t]))
    })
}

/// Haraka-512 v2 of 64 bytes: the VM's hash opcode.
pub fn haraka512(msg: &[u8; 64]) -> Digest {
    let mut st: [[u8; 16]; 4] = std::array::from_fn(|b| msg[16 * b..16 * b + 16].try_into().unwrap());
    for round in 0..5 {
        for sub in 0..2 {
            for blk in 0..4 {
                st[blk] = aes_round(&st[blk], 4 * (2 * round + sub) + blk);
            }
        }
        let old = st;
        for (dst, row) in MIX.iter().enumerate() {
            for (d, &(src_blk, src_d)) in row.iter().enumerate() {
                st[dst][4 * d..4 * d + 4].copy_from_slice(&old[src_blk][4 * src_d..4 * src_d + 4]);
            }
        }
    }
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        let (blk, b) = out_source(i);
        *o = st[blk][b] ^ msg[16 * blk + b];
    }
    Digest(out)
}

/// The chain over 32-byte blocks from the zero chaining value, closed by a `1` byte, zeros, and the
/// length as eight little-endian bytes ending a block.
pub fn hash(data: &[u8]) -> Digest {
    let mut cv = [0u8; 32];
    let mut step = |block: &[u8; 32]| {
        let mut input = [0u8; 64];
        input[..32].copy_from_slice(&cv);
        input[32..].copy_from_slice(block);
        cv = haraka512(&input).0;
    };
    let (blocks, tail) = data.as_chunks::<32>();
    for block in blocks {
        step(block);
    }
    let mut last = [[0u8; 32]; 2];
    last[0][..tail.len()].copy_from_slice(tail);
    last[0][tail.len()] = 1;
    let n = if tail.len() + 9 <= 32 { 1 } else { 2 };
    last[n - 1][24..].copy_from_slice(&(data.len() as u64).to_le_bytes());
    for block in &last[..n] {
        step(block);
    }
    Digest(cv)
}

/// 256 bits
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Digest(pub [u8; 32]);

impl Digest {
    /// Its four 64-bit words, the form the compression chain runs in.
    pub fn words(&self) -> [u64; 4] {
        std::array::from_fn(|i| u64::from_le_bytes(self.0.as_chunks().0[i]))
    }

    /// Its two 128-bit halves, the one form a digest travels in.
    pub fn halves(&self) -> (E, E) {
        let [w0, w1, w2, w3] = self.words();
        (E::new(w0, w1, 0), E::new(w2, w3, 0))
    }

    pub fn from_halves(low: E, high: E) -> Result<Self, VerifierError> {
        require(low.c2 == 0 && high.c2 == 0, VerifierError::DigestHalf)?;
        let mut value = [0u8; 32];
        for (bytes, word) in value
            .as_chunks_mut()
            .0
            .iter_mut()
            .zip([low.c0, low.c1, high.c0, high.c1])
        {
            *bytes = word.to_le_bytes();
        }
        Ok(Self(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_test_vector() {
        let input: [u8; 64] = std::array::from_fn(|i| i as u8);
        let got: String = haraka512(&input).0.iter().map(|byte| format!("{byte:02x}")).collect();
        assert_eq!(got, "be7f723b4e80a99813b292287f306f625a6d57331cae5f34dd9277b0945be2aa");
    }
}
