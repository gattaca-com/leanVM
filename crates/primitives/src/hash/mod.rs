//! Haraka-512 v2 (Kölbl, Lauridsen, Mendel, Rechberger), the repo's one hash function.
//!
//! - The 64-byte call: ten AES rounds on four states with a dword mix after every second, then the
//!   truncated feed-forward. Every hash here is a chain of them, and the VM proves one per opcode.
//! - [`hash`]: a Merkle–Damgård chain of calls over 32-byte blocks, the chaining value the first half of
//!   each call's input, with length strengthening.
//! - The batched form: many equal-length inputs at once, four calls in flight on AES-NI, for the PCS
//!   Merkle tree.
//!
//! Why Haraka: an AES S-box is one field inverse, which the prover's GF(2^8)-coefficient constraints
//! pay two rows and eight committed bits for, against the 32 AND gates of a BLAKE2s adder's worth.

#[cfg(all(target_arch = "x86_64", target_feature = "aes", target_feature = "sse2"))]
pub mod aesni;

/// Call input length in bytes.
pub const IN_LEN: usize = 64;
/// Digest length in bytes.
pub const OUT_LEN: usize = 32;
/// Bytes a chain step absorbs: the call's input past the chaining value.
pub const BLOCK_LEN: usize = IN_LEN - OUT_LEN;

pub const ROUNDS: usize = 5;
pub const AES_PER_ROUND: usize = 2;
pub const STATES: usize = 4;
pub const AES_ROUNDS: usize = ROUNDS * AES_PER_ROUND;
pub const SBOX_PER_CALL: usize = AES_ROUNDS * STATES * 16;

pub const RC32: [[u32; 4]; 40] = [
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

/// Position of an S-box among the call's [`SBOX_PER_CALL`].
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
    affine(gf_inv(x))
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

/// Portable Haraka-512.
pub fn haraka512_scalar(msg: &[u8; IN_LEN]) -> [u8; OUT_LEN] {
    let mut xs = [0u8; SBOX_PER_CALL];
    let mut us = [0u8; SBOX_PER_CALL];
    haraka512_witness_scalar(msg, &mut xs, &mut us)
}

/// The fastest available witness-recording Haraka-512.
pub fn haraka512_witness(
    msg: &[u8; IN_LEN],
    xs: &mut [u8; SBOX_PER_CALL],
    us: &mut [u8; SBOX_PER_CALL],
) -> [u8; OUT_LEN] {
    #[cfg(all(target_arch = "x86_64", target_feature = "aes", target_feature = "gfni"))]
    {
        aesni::haraka512_witness(msg, xs, us)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "aes", target_feature = "gfni")))]
    {
        haraka512_witness_scalar(msg, xs, us)
    }
}

/// Haraka-512 of 64 bytes: THE primitive, one VM opcode.
#[inline]
pub fn haraka512(msg: &[u8; IN_LEN]) -> [u8; OUT_LEN] {
    #[cfg(all(target_arch = "x86_64", target_feature = "aes", target_feature = "sse2"))]
    {
        aesni::haraka512(msg)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "aes", target_feature = "sse2")))]
    {
        haraka512_scalar(msg)
    }
}

/// Inputs one batched call hashes together.
#[cfg(all(target_arch = "x86_64", target_feature = "aes", target_feature = "sse2"))]
pub const LANES: usize = aesni::WIDTH;
#[cfg(not(all(target_arch = "x86_64", target_feature = "aes", target_feature = "sse2")))]
pub const LANES: usize = 1;
/// Inputs one batched step hashes together: a count that is a multiple of it has no remainder.
pub const BATCH: usize = LANES;

/// `LANES` calls at once: input `i` at `msgs[64i..]`, digest `i` at `outs[32i..]`.
#[inline]
fn haraka512_many(msgs: &[u8], outs: &mut [u8]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "aes", target_feature = "sse2"))]
    {
        aesni::haraka512_x4(msgs, outs)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "aes", target_feature = "sse2")))]
    {
        outs.copy_from_slice(&haraka512_scalar(msgs.try_into().unwrap()))
    }
}

/// The chain step: the next chaining value of `cv` after `block`.
#[inline]
pub fn compress(cv: &[u8; OUT_LEN], block: &[u8; BLOCK_LEN]) -> [u8; OUT_LEN] {
    let mut input = [0u8; IN_LEN];
    input[..OUT_LEN].copy_from_slice(cv);
    input[OUT_LEN..].copy_from_slice(block);
    haraka512(&input)
}

/// The chain's initial value.
pub const IV: [u8; OUT_LEN] = [0u8; OUT_LEN];

/// The blocks that end a chain of `len` bytes whose last `len % 32` are `tail`: a `1` byte, zeros, then
/// the length as eight little-endian bytes, closing a block.
#[inline]
fn final_blocks(tail: &[u8], len: u64) -> ([[u8; BLOCK_LEN]; 2], usize) {
    let mut blocks = [[0u8; BLOCK_LEN]; 2];
    blocks[0][..tail.len()].copy_from_slice(tail);
    blocks[0][tail.len()] = 1;
    let n = if tail.len() + 1 + 8 <= BLOCK_LEN { 1 } else { 2 };
    blocks[n - 1][BLOCK_LEN - 8..].copy_from_slice(&len.to_le_bytes());
    (blocks, n)
}

/// The digest as the 32-byte chaining value.
fn finish(mut cv: [u8; OUT_LEN], tail: &[u8], len: u64) -> [u8; OUT_LEN] {
    let (blocks, n) = final_blocks(tail, len);
    for block in &blocks[..n] {
        cv = compress(&cv, block);
    }
    cv
}

/// Streaming hash.
#[derive(Clone)]
pub struct Hasher {
    cv: [u8; OUT_LEN],
    buf: [u8; BLOCK_LEN],
    /// Bytes currently in `buf`, below `BLOCK_LEN`.
    buf_len: usize,
    /// Bytes absorbed so far, the buffer included.
    len: u64,
}

impl Hasher {
    pub fn new() -> Self {
        Self {
            cv: IV,
            buf: [0u8; BLOCK_LEN],
            buf_len: 0,
            len: 0,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) -> &mut Self {
        self.len += data.len() as u64;
        if self.buf_len > 0 {
            let take = (BLOCK_LEN - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == BLOCK_LEN {
                self.cv = compress(&self.cv, &self.buf);
                self.buf_len = 0;
            }
            if data.is_empty() {
                return self;
            }
        }
        let (blocks, rest) = data.as_chunks::<BLOCK_LEN>();
        for block in blocks {
            self.cv = compress(&self.cv, block);
        }
        self.buf[..rest.len()].copy_from_slice(rest);
        self.buf_len = rest.len();
        self
    }

    /// The input so far as the prefix of many inputs.
    pub fn prefix(self) -> Prefix {
        Prefix(self)
    }

    pub fn finalize(&self) -> [u8; OUT_LEN] {
        finish(self.cv, &self.buf[..self.buf_len], self.len)
    }
}

impl Default for Hasher {
    fn default() -> Self {
        Self::new()
    }
}

/// A [`Hasher::prefix`], hashed with each tail.
#[derive(Clone)]
pub struct Prefix(Hasher);

impl Prefix {
    /// The hash of the prefix then `tail`.
    pub fn hash(&self, tail: &[u8]) -> [u8; OUT_LEN] {
        self.0.clone().update(tail).finalize()
    }
}

/// One-shot hash of any length.
pub fn hash(data: &[u8]) -> [u8; OUT_LEN] {
    let (blocks, rest) = data.as_chunks::<BLOCK_LEN>();
    let cv = blocks.iter().fold(IV, |cv, block| compress(&cv, block));
    finish(cv, rest, data.len() as u64)
}

/// The chaining value after `n_blocks` all-zero blocks from the IV.
///
/// PCS leaves share such a prefix, their absent interleaving lanes: the committer absorbs it once and
/// starts every leaf here. Digests are unchanged, a prefix's steps depending on nothing after them.
pub fn zero_prefix_state(n_blocks: usize) -> [u8; OUT_LEN] {
    (0..n_blocks).fold(IV, |cv, _| compress(&cv, &[0u8; BLOCK_LEN]))
}

/// The one-shot hash, continued from a chaining value: `data` is the rest of the image and `t_offset`
/// counts the bytes already absorbed into `state`.
pub fn hash_from_state(data: &[u8], state: &[u8; OUT_LEN], t_offset: u64) -> [u8; OUT_LEN] {
    let (blocks, rest) = data.as_chunks::<BLOCK_LEN>();
    let cv = blocks.iter().fold(*state, |cv, block| compress(&cv, block));
    finish(cv, rest, t_offset + data.len() as u64)
}

/// The batched hash, every input continued from one shared chaining value.
///
/// Each digest equals the hash of its full image.
pub fn hash_many_dyn_from_state(data: &[u8], len: usize, state: &[u8; OUT_LEN], t_offset: u64, out: &mut [u8]) {
    assert!(len > 0, "batched inputs are not empty");
    let n = out.len() / OUT_LEN;
    assert_eq!(data.len(), n * len);
    assert_eq!(out.len(), n * OUT_LEN);
    let whole = len / BLOCK_LEN;
    let (pad, n_pad) = final_blocks(&[0u8; BLOCK_LEN][..len % BLOCK_LEN], t_offset + len as u64);
    let tail_len = len % BLOCK_LEN;
    let mut msgs = [0u8; IN_LEN * LANES];
    let mut outs = [0u8; OUT_LEN * LANES];
    for (inputs, digests) in data.chunks(len * LANES).zip(out.chunks_mut(OUT_LEN * LANES)) {
        let k = digests.len() / OUT_LEN;
        for lane in 0..k {
            msgs[IN_LEN * lane..IN_LEN * lane + OUT_LEN].copy_from_slice(state);
        }
        for b in 0..whole + n_pad {
            for lane in 0..k {
                let input = &inputs[len * lane..len * (lane + 1)];
                let block = &mut msgs[IN_LEN * lane + OUT_LEN..IN_LEN * (lane + 1)];
                if b < whole {
                    block.copy_from_slice(&input[BLOCK_LEN * b..BLOCK_LEN * (b + 1)]);
                } else {
                    block.copy_from_slice(&pad[b - whole]);
                    if b == whole {
                        block[..tail_len].copy_from_slice(&input[BLOCK_LEN * whole..]);
                    }
                }
            }
            if k == LANES {
                haraka512_many(&msgs, &mut outs);
            } else {
                for lane in 0..k {
                    let msg: &[u8; IN_LEN] = msgs[IN_LEN * lane..IN_LEN * (lane + 1)].try_into().unwrap();
                    outs[OUT_LEN * lane..OUT_LEN * (lane + 1)].copy_from_slice(&haraka512(msg));
                }
            }
            for lane in 0..k {
                msgs[IN_LEN * lane..IN_LEN * lane + OUT_LEN]
                    .copy_from_slice(&outs[OUT_LEN * lane..OUT_LEN * (lane + 1)]);
            }
        }
        digests.copy_from_slice(&outs[..OUT_LEN * k]);
    }
}

/// The batched hash with a runtime input length.
pub fn hash_many_dyn(data: &[u8], len: usize, out: &mut [u8]) {
    hash_many_dyn_from_state(data, len, &IV, 0, out);
}

/// Batched hash of `LEN`-byte inputs, one 32-byte digest each: each digest equals the one-shot hash of
/// its input.
pub fn hash_many<const LEN: usize>(data: &[u8], out: &mut [u8]) {
    hash_many_dyn(data, LEN, out);
}

/// Batched calls: `n` inputs of 64 bytes, one call each, into `n` digests.
pub fn haraka512_many_dyn(data: &[u8], out: &mut [u8]) {
    let n = out.len() / OUT_LEN;
    assert_eq!(data.len(), n * IN_LEN);
    let (full, rest) = (n / LANES, n % LANES);
    for i in 0..full {
        haraka512_many(
            &data[IN_LEN * LANES * i..IN_LEN * LANES * (i + 1)],
            &mut out[OUT_LEN * LANES * i..OUT_LEN * LANES * (i + 1)],
        );
    }
    for i in n - rest..n {
        let msg: &[u8; IN_LEN] = data[IN_LEN * i..IN_LEN * (i + 1)].try_into().unwrap();
        out[OUT_LEN * i..OUT_LEN * (i + 1)].copy_from_slice(&haraka512(msg));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector() {
        let msg: [u8; 64] = std::array::from_fn(|i| i as u8);
        let hex: String = haraka512(&msg).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "be7f723b4e80a99813b292287f306f625a6d57331cae5f34dd9277b0945be2aa");
        assert_eq!(haraka512_scalar(&msg), haraka512(&msg));
    }

    #[test]
    fn witness_paths_agree() {
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

    #[test]
    fn chain_forms_agree() {
        for len in [0usize, 1, 23, 24, 31, 32, 33, 55, 56, 64, 100, 1024] {
            let data: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
            let one_shot = hash(&data);
            let mut streaming = Hasher::new();
            for piece in data.chunks(13) {
                streaming.update(piece);
            }
            assert_eq!(streaming.finalize(), one_shot, "len {len}");
            let split = len / 3;
            let prefix = Hasher::new().update(&data[..split]).clone().prefix();
            assert_eq!(prefix.hash(&data[split..]), one_shot, "prefix {len}");
            let n = 7;
            let many: Vec<u8> = (0..n).flat_map(|_| data.clone()).collect();
            let mut out = vec![0u8; n * OUT_LEN];
            if len > 0 {
                hash_many_dyn(&many, len, &mut out);
                for d in out.chunks(OUT_LEN) {
                    assert_eq!(d, &one_shot, "batched {len}");
                }
            }
            if len >= 64 {
                let zeros = 2;
                let image = [vec![0u8; zeros * BLOCK_LEN], data.clone()].concat();
                let state = zero_prefix_state(zeros);
                assert_eq!(hash_from_state(&data, &state, (zeros * BLOCK_LEN) as u64), hash(&image));
                hash_many_dyn_from_state(&many, len, &state, (zeros * BLOCK_LEN) as u64, &mut out);
                assert_eq!(&out[..OUT_LEN], &hash(&image));
            }
        }
        let calls: Vec<u8> = (0..9 * 64).map(|i| (i * 5) as u8).collect();
        let mut out = vec![0u8; 9 * OUT_LEN];
        haraka512_many_dyn(&calls, &mut out);
        for i in 0..9 {
            assert_eq!(
                &out[32 * i..32 * i + 32],
                &haraka512(calls[64 * i..64 * i + 64].try_into().unwrap())
            );
        }
    }
}
