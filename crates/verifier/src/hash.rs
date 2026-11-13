//! Keccak-256 (Ethereum's), the permutation the VM's `KECCAK` applies, and digests.

use crate::field::E;

pub const STATE_LANES: usize = 25;
/// The sponge rate in bytes.
const RATE: usize = 136;
pub const ROUNDS: usize = 12;

pub const KECCAK_RC: [u64; ROUNDS] = [
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];
pub const KECCAK_RHO: [u32; STATE_LANES] = [
    0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14,
];
/// Where lane `x + 5y` lands.
pub const KECCAK_PI: [usize; STATE_LANES] = {
    let mut pi = [0; STATE_LANES];
    let mut i = 0;
    while i < STATE_LANES {
        pi[i] = i / 5 + 5 * ((2 * (i % 5) + 3 * (i / 5)) % 5);
        i += 1;
    }
    pi
};
pub fn keccak_f(a: &mut [u64; STATE_LANES]) {
    keccak_rounds(a, ROUNDS);
}

/// The first `rounds` rounds of the configured permutation.
pub fn keccak_rounds(a: &mut [u64; STATE_LANES], rounds: usize) {
    for rc in KECCAK_RC[..rounds].iter().copied() {
        // Row by row, so every index is a constant: an index computed per lane vectorizes to gathers.
        let c: [u64; 5] = std::array::from_fn(|x| a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20]);
        let d: [u64; 5] = std::array::from_fn(|x| c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1));
        for row in a.as_chunks_mut::<5>().0 {
            for (lane, d) in row.iter_mut().zip(d) {
                *lane ^= d;
            }
        }
        let mut b = [0u64; STATE_LANES];
        for i in 0..STATE_LANES {
            b[KECCAK_PI[i]] = a[i].rotate_left(KECCAK_RHO[i]);
        }
        for (row, b) in a.as_chunks_mut::<5>().0.iter_mut().zip(b.as_chunks::<5>().0) {
            *row = std::array::from_fn(|x| b[x] ^ (!b[(x + 1) % 5] & b[(x + 2) % 5]));
        }
        a[0] ^= rc;
    }
}

fn xor_block(state: &mut [u64; STATE_LANES], block: &[u8; RATE]) {
    for (lane, word) in state.iter_mut().zip(block.as_chunks::<8>().0) {
        *lane ^= u64::from_le_bytes(*word);
    }
}

/// Keccak-256 of `data`: 136-byte blocks, the last holding the remaining 0 to 135 bytes and the padding
/// `0x01 ‖ 0^* ‖ 0x80`.
pub fn keccak_hash(data: &[u8]) -> Digest {
    let mut state = [0u64; STATE_LANES];
    let (blocks, tail) = data.as_chunks::<RATE>();
    for block in blocks {
        xor_block(&mut state, block);
        keccak_f(&mut state);
    }
    let mut block = [0u8; RATE];
    block[..tail.len()].copy_from_slice(tail);
    block[tail.len()] ^= 0x01;
    block[RATE - 1] ^= 0x80;
    xor_block(&mut state, &block);
    keccak_f(&mut state);
    let mut value = [0u8; 32];
    for (bytes, lane) in value.as_chunks_mut().0.iter_mut().zip(state) {
        *bytes = lane.to_le_bytes();
    }
    Digest(value)
}

/// 256 bits
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Digest(pub [u8; 32]);

impl Digest {
    /// Its four 64-bit words, the form a seed takes in the sponge's lanes.
    pub fn words(&self) -> [u64; 4] {
        std::array::from_fn(|i| u64::from_le_bytes(self.0.as_chunks().0[i]))
    }

    /// Its two 128-bit halves, the one form a digest travels in.
    pub fn halves(&self) -> (E, E) {
        let [w0, w1, w2, w3] = self.words();
        (E::new(w0, w1), E::new(w2, w3))
    }

    pub fn from_halves(low: E, high: E) -> Self {
        let mut value = [0u8; 32];
        for (bytes, word) in value
            .as_chunks_mut()
            .0
            .iter_mut()
            .zip([low.c0, low.c1, high.c0, high.c1])
        {
            *bytes = word.to_le_bytes();
        }
        Self(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sponge over Keccak-p[1600, 12] with rate 136 and `0x01 .. 0x80` padding, as the `keccak` crate's
    /// permutation computes it: the reference the digests are tested against.
    fn reference_sponge(data: &[u8]) -> [u8; 32] {
        fn absorb(state: &mut [u64; STATE_LANES], block: &[u8]) {
            for (lane, bytes) in state.iter_mut().zip(block.chunks_exact(8)) {
                *lane ^= u64::from_le_bytes(bytes.try_into().unwrap());
            }
            keccak::Keccak::new().with_p1600::<{ ROUNDS }>(|f| f(state));
        }
        let mut state = [0u64; STATE_LANES];
        let (blocks, tail) = data.as_chunks::<RATE>();
        for block in blocks {
            absorb(&mut state, block);
        }
        let mut last = [0u8; RATE];
        last[..tail.len()].copy_from_slice(tail);
        last[tail.len()] ^= 0x01;
        last[RATE - 1] ^= 0x80;
        absorb(&mut state, &last);
        let mut out = [0u8; 32];
        for (bytes, lane) in out.chunks_exact_mut(8).zip(state) {
            bytes.copy_from_slice(&lane.to_le_bytes());
        }
        out
    }

    /// The permutation is Keccak-p[1600, 12], as the `keccak` crate computes it.
    #[test]
    fn permutation_matches_keccak_p_1600_12() {
        let mut seed = 0x1234_5678_9abc_def1u64;
        for case in 0..16 {
            let mut state: [u64; STATE_LANES] = std::array::from_fn(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                if case == 0 { 0 } else { seed }
            });
            let mut expected = state;
            keccak::Keccak::new().with_p1600::<{ ROUNDS }>(|f| f(&mut expected));
            keccak_f(&mut state);
            assert_eq!(state, expected, "case {case}");
        }
    }

    /// The digest of the bytes `00 01 ..` (mod 256) is the reference sponge's, on each side of a block boundary.
    #[test]
    fn matches_the_reference_sponge() {
        for n in [0, 1, 127, 128, 135, 136, 137, 200, 271, 272, 273, 512] {
            let input: Vec<u8> = (0..n).map(|i| i as u8).collect();
            assert_eq!(keccak_hash(&input).0, reference_sponge(&input), "{n} bytes");
        }
    }

    /// This crate's digest is `primitives`' hash, the one the prover and the guest compute.
    #[test]
    fn matches_the_primitives_hash() {
        for n in 0..700 {
            let input: Vec<u8> = (0..n).map(|i| (i * 7 + 3) as u8).collect();
            assert_eq!(
                keccak_hash(&input).0[..],
                primitives::hash::hash(&input)[..],
                "{n} bytes"
            );
        }
    }
}
