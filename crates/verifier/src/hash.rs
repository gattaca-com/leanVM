//! Keccak-256 (Ethereum's), the permutation the VM's `KECCAK` applies, and digests.

use crate::field::E;

pub const STATE_LANES: usize = 25;
/// The sponge rate in bytes.
pub(crate) const RATE: usize = 136;
pub(crate) const PAD_FIRST: u8 = 0x01;
pub const ROUNDS: usize = 24;

pub const KECCAK_RC: [u64; ROUNDS] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
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
    let mut aba = a[0];
    let mut abe = a[1];
    let mut abi = a[2];
    let mut abo = a[3];
    let mut abu = a[4];
    let mut aga = a[5];
    let mut age = a[6];
    let mut agi = a[7];
    let mut ago = a[8];
    let mut agu = a[9];
    let mut aka = a[10];
    let mut ake = a[11];
    let mut aki = a[12];
    let mut ako = a[13];
    let mut aku = a[14];
    let mut ama = a[15];
    let mut ame = a[16];
    let mut ami = a[17];
    let mut amo = a[18];
    let mut amu = a[19];
    let mut asa = a[20];
    let mut ase = a[21];
    let mut asi = a[22];
    let mut aso = a[23];
    let mut asu = a[24];

    for round in 0..ROUNDS / 2 {
        let ca = aba ^ aga ^ aka ^ ama ^ asa;
        let ce = abe ^ age ^ ake ^ ame ^ ase;
        let ci = abi ^ agi ^ aki ^ ami ^ asi;
        let co = abo ^ ago ^ ako ^ amo ^ aso;
        let cu = abu ^ agu ^ aku ^ amu ^ asu;
        let d_a = cu ^ ce.rotate_left(1);
        let d_e = ca ^ ci.rotate_left(1);
        let d_i = ce ^ co.rotate_left(1);
        let d_o = ci ^ cu.rotate_left(1);
        let d_u = co ^ ca.rotate_left(1);

        let ba = aba ^ d_a;
        let be = (age ^ d_e).rotate_left(44);
        let bi = (aki ^ d_i).rotate_left(43);
        let bo = (amo ^ d_o).rotate_left(21);
        let bu = (asu ^ d_u).rotate_left(14);
        let eba = ba ^ (!be & bi) ^ KECCAK_RC[2 * round];
        let ebe = be ^ (!bi & bo);
        let ebi = bi ^ (!bo & bu);
        let ebo = bo ^ (!bu & ba);
        let ebu = bu ^ (!ba & be);

        let ba = (abo ^ d_o).rotate_left(28);
        let be = (agu ^ d_u).rotate_left(20);
        let bi = (aka ^ d_a).rotate_left(3);
        let bo = (ame ^ d_e).rotate_left(45);
        let bu = (asi ^ d_i).rotate_left(61);
        let ega = ba ^ (!be & bi);
        let ege = be ^ (!bi & bo);
        let egi = bi ^ (!bo & bu);
        let ego = bo ^ (!bu & ba);
        let egu = bu ^ (!ba & be);

        let ba = (abe ^ d_e).rotate_left(1);
        let be = (agi ^ d_i).rotate_left(6);
        let bi = (ako ^ d_o).rotate_left(25);
        let bo = (amu ^ d_u).rotate_left(8);
        let bu = (asa ^ d_a).rotate_left(18);
        let eka = ba ^ (!be & bi);
        let eke = be ^ (!bi & bo);
        let eki = bi ^ (!bo & bu);
        let eko = bo ^ (!bu & ba);
        let eku = bu ^ (!ba & be);

        let ba = (abu ^ d_u).rotate_left(27);
        let be = (aga ^ d_a).rotate_left(36);
        let bi = (ake ^ d_e).rotate_left(10);
        let bo = (ami ^ d_i).rotate_left(15);
        let bu = (aso ^ d_o).rotate_left(56);
        let ema = ba ^ (!be & bi);
        let eme = be ^ (!bi & bo);
        let emi = bi ^ (!bo & bu);
        let emo = bo ^ (!bu & ba);
        let emu = bu ^ (!ba & be);

        let ba = (abi ^ d_i).rotate_left(62);
        let be = (ago ^ d_o).rotate_left(55);
        let bi = (aku ^ d_u).rotate_left(39);
        let bo = (ama ^ d_a).rotate_left(41);
        let bu = (ase ^ d_e).rotate_left(2);
        let esa = ba ^ (!be & bi);
        let ese = be ^ (!bi & bo);
        let esi = bi ^ (!bo & bu);
        let eso = bo ^ (!bu & ba);
        let esu = bu ^ (!ba & be);

        let ca = eba ^ ega ^ eka ^ ema ^ esa;
        let ce = ebe ^ ege ^ eke ^ eme ^ ese;
        let ci = ebi ^ egi ^ eki ^ emi ^ esi;
        let co = ebo ^ ego ^ eko ^ emo ^ eso;
        let cu = ebu ^ egu ^ eku ^ emu ^ esu;
        let d_a = cu ^ ce.rotate_left(1);
        let d_e = ca ^ ci.rotate_left(1);
        let d_i = ce ^ co.rotate_left(1);
        let d_o = ci ^ cu.rotate_left(1);
        let d_u = co ^ ca.rotate_left(1);

        let ba = eba ^ d_a;
        let be = (ege ^ d_e).rotate_left(44);
        let bi = (eki ^ d_i).rotate_left(43);
        let bo = (emo ^ d_o).rotate_left(21);
        let bu = (esu ^ d_u).rotate_left(14);
        aba = ba ^ (!be & bi) ^ KECCAK_RC[2 * round + 1];
        abe = be ^ (!bi & bo);
        abi = bi ^ (!bo & bu);
        abo = bo ^ (!bu & ba);
        abu = bu ^ (!ba & be);

        let ba = (ebo ^ d_o).rotate_left(28);
        let be = (egu ^ d_u).rotate_left(20);
        let bi = (eka ^ d_a).rotate_left(3);
        let bo = (eme ^ d_e).rotate_left(45);
        let bu = (esi ^ d_i).rotate_left(61);
        aga = ba ^ (!be & bi);
        age = be ^ (!bi & bo);
        agi = bi ^ (!bo & bu);
        ago = bo ^ (!bu & ba);
        agu = bu ^ (!ba & be);

        let ba = (ebe ^ d_e).rotate_left(1);
        let be = (egi ^ d_i).rotate_left(6);
        let bi = (eko ^ d_o).rotate_left(25);
        let bo = (emu ^ d_u).rotate_left(8);
        let bu = (esa ^ d_a).rotate_left(18);
        aka = ba ^ (!be & bi);
        ake = be ^ (!bi & bo);
        aki = bi ^ (!bo & bu);
        ako = bo ^ (!bu & ba);
        aku = bu ^ (!ba & be);

        let ba = (ebu ^ d_u).rotate_left(27);
        let be = (ega ^ d_a).rotate_left(36);
        let bi = (eke ^ d_e).rotate_left(10);
        let bo = (emi ^ d_i).rotate_left(15);
        let bu = (eso ^ d_o).rotate_left(56);
        ama = ba ^ (!be & bi);
        ame = be ^ (!bi & bo);
        ami = bi ^ (!bo & bu);
        amo = bo ^ (!bu & ba);
        amu = bu ^ (!ba & be);

        let ba = (ebi ^ d_i).rotate_left(62);
        let be = (ego ^ d_o).rotate_left(55);
        let bi = (eku ^ d_u).rotate_left(39);
        let bo = (ema ^ d_a).rotate_left(41);
        let bu = (ese ^ d_e).rotate_left(2);
        asa = ba ^ (!be & bi);
        ase = be ^ (!bi & bo);
        asi = bi ^ (!bo & bu);
        aso = bo ^ (!bu & ba);
        asu = bu ^ (!ba & be);
    }

    a[0] = aba;
    a[1] = abe;
    a[2] = abi;
    a[3] = abo;
    a[4] = abu;
    a[5] = aga;
    a[6] = age;
    a[7] = agi;
    a[8] = ago;
    a[9] = agu;
    a[10] = aka;
    a[11] = ake;
    a[12] = aki;
    a[13] = ako;
    a[14] = aku;
    a[15] = ama;
    a[16] = ame;
    a[17] = ami;
    a[18] = amo;
    a[19] = amu;
    a[20] = asa;
    a[21] = ase;
    a[22] = asi;
    a[23] = aso;
    a[24] = asu;
}

#[cfg(test)]
fn reference_keccak_f(a: &mut [u64; STATE_LANES]) {
    for rc in KECCAK_RC {
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
    block[tail.len()] ^= PAD_FIRST;
    block[RATE - 1] ^= 0x80;
    xor_block(&mut state, &block);
    keccak_f(&mut state);
    Digest::from_state(&state)
}

/// 256 bits
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Digest(pub [u8; 32]);

impl Digest {
    /// The first 32 bytes of a Keccak state, little-endian within each lane.
    pub(crate) fn from_state(state: &[u64; STATE_LANES]) -> Self {
        let mut value = [0u8; 32];
        for (bytes, lane) in value.as_chunks_mut().0.iter_mut().zip(state) {
            *bytes = lane.to_le_bytes();
        }
        Self(value)
    }

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

    #[test]
    fn permutation_matches_reference() {
        let mut seed = 0xD1B54A32D192ED03u64;
        for _ in 0..4096 {
            let mut state = std::array::from_fn(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            });
            let mut expected = state;
            reference_keccak_f(&mut expected);
            keccak_f(&mut state);
            assert_eq!(state, expected);
        }
    }

    #[test]
    fn permutation_of_zero() {
        // The full zero-state output, generated with reference_keccak_f.
        let expected = [
            0xF1258F7940E1DDE7,
            0x84D5CCF933C0478A,
            0xD598261EA65AA9EE,
            0xBD1547306F80494D,
            0x8B284E056253D057,
            0xFF97A42D7F8E6FD4,
            0x90FEE5A0A44647C4,
            0x8C5BDA0CD6192E76,
            0xAD30A6F71B19059C,
            0x30935AB7D08FFC64,
            0xEB5AA93F2317D635,
            0xA9A6E6260D712103,
            0x81A57C16DBCF555F,
            0x43B831CD0347C826,
            0x01F22F1A11A5569F,
            0x05E5635A21D9AE61,
            0x64BEFEF28CC970F2,
            0x613670957BC46611,
            0xB87C5A554FD00ECB,
            0x8C3EE88A1CCF32C8,
            0x940C7922AE3A2614,
            0x1841F924A2C509E4,
            0x16F53526E70465C2,
            0x75F644E97F30A13B,
            0xEAF1FF7B5CECA249,
        ];
        let mut reference = [0; STATE_LANES];
        reference_keccak_f(&mut reference);
        assert_eq!(reference, expected);
        let mut state = [0; STATE_LANES];
        keccak_f(&mut state);
        assert_eq!(state, expected);
    }

    /// Keccak-256 of the bytes `00 01 ..` (mod 256), from the `sha3` crate's `Keccak256`, on each side of a block
    /// boundary.
    const VECTORS: [(usize, &str); 11] = [
        (0, "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
        (1, "bc36789e7a1e281436464229828f817d6612f7b477d66591ff96a9e064bcc98a"),
        (127, "c52f0bd08793b9e8601b29753539e1bf47f8e483eed0a901e8761982449c9b4c"),
        (128, "ed4c9adc183fb8cb025b1500ec3eeae1b45517314441a187605de1bb8a64726e"),
        (135, "cbdfd9dee5faad3818d6b06f95a219fd290b0e1706f6a82e5a595b9ce9faca62"),
        (136, "7ce759f1ab7f9ce437719970c26b0a66ff11fe3e38e17df89cf5d29c7d7f807e"),
        (137, "ac73d4fae68b8453f764007c1a20ce95994187861f0c3227a3a8e99a73a3b1db"),
        (200, "bfb0aa97863e797943cf7c33bb7e880bb4543f3d2703c0923c6901c2af57b890"),
        (271, "7c974895b2a88303ff2dc6b58f438ceb0b298cac91099ac0539cc0f477506191"),
        (272, "fdf2ec49e749960d3c8521a0219af8d03e30e2b3bf19bd16150ee0eaf133d66e"),
        (512, "f55ba327291604f0e5be6651752398b7be2331aad65f5763ce067df95cc13be1"),
    ];

    #[test]
    fn matches_reference_vectors() {
        for (n, want) in VECTORS {
            let input: Vec<u8> = (0..n).map(|i| i as u8).collect();
            let got: String = keccak_hash(&input).0.iter().map(|byte| format!("{byte:02x}")).collect();
            assert_eq!(got, want, "{n} bytes");
        }
    }
}
