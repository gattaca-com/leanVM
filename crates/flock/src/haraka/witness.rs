//! Witness generation: the packed bits `z` (the committed column), the per-row
//! `c` bytes the zerocheck reads (`a` and `b` derive from them, see
//! [`super::kernels::abc_column`]) and the lincheck byte stripe of `z`.

use super::circuit::*;
use super::primitive::{IN_LEN, OUT_LEN, SBOX_PER_CALL, haraka512_witness};
use primitives::bits::bit_transpose_64bytes;

/// One call's input, 64 bytes.
pub type Input = [u8; IN_LEN];

pub struct Witness {
    /// `2^(K_LOG + n_log - 6)` words: bit `i` of the witness is bit `i % 64` of word `i / 64`.
    pub z: Vec<u64>,
    /// One byte per row, `2^(ROWS_LOG + n_log)`.
    pub c: Vec<u8>,
    /// The lincheck stripe layout of `z` (see `lincheck`'s module docs).
    pub z_lincheck: Vec<u8>,
}

fn set_bit(buf: &mut [u8], bit: usize) {
    buf[bit / 8] |= 1 << (bit % 8);
}

/// The call every unused call slot holds: zero bits do not satisfy the rows past the first round
/// (the round constants make the S-box inputs nonzero), so padding is a real call of this input.
pub const PADDING_INPUT: Input = [0u8; IN_LEN];

/// One slot into `zb` (`K / 8` bytes) and `c` (`ROWS_PER_SLOT` bytes), both zeroed.
fn gen_slot(inputs: &[Input], zb: &mut [u8], c: &mut [u8]) {
    let mut xs = [0u8; SBOX_PER_CALL];
    let mut us = [0u8; SBOX_PER_CALL];
    for call in 0..CALLS_PER_SLOT {
        let msg = inputs.get(call).unwrap_or(&PADDING_INPUT);
        let out = haraka512_witness(msg, &mut xs, &mut us);
        let zbase = call * CALL_BITS / 8;
        zb[zbase + IN_BASE / 8..zbase + IN_BASE / 8 + IN_LEN].copy_from_slice(msg);
        zb[zbase + OUT_BASE / 8..zbase + OUT_BASE / 8 + OUT_LEN].copy_from_slice(&out);
        zb[zbase + U_BASE / 8..zbase + U_BASE / 8 + SBOX_PER_CALL].copy_from_slice(&us);
        let rbase = call * CALL_ROWS;
        for k in 0..SBOX_PER_CALL {
            c[rbase + 2 * k] = xs[k];
            c[rbase + 2 * k + 1] = us[k];
        }
        for i in 0..OUT_BITS {
            c[rbase + SBOX_ROWS + i] = out[i / 8] >> (i % 8) & 1;
        }
    }
    set_bit(zb, CONST_COL);
}

/// Build the witness for `inputs.len()` calls in `2^n_log` slots; empty slots hold only the constant wire.
pub fn generate(inputs: &[Input], n_log: usize) -> Witness {
    let n_slots = 1usize << n_log;
    assert!(inputs.len() <= n_slots * CALLS_PER_SLOT);
    assert!(n_log >= 3, "the stripe layout needs at least 8 slots");
    let zbytes = K / 8;
    let n_groups = n_slots / 8;
    // SAFETY (x3): every byte is written below, groups partitioning each buffer.
    let mut z = unsafe { primitives::uninit_vec::<u64>(n_slots * K / 64) };
    let mut c = unsafe { primitives::uninit_vec::<u8>(n_slots * ROWS_PER_SLOT) };
    let mut z_lincheck = unsafe { primitives::uninit_vec::<u8>(n_groups * K) };
    let z_chunks = parallel::Chunks::new(&mut z, 8 * K / 64);
    let c_chunks = parallel::Chunks::new(&mut c, 8 * ROWS_PER_SLOT);
    let s_chunks = parallel::Chunks::new(&mut z_lincheck, K);
    parallel::for_each(n_groups, |g| {
        // SAFETY: one task per group, each chunk taken once.
        let (zw, cg, sg) = unsafe { (z_chunks.get(g), c_chunks.get(g), s_chunks.get(g)) };
        let mut zb = vec![0u8; 8 * zbytes];
        cg.fill(0);
        for r in 0..8 {
            let slot = 8 * g + r;
            let lo = (slot * CALLS_PER_SLOT).min(inputs.len());
            let hi = ((slot + 1) * CALLS_PER_SLOT).min(inputs.len());
            gen_slot(
                &inputs[lo..hi],
                &mut zb[r * zbytes..(r + 1) * zbytes],
                &mut cg[r * ROWS_PER_SLOT..(r + 1) * ROWS_PER_SLOT],
            );
        }
        for (w, out) in zw.iter_mut().enumerate() {
            *out = u64::from_le_bytes(zb[8 * w..8 * w + 8].try_into().unwrap());
        }
        // Stripe byte `i` holds bit `i` of the group's eight slots: a 64-byte block per 64 inner bits.
        let mut inp = [0u8; 64];
        let mut outp = [0u8; 64];
        for blk in 0..K / 64 {
            for r in 0..8 {
                inp[8 * r..8 * r + 8].copy_from_slice(&zb[r * zbytes + 8 * blk..r * zbytes + 8 * blk + 8]);
            }
            bit_transpose_64bytes(&inp, &mut outp);
            sg[64 * blk..64 * blk + 64].copy_from_slice(&outp);
        }
    });
    Witness { z, c, z_lincheck }
}

/// Bit `inner` of slot `slot` of the witness.
pub fn z_bit(w: &Witness, slot: usize, inner: usize) -> bool {
    let i = slot * K + inner;
    w.z[i / 64] >> (i % 64) & 1 == 1
}
