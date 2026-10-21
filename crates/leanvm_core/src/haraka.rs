//! Bridge to the flock Haraka-512 prover ([`flock::haraka`]), single-PCS.
//!
//! `q_flock` (flock's packed witness, 64 bits per `F64` word) is committed as a column in leanVM's ONE
//! stacked `F64` witness (§sec:stacking), with no separate flock commitment. The VM's `Haraka` table
//! binds to it by point-eval equality (its value columns and `q_flock`'s slots are point-evals of the
//! same committed stack), and flock's R1CS validity is discharged by the same stacked WHIR: the
//! reduction's claim passes through [`ring_switch_open`] and joins the batch-mixed opening
//! ([`::pcs::stack_open`]).
//!
//! ## The mapping
//!
//! The VM's `Haraka(m0, m1, m2, m3) -> out` is one Haraka-512 call: the four 16-byte message cells
//! laid out little-endian are its 64-byte input, the 32-byte digest lands in two cells. Every word is a
//! witness value in `q_flock`, and the memory interaction binds each of them.
//!
//! ## The layout (64-bit words)
//!
//! A call's `2^K_LOG` bits pack into `2^(K_LOG-6)` `F64` words (the [`SLOT_STRIDE_LOG`] stride); each
//! VM-visible 64-bit word is one whole packed word at a fixed within-call slot (bit position / 64):
//!
//! ```text
//!   m0..m3 = slots 0..8 (lo, hi lanes)    out = slots 8..12
//! ```

use crate::transcript::ProverState;
use ::pcs::pack::LOG_PACKING;
use flock::haraka::{HarakaSetup, Input, K_LOG, Witness, haraka512};
use primitives::field::{F64, F192};
use primitives::stream::Stream;

pub use flock::haraka::{SliceClaim, min_n_log as n_calls_log, qflock_kappa, ring_switch_open};

/// Names the circuit in the Fiat-Shamir seed, independent of the instance count.
pub const R1CS_LABEL: &[u8] = b"flock/haraka512/gf256-r1cs/v1";

/// Flock-native reduction buffers emitted in the same fused pass as the committed, flattened `q_flock`.
/// They stay alive across commit, bus, and constraint proving so reduction needs no second witness pass.
pub(crate) struct PreparedReductionWitness {
    n_calls: usize,
    witness: Witness,
}

impl PreparedReductionWitness {
    pub(crate) fn n_calls(&self) -> usize {
        self.n_calls
    }

    pub(crate) fn prove(&self, ps: &mut ProverState) -> SliceClaim {
        let setup = HarakaSetup::new(self.n_calls);
        let stage = setup.prove_zerocheck(&self.witness, ps);
        setup.prove_lincheck(stage, &self.witness.z_lincheck, ps)
    }
}

/// The twelve within-call value slots in canonical order `[m0.lo, m0.hi, .., m3.hi, out0.lo, .., out1.hi]`,
/// matching `tables::HARAKA_VALUE_COLS`.
pub const SLOTS: [usize; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// The call's 64-byte input from the four message cells (canonical 128-bit chunks, top limbs zero).
pub fn input(cells: [F192; 4]) -> Input {
    let mut m = [0u8; 64];
    for (i, c) in cells.iter().enumerate() {
        m[16 * i..16 * i + 8].copy_from_slice(&c.c0.to_le_bytes());
        m[16 * i + 8..16 * i + 16].copy_from_slice(&c.c1.to_le_bytes());
    }
    m
}

/// The digest of a call as its four little-endian words.
pub fn digest(input: &Input) -> [F64; 4] {
    let h = haraka512(input);
    std::array::from_fn(|k| F64(u64::from_le_bytes(h[8 * k..8 * k + 8].try_into().unwrap())))
}

/// The digest as the two cells it lands in.
pub fn digest_cells(input: &Input) -> [F192; 2] {
    let d = digest(input);
    [F192::new(d[0].0, d[1].0, 0), F192::new(d[2].0, d[3].0, 0)]
}

/// The twelve VM-visible words of flock's padding instance, the call on 64 zero bytes, in [`SLOTS`] order:
/// what the `Haraka` table's value columns hold on a padding row, since those columns ARE the `q_flock`
/// slots of the instance at that row.
pub fn padding_words() -> [F64; 12] {
    let d = digest(&flock::haraka::PADDING_INPUT);
    std::array::from_fn(|i| if i < 8 { F64::ZERO } else { d[i - 8] })
}

/// Lift flock's packed witness (64 bits per word, bit `i` at position `i`) into the committed `F64`
/// column: word for word, which is exactly `pcs::pack`'s LSB-first convention on the same bit string.
fn flatten_packed_into(packed: &[u64], out: &mut [F64]) {
    assert_eq!(out.len(), packed.len(), "q_flock's window is the wrong size");
    // SAFETY: `F64` is `repr(transparent)` over `u64`, so the two slices are the same bytes.
    let words: &mut [u64] = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast(), out.len()) };
    parallel::chunks_mut_zip(words, packed, 1 << 14, |_, dst, src| Stream::new().copy(dst, src));
}

/// Build the committed `q_flock` column (flock's packed witness) for `inputs`, padded to
/// `2^n_calls_log(max(inputs.len(),1))` instances (the unused ones flock's own padding call), and retain
/// the flock-native layouts produced by that same fused pass so reduction does not regenerate them.
pub(crate) fn build_qflock_prepared(inputs: &[Input], q_flock: &mut [F64]) -> PreparedReductionWitness {
    let n_calls = inputs.len().max(1);
    let mut witness = flock::haraka::generate(inputs, n_calls_log(n_calls));
    flatten_packed_into(&witness.z, q_flock);
    witness.z = Vec::new();
    PreparedReductionWitness { n_calls, witness }
}

/// `log2` of the within-call packed span (`2^7` words): the number of low coords of a `q_flock` point
/// that carry the slot's bits, and the stride between consecutive calls' same-slot words in `q_flock`.
/// A value claim on `q_flock` is thus a boolean-selector (strided) claim with this stride.
pub const SLOT_STRIDE_LOG: usize = K_LOG - LOG_PACKING;

#[cfg(test)]
mod tests {
    use super::*;

    fn build_qflock(inputs: &[Input]) -> Vec<F64> {
        let mut q_flock = vec![F64::ZERO; 1 << qflock_kappa(inputs.len())];
        build_qflock_prepared(inputs, &mut q_flock);
        q_flock
    }

    /// `q_flock`'s aligned packed slots hold the VM's 64-bit words, and the digest matches the primitive.
    #[test]
    fn qflock_words_match_layout() {
        let cells: Vec<[F192; 4]> = (0..5u64)
            .map(|i| std::array::from_fn(|k| F192::new(0x1000 * (k as u64 + 1) + i, 0x5000 * (k as u64 + 1) + i, 0)))
            .collect();
        let inputs: Vec<Input> = cells.iter().map(|&c| input(c)).collect();
        let q_flock = build_qflock(&inputs);
        assert_eq!(q_flock.len(), 1 << qflock_kappa(inputs.len()));
        let slot = |j: usize, s: usize| q_flock[j * (1 << SLOT_STRIDE_LOG) + s];
        for (j, (c, inp)) in cells.iter().zip(&inputs).enumerate() {
            for k in 0..4 {
                assert_eq!(slot(j, 2 * k), F64(c[k].c0));
                assert_eq!(slot(j, 2 * k + 1), F64(c[k].c1));
            }
            let d = digest(inp);
            assert_eq!(primitives::hash::haraka512(inp), {
                let mut h = [0u8; 32];
                for k in 0..4 {
                    h[8 * k..8 * k + 8].copy_from_slice(&d[k].0.to_le_bytes());
                }
                h
            });
            for k in 0..4 {
                assert_eq!(slot(j, 8 + k), d[k]);
            }
        }
    }

    /// The padding row every verifier evaluates the `Haraka` table's bus forms at is what `q_flock`
    /// holds at a padding instance.
    #[test]
    fn padding_words_are_the_padding_instance() {
        let q_flock = build_qflock(&[]);
        for (&s, w) in SLOTS.iter().zip(padding_words()) {
            assert_eq!(q_flock[s], w, "slot {s}");
        }
    }
}
