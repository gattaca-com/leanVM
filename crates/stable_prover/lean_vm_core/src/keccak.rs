//! The `KECCAK` glue to the Keccak reduction ([`keccak_gkr`], Annex C), over the one commitment.
//!
//! The VM's `KECCAK` is Keccak-f\[1600\] of the 25 lanes thirteen cells hold, read and written each in one of two
//! layouts ([`keccak_cell_lanes`]). The committed column `q_keccak` holds every step's input lanes and its state every
//! [`DEPTH`] rounds before the output, a slot per lane and a word per step; the output is not committed. The
//! `KECCAK` table commits no lane of its own: a claim on an input lane is a claim on its slot of `q_keccak`, and the
//! claims on the output lanes are the reduction's claim on the permutation's output ([`keccak_gkr::OutputClaim`]).
//! The reduction proves the rounds between, and its two claims on `q_keccak` join the one opening.

use keccak_gkr::{Openings, OutputClaim, Shape, TableInputs, Witness, ZERO_PAD};
use primitives::field::{F64, F128};
use primitives::hash::STATE_LANES;
use primitives::transcript::ProverState;
use verifier::layout::keccak_cell_lanes;

pub use isa::STATE_CELLS;

/// Rounds between two committed states.
pub const DEPTH: usize = verifier::layout::KECCAK_DEPTH;
/// The reduction's shape at its fewest steps, which the recursion guest's constants come from.
pub const MIN_SHAPE: Shape = shape(0);
/// `log2` of `q_keccak`'s slots a step: one per lane of each committed state, rounded up to a power of two.
pub const SLOT_LOG: usize = MIN_SHAPE.slot_log();
/// Slots a step's committed states occupy: the ones from here to `2^SLOT_LOG` carry the KECCAK table's own columns
/// (`cpu::keccak_value`).
pub const SLOTS: usize = MIN_SHAPE.slots();
/// The fewest steps the reduction is sized for, one of its blocks.
pub const MIN_STEPS: usize = MIN_SHAPE.n();
/// Cells one `KECCAK` row flushes: thirteen read, thirteen written or, past what it writes, cancelled.
pub(crate) const ROW_CELLS: usize = 2 * STATE_CELLS;
/// Cells one `KECCAK64` row flushes: four read, then the one or two written (the second cancelled when it is not).
pub(crate) const KECCAK64_ROW_CELLS: usize = 6;

/// `log2` of the tables the reduction proves together: `KECCAK` is half 0 of `q_keccak`, `KECCAK64` half 1.
pub const TABLE_BITS: usize = verifier::layout::TABLE_BITS;
/// The layout's slots the verifier owns: which `KECCAK64` slots are public, and the blocks of the others.
pub use verifier::layout::{KECCAK64_PUBLIC_SLOTS, public_slot, slot_blocks};
/// The tables the reduction proves together.
pub const TABLES: usize = 1 << TABLE_BITS;
/// The half of `q_keccak`, and the table bit, of `KECCAK` and of `KECCAK64`.
pub const KECCAK_HALF: usize = 0;
pub const KECCAK64_HALF: usize = 1;

const fn shape(n_steps: usize) -> Shape {
    Shape::with_tables(n_steps, DEPTH, TABLE_BITS)
}

/// `log2` of the steps each table is completed to, for tables of `rows` executed steps: the tallest's.
pub(crate) fn kappa(rows: [usize; TABLES]) -> usize {
    rows.into_iter()
        .map(|n| Shape::new(n, DEPTH).n_log())
        .max()
        .expect("hash tables")
}

/// The input every padding step of each table holds: zeros for `KECCAK`, a message of zeros for `KECCAK64`.
pub(crate) fn pad_inputs() -> [[u64; STATE_LANES]; TABLES] {
    [ZERO_PAD, verifier::layout::keccak64_pad()]
}

/// The 25 lanes of a `KECCAK64` row's state, from the four cells `cell(0..4)` it reads.
pub(crate) fn keccak64_state(cell: impl Fn(usize) -> F128) -> [u64; STATE_LANES] {
    let mut state = verifier::layout::keccak64_pad();
    for c in 0..4 {
        let w = cell(c);
        state[2 * c] = w.c0;
        state[2 * c + 1] = w.c1;
    }
    state
}

/// The 25 lanes the thirteen state cells `cell(0..13)` hold in a layout.
pub(crate) fn state_of_cells(odd: bool, cell: impl Fn(usize) -> F128) -> [u64; STATE_LANES] {
    // The layout a constant, the loop compiles to straight-line moves.
    fn read<const ODD: bool>(cell: impl Fn(usize) -> F128) -> [u64; STATE_LANES] {
        let mut s = [0; STATE_LANES];
        for c in 0..STATE_CELLS {
            let (lo, hi) = const { keccak_cell_lanes(ODD) }[c];
            let w = cell(c);
            if let Some(l) = lo {
                s[l] = w.c0;
            }
            if let Some(h) = hi {
                s[h] = w.c1;
            }
        }
        s
    }
    if odd { read::<true>(cell) } else { read::<false>(cell) }
}

/// The thirteen state cells holding 25 lanes in a layout, a free limb zero.
pub(crate) fn cells_of_state(s: &[u64; STATE_LANES], odd: bool) -> [F128; STATE_CELLS] {
    let lanes = keccak_cell_lanes(odd);
    std::array::from_fn(|c| {
        let (lo, hi) = lanes[c];
        F128::new(lo.map_or(0, |l| s[l]), hi.map_or(0, |h| s[h]))
    })
}

/// What a `KECCAK` instruction writes, given the cells it reads.
#[cfg(test)]
pub(crate) fn permute_cells(cells: &[F128; STATE_CELLS], in_odd: bool, out_odd: bool) -> [F128; STATE_CELLS] {
    let mut s = state_of_cells(in_odd, |c| cells[c]);
    primitives::hash::keccak_f(&mut s);
    cells_of_state(&s, out_odd)
}

/// The reduction's sizes. Its witness, every round's state of every step, is rebuilt from `q_keccak`'s input states
/// when it runs: held from the witness build instead, it would sit beside the bus's trees, at the prover's peak.
pub(crate) struct Reduction {
    shape: Shape,
    steps: [usize; TABLES],
}

impl Reduction {
    /// Rebuild the witness from `q_keccak`, the whole committed column, and run the reduction.
    pub(crate) fn prove(self, ps: &mut ProverState, q_keccak: &[F64], output: &OutputClaim) -> Openings {
        let witness = tracing::info_span!("Witness")
            .in_scope(|| Witness::from_committed(self.shape, &self.steps, &pad_inputs(), q_keccak));
        witness.prove(ps, q_keccak, output)
    }
}

/// Write the committed states of the steps `inputs` of each table, each padded with its padding steps up to `2^kappa`,
/// over `q_keccak`'s first [`SLOTS`] slots of each table, `states`: the spare slots past them are their columns' to fill.
pub(crate) fn place(inputs: [&[[u64; STATE_LANES]]; TABLES], kappa: usize, states: &mut [F64]) -> Reduction {
    let pads = pad_inputs();
    let tables: [TableInputs; TABLES] = std::array::from_fn(|t| TableInputs {
        inputs: inputs[t],
        pad: pads[t],
    });
    let shape = shape(1 << kappa);
    keccak_gkr::witness::commit_states(shape, &tables, states);
    Reduction {
        shape,
        steps: inputs.map(<[_]>::len),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_instances(n: usize) -> Vec<[u64; STATE_LANES]> {
        (0..n as u64)
            .map(|i| std::array::from_fn(|l| 0x0101_0101_0101_0101u64.wrapping_mul(i + 1) ^ l as u64))
            .collect()
    }

    /// Both layouts round-trip, and a padded message laid out in cells, block `k`'s nine cells
    /// from cell `17k / 2` XORed into the previous output, read and written in the layout of each
    /// block's parity, the last written even, hashes to its Keccak-256.
    #[test]
    fn layouts_absorb_a_message() {
        use primitives::hash::{PAD_FIRST, PAD_LAST, RATE, permutations};
        let s: [u64; STATE_LANES] = std::array::from_fn(|i| 0x1111 * (i as u64 + 1));
        for odd in [false, true] {
            let cells = cells_of_state(&s, odd);
            assert_eq!(state_of_cells(odd, |c| cells[c]), s);
        }
        for n in [0, 1, 64, 127, 135, 136, 137, 200, 271, 272, 300, 543, 1000] {
            let msg: Vec<u8> = (0..n).map(|i| (i * 7 + 3) as u8).collect();
            let blocks = permutations(n);
            let mut padded = msg.clone();
            padded.resize(blocks * RATE + 8, 0);
            padded[n] ^= PAD_FIRST;
            padded[blocks * RATE - 1] ^= PAD_LAST;
            let cells: Vec<F128> = padded
                .as_chunks::<16>()
                .0
                .iter()
                .map(|c| {
                    let w = |k: usize| u64::from_le_bytes(c[8 * k..8 * k + 8].try_into().unwrap());
                    F128::new(w(0), w(1))
                })
                .collect();
            let mut state = [F128::ZERO; STATE_CELLS];
            for k in 0..blocks {
                for (j, cell) in state.iter_mut().take(9).enumerate() {
                    *cell += cells[17 * k / 2 + j];
                }
                state = permute_cells(&state, k % 2 == 1, k + 1 < blocks && k % 2 == 0);
            }
            let digest: Vec<u8> = [state[0].c0, state[0].c1, state[1].c0, state[1].c1]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            assert_eq!(digest, primitives::hash::hash(&msg), "{n} bytes");
        }
    }

    /// The committed column holds every step's input lanes, a word per step. The output lanes
    /// have no slot.
    #[test]
    fn qkeccak_words_match_layout() {
        let (keccak, keccak64) = (sample_instances(5), sample_instances(9));
        let kappa = kappa([keccak.len(), keccak64.len()]);
        let n = 1usize << kappa;
        let mut q_keccak = vec![F64::ZERO; SLOTS * n * TABLES];
        place([&keccak, &keccak64], kappa, &mut q_keccak);
        // The layout is `(step, table, slot)`: a table's slot is a run of `n` words.
        for (t, instances) in [&keccak, &keccak64].into_iter().enumerate() {
            for (j, input) in instances.iter().enumerate() {
                for i in 0..STATE_LANES {
                    assert_eq!(q_keccak[(i * TABLES + t) * n + j], F64(input[i]));
                }
            }
        }
        // The steps past a table's own hold its padding step.
        let pads = pad_inputs();
        for i in 0..STATE_LANES {
            assert_eq!(q_keccak[i * TABLES * n + 5], F64(pads[0][i]));
            assert_eq!(q_keccak[(i * TABLES + 1) * n + 9], F64(pads[1][i]));
        }
    }
}
