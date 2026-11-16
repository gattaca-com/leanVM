//! Test-only forgeries: a prover that breaks one thing in the witness it commits, so that a test can check the
//! verifier refuses the proof. The forgery is set per thread and read by [`super::prove`] after it builds the witness.

use super::layout::{Witness, hash_kappa, keccak_value, qkeccak_slot_start, schema};
use super::{KeccakLane, QKECCAK};
use crate::tables;
use primitives::field::{F64, F128};
use primitives::hash::STATE_LANES;
use std::cell::RefCell;

#[derive(Clone, Copy, Debug)]
pub(crate) enum Forge {
    /// Lane `lane` of the first `KECCAK64` row's input state is not what the padding fixes.
    Keccak64InputLane(usize),
    /// The first `KECCAK64` row's output lanes are those of the second row.
    Keccak64OtherRowsOutput,
    /// The first `rows` steps of the two halves of `q_keccak` change places, in the slots of the states.
    SwapRows(usize),
    /// A value among the 21 output lanes `KECCAK64` sends is wrong.
    SentLane,
    /// The first padding row of `KECCAK64` holds the state of its first row.
    Keccak64PaddingRowIsReal,
    /// Row 0 of `KECCAK64`'s `out_two` column holds `to`, whatever the bytecode says.
    Keccak64OutTwo(bool),
    /// The first word of row piece `piece` of slot block `block` of table half `half`, as `(half, block, piece)`, changes
    /// in the committed stack, not in the reduction's witness.
    PieceWord(usize, (usize, usize), usize),
    /// Past its rows each half of `q_keccak` holds the padding step of the other half.
    SwapPads,
    /// Lane `lane` of the first `KECCAK64` row's input state is `value`, and the rest of the row follows the permutation:
    /// its state after the first rounds and its output lanes are those of the state it now holds.
    Keccak64Lane { lane: usize, value: u64 },
    /// The column `column` (`tables::keccak64::S_A`, `K_A` or `K_Y`) of the first `KECCAK64` row holds `xor` more than the bytecode.
    Keccak64Column { column: usize, xor: u64 },
    /// The blocks `[0, 8)` and `[24, 32)` of the first row piece of `KECCAK64` change places in the committed stack.
    SwapBlocks,
    /// The block `[24, 32)` of the first row piece of `KECCAK64` holds the second row piece's, in the committed stack.
    BorrowBlock,
}

thread_local! {
    static FORGE: RefCell<Option<Forge>> = const { RefCell::new(None) };
}

/// Forge the next proofs this thread makes.
pub(crate) fn set(forge: Option<Forge>) {
    FORGE.with(|f| *f.borrow_mut() = forge);
}

pub(super) fn current() -> Option<Forge> {
    FORGE.with(|f| *f.borrow())
}

/// Whether this thread's prover is forging.
pub(crate) fn active() -> bool {
    current().is_some()
}

/// Where a slot of a half of `q_keccak` starts in the witness's stack.
fn slot(w: &Witness, half: usize, slot: usize) -> usize {
    w.layout.placements[QKECCAK].offset + qkeccak_slot_start(hash_kappa(&w.layout.taus), half, slot)
}

/// Apply the forgery to a witness, before the commitment.
pub(crate) fn tamper(w: &mut Witness) {
    let Some(forge) = current() else { return };
    let keccak64_rows = w.layout.rows[verifier::layout::OP_KECCAK64];
    match forge {
        Forge::Keccak64InputLane(lane) => {
            let at = slot(w, crate::keccak::KECCAK64_HALF, lane);
            w.q[at].0 ^= 1;
        }
        Forge::Keccak64OtherRowsOutput => {
            let base = schema().base[verifier::layout::OP_KECCAK64] + tables::keccak64::V_OUT;
            for lane in 0..tables::keccak64::OUT_LANES {
                let (_, column) = w
                    .virt
                    .iter_mut()
                    .find(|(col, _)| *col == base + lane)
                    .expect("an output lane");
                column[0] = column[1];
            }
        }
        Forge::SwapRows(rows) => {
            for s in 0..crate::keccak::SLOTS {
                let (a, b) = (
                    slot(w, crate::keccak::KECCAK_HALF, s),
                    slot(w, crate::keccak::KECCAK64_HALF, s),
                );
                for row in 0..rows {
                    w.q.swap(a + row, b + row);
                }
            }
        }
        Forge::SwapPads => {
            // Past its rows each half holds the other's padding step. The reduction proves what is there.
            let kappa = hash_kappa(&w.layout.taus);
            for (half, &table) in verifier::layout::HASH_TABLES.iter().enumerate() {
                let rows = w.layout.rows[table];
                let other = super::qt_pad(1 - half);
                for (s, &word) in other.iter().enumerate() {
                    let word = crate::keccak::public_slot(1 - half, s).unwrap_or(word);
                    let start = slot(w, half, s);
                    w.q[start + rows..start + (1 << kappa)].fill(F64(word));
                }
            }
        }
        Forge::Keccak64Lane { lane, value } => {
            let half = crate::keccak::KECCAK64_HALF;
            let mut state: [u64; STATE_LANES] = std::array::from_fn(|l| w.q[slot(w, half, l)].0);
            state[lane] = value;
            for (l, &word) in state.iter().enumerate() {
                let at = slot(w, half, l);
                w.q[at] = F64(word);
            }
            // The state the reduction commits after its first rounds, and the output lanes the bus carries.
            let mut output = state;
            primitives::hash::keccak_f(&mut output);
            verifier::keccak_rounds(&mut state, crate::keccak::DEPTH);
            for (l, &word) in state.iter().enumerate() {
                let at = slot(w, half, STATE_LANES + l);
                w.q[at] = F64(word);
            }
            let state = output;
            let base = schema().base[verifier::layout::OP_KECCAK64] + tables::keccak64::V_OUT;
            for l in 0..tables::keccak64::OUT_LANES {
                let (_, column) = w
                    .virt
                    .iter_mut()
                    .find(|(col, _)| *col == base + l)
                    .expect("an output lane");
                column[0] = F64(state[l]);
            }
        }
        Forge::Keccak64Column { column, xor } => {
            let col = schema().base[verifier::layout::OP_KECCAK64] + column;
            let at = w.layout.placements[col].offset;
            w.q[at].0 ^= xor;
        }
        Forge::SentLane | Forge::PieceWord(..) | Forge::SwapBlocks | Forge::BorrowBlock => {}
        Forge::Keccak64PaddingRowIsReal => {
            for lane in 0..8 {
                let at = slot(w, crate::keccak::KECCAK64_HALF, lane);
                w.q[at + keccak64_rows] = w.q[at];
            }
        }
        Forge::Keccak64OutTwo(to) => {
            let col = schema().base[verifier::layout::OP_KECCAK64] + tables::keccak64::OUT_TWO;
            let Some((half, KeccakLane::Slot(s))) = keccak_value(col) else {
                panic!("the flag rides a slot of q_keccak")
            };
            let at = slot(w, half, s);
            w.q[at] = F64(u64::from(to));
        }
    }
}

/// Apply the forgery to the compact stack the prover commits, which the reduction never reads.
pub(super) fn tamper_compact(compact: &super::truncate::Compact, q: &mut [F64]) {
    let keccak64 = || {
        compact
            .qt
            .pieces
            .iter()
            .filter(|p| p.half == crate::keccak::KECCAK64_HALF)
    };
    let dst = |p: &verifier::QtPiece| p.placement.index as usize;
    match current() {
        Some(Forge::PieceWord(half, block, piece)) => {
            let it = (compact.qt.pieces.iter())
                .filter(|p| p.half == half && p.block == block)
                .nth(piece)
                .expect("a row piece of the block");
            q[dst(it)].0 ^= 1;
        }
        Some(Forge::SwapBlocks) => {
            let first_piece = keccak64().next().expect("KECCAK64 has a piece").off;
            let [a, b] = [(0, 3), (24, 3)].map(|block| {
                keccak64()
                    .find(|p| p.block == block && p.off == first_piece)
                    .expect("a block of the first row piece")
            });
            assert_eq!(a.b(), b.b());
            for i in 0..1usize << (a.b() + 3) {
                q.swap(dst(a) + i, dst(b) + i);
            }
        }
        Some(Forge::BorrowBlock) => {
            let block = |n: usize| keccak64().filter(|p| p.block == (24, 3)).nth(n);
            let [own, other] = [0, 1].map(|n| block(n).expect("KECCAK64 has two row pieces"));
            for i in 0..1usize << (other.b() + 3) {
                q[dst(own) + i] = q[dst(other) + i];
            }
        }
        _ => {}
    }
}

/// Apply the forgery to the output lanes `KECCAK64` sends.
pub(crate) fn tamper_sent(rest: &mut [F128]) {
    if let Some(Forge::SentLane) = current() {
        rest[3] += F128::ONE;
    }
}
