//! The compact stack: a table column or memory column is committed as the covered prefix of its real rows, cut into
//! power-of-two pieces ([`verifier::layout::pieces`]), when the proof is truncated. Everything before the PCS open reads
//! the full `w.q`; only the commit and the open see the compact stack built here.
//!
//! `q_keccak` is pieces in every proof, a half of it (a hash table) at a time: the covered rows of a half are row
//! pieces, and each is cut again by the slot blocks of the half ([`crate::keccak::slot_blocks`]): a piece holds the
//! slots of one block for the rows of one row piece, laid out `(row, slot within the block)`. The rest of a half is its
//! public pad in the committed slots, and the slots no block holds (the padding lanes of a `KECCAK64` row) are public
//! constants on every row.

use super::layout::{hash_kappa, qkeccak_slot_start};
use super::{BYTECODE_FINAL_COUNTERS, GLOBAL_COLUMN_BASES, MEMORY_FINAL_COUNTERS, ProveError, QKECCAK};
use crate::keccak::{public_slot, slot_blocks};
use crate::pcs::{self, RingSwitchClaim, RingSwitchOpen, StackClaim, StackPart};
use crate::tables;
use crate::witness::{self, Placement, StackShape};
use primitives::field::{F64, F128};
use verifier::layout::{HASH_TABLES, pieces};

pub use verifier::layout::qt_pad;

/// The pieces' digits of a column of `2^k` rows whose first `real` rows are real: bit `j` of the result says a piece
/// of `2^(k - j)` rows, `j = 0` being the whole column.
pub fn piece_digits(k: usize, real: usize) -> [bool; 5] {
    let mut digits = [false; 5];
    for (_, b) in pieces(k, real as u64) {
        digits[k - b] = true;
    }
    digits
}

/// The column of an item that is a piece of `q_keccak`'s halves: no real column.
const REGION: usize = usize::MAX;

/// One stacked piece of a column: `2^b` rows from row `off`, at `dst` in the compact stack. A piece of `q_keccak`
/// (`col` is [`REGION`], `half` its table) holds one slot block of its rows, `block = (first slot, log slots)`:
/// `2^(b + block.1)` words.
#[derive(Clone, Copy)]
struct Item {
    col: usize,
    half: usize,
    block: (usize, usize),
    off: usize,
    b: usize,
    dst: usize,
}

impl Item {
    /// The words the piece occupies, as a log.
    fn log_len(&self) -> usize {
        self.b + if self.col == REGION { self.block.1 } else { 0 }
    }
}

/// `eq(bits(i), point)`, bit 0 first.
fn eq_bits(i: usize, point: &[F128]) -> F128 {
    point
        .iter()
        .enumerate()
        .map(|(j, &x)| if i >> j & 1 == 1 { x } else { F128::ONE + x })
        .product()
}

/// The factor a slot block puts on a claim at slot point `slot`: `eq` of the slot bits the block fixes, at the point's
/// coordinates above the block's own.
fn fixed_slot_factor(block: (usize, usize), slot: &[F128]) -> F128 {
    (block.1..slot.len())
        .map(|i| {
            if block.0 >> i & 1 == 1 {
                slot[i]
            } else {
                F128::ONE + slot[i]
            }
        })
        .product()
}

/// `q_keccak` as pieces: those of each half are items of the compact stack, after the columns'.
struct Region {
    /// `log2` of the steps each table is completed to.
    kappa: usize,
    /// What a padding step holds in each slot of each half.
    pads: [[F64; 64]; HASH_TABLES.len()],
}

impl Region {
    /// `e_j` of a piece at the row point `zeta`: the weight of its rows in the column.
    fn weight(&self, piece: &Item, zeta: &[F128]) -> F128 {
        (piece.b..self.kappa)
            .map(|i| {
                if piece.off >> i & 1 == 1 {
                    zeta[i]
                } else {
                    F128::ONE + zeta[i]
                }
            })
            .product()
    }

    /// Asserts that `q_keccak` holds each half's pad past its covered rows in the committed slots, and the constants of
    /// the public slots on every row.
    fn check_pad(&self, qk: &[F64], pieces: &[Item]) {
        for half in 0..HASH_TABLES.len() {
            let covered = pieces
                .iter()
                .filter(|p| p.half == half)
                .map(|p| p.off + (1 << p.b))
                .max()
                .expect("a half has a piece");
            for (s, &want) in self.pads[half].iter().enumerate() {
                let col = &qk[qkeccak_slot_start(self.kappa, half, s)..][..1 << self.kappa];
                if let Some(constant) = public_slot(half, s) {
                    assert!(
                        col.iter().all(|&x| x.0 == constant),
                        "q_keccak half {half} slot {s} is public and is not its constant {constant:x}"
                    );
                    continue;
                }
                if covered == 1 << self.kappa {
                    continue;
                }
                assert!(
                    col[col.len() - 1] == want && col[covered] == want,
                    "q_keccak half {half} slot {s} does not end in the pad"
                );
                debug_assert!(
                    col[covered..].iter().all(|&x| x == want),
                    "half {half} slot {s} has data past the pad"
                );
            }
        }
    }
}

/// The compact stack's layout: the items of every committed column, in increasing column then offset, then those of
/// `q_keccak`.
pub(super) struct Compact {
    items: Vec<Item>,
    /// Column `c`'s items are `items[starts[c]..starts[c + 1]]`, none for a virtual column and for `q_keccak`.
    starts: Vec<usize>,
    /// The columns' items: the others are `q_keccak`'s pieces.
    n_columns: usize,
    pub(super) shape: StackShape,
    /// Words the items occupy.
    pub(super) placed: usize,
    region: Region,
    /// The memory cells the memory columns' pieces cover.
    mem_count: usize,
    /// Whether the memory and the table columns are cut: the rate word announces it.
    truncated: bool,
}

impl Compact {
    /// The items for a proof of `l` whose memory has `mem_count` cells in use. `truncated` cuts the memory and the table
    /// columns to their covered rows. `q_keccak` is pieces either way.
    pub(super) fn new(l: &super::Layout, mem_count: usize, truncated: bool) -> Result<Self, ProveError> {
        let mut items = Vec::new();
        let mut starts = Vec::with_capacity(l.placements.len() + 1);
        for (c, p) in l.placements.iter().enumerate() {
            starts.push(items.len());
            if p.is_virtual() || c == QKECCAK {
                continue;
            }
            let real = if !truncated || c == BYTECODE_FINAL_COUNTERS {
                1 << p.n_vars
            } else if c <= MEMORY_FINAL_COUNTERS {
                mem_count
            } else {
                let table = (0..tables::N_TABLES).rev().find(|&t| c >= GLOBAL_COLUMN_BASES[t]);
                l.rows[table.expect("a table column")]
            };
            items.extend(pieces(p.n_vars, real as u64).into_iter().map(|(off, b)| Item {
                col: c,
                half: 0,
                block: (0, 0),
                off: off as usize,
                b,
                dst: 0,
            }));
        }
        starts.push(items.len());
        let n_columns = items.len();
        let kappa = hash_kappa(&l.taus);
        assert_eq!(
            l.placements[QKECCAK].n_vars,
            crate::keccak::SLOT_LOG + crate::keccak::TABLE_BITS + kappa,
            "q_keccak is 64 slots of each hash table"
        );
        // `q_keccak`: for each half, each of its slot blocks, the row pieces covering the half's rows.
        for (half, &table) in HASH_TABLES.iter().enumerate() {
            for &block in slot_blocks(half) {
                items.extend(pieces(kappa, l.rows[table] as u64).into_iter().map(|(off, b)| Item {
                    col: REGION,
                    half,
                    block,
                    off: off as usize,
                    b,
                    dst: 0,
                }));
            }
        }
        let kappas: Vec<Option<usize>> = items.iter().map(|it| Some(it.log_len())).collect();
        let (placements, shape) = witness::placements_of(&kappas);
        for (it, p) in items.iter_mut().zip(&placements) {
            it.dst = p.offset;
        }
        let placed = items.iter().map(|it| 1usize << it.log_len()).sum();
        if !(pcs::MIN_STACKED_LOG..=pcs::MAX_STACKED_LOG).contains(&shape.mu) {
            return Err(ProveError::WitnessOutOfRange {
                log_committed: shape.mu,
            });
        }
        Ok(Self {
            items,
            starts,
            n_columns,
            shape,
            placed,
            region: Region {
                kappa,
                pads: [0, 1].map(|half| qt_pad(half).map(F64)),
            },
            mem_count,
            truncated,
        })
    }

    /// The compact layout of `l` whose stack reaches `2^floor` words, if one does, cutting the memory and the table
    /// columns. It truncates the memory to `mem_used` cells, else keeps it whole; past that it gives up.
    pub(super) fn reaching(l: &super::Layout, mem_used: usize, floor: usize) -> Result<Option<Self>, ProveError> {
        for mem_count in [mem_used, 1 << l.placements[MEMORY_FINAL_COUNTERS].n_vars] {
            let compact = Self::new(l, mem_count, true)?;
            if compact.shape.mu >= floor {
                return Ok(Some(compact));
            }
        }
        Ok(None)
    }

    /// What the rate word announces: the memory count of a truncated proof, nothing for a full one.
    pub(super) fn announced(&self) -> Option<usize> {
        self.truncated.then_some(self.mem_count)
    }

    fn column(&self, c: usize) -> &[Item] {
        &self.items[self.starts[c]..self.starts[c + 1]]
    }

    /// `q_keccak`'s pieces.
    fn region_items(&self) -> &[Item] {
        &self.items[self.n_columns..]
    }

    /// The public value of a truncated column past its prefix: the memory counters' `1`, zero elsewhere.
    fn pad(c: usize) -> F64 {
        if c == MEMORY_FINAL_COUNTERS {
            F64::ONE
        } else {
            F64::ZERO
        }
    }

    /// The compact stack: each item copied from the full stack `q`, the tail zero. A column past its prefix must hold
    /// its public value, which is what the verifier takes it to be.
    pub(super) fn copy(&self, q: &[F64], l: &super::Layout) -> Vec<F64> {
        for (c, p) in l.placements.iter().enumerate() {
            let Some(last) = self.column(c).last() else { continue };
            let end = last.off + (1 << last.b);
            if end < 1 << p.n_vars {
                let column = &q[p.offset..][..1 << p.n_vars];
                let pad = Self::pad(c);
                assert!(
                    column[end] == pad && column[column.len() - 1] == pad,
                    "column {c} is not its pad past row {end}"
                );
                debug_assert!(
                    column[end..].iter().all(|&x| x == pad),
                    "column {c} has data past row {end}"
                );
            }
        }
        let qk = l.placements[QKECCAK];
        let qk_words = &q[qk.offset..][..1 << qk.n_vars];
        self.region.check_pad(qk_words, self.region_items());
        let placements: Vec<Placement> = self
            .items
            .iter()
            .map(|it| Placement {
                n_vars: it.log_len(),
                offset: it.dst,
            })
            .collect();
        let mut compact = parallel::filled(self.shape.committed_len(), F64::ZERO);
        for (it, window) in self.items.iter().zip(witness::split_stack(&mut compact, &placements)) {
            if it.col == REGION {
                // Slot `block.0 + s` of the piece's rows: `2^b` words, at `s * 2^b` of the piece.
                let kappa = self.region.kappa;
                parallel::chunks_mut(window, 1 << it.b, |s, dst| {
                    let slot = it.block.0 + s;
                    dst.copy_from_slice(&qk_words[qkeccak_slot_start(kappa, it.half, slot) + it.off..][..dst.len()]);
                });
            } else {
                window.copy_from_slice(&q[l.placements[it.col].offset + it.off..][..window.len()]);
            }
        }
        compact
    }

    /// Every claim on the full stack as a claim on the compact one, in the same order, and the ring-switched claim the
    /// reduction leaves as one claim over `q_keccak`'s pieces: `sum_j c_j piece_j` has the reduction's slices less the
    /// share of what the pieces leave out, so the claim's slices are those plus the share, and each piece weighs its
    /// block by `c_j`. Nothing is sent, and the opening needs no slices.
    ///
    /// A claim on an untruncated column moves with its column. A claim `C(zeta)` on a truncated column is the weighted
    /// sum `sum_j e_j * piece_j(zeta[..b_j])`, with `e_j = prod_{i >= b_j} (bit_i(off_j) ? zeta_i : 1 + zeta_i)`: the
    /// verifier moves the claimed value by the public pad's share, which the opening does not need. A claim on a slot
    /// of `q_keccak` is that of a column of `2^kappa` rows, which are `q_keccak`'s pieces of the slot's block.
    pub(super) fn map_claims(
        &self,
        l: &super::Layout,
        claims: Vec<StackClaim>,
        mut ring: RingSwitchOpen,
    ) -> (Vec<StackClaim>, RingSwitchOpen) {
        let qk = l.placements[QKECCAK];
        let r = &self.region;
        let pieces = self.region_items();
        let [input] = <[RingSwitchClaim; 1]>::try_from(std::mem::take(&mut ring.claims)).expect("one claim");
        let (row, rest) = input.point.split_at(r.kappa);
        let (table, slot) = rest.split_at(crate::keccak::TABLE_BITS);
        ring.offset = 0;
        ring.claims = pieces
            .iter()
            .map(|piece| RingSwitchClaim {
                offset: piece.dst,
                point: [&row[..piece.b], &slot[..piece.block.1]].concat(),
                scale: eq_bits(piece.half, table) * r.weight(piece, row) * fixed_slot_factor(piece.block, slot),
            })
            .collect();
        let mapped = claims
            .into_iter()
            .map(|claim| {
                let StackClaim::Point {
                    offset,
                    low_point: zeta,
                } = claim
                else {
                    unreachable!("the prover builds Point claims")
                };
                if (qk.offset..qk.offset + (1 << qk.n_vars)).contains(&offset) {
                    let mut parts = Vec::new();
                    if zeta.len() <= r.kappa {
                        // A slot claim: the slot's column of its half, at the rows of its table, on the pieces of
                        // the slot's block.
                        let rel = offset - qk.offset;
                        let (half, slot) = ((rel >> r.kappa) & (crate::keccak::TABLES - 1), rel >> (r.kappa + 1));
                        let block = *slot_blocks(half)
                            .iter()
                            .find(|&&(start, log)| (start..start + (1 << log)).contains(&slot))
                            .expect("a claim on a committed slot");
                        let mut row = zeta;
                        row.resize(r.kappa, F128::ZERO);
                        let bits: Vec<F128> = (0..block.1)
                            .map(|j| {
                                if (slot - block.0) >> j & 1 == 1 {
                                    F128::ONE
                                } else {
                                    F128::ZERO
                                }
                            })
                            .collect();
                        for piece in pieces.iter().filter(|p| p.half == half && p.block == block) {
                            parts.push(StackPart {
                                offset: piece.dst,
                                low_point: [&row[..piece.b], &bits[..]].concat(),
                                scale: r.weight(piece, &row),
                            });
                        }
                    } else {
                        // The Keccak output claim, at the whole of `q_keccak`.
                        assert!(
                            offset == qk.offset && zeta.len() == r.kappa + crate::keccak::TABLE_BITS + 6,
                            "the output claim covers q_keccak"
                        );
                        let (row, rest) = zeta.split_at(r.kappa);
                        let (table, slot) = rest.split_at(crate::keccak::TABLE_BITS);
                        for half in 0..HASH_TABLES.len() {
                            let eq = eq_bits(half, table);
                            for piece in pieces.iter().filter(|p| p.half == half) {
                                parts.push(StackPart {
                                    offset: piece.dst,
                                    low_point: [&row[..piece.b], &slot[..piece.block.1]].concat(),
                                    scale: eq * r.weight(piece, row) * fixed_slot_factor(piece.block, slot),
                                });
                            }
                        }
                    }
                    return StackClaim::Sum(parts);
                }
                let c = l
                    .placements
                    .iter()
                    .position(|p| !p.is_virtual() && p.offset == offset)
                    .expect("a claim on a committed column");
                let k = l.placements[c].n_vars;
                let items = self.column(c);
                if let [whole] = items
                    && whole.b == k
                {
                    return StackClaim::Point {
                        offset: whole.dst,
                        low_point: zeta,
                    };
                }
                assert_eq!(zeta.len(), k, "a claim on a truncated column is at a full point");
                StackClaim::Sum(
                    items
                        .iter()
                        .map(|it| {
                            let bit = |i: usize| {
                                if it.off >> i & 1 == 1 {
                                    zeta[i]
                                } else {
                                    F128::ONE + zeta[i]
                                }
                            };
                            StackPart {
                                offset: it.dst,
                                low_point: zeta[..it.b].to_vec(),
                                scale: (it.b..k).map(bit).product(),
                            }
                        })
                        .collect(),
                )
            })
            .collect();
        (mapped, ring)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piece_digits_follow_the_pieces() {
        assert_eq!(piece_digits(19, 280940), [false, true, false, false, true]);
        assert_eq!(piece_digits(22, 2374000), [false, true, false, true, false]);
        assert_eq!(piece_digits(10, 5), [true, false, false, false, false]);
        assert_eq!(piece_digits(19, 1 << 19), [true, false, false, false, false]);
    }
}
