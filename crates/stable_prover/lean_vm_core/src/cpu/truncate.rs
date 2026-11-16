//! Truncated column: a table column or memory column is committed as the covered prefix of its real rows, cut into
//! power-of-two pieces ([`verifier::layout::pieces`]). `q_keccak` is cut the same way, a half of it (a hash table) at a
//! time, when that commits fewer words: the covered rows of a half are pieces, each holding every slot of its rows, laid
//! out `(row, slot)`, and the rest of a half is its public pad.
//! Everything before the PCS open reads the full `w.q`; only the commit and the open see the compact stack built here.

use super::layout::{hash_kappa, qkeccak_slot_start};
use super::{BYTECODE_FINAL_COUNTERS, GLOBAL_COLUMN_BASES, MEMORY_FINAL_COUNTERS, ProveError, QKECCAK};
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
/// (`col` is [`REGION`], `half` its table) is all 64 slots of its rows, `2^(b + 6)` words.
#[derive(Clone, Copy)]
struct Item {
    col: usize,
    half: usize,
    off: usize,
    b: usize,
    dst: usize,
}

impl Item {
    /// The words the piece occupies, as a log.
    fn log_len(&self) -> usize {
        self.b + if self.col == REGION { 6 } else { 0 }
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

/// The 64 bit slices of a block of words at a point: bit `z` of every word, as a multilinear, evaluated.
fn slices(block: &[F64], point: &[F128]) -> Vec<F128> {
    let eq = primitives::multilinear::eq_table(point);
    parallel::fold_reduce(
        block.len(),
        || vec![F128::ZERO; 64],
        |sums, i| {
            let mut bits = block[i].0;
            while bits != 0 {
                sums[bits.trailing_zeros() as usize] += eq[i];
                bits &= bits - 1;
            }
        },
        |mut a, b| {
            for (a, b) in a.iter_mut().zip(b) {
                *a += b;
            }
            a
        },
    )
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

    /// `1 + sum_j e_j`, the weight of the rows past a half's pieces.
    fn uncovered(&self, pieces: &[Item], half: usize, zeta: &[F128]) -> F128 {
        F128::ONE
            + pieces
                .iter()
                .filter(|p| p.half == half)
                .map(|p| self.weight(p, zeta))
                .sum::<F128>()
    }

    /// Asserts that `q_keccak` holds each half's pad past its covered rows.
    fn check_pad(&self, qk: &[F64], pieces: &[Item]) {
        for half in 0..HASH_TABLES.len() {
            let covered = pieces
                .iter()
                .filter(|p| p.half == half)
                .map(|p| p.off + (1 << p.b))
                .max()
                .expect("a half has a piece");
            if covered == 1 << self.kappa {
                continue;
            }
            for (s, &want) in self.pads[half].iter().enumerate() {
                let col = &qk[qkeccak_slot_start(self.kappa, half, s)..][..1 << self.kappa];
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
/// `q_keccak` if it is pieces.
pub(super) struct Compact {
    items: Vec<Item>,
    /// Column `c`'s items are `items[starts[c]..starts[c + 1]]`, none for a virtual column.
    starts: Vec<usize>,
    /// The columns' items: the others are `q_keccak`'s pieces.
    n_columns: usize,
    pub(super) shape: StackShape,
    /// Words the items occupy.
    pub(super) placed: usize,
    /// `q_keccak` as pieces, when that commits fewer words than keeping it whole: then the `QKECCAK` column has no item.
    region: Option<Region>,
    /// The memory cells the memory columns' pieces cover.
    mem_count: usize,
}

impl Compact {
    /// The items for a proof of `l` whose memory has `mem_count` cells in use.
    fn new(l: &super::Layout, mem_count: usize) -> Result<Self, ProveError> {
        let mut items = Vec::new();
        let mut starts = Vec::with_capacity(l.placements.len() + 1);
        for (c, p) in l.placements.iter().enumerate() {
            starts.push(items.len());
            if p.is_virtual() {
                continue;
            }
            let real = if c == BYTECODE_FINAL_COUNTERS || c == QKECCAK {
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
                off: off as usize,
                b,
                dst: 0,
            }));
        }
        starts.push(items.len());
        let kappa = hash_kappa(&l.taus);
        assert_eq!(
            l.placements[QKECCAK].n_vars,
            crate::keccak::SLOT_LOG + crate::keccak::TABLE_BITS + kappa,
            "q_keccak is 64 slots of each hash table"
        );
        let placed_of = |items: &[Item]| -> (Vec<usize>, StackShape) {
            let kappas: Vec<Option<usize>> = items.iter().map(|it| Some(it.log_len())).collect();
            let (placements, shape) = witness::placements_of(&kappas);
            (placements.iter().map(|p| p.offset).collect(), shape)
        };
        // The columns' items, `q_keccak` whole; then QT, `q_keccak` as the pieces of its halves in place of its item.
        let (mut dsts, mut shape) = placed_of(&items);
        let mut placed = items.iter().map(|it| 1usize << it.b).sum();
        let mut qt_items: Vec<Item> = items.iter().copied().filter(|it| it.col != QKECCAK).collect();
        let qt_columns = qt_items.len();
        for (half, &table) in HASH_TABLES.iter().enumerate() {
            qt_items.extend(pieces(kappa, l.rows[table] as u64).into_iter().map(|(off, b)| Item {
                col: REGION,
                half,
                off: off as usize,
                b,
                dst: 0,
            }));
        }
        let (qt_dsts, qt_shape) = placed_of(&qt_items);
        let mut region = None;
        let mut n_columns = items.len();
        if qt_shape.committed_len() < shape.committed_len() {
            region = Some(Region {
                kappa,
                pads: [0, 1].map(|half| qt_pad(half).map(F64)),
            });
            // The columns' starts: `q_keccak`'s column has no item now.
            starts = (0..=l.placements.len())
                .map(|c| qt_items[..qt_columns].iter().filter(|it| it.col < c).count())
                .collect();
            placed = qt_items.iter().map(|it| 1usize << it.log_len()).sum();
            (items, dsts, shape, n_columns) = (qt_items, qt_dsts, qt_shape, qt_columns);
        }
        for (it, dst) in items.iter_mut().zip(dsts) {
            it.dst = dst;
        }
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
            region,
            mem_count,
        })
    }

    /// The compact layout of `l` whose stack reaches `2^floor` words, if one does.
    /// It truncates the memory to `mem_used` cells, else keeps it whole; past that it gives up.
    pub(super) fn reaching(l: &super::Layout, mem_used: usize, floor: usize) -> Result<Option<Self>, ProveError> {
        for mem_count in [mem_used, 1 << l.placements[MEMORY_FINAL_COUNTERS].n_vars] {
            let compact = Self::new(l, mem_count)?;
            if compact.shape.mu >= floor {
                return Ok(Some(compact));
            }
        }
        Ok(None)
    }

    /// What the rate word announces: the memory count, and whether q_keccak is pieces.
    pub(super) fn announced(&self) -> (usize, bool) {
        (self.mem_count, self.region.is_some())
    }

    fn column(&self, c: usize) -> &[Item] {
        &self.items[self.starts[c]..self.starts[c + 1]]
    }

    /// `q_keccak`'s pieces, if it is pieces.
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
        if let Some(r) = &self.region {
            r.check_pad(qk_words, self.region_items());
        }
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
                // Slot `s` of the piece's rows: `2^b` words, at `s * 2^b` of the piece.
                let kappa = self.region.as_ref().expect("a region").kappa;
                parallel::chunks_mut(window, 1 << it.b, |s, dst| {
                    dst.copy_from_slice(&qk_words[qkeccak_slot_start(kappa, it.half, s) + it.off..][..dst.len()]);
                });
            } else {
                window.copy_from_slice(&q[l.placements[it.col].offset + it.off..][..window.len()]);
            }
        }
        compact
    }

    /// Where a full-stack offset inside `q_keccak` sits in the compact stack, when `q_keccak` is one item.
    fn qkeccak_offset(&self, l: &super::Layout, offset: usize) -> usize {
        offset - l.placements[QKECCAK].offset + self.column(QKECCAK)[0].dst
    }

    /// Every claim on the full stack as a claim on the compact one, in the same order, and the ring-switched claim the
    /// reduction leaves as the claims of `q_keccak`'s pieces, with the slices of those the verifier does not derive,
    /// which the prover sends. `input_slices` are the reduction's slices of `q_keccak` at the ring claim's point, and
    /// `compact` is the compact stack.
    ///
    /// A claim on `q_keccak` and on an untruncated column moves with its column. A claim `C(zeta)` on a truncated
    /// column is the weighted sum `sum_j e_j * piece_j(zeta[..b_j])`, with `e_j = prod_{i >= b_j} (bit_i(off_j) ?
    /// zeta_i : 1 + zeta_i)`: the verifier moves the claimed value by the public pad's share, which the opening does
    /// not need. A claim on a slot of `q_keccak` is that of a column of `2^kappa` rows, which are `q_keccak`'s pieces
    /// when it is.
    pub(super) fn map_claims(
        &self,
        l: &super::Layout,
        claims: Vec<StackClaim>,
        mut ring: RingSwitchOpen,
        input_slices: &[F128],
        compact: &[F64],
    ) -> (Vec<StackClaim>, RingSwitchOpen, Vec<Vec<F128>>) {
        let qk = l.placements[QKECCAK];
        let mut sent = Vec::new();
        if let Some(r) = &self.region {
            let [input] = <[RingSwitchClaim; 1]>::try_from(std::mem::take(&mut ring.claims)).expect("one claim");
            let (row, rest) = input.point.split_at(r.kappa);
            let (table, slot) = rest.split_at(crate::keccak::TABLE_BITS);
            let pieces = self.region_items();
            // The derived piece is the largest, the first of them: its slices are the reduction's less the others'.
            let derived = (0..pieces.len())
                .max_by_key(|&j| (pieces[j].b, std::cmp::Reverse(j)))
                .expect("a piece");
            let coefficient = |piece: &Item| eq_bits(piece.half, table) * r.weight(piece, row);
            // The pad's share of the reduction's slices: bit `i` of every pad word, weighed by its slot, half and rows.
            let mut pad_share = vec![F128::ZERO; 64];
            let eq_slot: Vec<F128> = (0..64).map(|s| eq_bits(s, slot)).collect();
            for half in 0..HASH_TABLES.len() {
                let u = eq_bits(half, table) * r.uncovered(pieces, half, row);
                for (i, share) in pad_share.iter_mut().enumerate() {
                    let mass: F128 = (0..64)
                        .filter(|&s| r.pads[half][s].0 >> i & 1 == 1)
                        .map(|s| eq_slot[s])
                        .sum();
                    *share += u * mass;
                }
            }
            let mut slices_of: Vec<Vec<F128>> = pieces
                .iter()
                .enumerate()
                .map(|(j, piece)| {
                    if j == derived {
                        Vec::new()
                    } else {
                        slices(
                            &compact[piece.dst..][..1 << piece.log_len()],
                            &[&row[..piece.b], slot].concat(),
                        )
                    }
                })
                .collect();
            // The derived piece: `S = sum_j c_j s_j + pad share`.
            let mut rest_sum: Vec<F128> = input_slices.iter().zip(&pad_share).map(|(&s, &pad)| s + pad).collect();
            for (piece, own) in pieces.iter().zip(&slices_of) {
                let c = coefficient(piece);
                for (acc, &s) in rest_sum.iter_mut().zip(own) {
                    *acc += c * s;
                }
            }
            let inverse = coefficient(&pieces[derived]).inv();
            slices_of[derived] = rest_sum.into_iter().map(|s| s * inverse).collect();
            ring.offset = 0;
            ring.claims = pieces
                .iter()
                .map(|piece| RingSwitchClaim {
                    offset: piece.dst,
                    point: [&row[..piece.b], slot].concat(),
                })
                .collect();
            sent = slices_of
                .into_iter()
                .enumerate()
                .filter(|&(j, _)| j != derived)
                .map(|(_, own)| own)
                .collect();
        } else {
            ring.offset = self.qkeccak_offset(l, ring.offset);
        }
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
                    let Some(r) = &self.region else {
                        return StackClaim::Point {
                            offset: self.qkeccak_offset(l, offset),
                            low_point: zeta,
                        };
                    };
                    let pieces = self.region_items();
                    let mut parts = Vec::new();
                    if zeta.len() <= r.kappa {
                        // A slot claim: the slot's column of its half, at the rows of its table.
                        let rel = offset - qk.offset;
                        let (half, slot) = ((rel >> r.kappa) & (crate::keccak::TABLES - 1), rel >> (r.kappa + 1));
                        let mut row = zeta;
                        row.resize(r.kappa, F128::ZERO);
                        let bits: Vec<F128> = (0..6)
                            .map(|j| if slot >> j & 1 == 1 { F128::ONE } else { F128::ZERO })
                            .collect();
                        for piece in pieces.iter().filter(|p| p.half == half) {
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
                                    low_point: [&row[..piece.b], slot].concat(),
                                    scale: eq * r.weight(piece, row),
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
        (mapped, ring, sent)
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
