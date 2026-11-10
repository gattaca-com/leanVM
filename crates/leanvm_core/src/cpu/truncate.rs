//! Truncated columns: a table column or memory column is committed as the covered prefix of its
//! real rows, cut into power-of-two pieces. Everything before the PCS open reads the full `w.q`;
//! only the commit and the open see the compact stack built here. The piece rule is the
//! verifier's (`verifier::layout::pieces`).

use super::{BFCNT, MFCNT, ProveError, QKECCAK, schema};
use crate::pcs::{self, RingSwitchOpen, SlotClaim, StackPart};
use crate::tables;
use crate::witness::{self, Placement, StackShape};
use primitives::field::{F64, F128};

/// A column keeps its rows to within `2^TRUNC_D` of its real count, in pieces of at least `2^MIN_PIECE_LOG` rows.
const TRUNC_D: usize = 4;
const MIN_PIECE_LOG: usize = 10;
/// Words per task of the compact copy.
const COPY_CHUNK: usize = 1 << 16;

/// The pieces `(offset in the column, log size)`, by increasing offset, that commit the first `real` rows of
/// a column of `2^k` rows: the prefix rounded up to a multiple of `2^max(k - TRUNC_D, MIN_PIECE_LOG)`, cut by
/// its binary digits. A column of at most `2^MIN_PIECE_LOG` rows, or one fully covered, is one piece.
fn pieces(k: usize, real: usize) -> impl Iterator<Item = (usize, usize)> {
    let g_log = (k.saturating_sub(TRUNC_D)).max(MIN_PIECE_LOG).min(k);
    let covered = (real.max(1).div_ceil(1 << g_log) << g_log).min(1 << k);
    let mask = if k <= MIN_PIECE_LOG { 1 << k } else { covered };
    let mut off = 0;
    (g_log..=k).rev().filter(move |&b| mask >> b & 1 == 1).map(move |b| {
        let piece = (off, b);
        off += 1 << b;
        piece
    })
}

/// One stacked piece of a column: `2^b` rows from row `off`, at `dst` in the compact stack.
#[derive(Clone, Copy)]
struct Item {
    col: usize,
    off: usize,
    b: usize,
    dst: usize,
}

/// The compact stack's layout: the items of every committed column, in increasing column then offset.
pub(super) struct Compact {
    items: Vec<Item>,
    /// Column `c`'s items are `items[starts[c]..starts[c + 1]]`, none for a virtual column.
    starts: Vec<usize>,
    pub(super) shape: StackShape,
    /// Words the items occupy.
    pub(super) placed: usize,
}

impl Compact {
    /// The items for a proof of `l` whose memory has `mem_used` cells in use.
    pub(super) fn new(l: &super::Layout, mem_used: usize) -> Result<Self, ProveError> {
        let sch = schema();
        let mut items = Vec::new();
        let mut starts = Vec::with_capacity(l.placements.len() + 1);
        for (c, p) in l.placements.iter().enumerate() {
            starts.push(items.len());
            if p.is_virtual() {
                continue;
            }
            let real = if c == BFCNT || c == QKECCAK {
                1 << p.n_vars
            } else if c <= MFCNT {
                mem_used
            } else {
                l.rows[(0..tables::N_TABLES)
                    .rev()
                    .find(|&t| c >= sch.base[t])
                    .expect("a table column")]
            };
            items.extend(pieces(p.n_vars, real).map(|(off, b)| Item { col: c, off, b, dst: 0 }));
        }
        starts.push(items.len());
        let kappas: Vec<Option<usize>> = items.iter().map(|it| Some(it.b)).collect();
        let (placements, shape) = witness::placements_of(&kappas);
        for (it, p) in items.iter_mut().zip(&placements) {
            it.dst = p.offset;
        }
        if !(pcs::MIN_STACKED_LOG..=pcs::MAX_STACKED_LOG).contains(&shape.mu) {
            return Err(ProveError::WitnessOutOfRange {
                log_committed: shape.mu,
            });
        }
        let placed = items.iter().map(|it| 1 << it.b).sum();
        Ok(Self {
            items,
            starts,
            shape,
            placed,
        })
    }

    fn column(&self, c: usize) -> &[Item] {
        &self.items[self.starts[c]..self.starts[c + 1]]
    }

    /// The public value of a truncated column past its prefix.
    fn pad(c: usize) -> F64 {
        if c == MFCNT { F64::ONE } else { F64::ZERO }
    }

    /// The compact stack: each item copied from the full stack `q`, the tail zero.
    /// Asserts first that every truncated column holds its pad past the prefix.
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
        // SAFETY: `split_stack` hands out one window per item, tiling `0..placed`, and zeroes the tail.
        // The loop below writes every window.
        let mut compact = unsafe { witness::alloc_stack(self.shape) };
        let placements: Vec<Placement> = self
            .items
            .iter()
            .map(|it| Placement {
                n_vars: it.b,
                offset: it.dst,
            })
            .collect();
        for (it, window) in self.items.iter().zip(witness::split_stack(&mut compact, &placements)) {
            let src = &q[l.placements[it.col].offset + it.off..][..window.len()];
            parallel::chunks_mut(window, COPY_CHUNK, |i, chunk| {
                chunk.copy_from_slice(&src[i * COPY_CHUNK..][..chunk.len()]);
            });
        }
        compact
    }

    /// Where a full-stack offset inside `q_keccak` sits in the compact stack.
    fn qkeccak_offset(&self, l: &super::Layout, offset: usize) -> usize {
        offset - l.placements[QKECCAK].offset + self.column(QKECCAK)[0].dst
    }

    /// Every claim on the full stack as a claim on the compact one, in the same order.
    ///
    /// A claim on `q_keccak` and on an untruncated column moves with its column. A claim `C(zeta) = v` on a
    /// truncated column becomes `sum_j e_j * piece_j(zeta[..b_j]) = v + pad * (1 + sum_j e_j)`
    /// (the field has characteristic 2, so `+` is the `-` of the design), with
    /// `e_j = prod_{i >= b_j} (bit_i(off_j) ? zeta_i : 1 + zeta_i)`.
    pub(super) fn map_claims(
        &self,
        l: &super::Layout,
        claims: Vec<SlotClaim>,
        mut ring: RingSwitchOpen,
    ) -> (Vec<SlotClaim>, RingSwitchOpen) {
        ring.offset = self.qkeccak_offset(l, ring.offset);
        let qk = l.placements[QKECCAK];
        let mapped = claims
            .into_iter()
            .map(|claim| {
                let SlotClaim::Point {
                    offset,
                    low_point: zeta,
                    value,
                } = claim
                else {
                    unreachable!("the prover builds Point claims")
                };
                if (qk.offset..qk.offset + (1 << qk.n_vars)).contains(&offset) {
                    return SlotClaim::Point {
                        offset: self.qkeccak_offset(l, offset),
                        low_point: zeta,
                        value,
                    };
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
                    return SlotClaim::Point {
                        offset: whole.dst,
                        low_point: zeta,
                        value,
                    };
                }
                assert_eq!(zeta.len(), k, "a claim on a truncated column is at a full point");
                let parts: Vec<StackPart> = items
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
                    .collect();
                let weight: F128 = parts.iter().map(|part| part.scale).sum();
                let pad = F128::from(Self::pad(c));
                SlotClaim::Sum {
                    value: value + pad * (F128::ONE + weight),
                    parts,
                }
            })
            .collect();
        (mapped, ring)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pieces_follow_the_verifier_rule() {
        let cut = |k, real| pieces(k, real).collect::<Vec<_>>();
        assert_eq!(cut(19, 280940), [(0, 18), (262144, 15)]);
        assert_eq!(cut(22, 2374000), [(0, 21), (2097152, 19)]);
        assert_eq!(cut(10, 5), [(0, 10)]);
        assert_eq!(cut(7, 5), [(0, 7)]);
        assert_eq!(cut(19, 1 << 19), [(0, 19)]);
        assert_eq!(cut(19, 0), [(0, 15)]);
    }
}
