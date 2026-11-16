//! Truncated columns: a table column or memory column is committed as the covered prefix of its real rows, cut into
//! power-of-two pieces ([`verifier::layout::pieces`]), and `q_keccak` as a region, when that commits fewer words.
//! Everything before the PCS open reads the full `w.q`; only the commit and the open see the compact stack built here.

use super::{BYTECODE_FINAL_COUNTERS, GLOBAL_COLUMN_BASES, MEMORY_FINAL_COUNTERS, ProveError, QKECCAK};
use crate::pcs::{self, RingSwitchOpen, StackClaim, StackPart};
use crate::tables;
use crate::witness::{self, Placement, StackShape};
use primitives::field::{F64, F128};
use verifier::layout::{MIN_PIECE_LOG, OP_KECCAK, TRUNC_D, pieces};

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

/// One stacked piece of a column: `2^b` rows from row `off`, at `dst` in the compact stack.
#[derive(Clone, Copy)]
struct Item {
    col: usize,
    off: usize,
    b: usize,
    dst: usize,
}

/// `q_keccak` in the QT layout: its chunks `< c` are copied to `o`, chunk `ch` of slot `s` at `g * (s + 64 * ch)`.
/// The chunks past them equal the public pad `pad[slot]`.
struct Region {
    tau: usize,
    g_log: usize,
    /// Chunks of `2^g_log` steps that are committed.
    c: usize,
    /// The region's offset: after the other items, a multiple of its length.
    o: usize,
    /// Length of the stack the opener gets: the region's end, to whole lanes.
    ext: usize,
    pad: [F64; 64],
}

impl Region {
    /// `rho = (steplo, chunk, slot)` as `(steplo, slot, chunk)`.
    fn rotate(&self, rho: &[F128]) -> Vec<F128> {
        let (steplo, rest) = rho.split_at(self.g_log);
        let (chunk, slot) = rest.split_at(self.tau - self.g_log);
        [steplo, slot, chunk].concat()
    }

    /// Asserts that `q_keccak` holds the pad past its covered chunks.
    fn check_pad(&self, qk: &[F64]) {
        if self.c << self.g_log == 1 << self.tau {
            return;
        }
        for (s, &want) in self.pad.iter().enumerate() {
            let col = &qk[s << self.tau..][..1 << self.tau];
            assert!(col[col.len() - 1] == want, "slot {s} does not end in the pad");
            debug_assert!(
                col[self.c << self.g_log..].iter().all(|&x| x == want),
                "slot {s} has data past the pad"
            );
        }
    }
}

/// The compact stack's layout: the items of every committed column, in increasing column then offset.
pub(super) struct Compact {
    items: Vec<Item>,
    /// Column `c`'s items are `items[starts[c]..starts[c + 1]]`, none for a virtual column.
    starts: Vec<usize>,
    pub(super) shape: StackShape,
    /// Words the items occupy.
    pub(super) placed: usize,
    /// The QT layout, when it commits fewer words than keeping `q_keccak` whole: then the `QKECCAK` item is unused.
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
                off: off as usize,
                b,
                dst: 0,
            }));
        }
        starts.push(items.len());
        let kappas = |skip: Option<usize>| -> Vec<Option<usize>> {
            items.iter().map(|it| (Some(it.col) != skip).then_some(it.b)).collect()
        };
        let (placements, mut shape) = witness::placements_of(&kappas(None));
        let mut dsts: Vec<usize> = placements.iter().map(|p| p.offset).collect();
        let mut placed = items.iter().map(|it| 1 << it.b).sum();
        // QT: `q_keccak` as the region `R`, after the other items.
        let (tau, rows) = (l.taus[OP_KECCAK], l.rows[OP_KECCAK]);
        assert_eq!(
            l.placements[QKECCAK].n_vars,
            6 + tau,
            "q_keccak is 64 slots of the KECCAK table"
        );
        let g_log = tau.min(tau.saturating_sub(TRUNC_D).max(MIN_PIECE_LOG));
        let c = rows.max(1).div_ceil(1 << g_log).min(1 << (tau - g_log));
        let (others, _) = witness::placements_of(&kappas(Some(QKECCAK)));
        let s_o: usize = items.iter().filter(|it| it.col != QKECCAK).map(|it| 1 << it.b).sum();
        let o = s_o.next_multiple_of(1 << (6 + tau));
        let placed_qt = o + ((c * 64) << g_log);
        let mu = crate::log2_ceil_usize(o + (1 << (6 + tau))).max(pcs::MIN_STACKED_LOG);
        let qt = StackShape {
            mu,
            n_lanes: placed_qt.div_ceil(1 << (mu - pcs::LOG_BATCH)).max(1),
        };
        let mut region = None;
        if qt.committed_len() < shape.committed_len() {
            region = Some(Region {
                tau,
                g_log,
                c,
                o,
                ext: (o + (1 << (6 + tau))).next_multiple_of(1 << (mu - pcs::LOG_BATCH)),
                pad: qt_pad().map(F64),
            });
            (dsts, shape, placed) = (others.iter().map(|p| p.offset).collect(), qt, placed_qt);
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

    /// What the rate word announces: the memory count, and whether q_keccak is a region.
    pub(super) fn announced(&self) -> (usize, bool) {
        (self.mem_count, self.region.is_some())
    }

    fn column(&self, c: usize) -> &[Item] {
        &self.items[self.starts[c]..self.starts[c + 1]]
    }

    /// The compact stack: each item copied from the full stack `q`, the tail zero. A column past its prefix must hold
    /// its public value, which is what the verifier takes it to be: the memory counters' `1`, zero elsewhere.
    pub(super) fn copy(&self, q: &[F64], l: &super::Layout) -> Vec<F64> {
        for (c, p) in l.placements.iter().enumerate() {
            let Some(last) = self.column(c).last() else { continue };
            let end = last.off + (1 << last.b);
            if end < 1 << p.n_vars {
                let column = &q[p.offset..][..1 << p.n_vars];
                let pad = if c == MEMORY_FINAL_COUNTERS {
                    F64::ONE
                } else {
                    F64::ZERO
                };
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
        let items: Vec<&Item> = self
            .items
            .iter()
            .filter(|it| self.region.is_none() || it.col != QKECCAK)
            .collect();
        let placements: Vec<Placement> = items
            .iter()
            .map(|it| Placement {
                n_vars: it.b,
                offset: it.dst,
            })
            .collect();
        let len = self.region.as_ref().map_or(self.shape.committed_len(), |r| r.ext);
        let mut compact = parallel::filled(len, F64::ZERO);
        for (it, window) in items.iter().zip(witness::split_stack(&mut compact, &placements)) {
            let src = &q[l.placements[it.col].offset + it.off..][..window.len()];
            window.copy_from_slice(src);
        }
        if let Some(r) = &self.region {
            let qk_words = &q[qk.offset..][..1 << qk.n_vars];
            r.check_pad(qk_words);
            let g = 1 << r.g_log;
            let region = &mut compact[r.o..][..(r.c * 64) << r.g_log];
            parallel::chunks_mut(region, 64 * g, |ch, block| {
                for (s, dst) in block.chunks_exact_mut(g).enumerate() {
                    dst.copy_from_slice(&qk_words[(s << r.tau) + ch * g..][..g]);
                }
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
    /// A claim on `q_keccak` and on an untruncated column moves with its column. A claim `C(zeta)` on a truncated
    /// column is the weighted sum `sum_j e_j * piece_j(zeta[..b_j])`, with `e_j = prod_{i >= b_j} (bit_i(off_j) ? zeta_i
    /// : 1 + zeta_i)`: the verifier moves the claimed value by the public pad's share, which the opening does not need.
    pub(super) fn map_claims(
        &self,
        l: &super::Layout,
        claims: Vec<StackClaim>,
        mut ring: RingSwitchOpen,
    ) -> (Vec<StackClaim>, RingSwitchOpen) {
        let qk = l.placements[QKECCAK];
        if let Some(r) = &self.region {
            ring.offset = r.o;
            ring.points = ring.points.iter().map(|point| r.rotate(point)).collect();
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
                    let point = if zeta.len() == r.tau {
                        // A slot claim: the slot's bits go between the step's and the chunk's.
                        let slot = (offset - qk.offset) >> r.tau;
                        let bits = (0..6).map(|j| if slot >> j & 1 == 1 { F128::ONE } else { F128::ZERO });
                        let (steplo, chunk) = zeta.split_at(r.g_log);
                        [steplo, &bits.collect::<Vec<_>>(), chunk].concat()
                    } else {
                        // The Keccak output claim, at the whole region.
                        assert!(
                            offset == qk.offset && zeta.len() == 6 + r.tau,
                            "the output claim covers q_keccak"
                        );
                        r.rotate(&zeta)
                    };
                    return StackClaim::Point {
                        offset: r.o,
                        low_point: point,
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
