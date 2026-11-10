//! Truncated columns: a table column or memory column is committed as the covered prefix of its
//! real rows, cut into power-of-two pieces. Everything before the PCS open reads the full `w.q`;
//! only the commit and the open see the compact stack built here. The piece rule is the
//! verifier's (`verifier::layout::pieces`).

use super::{BFCNT, MFCNT, ProveError, QKECCAK, schema};
use crate::pcs::{self, RingSwitchOpen, SlotClaim, StackPart};
use crate::tables;
use crate::witness::{self, Placement, StackShape};
use primitives::field::{F64, F128};
use primitives::hash::{PI, RC, RHO, STATE_LANES};

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

/// Keccak-f's first `n` rounds on `a`.
fn keccak_rounds(a: &mut [u64; STATE_LANES], n: usize) {
    for &rc in &RC[..n] {
        let c: [u64; 5] = std::array::from_fn(|x| a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20]);
        let mut b = [0u64; STATE_LANES];
        for i in 0..STATE_LANES {
            let d = c[(i % 5 + 4) % 5] ^ c[(i % 5 + 1) % 5].rotate_left(1);
            b[PI[i]] = (a[i] ^ d).rotate_left(RHO[i]);
        }
        for i in 0..STATE_LANES {
            let row = i / 5 * 5;
            a[i] = b[i] ^ (!b[row + (i % 5 + 1) % 5] & b[row + (i % 5 + 2) % 5]);
        }
        a[0] ^= rc;
    }
}

/// What a QT padding step holds in each slot of `q_keccak`: zero but for slots `25..50`,
/// the state after the first `keccak::DEPTH` rounds on the zero state.
pub fn qt_pad() -> [u64; 64] {
    let mut pad = [0u64; 64];
    let mut state = [0u64; STATE_LANES];
    keccak_rounds(&mut state, crate::keccak::DEPTH);
    pad[STATE_LANES..2 * STATE_LANES].copy_from_slice(&state);
    pad
}

/// The pieces' digits of a column of `2^k` rows whose first `real` rows are real: bit `j` of the result says a piece
/// of `2^(k - j)` rows, `j = 0` being the whole column.
pub fn piece_digits(k: usize, real: usize) -> [bool; 5] {
    let mut digits = [false; 5];
    for (_, b) in pieces(k, real) {
        digits[k - b] = true;
    }
    digits
}

/// `eq(bits(i), point)`, bit 0 first.
fn eq_bits(i: usize, point: &[F128]) -> F128 {
    point
        .iter()
        .enumerate()
        .map(|(j, &x)| if i >> j & 1 == 1 { x } else { F128::ONE + x })
        .product()
}

/// `q_keccak` in the QT layout: its chunks `< c` are copied to `o`, chunk `ch` of slot `s` at `g * (s + 64 * ch)`.
/// The chunks past them equal the public pad `pad[slot]`.
struct Region {
    tau: usize,
    g_log: usize,
    /// Chunks of `2^g_log` steps that are committed.
    c: usize,
    /// Words the other items occupy, and the region's offset.
    s_o: usize,
    o: usize,
    /// Length of the stack the opener gets: the region's end, to whole lanes.
    ext: usize,
    pad: [F64; 64],
}

impl Region {
    /// `1 + sum_{i < c} eq(bits(i), rho_chunk)`, the weight of the pad's chunks.
    fn u(&self, rho_chunk: &[F128]) -> F128 {
        F128::ONE + (0..self.c).map(|i| eq_bits(i, rho_chunk)).sum::<F128>()
    }

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
        let mut bad = Vec::new();
        for (s, &want) in self.pad.iter().enumerate() {
            let col = &qk[s << self.tau..][..1 << self.tau];
            if col[col.len() - 1] != want {
                bad.push((s, col[col.len() - 1].0, want.0));
            }
            debug_assert!(
                col[self.c << self.g_log..].iter().all(|&x| x == want),
                "slot {s} has data past the pad"
            );
        }
        assert!(
            bad.is_empty(),
            "q_keccak does not end in the pad; (slot, held, expected): {bad:x?}"
        );
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
        let kappas = |skip: Option<usize>| -> Vec<Option<usize>> {
            items.iter().map(|it| (Some(it.col) != skip).then_some(it.b)).collect()
        };
        let (placements, mut shape) = witness::placements_of(&kappas(None));
        let mut dsts: Vec<usize> = placements.iter().map(|p| p.offset).collect();
        let mut placed = items.iter().map(|it| 1 << it.b).sum();
        // QT: `q_keccak` as the region `R`, after the other items.
        let kt = tables::KECCAK_TABLE;
        let (tau, rows) = (l.taus[kt], l.rows[kt]);
        assert_eq!(
            l.placements[QKECCAK].n_vars,
            6 + tau,
            "q_keccak is 64 slots of the KECCAK table"
        );
        let g_log = tau.min(tau.saturating_sub(TRUNC_D).max(MIN_PIECE_LOG));
        let c = rows.max(1).div_ceil(1 << g_log).min(1 << (tau - g_log));
        let (offsets, s_o) = witness::stack_offsets(&kappas(Some(QKECCAK)));
        let o = s_o.next_multiple_of(1 << (6 + tau));
        let placed_qt = o + ((c * 64) << g_log);
        let mu = crate::log2_ceil_usize(o + (1 << (6 + tau))).max(pcs::MIN_STACKED_LOG);
        let qt = StackShape {
            mu,
            n_lanes: placed_qt.div_ceil(1 << (mu - pcs::LOG_BATCH)).max(1),
        };
        let mut region = None;
        if qt.committed_len() < shape.committed_len() {
            let pad = qt_pad().map(F64);
            let ext = (o + (1 << (6 + tau))).next_multiple_of(1 << (mu - pcs::LOG_BATCH));
            region = Some(Region {
                tau,
                g_log,
                c,
                s_o,
                o,
                ext,
                pad,
            });
            (dsts, shape, placed) = (offsets, qt, placed_qt);
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
            mem_count: mem_used,
        })
    }

    /// The compact layout of `l` whose stack reaches `2^floor` words, if one does.
    /// It truncates the memory to `mem_used` cells, else keeps it whole; past that it gives up.
    pub(super) fn reaching(l: &super::Layout, mem_used: usize, floor: usize) -> Result<Option<Self>, ProveError> {
        for mem_count in [mem_used, 1 << l.placements[MFCNT].n_vars] {
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
        let qk = l.placements[QKECCAK];
        let qk_words = &q[qk.offset..][..1 << qk.n_vars];
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
        // SAFETY: `split_stack` hands out one window per item, tiling `0..placed` (a QT stack's items tile
        // `0..s_o`), and zeroes the tail. The loops below write every window, and in a QT stack the gap
        // before the region, the region and the zeros past it.
        let mut compact = match &self.region {
            None => unsafe { witness::alloc_stack(self.shape) },
            Some(r) => unsafe { primitives::uninit_vec::<F64>(r.ext) },
        };
        let split = self.region.as_ref().map_or(compact.len(), |r| r.s_o);
        let (head, rest) = compact.split_at_mut(split);
        for (it, window) in items.iter().zip(witness::split_stack(head, &placements)) {
            let src = &q[l.placements[it.col].offset + it.off..][..window.len()];
            parallel::chunks_mut(window, COPY_CHUNK, |i, chunk| {
                chunk.copy_from_slice(&src[i * COPY_CHUNK..][..chunk.len()]);
            });
        }
        if let Some(r) = &self.region {
            r.check_pad(qk_words);
            let (gap, rest) = rest.split_at_mut(r.o - r.s_o);
            let (region, tail) = rest.split_at_mut((r.c * 64) << r.g_log);
            parallel::chunks_mut(gap, COPY_CHUNK, |_, chunk| chunk.fill(F64::ZERO));
            parallel::chunks_mut(tail, COPY_CHUNK, |_, chunk| chunk.fill(F64::ZERO));
            let g = 1 << r.g_log;
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
        let qk = l.placements[QKECCAK];
        if let Some(r) = &self.region {
            // Characteristic 2: the pad's share adds where the design subtracts it.
            ring.offset = r.o;
            for claim in &mut ring.claims {
                let (chunk, slot) = claim.suffix_point[r.g_log..].split_at(r.tau - r.g_log);
                let u = r.u(chunk);
                let eq_slots: Vec<F128> = (0..64).map(|s| eq_bits(s, slot)).collect();
                let slices = claim.s_hat_v.as_mut().expect("the reduction sends its slices");
                for (i, slice) in slices.iter_mut().enumerate() {
                    let share: F128 = (0..64).filter(|&s| r.pad[s].0 >> i & 1 == 1).map(|s| eq_slots[s]).sum();
                    *slice += u * share;
                }
                claim.suffix_point = r.rotate(&claim.suffix_point);
            }
        } else {
            ring.offset = self.qkeccak_offset(l, ring.offset);
        }
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
                    if let Some(r) = &self.region {
                        let tau = r.tau;
                        let (point, pad) = if zeta.len() == tau {
                            // A slot claim: the slot's bits go between the step's and the chunk's.
                            let slot = (offset - qk.offset) >> tau;
                            let bits: Vec<F128> = (0..6)
                                .map(|j| if slot >> j & 1 == 1 { F128::ONE } else { F128::ZERO })
                                .collect();
                            let (steplo, chunk) = zeta.split_at(r.g_log);
                            ([steplo, &bits, chunk].concat(), F128::from(r.pad[slot]) * r.u(chunk))
                        } else {
                            // The Keccak output claim, at the whole region.
                            assert!(
                                offset == qk.offset && zeta.len() == 6 + tau,
                                "the output claim covers q_keccak"
                            );
                            let chunk = &zeta[r.g_log..tau];
                            let phat: F128 = (0..64).map(|s| F128::from(r.pad[s]) * eq_bits(s, &zeta[tau..])).sum();
                            (r.rotate(&zeta), phat * r.u(chunk))
                        };
                        return SlotClaim::Point {
                            offset: r.o,
                            low_point: point,
                            value: value + pad,
                        };
                    }
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

    /// A xorshift generator, enough for test data.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self((seed * 0x9E37_79B9_7F4A_7C15) | 1)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn ext_vec(&mut self, n: usize) -> Vec<F128> {
            (0..n).map(|_| F128::new(self.next_u64(), self.next_u64())).collect()
        }
    }

    fn mle(w: &[F64], point: &[F128]) -> F128 {
        w.iter()
            .enumerate()
            .map(|(i, &x)| F128::from(x) * eq_bits(i, point))
            .sum()
    }

    #[test]
    fn rounds_match_the_permutation() {
        let mut rng = Rng::new(7);
        let mut a: [u64; STATE_LANES] = std::array::from_fn(|_| rng.next_u64());
        let mut expected = a;
        keccak_rounds(&mut a, primitives::hash::ROUNDS);
        primitives::hash::keccak_f(&mut expected);
        assert_eq!(a, expected);
    }

    #[test]
    fn rotation_and_pad_identity() {
        let (tau, g_log, c) = (12, 10, 2);
        let mut rng = Rng::new(11);
        let pad: [F64; 64] = std::array::from_fn(|_| F64(rng.next_u64()));
        let r = Region {
            tau,
            g_log,
            c,
            s_o: 0,
            o: 0,
            ext: 1 << (6 + tau),
            pad,
        };
        let g = 1 << g_log;
        let mut q = vec![F64::ZERO; 64 << tau];
        for (i, x) in q.iter_mut().enumerate() {
            *x = if i % (1 << tau) < c * g {
                F64(rng.next_u64())
            } else {
                pad[i >> tau]
            };
        }
        let mut region = vec![F64::ZERO; 64 << tau];
        for s in 0..64 {
            for ch in 0..c {
                region[g * (s + 64 * ch)..][..g].copy_from_slice(&q[(s << tau) + ch * g..][..g]);
            }
        }
        let rho = rng.ext_vec(6 + tau);
        let (chunk, slot) = rho[g_log..].split_at(tau - g_log);
        let phat: F128 = (0..64).map(|s| F128::from(pad[s]) * eq_bits(s, slot)).sum();
        assert_eq!(mle(&q, &rho), mle(&region, &r.rotate(&rho)) + phat * r.u(chunk));
    }

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
