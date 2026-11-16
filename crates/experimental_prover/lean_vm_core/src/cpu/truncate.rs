//! Truncated columns: a table column or memory column is committed as the covered prefix of its
//! real rows, cut into power-of-two pieces. Everything before the PCS open reads the full `w.q`;
//! only the commit and the open see the compact stack built here. The piece rule is the
//! verifier's (`verifier::layout::pieces`).
//!
//! `q_keccak` is cut the same way, a half of it (a hash table) at a time: the covered rows of a half are pieces,
//! each holding every slot of its rows, laid out `(row, slot)`. The rest of a half is its public pad.

use super::layout::{hash_kappa, qkeccak_slot_start};
use super::{BYTECODE_FINAL_COUNTERS, MEMORY_FINAL_COUNTERS, ProveError, QKECCAK, schema};
use crate::pcs::{self, RingSwitchClaim, RingSwitchOpen, StackClaim, StackPart};
use crate::tables;
use crate::witness::{self, Placement, StackShape};
use primitives::field::{F64, F128};
use primitives::hash::{PI, RC, RHO, STATE_LANES};

/// A column keeps its rows to within `2^TRUNC_D` of its real count, in pieces of at least `2^MIN_PIECE_LOG` rows.
const TRUNC_D: usize = 4;
const MIN_PIECE_LOG: usize = 10;

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

/// One stacked piece of a column: `2^b` rows from row `off`, at `dst` in the compact stack. A piece of `q_keccak`
/// (`col` is [`QKECCAK`], `half` its table) is all 64 slots of its rows, `2^(b + 6)` words.
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
    fn log_len(&self, region: bool) -> usize {
        self.b + if region { 6 } else { 0 }
    }
}

/// Keccak-f's first `n` rounds on `a`.
pub(super) fn keccak_rounds(a: &mut [u64; STATE_LANES], n: usize) {
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

/// What a padding step of table half `half` holds in each slot of `q_keccak`: its input state in slots `0..25`, the
/// state after the first `keccak::DEPTH` rounds on it in slots `25..50`, and zero in the others.
pub fn qt_pad(half: usize) -> [u64; 64] {
    let mut pad = [0u64; 64];
    let mut state = crate::keccak::pad_inputs()[half];
    pad[..STATE_LANES].copy_from_slice(&state);
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

/// `q_keccak` as pieces: those of each half are items of the compact stack, after the columns'.
struct Region {
    /// `log2` of the steps each table is completed to.
    kappa: usize,
    /// What a padding step holds in each slot of each half.
    pads: [[F64; 64]; verifier::layout::HASH_TABLES.len()],
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

    /// `sum_s pad[half][s] eq(bits(s), slot)`.
    fn pad_mle(&self, half: usize, slot: &[F128]) -> F128 {
        (0..64).map(|s| F128::from(self.pads[half][s]) * eq_bits(s, slot)).sum()
    }

    /// Asserts that `q_keccak` holds each half's pad past its covered rows.
    fn check_pad(&self, qk: &[F64], pieces: &[Item]) {
        for half in 0..verifier::layout::HASH_TABLES.len() {
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
                let start = qkeccak_slot_start(self.kappa, half, s);
                let col = &qk[start..][..1 << self.kappa];
                assert!(
                    col[col.len() - 1] == want && col[covered] == want,
                    "q_keccak half {half} slot {s} does not end in the pad: held {:x} and {:x}, expected {:x}",
                    col[covered].0,
                    col[col.len() - 1].0,
                    want.0
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
            let real = if c == BYTECODE_FINAL_COUNTERS || c == QKECCAK {
                1 << p.n_vars
            } else if c <= MEMORY_FINAL_COUNTERS {
                mem_used
            } else {
                l.rows[(0..tables::N_TABLES)
                    .rev()
                    .find(|&t| c >= sch.base[t])
                    .expect("a table column")]
            };
            items.extend(pieces(p.n_vars, real).map(|(off, b)| Item {
                col: c,
                half: 0,
                off,
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
        let kappas_of = |items: &[Item]| -> (Vec<usize>, StackShape) {
            let kappas: Vec<Option<usize>> = items.iter().map(|it| Some(it.log_len(it.col == REGION))).collect();
            let (placements, shape) = witness::placements_of(&kappas);
            (placements.iter().map(|p| p.offset).collect(), shape)
        };
        // The columns' items, `q_keccak` whole.
        let (mut dsts, mut shape) = kappas_of(&items);
        let mut placed = items.iter().map(|it| 1usize << it.b).sum();
        // QT: `q_keccak` as the pieces of its halves, in place of its column's item.
        let mut qt_items: Vec<Item> = items.iter().copied().filter(|it| it.col != QKECCAK).collect();
        let qt_columns = qt_items.len();
        for (half, &table) in verifier::layout::HASH_TABLES.iter().enumerate() {
            qt_items.extend(pieces(kappa, l.rows[table]).map(|(off, b)| Item {
                col: REGION,
                half,
                off,
                b,
                dst: 0,
            }));
        }
        let (qt_dsts, qt_shape) = kappas_of(&qt_items);
        let mut region = None;
        let mut n_columns = n_columns;
        if qt_shape.committed_len() < shape.committed_len() {
            let pads = [0, 1].map(|half| qt_pad(half).map(F64));
            region = Some(Region { kappa, pads });
            // The columns' starts: `q_keccak`'s column has no item now.
            starts = (0..=l.placements.len())
                .map(|c| qt_items[..qt_columns].iter().filter(|it| it.col < c).count())
                .collect();
            placed = qt_items.iter().map(|it| 1usize << it.log_len(it.col == REGION)).sum();
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
            mem_count: mem_used,
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

    /// Test-only forgery: flips the first word of `q_keccak`'s second piece in the compact stack `q`.
    #[cfg(test)]
    pub(super) fn flip_piece_word(&self, q: &mut [F64]) {
        let piece = self.region_items().get(1).expect("q_keccak has a second piece");
        q[piece.dst].0 ^= 1;
    }

    /// The public value of a truncated column past its prefix.
    fn pad(c: usize) -> F64 {
        if c == MEMORY_FINAL_COUNTERS {
            F64::ONE
        } else {
            F64::ZERO
        }
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
        // A forging prover goes on past its own check.
        #[cfg(test)]
        let checks = !super::forge::active();
        #[cfg(not(test))]
        let checks = true;
        if let (Some(r), true) = (&self.region, checks) {
            r.check_pad(qk_words, self.region_items());
        }
        let placements: Vec<Placement> = self
            .items
            .iter()
            .map(|it| Placement {
                n_vars: it.log_len(it.col == REGION),
                offset: it.dst,
            })
            .collect();
        // SAFETY: `split_stack` hands out one window per item, tiling `0..placed`, and zeroes the tail. The loops
        // below write every window. The zeros end at the committed lanes' end, which is where the stack ends.
        let mut compact = unsafe { witness::alloc_stack(self.shape) };
        for (it, window) in self.items.iter().zip(witness::split_stack(&mut compact, &placements)) {
            if it.col == REGION {
                // Slot `s` of the piece's rows: `2^b` words, at `s * 2^b` of the piece.
                let kappa = self.region.as_ref().expect("a region").kappa;
                parallel::chunks_mut(window, 1 << it.b, |s, dst| {
                    let src = &qk_words[qkeccak_slot_start(kappa, it.half, s) + it.off..][..dst.len()];
                    // The fence lands when the stream drops, before this task returns.
                    primitives::stream::Stream::new().copy(dst, src);
                });
            } else {
                let src = &q[l.placements[it.col].offset + it.off..][..window.len()];
                parallel::chunks_mut(window, COPY_CHUNK, |i, chunk| {
                    // The fence lands when the stream drops, before this task returns.
                    primitives::stream::Stream::new().copy(chunk, &src[i * COPY_CHUNK..][..chunk.len()]);
                });
            }
        }
        compact
    }

    /// Where a full-stack offset inside `q_keccak` sits in the compact stack, when `q_keccak` is one item.
    fn qkeccak_offset(&self, l: &super::Layout, offset: usize) -> usize {
        offset - l.placements[QKECCAK].offset + self.column(QKECCAK)[0].dst
    }

    /// Every claim on the full stack as a claim on the compact one, in the same order, and the ring-switched claim the
    /// reduction leaves as one claim over `q_keccak`'s pieces: `sum_j c_j piece_j` has the reduction's slices less the
    /// pad's share, so the claim's slices are those plus the share, and each piece weighs its block by `c_j`.
    /// Nothing is sent.
    ///
    /// A claim on an untruncated column moves with its column. A claim `C(zeta) = v` on a
    /// truncated column becomes `sum_j e_j * piece_j(zeta[..b_j]) = v + pad * (1 + sum_j e_j)`
    /// (the field has characteristic 2, so `+` is the `-` of the design), with
    /// `e_j = prod_{i >= b_j} (bit_i(off_j) ? zeta_i : 1 + zeta_i)`. A claim on a slot of `q_keccak` is that of a
    /// column of `2^kappa` rows, which are `q_keccak`'s pieces when it is: the pad of its half past its pieces.
    pub(super) fn map_claims(
        &self,
        l: &super::Layout,
        claims: Vec<StackClaim>,
        mut ring: RingSwitchOpen,
    ) -> (Vec<StackClaim>, RingSwitchOpen) {
        let qk = l.placements[QKECCAK];
        if let Some(r) = &self.region {
            let [input] = <[RingSwitchClaim; 1]>::try_from(std::mem::take(&mut ring.claims)).expect("one claim");
            let (row, rest) = input.suffix_point.split_at(r.kappa);
            let (table, slot) = rest.split_at(crate::keccak::TABLE_BITS);
            let pieces = self.region_items();
            let mut pads = vec![F128::ZERO; 64];
            for half in 0..verifier::layout::HASH_TABLES.len() {
                let u = eq_bits(half, table) * r.uncovered(pieces, half, row);
                let eq_slot: Vec<F128> = (0..64).map(|s| eq_bits(s, slot)).collect();
                for (i, pad) in pads.iter_mut().enumerate() {
                    let share: F128 = (0..64)
                        .filter(|&s| r.pads[half][s].0 >> i & 1 == 1)
                        .map(|s| eq_slot[s])
                        .sum();
                    *pad += u * share;
                }
            }
            // `S' = S + pad share` (the field has characteristic 2) is `sum_j c_j s_j`, `s_j` the slices of piece `j`.
            let slices = ring.s_hat_v.take().expect("the reduction sends its slices");
            ring.s_hat_v = Some(slices.iter().zip(&pads).map(|(&s, &pad)| s + pad).collect());
            ring.offset = 0;
            ring.claims = pieces
                .iter()
                .map(|piece| RingSwitchClaim {
                    offset: piece.dst,
                    suffix_point: [&row[..piece.b], slot].concat(),
                    scale: eq_bits(piece.half, table) * r.weight(piece, row),
                })
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
                    value,
                } = claim
                else {
                    unreachable!("the prover builds Point claims")
                };
                if (qk.offset..qk.offset + (1 << qk.n_vars)).contains(&offset) {
                    if let Some(r) = &self.region {
                        let pieces = self.region_items();
                        let mut parts = Vec::new();
                        let mut value = value;
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
                            value += F128::from(r.pads[half][slot]) * r.uncovered(pieces, half, &row);
                        } else {
                            // The Keccak output claim, at the whole of `q_keccak`.
                            assert!(
                                offset == qk.offset && zeta.len() == r.kappa + crate::keccak::TABLE_BITS + 6,
                                "the output claim covers q_keccak"
                            );
                            let (row, rest) = zeta.split_at(r.kappa);
                            let (table, slot) = rest.split_at(crate::keccak::TABLE_BITS);
                            for half in 0..verifier::layout::HASH_TABLES.len() {
                                let eq = eq_bits(half, table);
                                value += eq * r.pad_mle(half, slot) * r.uncovered(pieces, half, row);
                                for piece in pieces.iter().filter(|p| p.half == half) {
                                    parts.push(StackPart {
                                        offset: piece.dst,
                                        low_point: [&row[..piece.b], slot].concat(),
                                        scale: eq * r.weight(piece, row),
                                    });
                                }
                            }
                        }
                        return StackClaim::Sum { value, parts };
                    }
                    return StackClaim::Point {
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
                    return StackClaim::Point {
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
                StackClaim::Sum {
                    value: value + pad * (F128::ONE + weight),
                    parts,
                }
            })
            .collect();
        (mapped, ring)
    }
}

/// The column of an item that is a piece of `q_keccak`'s halves: no real column.
const REGION: usize = usize::MAX;

/// Words per task of the compact copy.
const COPY_CHUNK: usize = 1 << 16;

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

    /// The compact stack has exactly the committed lanes' words, and holds the full stack's pieces: each item of a
    /// column, each piece of `q_keccak` slot by slot, then zeros. `q_keccak` is a column's items or pieces.
    #[test]
    fn a_compact_stack_holds_the_pieces_up_to_its_committed_lanes() {
        let program = isa::Program::from(isa::CompiledProgram {
            prog: vec![super::super::Op::Xor { a: 0, b: 0, c: 0 }; 16],
            hints: Default::default(),
            main_frame: 64,
            fn_ranges: Vec::new(),
            src_lines: Vec::new(),
        });
        let log_mem = 20;
        let mut rng = Rng::new(5);
        let mut regions = 0;
        for (keccak_rows, keccak64_rows) in [
            (1, 1),
            (5000, 70_000),
            ((1 << 14) + 1, 1 << 17),
            ((1 << 15) + 1, (1 << 16) + 5),
        ] {
            let mut rows = [0; tables::N_TABLES];
            rows[verifier::layout::OP_KECCAK] = keccak_rows;
            rows[verifier::layout::OP_KECCAK64] = keccak64_rows;
            let layout = crate::cpu::layout_of(
                crate::log2_strict_usize(program.prog.len()),
                log_mem,
                rows,
                [F128::ZERO; 2],
            );
            let compact = Compact::new(&layout, 1 << log_mem).expect("a stack in range");
            // A column is its pad past its pieces, which hold random words; the memory is all real.
            let mut q = vec![F64::ZERO; layout.shape.committed_len()];
            for it in compact.items.iter().filter(|it| it.col != REGION) {
                let words = &mut q[layout.placements[it.col].offset + it.off..][..1 << it.b];
                words.iter_mut().for_each(|w| *w = F64(rng.next_u64()));
            }
            let qk = layout.placements[QKECCAK];
            if let Some(r) = &compact.region {
                regions += 1;
                for half in 0..verifier::layout::HASH_TABLES.len() {
                    for (s, &pad) in r.pads[half].iter().enumerate() {
                        let start = qk.offset + qkeccak_slot_start(r.kappa, half, s);
                        q[start..][..1 << r.kappa].fill(pad);
                    }
                }
                for it in compact.region_items() {
                    for s in 0..64 {
                        let start = qk.offset + qkeccak_slot_start(r.kappa, it.half, s) + it.off;
                        q[start..][..1 << it.b]
                            .iter_mut()
                            .for_each(|w| *w = F64(rng.next_u64()));
                    }
                }
            }
            let stack = compact.copy(&q, &layout);
            assert_eq!(stack.len(), compact.shape.committed_len(), "{keccak_rows} Keccak rows");
            let zero = |words: &[F64]| words.iter().all(|&w| w == F64::ZERO);
            for it in &compact.items {
                if it.col == REGION {
                    let kappa = compact.region.as_ref().expect("a region").kappa;
                    for s in 0..64 {
                        let src = &q[qk.offset + qkeccak_slot_start(kappa, it.half, s) + it.off..][..1 << it.b];
                        let dst = &stack[it.dst + (s << it.b)..][..1 << it.b];
                        assert!(dst == src, "piece of half {} at {}, slot {s}", it.half, it.off);
                    }
                } else {
                    let src = &q[layout.placements[it.col].offset + it.off..][..1 << it.b];
                    assert!(stack[it.dst..][..1 << it.b] == *src, "item of column {}", it.col);
                }
            }
            assert!(
                zero(&stack[compact.placed..]),
                "the tail past the items, {keccak_rows} Keccak rows"
            );
        }
        assert!(regions > 0, "some layout makes q_keccak pieces");
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

    /// The pad of a half is its padding step: the state it permutes in slots `0..25`, the middle state next.
    #[test]
    fn qt_pad_is_the_padding_step() {
        let [zero, keccak64] = [qt_pad(0), qt_pad(1)];
        assert!(zero[..25].iter().chain(&zero[50..]).all(|&w| w == 0));
        assert_ne!(zero[25], 0);
        assert_eq!(keccak64[..25], crate::keccak::keccak64_pad());
        assert!(keccak64[50..].iter().all(|&w| w == 0));
    }
}
