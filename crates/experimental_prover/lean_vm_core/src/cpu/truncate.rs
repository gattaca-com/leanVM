//! The compact stack: a table column or memory column is committed as the covered prefix of its real rows, cut into
//! power-of-two pieces, when the proof is truncated. Everything before the PCS open reads the full `w.q`;
//! only the commit and the open see the compact stack built here. The piece rule is the verifier's
//! (`verifier::layout::pieces`).
//!
//! `q_keccak` is pieces in every proof, a half of it (a hash table) at a time: the covered rows of a half are row
//! pieces, and each is cut again by the slot blocks of the half ([`crate::keccak::slot_blocks`]): a piece holds the
//! slots of one block for the rows of one row piece, laid out `(row, slot within the block)`. The rest of a half is its
//! public pad in the committed slots, and the slots no block holds (the padding lanes of a `KECCAK64` row) are public
//! constants on every row.

use super::layout::{hash_kappa, qkeccak_slot_start};
use super::{BYTECODE_FINAL_COUNTERS, MEMORY_FINAL_COUNTERS, ProveError, QKECCAK, from_scalar, schema, to_scalar};
use crate::keccak::public_slot;
use crate::pcs::{self, RingSwitchClaim, RingSwitchOpen, StackClaim, StackPart};
use crate::tables;
use crate::witness::{self, Placement, StackShape};
use primitives::field::{F64, F128};
use verifier::Qt;

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

/// The pieces' digits of a column of `2^k` rows whose first `real` rows are real: bit `j` of the result says a piece
/// of `2^(k - j)` rows, `j = 0` being the whole column.
pub fn piece_digits(k: usize, real: usize) -> [bool; 5] {
    let mut digits = [false; 5];
    for (_, b) in verifier::pieces(k, real as u64) {
        digits[k - b] = true;
    }
    digits
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
    /// `q_keccak`'s items as the verifier's pieces, at their offsets in this stack.
    pub(super) qt: Qt,
    /// The memory cells the memory columns' pieces cover.
    mem_count: usize,
    /// Whether the memory and the table columns are cut: the rate word announces it.
    truncated: bool,
}

impl Compact {
    /// The items for a proof of `l` whose memory has `mem_used` cells in use. `truncated` cuts the memory and the table
    /// columns to their covered rows. `q_keccak` is pieces either way.
    pub(super) fn new(l: &super::Layout, mem_used: usize, truncated: bool) -> Result<Self, ProveError> {
        let sch = schema();
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
                mem_used
            } else {
                l.rows[(0..tables::N_TABLES)
                    .rev()
                    .find(|&t| c >= sch.base[t])
                    .expect("a table column")]
            };
            items.extend(
                verifier::pieces(p.n_vars, real as u64)
                    .into_iter()
                    .map(|(off, b)| Item {
                        col: c,
                        half: 0,
                        block: (0, 0),
                        off: off as usize,
                        b,
                        dst: 0,
                    }),
            );
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
        let shapes = verifier::qt_piece_shapes(kappa, &l.rows.map(|rows| rows as u64));
        items.extend(shapes.iter().map(|&(half, block, off, b)| Item {
            col: REGION,
            half,
            block,
            off: off as usize,
            b,
            dst: 0,
        }));
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
        let offsets = items[n_columns..].iter().map(|it| it.dst as u64).collect();
        Ok(Self {
            qt: Qt::new(kappa, &shapes, offsets),
            items,
            starts,
            n_columns,
            shape,
            placed,
            mem_count: mem_used,
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

    /// The public value of a truncated column past its prefix.
    fn pad(c: usize) -> F64 {
        if c == MEMORY_FINAL_COUNTERS {
            F64::ONE
        } else {
            F64::ZERO
        }
    }

    /// The compact stack: each item copied from the full stack `q`, the tail zero.
    /// Asserts first that every truncated column holds its pad past the prefix, and so does each slot of `q_keccak`, a
    /// column of `2^kappa` rows: its half's pad past the half's covered rows, or a public slot's constant on all rows.
    pub(super) fn copy(&self, q: &[F64], l: &super::Layout) -> Vec<F64> {
        let check = |words: &[F64], end: usize, pad: F64, what: std::fmt::Arguments| {
            if end < words.len() {
                assert!(
                    words[end] == pad && words[words.len() - 1] == pad,
                    "{what} is not its pad past row {end}"
                );
                debug_assert!(words[end..].iter().all(|&x| x == pad), "{what} has data past row {end}");
            }
        };
        for (c, p) in l.placements.iter().enumerate() {
            let Some(last) = self.column(c).last() else { continue };
            let column = &q[p.offset..][..1 << p.n_vars];
            check(
                column,
                last.off + (1 << last.b),
                Self::pad(c),
                format_args!("column {c}"),
            );
        }
        let qk = l.placements[QKECCAK];
        let qk_words = &q[qk.offset..][..1 << qk.n_vars];
        // A forging prover goes on past its own check.
        #[cfg(test)]
        let checks = !super::forge::active();
        #[cfg(not(test))]
        let checks = true;
        let kappa = self.qt.log_steps;
        if checks {
            for (half, pads) in self.qt.pads.iter().enumerate() {
                let items = self.region_items().iter().filter(|it| it.half == half);
                let covered = items.map(|it| it.off + (1 << it.b)).max().expect("a half has a piece");
                for (s, &pad) in pads.iter().enumerate() {
                    let words = &qk_words[qkeccak_slot_start(kappa, half, s)..][..1 << kappa];
                    match public_slot(half, s) {
                        // The verifier fills a public slot in itself, so every row is checked, in release too.
                        Some(constant) => assert!(
                            words.iter().all(|&x| x.0 == constant),
                            "q_keccak half {half} slot {s} is public and is not its constant {constant:x}"
                        ),
                        None => check(words, covered, F64(pad), format_args!("q_keccak half {half} slot {s}")),
                    }
                }
            }
        }
        let placements: Vec<Placement> = self
            .items
            .iter()
            .map(|it| Placement {
                n_vars: it.log_len(),
                offset: it.dst,
            })
            .collect();
        // SAFETY: `split_stack` hands out one window per item, tiling `0..placed`, and zeroes the tail. The loops
        // below write every window. The zeros end at the committed lanes' end, which is where the stack ends.
        let mut compact = unsafe { witness::alloc_stack(self.shape) };
        for (it, window) in self.items.iter().zip(witness::split_stack(&mut compact, &placements)) {
            if it.col == REGION {
                // Slot `j` of the piece's block, for its rows: `2^b` words, at `j * 2^b` of the piece.
                parallel::chunks_mut(window, 1 << it.b, |j, dst| {
                    let src = &qk_words[qkeccak_slot_start(kappa, it.half, it.block.0 + j) + it.off..][..dst.len()];
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

    /// Every claim on the full stack as a claim on the compact one, in the same order, and the ring-switched claim the
    /// reduction leaves as one claim over `q_keccak`'s pieces: `sum_j c_j piece_j` has the reduction's slices less the
    /// share of what the pieces leave out, so the claim's slices are those plus the share, and each piece weighs its
    /// block by `c_j`. Nothing is sent.
    ///
    /// A claim on an untruncated column moves with its column. A claim `C(zeta) = v` on a
    /// truncated column becomes `sum_j e_j * piece_j(zeta[..b_j]) = v + pad * (1 + sum_j e_j)`
    /// (the field has characteristic 2, so `+` is the `-` of the design), with
    /// `e_j = prod_{i >= b_j} (bit_i(off_j) ? zeta_i : 1 + zeta_i)`. A claim on a slot of `q_keccak` is that of a
    /// column of `2^kappa` rows, which are `q_keccak`'s pieces of the slot's block: the pad of its half past them.
    pub(super) fn map_claims(
        &self,
        l: &super::Layout,
        claims: Vec<StackClaim>,
        mut ring: RingSwitchOpen,
    ) -> (Vec<StackClaim>, RingSwitchOpen) {
        let qk = l.placements[QKECCAK];
        let qt = &self.qt;
        let kappa = qt.log_steps;
        let [input] = <[RingSwitchClaim; 1]>::try_from(std::mem::take(&mut ring.claims)).expect("one claim");
        let point: Vec<_> = input.suffix_point.iter().map(to_scalar).collect();
        // `S' = S + pad share` (the field has characteristic 2) is `sum_j c_j s_j`, `s_j` the slices of piece `j`.
        let slices = ring.s_hat_v.take().expect("the reduction sends its slices");
        let share = qt.pad_share(&point).into_iter().map(from_scalar);
        ring.s_hat_v = Some(slices.iter().zip(share).map(|(&s, pad)| s + pad).collect());
        ring.offset = 0;
        ring.claims = (qt.pieces.iter().zip(qt.ring_split(&point)))
            .map(|(piece, (suffix_point, scale))| RingSwitchClaim {
                offset: piece.placement.index as usize,
                suffix_point: suffix_point.into_iter().map(from_scalar).collect(),
                scale: from_scalar(scale),
            })
            .collect();
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
                    let (zeta, value) = (zeta.iter().map(to_scalar).collect::<Vec<_>>(), to_scalar(&value));
                    let (parts, value) = if zeta.len() <= kappa {
                        // A slot claim: the slot's column of its half, at the rows of its table, on the pieces of
                        // the slot's block.
                        let rel = offset - qk.offset;
                        let (half, slot) = ((rel >> kappa) & (crate::keccak::TABLES - 1), rel >> (kappa + 1));
                        qt.slot_claim(half, slot, &zeta, value)
                            .expect("a claim on a committed slot")
                    } else {
                        // The Keccak output claim, at the whole of `q_keccak`.
                        assert!(
                            offset == qk.offset && zeta.len() == kappa + crate::keccak::TABLE_BITS + 6,
                            "the output claim covers q_keccak"
                        );
                        qt.output_claim(&zeta, value)
                    };
                    let part = |(piece, at, scale): (&verifier::QtPiece, Vec<_>, _)| StackPart {
                        offset: piece.placement.index as usize,
                        low_point: at.into_iter().map(from_scalar).collect(),
                        scale: from_scalar(scale),
                    };
                    return StackClaim::Sum {
                        value: from_scalar(value),
                        parts: parts.into_iter().map(part).collect(),
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

    /// The compact stack has exactly the committed lanes' words, and holds the full stack's pieces: each item of a
    /// column, each piece of `q_keccak` slot by slot over its block, then zeros. A `KECCAK64` half commits 48 words a covered
    /// row.
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
            for truncated in [false, true] {
                let compact = Compact::new(&layout, 1 << log_mem, truncated).expect("a stack in range");
                // A column is its pad past its pieces, which hold random words; the memory is all real.
                let mut q = vec![F64::ZERO; layout.shape.committed_len()];
                for it in compact.items[..compact.n_columns].iter() {
                    let words = &mut q[layout.placements[it.col].offset + it.off..][..1 << it.b];
                    words.iter_mut().for_each(|w| *w = F64(rng.next_u64()));
                }
                let qk = layout.placements[QKECCAK];
                let kappa = compact.qt.log_steps;
                for half in 0..verifier::layout::HASH_TABLES.len() {
                    for (s, &pad) in compact.qt.pads[half].iter().enumerate() {
                        let start = qk.offset + qkeccak_slot_start(kappa, half, s);
                        q[start..][..1 << kappa].fill(public_slot(half, s).map_or(F64(pad), F64));
                    }
                }
                for it in compact.region_items() {
                    for s in it.block.0..it.block.0 + (1 << it.block.1) {
                        if public_slot(it.half, s).is_some() {
                            continue;
                        }
                        let start = qk.offset + qkeccak_slot_start(kappa, it.half, s) + it.off;
                        q[start..][..1 << it.b]
                            .iter_mut()
                            .for_each(|w| *w = F64(rng.next_u64()));
                    }
                }
                let stack = compact.copy(&q, &layout);
                assert_eq!(stack.len(), compact.shape.committed_len(), "{keccak_rows} Keccak rows");
                let zero = |words: &[F64]| words.iter().all(|&w| w == F64::ZERO);
                for it in &compact.items {
                    if it.col == REGION {
                        for j in 0..1 << it.block.1 {
                            let src = &q[qk.offset + qkeccak_slot_start(kappa, it.half, it.block.0 + j) + it.off..]
                                [..1 << it.b];
                            let dst = &stack[it.dst + (j << it.b)..][..1 << it.b];
                            assert!(
                                dst == src,
                                "piece of half {} at {}, slot {}",
                                it.half,
                                it.off,
                                it.block.0 + j
                            );
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
                // A `KECCAK64` half commits 48 words per covered row, a `KECCAK` half 64.
                let words = |half: usize| {
                    compact
                        .region_items()
                        .iter()
                        .filter(|it| it.half == half)
                        .map(|it| 1usize << it.log_len())
                        .sum::<usize>()
                };
                let covered = |half: usize| {
                    let first = crate::keccak::slot_blocks(half)[0];
                    (compact.region_items().iter())
                        .filter(|it| it.half == half && it.block == first)
                        .map(|it| 1usize << it.b)
                        .sum::<usize>()
                };
                assert_eq!(
                    words(crate::keccak::KECCAK_HALF),
                    64 * covered(crate::keccak::KECCAK_HALF)
                );
                assert_eq!(
                    words(crate::keccak::KECCAK64_HALF),
                    48 * covered(crate::keccak::KECCAK64_HALF)
                );
                assert_eq!(compact.announced().is_some(), truncated);
            }
        }
    }
}
