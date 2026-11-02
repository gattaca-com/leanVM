//! `K`-valued columns stacked into one committed witness (§sec:stacking): columns laid
//! end to end, largest first at aligned offsets, into one multilinear `q` over
//! `F64`. An evaluation claim on column `i` at `ζ ∈ E` becomes the claim
//! `q̂(ζ, sel_i) = c` on the stack, where `sel_i` is the high-bit selector of
//! the column's offset.

use crate::tables::ColumnOut;
use ::pcs::stack_open::EQUAL_BLOCK_LOG;
use primitives::field::F64;

/// Where a column sits in the stacked witness. A [`Placement::VIRTUAL`] column has no
/// placement of its own: either its words are committed inside another column, and its
/// evaluation claims settle there (the KECCAK columns `q_keccak` carries), or they are not
/// committed at all and something else proves them (the KECCAK output lanes).
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    pub n_vars: usize,
    pub offset: usize,
}

impl Placement {
    pub const VIRTUAL: Placement = Placement {
        n_vars: usize::MAX,
        offset: 0,
    };

    pub fn is_virtual(&self) -> bool {
        self.n_vars == usize::MAX
    }
}

/// The committed stack's shape. `mu` is the log size everything public is derived
/// from (the claims' selector coords, the PCS level ladder), while `n_lanes` counts
/// the `2^(mu - LOG_BATCH)`-word lane blocks that actually carry data: the PCS
/// commits `committed_len()` words and never encodes or hashes the zero tail past
/// them. The verifier's view of the commitment is the same `2^mu`-word witness
/// either way; it derives `n_lanes` from the announced layout only to read the
/// pruned level-0 rows, which leave the zero lanes out.
#[derive(Clone, Copy, Debug)]
pub struct StackShape {
    pub mu: usize,
    pub n_lanes: usize,
}

impl StackShape {
    /// Words actually committed: the placed columns rounded up to a whole lane.
    pub fn committed_len(&self) -> usize {
        // Both fields are `pub`, and in release the shift below would mask an
        // underflowed amount into a plausible wrong length rather than panic.
        assert!(self.mu >= crate::pcs::LOG_BATCH, "a stack is at least one lane block");
        self.n_lanes << (self.mu - crate::pcs::LOG_BATCH)
    }
}

/// Lay `2^kappa`-sized items end to end, largest first at aligned offsets and ties
/// broken by index, returning the per-item offset and `Σ 2^kappa`. A `None`
/// kappa takes no space and its offset is meaningless. Depends only on the sizes, so
/// the verifier reconstructs the same tiling.
pub(crate) fn stack_offsets(kappas: &[Option<usize>]) -> (Vec<usize>, usize) {
    let n = kappas.len();
    let mut order: Vec<usize> = (0..n).filter(|&i| kappas[i].is_some()).collect();
    order.sort_by(|&a, &b| kappas[b].unwrap().cmp(&kappas[a].unwrap()).then(a.cmp(&b)));

    let mut offsets = vec![0usize; n];
    let mut off = 0usize;
    for &i in &order {
        let kappa = kappas[i].unwrap();
        assert!(
            kappa < usize::BITS as usize,
            "kappa {kappa} is a log-size, not a length"
        );
        offsets[i] = off;
        off += 1 << kappa;
    }
    (offsets, off)
}

/// Per-column placements (offset + n_vars) and the stack's [`StackShape`] from the
/// columns' log-sizes alone, largest-first at aligned offsets. A `None` kappa marks
/// a virtual (uncommitted) column. Depends only on lengths, so the verifier can
/// reconstruct it.
pub fn placements_of(kappas: &[Option<usize>]) -> (Vec<Placement>, StackShape) {
    let (offsets, placed) = stack_offsets(kappas);
    let placements = kappas
        .iter()
        .zip(&offsets)
        .map(|(k, &offset)| match *k {
            Some(n_vars) => Placement { n_vars, offset },
            None => Placement::VIRTUAL,
        })
        .collect();
    // Floor at the PCS minimum (WHIR's level ladder needs room); tiny
    // witnesses zero-pad up. Both sides derive this identically from the kappas.
    let mu = crate::log2_ceil_usize(placed.max(1)).max(crate::pcs::MIN_MU);
    // The columns tile from 0, so the padding is the tail: round it up to a whole
    // lane and the lanes past that are never committed at all.
    let n_lanes = placed.div_ceil(1 << (mu - crate::pcs::LOG_BATCH)).max(1);
    (placements, StackShape { mu, n_lanes })
}

/// Rows of a jagged chunk: a column's run inside one chunk of its group.
pub const CHUNK_LOG: usize = ::pcs::stack_open::CHUNK_LOG;
/// The jagged groups start at a multiple of `2^JAGGED_ALIGN_LOG` words, which every group's chunk divides.
pub const JAGGED_ALIGN_LOG: usize = 15;

/// A jagged group: a table's committed columns (or the memory's), cut into chunks of `2^CHUNK_LOG` rows. A chunk
/// holds `2^log_width` column runs one after another, the runs past the last column zero.
#[derive(Clone, Debug)]
pub struct JaggedGroup {
    /// Global column indices, in local order.
    pub columns: Vec<usize>,
    pub log_width: usize,
    pub rows: usize,
    pub offset: usize,
}

impl JaggedGroup {
    pub fn chunks(&self) -> usize {
        self.rows.div_ceil(1 << CHUNK_LOG)
    }

    pub fn words(&self) -> usize {
        self.chunks() << (CHUNK_LOG + self.log_width)
    }
}

/// Where a committed column lives in the committed (dense) stack.
#[derive(Clone, Copy, Debug)]
pub enum Slot {
    /// A whole `2^n_vars` block, as in the padded working stack.
    Padded(Placement),
    /// Local column `column` of jagged group `group`.
    Jagged { group: usize, column: usize },
}

/// The committed stack: the padded blocks by descending size, then the jagged groups by descending width.
///
/// The prover works on the padded stack ([`placements_of`]) and commits this one, a copy that leaves every row
/// past a group's chunks out. Both sides derive it from the announced sizes alone.
#[derive(Clone, Debug)]
pub struct Dense {
    pub groups: Vec<JaggedGroup>,
    pub slots: Vec<Option<Slot>>,
    pub shape: StackShape,
    pub placed: usize,
}

/// The committed stack for the columns' log sizes `kappas`: the `padded` columns as whole blocks, every column of a
/// `groups` entry (its global columns and its rows) jagged.
pub fn dense_of(kappas: &[Option<usize>], padded: &[usize], groups: Vec<(Vec<usize>, usize)>) -> Dense {
    let padded_kappas: Vec<Option<usize>> = padded.iter().map(|&c| kappas[c]).collect();
    let (padded_offsets, padded_words) = stack_offsets(&padded_kappas);
    let mut groups: Vec<JaggedGroup> = groups
        .into_iter()
        .map(|(columns, rows)| JaggedGroup {
            log_width: crate::log2_ceil_usize(columns.len()),
            columns,
            rows,
            offset: 0,
        })
        .collect();
    let mut order: Vec<usize> = (0..groups.len()).collect();
    order.sort_by_key(|&g| (std::cmp::Reverse(groups[g].log_width), g));
    let mut placed = padded_words.next_multiple_of(1 << JAGGED_ALIGN_LOG);
    for &g in &order {
        groups[g].offset = placed;
        placed += groups[g].words();
    }
    let mut slots = vec![None; kappas.len()];
    for (&column, &offset) in padded.iter().zip(&padded_offsets) {
        let n_vars = kappas[column].expect("a padded column is committed");
        slots[column] = Some(Slot::Padded(Placement { n_vars, offset }));
    }
    for (g, group) in groups.iter().enumerate() {
        for (column, &global) in group.columns.iter().enumerate() {
            slots[global] = Some(Slot::Jagged { group: g, column });
        }
    }
    let mu = crate::log2_ceil_usize(placed.max(1)).max(crate::pcs::MIN_MU);
    let n_lanes = placed.div_ceil(1 << (mu - crate::pcs::LOG_BATCH)).max(1);
    Dense {
        groups,
        slots,
        shape: StackShape { mu, n_lanes },
        placed,
    }
}

impl Dense {
    /// The committed stack from the padded one `q`, laid out by `placements` with repeat map `q_repeats`, and its
    /// own repeat map.
    ///
    /// A run of a column past its padded block (a table under one chunk) and a run past a group's columns are
    /// zero. A padded block keeps its tags; in the jagged region only the zero runs that follow another are tagged.
    pub fn gather(&self, q: &[F64], q_repeats: &[bool], placements: &[Placement]) -> (Vec<F64>, Vec<bool>) {
        const RUN: usize = 1 << CHUNK_LOG;
        let len = self.shape.committed_len();
        let mut repeats = vec![false; len >> EQUAL_BLOCK_LOG];
        // Each run of the jagged region: the padded words it copies (start, length), or none for zeros.
        let start = self.groups.iter().map(|g| g.offset).min().unwrap_or(self.placed);
        let mut runs: Vec<Option<(usize, usize)>> = vec![None; (len - start) / RUN];
        for group in &self.groups {
            for chunk in 0..group.chunks() {
                for column in 0..1 << group.log_width {
                    let at = group.offset + (chunk << (CHUNK_LOG + group.log_width)) + column * RUN;
                    let run = (at - start) / RUN;
                    match group.columns.get(column) {
                        Some(&global) => {
                            let p = placements[global];
                            let height = 1usize << p.n_vars;
                            let first = chunk * RUN;
                            runs[run] = (first < height).then(|| (p.offset + first, RUN.min(height - first)));
                        }
                        None if column > group.columns.len() => {
                            if let Some(tag) = repeats.get_mut(at >> EQUAL_BLOCK_LOG) {
                                *tag = RUN >= 1 << EQUAL_BLOCK_LOG;
                            }
                        }
                        None => {}
                    }
                }
            }
        }
        let mut padded_words = 0;
        for (global, slot) in self.slots.iter().enumerate() {
            if let Some(Slot::Padded(p)) = *slot {
                let n = 1usize << p.n_vars;
                padded_words += n;
                let from = placements[global].offset;
                if n >= 1 << EQUAL_BLOCK_LOG && from.is_multiple_of(1 << EQUAL_BLOCK_LOG) {
                    let blocks = n >> EQUAL_BLOCK_LOG;
                    repeats[p.offset >> EQUAL_BLOCK_LOG..][..blocks]
                        .copy_from_slice(&q_repeats[from >> EQUAL_BLOCK_LOG..][..blocks]);
                }
            }
        }
        let dense = primitives::filled_vec(len, |dense| {
            let (head, jagged) = dense.split_at_mut(start);
            head[padded_words..].fill(F64::ZERO);
            for (global, slot) in self.slots.iter().enumerate() {
                if let Some(Slot::Padded(p)) = *slot {
                    let n = 1usize << p.n_vars;
                    let from = placements[global].offset;
                    parallel::chunks_mut(&mut head[p.offset..p.offset + n], FILL_CHUNK, |t, out| {
                        out.copy_from_slice(&q[from + t * FILL_CHUNK..][..out.len()]);
                    });
                }
            }
            parallel::chunks_mut(jagged, FILL_CHUNK, |t, out| {
                for (r, out) in out.chunks_mut(RUN).enumerate() {
                    match runs.get(t * (FILL_CHUNK / RUN) + r).copied().flatten() {
                        Some((from, n)) => {
                            out[..n].copy_from_slice(&q[from..from + n]);
                            out[n..].fill(F64::ZERO);
                        }
                        None => out.fill(F64::ZERO),
                    }
                }
            });
        });
        (dense, repeats)
    }
}

/// Chunk width for the bulk writes below: big enough to amortize the dispatch,
/// small enough to spread one column across cores.
const FILL_CHUNK: usize = 1 << 16;

/// The stack's repeat map, all false: one tag per block of `2^EQUAL_BLOCK_LOG` words.
pub fn alloc_repeats(shape: StackShape) -> Vec<bool> {
    vec![false; shape.committed_len() >> EQUAL_BLOCK_LOG]
}

/// Carve the stack into one mutable window per committed column, in column order,
/// and zero the pad tail past the last one. A virtual column gets an empty window:
/// it is not in the stack, so its values need storage of their own.
///
/// `repeats` is the stack's repeat map ([`::pcs::stack_open::equal_blocks`] is its reference): each column
/// gets its blocks' tags to set, and the zero pad tail is tagged here. A block shared by columns smaller
/// than a block stays untagged, as does the first block of a column.
///
/// Writing each column into its final place is what lets the whole witness be
/// written once. Copying columns in afterwards would move the entire stack a
/// second time (a gigabyte at scale) for nothing, and allocating the stack zeroed
/// would memset all `2^m` slots only for the columns to overwrite them.
///
/// Safe over an unwritten stack: [`placements_of`]
/// tiles the columns from offset 0, checked here, so consecutive `split_at_mut`
/// hands out disjoint windows covering `[0, placed)` and this zeroes the rest.
pub fn split_stack<'a>(q: &'a mut [F64], repeats: &'a mut [bool], placements: &[Placement]) -> Vec<ColumnOut<'a>> {
    let mut order: Vec<usize> = (0..placements.len()).filter(|&i| !placements[i].is_virtual()).collect();
    order.sort_unstable_by_key(|&i| placements[i].offset);

    let mut windows: Vec<ColumnOut> = Vec::with_capacity(placements.len());
    windows.resize_with(placements.len(), || ColumnOut {
        words: &mut [],
        repeats: &mut [],
    });
    let total = repeats.len();
    let (mut rest, mut tags) = (q, repeats);
    let mut placed = 0usize;
    for &i in &order {
        assert_eq!(
            placements[i].offset, placed,
            "the stacked columns must tile from 0 with no gap"
        );
        let (words, tail) = rest.split_at_mut(1 << placements[i].n_vars);
        let (repeats, tag_tail) = tags.split_at_mut(words.len() >> EQUAL_BLOCK_LOG);
        windows[i] = ColumnOut { words, repeats };
        (rest, tags) = (tail, tag_tail);
        placed += 1 << placements[i].n_vars;
    }
    parallel::chunks_mut(rest, FILL_CHUNK, |_, pad| pad.fill(F64::ZERO));
    // The pad's blocks are zeros, so a block repeats its predecessor once that is in the pad too.
    let first = total - tags.len();
    for (b, tag) in (first..).zip(tags.iter_mut()) {
        *tag = b > 0 && (b - 1) << EQUAL_BLOCK_LOG >= placed;
    }
    windows
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape has to cover the placed columns with as few lanes as possible: the
    /// prover commits `committed_len()` words, so anything short of `placed` would
    /// leave a claim outside the commitment, and any extra whole lane is work the
    /// change exists to avoid. `mu` stays the announced `2^mu` size both sides
    /// derive their layout from.
    #[test]
    fn placements_cover_the_columns_in_the_fewest_lanes() {
        let lanes = 1usize << crate::pcs::LOG_BATCH;
        // A power-of-two total, one word over one, one just under, a virtual column
        // in the middle, and a stack far below the MIN_MU floor.
        let cases: [Vec<Option<usize>>; 5] = [
            vec![Some(20), Some(20)],
            vec![Some(20), Some(20), Some(0)],
            vec![Some(20), Some(19), Some(18), Some(17)],
            vec![Some(21), None, Some(19), None, Some(15)],
            vec![Some(3), Some(2)],
        ];
        for kappas in cases {
            let placed: usize = kappas.iter().flatten().map(|k| 1usize << k).sum();
            let (placements, shape) = placements_of(&kappas);
            let lane_block = 1usize << (shape.mu - crate::pcs::LOG_BATCH);

            assert_eq!(shape.mu, crate::log2_ceil_usize(placed).max(crate::pcs::MIN_MU));
            assert!((1..=lanes).contains(&shape.n_lanes), "lane count out of range");
            assert!(shape.committed_len() >= placed, "columns must fit the commitment");
            assert!(
                shape.committed_len() - placed < lane_block,
                "a whole lane of the commitment carries no data"
            );
            assert!(shape.committed_len() <= 1usize << shape.mu);
            // And every column really does live inside those lanes.
            for p in placements.iter().filter(|p| !p.is_virtual()) {
                assert!(p.offset + (1usize << p.n_vars) <= shape.committed_len());
            }
        }
    }
}
