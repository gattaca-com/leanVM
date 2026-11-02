//! `K`-valued columns stacked into one committed witness (§sec:stacking): columns laid
//! end to end, largest first at aligned offsets, into one multilinear `q` over
//! `F64`. An evaluation claim on column `i` at `ζ ∈ E` becomes the claim
//! `q̂(ζ, sel_i) = c` on the stack, where `sel_i` is the high-bit selector of
//! the column's offset.

use crate::tables::ColumnOut;
use primitives::stream::Stream;
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
    /// A whole `2^n_vars` block, its padding rows included.
    Padded(Placement),
    /// Local column `column` of jagged group `group`.
    Jagged { group: usize, column: usize },
}

/// The committed stack: the padded blocks by descending size, then the jagged groups by descending width.
///
/// It leaves every row past a group's chunks out. The prover fills it directly and reads its columns as
/// [`Column`] views, so no padded copy of a committed column exists. Both sides derive it from the announced sizes.
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

/// Rows of a column's run in the committed stack.
pub const RUN: usize = 1 << CHUNK_LOG;

/// A run of zeros and a run of ones: the rows a column holds past its runs.
static ZEROS: [F64; RUN] = [F64::ZERO; RUN];
static ONES: [F64; RUN] = [F64::ONE; RUN];

/// A column as the prover reads it: runs of [`RUN`] rows `2^stride_log` words apart, then its padding value.
///
/// A jagged column's runs are its chunks in the committed stack. A contiguous column is one long run. Rows
/// past the runs are never stored: they read from a static run of the padding value.
#[derive(Clone, Copy, Debug)]
pub struct Column<'a> {
    words: &'a [F64],
    stride_log: usize,
    stored: usize,
    len: usize,
    pad: &'static [F64; RUN],
}

impl<'a> Column<'a> {
    /// The contiguous column `words`.
    pub fn flat(words: &'a [F64]) -> Self {
        Self {
            words,
            stride_log: CHUNK_LOG,
            stored: words.len(),
            len: words.len(),
            pad: &ZEROS,
        }
    }

    /// A column of `len` rows whose first `stored` sit in runs from `words` on, `2^stride_log` words apart. The
    /// rows past them hold `pad`, zero or one.
    pub fn strided(words: &'a [F64], stride_log: usize, stored: usize, len: usize, pad: F64) -> Self {
        assert!(stored <= len && stride_log >= CHUNK_LOG);
        let pad = match pad {
            F64::ZERO => &ZEROS,
            F64::ONE => &ONES,
            _ => panic!("a column pads with zero or one"),
        };
        Self {
            words,
            stride_log,
            stored,
            len,
            pad,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The rows held in runs: every row past them is [`Self::pad`].
    pub fn stored(&self) -> usize {
        self.stored
    }

    pub fn pad(&self) -> F64 {
        self.pad[0]
    }

    /// Whether the rows lie in one contiguous run.
    pub fn is_contiguous(&self) -> bool {
        self.stride_log == CHUNK_LOG
    }

    /// Whether both are the same column.
    pub fn same(&self, other: &Self) -> bool {
        std::ptr::eq(self.words, other.words) && (self.stride_log, self.stored, self.len) == (other.stride_log, other.stored, other.len)
    }

    #[inline(always)]
    fn at(&self, i: usize) -> usize {
        ((i >> CHUNK_LOG) << self.stride_log) | (i & (RUN - 1))
    }

    /// Row `i`.
    #[inline(always)]
    pub fn get(&self, i: usize) -> F64 {
        debug_assert!(i < self.len);
        if i < self.stored { self.words[self.at(i)] } else { self.pad[0] }
    }

    /// Rows `i..i + n`, which lie in one run.
    #[inline(always)]
    pub fn span(&self, i: usize, n: usize) -> &'a [F64] {
        debug_assert!((i & (RUN - 1)) + n <= RUN);
        if i < self.stored {
            &self.words[self.at(i)..][..n]
        } else {
            &self.pad[..n]
        }
    }

    /// Rows `i..i + 8`, which lie in one run.
    #[inline(always)]
    pub fn eight(&self, i: usize) -> &'a [F64; 8] {
        self.span(i, 8).try_into().expect("eight rows")
    }

    /// Where row `i` lives, for a prefetch: computed without a bounds check, it may point past the column.
    #[inline(always)]
    pub fn ptr(&self, i: usize) -> *const F64 {
        if i < self.stored {
            self.words.as_ptr().wrapping_add(self.at(i))
        } else {
            self.pad.as_ptr()
        }
    }

    /// The stored rows from `start`, as contiguous pieces: `(first row, words)`, the first cut at `start`.
    pub fn pieces(&self, start: usize) -> impl Iterator<Item = (usize, &'a [F64])> + '_ {
        let step = if self.is_contiguous() { self.stored.max(1) } else { RUN };
        (start / step * step..self.stored).step_by(step).map(move |first| {
            let from = first.max(start);
            let end = (first + step).min(self.stored);
            (from, &self.words[self.at(from)..][..end - from])
        })
    }

    /// Every row, padding included. For tests and forgeries only: the prover never pads a column out.
    pub fn to_vec(&self) -> Vec<F64> {
        (0..self.len).map(|i| self.get(i)).collect()
    }
}

impl Dense {
    /// The committed stack's repeat map. In the jagged region only the zero runs that follow another are tagged.
    /// A padded block's fill sets its own tags.
    pub fn alloc_repeats(&self) -> Vec<bool> {
        let mut repeats = vec![false; self.shape.committed_len() >> EQUAL_BLOCK_LOG];
        for group in &self.groups {
            for chunk in 0..group.chunks() {
                for column in group.columns.len() + 1..1 << group.log_width {
                    let at = group.offset + (chunk << (CHUNK_LOG + group.log_width)) + column * RUN;
                    if let Some(tag) = repeats.get_mut(at >> EQUAL_BLOCK_LOG) {
                        *tag = RUN >= 1 << EQUAL_BLOCK_LOG;
                    }
                }
            }
        }
        repeats
    }

    /// Carve the committed stack `dense` into one window per jagged column, its runs, and zero every word no
    /// column holds. Returns the windows in global column order, `len` the column's padded height from
    /// `placements`, and each padded block as `(global column, words, tags)`.
    ///
    /// A jagged window's rows past its runs are held nowhere, and it takes no tags.
    #[allow(clippy::type_complexity)]
    pub fn split<'a>(
        &self,
        dense: &'a mut [F64],
        repeats: &'a mut [bool],
        placements: &[Placement],
    ) -> (Vec<ColumnOut<'a>>, Vec<(usize, &'a mut [F64], &'a mut [bool])>) {
        let total = dense.len();
        let mut windows: Vec<ColumnOut> = Vec::with_capacity(self.slots.len());
        windows.resize_with(self.slots.len(), ColumnOut::default);
        let mut padded: Vec<(usize, Placement)> = (self.slots.iter().enumerate())
            .filter_map(|(c, slot)| match *slot {
                Some(Slot::Padded(p)) => Some((c, p)),
                _ => None,
            })
            .collect();
        padded.sort_unstable_by_key(|&(_, p)| p.offset);
        let mut groups: Vec<&JaggedGroup> = self.groups.iter().collect();
        // A group without chunks shares its offset with the next one.
        groups.sort_unstable_by_key(|g| (g.offset, g.words()));
        let start = groups.first().map_or(self.placed, |g| g.offset);
        let (mut head, mut rest) = dense.split_at_mut(start);
        let mut tags = &mut repeats[..];
        let mut blocks = Vec::with_capacity(padded.len());
        let mut at = 0;
        for (column, p) in padded {
            assert_eq!(p.offset, at, "the padded blocks tile from 0");
            let (words, tail) = head.split_at_mut(1 << p.n_vars);
            let (block_tags, tag_tail) = tags.split_at_mut(words.len() >> EQUAL_BLOCK_LOG);
            blocks.push((column, words, block_tags));
            (head, tags, at) = (tail, tag_tail, at + (1 << p.n_vars));
        }
        let mut zeros: Vec<&mut [F64]> = head.chunks_mut(RUN).collect();
        for group in groups {
            assert_eq!(total - rest.len(), group.offset, "the groups tile");
            let (words, tail) = rest.split_at_mut(group.words());
            let width = 1 << group.log_width;
            for (r, run) in words.chunks_exact_mut(RUN).enumerate() {
                match group.columns.get(r % width) {
                    Some(&global) => windows[global].runs.push(run),
                    None => zeros.push(run),
                }
            }
            for &global in &group.columns {
                windows[global].len = 1 << placements[global].n_vars;
            }
            rest = tail;
        }
        zeros.extend(rest.chunks_mut(RUN));
        // Nothing reads the stack before the commit, so the zeros stream past the cache.
        parallel::chunks_mut(&mut zeros, FILL_CHUNK / RUN, |_, runs| {
            let stream = Stream::new();
            for run in runs {
                let n = run.len();
                stream.copy(run, &ZEROS[..n]);
            }
        });
        (windows, blocks)
    }

    /// Global column `column` of `2^n_vars` rows, from the committed stack `dense`, its rows past its chunks `pad`.
    pub fn column<'a>(&self, dense: &'a [F64], column: usize, n_vars: usize, pad: F64) -> Column<'a> {
        let len = 1usize << n_vars;
        match self.slots[column].expect("a committed column") {
            Slot::Padded(p) => Column::flat(&dense[p.offset..][..len]),
            Slot::Jagged { group, column } => {
                let group = &self.groups[group];
                let stored = (group.chunks() << CHUNK_LOG).min(len);
                let words = dense.get(group.offset + column * RUN..).unwrap_or_default();
                Column::strided(words, CHUNK_LOG + group.log_width, stored, len, pad)
            }
        }
    }
}

/// Chunk width for the bulk writes above: big enough to amortize the dispatch,
/// small enough to spread one column across cores.
const FILL_CHUNK: usize = 1 << 16;

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
