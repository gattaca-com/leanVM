//! The bus: a single shared channel balanced by a grand product (§sec:gp through §sec:leafstack). Each
//! interaction wires a table's columns into width-`m` tuples and flushes them in a
//! direction; the bus balances when pushed and pulled tuples form the same
//! multiset, proven by grand products over the leaf vectors `β − π_α(σ)` under one
//! fingerprint `(α⃗, β)`, `α⃗` a weight per tuple slot, plus the count tree. One
//! batched GKR reduces the three trees to leaf claims `Ṽ₀(ζ)` at one point,
//! decomposed into evaluation claims on the committed columns. Tuple coordinates
//! `σ_i` are `K`-valued (column entries, g-powers, separators); the fingerprint
//! challenges `α, β` are `E`-valued, so a leaf accumulates via the mixed `mul_base`
//! product (2 PMULL per coordinate).

use crate::PAR_THRESHOLD;
use crate::gkr::{self, Lanes};
use crate::transcript::ProverState;
use primitives::cast_slice;
use primitives::field::{F64, F64x8, F128, F128Unreduced, F128x8, Prod8, g_pow, mul_base_splat8, mul_by_g};
use primitives::multilinear::{eq_eval, interp, mle_eval_par};
use primitives::stream::Stream;
use std::ops::Range;
use verifier::N_BUS_TREES;
use verifier::layout::BUS_BITS;

/// One tuple coordinate as a function of the block's row `z`.
#[derive(Clone, Debug)]
pub enum Coord {
    /// A public constant (domain separator, opcode, the seed count `1`).
    Const(F64),
    /// A committed column, value `col[z]`.
    Col(usize),
    /// `g^k · col[z]`, a virtual column (§sec:vm): a count's increment or the fall-through pc, `k = 1`.
    GCol(usize, u32),
    /// The product `g^k · col_a[z] · col_b[z]` of two committed columns. An address
    /// is `fp·g^o`, so this carries one on the bus without committing it: the
    /// coordinate IS the product, so no column can disagree with it and the binding
    /// constraint that used to say so is unnecessary (§sec:m3).
    Prod(usize, usize, u32),
    /// The index column `g^z` (§sec:idxcol), free via the factored MLE.
    Index,
    /// Column `s` of the stacked bytecode (§sec:e2e-bc, [`isa::Program::stacked_bytecode`]): public, not committed,
    /// both parties form its MLE.
    Public(usize),
    /// A sum of `Const`/`Col`/`GCol`/`Prod` terms: any degree-2 form over the
    /// table's columns, which is all §sec:m3 asks of a coordinate. This is what
    /// carries a value a row DERIVES from its columns (an `XOR`/`MUL` result, a
    /// `DEREF` store, a `JUMP` successor) without committing a column for it, and
    /// with it the identity that would have tied the two. Like [`Coord::Prod`],
    /// only a table's blocks may carry one: the table sumcheck settles them,
    /// while a framework block has to split into per-column openings.
    Sum(Vec<Coord>),
}

/// A flushing rule: `2^kappa` rows, each a tuple of coordinates. The first `rows` flush;
/// the rest are a table's padding rows (§sec:e2e-pad), whose leaves are `1`.
#[derive(Clone, Debug)]
pub struct Block {
    pub kappa: usize,
    pub rows: usize,
    pub coords: Vec<Coord>,
}

impl Block {
    /// A block every row of which flushes.
    pub fn full(kappa: usize, coords: Vec<Coord>) -> Self {
        Self {
            kappa,
            rows: 1 << kappa,
            coords,
        }
    }
}

/// Placement of each block in the stacked leaf vector (input order).
#[derive(Clone, Debug)]
pub struct Layout {
    pub mu: usize,
    pub offsets: Vec<usize>,
}

/// An evaluation claim on a committed column, settled against the witness.
/// Reconstructed identically by both sides (its value rides the stream).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnClaim {
    pub col: usize,
    pub point: Vec<F128>,
    pub value: F128,
}

/// The count tree's weights: the count alone, in slot `0` (`α⃗ = (1, 0, …, 0)`, `β = 0`).
fn count_weights() -> Vec<F128> {
    let mut w = vec![F128::ZERO; 1 << BUS_BITS];
    w[0] = F128::ONE;
    w
}

/// Check the bus layout's shape. Push and pull have the same logical height, and the count
/// product is checked by the same GKR rather than by a separate root-at-random test.
fn check_bus_shape(push_blocks: &[Block], pull_blocks: &[Block], push: &Layout, pull: &Layout, count: &Layout) {
    assert_eq!(
        push.mu, pull.mu,
        "push/pull bus blocks are paired, so their layouts match"
    );
    assert!(count.mu <= push.mu, "count sums fewer bus messages than push");
    let widest = push_blocks
        .iter()
        .chain(pull_blocks)
        .map(|block| block.coords.len())
        .max()
        .unwrap_or(0);
    assert!(widest <= 1 << BUS_BITS, "a tuple's coordinates index its slots");
}

/// Stack blocks largest-first at aligned offsets; `μ = ⌈log2 Σ 2^{κ_b}⌉`.
pub fn layout(blocks: &[Block]) -> Layout {
    let kappas: Vec<usize> = blocks.iter().map(|b| b.kappa).collect();
    let (offsets, mu) = verifier::stack_offsets(&kappas);
    Layout {
        mu,
        offsets: offsets.into_iter().map(|offset| offset as usize).collect(),
    }
}

/// What a leaf term reads at row `z`, in `K`: a column (the stacked bytecode's numbered after
/// the committed ones), the product of two, or the index column `g^z`.
#[derive(Clone, Copy, PartialEq)]
enum Source {
    Col(usize),
    Prod(usize, usize),
    Index,
}

/// A non-constant coordinate as `(source, coefficient)`: its leaf contribution is
/// the mixed product `coeff · source(z)` with `coeff ∈ E`.
/// `GCol` folds the `g^k` factor into the coefficient.
type Term = (Source, F128);

/// Add a term to a formula's, summed into the one of the same source, which a zero sum drops.
fn add_term(terms: &mut Vec<Term>, (source, coeff): Term) {
    match terms.iter().position(|t| t.0 == source) {
        Some(i) => {
            terms[i].1 += coeff;
            if terms[i].1 == F128::ZERO {
                terms.swap_remove(i);
            }
        }
        None => terms.push((source, coeff)),
    }
}

/// Flatten one coordinate into leaf terms at coefficient `w`, the stacked bytecode's
/// columns numbered from `public`. A [`Coord::Sum`] spreads its children over the
/// SAME `w`: they are one coordinate, so they share its `α`-power.
fn push_terms(c: &Coord, w: F128, public: usize, terms: &mut Vec<Term>, constant: &mut F128) {
    let g = |k: u32| w.mul_base(g_pow(k as usize));
    match *c {
        Coord::Const(v) => *constant += w.mul_base(v),
        Coord::Col(i) => add_term(terms, (Source::Col(i), w)),
        Coord::GCol(i, k) => add_term(terms, (Source::Col(i), g(k))),
        Coord::Prod(i, j, k) => add_term(terms, (Source::Prod(i, j), g(k))),
        Coord::Index => add_term(terms, (Source::Index, w)),
        Coord::Public(s) => add_term(terms, (Source::Col(public + s), w)),
        Coord::Sum(ref cs) => {
            for c in cs {
                push_terms(c, w, public, terms, constant);
            }
        }
    }
}

/// One block's leaf formula: `const_part + Σ terms(z)` on its real rows, the identity
/// on its padding rows.
#[derive(Clone)]
struct LeafBlock {
    side: usize,
    offset: usize,
    kappa: usize,
    rows: usize,
    const_part: F128,
    terms: Vec<Term>,
    /// Whether the leaves are those of the block before it in its [`Group`] plus this formula,
    /// which is then their [`delta`](Self::delta).
    onto: bool,
    /// The whole GKR blocks of leaves that the parallel pass builds: the block's real rows, if it starts on a
    /// GKR block and is long enough for the pool. The rest of its rows go one by one.
    bulk: usize,
    /// Of those, the leading blocks that GKR recomputes from the columns, in whole groups of four so that
    /// two folds write complete destination blocks.
    unstored: usize,
    tail: Option<Tail>,
}

/// A memory block's leaves from row `start` on, past the cells the run used: its memory columns are
/// constant there, so a leaf is `value + index · g^z` with no column read.
#[derive(Clone, Copy)]
struct Tail {
    start: usize,
    value: F128,
    index: F128,
}

impl Tail {
    /// The tail of a block that flushes only memory columns and the index, from row `start`, if it has one.
    /// The memory columns must be constant from `start` to the block's last row.
    fn of(terms: &[Term], const_part: F128, cols: &[&[F64]], start: usize, rows: usize) -> Option<Self> {
        let memory = |i: usize| i <= verifier::layout::MEMORY_FINAL_COUNTERS;
        let columns = terms
            .iter()
            .filter(|t| matches!(t.0, Source::Col(i) if memory(i)))
            .count();
        let only_memory = terms.iter().all(|t| match t.0 {
            Source::Col(i) => memory(i),
            Source::Index => true,
            Source::Prod(..) => false,
        });
        if start >= rows || columns == 0 || !only_memory {
            return None;
        }
        let mut tail = Self {
            start,
            value: const_part,
            index: F128::ZERO,
        };
        for &(source, c) in terms {
            match source {
                Source::Col(i) => {
                    debug_assert!(cols[i][start..rows].iter().all(|&v| v == cols[i][start]));
                    tail.value += c.mul_base(cols[i][start]);
                }
                Source::Index => tail.index += c,
                Source::Prod(..) => unreachable!("memory blocks carry no products"),
            }
        }
        Some(tail)
    }
}

impl LeafBlock {
    /// `other`'s formula less this one's, for a block of the same shape: shared terms cancel.
    fn delta(&self, other: &Self) -> Self {
        let mut terms = self.terms.clone();
        for &t in &other.terms {
            add_term(&mut terms, t);
        }
        // Both tails start at one row, or the difference reads its columns throughout.
        let tail = match (self.tail, other.tail) {
            (Some(a), Some(b)) if a.start == b.start => Some(Tail {
                start: a.start,
                value: a.value + b.value,
                index: a.index + b.index,
            }),
            _ => None,
        };
        Self {
            const_part: self.const_part + other.const_part,
            terms,
            onto: true,
            tail,
            ..*other
        }
    }

    /// Whether the formula reads the index column.
    fn reads_index(&self) -> bool {
        self.terms.iter().any(|t| t.0 == Source::Index)
    }

    /// The leaf of real row `z`, whose index column value `g^z` is `index`.
    #[inline]
    fn leaf(&self, z: usize, cols: &[&[F64]], index: F64) -> F128 {
        // The α-weighted coordinate sum defers its reductions: each mixed
        // product contributes its two raw limb products (2 PMULL, no
        // reduction tail), one combined reduction per row at the end,
        // bit-identical to summing reduced `mul_base` terms.
        let mut acc = F128Unreduced::ZERO;
        for &(source, c) in &self.terms {
            acc ^= c.mul_base_unreduced(match source {
                Source::Col(i) => cols[i][z],
                Source::Prod(i, j) => cols[i][z] * cols[j][z],
                Source::Index => index,
            });
        }
        self.const_part + acc.reduce()
    }
}

/// A block's leaves eight rows at a time, on consecutive rows from `z`. Every term is one wide kernel
/// call and the reduction happens once per eight rows. The index column's `g^z` for the eight rows
/// steps by `g^8` in a register, so no table of it is read.
struct Rows8<'a> {
    block: &'a LeafBlock,
    cols: &'a [&'a [F64]],
    z: usize,
    reads_index: bool,
    index: F64x8,
    step: F64x8,
}

impl<'a> Rows8<'a> {
    fn new(block: &'a LeafBlock, cols: &'a [&'a [F64]], z: usize) -> Self {
        let reads_index = block.reads_index();
        // `g^z, …, g^(z+7)`: one power by square-and-multiply, then a shift per lane.
        let mut g = if reads_index { g_pow(z) } else { F64::ZERO };
        let lanes = std::array::from_fn(|_| {
            let lane = g;
            g = mul_by_g(g);
            lane
        });
        Self {
            block,
            cols,
            z,
            reads_index,
            index: F64x8::from_array(&lanes),
            step: F64x8::from_array(&[g_pow(8); 8]),
        }
    }

    /// The leaves of the next eight rows.
    #[inline(always)]
    fn next(&mut self) -> F128x8 {
        let (block, z) = (self.block, self.z);
        let cols = self.cols;
        let eight = |i: usize| -> &[F64; 8] { cols[i][z..z + 8].try_into().expect("eight words") };
        let mut acc = Prod8::ZERO;
        if let Some(tail) = block.tail.filter(|tail| z >= tail.start) {
            if self.reads_index {
                acc ^= mul_base_splat8(tail.index, &self.index.to_array());
                self.index = self.index * self.step;
            }
            self.z += 8;
            return acc.reduce_lanes() + F128x8::splat(tail.value);
        }
        for &(source, c) in &block.terms {
            acc ^= match source {
                Source::Col(i) => mul_base_splat8(c, eight(i)),
                Source::Prod(i, j) => {
                    let product = F64x8::from_array(eight(i)) * F64x8::from_array(eight(j));
                    mul_base_splat8(c, &product.to_array())
                }
                Source::Index => mul_base_splat8(c, &self.index.to_array()),
            };
        }
        if self.reads_index {
            self.index = self.index * self.step;
        }
        self.z += 8;
        acc.reduce_lanes() + F128x8::splat(block.const_part)
    }

    /// [`next`](Self::next) for a block of the chain at `staged`, the leaves of the block before it: with
    /// [`LeafBlock::onto`] they are added on.
    #[inline(always)]
    fn next_onto(&mut self, staged: F128x8) -> F128x8 {
        let leaves = self.next();
        if self.block.onto { staged + leaves } else { leaves }
    }
}

/// Build and publish one task of consecutive GKR `blocks` of the leaf
/// vector: `compute(l)` returns the eight leaves from leaf `l`, and each block's first radix-four
/// layer goes to `quads`, sixteen per block. The next two layers go to `tiers`: four nodes of level
/// 4 per block, and one node of level 6 for each group of eight blocks that starts at a multiple of
/// eight and lies in the task.
///
/// Staged in L1 and published with streaming stores: nothing reads a leaf until GKR, long after a
/// vector this size is evicted, so an ordinary store's fetch of every line it overwrites would
/// double the DRAM traffic of the write.
///
/// Block `k` is published, and its quads multiplied, during the compute of block `k + 1`. The
/// stores then drain in the background instead of queueing up behind each other, the products'
/// reduction chains fill the gaps of the carry-less multiplier's queue, and the fence waits for the
/// stores once per task rather than once per block.
///
/// The first `columns` blocks of the task are computed but not stored: GKR recomputes them.
///
/// `quads` and `tiers` are the task's blocks of those tables, sixteen, four and one per block, and
/// `task`, if present, holds their leaves.
fn publish_task<G: Lanes>(
    mut task: Option<&mut [[G; 8]]>,
    quads: &mut [G::Elem],
    [tier4, tier6]: [&mut [G::Elem]; 2],
    blocks: Range<usize>,
    columns: usize,
    mut compute: impl FnMut(usize) -> G,
) {
    let (first, count) = (blocks.start, blocks.len());
    let stream = Stream::new();
    let mut stage = [[G::ZERO; 8]; 2];
    let zero = G::ZERO.to_array()[0];
    let mut stage_quads = [[zero; 16]; 2];
    let mut pairs = [G::ZERO; 4];
    // Level 4 of the group of eight blocks in progress, as nodes and as the lanes of each pair of
    // blocks, and an even block's lanes, each the product of eight leaves, while it waits for its
    // odd neighbour.
    let mut nodes = [zero; 32];
    let mut node_lanes = [G::ZERO; 4];
    let mut even_block = None;
    for k in 0..=count {
        let (cur, prev) = (k % 2, (k + 1) % 2);
        let base = gkr::BLOCK_VALUES * (first + k);
        let mut groups = [G::ZERO; 8];
        let (mut evens, mut eights) = (G::ZERO, G::ZERO);
        for (j, group) in groups.iter_mut().enumerate() {
            if k < count {
                *group = compute(base + 8 * j);
            }
            if k > 0 {
                let v = &stage[prev];
                if k > columns {
                    let task = task.as_deref_mut().expect("stored leaves have a destination");
                    stream.copy(std::slice::from_mut(&mut task[k - 1][j]), &v[j..=j]);
                }
                // Lane `j` of the two quads of each group, a product per step.
                match j {
                    0..4 => pairs[j] = v[2 * j] * v[2 * j + 1],
                    4 | 5 => {
                        let lanes = pairs[2 * j - 8] * pairs[2 * j - 7];
                        for (i, q) in lanes.to_array().into_iter().enumerate() {
                            stage_quads[prev][2 * i + j - 4] = q;
                        }
                        if j == 4 {
                            evens = lanes;
                        } else {
                            eights = evens * lanes;
                        }
                    }
                    _ => {}
                }
            }
        }
        if k > 0 {
            let block = first + k - 1;
            stream.copy(&mut quads[16 * (k - 1)..][..16], &stage_quads[prev]);
            let (slot, last) = (block % 8, k == count);
            if block.is_multiple_of(2) && !last {
                even_block = Some(eights);
            } else {
                // Even and odd lanes pair into the four nodes of level 4 of a block. An odd block
                // takes the even one before it into the other four lanes, and a block without its
                // neighbour in the task leaves them unused.
                let partner = even_block.take();
                let (even, odd) = G::even_odd(partner.unwrap_or(eights), eights);
                let products = even * odd;
                node_lanes[slot / 2] = products;
                let products = products.to_array();
                let (at, count) = if partner.is_some() { (slot - 1, 8) } else { (slot, 4) };
                nodes[4 * at..4 * at + count].copy_from_slice(&products[..count]);
                if slot == 7 || last {
                    let group = block - slot;
                    let from = first.max(group);
                    stream.copy(
                        &mut tier4[4 * (from - first)..][..4 * (block + 1 - from)],
                        &nodes[4 * (from - group)..4 * (slot + 1)],
                    );
                    if slot == 7 && from == group {
                        let pair = |x, y| {
                            let (even, odd) = G::even_odd(x, y);
                            even * odd
                        };
                        let (even, odd) =
                            G::even_odd(pair(node_lanes[0], node_lanes[1]), pair(node_lanes[2], node_lanes[3]));
                        let top = (even * odd).to_array();
                        stream.copy(&mut tier6[group - first..][..8], &top);
                    }
                }
            }
        }
        if k < count {
            // Group `j` of eight leaves becomes lane `j` of the block's eight vectors.
            stage[cur] = G::transpose(groups);
        }
    }
}

/// GKR blocks per task of a pass over a table's columns, at most.
const BLOCKS: usize = 128;

/// GKR blocks per task for a group of `jobs` blocks that read the same table.
/// A task covers about `2 * BLOCKS` GKR blocks over the whole group, in whole groups of eight.
///
/// A table with many bus blocks costs many leaves per row.
/// Its tasks are cut finer, so they weigh the same as those of a table with one block.
/// Whole groups of eight keep the double folds inside one task.
/// A level-6 group cut at a task edge is filled by `build_layers`.
fn task_blocks(jobs: usize) -> usize {
    (2 * BLOCKS / jobs.max(1) / 8 * 8).clamp(8, BLOCKS)
}

/// GKR blocks of a chain staged in L1 at a time, a multiple of eight.
const RUN_BLOCKS: usize = 16;

/// The blocks of one row count, which read the same table: one pass builds them together, so the
/// table's columns come from DRAM once however many blocks flush them.
///
/// The blocks are by offset. An interaction's pull block sits at its push block's offset, laid out alike and
/// flushing mostly the same coordinates. When their difference has fewer terms than the pull block's own
/// formula, the pull block is that difference, `onto` the push block's leaves, so the products they share
/// are formed once. A chain is a block and the block `onto` it, which a pass builds run by run, each run
/// staged in L1.
///
/// Chaining a block to the one before it at another offset measured slower: on the desktop,
/// `Prove bus` took 77.3 ms against 74.1 ms with pairs only.
struct Group {
    blocks: Vec<LeafBlock>,
}

impl Group {
    /// The blocks of `blocks` with `bulk > 0`, grouped by row count.
    fn all(blocks: &[LeafBlock]) -> Vec<Self> {
        let mut bulk: Vec<&LeafBlock> = blocks.iter().filter(|b| b.bulk > 0).collect();
        // Stable, so a push block precedes the pull block at its offset.
        bulk.sort_by_key(|b| (b.rows, b.offset));
        bulk.chunk_by(|a, b| a.rows == b.rows)
            .map(|group| {
                let mut filled = vec![group[0].clone()];
                for pair in group.windows(2) {
                    let delta = pair[0].delta(pair[1]);
                    // Equal stores keep the chain's blocks in step: both stored, or both recomputed.
                    let cheaper = pair[0].offset == pair[1].offset
                        && delta.terms.len() < pair[1].terms.len()
                        && pair[0].unstored == pair[1].unstored;
                    filled.push(if cheaper { delta } else { pair[1].clone() });
                }
                Self { blocks: filled }
            })
            .collect()
    }

    fn chains(&self) -> impl Iterator<Item = &[LeafBlock]> {
        self.blocks.chunk_by(|_, next| next.onto)
    }

    /// GKR blocks of each block built by the pass: they share their row count.
    fn bulk(&self) -> usize {
        self.blocks[0].bulk
    }

    /// GKR blocks of each block per task.
    fn per(&self) -> usize {
        task_blocks(self.blocks.len())
    }
}

/// One side's leaf vector while it is built.
struct Side {
    explicit: usize,
    tiled: bool,
    leaves: gkr::Store<gkr::Block>,
    quads: Vec<F128>,
    /// Levels 4 and 6 of the blocks `publish_task` builds, in the `staged` runs of blocks; empty
    /// when the side has none.
    tiers: [Vec<F128>; 2],
    staged: Vec<Range<usize>>,
}

/// The blocks of both bus sides that [`build_leaves`] left unstored, and what computes them again
/// from the columns, in the same groups and tasks as the build.
pub(crate) struct ColumnLeaves<'a> {
    /// The committed columns, then the stacked bytecode's.
    cols: &'a [&'a [F64]],
    /// The chains of unstored blocks, by row count.
    groups: Vec<Group>,
    runs: [Vec<Range<usize>>; 2],
    tasks: usize,
}

impl ColumnLeaves<'_> {
    /// The tasks of a group: one per `per` unstored blocks.
    fn chunks(group: &Group) -> usize {
        group.blocks[0].unstored.div_ceil(group.per())
    }

    /// Computes the unstored blocks `blocks` of a chain, calling `f` for each. A run of
    /// [`RUN_BLOCKS`] is built block by block, the staged leaves of one block being
    /// the base of the next.
    fn chain(chain: &[LeafBlock], blocks: Range<usize>, cols: &[&[F64]], f: &mut dyn FnMut(usize, usize, &gkr::Block)) {
        let mut rows: Vec<Rows8> = chain
            .iter()
            .map(|blk| Rows8::new(blk, cols, 64 * blocks.start))
            .collect();
        let mut stage = [[F128x8::ZERO; 8]; RUN_BLOCKS];
        for start in blocks.clone().step_by(RUN_BLOCKS) {
            let stop = (start + RUN_BLOCKS).min(blocks.end);
            for (blk, rows) in chain.iter().zip(&mut rows) {
                for (k, staged) in (start..stop).zip(&mut stage) {
                    *staged = std::array::from_fn(|j| rows.next_onto(staged[j]));
                    f(
                        blk.side,
                        blk.offset / gkr::BLOCK_VALUES + k,
                        &F128x8::transpose(*staged),
                    );
                }
            }
        }
    }
}

impl gkr::LeafColumns for ColumnLeaves<'_> {
    fn runs(&self, s: usize) -> &[Range<usize>] {
        &self.runs[s]
    }

    fn tasks(&self) -> usize {
        self.tasks
    }

    fn task(&self, t: usize, f: &mut dyn FnMut(usize, usize, &gkr::Block)) {
        // The groups run from the fewest rows up, and those hold the most blocks per row.
        // Guided claims start large, so the tasks are numbered from the last group.
        // The light tasks of the tall groups go first, and the heavy ones are claimed one by one.
        let mut t = self.tasks - 1 - t;
        for group in &self.groups {
            let chunks = Self::chunks(group);
            if t >= chunks {
                t -= chunks;
                continue;
            }
            let per = group.per();
            let blocks = t * per..((t + 1) * per).min(group.blocks[0].unstored);
            for chain in group.chains() {
                Self::chain(chain, blocks.clone(), self.cols, f);
            }
            return;
        }
    }
}

/// Build the leaf vectors of the bus's push and pull sides: block `b` row `z` holds `β − Σ_i w_i c_i(z)`
/// for the side's fingerprint weights `w`, followed implicitly by the identity `1` up to `2^μ`. The
/// row-invariant weights and constant coordinates are folded once per block into its `const_part`.
///
/// The blocks of both sides with the same number of rows read the same table, so one pass runs them
/// together over each task's rows ([`Group`]): the table's columns come from DRAM once for all of them,
/// and every block publishes exactly what its own pass would.
///
/// With `columns`, aligned groups of four bulk GKR blocks are left unstored; incomplete boundary groups
/// remain stored. The returned [`ColumnLeaves`] recomputes them from the columns, reducing traffic compared
/// with reading stored leaves. A side that does not tile its layout keeps all its leaves.
///
/// `cols` are the `committed` columns, then the stacked bytecode's, which [`Coord::Public`] numbers from zero.
/// The memory columns must be constant from `memory_len` to the end of their blocks.
pub(crate) fn build_leaves<'a>(
    sides: [(&[Block], &Layout, &[F128], F128); 2],
    cols: &'a [&'a [F64]],
    committed: usize,
    columns: bool,
    memory_len: usize,
) -> ([gkr::Leaves<'static>; 2], Option<ColumnLeaves<'a>>) {
    let mut built: Vec<Side> = Vec::with_capacity(2);
    let mut blocks: Vec<LeafBlock> = Vec::new();
    for (side, &(side_blocks, lay, w, beta)) in sides.iter().enumerate() {
        let explicit = side_blocks
            .iter()
            .enumerate()
            .map(|(b, block)| lay.offsets[b] + (1usize << block.kappa))
            .max()
            .unwrap_or(1);
        debug_assert!(explicit <= 1usize << lay.mu);
        // `verifier::stack_offsets` packs the power-of-two blocks contiguously from zero, so the
        // blocks tile `0..explicit` and every slot below is written by one of them: the
        // identity fill would be overwritten in full, and this is the largest buffer in
        // the proof. The `covered` test is what licenses skipping it, so a layout that
        // ever left a hole falls back to filling rather than reading uninitialized rows.
        // The leaves are GKR's blocks of 64, so the identity past `explicit` is written to
        // the end of the last one, which is where GKR expects it. The padding rows inside the layout
        // are runs of identity that GKR skips (`Leaves::ones`), so their whole blocks stay unwritten.
        let covered: usize = side_blocks.iter().map(|blk| 1usize << blk.kappa).sum();
        let block_count = explicit.div_ceil(gkr::BLOCK_VALUES);
        let tiled = covered == explicit;
        // With `columns`, GKR recomputes most blocks and skips the identity runs.
        let mut leaves = if tiled && columns {
            gkr::Store::sparse(block_count)
        } else if tiled {
            // SAFETY: the per-block fills below cover `0..explicit` but for the identity blocks GKR
            // never reads, each joins before this function returns, and the tail of the last block
            // is written next.
            gkr::Store::Heap(unsafe { primitives::uninit_vec(block_count) })
        } else {
            gkr::Store::Heap(vec![[F128x8::ONE; 8]; block_count])
        };
        for l in explicit..gkr::BLOCK_VALUES * block_count {
            gkr::set_leaf(&mut leaves, l, F128::ONE);
        }
        // The first radix-four layer, `quads[k] = leaves[4k] · … · leaves[4k + 3]`, built from the
        // staged rows while they are in L1, so GKR never reads the leaves to build it.
        //
        // SAFETY: every slot is written before this function returns: the staged groups write
        // theirs, and `finish_quads` writes every slot past them.
        let quads = unsafe { primitives::uninit_vec::<F128>(explicit.div_ceil(4)) };
        let side_columns = columns && tiled;
        for (b, blk) in side_blocks.iter().enumerate() {
            let offset = lay.offsets[b];
            // Whole GKR blocks of the block's leaves, built in parallel. Blocks off the GKR block
            // boundary or too few for the pool go row by row.
            let bulk = if blk.rows >= PAR_THRESHOLD && offset.is_multiple_of(gkr::BLOCK_VALUES) {
                blk.rows / gkr::BLOCK_VALUES
            } else {
                0
            };
            let unstored = if side_columns && offset.is_multiple_of(4 * gkr::BLOCK_VALUES) {
                bulk & !3
            } else {
                0
            };
            let mut const_part = beta;
            let mut terms: Vec<Term> = Vec::with_capacity(blk.coords.len());
            for (i, c) in blk.coords.iter().enumerate() {
                push_terms(c, w[i], committed, &mut terms, &mut const_part);
            }
            // Whole vectors of rows, so a call of eight is wholly in the tail or out of it.
            let tail = Tail::of(&terms, const_part, cols, memory_len.next_multiple_of(8), blk.rows);
            blocks.push(LeafBlock {
                side,
                offset,
                kappa: blk.kappa,
                rows: blk.rows,
                const_part,
                terms,
                onto: false,
                bulk,
                unstored,
                tail,
            });
        }
        built.push(Side {
            explicit,
            tiled,
            leaves,
            quads,
            tiers: [Vec::new(), Vec::new()],
            staged: Vec::new(),
        });
    }
    let groups = Group::all(&blocks);
    for group in &groups {
        let (bulk, per) = (group.bulk(), group.per());
        for blk in &group.blocks {
            let first = blk.offset / gkr::BLOCK_VALUES;
            built[blk.side]
                .staged
                .extend((0..bulk.div_ceil(per)).map(|c| first + c * per..first + ((c + 1) * per).min(bulk)));
        }
    }
    for (s, side) in built.iter_mut().enumerate() {
        if blocks.iter().any(|blk| blk.side == s && blk.bulk > 0) {
            let level4 = side.quads.len().div_ceil(4);
            // SAFETY: every entry is written by a task below or by `build_layers`, which
            // computes the entries of the blocks outside `staged` unless they are identity.
            side.tiers = unsafe {
                [
                    primitives::uninit_vec(level4),
                    primitives::uninit_vec(level4.div_ceil(4)),
                ]
            };
        }
    }
    let ptrs: Vec<_> = built
        .iter_mut()
        .map(|side| {
            let [l4, l6] = side.tiers.each_mut().map(|t| parallel::SendPtr(t.as_mut_ptr()));
            (
                parallel::SendPtr(side.leaves.as_mut_ptr()),
                parallel::SendPtr(side.quads.as_mut_ptr()),
                [l4, l6],
            )
        })
        .collect();
    let tasks: Vec<(usize, usize)> = groups
        .iter()
        .enumerate()
        .flat_map(|(g, group)| (0..group.bulk().div_ceil(group.per())).map(move |c| (g, c)))
        .collect();
    parallel::for_each(tasks.len(), |task| {
        let (g, c) = tasks[task];
        let group = &groups[g];
        let (bulk, per) = (group.bulk(), group.per());
        let task_blocks = c * per..bulk.min((c + 1) * per);
        // A block's leaves, quads and tiers of its GKR blocks `run`, as `publish_task` takes them.
        let parts = |blk: &LeafBlock, run: Range<usize>| {
            let first = blk.offset / gkr::BLOCK_VALUES + run.start;
            let (side, len) = (blk.side, run.len());
            // SAFETY: task `c` owns these blocks of the leaves, their quads and their tiers, which no
            // other task or block of the group touches.
            let parts = unsafe {
                (
                    Some(ptrs[side].0.slice(first, len)),
                    ptrs[side].1.slice(16 * first, 16 * len),
                    [
                        ptrs[side].2[0].slice(4 * first, 4 * len),
                        ptrs[side].2[1].slice(first, len),
                    ],
                )
            };
            (parts, first..first + len)
        };
        for chain in group.chains() {
            if let [blk] = chain {
                let columns = blk.unstored.saturating_sub(task_blocks.start).min(task_blocks.len());
                let ((leaves, quads, tiers), blocks) = parts(blk, task_blocks.clone());
                let mut rows = Rows8::new(blk, cols, gkr::BLOCK_VALUES * task_blocks.start);
                publish_task(leaves, quads, tiers, blocks, columns, |_| rows.next());
                continue;
            }
            // The chain publishes each run of `RUN_BLOCKS` block by block, from the leaves staged in L1
            // that the block before it left. The runs split at multiples of eight, as the level-6 groups do.
            let mut rows: Vec<Rows8> = chain
                .iter()
                .map(|blk| Rows8::new(blk, cols, gkr::BLOCK_VALUES * task_blocks.start))
                .collect();
            let mut stage = [[F128x8::ZERO; 8]; RUN_BLOCKS];
            let slot = |z: usize| (z / gkr::BLOCK_VALUES % RUN_BLOCKS, z % gkr::BLOCK_VALUES / 8);
            for start in task_blocks.clone().step_by(RUN_BLOCKS) {
                let stop = (start + RUN_BLOCKS).min(task_blocks.end);
                for (blk, rows) in chain.iter().zip(&mut rows) {
                    let columns = blk.unstored.saturating_sub(start).min(stop - start);
                    let ((leaves, quads, tiers), blocks) = parts(blk, start..stop);
                    publish_task(leaves, quads, tiers, blocks, columns, |l| {
                        let (i, j) = slot(l - blk.offset);
                        stage[i][j] = rows.next_onto(stage[i][j]);
                        stage[i][j]
                    });
                }
            }
        }
    });
    let column_groups: Vec<Group> = groups
        .iter()
        .filter_map(|group| {
            let blocks: Vec<LeafBlock> = group
                .chains()
                .filter(|chain| chain[0].unstored > 0)
                .flatten()
                .cloned()
                .collect();
            (!blocks.is_empty()).then_some(Group { blocks })
        })
        .collect();
    let mut runs = [Vec::new(), Vec::new()];
    for blk in column_groups.iter().flat_map(|group| &group.blocks) {
        let first = blk.offset / gkr::BLOCK_VALUES;
        runs[blk.side].push(first..first + blk.unstored);
    }
    for run in &mut runs {
        run.sort_unstable_by_key(|run| run.start);
    }
    let tasks = column_groups.iter().map(ColumnLeaves::chunks).sum();
    let column_leaves = (!column_groups.is_empty()).then_some(ColumnLeaves {
        cols,
        groups: column_groups,
        runs,
        tasks,
    });
    let leaves: Vec<gkr::Leaves<'static>> = built
        .into_iter()
        .enumerate()
        .map(|(s, mut side)| {
            let (side_blocks, lay, ..) = sides[s];
            let own = || blocks.iter().filter(|blk| blk.side == s);
            for blk in own() {
                let mut index = if blk.reads_index() {
                    g_pow(gkr::BLOCK_VALUES * blk.bulk)
                } else {
                    F64::ZERO
                };
                for z in gkr::BLOCK_VALUES * blk.bulk..blk.rows {
                    gkr::set_leaf(&mut side.leaves, blk.offset + z, blk.leaf(z, cols, index));
                    index = mul_by_g(index);
                }
                // A padding row's leaf is the identity, so it leaves the product alone.
                if side.tiled {
                    fill_ones(
                        &mut side.leaves,
                        blk.offset + blk.rows,
                        blk.offset + (1usize << blk.kappa),
                        side.explicit,
                    );
                }
            }
            let Side {
                explicit,
                tiled,
                leaves,
                mut quads,
                tiers,
                mut staged,
                ..
            } = side;
            staged.sort_unstable_by_key(|run| run.start);
            let mut ones: Vec<Range<usize>> = Vec::new();
            if tiled {
                let quads_done: Vec<usize> = own().map(|blk| blk.offset + gkr::BLOCK_VALUES * blk.bulk).collect();
                finish_quads(&leaves, &mut quads, side_blocks, lay, &quads_done);
                let mut runs: Vec<_> = side_blocks
                    .iter()
                    .enumerate()
                    .map(|(b, blk)| lay.offsets[b] + blk.rows..lay.offsets[b] + (1usize << blk.kappa))
                    .filter(|run| !run.is_empty())
                    .collect();
                runs.sort_unstable_by_key(|run| run.start);
                for run in runs {
                    match ones.last_mut() {
                        Some(last) if last.end == run.start => last.end = run.end,
                        _ => ones.push(run),
                    }
                }
            } else {
                parallel::fill(&mut quads, |k| gkr::quad_at(&leaves, k));
            }
            gkr::Leaves {
                leaves: gkr::LeafData::Blocks(leaves),
                len: explicit,
                quads,
                ones,
                tiers,
                staged,
            }
        })
        .collect();
    let leaves = leaves.try_into().ok().expect("two sides");
    (leaves, column_leaves)
}

/// The values at `level`, of a table of `len`, that GKR never reads inside the run of identity
/// leaves `run`: its whole blocks of 64.
fn unread(run: Range<usize>, level: usize, len: usize) -> Range<usize> {
    let blocks = gkr::identity_blocks(&[run], level, len).pop();
    blocks.map_or(0..0, |run| run.start * gkr::BLOCK_VALUES..run.end * gkr::BLOCK_VALUES)
}

/// `range` less `skip`, which is empty or inside it.
fn outside(range: Range<usize>, skip: Range<usize>) -> impl Iterator<Item = usize> {
    let (start, end) = (
        skip.start.clamp(range.start, range.end),
        skip.end.clamp(range.start, range.end),
    );
    (range.start..start).chain(end..range.end)
}

/// Leaves `start..end` of a block table of `len` leaves set to the identity, but for whole blocks.
fn fill_ones(leaves: &mut [gkr::Block], start: usize, end: usize, len: usize) {
    for l in outside(start..end, unread(start..end, 0, len)) {
        gkr::set_leaf(leaves, l, F128::ONE);
    }
}

/// Write every quad the staged groups did not: the rest of each block from its leaves, and
/// whole-identity runs of its padding as the identity.
fn finish_quads(leaves: &[gkr::Block], quads: &mut [F128], blocks: &[Block], lay: &Layout, quads_done: &[usize]) {
    let quad = |k: usize| gkr::quad_at(leaves, k);
    // Blocks below four slots share quads with their neighbours, so the quads from the first
    // of them on are computed from the finished leaves. They sit last: blocks are placed
    // largest first.
    let small_from = blocks
        .iter()
        .enumerate()
        .filter(|(_, blk)| blk.kappa < 2)
        .map(|(b, _)| lay.offsets[b] / 4)
        .min()
        .unwrap_or(quads.len());
    for (b, blk) in blocks.iter().enumerate() {
        if blk.kappa < 2 {
            continue;
        }
        let (start, end) = (
            quads_done[b] / 4,
            ((lay.offsets[b] + (1usize << blk.kappa)) / 4).min(small_from),
        );
        if start >= end {
            continue;
        }
        let ones_from = (lay.offsets[b] + blk.rows).div_ceil(4).clamp(start, end);
        let computed = &mut quads[start..ones_from];
        if computed.len() >= PAR_THRESHOLD {
            parallel::fill(computed, |i| quad(start + i));
        } else {
            for (i, q) in computed.iter_mut().enumerate() {
                *q = quad(start + i);
            }
        }
        let run = lay.offsets[b] + blk.rows..lay.offsets[b] + (1usize << blk.kappa);
        for k in outside(ones_from..end, unread(run, 2, quads.len())) {
            quads[k] = F128::ONE;
        }
    }
    for k in small_from..quads.len() {
        quads[k] = quad(k);
    }
}

/// The count side reads its base-field leaves directly from the columns. Only product levels
/// are stored, with whole blocks of identity padding left implicit.
pub(crate) fn build_count_leaves<'a>(blocks: &[Block], lay: &Layout, cols: &[&'a [F64]]) -> gkr::Leaves<'a, F64> {
    // `(offset, end, column)` of each block, the column cut to its rows, by offset.
    let mut segments: Vec<(usize, usize, &[F64])> = blocks
        .iter()
        .enumerate()
        .map(|(b, blk)| {
            let [Coord::Col(i)] = blk.coords.as_slice() else {
                panic!("a count block is one column");
            };
            (
                lay.offsets[b],
                lay.offsets[b] + (1usize << blk.kappa),
                &cols[*i][..blk.rows],
            )
        })
        .collect();
    segments.sort_unstable_by_key(|segment| segment.0);
    let explicit = segments.iter().map(|segment| segment.1).max().unwrap_or(1);
    let block_count = explicit.div_ceil(gkr::BLOCK_VALUES);
    let columns = gkr::Columns {
        segments: segments.into_iter().map(|(off, _, data)| (off, data)).collect(),
        len: explicit,
    };
    let mut ones = Vec::new();
    let mut end = 0;
    for &(off, data) in &columns.segments {
        if end < off {
            ones.push(end..off);
        }
        end = off + data.len();
    }
    if end < explicit {
        ones.push(end..explicit);
    }
    let mut staged = Vec::new();
    let mut first = 0;
    for skip in gkr::identity_blocks(&ones, 0, explicit)
        .into_iter()
        .chain(std::iter::once(block_count..block_count))
    {
        staged.extend(
            (first..skip.start)
                .step_by(BLOCKS)
                .map(|start| start..(start + BLOCKS).min(skip.start)),
        );
        first = skip.end;
    }
    // The staged runs write their quads and tiers, and the boundary fill below the identity quads GKR
    // reads. Whole identity blocks stay unread, as in `build_leaves`.
    let [mut quads, mut tier4, mut tier6] =
        // SAFETY: the fill writes every element before it reads any.
        unsafe { primitives::filled_vecs([16 * block_count, 4 * block_count, block_count], |bufs| {
            let ptrs = bufs.map(|buf| parallel::SendPtr(buf.as_mut_ptr()));
            parallel::for_each(staged.len(), |c| {
                let run = staged[c].clone();
                let (first, len) = (run.start, run.len());
                // The staged runs are disjoint, and so are their blocks of the product levels.
                let (quads, tiers) = (
                    ptrs[0].slice(16 * first, 16 * len),
                    [ptrs[1].slice(4 * first, 4 * len), ptrs[2].slice(first, len)],
                );
                publish_task(None, quads, tiers, run, len, |l| {
                    F64x8::from_array(&columns.read::<8>(l))
                });
            });
        }) };
    quads.truncate(explicit.div_ceil(4));
    for run in &ones {
        let range = run.start.div_ceil(4)..run.end / 4;
        if !range.is_empty() {
            for q in outside(range, unread(run.clone(), 2, quads.len())) {
                quads[q] = F64::ONE;
            }
        }
    }
    tier4.truncate(quads.len().div_ceil(4));
    tier6.truncate(quads.len().div_ceil(4).div_ceil(4));
    let tiers = [tier4, tier6];
    gkr::Leaves {
        leaves: gkr::LeafData::Columns(columns),
        len: explicit,
        quads,
        ones,
        tiers,
        staged,
    }
}

/// One table's bus contribution on one side, as a form over that table's committed
/// columns: `Σ_c coeffs[c]·col_c(z) + Σ (a,b,c) c·col_a(z)·col_b(z) + constant`.
/// Every coefficient is a public function of `α`, `β` and the block selectors at
/// `ζ`, because a table's bus blocks carry only `Const`/`Col`/`GCol`/`Prod`
/// coordinates. The table sumcheck sums this against `eq(ζ[..τ], ·)` instead of
/// opening each column at `ζ`, which is why those per-column claims no longer reach
/// the PCS.
///
/// The quadratic part comes from [`Coord::Prod`] and is free: the AIR identities are
/// already degree 2, so a degree-2 form does not raise the round-polynomial degree
/// the batch pays for.
#[derive(Clone, Debug)]
pub struct BusForm {
    pub coeffs: Vec<F128>,
    /// `(col_a, col_b, coeff)` in LOCAL column indices.
    pub(crate) prods: Vec<(usize, usize, F128)>,
    pub constant: F128,
}

impl BusForm {
    /// The zero form over `n_cols` columns.
    pub(crate) fn new(n_cols: usize) -> Self {
        Self {
            coeffs: vec![F128::ZERO; n_cols],
            prods: Vec::new(),
            constant: F128::ZERO,
        }
    }

    /// The same form scaled by `w`. Every coefficient is `E`-valued already, so
    /// folding the side's `ξ`-power in here costs three multiplies once per table
    /// instead of one per row and round of the zerocheck.
    pub fn scaled(&self, w: F128) -> Self {
        Self {
            coeffs: self.coeffs.iter().map(|&c| c * w).collect(),
            prods: self.prods.iter().map(|&(a, b, c)| (a, b, c * w)).collect(),
            constant: self.constant * w,
        }
    }

    /// The pointwise sum of several forms over the same columns. Their products are
    /// listed together, a pair named in two forms twice.
    pub fn sum(forms: impl IntoIterator<Item = Self>) -> Self {
        let mut forms = forms.into_iter();
        let mut out = forms.next().expect("a table has at least one bus form");
        for form in forms {
            assert_eq!(out.coeffs.len(), form.coeffs.len(), "forms over the same columns");
            for (slot, c) in out.coeffs.iter_mut().zip(form.coeffs) {
                *slot += c;
            }
            out.constant += form.constant;
            out.prods.extend(form.prods);
        }
        out
    }

    /// The form at one point, `evals` the columns' values there; only its degree-two terms when `quadratic`.
    #[cfg(test)]
    pub(crate) fn eval(&self, evals: &[F128], quadratic: bool) -> F128 {
        let linear = if quadratic { F128::ZERO } else { self.linear(evals) };
        self.prods
            .iter()
            .fold(linear, |acc, &(a, b, c)| acc + evals[a] * evals[b] * c)
    }

    /// `constant + Σ coeffs[i]·evals[i]`.
    #[cfg(test)]
    fn linear(&self, evals: &[F128]) -> F128 {
        self.coeffs
            .iter()
            .zip(evals)
            .fold(self.constant, |acc, (&w, &v)| acc + w * v)
    }
}

/// Accumulate one coordinate of a table's block into that table's form, at
/// coefficient `w`. A [`Coord::Sum`]'s children share `w`, so a derived value
/// lands as the several coefficients and products it is made of.
fn accumulate_form(c: &Coord, w: F128, base: usize, form: &mut BusForm) {
    match c {
        Coord::Const(v) => form.constant += w.mul_base(*v),
        Coord::Col(i) => form.coeffs[*i - base] += w,
        Coord::GCol(i, k) => form.coeffs[*i - base] += w.mul_base(g_pow(*k as usize)),
        Coord::Prod(i, j, k) => form.prods.push((*i - base, *j - base, w.mul_base(g_pow(*k as usize)))),
        Coord::Sum(cs) => {
            for c in cs {
                accumulate_form(c, w, base, form);
            }
        }
        Coord::Index | Coord::Public(_) => {
            unreachable!("a table's bus block carries no virtual coordinate")
        }
    }
}

/// Look up an already-recorded claim on `(col, point)`. Push and pull share
/// their GKR point, so a column read by both sides (or by two same-κ blocks of
/// one side) is streamed and opened ONCE.
fn known_claim(claims: &[ColumnClaim], col: usize, point: &[F128]) -> bool {
    claims.iter().any(|c| c.col == col && c.point == point)
}

/// Walk one side's blocks. A block owned by table `t` (with column base `base`)
/// accumulates into `forms[t]`, for the table sumcheck to settle; a framework block's
/// committed columns are evaluated at `ζ[..κ]` and each FRESH value is written onto the
/// stream and recorded as a claim, in block/coord order.
fn decompose_prove(
    blocks: &[Block],
    lay: &Layout,
    cols: &[&[F64]],
    zeta: &[F128],
    w: &[F128],
    beta: F128,
    owners: &[Option<(usize, usize)>],
    forms: &mut [BusForm],
    claims: &mut Vec<ColumnClaim>,
    ps: &mut ProverState,
) {
    assert_eq!(zeta.len(), lay.mu);
    let mut jobs: Vec<(usize, usize)> = Vec::new();
    for (b, blk) in blocks.iter().enumerate() {
        let kappa = blk.kappa;
        if let Some((t, base)) = owners[b] {
            let sel = lay.offsets[b] >> kappa;
            let sel_bits: Vec<F128> = (0..(lay.mu - kappa))
                .map(|k| F128::new(((sel >> k) & 1) as u64, 0))
                .collect();
            let eq_hi = eq_eval(&sel_bits, &zeta[kappa..]);
            let form = &mut forms[t];
            form.constant += eq_hi * beta;
            for (i, c) in blk.coords.iter().enumerate() {
                accumulate_form(c, eq_hi * w[i], base, form);
            }
            continue;
        }
        for c in &blk.coords {
            if let Coord::Col(i) | Coord::GCol(i, _) = c
                && !known_claim(claims, *i, &zeta[..kappa])
                && !jobs.contains(&(*i, kappa))
            {
                jobs.push((*i, kappa));
            }
        }
    }
    // One evaluation at a time, each across the whole pool: a few long columns, one
    // per worker, would leave the rest of it idle.
    for (col, kappa) in jobs {
        let point = &zeta[..kappa];
        let value = mle_eval_par(cols[col], point);
        ps.add_scalar(value);
        claims.push(ColumnClaim {
            col,
            point: point.to_vec(),
            value,
        });
    }
}

/// The bus trees in GKR order (push, pull, count), each its blocks, layout, fingerprint
/// weights and `β`. The count channel's leaf is the count itself (a single `Col`), so it
/// runs at [`count_weights`], `β = 0`, and its GKR root is the product of all counts.
fn sides<'a>(
    blocks: [&'a [Block]; N_BUS_TREES],
    lays: [&'a Layout; N_BUS_TREES],
    w: &'a [F128],
    count_w: &'a [F128],
    beta: F128,
) -> [(&'a [Block], &'a Layout, &'a [F128], F128); N_BUS_TREES] {
    [
        (blocks[0], lays[0], w, beta),
        (blocks[1], lays[1], w, beta),
        (blocks[2], lays[2], count_w, F128::ZERO),
    ]
}

/// The program's share of a bytecode leaf, `S = Σ_x α_x·P(ζ_lo, x)` over the slots `x` of the
/// stacked bytecode `P` (§sec:e2e-bc), reduced to one evaluation of `P`. The stream: `S`, a
/// sumcheck of [`BUS_BITS`] degree-two rounds over `x`, the low bit bound first, ending at
/// `r⃗`, then `P(ζ_lo, r⃗)`, whose product with `α̃(r⃗)` the verifier checks against the last round.
fn prove_bytecode_share(push: &[Block], zeta: &[F128], w: &[F128], public: &[&[F64]], ps: &mut ProverState) {
    let block = push
        .iter()
        .find(|blk| blk.coords.iter().any(|c| matches!(c, Coord::Public(_))))
        .expect("the push side seeds the bytecode");
    let zeta_lo = &zeta[..block.kappa];
    let mut slots: Vec<F128> = (0..1 << BUS_BITS)
        .map(|x| match block.coords.get(x) {
            Some(Coord::Public(s)) => mle_eval_par(public[*s], zeta_lo),
            _ => F128::ZERO,
        })
        .collect();
    let mut weights = w.to_vec();
    ps.add_scalar(weights.iter().zip(&slots).map(|(&a, &p)| a * p).sum());
    for _ in 0..BUS_BITS {
        let pairs = || weights.as_chunks::<2>().0.iter().zip(slots.as_chunks::<2>().0);
        let c0: F128 = pairs().map(|(a, p)| a[0] * p[0]).sum();
        let c2: F128 = pairs().map(|(a, p)| (a[0] + a[1]) * (p[0] + p[1])).sum();
        let c1: F128 = pairs().map(|(a, p)| a[0] * (p[0] + p[1]) + (a[0] + a[1]) * p[0]).sum();
        ps.add_round_poly(&[c0, c1, c2], false);
        let r = ps.sample();
        let fold = |v: &[F128]| -> Vec<F128> { v.as_chunks::<2>().0.iter().map(|h| interp(h[0], h[1], r)).collect() };
        (weights, slots) = (fold(&weights), fold(&slots));
    }
    ps.add_scalar(slots[0]);
}

/// Everything the bus hands on: the framework blocks' column claims,
/// the shared GKR point (the table sumcheck's eq point), and
/// per tree the tables' forms.
pub(crate) struct BusProof {
    pub claims: Vec<ColumnClaim>,
    /// The GKR point ζ: the zerocheck reuses it, so no fresh point is sampled.
    pub point: Vec<F128>,
    /// `forms[tree][table]`, in GKR order ([`N_BUS_TREES`]).
    pub forms: [Vec<BusForm>; N_BUS_TREES],
}

/// The workers from which an x86 build without AVX-512 rebuilds the bus leaves at every depth.
/// With fewer workers it stores them at `μ <= AVX2_STORE_MAX_MU`.
///
/// Rebuilding lost 45 ms at `μ = 25` (the leaf) only while the column tasks were of uneven weight, and
/// the heaviest task held back the rest. With tasks of even weight ([`task_blocks`]) it wins on 8 workers
/// and the leaf store of 786 MiB is gone. At 4 workers the store still wins: `Prove bus` took 163 ms
/// stored and 186 ms rebuilt. 5 to 7 workers were not measured.
const AVX2_REBUILD_MIN_WORKERS: usize = 8;

/// The largest bus depth `μ` whose leaves an x86 build without AVX-512 stores rather than rebuilds,
/// with fewer than [`AVX2_REBUILD_MIN_WORKERS`] workers.
/// Above it the store is 1 GiB or more of peak memory, and rebuilding costs no time.
/// At or below it the store is small, and rebuilding costs time.
const AVX2_STORE_MAX_MU: usize = 25;

/// Whether a bus of depth `mu` rebuilds its leaves from the columns with `workers` workers, if the
/// build allows it at all.
fn rebuilds_leaves(mu: usize, workers: usize) -> bool {
    cfg!(any(not(target_arch = "x86_64"), target_feature = "avx512f"))
        || workers >= AVX2_REBUILD_MIN_WORKERS
        || mu > AVX2_STORE_MAX_MU
}

/// Prove the bus balances; returns the per-column claims to open (§sec:leafstack). The
/// fingerprint follows the witness commitment (the only ordering the grand product
/// needs), and the block structure is public, so no shape is observed.
///
/// The stream, in order: the [`verifier::bus_grinding_bits`] grind and the fingerprint `(α⃗, β)`; the
/// batched GKR ([`gkr::prove_products_from`]); the framework blocks' fresh column values; the
/// bytecode's share ([`prove_bytecode_share`]).
///
/// The three memory columns must be constant from `memory_len` to their logical end.
pub(crate) fn prove_balance(
    push: &[Block],
    pull: &[Block],
    count: &[Block],
    cols: &[&[F64]],
    stacked_bytecode: &[u64],
    owners: &[Vec<Option<(usize, usize)>>; 3],
    tables: &[(usize, usize)],
    memory_len: usize,
    ps: &mut ProverState,
) -> BusProof {
    // The columns [`Coord::Public`] names.
    let public: Vec<&[F64]> = cast_slice(stacked_bytecode)
        .chunks_exact(stacked_bytecode.len() >> BUS_BITS)
        .collect();
    let push_lay = layout(push);
    let pull_lay = layout(pull);
    let mut count_lay = layout(count);
    check_bus_shape(push, pull, &push_lay, &pull_lay, &count_lay);
    let bits = verifier::bus_grinding_bits(push_lay.mu);
    tracing::info_span!("Bus grind", mu = push_lay.mu, bits).in_scope(|| ps.grind(bits));
    let w = ps.sample_vec(1 << BUS_BITS);
    let beta = ps.sample();
    let count_w = count_weights();
    // The leaves keep the all-one padding implicit; decomposition uses the full logical depth.
    count_lay.mu = push_lay.mu;
    let blocks = [push, pull, count];
    let lays = [&push_lay, &pull_lay, &count_lay];
    let sides = sides(blocks, lays, &w, &count_w, beta);
    // A term reads the stacked bytecode's columns after the committed ones.
    let all_cols: Vec<&[F64]> = [cols, &public[..]].concat();
    // On machines that favor rebuilding, GKR recomputes most leaves from the columns with several threads.
    // On x86 without AVX-512 its first rounds are bound by the products, so with fewer than
    // `AVX2_REBUILD_MIN_WORKERS` workers a small bus stores its leaves.
    // Above `AVX2_STORE_MAX_MU` the store would hold 1 GiB or more at the peak.
    let columns = parallel::num_threads() > 1
        && primitives::REBUILD_OVER_STORE
        && rebuilds_leaves(push_lay.mu, parallel::num_threads());
    // Push and pull are built together, so a table's columns are read once for both sides.
    let (push_leaves, pull_leaves, count_leaves, column_leaves) = crate::stage!("Bus leaves", || {
        let (leaves, column_leaves) = build_leaves([sides[0], sides[1]], &all_cols, cols.len(), columns, memory_len);
        let [push_leaves, pull_leaves] = leaves;
        (
            push_leaves,
            pull_leaves,
            build_count_leaves(count, &count_lay, cols),
            column_leaves,
        )
    });
    // All three trees run as ONE RLC-batched GKR (equal μ: push/pull match
    // block-for-block, count is padded), so every claim lands on ONE point ζ.
    let bus_gkr = crate::stage!("Bus GKR", || {
        gkr::prove_products_from(
            push_leaves,
            pull_leaves,
            count_leaves,
            column_leaves.as_ref().map(|c| c as &dyn gkr::LeafColumns),
            ps,
        )
    });

    // Framework blocks keep their per-column claims (deduped: push/pull share ζ);
    // every table block becomes a form for the zerocheck instead.
    let mut claims: Vec<ColumnClaim> = Vec::new();
    // No table total travels: the verifier derives each tree's table share as `Ṽ₀(ζ)` less the
    // framework decomposition and the batch settles it. A transmitted total would appear in
    // exactly one check, which it could always be solved to satisfy, and would settle nothing.
    let mut forms = std::array::from_fn(|_| tables.iter().map(|&(_, n)| BusForm::new(n)).collect::<Vec<_>>());
    crate::stage!("Bus decompose", || {
        for (s, &(blocks, lay, a, g)) in sides.iter().enumerate() {
            decompose_prove(
                blocks,
                lay,
                cols,
                &bus_gkr.point,
                a,
                g,
                &owners[s],
                &mut forms[s],
                &mut claims,
                ps,
            );
        }
        prove_bytecode_share(push, &bus_gkr.point, &w, &public, ps);
    });

    BusProof {
        claims,
        point: bus_gkr.point,
        forms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gkr::{TreeField, prove_products};

    /// A coordinate's value at row `z`, from its definition.
    fn coord_value(c: &Coord, cols: &[&[F64]], public: &[&[F64]], z: usize) -> F64 {
        match c {
            Coord::Const(v) => *v,
            Coord::Col(i) => cols[*i][z],
            Coord::GCol(i, k) => g_pow(*k as usize) * cols[*i][z],
            Coord::Prod(i, j, k) => g_pow(*k as usize) * cols[*i][z] * cols[*j][z],
            Coord::Index => g_pow(z),
            Coord::Public(s) => public[*s][z],
            Coord::Sum(cs) => cs.iter().map(|c| coord_value(c, cols, public, z)).sum(),
        }
    }

    /// The leaves by their definition, every block at its offset.
    fn naive_leaves(
        blocks: &[Block],
        lay: &Layout,
        cols: &[&[F64]],
        public: &[&[F64]],
        w: &[F128],
        beta: F128,
    ) -> Vec<F128> {
        let len = blocks.iter().map(|b| 1 << b.kappa).sum::<usize>().max(1);
        let mut leaves = vec![F128::ONE; len];
        for (b, blk) in blocks.iter().enumerate() {
            for z in 0..blk.rows {
                leaves[lay.offsets[b] + z] = blk
                    .coords
                    .iter()
                    .zip(w)
                    .fold(beta, |acc, (c, &wi)| acc + wi.mul_base(coord_value(c, cols, public, z)));
            }
        }
        leaves
    }

    /// Whether GKR never reads leaf `l`, being inside a whole block of identity leaves.
    fn unread_leaf(leaves: &gkr::Leaves, l: usize) -> bool {
        leaves
            .ones
            .iter()
            .any(|run| unread(run.clone(), 0, leaves.len).contains(&l))
    }

    /// The stored blocks and every one the column tasks yield equal the definition's leaves, and the quads
    /// the product of four leaves, for all that GKR reads.
    fn check_leaves(built: &[gkr::Leaves<'static>; 2], from_columns: Option<&ColumnLeaves>, want: &[Vec<F128>]) {
        let blocks_of = |side: usize| {
            let gkr::LeafData::Blocks(ref blocks) = built[side].leaves else {
                unreachable!("the builder stores blocks")
            };
            blocks
        };
        let unstored = |side: usize, b: usize| {
            from_columns.is_some_and(|c| gkr::LeafColumns::runs(c, side).iter().any(|run| run.contains(&b)))
        };
        for (side, want) in want.iter().take(2).enumerate() {
            assert_eq!(built[side].len, want.len(), "side {side}");
            for (l, &leaf) in want.iter().enumerate() {
                let skip = unread_leaf(&built[side], l) || unstored(side, l / gkr::BLOCK_VALUES);
                if !skip {
                    assert_eq!(F128::leaf(blocks_of(side), l), leaf, "side {side} leaf {l}");
                }
            }
            for (k, &quad) in built[side].quads.iter().enumerate() {
                let inside = |run: &Range<usize>| run.start <= 4 * k && 4 * k + 4 <= run.end;
                if !built[side].ones.iter().any(inside) {
                    let expected: F128 = (0..4)
                        .map(|i| want.get(4 * k + i).copied().unwrap_or(F128::ONE))
                        .product();
                    assert_eq!(quad, expected, "side {side} quad {k}");
                }
            }
        }
        let Some(columns) = from_columns else { return };
        let mut seen = [0usize; 2];
        for t in 0..gkr::LeafColumns::tasks(columns) {
            let mut last = [None::<usize>; 2];
            gkr::LeafColumns::task(columns, t, &mut |side, b, got| {
                assert!(b % 4 == 0 || last[side] == Some(b - 1), "side={side} block={b}");
                last[side] = Some(b);
                seen[side] += 1;
                assert!(unstored(side, b), "side={side} block={b} is stored");
                for (p, vector) in got.iter().enumerate() {
                    for (j, lane) in vector.to_array().into_iter().enumerate() {
                        let l = gkr::BLOCK_VALUES * b + 8 * j + p;
                        assert_eq!(lane, want[side][l], "side={side} leaf {l}");
                    }
                }
            });
        }
        for side in 0..2 {
            let runs: usize = gkr::LeafColumns::runs(columns, side).iter().map(Range::len).sum();
            assert_eq!(seen[side], runs, "side {side}: every unstored block once");
        }
    }

    /// The builder's leaves and products are their definitions, whether it stores every leaf or leaves
    /// most to the column tasks, and its identity runs leave the proof as it is.
    #[test]
    fn built_leaves_match_the_definition() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let columns: Vec<Vec<F64>> = (0..6).map(|_| (0..1 << 12).map(|_| F64(next())).collect()).collect();
        let cols: Vec<&[F64]> = columns.iter().map(Vec::as_slice).collect();
        let bytecode: Vec<F64> = (0..1 << 11).map(|_| F64(next())).collect();
        let public = [bytecode.as_slice()];
        let all: Vec<&[F64]> = [&cols[..], &public[..]].concat();
        let block = |kappa: usize, rows: usize, coords: Vec<Coord>| Block { kappa, rows, coords };
        use Coord::*;
        let wide = |count: Coord| {
            vec![
                Const(F64(5)),
                Index,
                count,
                Prod(0, 2, 1),
                Sum(vec![Col(4), GCol(5, 3)]),
            ]
        };
        let narrow = |count: Coord| vec![Const(F64(7)), Col(3), count, Public(0)];
        let push = vec![
            block(12, 3001, wide(GCol(1, 1))),
            block(11, 1 << 11, narrow(Col(2))),
            block(2, 3, vec![Const(F64(1)), Col(4)]),
            block(0, 1, vec![Const(F64(2))]),
        ];
        let pull = vec![
            block(12, 3001, wide(Col(1))),
            block(11, 1 << 11, narrow(GCol(2, 1))),
            block(1, 2, vec![Const(F64(3)), Col(0)]),
        ];
        let count = vec![block(12, 3001, vec![Col(1)]), block(11, 1 << 11, vec![Col(2)])];
        let lays = [&push, &pull, &count].map(|b| layout(b));
        let w: Vec<F128> = (0..1 << BUS_BITS).map(|_| F128::new(next(), next())).collect();
        let count_w = count_weights();
        let beta = F128::new(27, 18);
        let sides = sides(
            [&push, &pull, &count],
            [&lays[0], &lays[1], &lays[2]],
            &w,
            &count_w,
            beta,
        );
        let want = sides.map(|(blocks, lay, w, beta)| naive_leaves(blocks, lay, &cols, &public, w, beta));
        let two = [sides[0], sides[1]];

        let (dense, none) = build_leaves(two, &all, cols.len(), false, 1 << 12);
        assert!(none.is_none());
        check_leaves(&dense, None, &want);
        let (sparse, from_columns) = build_leaves(two, &all, cols.len(), true, 1 << 12);
        let from_columns = from_columns.expect("the tall blocks have unstored leaves");
        check_leaves(&sparse, Some(&from_columns), &want);
        assert!(dense.iter().all(|leaves| !leaves.ones.is_empty()));

        // The GKR wants its first two products equal, so each side is paired with itself.
        let leaves = |side: usize| {
            build_leaves(two, &all, cols.len(), false, 1 << 12)
                .0
                .into_iter()
                .nth(side)
                .unwrap()
        };
        for side in [0, 1] {
            let mut proofs = [ProverState::from_label(b"leaves"), ProverState::from_label(b"leaves")];
            let [ps0, ps1] = &mut proofs;
            let counts = build_count_leaves(&count, &lays[2], &cols);
            let planned = gkr::prove_products_from(leaves(side), leaves(side), counts, None, ps0);
            let reference = prove_products([want[side].clone(), want[side].clone(), want[2].clone()], ps1);
            assert_eq!((planned.point, planned.values), (reference.point, reference.values));
            let [a, b] = proofs.map(|ps| ps.into_proof().stream);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn rebuild_choice_follows_workers_and_depth() {
        let always = cfg!(any(not(target_arch = "x86_64"), target_feature = "avx512f"));
        for mu in [20, AVX2_STORE_MAX_MU, AVX2_STORE_MAX_MU + 1, 30] {
            for workers in [1, 4, AVX2_REBUILD_MIN_WORKERS - 1, AVX2_REBUILD_MIN_WORKERS, 32] {
                let expected = always || workers >= 8 || mu > 25;
                assert_eq!(rebuilds_leaves(mu, workers), expected, "mu={mu} workers={workers}");
            }
        }
    }

    #[test]
    fn task_blocks_are_whole_groups_of_eight() {
        for jobs in 0..=200 {
            let per = task_blocks(jobs);
            assert!(per.is_multiple_of(8) && (8..=BLOCKS).contains(&per), "jobs={jobs}");
        }
        let expected = [(2, 128), (6, 40), (10, 24), (12, 16), (16, 16), (17, 8), (56, 8)];
        for (jobs, per) in expected {
            assert_eq!(task_blocks(jobs), per, "jobs={jobs}");
        }
    }

    /// A memory block whose columns are constant from `real` on, built with its tail and without: the same
    /// quads, and column tasks that yield the leaves of the one without.
    #[test]
    fn memory_tail_leaves_match_dense_at_boundaries() {
        let n = 1 << 14;
        let weights: Vec<F128> = (0..1 << BUS_BITS).map(|i| F128::new(i as u64 + 3, 17)).collect();
        let block = Block::full(
            14,
            vec![
                Coord::Const(F64(7)),
                Coord::Index,
                Coord::Col(0),
                Coord::Col(1),
                Coord::Col(2),
            ],
        );
        let blocks = std::slice::from_ref(&block);
        let lay = layout(blocks);
        let sides = [(blocks, &lay, &weights[..], F128::Y); 2];
        for real in [0, 1, 7, 8, 9, 63, 64, 65, 1023, 1024, 1025, 8191, 8192, 8193, n] {
            let values: Vec<_> = (0..3 * n)
                .map(|i| {
                    if i % n < real {
                        F64((i as u64).wrapping_mul(0xabcdef))
                    } else {
                        F64(u64::from(i / n == 2))
                    }
                })
                .collect();
            let cols = std::array::from_fn::<_, 3, _>(|c| &values[c * n..(c + 1) * n]);
            let (dense, _) = build_leaves(sides, &cols, cols.len(), false, n);
            let (sparse, from_columns) = build_leaves(sides, &cols, cols.len(), true, real);
            let from_columns = from_columns.unwrap();
            for side in 0..2 {
                assert_eq!(dense[side].quads, sparse[side].quads, "real={real}");
            }
            for t in 0..gkr::LeafColumns::tasks(&from_columns) {
                gkr::LeafColumns::task(&from_columns, t, &mut |side, b, got| {
                    let gkr::LeafData::Blocks(ref expected) = dense[side].leaves else {
                        unreachable!()
                    };
                    for (a, e) in got.iter().zip(expected[b]) {
                        assert_eq!(a.to_array(), e.to_array(), "real={real}");
                    }
                });
            }
        }
    }

    #[test]
    fn a_block_follows_the_one_before_it_as_their_difference_when_that_is_cheaper() {
        let w = |i: u64| F128::new(i + 2, 1);
        let block = |side: usize, offset: usize, unstored: usize, terms: Vec<Term>| LeafBlock {
            side,
            offset,
            kappa: 12,
            rows: 4096,
            const_part: F128::ZERO,
            terms,
            onto: false,
            bulk: 64,
            unstored,
            tail: None,
        };
        let push = vec![(Source::Col(0), w(0)), (Source::Col(1), w(1)), (Source::Index, w(2))];
        let near = vec![(Source::Col(0), w(0)), (Source::Col(2), w(1)), (Source::Index, w(2))];
        // One term from `near`, but at another offset.
        let elsewhere = vec![(Source::Col(0), w(0)), (Source::Col(2), w(1)), (Source::Index, w(9))];
        let blocks = [
            block(0, 0, 64, push),
            block(1, 0, 64, near),
            block(0, 4096, 64, elsewhere.clone()),
            // One term from the block before it, at its offset, but stored where it is not.
            block(1, 4096, 0, elsewhere),
        ];
        let groups = Group::all(&blocks);
        assert_eq!(groups.len(), 1);
        let chained: Vec<(usize, bool, usize)> = groups[0]
            .blocks
            .iter()
            .map(|b| (b.side, b.onto, b.terms.len()))
            .collect();
        // Sorted by offset, then side. The pull block is the difference of two terms. The block at another
        // offset and the stored one each start a chain of their own.
        assert_eq!(chained, [(0, false, 3), (1, true, 2), (0, false, 3), (1, false, 3)]);
        assert_eq!(
            groups[0].chains().map(<[LeafBlock]>::len).collect::<Vec<_>>(),
            [2, 1, 1]
        );
    }

    /// A table of `per_side` blocks a side, each of `rows` of `2^kappa` rows, over seven columns:
    /// columns 3 to 5 copy columns 0 to 2. A pull block reads the copies, so its leaves equal its
    /// push block's and the two products agree. An even pull block differs from its push block in one
    /// coordinate, an odd one in two, and every fourth also in the product it reads.
    fn bus_blocks(per_side: usize, kappa: usize, rows: usize) -> [Vec<Block>; 2] {
        let block = |i: usize, pull: bool| {
            let shift = if pull { 3 } else { 0 };
            let mut coords = vec![
                Coord::Const(F64(i as u64 + 1)),
                Coord::Col(i % 3 + shift),
                if i % 2 == 1 {
                    Coord::Prod(shift, 1, 0)
                } else {
                    Coord::Col(6)
                },
                Coord::Index,
            ];
            if pull && i % 4 == 3 {
                coords[2] = Coord::Prod(3, 4, 0);
            }
            Block { kappa, rows, coords }
        };
        [false, true].map(|pull| (0..per_side).map(|i| block(i, pull)).collect())
    }

    /// `build_leaves` with and without `columns` on a table of `per_side` blocks a side: the same quads and
    /// proofs, and the column tasks yield every unstored block once, in groups of four, equal to the dense one.
    fn check_column_tasks(per_side: usize, kappa: usize, rows: usize) {
        let n = 1usize << kappa;
        let random = |i: usize| F64((i as u64 + 5).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let values: Vec<_> = (0..7 * n)
            .map(|i| random(if (3 * n..6 * n).contains(&i) { i - 3 * n } else { i }))
            .collect();
        let cols = std::array::from_fn::<_, 7, _>(|c| &values[c * n..(c + 1) * n]);
        let weights: Vec<F128> = (0..1 << BUS_BITS).map(|i| F128::new(i as u64 + 3, 17)).collect();
        let [push, pull] = bus_blocks(per_side, kappa, rows);
        let lay = layout(&push);
        let sides = [
            (&push[..], &lay, &weights[..], F128::Y),
            (&pull[..], &lay, &weights[..], F128::Y),
        ];
        let (dense, _) = build_leaves(sides, &cols, cols.len(), false, n);
        let (sparse, from_columns) = build_leaves(sides, &cols, cols.len(), true, n);
        let from_columns = from_columns.unwrap();
        let tasks = gkr::LeafColumns::tasks(&from_columns);
        assert!(tasks > 0);
        let mut seen = [
            vec![0u8; dense[0].len.div_ceil(gkr::BLOCK_VALUES)],
            vec![0u8; dense[1].len.div_ceil(gkr::BLOCK_VALUES)],
        ];
        for t in 0..tasks {
            let mut last = [None::<usize>; 2];
            gkr::LeafColumns::task(&from_columns, t, &mut |side, b, got| {
                assert!(b % 4 == 0 || last[side] == Some(b - 1), "side={side} block={b}");
                last[side] = Some(b);
                seen[side][b] += 1;
                let gkr::LeafData::Blocks(ref expected) = dense[side].leaves else {
                    unreachable!()
                };
                for (a, e) in got.iter().zip(expected[b]) {
                    assert_eq!(a.to_array(), e.to_array(), "side={side} block={b}");
                }
            });
        }
        for side in 0..2 {
            // The quads of the padding rows are left unwritten.
            for b in 0..per_side {
                let start = lay.offsets[b] / 4;
                let end = start + rows / 4;
                assert_eq!(
                    dense[side].quads[start..end],
                    sparse[side].quads[start..end],
                    "side={side} block={b}"
                );
            }
            let mut unstored = vec![0u8; seen[side].len()];
            for run in gkr::LeafColumns::runs(&from_columns, side) {
                run.clone().for_each(|b| unstored[b] += 1);
            }
            assert!(unstored.iter().any(|&u| u > 0), "no unstored block");
            assert_eq!(seen[side], unstored, "side={side}");
        }
        let prove = |[push, pull]: [gkr::Leaves<'static>; 2], columns: Option<&dyn gkr::LeafColumns>| {
            let len = push.len;
            let mut ps = ProverState::from_label(b"fine-column-tasks");
            let result = gkr::prove_products_from(push, pull, gkr::Leaves::new(vec![F128::Y; len]), columns, &mut ps);
            (result, ps.into_proof().stream)
        };
        let (actual, actual_proof) = prove(sparse, Some(&from_columns));
        let (expected, expected_proof) = prove(dense, None);
        assert_eq!(actual.values, expected.values);
        assert_eq!(actual.point, expected.point);
        assert_eq!(actual_proof, expected_proof);
    }

    #[test]
    fn column_tasks_of_many_blocks_match_dense() {
        // 68 jobs: eight blocks a task. Rows are not a multiple of eight GKR blocks.
        check_column_tasks(34, 12, 4096 - 100);
    }

    #[test]
    fn column_tasks_of_few_blocks_match_dense() {
        // 12 jobs: sixteen blocks a task.
        check_column_tasks(6, 12, 3000);
    }

    #[test]
    fn borrowed_count_columns_match_dense_proofs() {
        let mut cases = vec![
            vec![],
            vec![(0, 0)],
            vec![(0, 1)],
            vec![(1, 1), (0, 1)],
            vec![(11, 65), (10, 513), (4, 7), (0, 1), (9, 0), (3, 2)],
        ];
        for rows in [
            0, 1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 8191, 8192, 8193, 16384,
        ] {
            cases.push(vec![(14, rows)]);
        }
        for case in cases {
            let values: Vec<Vec<F64>> = case
                .iter()
                .enumerate()
                .map(|(b, &(_, rows))| (0..rows).map(|i| F64((13 * i + 97 * b + 2) as u64)).collect())
                .collect();
            let cols: Vec<&[F64]> = values.iter().map(Vec::as_slice).collect();
            let blocks: Vec<Block> = case
                .iter()
                .enumerate()
                .map(|(i, &(kappa, rows))| Block {
                    kappa,
                    rows,
                    coords: vec![Coord::Col(i)],
                })
                .collect();
            let lay = layout(&blocks);
            let sparse = build_count_leaves(&blocks, &lay, &cols);
            let len = sparse.len;
            let mut dense = vec![F64::ONE; len];
            for (b, data) in values.iter().enumerate() {
                dense[lay.offsets[b]..lay.offsets[b] + data.len()].copy_from_slice(data);
            }
            let prove = |count| {
                let mut ps = ProverState::from_label(b"borrowed-count-columns");
                let result = gkr::prove_products_from(
                    gkr::Leaves::new(vec![F128::Y; len]),
                    gkr::Leaves::new(vec![F128::Y; len]),
                    count,
                    None,
                    &mut ps,
                );
                (result, ps.into_proof().stream)
            };
            let (actual, actual_proof) = prove(sparse);
            let (expected, expected_proof) = prove(gkr::Leaves::new(dense));
            assert_eq!(actual.values, expected.values, "{case:?}");
            assert_eq!(actual.point, expected.point, "{case:?}");
            assert_eq!(actual_proof, expected_proof, "{case:?}");
        }
    }
}
