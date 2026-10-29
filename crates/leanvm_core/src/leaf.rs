//! The bus: a single shared channel balanced by a grand product (§sec:gp through §sec:leafstack). Each
//! interaction wires a table's columns into width-`m` tuples and flushes them in a
//! direction; the bus balances when pushed and pulled tuples form the same
//! multiset, proven by two GKR passes over the leaf vectors `β − π_α(σ)`. Each pass
//! reduces to a leaf claim `Ṽ₀(ζ)`, decomposed into evaluation claims on the
//! committed columns. Tuple coordinates `σ_i` are `K`-valued (column entries,
//! g-powers, separators); the fingerprint challenges `α, β` are `E`-valued, so a
//! leaf accumulates via the mixed `mul_base` product (2 PMULL per coordinate).

use crate::PAR_THRESHOLD;
use crate::constraints::{self, TableBlocks};
use crate::gkr;
use crate::transcript::ProverState;
use flux_profiler::timed;
use primitives::field::{
    Acc8, F64, F64x8, F192, F192Unreduced, F192x8, Weights8, dot_base, g_pow, index_mle, mul_base_splat8, mul_f64x8,
    mul_slices,
};
use primitives::multilinear::{eq_eval_bits, eq_table, mle_eval_many};
use primitives::stream::Stream;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

/// One tuple coordinate as a function of the block's row `z`.
#[derive(Clone, Debug)]
pub enum Coord {
    /// A public constant (domain separator, opcode, the seed count `1`).
    Const(F64),
    /// A committed column, value `col[z]`.
    Col(usize),
    /// The free increment `g^k · col[z]` (a virtual column, §sec:vm): `k = 1` for the
    /// count/state steps, `k ≤ 12` for a KECCAK state's consecutive cells.
    GCol(usize, u32),
    /// The product `g^k · col_a[z] · col_b[z]` of two committed columns. An address
    /// is `fp·g^o`, so this carries one on the bus without committing it: the
    /// coordinate IS the product, so no column can disagree with it and the binding
    /// constraint that used to say so is unnecessary (§sec:m3).
    Prod(usize, usize, u32),
    /// The index column `g^z` (§sec:idxcol), free via the factored MLE.
    Index,
    /// A public column (the bytecode program, §sec:e2e-bc): not committed; both parties form
    /// its MLE directly, so it raises no claim. Shared rather than owned: push and
    /// pull carry the same eleven columns, tens of megabytes at production sizes.
    Public(Arc<Vec<F64>>),
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
/// the rest are a table's padding rows (§sec:e2e-pad), whose leaves are `1`, and
/// `padding` is the tuple at the table's padding row.
#[derive(Clone, Debug)]
pub struct Block {
    pub kappa: usize,
    pub rows: usize,
    pub coords: Vec<Coord>,
    pub padding: Vec<F64>,
}

impl Block {
    /// A block every row of which flushes.
    pub fn full(kappa: usize, coords: Vec<Coord>) -> Self {
        Self {
            kappa,
            rows: 1 << kappa,
            coords,
            padding: Vec::new(),
        }
    }
}

/// `pad_n(point)`, the multilinear extension of `[z ≥ n]`, `n ≤ 2^|point|` (lemma lem:suffix).
fn padding_weight(point: &[F192], n: usize) -> F192 {
    let bit = |k: usize| F192::new(((n >> k) & 1) as u64, 0, 0);
    let low = point.iter().enumerate().fold(F192::ONE, |pad, (k, &r)| {
        (F192::ONE + bit(k)) * r + (F192::ONE + bit(k) + r) * pad
    });
    (F192::ONE + bit(point.len())) * low
}

/// A table coordinate's value on a row whose local columns are `row`.
pub(crate) fn coord_at(c: &Coord, row: &[F64]) -> F64 {
    match c {
        Coord::Const(v) => *v,
        Coord::Col(i) => row[*i],
        Coord::GCol(i, k) => g_pow(*k as usize) * row[*i],
        Coord::Prod(i, j, k) => g_pow(*k as usize) * row[*i] * row[*j],
        Coord::Sum(cs) => cs.iter().fold(F64::ZERO, |acc, c| acc + coord_at(c, row)),
        Coord::Index | Coord::Public(_) => unreachable!("a table's bus block carries no virtual coordinate"),
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
    pub point: Vec<F192>,
    pub value: F192,
}

/// The fingerprint weights `eq(α⃗, x)` over the `2^N_TUPLE_BITS` slots (§sec:gp).
/// A tuple is fingerprinted as `Σ_x eq(α⃗, x)·σ_x`, a MULTILINEAR combination
/// rather than a power chain: each leaf factor is then of total degree
/// `N_TUPLE_BITS` in the challenges instead of the tuple width, and slot `x`'s
/// weight is an `eq` weight, which is what lets the aligned bytecode polynomial
/// be read off at `α⃗` itself (§sec:e2e-bc).
pub fn fingerprint_weights(alphas: &[F192]) -> Vec<F192> {
    debug_assert_eq!(alphas.len(), N_TUPLE_BITS);
    let mut w = vec![F192::ONE; 1 << N_TUPLE_BITS];
    for (bit, &a) in alphas.iter().enumerate() {
        for (x, wx) in w.iter_mut().enumerate() {
            *wx *= if (x >> bit) & 1 == 1 { a } else { a + F192::ONE };
        }
    }
    w
}

/// Bits indexing a bus tuple's coordinates: `m = 14` coordinates live in the
/// `2^4` slots of the bytecode encoding (§sec:m3, §sec:e2e-bc).
pub const N_TUPLE_BITS: usize = 4;

/// Conservative sum of the degree bounds for every random-challenge failure in
/// the bus argument. A side contains at most `2^mu` leaf factors, each of total
/// degree `N_TUPLE_BITS` in `α⃗` and one in `β`. The second term covers all
/// radix-four GKR batching and sumcheck challenges.
fn soundness_degree_bound(mu: usize) -> u128 {
    assert!(mu < u128::BITS as usize, "bus layout is too large to bound");
    let fingerprint = (N_TUPLE_BITS as u128 + 1) * (1u128 << mu);
    let gkr = 8u128 * (mu as u128 + 1).pow(2);
    fingerprint + gkr
}

fn soundness_bits(mu: usize) -> u32 {
    let degree = soundness_degree_bound(mu);
    192u32.saturating_sub(u128::BITS - degree.leading_zeros())
}

/// Check that the 192-bit challenge field supplies the target bus soundness.
/// Push and pull have the same logical height, and the count product is checked
/// by the same GKR rather than by a separate root-at-random test.
fn assert_grinding_unnecessary(
    push_blocks: &[Block],
    pull_blocks: &[Block],
    push: &Layout,
    pull: &Layout,
    count: &Layout,
) {
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
    assert!(widest <= 1 << N_TUPLE_BITS, "a tuple's coordinates index its slots");
    assert!(
        soundness_bits(push.mu) >= crate::SECURITY_BITS,
        "bus layout exceeds the unground F192 soundness budget"
    );
}

/// Stack blocks largest-first at aligned offsets; `μ = ⌈log2 Σ 2^{κ_b}⌉`.
pub fn layout(blocks: &[Block]) -> Layout {
    let kappas: Vec<Option<usize>> = blocks.iter().map(|b| Some(b.kappa)).collect();
    let (offsets, placed) = crate::witness::stack_offsets(&kappas);
    Layout {
        mu: crate::log2_ceil_usize(placed.max(1)),
        offsets,
    }
}

/// The non-constant terms of a block's leaf: each contributes the mixed product `coeff · source(z)`
/// with `source(z) ∈ K`, `coeff ∈ E`. A column, the index column and a public column are all one
/// slice read at `z`; `GCol` folds its `g^k` factor into the coefficient.
#[derive(Default)]
struct Terms<'a> {
    lins: Vec<(&'a [F64], F192)>,
    prods: Vec<(&'a [F64], &'a [F64], F192)>,
}

/// Flatten one coordinate into leaf terms at coefficient `w`. A [`Coord::Sum`]
/// spreads its children over the SAME `w`: they are one coordinate, so they share
/// its `α`-power.
fn push_terms<'a>(
    c: &'a Coord,
    w: F192,
    cols: &[&'a [F64]],
    gpow: &'a [F64],
    terms: &mut Terms<'a>,
    constant: &mut F192,
) {
    match c {
        Coord::Const(v) => *constant += w.mul_base(*v),
        Coord::Col(i) => terms.lins.push((cols[*i], w)),
        Coord::GCol(i, k) => terms.lins.push((cols[*i], w.mul_base(g_pow(*k as usize)))),
        Coord::Prod(i, j, k) => terms.prods.push((cols[*i], cols[*j], w.mul_base(g_pow(*k as usize)))),
        Coord::Index => terms.lins.push((gpow, w)),
        Coord::Public(vals) => terms.lins.push((vals.as_slice(), w)),
        Coord::Sum(cs) => {
            for c in cs {
                push_terms(c, w, cols, gpow, terms, constant);
            }
        }
    }
}

/// Eight leaves of a first layer in one register: `F192x8` for the bus sides, `F64x8` for the
/// count side, whose leaves stay in the base field.
trait Lanes: Copy + std::ops::Mul<Output = Self> {
    type Elem: Copy;
    const ZERO: Self;
    fn transpose(g: [Self; 8]) -> [Self; 8];
    fn even_odd(a: Self, b: Self) -> (Self, Self);
    fn to_array(self) -> [Self::Elem; 8];
}

impl Lanes for F192x8 {
    type Elem = F192;
    const ZERO: Self = F192x8::ZERO;

    fn transpose(g: [Self; 8]) -> [Self; 8] {
        F192x8::transpose(g)
    }

    fn even_odd(a: Self, b: Self) -> (Self, Self) {
        F192x8::even_odd(a, b)
    }

    fn to_array(self) -> [F192; 8] {
        F192x8::to_array(self)
    }
}

impl Lanes for F64x8 {
    type Elem = F64;
    const ZERO: Self = F64x8::ZERO;

    fn transpose(g: [Self; 8]) -> [Self; 8] {
        F64x8::transpose(g)
    }

    fn even_odd(a: Self, b: Self) -> (Self, Self) {
        F64x8::even_odd(a, b)
    }

    fn to_array(self) -> [F64; 8] {
        F64x8::to_array(self)
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
    compute: impl Fn(usize) -> G,
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

/// One leaf of a block at row `z`: the α-weighted coordinate sum defers its reductions. Each mixed
/// product contributes its three raw limb products (3 PMULL, no reduction tail) and one combined
/// reduction ends the row, bit-identical to summing reduced `mul_base` terms.
fn leaf_row((constant, terms): &(F192, Terms), z: usize) -> F192 {
    let mut acc = F192Unreduced::ZERO;
    for (k, c) in &terms.lins {
        acc ^= c.mul_base_unreduced(k[z]);
    }
    for (a, b, c) in &terms.prods {
        acc ^= c.mul_base_unreduced(a[z] * b[z]);
    }
    *constant + acc.reduce()
}

/// Eight leaves of a block from row `z`: every mixed product of a term is one wide kernel call,
/// and the reduction happens once per eight rows.
fn leaf_rows8((constant, terms): &(F192, Terms), z: usize) -> F192x8 {
    fn eight(k: &[F64], z: usize) -> &[F64; 8] {
        k[z..z + 8].try_into().expect("eight words")
    }
    let mut acc = Acc8::ZERO;
    for (k, c) in &terms.lins {
        acc ^= mul_base_splat8(*c, eight(k, z));
    }
    for (a, b, c) in &terms.prods {
        acc ^= mul_base_splat8(*c, &mul_f64x8(eight(a, z), eight(b, z)));
    }
    acc.reduce_lanes() + F192x8::splat(*constant)
}

/// GKR blocks per task of the passes over the columns.
const BLOCKS: usize = 128;

/// One side's leaf vector while it is built, with the leaf terms of each block.
struct Side<'a> {
    blocks: &'a [Block],
    lay: &'a Layout,
    explicit: usize,
    tiled: bool,
    /// Whether the pairs of GKR blocks inside a block's bulk are left unstored for GKR to recompute.
    columns: bool,
    leaves: Vec<gkr::Block>,
    quads: Vec<F192>,
    quads_done: Vec<usize>,
    /// Levels 4 and 6 of the blocks `publish_task` builds, in the `staged` runs of blocks; empty
    /// when the side has none.
    tiers: [Vec<F192>; 2],
    staged: Vec<Range<usize>>,
    terms: Vec<(F192, Terms<'a>)>,
}

impl Side<'_> {
    /// Whole GKR blocks of block `b`'s leaves, built in parallel. Blocks off the GKR block
    /// boundary or too few for the pool go row by row.
    fn bulk(&self, b: usize) -> usize {
        let blk = &self.blocks[b];
        let aligned = self.lay.offsets[b].is_multiple_of(gkr::BLOCK_VALUES);
        if blk.rows >= PAR_THRESHOLD && aligned {
            blk.rows / gkr::BLOCK_VALUES
        } else {
            0
        }
    }

    /// The leading GKR blocks of block `b`'s bulk that are not stored: an even number, so that an odd
    /// last block stays stored, where it pairs with the block after it.
    fn unstored(&self, b: usize) -> usize {
        if self.columns { self.bulk(b) & !1 } else { 0 }
    }
}

/// The blocks of both bus sides that [`build_leaves`] left unstored, and what computes them again
/// from the columns, in the same groups and tasks as the build.
pub struct ColumnLeaves<'a> {
    terms: [Vec<(F192, Terms<'a>)>; 2],
    offsets: [&'a [usize]; 2],
    /// `(rows, side, block)` of the blocks with unstored leaves, sorted by `rows`.
    jobs: Vec<(usize, usize, usize)>,
    runs: [Vec<Range<usize>>; 2],
    tasks: usize,
}

impl ColumnLeaves<'_> {
    /// The unstored GKR blocks of a block of `rows` rows, as [`Side::unstored`] counts them.
    fn blocks(rows: usize) -> usize {
        (rows / gkr::BLOCK_VALUES) & !1
    }

    /// The tasks of the group of `rows`: one per `BLOCKS` unstored blocks.
    fn chunks(rows: usize) -> usize {
        Self::blocks(rows).div_ceil(BLOCKS)
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
        let mut t = t;
        for group in self.jobs.chunk_by(|x, y| x.0 == y.0) {
            let chunks = Self::chunks(group[0].0);
            if t >= chunks {
                t -= chunks;
                continue;
            }
            for &(rows, s, b) in group {
                let off = self.offsets[s][b];
                for k in t * BLOCKS..((t + 1) * BLOCKS).min(Self::blocks(rows)) {
                    let block = F192x8::transpose(std::array::from_fn(|j| {
                        leaf_rows8(&self.terms[s][b], gkr::BLOCK_VALUES * k + 8 * j)
                    }));
                    f(s, off / gkr::BLOCK_VALUES + k, &block);
                }
            }
            return;
        }
    }
}

/// Build the leaf vectors of several sides: block `b` row `z` holds `β − Σ_i w_i c_i(z)` for
/// the fingerprint weights `w = eq(α⃗, ·)`, followed implicitly by the identity `1`
/// up to `2^μ`. The row-invariant weights and constant coordinates are folded once
/// per block into `const_part`. `gpow` supplies `g^z` for [`Coord::Index`] and is
/// the caller's, shared by the sides.
///
/// The blocks of all sides with the same number of rows read the same table, so one pass runs
/// them together over each chunk of rows: the table's columns come from DRAM once for all of them,
/// and every block publishes exactly what its own pass would.
///
/// With `columns`, the leaves of the bulk of a block are not stored, but for an odd last GKR block:
/// the returned [`ColumnLeaves`] computes them again from the columns, which costs less DRAM
/// traffic than reading them back. A side that does not tile its layout keeps all its leaves.
#[timed]
pub fn build_leaves<'a>(
    sides: &[(&'a [Block], &'a Layout); 2],
    cols: &[&'a [F64]],
    w: &[F192],
    beta: F192,
    gpow: &'a [F64],
    columns: bool,
) -> (Vec<gkr::Leaves<'static>>, Option<ColumnLeaves<'a>>) {
    let mut sides: Vec<Side> = sides
        .iter()
        .map(|&(blocks, lay)| {
            let explicit = blocks
                .iter()
                .enumerate()
                .map(|(b, block)| lay.offsets[b] + (1usize << block.kappa))
                .max()
                .unwrap_or(1);
            debug_assert!(explicit <= 1usize << lay.mu);
            // `stack_offsets` packs the power-of-two blocks contiguously from zero, so the
            // blocks tile `0..explicit` and every slot below is written by one of them: the
            // identity fill would be overwritten in full, and this is the largest buffer in
            // the proof. The `covered` test is what licenses skipping it, so a layout that
            // ever left a hole falls back to filling rather than reading uninitialized rows.
            // The leaves are GKR's blocks of 64, so the identity past `explicit` is written to
            // the end of the last one, which is where GKR expects it. The padding rows inside the layout
            // are runs of identity that GKR skips (`Leaves::ones`), so their whole blocks stay unwritten.
            let covered: usize = blocks.iter().map(|blk| 1usize << blk.kappa).sum();
            let block_count = explicit.div_ceil(gkr::BLOCK_VALUES);
            let tiled = covered == explicit;
            let mut leaves: Vec<gkr::Block> = if tiled {
                // SAFETY: the per-block fills below cover `0..explicit` but for the identity blocks GKR
                // never reads, each joins before this function returns, and the tail of the last block
                // is written next.
                unsafe { primitives::uninit_vec(block_count) }
            } else {
                vec![[F192x8::ONE; 8]; block_count]
            };
            for l in explicit..gkr::BLOCK_VALUES * block_count {
                gkr::set_leaf(&mut leaves, l, F192::ONE);
            }
            // The first radix-four layer, `quads[k] = leaves[4k] · … · leaves[4k + 3]`, built from the
            // staged rows while they are in L1, so GKR never reads the leaves to build it.
            //
            // SAFETY: every slot is written before this function returns: the staged groups write
            // theirs, and `finish_quads` writes every slot past them.
            let quads = unsafe { primitives::uninit_vec::<F192>(explicit.div_ceil(4)) };
            let terms = blocks
                .iter()
                .map(|blk| {
                    let mut const_part = beta;
                    let mut terms = Terms::default();
                    for (i, c) in blk.coords.iter().enumerate() {
                        push_terms(c, w[i], cols, gpow, &mut terms, &mut const_part);
                    }
                    (const_part, terms)
                })
                .collect();
            Side {
                blocks,
                lay,
                explicit,
                tiled,
                columns: columns && tiled,
                leaves,
                quads,
                quads_done: lay.offsets[..blocks.len()].to_vec(),
                tiers: [Vec::new(), Vec::new()],
                staged: Vec::new(),
                terms,
            }
        })
        .collect();
    for side in &mut sides {
        if (0..side.blocks.len()).any(|b| side.bulk(b) > 0) {
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
    let mut jobs: Vec<(usize, usize, usize)> = (0..sides.len())
        .flat_map(|s| (0..sides[s].blocks.len()).map(move |b| (s, b)))
        .filter(|&(s, b)| sides[s].bulk(b) > 0)
        .map(|(s, b)| (sides[s].blocks[b].rows, s, b))
        .collect();
    jobs.sort_by_key(|&(rows, ..)| rows);
    for group in jobs.chunk_by(|x, y| x.0 == y.0) {
        let bulk = group[0].0 / gkr::BLOCK_VALUES;
        for &(_, s, b) in group {
            let off = sides[s].lay.offsets[b] / gkr::BLOCK_VALUES;
            sides[s]
                .staged
                .extend((0..bulk.div_ceil(BLOCKS)).map(|c| off + c * BLOCKS..off + ((c + 1) * BLOCKS).min(bulk)));
        }
        let ptrs: Vec<_> = sides
            .iter_mut()
            .map(|s| {
                let [l4, l6] = s.tiers.each_mut().map(|t| parallel::SendPtr(t.as_mut_ptr()));
                (
                    parallel::SendPtr(s.leaves.as_mut_ptr()),
                    parallel::SendPtr(s.quads.as_mut_ptr()),
                    [l4, l6],
                )
            })
            .collect();
        parallel::for_each(bulk.div_ceil(BLOCKS), |c| {
            for &(_, s, b) in group {
                let side = &sides[s];
                let off = side.lay.offsets[b];
                let first = off / gkr::BLOCK_VALUES + c * BLOCKS;
                let len = BLOCKS.min(bulk - c * BLOCKS);
                let columns = side.unstored(b).saturating_sub(c * BLOCKS).min(BLOCKS);
                // SAFETY: task `c` owns these blocks of the leaves, their quads and their tiers, which no
                // other task or block of the group touches.
                let (task, quads, tiers) = unsafe {
                    (
                        ptrs[s].0.slice(first, len),
                        ptrs[s].1.slice(16 * first, 16 * len),
                        [ptrs[s].2[0].slice(4 * first, 4 * len), ptrs[s].2[1].slice(first, len)],
                    )
                };
                publish_task(Some(task), quads, tiers, first..first + len, columns, |l| {
                    leaf_rows8(&side.terms[b], l - off)
                });
            }
        });
    }
    let mut runs = [Vec::new(), Vec::new()];
    let mut offsets = [&[][..]; 2];
    let mut column_terms = [Vec::new(), Vec::new()];
    let column_jobs: Vec<_> = jobs
        .iter()
        .copied()
        .filter(|&(_, s, b)| sides[s].unstored(b) > 0)
        .collect();
    for &(_, s, b) in &column_jobs {
        let first = sides[s].lay.offsets[b] / gkr::BLOCK_VALUES;
        runs[s].push(first..first + sides[s].unstored(b));
    }
    for (s, run) in runs.iter_mut().enumerate() {
        run.sort_unstable_by_key(|run| run.start);
        offsets[s] = &sides[s].lay.offsets;
    }
    let leaves = sides
        .into_iter()
        .enumerate()
        .map(|(s, mut side)| {
            let (blocks, lay, explicit, tiled) = (side.blocks, side.lay, side.explicit, side.tiled);
            for b in 0..blocks.len() {
                let (blk, off) = (&blocks[b], lay.offsets[b]);
                let bulk = side.bulk(b);
                side.quads_done[b] = off + gkr::BLOCK_VALUES * bulk;
                for z in gkr::BLOCK_VALUES * bulk..blk.rows {
                    gkr::set_leaf(&mut side.leaves, off + z, leaf_row(&side.terms[b], z));
                }
                // A padding row's leaf is the identity, so it leaves the product alone.
                if tiled {
                    fill_ones(&mut side.leaves, off + blk.rows, off + (1usize << blk.kappa), explicit);
                }
            }
            if !runs[s].is_empty() {
                column_terms[s] = std::mem::take(&mut side.terms);
            }
            let Side {
                leaves,
                mut quads,
                quads_done,
                tiers,
                mut staged,
                ..
            } = side;
            staged.sort_unstable_by_key(|run| run.start);
            let mut ones: Vec<Range<usize>> = Vec::new();
            if tiled {
                finish_quads(&leaves, &mut quads, blocks, lay, &quads_done);
                let mut runs: Vec<_> = blocks
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
    let tasks = column_jobs
        .chunk_by(|x, y| x.0 == y.0)
        .map(|group| ColumnLeaves::chunks(group[0].0))
        .sum();
    let columns = (!column_jobs.is_empty()).then_some(ColumnLeaves {
        terms: column_terms,
        offsets,
        jobs: column_jobs,
        runs,
        tasks,
    });
    (leaves, columns)
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
        gkr::set_leaf(leaves, l, F192::ONE);
    }
}

/// Write every quad the staged groups did not: the rest of each block from its leaves, and
/// whole-identity runs of its padding as the identity.
#[timed]
fn finish_quads(leaves: &[gkr::Block], quads: &mut [F192], blocks: &[Block], lay: &Layout, quads_done: &[usize]) {
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
            quads[k] = F192::ONE;
        }
    }
    for k in small_from..quads.len() {
        quads[k] = quad(k);
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
    pub coeffs: Vec<F192>,
    /// `(col_a, col_b, coeff)` in LOCAL column indices.
    pub prods: Vec<(usize, usize, F192)>,
    pub constant: F192,
}

impl BusForm {
    fn new(n_cols: usize) -> Self {
        Self {
            coeffs: vec![F192::ZERO; n_cols],
            prods: Vec::new(),
            constant: F192::ZERO,
        }
    }

    /// The same form scaled by `w`. Every coefficient is `E`-valued already, so
    /// folding the side's `η`-power in here costs three multiplies once per table
    /// instead of one per [`eval_unreduced`](Self::eval_unreduced), which the
    /// zerocheck calls per row per round.
    pub fn scaled(&self, w: F192) -> Self {
        Self {
            coeffs: self.coeffs.iter().map(|&c| c * w).collect(),
            prods: self.prods.iter().map(|&(a, b, c)| (a, b, c * w)).collect(),
            constant: self.constant * w,
        }
    }

    /// The pointwise sum of several forms over the same columns.
    ///
    /// Evaluating the sum is evaluating each and adding, and the constraint batch
    /// only ever wants a table's total, so its three bus sides collapse to one
    /// dot product and one product list: the row loop then reads the column
    /// values once for all three rather than once each.
    pub fn sum(forms: impl IntoIterator<Item = Self>) -> Self {
        let mut forms = forms.into_iter();
        let mut out = forms.next().expect("a table has at least one bus form");
        for form in forms {
            assert_eq!(out.coeffs.len(), form.coeffs.len(), "forms over the same columns");
            for (slot, c) in out.coeffs.iter_mut().zip(form.coeffs) {
                *slot += c;
            }
            out.constant += form.constant;
            for (a, b, c) in form.prods {
                match out.prods.iter_mut().find(|p| (p.0, p.1) == (a, b)) {
                    Some(p) => p.2 += c,
                    None => out.prods.push((a, b, c)),
                }
            }
        }
        out.prods.retain(|p| p.2 != F192::ZERO);
        out
    }

    /// The form at one point, unreduced: `evals` are the columns' values there.
    /// This is what the zerocheck evaluates, per row while a table is unfolded and
    /// at the sumcheck point after, so a table's several forms share one reduction
    /// rather than paying one per term.
    /// `quadratic` selects only degree-two terms, for a sumcheck round coefficient.
    pub fn eval_unreduced(&self, evals: &[F192], quadratic: bool) -> F192Unreduced {
        let linear = if quadratic {
            F192Unreduced::ZERO
        } else {
            self.linear_unreduced(evals)
        };
        self.prods
            .iter()
            .fold(linear, |acc, &(a, b, c)| acc ^ (evals[a] * evals[b]).mul_unreduced(c))
    }

    /// `constant + Σ coeffs[i]·evals[i]`, unreduced.
    fn linear_unreduced(&self, evals: &[F192]) -> F192Unreduced {
        self.coeffs
            .iter()
            .zip(evals)
            .fold(F192Unreduced::from(self.constant), |acc, (&w, &v)| {
                acc ^ w.mul_unreduced(v)
            })
    }

    /// What the form sums to over the table's rows against `eq(ζ, ·)`, the target the
    /// zerocheck settles. The linear part factors through the columns' evaluations at
    /// `ζ`, which is the whole point of a form; a product coordinate does NOT, so
    /// `prod_sums` supplies `Σ_z eq(ζ,z)·col_a(z)·col_b(z)` for each pair it uses.
    fn sum_at(&self, evals: &[F192], prod_sums: &[(usize, usize, F192)]) -> F192 {
        self.prods
            .iter()
            .fold(self.linear_unreduced(evals).reduce(), |acc, &(a, b, c)| {
                let s = prod_sums
                    .iter()
                    .find(|p| (p.0, p.1) == (a, b))
                    .expect("every pair was summed");
                acc + c * s.2
            })
    }
}

/// Accumulate one coordinate of a table's block into that table's form, at
/// coefficient `w`. A [`Coord::Sum`]'s children share `w`, so a derived value
/// lands as the several coefficients and products it is made of.
fn accumulate_form(c: &Coord, w: F192, base: usize, form: &mut BusForm) {
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

/// Walk one side's blocks. A block owned by table `t` (with column base `base`)
/// accumulates into `forms[t]`; the framework blocks are decomposed into per-column
/// claims as before, `fresh` supplying values not already in `claims`. Returns the
/// framework blocks' contribution to `Ṽ₀(ζ)` plus the padding mass, so the caller
/// can settle the side once the zerocheck has proven the tables' forms.
///
/// A table's form is summed over all its rows, padding rows included, where the leaf
/// holds `1` instead: the difference, `padding_weight · (β + fingerprint(pad) + 1)`, is
/// public, so it joins the padding mass.
fn decompose_formula<F: FnMut(usize, &[F192]) -> F192>(
    blocks: &[Block],
    lay: &Layout,
    zeta: &[F192],
    w: &[F192],
    beta: F192,
    owners: &[Option<(usize, usize)>],
    forms: &mut [BusForm],
    claims: &mut Vec<ColumnClaim>,
    public: &mut PublicEvals,
    mut fresh: F,
) -> F192 {
    assert_eq!(zeta.len(), lay.mu);
    let mut acc = F192::ZERO;
    let mut sel_sum = F192::ZERO;
    for (b, blk) in blocks.iter().enumerate() {
        let kappa = blk.kappa;
        let zeta_lo = &zeta[..kappa];
        let zeta_hi = &zeta[kappa..];
        let sel = lay.offsets[b] >> kappa;
        let eq_hi = eq_eval_bits(sel, zeta_hi);
        sel_sum += eq_hi;

        // A table's block becomes a linear form the zerocheck will sum; only the
        // framework blocks (boundary, memory, bytecode) still open columns at ζ.
        if let Some((t, base)) = owners[b] {
            let form = &mut forms[t];
            form.constant += eq_hi * beta;
            for (i, c) in blk.coords.iter().enumerate() {
                accumulate_form(c, eq_hi * w[i], base, form);
            }
            let phi_padding = blk
                .padding
                .iter()
                .zip(w)
                .fold(beta, |acc, (&v, &wi)| acc + wi.mul_base(v));
            acc += eq_hi * padding_weight(zeta_lo, blk.rows) * (F192::ONE + phi_padding);
            continue;
        }

        // Column `i` at ζ_lo: reuse the recorded claim, else take a fresh value and
        // record it. The push ORDER is the stream order, so every coordinate that
        // needs a column value must go through here.
        let mut col_val = |i: usize| -> F192 {
            if let Some(v) = known_claim(claims, i, zeta_lo) {
                return v;
            }
            let v = fresh(i, zeta_lo);
            claims.push(ColumnClaim {
                col: i,
                point: zeta_lo.to_vec(),
                value: v,
            });
            v
        };
        let mut inner = F192::ZERO;
        for (i, c) in blk.coords.iter().enumerate() {
            let coord_val = match c {
                Coord::Const(v) => F192::from(*v),
                Coord::Index => index_mle(zeta_lo),
                Coord::Col(i) => col_val(*i),
                Coord::GCol(i, k) => col_val(*i).mul_base(g_pow(*k as usize)),
                Coord::Prod(..) | Coord::Sum(..) => {
                    unreachable!("only a table's bus block carries a degree-2 coordinate")
                }
                Coord::Public(vals) => public_eval(vals, zeta_lo, public),
            };
            inner += w[i] * coord_val;
        }
        acc += eq_hi * (beta + inner);
    }
    // The padding rows (identity `1`) contribute the leftover mass `1 - Σ_b sel_b`.
    acc + (F192::ONE + sel_sum)
}

/// Look up an already-recorded claim on `(col, point)`. Push and pull share
/// their GKR point, so a column read by both sides (or by two same-κ blocks of
/// one side) is streamed and opened ONCE; later occurrences reuse the value.
fn known_claim(claims: &[ColumnClaim], col: usize, point: &[F192]) -> Option<F192> {
    claims
        .iter()
        .find(|c| c.col == col && c.point == point)
        .map(|c| c.value)
}

// One bus GKR point per cache; its prefixes are keyed by length and shared column identity.
type PublicEvals = HashMap<(usize, usize), F192>;

fn public_eval(vals: &Arc<Vec<F64>>, point: &[F192], cache: &mut PublicEvals) -> F192 {
    *cache
        .entry((Arc::as_ptr(vals) as usize, point.len()))
        .or_insert_with(|| primitives::multilinear::mle_eval_par(vals, point))
}

/// The FRESH committed coords of one side, as `decompose_formula` visits them (blocks in order, coords
/// in order, `Col`/`GCol` only, first occurrence per `(col, point)`, the same dedup as `known_claim`),
/// and the public columns it evaluates, each once per `(column, point length)`.
fn fresh_coords<'a>(
    blocks: &'a [Block],
    owners: &[Option<(usize, usize)>],
    claims: &[ColumnClaim],
    zeta: &[F192],
    jobs: &mut Vec<(usize, usize)>,
    publics: &mut Vec<(&'a Arc<Vec<F64>>, usize)>,
) {
    for (blk, _) in blocks.iter().zip(owners).filter(|(_, o)| o.is_none()) {
        for c in &blk.coords {
            match c {
                Coord::Col(i) | Coord::GCol(i, _) => {
                    let job = (*i, blk.kappa);
                    if known_claim(claims, *i, &zeta[..blk.kappa]).is_none() && !jobs.contains(&job) {
                        jobs.push(job);
                    }
                }
                Coord::Public(vals) if !publics.iter().any(|&(v, k)| Arc::ptr_eq(v, vals) && k == blk.kappa) => {
                    publics.push((vals, blk.kappa));
                }
                _ => {}
            }
        }
    }
}

/// Prover-side decomposition: writes each FRESH committed value onto the stream and records the
/// matching claim (block/coord order); duplicates reuse the recorded value.
///
/// Within one `decompose_formula` call no challenge is sampled between claims (`zeta`, `alpha`, `beta`
/// are fixed arguments and each claim's point is `zeta[..kappa]` of its block), so the values are
/// independent of the transcript and only their `add_scalar` ORDER matters. The caller evaluates them
/// for every side at once, in the order of `fresh_coords`; this replays them through the transcript in
/// the original block/coord order, keeping the stream byte-identical to the serial form.
fn decompose_prove<'a>(
    blocks: &[Block],
    lay: &Layout,
    zeta: &[F192],
    w: &[F192],
    beta: F192,
    owners: &[Option<(usize, usize)>],
    forms: &mut [BusForm],
    claims: &mut Vec<ColumnClaim>,
    public: &mut PublicEvals,
    fresh: &mut impl Iterator<Item = (&'a (usize, usize), &'a F192)>,
    ps: &mut ProverState,
) -> F192 {
    decompose_formula(
        blocks,
        lay,
        zeta,
        w,
        beta,
        owners,
        forms,
        claims,
        public,
        |col, zeta_lo| {
            let (&(jc, jk), &v) = fresh
                .next()
                .expect("job enumeration matches decompose_formula's col_val order");
            debug_assert_eq!((jc, jk), (col, zeta_lo.len()), "job/coord order drift");
            ps.add_scalar(v);
            v
        },
    )
}

/// Selector bits of the stacked bytecode polynomial: the public encoding
/// columns (opcode + ten operand/immediate slots = eleven) stack along
/// `2^N_BYTECODE_SELECTORS` slots. A column's slot is its bus tuple coordinate,
/// which is what fixes the width at sixteen rather than at the column count.
pub const N_BYTECODE_SELECTORS: usize = 4;

/// Slot of the first public column: the bytecode block leads with three
/// non-public coordinates (tag, index, read count), and a column's slot IS its
/// bus tuple coordinate.
pub const BYTECODE_PUBLIC_SLOT: usize = 3;

/// The stacked bytecode polynomial as a dense table: eleven public encoding
/// columns at their tuple coordinates, padded to sixteen selector slots. One
/// multilinear polynomial B̃ in `κ_bc + 4` variables, which is what a bus leaf
/// reads at `(ζ_lo, α⃗)` (§sec:e2e-bc); the outermost verifier evaluates it.
pub fn stacked_bytecode_table(blocks: &[Block]) -> Vec<F64> {
    let mut kbc = 0;
    let mut cols: Vec<&[F64]> = Vec::new();
    for blk in blocks {
        for c in &blk.coords {
            if let Coord::Public(vals) = c {
                kbc = blk.kappa;
                cols.push(vals.as_slice());
            }
        }
    }
    let mut table = vec![F64::ZERO; 1 << (N_BYTECODE_SELECTORS + kbc)];
    for (i, vals) in cols.into_iter().enumerate() {
        let slot = BYTECODE_PUBLIC_SLOT + i;
        assert!(slot < 1 << N_BYTECODE_SELECTORS, "a public slot is a tuple coordinate");
        assert_eq!(vals.len(), 1 << kbc);
        table[(slot << kbc)..((slot + 1) << kbc)].copy_from_slice(vals);
    }
    table
}

/// The count side reads its base-field leaves directly from the columns. Only product levels
/// are stored, with whole blocks of identity padding left implicit.
#[timed]
fn build_count_leaves<'a>(blocks: &[Block], lay: &Layout, cols: &[&'a [F64]]) -> gkr::Leaves<'a, F64> {
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
    let [mut quads, mut tier4, mut tier6] = primitives::filled_vecs([16 * block_count, 4 * block_count, block_count], |bufs| {
        let ptrs = bufs.map(|buf| parallel::SendPtr(buf.as_mut_ptr()));
        parallel::for_each(staged.len(), |c| {
            let run = staged[c].clone();
            let (first, len) = (run.start, run.len());
            // SAFETY: the staged runs are disjoint, and so are their blocks of the product levels.
            let (quads, tiers) = unsafe {
                (
                    ptrs[0].slice(16 * first, 16 * len),
                    [ptrs[1].slice(4 * first, 4 * len), ptrs[2].slice(first, len)],
                )
            };
            publish_task(None, quads, tiers, run, len, |l| F64x8::from_array(&columns.read::<8>(l)));
        });
    });
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

/// The three bus sides in `[push, pull, count]` order with their fingerprint
/// weights. The count channel's leaf is the count itself (a single `Col`), so it
/// runs at `α⃗ = 0` (weight `1` on slot `0`, zero elsewhere), `β = 0`, and its GKR
/// root is the product of all counts.
fn sides<'a>(
    blocks: [&'a [Block]; 3],
    lays: [&'a Layout; 3],
    w: &'a [F192],
    count_w: &'a [F192],
    beta: F192,
) -> [(&'a [Block], &'a Layout, &'a [F192], F192); 3] {
    [
        (blocks[0], lays[0], w, beta),
        (blocks[1], lays[1], w, beta),
        (blocks[2], lays[2], count_w, F192::ZERO),
    ]
}

/// Prove the bus balances; returns the per-column claims to open (§sec:leafstack). `alpha`/
/// `beta` follow the witness commitment (the only ordering the grand product
/// needs), and the block structure is public, so no shape is observed.
/// Everything the bus hands on: the framework blocks' column claims,
/// the shared GKR point (the table sumcheck's eq point), and
/// per side the tables' linear forms plus what each is claimed to sum to.
pub struct BusProof {
    pub claims: Vec<ColumnClaim>,
    /// The GKR point ζ: the zerocheck reuses it, so no fresh point is sampled.
    pub point: Vec<F192>,
    /// `forms[side][table]`, in `[push, pull, count]` order.
    pub forms: [Vec<BusForm>; 3],
    /// `sigmas[side][table]`: each form's eq-weighted sum over its table's rows.
    /// Prover-side only. NOTHING here travels: the batch's target is the caller's
    /// derived `Σ_s η^·totals[s]`, and the shares serve only to build each round's
    /// waiting line, which rides inside the round polynomial.
    pub sigmas: [Vec<F192>; 3],
    /// `blocks[table]`: the per-block dots the table sums came from, for the constraint sumcheck.
    pub blocks: Vec<TableBlocks>,
}

#[timed]
pub fn prove_balance(
    push: &[Block],
    pull: &[Block],
    count: &[Block],
    cols: &[&[F64]],
    owners: &[Vec<Option<(usize, usize)>>; 3],
    tables: &[(usize, usize)],
    rows: &[usize],
    ps: &mut ProverState,
) -> BusProof {
    let push_lay = layout(push);
    let pull_lay = layout(pull);
    let mut count_lay = layout(count);
    assert_grinding_unnecessary(push, pull, &push_lay, &pull_lay, &count_lay);
    let alphas: Vec<F192> = (0..N_TUPLE_BITS).map(|_| ps.sample()).collect();
    let w = fingerprint_weights(&alphas);
    let count_w = fingerprint_weights(&[F192::ZERO; N_TUPLE_BITS]);
    let beta = ps.sample();
    // The `g^z` table backing `Coord::Index`, built once for the three sides: push
    // and pull would otherwise build the same table twice and count, which carries
    // no `Index` coordinate, would build one it never reads.
    let index_k = [push, pull, count]
        .into_iter()
        .flatten()
        .filter(|b| b.coords.iter().any(|c| matches!(c, Coord::Index)))
        .map(|b| b.kappa)
        .max();
    let gpow = index_k.map_or_else(Vec::new, |k| primitives::field::g_powers(1usize << k));
    // Push and pull are built together, so a table's columns are read once for both sides.
    // With several threads, GKR recomputes most of their leaves from the columns instead of storing them.
    let columns = parallel::num_threads() > 1;
    let (push_leaves, pull_leaves, count_leaves, column_leaves) = crate::stage!("Bus leaves", || {
        let (leaves, column_leaves) =
            build_leaves(&[(push, &push_lay), (pull, &pull_lay)], cols, &w, beta, &gpow, columns);
        let [push_leaves, pull_leaves]: [_; 2] = leaves.try_into().ok().expect("two sides");
        (
            push_leaves,
            pull_leaves,
            build_count_leaves(count, &count_lay, cols),
            column_leaves,
        )
    });
    // Leaf construction keeps the all-one padding implicit; decomposition uses the full logical depth.
    count_lay.mu = push_lay.mu;
    // All three trees run as ONE RLC-batched GKR (equal μ: push/pull match
    // block-for-block, count is padded), so every claim lands on ONE point ζ.
    let bus_gkr = crate::stage!("Bus GKR", || {
        gkr::prove_product_triple_from(
            push_leaves,
            pull_leaves,
            count_leaves,
            column_leaves.as_ref().map(|c| c as &dyn gkr::LeafColumns),
            ps,
            gkr::RootShape::FirstTwoShared,
        )
    });

    // Framework blocks keep their per-column claims (deduped: push/pull share ζ);
    // every table block becomes a form for the zerocheck instead.
    let mut claims: Vec<ColumnClaim> = Vec::new();
    let sides = sides(
        [push, pull, count],
        [&push_lay, &pull_lay, &count_lay],
        &w,
        &count_w,
        beta,
    );
    // Each table's columns at ζ[..τ], computed once and shared by the three sides
    // (a form's linear part factors through them). Nothing here travels, neither the
    // evaluations nor any total: the verifier derives each side's table share as `Ṽ₀(ζ)` less the
    // framework decomposition and the batch settles it. A
    // transmitted total would appear in exactly one check, which it could always be
    // solved to satisfy, and would settle nothing.
    let mut forms = std::array::from_fn(|_| tables.iter().map(|&(_, n)| BusForm::new(n)).collect::<Vec<_>>());
    let mut frameworks = [F192::ZERO; 3];
    let mut public = PublicEvals::new();
    crate::stage!("Bus decompose", || {
        // Every value the three sides take fresh, and every public column they evaluate, in one dispatch.
        let (mut jobs, mut publics) = (Vec::new(), Vec::new());
        for (s, &(blocks, ..)) in sides.iter().enumerate() {
            fresh_coords(blocks, &owners[s], &claims, &bus_gkr.point, &mut jobs, &mut publics);
        }
        let tasks: Vec<(&[F64], &[F192])> = jobs
            .iter()
            .map(|&(col, kappa)| (cols[col], &bus_gkr.point[..kappa]))
            .chain(
                publics
                    .iter()
                    .map(|&(vals, kappa)| (&vals[..], &bus_gkr.point[..kappa])),
            )
            .collect();
        let values = mle_eval_many(&tasks);
        let (fresh, public_values) = values.split_at(jobs.len());
        for (&(vals, kappa), &v) in publics.iter().zip(public_values) {
            public.insert((Arc::as_ptr(vals) as usize, kappa), v);
        }
        let mut fresh = jobs.iter().zip(fresh);
        for (s, &(blocks, lay, a, g)) in sides.iter().enumerate() {
            frameworks[s] = decompose_prove(
                blocks,
                lay,
                &bus_gkr.point,
                a,
                g,
                &owners[s],
                &mut forms[s],
                &mut claims,
                &mut public,
                &mut fresh,
                ps,
            );
        }
    });
    let (table_evals, prod_sums, blocks) = crate::stage!("Bus table evals", || tables_and_prods_at(
        cols,
        tables,
        rows,
        &forms,
        &bus_gkr.point
    ));
    let sigmas: [Vec<F192>; 3] = std::array::from_fn(|s| {
        let sigmas: Vec<F192> = forms[s]
            .iter()
            .zip(&table_evals)
            .zip(&prod_sums)
            .map(|((f, e), p)| f.sum_at(e, p))
            .collect();
        // Completeness only: the verifier derives this identity rather than checking
        // it, so a mismatch here is a prover bug, not a rejection path.
        debug_assert_eq!(
            sigmas.iter().fold(frameworks[s], |acc, &b| acc + b),
            bus_gkr.values[s],
            "side {s} must decompose into its leaf value"
        );
        sigmas
    });

    BusProof {
        claims,
        point: bus_gkr.point,
        forms,
        sigmas,
        blocks,
    }
}

/// Every table's committed columns at `ζ[..τ_t]`, and, for every column pair its
/// forms multiply, `Σ_z eq(ζ[..τ], z)·col_a(z)·col_b(z)`. A table of more than one block also
/// returns its per-block dots, which the sums are the `high`-weighted total of.
///
/// The eq table is a tensor `high[h]·low[l]`, so a column's sum is `Σ_h high[h]` times
/// the dot of one block of rows against the packed `low`, which stays in L1. Nothing
/// of length `2^τ` is built but the columns' own base products. Pairs are deduped
/// across the three sides and the several blocks that carry the same address, so an
/// address costs one pass however often it is flushed. `tables[t] = (base, n_cols)` in
/// the global schema; a pair names LOCAL column indices.
///
/// A block wholly past `rows[t]` holds the padding row in every row, so each column is a constant
/// `v` there and its dot is `v`, since `low` sums to one. Those blocks are not swept: each
/// accumulator takes `v` times the sum of their `high` weights, and each dot is written as `v`.
#[allow(clippy::type_complexity)]
#[timed]
fn tables_and_prods_at(
    cols: &[&[F64]],
    tables: &[(usize, usize)],
    rows: &[usize],
    forms: &[Vec<BusForm>; 3],
    zeta: &[F192],
) -> (Vec<Vec<F192>>, Vec<Vec<(usize, usize, F192)>>, Vec<TableBlocks>) {
    /// Variables of the packed `low` table: `2^LOW_VARS` rows per block.
    const LOW_VARS: usize = constraints::BLOCK_VARS;
    /// Blocks per task.
    const BLOCKS: usize = 4;

    let (evals, (prods, blocks)) = tables
        .iter()
        .zip(rows)
        .enumerate()
        .map(|(t, (&(base, n_cols), &rows))| {
            let tau = crate::log2_strict_usize(cols[base].len());
            let mut pairs: Vec<(usize, usize)> = forms
                .iter()
                .flat_map(|side| side[t].prods.iter().map(|&(a, b, _)| (a, b)))
                .collect();
            pairs.sort_unstable();
            pairs.dedup();

            let low_vars = tau.min(LOW_VARS);
            let mut low = eq_table(&zeta[..low_vars]);
            // A table below one vector of rows is zero-padded.
            low.resize(low.len().max(8), F192::ZERO);
            let low: Vec<Weights8> = low.as_chunks::<8>().0.iter().map(Weights8::new).collect();
            let high = eq_table(&zeta[low_vars..tau]);
            let block = 1usize << low_vars;
            let dot = |rows: &[F64]| {
                let mut padded = [F64::ZERO; 8];
                let rows = if rows.len() < 8 {
                    padded[..rows.len()].copy_from_slice(rows);
                    &padded[..]
                } else {
                    rows
                };
                dot_base(&low, rows).reduce()
            };

            let n_acc = n_cols + pairs.len();
            // A table of one block keeps no dots: nothing sums over blocks.
            let n_blocks = if tau > LOW_VARS { high.len() } else { 0 };
            // Blocks with a real row; the rest are constant. A table of one block is never skipped.
            let real = if n_blocks > 0 { rows.div_ceil(block) } else { high.len() };
            let mut sums = Vec::new();
            // The tasks and the constant tail write every dot, the one of accumulator `a` at block `h` to
            // `a * n_blocks + h`: a task owns blocks `task * BLOCKS ..` of each accumulator's run.
            let dots = primitives::filled_vec::<F192>(n_blocks * n_acc, |dots| {
                let mut parts: Vec<&mut [F192]> = Vec::with_capacity(n_acc * n_blocks.div_ceil(BLOCKS));
                if n_blocks > 0 {
                    let mut runs: Vec<_> = dots.chunks_mut(n_blocks).map(|run| run.chunks_mut(BLOCKS)).collect();
                    for _ in 0..n_blocks.div_ceil(BLOCKS) {
                        parts.extend(runs.iter_mut().filter_map(Iterator::next));
                    }
                }
                let fold = |prod: &mut Vec<F64>, acc: &mut Vec<F192Unreduced>, task: usize, out: &mut [&mut [F192]]| {
                    let keep = |out: &mut [&mut [F192]], a: usize, h: usize, d: F192| {
                        if let Some(run) = out.get_mut(a) {
                            run[h - task * BLOCKS] = d;
                        }
                    };
                    for h in task * BLOCKS..((task + 1) * BLOCKS).min(real) {
                        let span = |c: usize| &cols[base + c][h * block..(h + 1) * block];
                        for (c, slot) in acc[..n_cols].iter_mut().enumerate() {
                            let d = dot(span(c));
                            keep(out, c, h, d);
                            *slot ^= high[h].mul_unreduced(d);
                        }
                        for (p, (&(a, b), slot)) in pairs.iter().zip(&mut acc[n_cols..]).enumerate() {
                            mul_slices(prod, span(a), span(b));
                            let d = dot(prod);
                            keep(out, n_cols + p, h, d);
                            *slot ^= high[h].mul_unreduced(d);
                        }
                    }
                };
                let combine = |mut left: Vec<F192Unreduced>, right: Vec<F192Unreduced>| {
                    for (slot, part) in left.iter_mut().zip(right) {
                        *slot ^= part;
                    }
                    left
                };
                let (init_prod, init_acc) = (|| vec![F64::ZERO; block], || vec![F192Unreduced::ZERO; n_acc]);
                sums = if n_blocks > 0 {
                    parallel::map_reduce_chunks_mut(
                        [&mut parts[..]],
                        [n_acc],
                        init_prod,
                        init_acc,
                        |prod, acc, task, [out]| fold(prod, acc, task, out),
                        combine,
                    )
                } else {
                    parallel::map_reduce_with_state(
                        real.div_ceil(BLOCKS),
                        init_prod,
                        init_acc,
                        |prod, acc, task| fold(prod, acc, task, &mut []),
                        combine,
                    )
                };
                if real < high.len() {
                    let mass = high[real..].iter().fold(F192::ZERO, |acc, &w| acc + w);
                    let row = |c: usize| cols[base + c][real * block];
                    let constants = (0..n_cols)
                        .map(row)
                        .chain(pairs.iter().map(|&(a, b)| row(a) * row(b)));
                    for (a, (v, slot)) in constants.zip(&mut sums).enumerate() {
                        *slot ^= mass.mul_base_unreduced(v);
                        if n_blocks > 0 {
                            for h in real..high.len() {
                                parts[h / BLOCKS * n_acc + a][h % BLOCKS] = F192::from(v);
                            }
                        }
                    }
                }
            });
            let evals = sums[..n_cols].iter().map(|s| s.reduce()).collect();
            let prods = pairs
                .iter()
                .zip(&sums[n_cols..])
                .map(|(&(a, b), s)| (a, b, s.reduce()))
                .collect();
            let blocks = TableBlocks {
                n_blocks,
                n_cols,
                pairs,
                dots,
            };
            (evals, (prods, blocks))
        })
        .unzip();
    (evals, prods, blocks)
}

#[cfg(test)]
mod tests {
    use super::*;

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
                    padding: vec![F64::ONE],
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
                let result = gkr::prove_product_triple_from(
                    gkr::Leaves::new(vec![F192::Y; len]),
                    gkr::Leaves::new(vec![F192::Y + F192::ONE; len]),
                    count,
                    None,
                    &mut ps,
                    gkr::RootShape::Distinct,
                );
                (result, ps.into_proof().stream)
            };
            let (actual, actual_proof) = prove(sparse);
            let (expected, expected_proof) = prove(gkr::Leaves::new(dense));
            assert_eq!(actual.roots, expected.roots, "{case:?}");
            assert_eq!(actual.values, expected.values, "{case:?}");
            assert_eq!(actual.point, expected.point, "{case:?}");
            assert_eq!(actual_proof, expected_proof, "{case:?}");
        }
    }

    /// The bound is `(N_TUPLE_BITS + 1)·2^mu` plus the GKR terms: only the bus
    /// DEPTH costs bits now, the multilinear fingerprint having fixed each factor's
    /// degree at four in `α⃗` and one in `β`, whatever the tuple's width.
    #[test]
    fn bus_soundness_tracks_depth_only() {
        assert!(soundness_bits(38) >= crate::SECURITY_BITS);
        assert!(soundness_bits(61) >= crate::SECURITY_BITS);
        assert!(soundness_bits(62) < crate::SECURITY_BITS);
    }
}
