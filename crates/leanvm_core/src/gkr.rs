//! The grand product via GKR (§sec:gkr): given leaves `v_0…v_{2^μ-1}`, prove the
//! root `P = ∏ v_k` of the product tree, reducing the root to one leaf evaluation
//! `Ṽ_0(ζ)`. Two binary levels are contracted at a time: a radix-four layer has
//! relation `V_i(x)=∏_{a,b∈{0,1}}V_{i-2}(a,b,x)`. Its normalized eq-trick
//! sumcheck has degree four. An odd-depth tree starts with one binary layer.
//! Leaves and layers are `E`-valued (the bus fingerprints mix `K`-columns into `E`
//! upstream, [`crate::leaf`]), except that a tree whose leaves are `K`-valued stays in `K`
//! until its first fold with an `E` challenge ([`TreeField`]).

use crate::PAR_THRESHOLD;
use crate::transcript::ProverState;
use flux_profiler::timed;
use primitives::field::{F64, F64x8, F192, F192Unreduced, F192x8, Prod8, mul_unreduced4, mul2, mul4};
use primitives::multilinear::{eq_table, interp};
use primitives::stream::Stream;
use std::ops::{Deref, Range};

/// Rows per parallel window: enough tasks to keep every thread fed, but not so few
/// rows per task that dispatch dominates.
fn window_rows(total: usize) -> usize {
    let tasks = parallel::num_threads() * 16;
    total.div_ceil(tasks).clamp(64, 1 << 10)
}

/// A product tree's leaves with its first radix-four layer, `quads[k] = leaves[4k] · … ·
/// leaves[4k + 3]` (absent leaves are the identity), built by whoever wrote the leaves.
///
/// The leaves cover `len` values, stored as blocks or borrowed from columns, with the identity after.
///
/// `ones` are sorted, disjoint runs of leaves that are the identity. The GKR never reads a whole
/// block of 64 inside a run, at the leaves or at any level above them (see [`identity_blocks`]),
/// so the builder may leave those blocks of `leaves` and `quads` unwritten.
///
/// A builder that has the quads in hand may also write the levels above them for whole blocks of 64
/// leaves: `tiers` are levels 4 and 6, of the length the levels have (or empty), and `staged` are
/// sorted, disjoint runs of blocks. A run holds the level-4 entries of its blocks, and the level-6
/// entry of each group of eight blocks that starts at a multiple of eight and lies inside the run.
/// [`build_layers`] computes the entries no run holds.
pub struct Leaves<'a, F: TreeField = F192> {
    pub leaves: LeafData<'a, F>,
    pub len: usize,
    pub quads: Vec<F>,
    pub ones: Vec<Range<usize>>,
    pub tiers: [Vec<F>; 2],
    pub staged: Vec<Range<usize>>,
}

impl<F: TreeField> Leaves<'_, F> {
    pub fn new(values: Vec<F>) -> Self {
        let leaves = blocks_from_values(&values);
        let quads = (0..values.len().div_ceil(4)).map(|k| quad_at(&leaves, k)).collect();
        Self {
            leaves: LeafData::Blocks(leaves),
            len: values.len(),
            quads,
            ones: Vec::new(),
            tiers: [Vec::new(), Vec::new()],
            staged: Vec::new(),
        }
    }
}

/// Stored blocks, or columns read directly until the first fold.
pub enum LeafData<'a, F: TreeField> {
    Blocks(Vec<F::Block>),
    Columns(Columns<'a, F>),
}

impl<F: TreeField> Default for LeafData<'_, F> {
    fn default() -> Self {
        Self::Blocks(Vec::new())
    }
}

impl<'a, F: TreeField> LeafData<'a, F> {
    fn leaf(&self, l: usize) -> F {
        match self {
            Self::Blocks(blocks) => F::leaf(blocks, l),
            Self::Columns(columns) => columns.read::<1>(l)[0],
        }
    }

    fn state(self, len: usize, width: usize) -> QuaternaryLayerState<'a> {
        match self {
            Self::Blocks(blocks) => F::state_from_blocks(blocks, len, width),
            Self::Columns(columns) if len <= BLOCK_VALUES => {
                QuaternaryLayerState::from_rows((0..len).map(|l| columns.read::<1>(l)[0].embed()).collect(), width)
            }
            Self::Columns(columns) => QuaternaryLayerState::from_table(Box::new(columns), width),
        }
    }
}

/// Sorted, disjoint columns at their leaf offsets, with identity values between them.
pub struct Columns<'a, F> {
    pub segments: Vec<(usize, &'a [F])>,
    pub len: usize,
}

impl<F: TreeField> Columns<'_, F> {
    pub fn read<const N: usize>(&self, start: usize) -> [F; N] {
        let mut out = [F::ONE; N];
        let first = self.segments.partition_point(|&(off, data)| off + data.len() <= start);
        for &(off, data) in self.segments[first..].iter().take_while(|&&(off, _)| off < start + N) {
            let (lo, hi) = (start.max(off), (start + N).min(off + data.len()));
            out[lo - start..hi - start].copy_from_slice(&data[lo - off..hi - off]);
        }
        out
    }
}

/// The blocks of 64 values of a level holding `len` values (the products of `2^level` leaves) that
/// lie wholly inside `ones`, a run of leaves being the identity at every level. A level of one
/// block or less is read as rows, so it has none.
pub(crate) fn identity_blocks(ones: &[Range<usize>], level: usize, len: usize) -> Vec<Range<usize>> {
    let shift = BLOCK_VALUES.ilog2() as usize + level;
    let whole = |run: &Range<usize>| run.start.div_ceil(1 << shift)..run.end >> shift;
    if len <= BLOCK_VALUES {
        return Vec::new();
    }
    ones.iter().map(whole).filter(|run| !run.is_empty()).collect()
}

/// The runs of blocks after a fold: a folded block is the identity when both its sources are. A
/// single block is always computed, since the last round reads it.
fn halve(ones: &[Range<usize>], count: usize) -> Vec<Range<usize>> {
    if count <= 1 {
        return Vec::new();
    }
    ones.iter()
        .map(|run| run.start.div_ceil(2)..run.end / 2)
        .filter(|run| !run.is_empty())
        .collect()
}

/// Whether `blocks` lies inside one of the runs.
fn covers(runs: &[Range<usize>], blocks: Range<usize>) -> bool {
    runs.get(runs.partition_point(|run| run.end <= blocks.start))
        .is_some_and(|run| run.start <= blocks.start && blocks.end <= run.end)
}

fn in_runs(ones: &[Range<usize>], b: usize) -> bool {
    ones.get(ones.partition_point(|run| run.end <= b))
        .is_some_and(|run| run.start <= b)
}

/// The blocks of `range` outside `ones`.
fn live(ones: &[Range<usize>], range: Range<usize>) -> impl Iterator<Item = usize> + '_ {
    let mut at = range.start;
    let mut next = ones.partition_point(|run| run.end <= at);
    std::iter::from_fn(move || {
        while let Some(run) = ones.get(next).filter(|run| run.start <= at) {
            at = run.end;
            next += 1;
        }
        (at < range.end).then(|| {
            at += 1;
            at - 1
        })
    })
}

/// Sixty-four consecutive base-field table values, laid out as a [`Block`].
pub type KBlock = [F64x8; 8];

/// A block of table values: what a tree's first round reads, whichever field holds them.
pub trait TableBlock: primitives::Plain + Send + Sync {
    /// `quartic_summand` of the block's eight row pairs, one pair per lane, weighted by `equality` and
    /// left as four unreduced lane-wise sums.
    fn summand(&self, equality: F192x8) -> [Prod8; 4];

    /// The fold of the block by `challenge`: `V[c] + r (V[c] + V[c + 4])` for each child `c`.
    fn fold(&self, challenge: F192x8) -> [F192x8; 4];

    /// The block of identity rows.
    const ONE: Self;

    /// Adds the first two round polynomials of the blocks `self` and `second`, one pair of blocks,
    /// weighted by `weight`, to `sum` ([`bivariate_summand`]).
    fn bivariate(&self, second: &Self, weight: F192x8, sum: &mut BivariateSum);
}

/// The field a product tree's values live in: `E`, or `K` for a tree that the first fold lifts to
/// `E`. The two share every level's shape, so a tree is built and proved once, generically, and
/// only the arithmetic differs.
pub trait TreeField: primitives::Plain + Send + Sync + std::ops::Mul<Output = Self> + 'static {
    type Block: TableBlock;
    const ONE: Self;

    fn embed(self) -> F192;

    /// `c[0] · c[1] · c[2] · c[3]`.
    fn product4(c: [Self; 4]) -> Self;

    /// The products of eight consecutive rows of four, from their 32 children: an even-odd split
    /// pairs neighbours, so two rounds of it leave four-fold products in order.
    fn eight_products(children: &[Self]) -> [Self; 8];

    /// The block of up to 64 values, the missing tail being the identity.
    fn block(values: &[Self]) -> Self::Block;

    /// Leaf `l` of a block table.
    fn leaf(blocks: &[Self::Block], l: usize) -> Self;

    /// The layer state over a table of `len` values held as blocks, whose tail is the identity.
    fn state_from_blocks(blocks: Vec<Self::Block>, len: usize, width: usize) -> QuaternaryLayerState<'static>;
}

impl TreeField for F192 {
    type Block = Block;
    const ONE: Self = F192::ONE;

    fn embed(self) -> F192 {
        self
    }

    #[inline(always)]
    fn product4([c0, c1, c2, c3]: [Self; 4]) -> Self {
        let [lo, hi] = mul2([c0, c2], [c1, c3]);
        lo * hi
    }

    #[inline(always)]
    fn eight_products(children: &[Self]) -> [Self; 8] {
        let [a, b, c, d] = std::array::from_fn(|i| F192x8::from_array(children[8 * i..8 * i + 8].try_into().unwrap()));
        let pairs = |x, y| {
            let (even, odd) = F192x8::even_odd(x, y);
            even * odd
        };
        let (even, odd) = F192x8::even_odd(pairs(a, b), pairs(c, d));
        (even * odd).to_array()
    }

    #[inline(always)]
    fn block(values: &[Self]) -> Block {
        if let Ok(chunk) = <&[F192; BLOCK_VALUES]>::try_from(values) {
            // Group `j` of eight values is lane `j` of the eight vectors.
            return F192x8::transpose(std::array::from_fn(|j| {
                F192x8::from_array(chunk[8 * j..8 * j + 8].try_into().unwrap())
            }));
        }
        let mut block = [F192x8::ZERO; 8];
        for (p, vector) in block.iter_mut().enumerate() {
            for j in 0..8 {
                vector.set_lane(j, values.get(8 * j + p).copied().unwrap_or(F192::ONE));
            }
        }
        block
    }

    #[inline(always)]
    fn leaf(blocks: &[Block], l: usize) -> Self {
        blocks[l / BLOCK_VALUES][l % 8].lane(l / 8 % 8)
    }

    fn state_from_blocks(blocks: Vec<Block>, len: usize, width: usize) -> QuaternaryLayerState<'static> {
        QuaternaryLayerState::from_blocks(blocks, len, width)
    }
}

impl TreeField for F64 {
    type Block = KBlock;
    const ONE: Self = F64::ONE;

    fn embed(self) -> F192 {
        self.into()
    }

    #[inline(always)]
    fn product4([c0, c1, c2, c3]: [Self; 4]) -> Self {
        (c0 * c1) * (c2 * c3)
    }

    #[inline(always)]
    fn eight_products(children: &[Self]) -> [Self; 8] {
        let [a, b, c, d] = std::array::from_fn(|i| F64x8::from_array(children[8 * i..8 * i + 8].try_into().unwrap()));
        let pairs = |x, y| {
            let (even, odd) = F64x8::even_odd(x, y);
            even * odd
        };
        let (even, odd) = F64x8::even_odd(pairs(a, b), pairs(c, d));
        (even * odd).to_array()
    }

    #[inline(always)]
    fn block(values: &[Self]) -> KBlock {
        let mut padded = [F64::ONE; BLOCK_VALUES];
        let chunk = match <&[F64; BLOCK_VALUES]>::try_from(values) {
            Ok(chunk) => chunk,
            Err(_) => {
                padded[..values.len()].copy_from_slice(values);
                &padded
            }
        };
        F64x8::transpose(std::array::from_fn(|j| {
            F64x8::from_array(chunk[8 * j..8 * j + 8].try_into().unwrap())
        }))
    }

    #[inline(always)]
    fn leaf(blocks: &[KBlock], l: usize) -> Self {
        blocks[l / BLOCK_VALUES][l % 8].lane(l / 8 % 8)
    }

    fn state_from_blocks(blocks: Vec<KBlock>, len: usize, width: usize) -> QuaternaryLayerState<'static> {
        QuaternaryLayerState::from_k_blocks(blocks, len, width)
    }
}

/// Overwrite leaf `l` of a block table.
pub(crate) fn set_leaf(blocks: &mut [Block], l: usize, value: F192) {
    blocks[l / BLOCK_VALUES][l % 8].set_lane(l / 8 % 8, value);
}

/// `leaves[4k] · … · leaves[4k + 3]` of a block table, whose tail is the identity.
pub(crate) fn quad_at<F: TreeField>(leaves: &[F::Block], k: usize) -> F {
    F::product4(std::array::from_fn(|i| F::leaf(leaves, 4 * k + i)))
}

/// A product tree's levels: `layers[level]` holds the products of `2^level` leaves, built only for
/// the levels radix four consumes (`0, 2, 4, …`, plus the binary root when the depth is odd), and
/// the leaves stay in their original representation until the first fold.
struct Levels<'a, F: TreeField> {
    layers: Vec<Vec<F>>,
    bottom: LeafData<'a, F>,
    len: usize,
    ones: Vec<Range<usize>>,
}

/// A tree as the proof reads it, whatever field its values are in.
trait Tree<'a> {
    /// The product of the tree's leaves.
    fn root(&self) -> F192;

    /// The explicit nodes of a level, of which the root-most levels have at most two.
    fn nodes(&self, level: usize) -> Vec<F192>;

    /// The layer state over the children of a layer, the table two levels below it.
    fn state(&mut self, layer: usize, width: usize) -> QuaternaryLayerState<'a>;
}

impl<'a, F: TreeField> Tree<'a> for Levels<'a, F> {
    fn root(&self) -> F192 {
        self.layers.last().expect("a tree has a root level")[0].embed()
    }

    fn nodes(&self, level: usize) -> Vec<F192> {
        self.layers[level].iter().map(|node| node.embed()).collect()
    }

    fn state(&mut self, layer: usize, width: usize) -> QuaternaryLayerState<'a> {
        let level = layer - 2;
        let mut state = if layer == 2 {
            std::mem::take(&mut self.bottom).state(self.len, width)
        } else {
            QuaternaryLayerState::new(std::mem::take(&mut self.layers[level]), width)
        };
        state.ones = identity_blocks(&self.ones, level, self.len.div_ceil(1 << level));
        state
    }
}

#[timed]
fn build_layers<F: TreeField>(leaves: Leaves<'_, F>, mu: usize) -> Levels<'_, F> {
    let Leaves {
        leaves,
        len,
        quads,
        ones,
        tiers,
        staged,
    } = leaves;
    let mut tiers = tiers.map(Some);
    assert!(len > 0);
    assert!(len <= 1usize << mu);
    assert_eq!(quads.len(), len.div_ceil(4));
    // At mu = 22 the leaf level alone is hundreds of megabytes, and every level
    // dies with the proof. Only a tree too small for a radix-four layer reads its leaves here,
    // as rows: otherwise they stay blocks until the last layer.
    let mut layers: Vec<Vec<F>> = (0..=mu).map(|_| Vec::new()).collect();
    let mut level = 0;
    if mu < 2 {
        layers[0] = (0..len).map(|l| leaves.leaf(l)).collect();
    }
    if mu >= 2 {
        layers[2] = quads;
        level = 2;
    }
    while level + 2 <= mu {
        let current = &layers[level];
        let skip_in = identity_blocks(&ones, level, current.len());
        let skip_out = identity_blocks(&ones, level + 2, current.len().div_ceil(4));
        let full_rows = current.len() / 4;
        let product = |row: usize| F::product4(std::array::from_fn(|i| current[4 * row + i]));
        let mut next: Vec<F> = if current.len() == 1 {
            Vec::from_iter([current[0]])
        } else if current.len() == 2 {
            Vec::from_iter([current[0] * current[1]])
        } else {
            // The builder wrote the entries of the staged blocks, whose rows the chunks below skip.
            let tier = tiers
                .get_mut(level / 2 - 1)
                .and_then(Option::take)
                .filter(|tier| !tier.is_empty());
            let from_builder = tier.is_some();
            let vector_rows = full_rows / 8 * 8;
            const CHUNK: usize = 512;
            // Eight products come from 32 children, which lie in one block of 64 or none, and
            // from the leaves of `2^(level - 1)` blocks of 64.
            let span = 1usize << (level - 1);
            let fill = |c: usize, out: &mut [F]| {
                for (i, slots) in out.chunks_exact_mut(8).enumerate() {
                    let g = c * CHUNK / 8 + i;
                    if from_builder && covers(&staged, g * span..(g + 1) * span) {
                        continue;
                    }
                    if !in_runs(&skip_in, g / 2) {
                        slots.copy_from_slice(&F::eight_products(&current[32 * g..][..32]));
                    } else if !in_runs(&skip_out, g / 8) {
                        slots.fill(F::ONE);
                    }
                }
            };
            // The vector rows and the scalar rows after them cover every slot.
            let fill_rows = |next: &mut [F]| {
                if vector_rows >= PAR_THRESHOLD {
                    parallel::chunks_mut(&mut next[..vector_rows], CHUNK, fill);
                } else {
                    fill(0, &mut next[..vector_rows]);
                }
                for row in vector_rows..full_rows {
                    next[row] = product(row);
                }
            };
            match tier {
                Some(mut next) => {
                    next.truncate(full_rows);
                    fill_rows(&mut next);
                    next
                }
                None => primitives::filled_vec(full_rows, fill_rows),
            }
        };
        if !current.len().is_multiple_of(4) && current.len() > 2 {
            let row = full_rows;
            let child = |index| current.get(4 * row + index).copied().unwrap_or(F::ONE);
            next.push(F::product4([child(0), child(1), child(2), child(3)]));
        }
        level += 2;
        layers[level] = next;
    }
    if level < mu {
        layers[mu] = match layers[level].as_slice() {
            [root] => Vec::from_iter([*root]),
            [left, right] => Vec::from_iter([*left * *right]),
            _ => unreachable!("the final binary layer has at most two explicit nodes"),
        };
    }
    Levels {
        layers,
        bottom: leaves,
        len,
        ones,
    }
}

#[inline(always)]
fn quartic_summand(lines: [[F192; 2]; 4], equality: F192) -> [F192Unreduced; 4] {
    let [left0, left2, right0, right2] = mul4(
        [lines[0][0], lines[0][1], lines[2][0], lines[2][1]],
        [lines[1][0], lines[1][1], lines[3][0], lines[3][1]],
    );
    let [left_at_one, right_at_one, c0, c4] = mul4(
        [lines[0][0] + lines[0][1], lines[2][0] + lines[2][1], left0, left2],
        [lines[1][0] + lines[1][1], lines[3][0] + lines[3][1], right0, right2],
    );
    let left1 = left_at_one + left0 + left2;
    let right1 = right_at_one + right0 + right2;
    let [middle, at_one, cross_even, cross_high] = mul4(
        [left1, left0 + left1 + left2, left0 + left2, left1 + left2],
        [right1, right0 + right1 + right2, right0 + right2, right1 + right2],
    );
    let c2 = cross_even + c0 + c4 + middle;
    let c3 = cross_high + middle + c4;
    mul_unreduced4([equality; 4], [c0 + at_one, c2, c3, c4])
}

/// Sixty-four consecutive table values, `[F192x8; 8]`: block `b` holds `L = 64 b + 8 j + p` in vector
/// `p`, lane `j`. Lane `j` is then the row pair `8 b + j`, and `p = 4 parity + child`, so every
/// operation of a round is lane-wise.
pub type Block = [F192x8; 8];

pub const BLOCK_VALUES: usize = 64;

/// Pairs of rows in a block, the width of one equality vector.
const BLOCK_PAIRS: usize = 8;

/// Blocks per window of a parallel pass.
fn window_blocks(total: usize) -> usize {
    let tasks = parallel::num_threads() * 16;
    total.div_ceil(tasks).clamp(8, 64)
}

/// The values as blocks, the missing tail of the last block being the identity.
pub(crate) fn blocks_from_values<F: TreeField>(values: &[F]) -> Vec<F::Block> {
    let count = values.len().div_ceil(BLOCK_VALUES);
    let fill = |index: usize, destination: &mut [F::Block]| {
        for (offset, block) in destination.iter_mut().enumerate() {
            let base = (index * WINDOW + offset) * BLOCK_VALUES;
            *block = F::block(&values[base..(base + BLOCK_VALUES).min(values.len())]);
        }
    };
    const WINDOW: usize = 64;
    primitives::filled_vec(count, |blocks| {
        if count * BLOCK_PAIRS >= PAR_THRESHOLD {
            parallel::chunks_mut(blocks, WINDOW, fill);
        } else {
            fill(0, blocks);
        }
    })
}

/// The weights `eq(r, x)` of a sumcheck round, split as `hi[x >> k] * lo[x & (2^k - 1)]`.
///
/// Two tables of about the square root of the size stay in cache, where one table of every weight
/// is streamed from memory by each of the three trees in every round.
struct Equality {
    lo: Vec<F192>,
    hi: Vec<F192>,
}

impl Equality {
    /// The weights of `eq(point, .)`, least significant variable first. A block of eight weights must
    /// sit inside one `lo`, so a table of fewer than eight entries is all `lo`.
    fn new(point: &[F192]) -> Self {
        let k = (point.len() / 2).max(3).min(point.len());
        Self {
            lo: eq_table(&point[..k]),
            hi: eq_table(&point[k..]),
        }
    }

    /// The weight of row pair `i`.
    fn at(&self, i: usize) -> F192 {
        self.hi[i >> self.lo.len().ilog2()] * self.lo[i & (self.lo.len() - 1)]
    }

    /// The weights of block `b`'s eight row pairs; pairs past the table weigh nothing.
    fn block(&self, b: usize) -> F192x8 {
        let start = BLOCK_PAIRS * b;
        if self.lo.len() < BLOCK_PAIRS {
            let tail = self.lo.get(start..).unwrap_or_default();
            let mut padded = [F192::ZERO; BLOCK_PAIRS];
            padded[..tail.len()].copy_from_slice(tail);
            return F192x8::from_array(&padded);
        }
        let Some(&high) = self.hi.get(start / self.lo.len()) else {
            return F192x8::ZERO;
        };
        let low = start % self.lo.len();
        let weights = F192x8::from_array(self.lo[low..low + BLOCK_PAIRS].try_into().unwrap());
        if self.hi.len() == 1 {
            weights
        } else {
            F192x8::splat(high) * weights
        }
    }
}

/// The constant, linear and quadratic coefficients of the product of two lines `a + d X`, from
/// `a0`, `d0`, `a1`, `d1`, left unreduced.
#[inline(always)]
fn line_product(a0: F192x8, d0: F192x8, a1: F192x8, d1: F192x8) -> [F192x8; 3] {
    let low = a0.mul_unreduced(a1);
    let high = d0.mul_unreduced(d1);
    let at_one = (a0 + d0).mul_unreduced(a1 + d1);
    [
        low.reduce_lanes(),
        (at_one ^ low ^ high).reduce_lanes(),
        high.reduce_lanes(),
    ]
}

impl TableBlock for Block {
    /// Child `c` has the line `a + d X`, `a = v[c]` and `d = v[c] + v[c + 4]`. The weight multiplies the
    /// first line, so the lanes need no reduction before the sum over blocks.
    #[inline(always)]
    fn summand(&self, equality: F192x8) -> [Prod8; 4] {
        let v = self;
        let (a0, d0) = (equality * v[0], equality * (v[0] + v[4]));
        let [left0, left1, left2] = line_product(a0, d0, v[1], v[1] + v[5]);
        let [right0, right1, right2] = line_product(v[2], v[2] + v[6], v[3], v[3] + v[7]);
        let product = |a: F192x8, b: F192x8| a.mul_unreduced(b).merge_parities();
        let c0 = product(left0, right0);
        let c4 = product(left2, right2);
        let middle = product(left1, right1);
        let at_one = product(left0 + left1 + left2, right0 + right1 + right2);
        let cross_even = product(left0 + left2, right0 + right2);
        let cross_high = product(left1 + left2, right1 + right2);
        [c0 ^ at_one, cross_even ^ c0 ^ c4 ^ middle, cross_high ^ middle ^ c4, c4]
    }

    #[inline(always)]
    fn fold(&self, challenge: F192x8) -> [F192x8; 4] {
        std::array::from_fn(|c| self[c] + challenge * (self[c] + self[c + 4]))
    }

    const ONE: Self = [F192x8::ONE; 8];

    fn bivariate(&self, second: &Self, weight: F192x8, sum: &mut BivariateSum) {
        bivariate_summand(self, second, weight, sum);
    }
}

#[inline(always)]
fn bilinear_product(a: [[F192x8; 2]; 2], b: [[F192x8; 2]; 2]) -> [[F192x8; 3]; 3] {
    let low = line_product(a[0][0], a[0][1], b[0][0], b[0][1]);
    let high = line_product(a[1][0], a[1][1], b[1][0], b[1][1]);
    let sum = line_product(
        a[0][0] + a[1][0],
        a[0][1] + a[1][1],
        b[0][0] + b[1][0],
        b[0][1] + b[1][1],
    );
    [low, std::array::from_fn(|j| sum[j] + low[j] + high[j]), high]
}

#[inline(always)]
fn quadratic_product(a: [F192x8; 3], b: [F192x8; 3]) -> [Prod8; 5] {
    let p0 = a[0].mul_unreduced(b[0]).merge_parities();
    let p1 = a[1].mul_unreduced(b[1]).merge_parities();
    let p2 = a[2].mul_unreduced(b[2]).merge_parities();
    let p01 = (a[0] + a[1]).mul_unreduced(b[0] + b[1]).merge_parities();
    let p02 = (a[0] + a[2]).mul_unreduced(b[0] + b[2]).merge_parities();
    let p12 = (a[1] + a[2]).mul_unreduced(b[1] + b[2]).merge_parities();
    [p0, p01 ^ p0 ^ p1, p02 ^ p0 ^ p2 ^ p1, p12 ^ p1 ^ p2, p2]
}

/// The six Karatsuba products in `Y` of [`bivariate_summand`], each a quadratic in `X`, summed over
/// pairs: `a_0 b_0`, `a_1 b_1`, `a_2 b_2`, then the cross terms of `0, 1`, `0, 2` and `1, 2`.
type BivariateSum = [[Prod8; 5]; 6];

/// The coefficients of X^i Y^j of a [`BivariateSum`]: row `i` takes the products whose `Y` degrees
/// reach it, as the Karatsuba recombination does.
fn bivariate_coefficients(sum: &BivariateSum) -> [[F192; 5]; 5] {
    let [p0, p1, p2, p01, p02, p12] = sum;
    let row =
        |parts: &[&[Prod8; 5]]| std::array::from_fn(|j| parts.iter().fold(Prod8::ZERO, |acc, p| acc ^ p[j]).reduce());
    [
        row(&[p0]),
        row(&[p0, p1, p01]),
        row(&[p0, p1, p2, p02]),
        row(&[p1, p2, p12]),
        row(&[p2]),
    ]
}

/// Adds the products of a pair to `sum`; X folds the low row bit, and Y folds the next.
#[inline(never)]
fn bivariate_summand(first: &Block, second: &Block, weight: F192x8, sum: &mut BivariateSum) {
    let factors: [[[F192x8; 2]; 2]; 4] = std::array::from_fn(|c| {
        let (v0, v2) = F192x8::even_odd(first[c], second[c]);
        let (v1, v3) = F192x8::even_odd(first[c + 4], second[c + 4]);
        [[v0, v0 + v2], [v0 + v1, v0 + v1 + v2 + v3]]
    });
    let weighted = factors[0].map(|row| row.map(|v| weight * v));
    let a = bilinear_product(weighted, factors[1]);
    let b = bilinear_product(factors[2], factors[3]);
    add_product(&mut sum[0], a[0], b[0]);
    add_product(&mut sum[1], a[1], b[1]);
    add_product(&mut sum[2], a[2], b[2]);
    for (k, (i, l)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
        add_product(
            &mut sum[3 + k],
            std::array::from_fn(|j| a[i][j] + a[l][j]),
            std::array::from_fn(|j| b[i][j] + b[l][j]),
        );
    }
}

/// Adds `x · y` to `sum`. Kept out of line: the caller's twelve factors and this product's
/// temporaries do not fit the registers together.
#[inline(never)]
fn add_product(sum: &mut [Prod8; 5], x: [F192x8; 3], y: [F192x8; 3]) {
    for (slot, product) in sum.iter_mut().zip(quadratic_product(x, y)) {
        *slot ^= product;
    }
}

fn bivariate_messages(coefficients: [[F192; 5]; 5], eta: F192, r1: F192) -> [[F192; 4]; 2] {
    let first: [F192; 5] = coefficients.map(|c| c[0] + eta * (c[1] + c[2] + c[3] + c[4]));
    let second: [F192; 5] = std::array::from_fn(|j| coefficients.iter().rev().fold(F192::ZERO, |v, c| v * r1 + c[j]));
    [first, second].map(|c| [c[1] + c[2] + c[3] + c[4], c[2], c[3], c[4]])
}

/// The same with `K`-valued children: the lines and their products stay in `K`, and the weight, the one
/// `E` factor, multiplies the four coefficients as mixed products.
impl TableBlock for KBlock {
    #[inline(always)]
    fn summand(&self, equality: F192x8) -> [Prod8; 4] {
        let v = self;
        let lines = |c: usize| {
            let (d0, d1) = (v[c] + v[c + 4], v[c + 1] + v[c + 5]);
            let (low, high, at_one) = (v[c] * v[c + 1], d0 * d1, v[c + 4] * v[c + 5]);
            [low, at_one + low + high, high]
        };
        let [left0, left1, left2] = lines(0);
        let [right0, right1, right2] = lines(2);
        let c0 = left0 * right0;
        let c4 = left2 * right2;
        let middle = left1 * right1;
        let at_one = (left0 + left1 + left2) * (right0 + right1 + right2);
        let cross_even = (left0 + left2) * (right0 + right2);
        let cross_high = (left1 + left2) * (right1 + right2);
        [c0 + at_one, cross_even + c0 + c4 + middle, cross_high + middle + c4, c4]
            .map(|coefficient| equality.mul_base_unreduced(coefficient).merge_parities())
    }

    #[inline(always)]
    fn fold(&self, challenge: F192x8) -> [F192x8; 4] {
        std::array::from_fn(|c| {
            F192x8::from(self[c]) + challenge.mul_base_unreduced(self[c] + self[c + 4]).reduce_lanes()
        })
    }

    const ONE: Self = [F64x8::ONE; 8];

    /// [`bivariate_summand`] with every product in `K`: only the weight is in `E`, and it multiplies
    /// each product's five coefficients as mixed products.
    #[inline(never)]
    fn bivariate(&self, second: &Self, weight: F192x8, sum: &mut BivariateSum) {
        let line = |a0: F64x8, d0: F64x8, a1: F64x8, d1: F64x8| {
            let (low, high) = (a0 * a1, d0 * d1);
            [low, (a0 + d0) * (a1 + d1) + low + high, high]
        };
        let bilinear = |a: [[F64x8; 2]; 2], b: [[F64x8; 2]; 2]| {
            let low = line(a[0][0], a[0][1], b[0][0], b[0][1]);
            let high = line(a[1][0], a[1][1], b[1][0], b[1][1]);
            let cross = line(
                a[0][0] + a[1][0],
                a[0][1] + a[1][1],
                b[0][0] + b[1][0],
                b[0][1] + b[1][1],
            );
            [low, std::array::from_fn(|j| cross[j] + low[j] + high[j]), high]
        };
        let factors: [[[F64x8; 2]; 2]; 4] = std::array::from_fn(|c| {
            let (v0, v2) = F64x8::even_odd(self[c], second[c]);
            let (v1, v3) = F64x8::even_odd(self[c + 4], second[c + 4]);
            [[v0, v0 + v2], [v0 + v1, v0 + v1 + v2 + v3]]
        });
        let (a, b) = (bilinear(factors[0], factors[1]), bilinear(factors[2], factors[3]));
        let both = |x: [[F64x8; 3]; 3], i: usize, l: usize| std::array::from_fn(|j| x[i][j] + x[l][j]);
        let pairs = [
            (a[0], b[0]),
            (a[1], b[1]),
            (a[2], b[2]),
            (both(a, 0, 1), both(b, 0, 1)),
            (both(a, 0, 2), both(b, 0, 2)),
            (both(a, 1, 2), both(b, 1, 2)),
        ];
        for (slot, (x, y)) in sum.iter_mut().zip(pairs) {
            let (p0, p1, p2) = (x[0] * y[0], x[1] * y[1], x[2] * y[2]);
            let p01 = (x[0] + x[1]) * (y[0] + y[1]);
            let p02 = (x[0] + x[2]) * (y[0] + y[2]);
            let p12 = (x[1] + x[2]) * (y[1] + y[2]);
            let coefficients = [p0, p01 + p0 + p1, p02 + p0 + p2 + p1, p12 + p1 + p2, p2];
            for (slot, c) in slot.iter_mut().zip(coefficients) {
                *slot ^= weight.mul_base_unreduced(c).merge_parities();
            }
        }
    }
}

/// A table read block by block, stored as blocks or as rows that are packed as they are read.
trait Source: Sync {
    type Block: TableBlock;

    fn count(&self) -> usize;

    /// Block `b`, which must be inside the table.
    fn block(&self, b: usize) -> Self::Block;

    /// Starts fetching block `b`, if there is one.
    fn prefetch(&self, b: usize);
}

/// Blocks ahead that a pass prefetches. An `E` block is 24 cache lines and the hardware prefetcher alone
/// does not keep a kernel this compute-bound fed from memory.
const PREFETCH: usize = 4;
const PREFETCH_FOLD: usize = 2;

/// Starts fetching the `bytes` at `base`. Nothing is read, so any address will do.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn prefetch_lines(base: *const u8, bytes: usize) {
    for line in 0..bytes / 64 {
        // SAFETY: a prefetch of any address is harmless.
        unsafe {
            core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(base.wrapping_add(64 * line).cast())
        };
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn prefetch_lines(_base: *const u8, _bytes: usize) {}

/// Stored blocks.
impl<B: TableBlock, V: Deref<Target = [B]> + Sync> Source for V {
    type Block = B;

    fn count(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn block(&self, b: usize) -> B {
        self[b]
    }

    #[inline(always)]
    fn prefetch(&self, b: usize) {
        if let Some(block) = self.get(b) {
            prefetch_lines(core::ptr::from_ref(block).cast(), size_of::<B>());
        }
    }
}

/// A table of rows, the products of the level below. The first message and the first fold of a layer
/// read them in place, so no table of blocks is written and read back.
struct Rows<F>(Vec<F>);

impl<F: TreeField> Source for Columns<'_, F> {
    type Block = F::Block;

    fn count(&self) -> usize {
        self.len.div_ceil(BLOCK_VALUES)
    }

    fn block(&self, b: usize) -> Self::Block {
        F::block(&self.read::<BLOCK_VALUES>(b * BLOCK_VALUES))
    }

    fn prefetch(&self, b: usize) {
        let start = b * BLOCK_VALUES;
        let next = self.segments.partition_point(|&(off, data)| off + data.len() <= start);
        if let Some(&(off, data)) = self.segments.get(next) {
            let at = start.saturating_sub(off);
            let values = &data[at..(at + BLOCK_VALUES).min(data.len())];
            prefetch_lines(values.as_ptr().cast(), size_of_val(values));
        }
    }
}

impl<F: TreeField> Source for Rows<F> {
    type Block = F::Block;

    fn count(&self) -> usize {
        self.0.len().div_ceil(BLOCK_VALUES)
    }

    #[inline(always)]
    fn block(&self, b: usize) -> F::Block {
        let start = b * BLOCK_VALUES;
        F::block(&self.0[start..(start + BLOCK_VALUES).min(self.0.len())])
    }

    #[inline(always)]
    fn prefetch(&self, b: usize) {
        if let Some(row) = self.0.get(b * BLOCK_VALUES) {
            prefetch_lines(core::ptr::from_ref(row).cast(), BLOCK_VALUES * size_of::<F>());
        }
    }
}

/// A layer's first table, whichever form it has, seen through its first message and first fold.
trait Table: Send + Sync {
    fn count(&self) -> usize;

    fn message(&self, ones: &[Range<usize>], equality: &Equality) -> [F192; 4];

    fn polynomial(&self, ones: &[Range<usize>], equality: &Equality) -> [[F192; 5]; 5];

    fn fold(
        &self,
        next_blocks: &mut [Block],
        ones: &[Range<usize>],
        challenge: F192,
        second: Option<F192>,
        equality: Option<&Equality>,
    ) -> [F192; 4];
}

impl<S: Source + Send> Table for S {
    fn count(&self) -> usize {
        Source::count(self)
    }

    fn message(&self, ones: &[Range<usize>], equality: &Equality) -> [F192; 4] {
        blocks_message(self, ones, equality)
    }

    fn polynomial(&self, ones: &[Range<usize>], equality: &Equality) -> [[F192; 5]; 5] {
        blocks_polynomial(self, ones, equality)
    }

    fn fold(
        &self,
        next_blocks: &mut [Block],
        ones: &[Range<usize>],
        challenge: F192,
        second: Option<F192>,
        equality: Option<&Equality>,
    ) -> [F192; 4] {
        fold_block_pairs(self, next_blocks, ones, challenge, second, equality)
    }
}

/// Block `k` of the folded table: blocks `2k` and `2k + 1` folded by `challenge`, their even and
/// odd lanes (the two rows of each pair) interleaved. A missing block, or one inside `ones`, is all
/// identity, and folds to it.
#[inline(always)]
fn fold_block_pair<S: Source + ?Sized>(source: &S, ones: &[Range<usize>], k: usize, challenge: F192x8) -> Block {
    let fold = |b: usize| {
        if b < source.count() && !in_runs(ones, b) {
            source.block(b).fold(challenge)
        } else {
            [F192x8::ONE; 4]
        }
    };
    interleave(fold(2 * k), fold(2 * k + 1))
}

/// The folded block of two folded source blocks: their even and odd lanes interleaved.
#[inline(always)]
fn interleave(first: [F192x8; 4], second: [F192x8; 4]) -> Block {
    let pairs: [(F192x8, F192x8); 4] = std::array::from_fn(|c| F192x8::even_odd(first[c], second[c]));
    std::array::from_fn(|p| if p < 4 { pairs[p].0 } else { pairs[p - 4].1 })
}

/// The leaf layer's blocks recomputed from the builder's inputs for its initial messages and folds.
/// Subsequent rounds read the stored folded table.
pub trait LeafColumns: Sync {
    /// Sorted, disjoint runs of blocks of side `s` (0 push, 1 pull), aligned to four blocks.
    fn runs(&self, s: usize) -> &[Range<usize>];

    /// The number of tasks of a pass.
    fn tasks(&self) -> usize;

    /// Computes task `t`'s blocks, calling `f(side, block index, block)` for each, in groups of four:
    /// `4k` through `4k + 3` in that order, and the groups of a run in increasing order. The two
    /// sides' calls may interleave block by block.
    fn task(&self, t: usize, f: &mut dyn FnMut(usize, usize, &Block));
}

/// Runs the tasks of a pass and adds up each side's message coefficients.
fn reduce_sides(columns: &dyn LeafColumns, task: impl Fn(usize) -> [[Prod8; 4]; 2] + Sync) -> [[F192; 4]; 2] {
    let xor = |left: [[Prod8; 4]; 2], right: [[Prod8; 4]; 2]| {
        [xor_coefficients(left[0], right[0]), xor_coefficients(left[1], right[1])]
    };
    parallel::map_reduce(columns.tasks(), || [[Prod8::ZERO; 4]; 2], task, xor).map(|sum| sum.map(Prod8::reduce))
}

/// The first message of both sides' blocks of the runs.
#[timed]
fn columns_message(columns: &dyn LeafColumns, equality: &Equality) -> [[F192; 4]; 2] {
    let task = |t: usize| {
        let mut sums = [[Prod8::ZERO; 4]; 2];
        columns.task(t, &mut |s, b, block| {
            sums[s] = xor_coefficients(sums[s], block.summand(equality.block(b)));
        });
        sums
    };
    reduce_sides(columns, task)
}

/// The first two round polynomials, accumulated before either challenge is sampled.
#[timed]
fn columns_polynomial(columns: &dyn LeafColumns, equality: &Equality) -> [[[F192; 5]; 5]; 2] {
    let task = |t| {
        let mut sums = [[[Prod8::ZERO; 5]; 6]; 2];
        let mut even = [[F192x8::ZERO; 8]; 2];
        columns.task(t, &mut |side, b, block| {
            if b % 2 == 0 {
                even[side] = *block;
            } else {
                bivariate_summand(&even[side], block, equality.block(b / 2), &mut sums[side]);
            }
        });
        sums.map(|sum| bivariate_coefficients(&sum))
    };
    parallel::map_reduce(
        columns.tasks(),
        || [[[F192::ZERO; 5]; 5]; 2],
        task,
        |a, b| std::array::from_fn(|s| std::array::from_fn(|i| std::array::from_fn(|j| a[s][i][j] + b[s][i][j]))),
    )
}

/// Folds the blocks of the runs by `challenge` into the sides' `next_blocks`, and with an `equality`
/// returns the message of the folded blocks.
#[timed]
fn columns_fold(
    columns: &dyn LeafColumns,
    next: [&mut [Block]; 2],
    challenge: F192,
    second: Option<F192>,
    equality: Option<&Equality>,
) -> [[F192; 4]; 2] {
    let second = second.map(F192x8::splat);
    let dst = next.map(|blocks| parallel::SendPtr(blocks.as_mut_ptr()));
    let challenge = F192x8::splat(challenge);
    let task = |t: usize| {
        // The next round reads the destination; this round reads only the stage.
        let stream = Stream::new();
        let mut sums = [[Prod8::ZERO; 4]; 2];
        let mut even = [[F192x8::ZERO; 4]; 2];
        let mut quarter = [[F192x8::ZERO; 4]; 2];
        columns.task(t, &mut |s, b, block| {
            let folded = block.fold(challenge);
            if b % 2 == 0 {
                even[s] = folded;
                return;
            }
            let mut block = interleave(even[s], folded);
            let mut index = b / 2;
            if let Some(second) = second {
                let folded = block.fold(second);
                if b % 4 == 1 {
                    quarter[s] = folded;
                    return;
                }
                block = interleave(quarter[s], folded);
                index = b / 4;
            }
            // SAFETY: each task owns complete pairs, or complete groups of four for two folds.
            unsafe { stream.copy_lines(dst[s].add(index), &block) };
            if let Some(equality) = equality {
                sums[s] = xor_coefficients(sums[s], block.summand(equality.block(index)));
            }
        });
        sums
    };
    reduce_sides(columns, task)
}

fn xor_coefficients<const N: usize>(mut left: [Prod8; N], right: [Prod8; N]) -> [Prod8; N] {
    for (a, b) in left.iter_mut().zip(right) {
        *a ^= b;
    }
    left
}

/// The round message of a table of blocks. A block of identity rows adds nothing: its polynomial is
/// the constant weight, which `q(0) + q(1)` cancels.
fn blocks_message<S: Source + ?Sized>(source: &S, ones: &[Range<usize>], equality: &Equality) -> [F192; 4] {
    let count = source.count();
    let rows = window_blocks(count);
    let window = |index: usize| {
        live(ones, index * rows..((index + 1) * rows).min(count)).fold([Prod8::ZERO; 4], |sum, b| {
            source.prefetch(b + PREFETCH);
            xor_coefficients(sum, source.block(b).summand(equality.block(b)))
        })
    };
    let windows = count.div_ceil(rows);
    let message = if BLOCK_PAIRS * count >= PAR_THRESHOLD {
        parallel::map_reduce(windows, || [Prod8::ZERO; 4], window, xor_coefficients)
    } else {
        (0..windows).map(window).fold([Prod8::ZERO; 4], xor_coefficients)
    };
    message.map(Prod8::reduce)
}

/// The first two round polynomials of a table of blocks, before either challenge is sampled: the
/// pairs of blocks weighted by `equality`. A pair of identity blocks adds only to the constant
/// coefficient, which no message reads.
fn blocks_polynomial<S: Source + ?Sized>(source: &S, ones: &[Range<usize>], equality: &Equality) -> [[F192; 5]; 5] {
    let count = source.count();
    let pairs = count.div_ceil(2);
    let pair_ones = halve(ones, pairs);
    let block = |b: usize| {
        if b < count && !in_runs(ones, b) {
            source.block(b)
        } else {
            S::Block::ONE
        }
    };
    let rows = window_blocks(pairs);
    let window = |index: usize| {
        let mut sum = [[Prod8::ZERO; 5]; 6];
        for k in live(&pair_ones, index * rows..((index + 1) * rows).min(pairs)) {
            source.prefetch(2 * (k + PREFETCH));
            source.prefetch(2 * (k + PREFETCH) + 1);
            block(2 * k).bivariate(&block(2 * k + 1), equality.block(k), &mut sum);
        }
        sum
    };
    let xor = |left: BivariateSum, right: BivariateSum| std::array::from_fn(|k| xor_coefficients(left[k], right[k]));
    let windows = pairs.div_ceil(rows);
    let sum = if BLOCK_PAIRS * count >= PAR_THRESHOLD {
        parallel::map_reduce(windows, || [[Prod8::ZERO; 5]; 6], window, xor)
    } else {
        (0..windows).map(window).fold([[Prod8::ZERO; 5]; 6], xor)
    };
    bivariate_coefficients(&sum)
}

/// Fold every pair of blocks of `source` by `challenge`, and with a `second` challenge every pair of
/// the folded blocks again, into `E` blocks, and with an `equality` weigh the folded table's round
/// message as it is produced. Folded blocks that are wholly identity are not written.
fn fold_block_pairs<S: Source + ?Sized>(
    source: &S,
    next_blocks: &mut [Block],
    ones: &[Range<usize>],
    challenge: F192,
    second: Option<F192>,
    equality: Option<&Equality>,
) -> [F192; 4] {
    let count = next_blocks.len();
    let next_ones = match second {
        Some(_) => halve(&halve(ones, source.count().div_ceil(2)), count),
        None => halve(ones, count),
    };
    let span = if second.is_some() { 4 } else { 2 };
    let second = second.map(F192x8::splat);
    let challenge = F192x8::splat(challenge);
    let window = window_blocks(count);
    let task = |index: usize, out: &mut [Block]| {
        // The next round reads the destination; this round reads only the stage.
        let stream = Stream::new();
        let first = index * window;
        let mut message = [Prod8::ZERO; 4];
        for k in live(&next_ones, first..first + out.len()) {
            for b in 0..span {
                source.prefetch(span * (k + PREFETCH_FOLD) + b);
            }
            let block = match second {
                Some(second) => interleave(
                    fold_block_pair(source, ones, 2 * k, challenge).fold(second),
                    fold_block_pair(source, ones, 2 * k + 1, challenge).fold(second),
                ),
                None => fold_block_pair(source, ones, k, challenge),
            };
            // The stores go before the summand, so they overlap its products.
            stream.copy_lines_into(&mut out[k - first], &block);
            if let Some(equality) = equality {
                message = xor_coefficients(message, block.summand(equality.block(k)));
            }
        }
        message
    };
    let message = if BLOCK_PAIRS * count >= PAR_THRESHOLD {
        parallel::map_reduce_chunks_mut(
            [next_blocks],
            [window],
            || (),
            || [Prod8::ZERO; 4],
            |(), acc, index, [out]| *acc = xor_coefficients(*acc, task(index, out)),
            xor_coefficients,
        )
    } else {
        next_blocks.chunks_mut(window).enumerate().map(|(index, out)| task(index, out)).fold([Prod8::ZERO; 4], xor_coefficients)
    };
    message.map(Prod8::reduce)
}

/// Two binary product levels contracted into one degree-four layer.
pub struct QuaternaryLayerState<'a> {
    /// Four child tables interleaved in their original order. This lets the
    /// prover consume a product-tree level without first transposing it.
    /// Empty once the table is held as `blocks`.
    values: Vec<F192>,
    next: Vec<F192>,
    /// The same table as blocks, when it spans at least two of them; empty otherwise.
    blocks: Vec<Block>,
    next_blocks: Vec<Block>,
    /// A table of at least two blocks that is not yet `E` blocks (base-field blocks, or rows), until
    /// the first fold: it then replaces `blocks`.
    first: Option<Box<dyn Table + 'a>>,
    /// Logical row count after identity padding. `values` stores an arbitrary
    /// prefix; every omitted row is the constant four-tuple one.
    logical_rows: usize,
    /// Runs of blocks that are wholly identity, which the table does not store.
    ones: Vec<Range<usize>>,
    /// Runs of blocks supplied by a [`LeafColumns`] pass rather than stored in the table.
    columns: Vec<Range<usize>>,
}

impl<'a> QuaternaryLayerState<'a> {
    fn new<F: TreeField>(values: Vec<F>, width: usize) -> Self {
        if values.len() <= BLOCK_VALUES {
            return Self::from_rows(values.into_iter().map(F::embed).collect(), width);
        }
        Self::from_table(Box::new(Rows(values)), width)
    }

    /// The state over a table of `len` values held as blocks, whose tail is the identity.
    fn from_blocks(blocks: Vec<Block>, len: usize, width: usize) -> Self {
        if blocks.len() < 2 {
            return Self::from_rows((0..len).map(|l| F192::leaf(&blocks, l)).collect(), width);
        }
        Self {
            values: Vec::new(),
            next: Vec::new(),
            // SAFETY: the first fold writes every slot of `next_blocks` before any read.
            next_blocks: unsafe { primitives::uninit_vec(blocks.len().div_ceil(2)) },
            blocks,
            first: None,
            logical_rows: width,
            ones: Vec::new(),
            columns: Vec::new(),
        }
    }

    /// [`from_blocks`](Self::from_blocks) over a base-field table.
    fn from_k_blocks(k_blocks: Vec<KBlock>, len: usize, width: usize) -> Self {
        if k_blocks.len() < 2 {
            return Self::from_rows((0..len).map(|l| F64::leaf(&k_blocks, l).embed()).collect(), width);
        }
        Self::from_table(Box::new(k_blocks), width)
    }

    /// The state over a table of at least two blocks that the first fold turns into `E` blocks.
    fn from_table(table: Box<dyn Table + 'a>, width: usize) -> Self {
        Self {
            values: Vec::new(),
            next: Vec::new(),
            // SAFETY: the first fold writes every slot of `next_blocks` before any read.
            next_blocks: unsafe { primitives::uninit_vec(table.count().div_ceil(2)) },
            blocks: Vec::new(),
            first: Some(table),
            logical_rows: width,
            ones: Vec::new(),
            columns: Vec::new(),
        }
    }

    /// Whether the table is held as blocks, or as something the first fold turns into them.
    fn blocked(&self) -> bool {
        !self.blocks.is_empty() || self.first.is_some()
    }

    /// The state over the rows themselves, for a table of at most one block.
    fn from_rows(mut values: Vec<F192>, width: usize) -> Self {
        // Materialize only the incomplete final four-tuple. Every complete
        // all-one row after the arbitrary explicit prefix remains implicit.
        values.resize(4 * values.len().max(1).div_ceil(4), F192::ONE);
        debug_assert_eq!(values.len() % 4, 0);
        debug_assert!(values.len() <= 4 * width);
        let rows = (values.len() / 4).div_ceil(2);
        Self {
            values,
            // SAFETY: the first `fold` writes every slot of `next[..4 * rows]` before
            // any read (its windows cover the full pairs, its tail block the odd row),
            // and neither `round_message` nor `children` reads `next`.
            next: unsafe { primitives::uninit_vec(4 * rows) },
            blocks: Vec::new(),
            next_blocks: Vec::new(),
            first: None,
            logical_rows: width,
            ones: Vec::new(),
            columns: Vec::new(),
        }
    }

    /// The runs of blocks that the stored table does not hold.
    fn unstored(&self) -> Vec<Range<usize>> {
        let mut runs = [self.ones.as_slice(), self.columns.as_slice()].concat();
        runs.sort_unstable_by_key(|run| run.start);
        runs
    }

    /// `(q(0)+q(1), [X²]q, [X³]q, [X⁴]q)`.
    #[timed]
    fn round_message(&self, equality: &Equality) -> [F192; 4] {
        if let Some(first) = &self.first {
            return first.message(&self.ones, equality);
        }
        if self.blocked() {
            return blocks_message(&self.blocks, &self.unstored(), equality);
        }
        let stored_rows = self.values.len() / 4;
        let full_pairs = stored_rows / 2;
        let summand = |row: usize| -> [F192Unreduced; 4] {
            let (lo, hi) = (8 * row, 8 * row + 4);
            let lines = [0, 1, 2, 3].map(|child| {
                let at_zero = self.values[lo + child];
                [at_zero, at_zero + self.values[hi + child]]
            });
            quartic_summand(lines, equality.at(row))
        };
        let xor = |mut left: [F192Unreduced; 4], right: [F192Unreduced; 4]| {
            for coefficient in 0..4 {
                left[coefficient] ^= right[coefficient];
            }
            left
        };
        let rows = window_rows(full_pairs);
        let window = |index: usize| -> [F192Unreduced; 4] {
            let base = index * rows;
            (base..(base + rows).min(full_pairs)).fold([F192Unreduced::ZERO; 4], |sum, row| xor(sum, summand(row)))
        };
        let windows = full_pairs.div_ceil(rows);
        let mut message = if full_pairs >= PAR_THRESHOLD {
            parallel::map_reduce(windows, || [F192Unreduced::ZERO; 4], window, xor)
        } else {
            (0..windows).map(window).fold([F192Unreduced::ZERO; 4], xor)
        };
        if !stored_rows.is_multiple_of(2) {
            let lo = 8 * full_pairs;
            let lines = [0, 1, 2, 3].map(|child| {
                let at_zero = self.values[lo + child];
                [at_zero, at_zero + F192::ONE]
            });
            message = xor(message, quartic_summand(lines, equality.at(full_pairs)));
        }
        message.map(F192Unreduced::reduce)
    }

    /// Fold every pair of blocks by `challenge` into `next_blocks`, and with an `equality` weigh the
    /// folded table's round message as it is produced.
    /// With a `second` challenge, the folded table is folded again by it in the same pass.
    fn fold_blocks(
        &mut self,
        challenge: F192,
        second: Option<F192>,
        equality: Option<&Equality>,
        keep_columns: bool,
    ) -> [F192; 4] {
        let blocks = self.first.as_ref().map_or(self.blocks.len(), |first| first.count());
        let half = blocks.div_ceil(2);
        let count = if second.is_some() { half.div_ceil(2) } else { half };
        self.next_blocks.truncate(count);
        let first = self.first.take();
        let unstored = self.unstored();
        let message = match &first {
            Some(first) => first.fold(&mut self.next_blocks, &unstored, challenge, second, equality),
            None => fold_block_pairs(
                &self.blocks,
                &mut self.next_blocks,
                &unstored,
                challenge,
                second,
                equality,
            ),
        };
        self.ones = halve(&self.ones, half);
        if second.is_some() {
            self.ones = halve(&self.ones, count);
            self.logical_rows /= 2;
        }
        if keep_columns {
            self.columns = halve(&self.columns, count);
        } else {
            self.columns.clear();
        }
        std::mem::swap(&mut self.blocks, &mut self.next_blocks);
        if first.is_some() {
            // The leaf table dies first, so the allocator can hand its pages to the next buffer.
            drop(first);
            // SAFETY: the next fold writes every slot of `next_blocks` before any read.
            self.next_blocks = unsafe { primitives::uninit_vec(count.div_ceil(2)) };
        }
        self.logical_rows /= 2;
        message
    }

    #[timed]
    fn fold(&mut self, challenge: F192) {
        if self.blocked() {
            self.fold_blocks(challenge, None, None, false);
            return;
        }
        let stored_rows = self.values.len() / 4;
        let full_rows = stored_rows / 2;
        let rows = stored_rows.div_ceil(2);
        self.next.truncate(4 * rows);
        let (values, next) = (&self.values, &mut self.next);
        let fold_row = |row: usize| -> [F192; 4] {
            // One slice, not eight indexes: the bounds checks and the
            // index-by-24 multiplies fall out.
            let v = &values[8 * row..8 * row + 8];
            let folds = mul4(std::array::from_fn(|child| v[child] + v[4 + child]), [challenge; 4]);
            std::array::from_fn(|child| v[child] + folds[child])
        };
        // The next round is what reads the output, and a layer this size is long
        // evicted by then, so where an ordinary store fetches the line it
        // overwrites the pair is staged and published with streaming stores.
        // Where it does not, the stage buys nothing and costs a real call, the
        // `slot.len()` being one the compiler cannot fold away.
        #[cfg(target_arch = "x86_64")]
        let window = |base: usize, destination: &mut [F192]| {
            let stream = Stream::new();
            for (pair, slot) in destination.chunks_mut(8).enumerate() {
                let mut both = [F192::ZERO; 8];
                both[..4].copy_from_slice(&fold_row(base + 2 * pair));
                if slot.len() == 8 {
                    both[4..].copy_from_slice(&fold_row(base + 2 * pair + 1));
                }
                stream.copy(slot, &both[..slot.len()]);
            }
        };
        #[cfg(not(target_arch = "x86_64"))]
        let window = |base: usize, destination: &mut [F192]| {
            for (pair, slot) in destination.chunks_mut(8).enumerate() {
                slot[..4].copy_from_slice(&fold_row(base + 2 * pair));
                if slot.len() == 8 {
                    slot[4..].copy_from_slice(&fold_row(base + 2 * pair + 1));
                }
            }
        };
        if full_rows >= PAR_THRESHOLD {
            let rows = window_rows(full_rows);
            parallel::chunks_mut(&mut next[..4 * full_rows], 4 * rows, |index, destination| {
                window(index * rows, destination);
            });
        } else {
            window(0, &mut next[..4 * full_rows]);
        }
        if !stored_rows.is_multiple_of(2) {
            let lo = 8 * full_rows;
            let folds = mul4(
                [0, 1, 2, 3].map(|child| self.values[lo + child] + F192::ONE),
                [challenge; 4],
            );
            for child in 0..4 {
                self.next[4 * full_rows + child] = self.values[lo + child] + folds[child];
            }
        }
        std::mem::swap(&mut self.values, &mut self.next);
        self.logical_rows /= 2;
    }

    #[timed]
    fn fold_and_message(&mut self, challenge: F192, equality: &Equality) -> [F192; 4] {
        if self.blocked() {
            return self.fold_blocks(challenge, None, Some(equality), false);
        }
        let stored_rows = self.values.len() / 4;
        let rows = stored_rows.div_ceil(2);
        self.next.truncate(4 * rows);
        let values = &self.values;
        const PAIRS: usize = 16;
        let pairs = rows.div_ceil(2);
        let task = |index: usize, out: &mut [F192]| {
            let first = index * PAIRS;
            let end = (first + PAIRS).min(pairs);
            let mut stage = [F192::ZERO; 8 * PAIRS];
            let end_row = (2 * end).min(rows);
            for row in 2 * first..end_row {
                let lo = 8 * row;
                let left = &values[lo..lo + 4];
                let right = values.get(lo + 4..lo + 8).unwrap_or(&[F192::ONE; 4]);
                let product = mul4(std::array::from_fn(|i| left[i] + right[i]), [challenge; 4]);
                let offset = 4 * (row - 2 * first);
                for i in 0..4 {
                    stage[offset + i] = left[i] + product[i];
                }
            }
            let mut message = [F192Unreduced::ZERO; 4];
            for pair in first..end {
                let lo = 8 * (pair - first);
                let left = &stage[lo..lo + 4];
                let right = if 2 * pair + 1 < rows {
                    &stage[lo + 4..lo + 8]
                } else {
                    &[F192::ONE; 4]
                };
                let lines = std::array::from_fn(|i| [left[i], left[i] + right[i]]);
                let terms = quartic_summand(lines, equality.at(pair));
                for i in 0..4 {
                    message[i] ^= terms[i];
                }
            }
            // The next round reads the destination; this round reads only the local stage.
            let stream = Stream::new();
            let len = 4 * (end_row - 2 * first);
            stream.copy(out, &stage[..len]);
            message
        };
        let xor = |mut a: [F192Unreduced; 4], b: [F192Unreduced; 4]| {
            for i in 0..4 {
                a[i] ^= b[i];
            }
            a
        };
        // A task's rows are the `8 * PAIRS` values of its chunk, fewer in the last.
        let out = &mut self.next[..4 * rows];
        let message = if rows >= PAR_THRESHOLD {
            parallel::map_reduce_chunks_mut(
                [out],
                [8 * PAIRS],
                || (),
                || [F192Unreduced::ZERO; 4],
                |(), acc, index, [out]| *acc = xor(*acc, task(index, out)),
                xor,
            )
        } else {
            out.chunks_mut(8 * PAIRS).enumerate().map(|(index, out)| task(index, out)).fold([F192Unreduced::ZERO; 4], xor)
        };
        std::mem::swap(&mut self.values, &mut self.next);
        self.logical_rows /= 2;
        message.map(F192Unreduced::reduce)
    }

    fn children(&self) -> [F192; 4] {
        debug_assert_eq!(self.logical_rows, 1);
        if let Some(block) = self.blocks.first() {
            return std::array::from_fn(|child| block[child].lane(0));
        }
        debug_assert_eq!(self.values.len(), 4);
        self.values[..4].try_into().unwrap()
    }

    /// The table as `4 * rows` values, whichever form holds it.
    #[cfg(test)]
    fn table(&self) -> Vec<F192> {
        assert!(self.first.is_none(), "the table is not `E` blocks yet");
        if !self.blocked() {
            return self.values.clone();
        }
        (0..BLOCK_VALUES * self.blocks.len())
            .map(|l| F192::leaf(&self.blocks, l))
            .collect()
    }
}

/// The result of a batched grand-product proof: the three roots and leaf
/// evaluations, all reduced to one shared point. Under [`RootShape::FirstTwoShared`],
/// `roots[0] == roots[1]` by construction rather than by a check.
pub struct ProductTriple {
    pub roots: [F192; 3],
    pub point: Vec<F192>,
    pub values: [F192; 3],
}

/// How many of the three roots ride the stream, which is a property of the statement rather than
/// of the reduction below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootShape {
    /// Three unrelated products: every root is sent.
    Distinct,
    /// The first two trees share a product by construction, as the bus's two sides do
    /// (a padding row's leaf is the identity on both, so they balance outright). ONE root
    /// is then sent for both, and no verifier can be handed an unbalanced pair to check.
    FirstTwoShared,
}

/// Adds the column pass's messages of the first two trees to the stored blocks' messages.
fn add_messages(messages: &mut [[F192; 4]; 3], from_columns: [[F192; 4]; 2]) {
    for (message, extra) in messages.iter_mut().zip(from_columns) {
        for (coefficient, extra) in message.iter_mut().zip(extra) {
            *coefficient += extra;
        }
    }
}

/// [`columns_fold`] into the first two trees' `next_blocks`, which their own fold then completes.
fn fold_columns(
    columns: &dyn LeafColumns,
    trees: &mut [QuaternaryLayerState; 3],
    challenge: F192,
    second: Option<F192>,
    equality: Option<&Equality>,
) -> [[F192; 4]; 2] {
    let [push, pull, _] = trees;
    columns_fold(
        columns,
        [&mut push.next_blocks, &mut pull.next_blocks],
        challenge,
        second,
        equality,
    )
}

/// Prove three identity-padded grand products as one RLC-batched radix-four GKR.
pub fn prove_product_triple(leaves: [Vec<F192>; 3], ps: &mut ProverState, shape: RootShape) -> ProductTriple {
    let [push, pull, count] = leaves.map(Leaves::new);
    prove_product_triple_from(push, pull, count, None, ps, shape)
}

/// [`prove_product_triple`] over leaves whose first radix-four layer is already built, the third
/// tree's in `C`. The proof is the same whichever field holds the third tree's values.
///
/// The `columns` are blocks of the first two trees' leaves that `push` and `pull` do not store.
#[timed]
pub fn prove_product_triple_from<C: TreeField>(
    push: Leaves,
    pull: Leaves,
    count: Leaves<C>,
    columns: Option<&dyn LeafColumns>,
    ps: &mut ProverState,
    shape: RootShape,
) -> ProductTriple {
    let mu = crate::log2_ceil_usize(push.len);
    assert!(
        [pull.len, count.len].iter().all(|&len| len > 0 && len <= 1 << mu),
        "batched trees must be nonempty prefixes of the first tree's logical tree"
    );
    let mut levels: [Box<dyn Tree<'_> + '_>; 3] = [
        Box::new(build_layers(push, mu)),
        Box::new(build_layers(pull, mu)),
        Box::new(build_layers(count, mu)),
    ];
    let roots = levels.each_ref().map(|tree| tree.root());
    match shape {
        RootShape::Distinct => {
            for root in roots {
                ps.add_scalar(root);
            }
        }
        RootShape::FirstTwoShared => {
            assert_eq!(roots[0], roots[1], "FirstTwoShared needs the two products to agree");
            ps.add_scalar(roots[0]);
            ps.add_scalar(roots[2]);
        }
    }
    let mut point = Vec::new();
    let mut values = roots;

    let mut layer = mu;
    while layer > 0 {
        let _layer_guard = flux_profiler::TimerGuard::new("gkr::layer");
        let lambda = ps.sample();
        let round_count = mu - layer;
        if layer % 2 == 1 {
            debug_assert_eq!(round_count, 0, "only the root-most layer may be binary");
            let tails = levels.each_ref().map(|tree| match tree.nodes(layer - 1).as_slice() {
                [left, right] => [*left, *right],
                [left] => [*left, F192::ONE],
                _ => unreachable!("the root's children have at most two explicit nodes"),
            });
            for tail in &tails {
                ps.add_scalars(tail);
            }
            let challenge = ps.sample();
            for (value, [left, right]) in values.iter_mut().zip(tails) {
                *value = interp(left, right, challenge);
            }
            point = vec![challenge];
            layer -= 1;
            continue;
        }

        let width = 1usize << round_count;
        let mut trees = levels.each_mut().map(|tree| tree.state(layer, width));
        let columns = columns.filter(|_| layer == 2 && round_count > 0);
        if let Some(columns) = columns {
            for (s, tree) in trees.iter_mut().take(2).enumerate() {
                tree.columns = columns.runs(s).to_vec();
            }
        }
        let mut equality = Equality::new(point.get(1..).unwrap_or_default());
        let mut round_point = Vec::with_capacity(round_count);
        // With enough workers to hide the extra products behind the saved traffic, a layer's first two
        // rounds come from one pass over its columns or its first table, and one double fold.
        let two_rounds = round_count >= 2
            && parallel::num_threads() >= 4
            && cfg!(all(
                target_arch = "x86_64",
                target_feature = "vpclmulqdq",
                target_feature = "avx512f"
            ));
        let pairs = two_rounds.then(|| Equality::new(&point[2..]));
        // A first table's double fold carries the third round's message, so it needs three rounds.
        let firsts: [Option<[[F192; 5]; 5]>; 3] = trees.each_ref().map(|tree| {
            let first = tree.first.as_ref().filter(|_| round_count >= 3)?;
            Some(first.polynomial(&tree.ones, pairs.as_ref()?))
        });
        let mut messages = if round_count > 0 {
            std::array::from_fn(|s| match firsts[s] {
                Some(polynomial) => bivariate_messages(polynomial, point[1], F192::ZERO)[0],
                None => trees[s].round_message(&equality),
            })
        } else {
            [[F192::ZERO; 4]; 3]
        };
        let polynomial = columns
            .zip(pairs.as_ref())
            .map(|(columns, pairs)| columns_polynomial(columns, pairs));
        if let Some(columns) = columns {
            let from_columns = polynomial.map_or_else(
                || columns_message(columns, &equality),
                |polynomial| polynomial.map(|p| bivariate_messages(p, point[1], F192::ZERO)[0]),
            );
            add_messages(&mut messages, from_columns);
        }
        for round in 0..round_count {
            let mut coeffs = [0, 1, 2, 3].map(|coefficient| {
                messages[0][coefficient] + lambda * (messages[1][coefficient] + lambda * messages[2][coefficient])
            });
            // The kernel accumulates `q(0) + q(1)`, the wire carries `c1`. This is
            // `ProverState::add_round_poly(_, true)` bar its dropped `c0`, which the
            // claim fixes and the prover therefore never forms.
            coeffs[0] = coeffs[0] + coeffs[1] + coeffs[2] + coeffs[3];
            ps.add_scalars(&coeffs);
            let challenge = ps.sample();
            round_point.push(challenge);
            if round + 1 < round_count {
                equality = Equality::new(&point[round + 2..]);
                let from_columns = if let Some(polynomial) = polynomial.filter(|_| round == 0) {
                    Some(polynomial.map(|p| bivariate_messages(p, F192::ZERO, challenge)[1]))
                } else if let Some(columns) = columns.filter(|_| round == usize::from(polynomial.is_some())) {
                    let (first, second) = if polynomial.is_some() {
                        (round_point[0], Some(challenge))
                    } else {
                        (challenge, None)
                    };
                    Some(fold_columns(columns, &mut trees, first, second, Some(&equality)))
                } else {
                    None
                };
                messages = std::array::from_fn(|s| {
                    if let Some(polynomial) = firsts[s] {
                        match round {
                            0 => bivariate_messages(polynomial, F192::ZERO, challenge)[1],
                            1 => trees[s].fold_blocks(round_point[0], Some(challenge), Some(&equality), false),
                            _ => trees[s].fold_and_message(challenge, &equality),
                        }
                    } else if round == 0 && polynomial.is_some() && s < 2 && trees[s].blocked() {
                        trees[s].fold_blocks(challenge, None, Some(&equality), true)
                    } else {
                        trees[s].fold_and_message(challenge, &equality)
                    }
                });
                if let Some(from_columns) = from_columns {
                    add_messages(&mut messages, from_columns);
                }
            } else {
                // The last round has no weights left to weight a message by and needs
                // the fold alone. Both kernels stay for that reason; `fold` is not dead.
                if let Some(columns) = columns.filter(|_| round == usize::from(polynomial.is_some())) {
                    let (first, second) = if polynomial.is_some() {
                        (round_point[0], Some(challenge))
                    } else {
                        (challenge, None)
                    };
                    fold_columns(columns, &mut trees, first, second, None);
                }
                for tree in &mut trees {
                    tree.fold(challenge);
                }
            }
        }

        for tree in &trees {
            ps.add_scalars(&tree.children());
        }
        let low_challenge = ps.sample();
        let high_challenge = ps.sample();
        for (value, tree) in values.iter_mut().zip(&trees) {
            let tail = tree.children();
            *value = interp(
                interp(tail[0], tail[1], low_challenge),
                interp(tail[2], tail[3], low_challenge),
                high_challenge,
            );
        }
        point = vec![low_challenge, high_challenge];
        point.extend_from_slice(&round_point);
        layer -= 2;
    }

    ProductTriple { roots, point, values }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mle_eval_e(table: &[F192], point: &[F192]) -> F192 {
        assert_eq!(table.len(), 1 << point.len());
        let mut folded = table.to_vec();
        for &challenge in point {
            let half = folded.len() / 2;
            for row in 0..half {
                folded[row] = interp(folded[2 * row], folded[2 * row + 1], challenge);
            }
            folded.truncate(half);
        }
        folded[0]
    }

    #[test]
    fn quartic_round_message_matches_direct_evaluation() {
        for width in [2, 4, 8, 16, 32, 128] {
            let below: Vec<F192> = (0..4 * width)
                .map(|i| F192::new((17 * i + width + 1) as u64, (i * i + 3) as u64, (5 * i + 7) as u64))
                .collect();
            let table = below.clone();
            let state = QuaternaryLayerState::new(below, width);
            let equality = Equality {
                lo: (0..width / 2)
                    .map(|i| F192::new((31 * i + 5) as u64, (7 * i + 1) as u64, (11 * i + 9) as u64))
                    .collect(),
                hi: vec![F192::ONE],
            };
            let [difference, c2, c3, c4] = state.round_message(&equality);
            let direct = |point: F192| {
                (0..width / 2).fold(F192::ZERO, |sum, row| {
                    let values =
                        [0, 1, 2, 3].map(|child| interp(table[8 * row + child], table[8 * row + 4 + child], point));
                    sum + equality.at(row) * values[0] * values[1] * values[2] * values[3]
                })
            };
            let c0 = direct(F192::ZERO);
            let c1 = difference + c2 + c3 + c4;
            for point in [F192::ZERO, F192::ONE, F192::Y, F192::Y.square()] {
                assert_eq!(
                    c0 + point * (c1 + point * (c2 + point * (c3 + point * c4))),
                    direct(point)
                );
            }
        }
    }

    /// The tables agree wherever either stores a value, the rest being the identity.
    fn assert_same_table(left: &QuaternaryLayerState, right: &QuaternaryLayerState, context: &str) {
        let (left, right) = (left.table(), right.table());
        for i in 0..left.len().max(right.len()) {
            let at = |table: &[F192]| table.get(i).copied().unwrap_or(F192::ONE);
            assert_eq!(at(&left), at(&right), "{context}, value {i}");
        }
    }

    #[test]
    fn fused_fold_matches_separate_fold_and_message() {
        for width in [4usize, 16, 64, 256, 1 << 14] {
            let lengths = [
                1, 4, 5, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 192, 193, 320, 1000, 4097,
            ];
            for len in lengths.into_iter().chain([4 * width - 5, 4 * width - 1, 4 * width]) {
                if len > 4 * width {
                    continue;
                }
                let values: Vec<F192> = (0..len)
                    .map(|i| F192::new((17 * i + 1) as u64, (i * i + 3) as u64, (5 * i + 7) as u64))
                    .collect();
                let mut reference = QuaternaryLayerState::from_rows(values.to_vec(), width);
                let mut separate = QuaternaryLayerState::new(values.to_vec(), width);
                let mut fused = QuaternaryLayerState::new(values, width);
                let context = format!("width={width}, len={len}");
                let point: Vec<F192> = (0..width.ilog2() - 1)
                    .map(|i| F192::new(31 + u64::from(i), 7, 11))
                    .collect();
                let mut dropped = 1;
                while reference.logical_rows > 2 {
                    let challenge = F192::new(reference.logical_rows as u64, 13, 19);
                    reference.fold(challenge);
                    separate.fold(challenge);
                    let equality = Equality::new(&point[dropped..]);
                    dropped += 1;
                    let message = fused.fold_and_message(challenge, &equality);
                    assert_eq!(message, reference.round_message(&equality), "{context}");
                    assert_eq!(message, separate.round_message(&equality), "{context}");
                    assert_same_table(&fused, &reference, &context);
                    assert_same_table(&separate, &reference, &context);
                }
                reference.fold(F192::Y);
                separate.fold(F192::Y);
                fused.fold(F192::Y);
                assert_eq!(fused.children(), reference.children(), "{context}");
                assert_eq!(separate.children(), reference.children(), "{context}");
            }
        }
    }

    /// Distinct nonzero base-field values.
    fn base_values(len: usize, seed: u64) -> Vec<F64> {
        (0..len as u64)
            .map(|i| F64((i + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed.wrapping_mul(0xD6E8_FEB8_6659_FD93) | 1))
            .collect()
    }

    #[test]
    fn base_field_first_round_matches_the_extension_path() {
        for width in [4usize, 16, 64, 256, 1 << 14] {
            let lengths = [1, 4, 5, 33, 64, 65, 127, 128, 129, 192, 193, 320, 1000, 4097];
            for len in lengths.into_iter().chain([4 * width - 5, 4 * width]) {
                if len > 4 * width {
                    continue;
                }
                let values = base_values(len, width as u64);
                let embedded: Vec<F192> = values.iter().map(|&k| F192::from(k)).collect();
                let mut reference = QuaternaryLayerState::from_rows(embedded.clone(), width);
                let mut separate = QuaternaryLayerState::new(values.clone(), width);
                let mut fused = QuaternaryLayerState::new(values, width);
                let context = format!("width={width}, len={len}");
                let point: Vec<F192> = (0..width.ilog2() - 1)
                    .map(|i| F192::new(31 + u64::from(i), 7, 11))
                    .collect();
                let mut equality = Equality::new(&point);
                let mut dropped = 1;
                assert_eq!(
                    fused.round_message(&equality),
                    reference.round_message(&equality),
                    "{context}"
                );
                while reference.logical_rows > 2 {
                    let challenge = F192::new(reference.logical_rows as u64, 13, 19);
                    reference.fold(challenge);
                    separate.fold(challenge);
                    equality = Equality::new(&point[dropped..]);
                    dropped += 1;
                    let message = fused.fold_and_message(challenge, &equality);
                    assert_eq!(message, reference.round_message(&equality), "{context}");
                    assert_eq!(message, separate.round_message(&equality), "{context}");
                    assert_same_table(&fused, &reference, &context);
                    assert_same_table(&separate, &reference, &context);
                }
                reference.fold(F192::Y);
                separate.fold(F192::Y);
                fused.fold(F192::Y);
                assert_eq!(fused.children(), reference.children(), "{context}");
                assert_eq!(separate.children(), reference.children(), "{context}");
            }
        }
    }

    #[test]
    fn base_field_count_tree_gives_the_same_proof() {
        for mu in 0..=13usize {
            let lengths = [
                1usize << mu,
                (1usize << mu) - 3,
                (1usize << mu.saturating_sub(1)) + 1,
                1,
            ];
            for count_len in lengths {
                if count_len == 0 || count_len > 1 << mu {
                    continue;
                }
                let push: Vec<F192> = (0..1usize << mu)
                    .map(|row| F192::new((1 + row) as u64, row as u64, 3))
                    .collect();
                let pull: Vec<F192> = push.iter().map(|&value| value + F192::Y).collect();
                let count = base_values(count_len, mu as u64);
                let mut dense: Vec<F192> = count.iter().map(|&k| F192::from(k)).collect();
                let mut extension_ps = ProverState::from_label(b"base-field-gkr-test");
                let extension = prove_product_triple(
                    [push.clone(), pull.clone(), dense.clone()],
                    &mut extension_ps,
                    RootShape::Distinct,
                );
                let mut base_ps = ProverState::from_label(b"base-field-gkr-test");
                let base = prove_product_triple_from(
                    Leaves::new(push),
                    Leaves::new(pull),
                    Leaves::<F64>::new(count),
                    None,
                    &mut base_ps,
                    RootShape::Distinct,
                );
                dense.resize(1 << mu, F192::ONE);
                assert_eq!(
                    base.values[2],
                    mle_eval_e(&dense, &base.point),
                    "mu={mu}, len={count_len}"
                );
                assert_eq!(base.roots, extension.roots, "mu={mu}, len={count_len}");
                assert_eq!(base.point, extension.point, "mu={mu}, len={count_len}");
                assert_eq!(base.values, extension.values, "mu={mu}, len={count_len}");
                assert_eq!(base_ps.into_proof().stream, extension_ps.into_proof().stream, "mu={mu}");
            }
        }
    }

    #[test]
    fn radix_four_claims_are_true_at_even_and_odd_depths() {
        for mu in 0..=12 {
            let leaves: [Vec<F192>; 3] = [0, 1, 2].map(|lane| {
                (0..1usize << mu)
                    .map(|row| F192::new((1 + row + lane * 100_003) as u64, row as u64, lane as u64))
                    .collect()
            });
            let expected_roots = leaves
                .each_ref()
                .map(|lane| lane.iter().copied().fold(F192::ONE, |product, value| product * value));
            let mut ps = ProverState::from_label(b"radix-four-gkr-test");
            let proved = prove_product_triple(leaves.each_ref().map(|l| l.to_vec()), &mut ps, RootShape::Distinct);
            assert_eq!(proved.roots, expected_roots);
            for lane in 0..3 {
                assert_eq!(proved.values[lane], mle_eval_e(&leaves[lane], &proved.point));
            }
        }
    }

    #[test]
    fn implicit_identity_suffix_matches_dense_padding() {
        for mu in 3..=10 {
            let lengths = [(1usize << mu) - 3, (1usize << (mu - 1)) + 1, (1usize << (mu - 2)) + 3];
            let leaves: [Vec<F192>; 3] = std::array::from_fn(|lane| {
                (0..lengths[lane])
                    .map(|row| F192::new((3 + row + lane * 10_007) as u64, row as u64, lane as u64))
                    .collect()
            });
            let dense = leaves.each_ref().map(|lane| {
                let mut padded = lane.clone();
                padded.resize(1 << mu, F192::ONE);
                padded
            });
            let mut sparse_ps = ProverState::from_label(b"sparse-radix-four-gkr-test");
            let proved = prove_product_triple(
                leaves.each_ref().map(|l| l.to_vec()),
                &mut sparse_ps,
                RootShape::Distinct,
            );
            for lane in 0..3 {
                assert_eq!(proved.values[lane], mle_eval_e(&dense[lane], &proved.point));
                assert_eq!(
                    proved.roots[lane],
                    dense[lane]
                        .iter()
                        .copied()
                        .fold(F192::ONE, |product, value| product * value)
                );
            }
            let proof = sparse_ps.into_proof();
            let mut dense_ps = ProverState::from_label(b"sparse-radix-four-gkr-test");
            let dense_proved =
                prove_product_triple(dense.each_ref().map(|l| l.to_vec()), &mut dense_ps, RootShape::Distinct);
            assert_eq!(dense_proved.roots, proved.roots);
            assert_eq!(dense_proved.point, proved.point);
            assert_eq!(dense_proved.values, proved.values);
            assert_eq!(dense_ps.into_proof().stream, proof.stream);
        }
    }

    /// Runs of identity leaves in the middle of the tree, whose whole blocks hold garbage at the leaves
    /// and at the quads, give the proof of the dense tree.
    #[test]
    fn interior_identity_runs_match_dense_padding() {
        for mu in 10..=14 {
            let len = (1usize << mu) - 5;
            let runs = vec![
                len / 8 + 5..len / 4 - 3,
                len / 4 + 70..len / 2 + 7,
                5 * len / 8..len - 70,
            ];
            let tree = |lane: usize| -> Vec<F192> {
                (0..len)
                    .map(|row| {
                        if runs.iter().any(|run| run.contains(&row)) {
                            F192::ONE
                        } else {
                            F192::new((3 + row + lane * 10_007) as u64, row as u64, lane as u64)
                        }
                    })
                    .collect()
            };
            let garbage = F192::new(0xDEAD, 0xBEEF, 0xF00D);
            let sparse = |lane: usize| {
                let mut leaves = Leaves::new(tree(lane));
                let LeafData::Blocks(blocks) = &mut leaves.leaves else {
                    unreachable!();
                };
                for b in identity_blocks(&runs, 0, leaves.len).into_iter().flatten() {
                    blocks[b] = [F192x8::splat(garbage); 8];
                }
                for b in identity_blocks(&runs, 2, leaves.quads.len()).into_iter().flatten() {
                    leaves.quads[BLOCK_VALUES * b..BLOCK_VALUES * (b + 1)].fill(garbage);
                }
                leaves.ones = runs.clone();
                leaves
            };
            let mut sparse_ps = ProverState::from_label(b"interior-identity-gkr-test");
            let proved = prove_product_triple_from(
                sparse(0),
                sparse(1),
                Leaves::new(tree(2)),
                None,
                &mut sparse_ps,
                RootShape::Distinct,
            );
            let mut dense_ps = ProverState::from_label(b"interior-identity-gkr-test");
            let dense = prove_product_triple([tree(0), tree(1), tree(2)], &mut dense_ps, RootShape::Distinct);
            assert_eq!(dense.roots, proved.roots, "mu={mu}");
            assert_eq!(dense.point, proved.point, "mu={mu}");
            assert_eq!(dense.values, proved.values, "mu={mu}");
            assert_eq!(dense_ps.into_proof().stream, sparse_ps.into_proof().stream, "mu={mu}");
        }
    }
}

#[cfg(test)]
mod two_round_tests {
    use super::*;

    #[inline(never)]
    fn first_weights(weights: &[F192x8], eta: F192) -> Vec<F192x8> {
        weights
            .iter()
            .flat_map(|w| {
                let w = w.to_array();
                std::array::from_fn::<_, 2, _>(|b| {
                    F192x8::from_array(&std::array::from_fn(|i| {
                        w[4 * b + i / 2] * (if i % 2 == 0 { F192::ONE + eta } else { eta })
                    }))
                })
            })
            .collect()
    }

    #[inline(never)]
    fn baseline(
        input: &[Block],
        half: &mut [Block],
        quarter: &mut [[F192x8; 4]],
        weights: &[F192x8],
        first_weights: &[F192x8],
        r1: F192,
        r2: F192,
    ) -> [[F192; 4]; 2] {
        let mut sums = [[Prod8::ZERO; 4]; 2];
        for (block, weight) in input.iter().zip(first_weights) {
            sums[0] = xor_coefficients(sums[0], block.summand(*weight));
        }
        let r1 = F192x8::splat(r1);
        for ((pair, out), weight) in input.chunks_exact(2).zip(half.iter_mut()).zip(weights) {
            let folded = interleave(pair[0].fold(r1), pair[1].fold(r1));
            sums[1] = xor_coefficients(sums[1], folded.summand(*weight));
            *out = folded;
        }
        let r2 = F192x8::splat(r2);
        for (block, out) in half.iter().zip(quarter) {
            *out = block.fold(r2);
        }
        sums.map(|sum| sum.map(Prod8::reduce))
    }

    #[inline(never)]
    fn fused(
        input: &[Block],
        quarter: &mut [[F192x8; 4]],
        weights: &[F192x8],
        eta: F192,
        r1: F192,
        r2: F192,
    ) -> [[F192; 4]; 2] {
        let mut sum = [[Prod8::ZERO; 5]; 6];
        for (pair, weight) in input.chunks_exact(2).zip(weights) {
            bivariate_summand(&pair[0], &pair[1], *weight, &mut sum);
        }
        let result = bivariate_messages(bivariate_coefficients(&sum), eta, r1);
        let (r1, r2) = (F192x8::splat(r1), F192x8::splat(r2));
        for (pair, out) in input.chunks_exact(2).zip(quarter) {
            *out = interleave(pair[0].fold(r1), pair[1].fold(r1)).fold(r2);
        }
        result
    }

    fn random(seed: &mut u64) -> F192 {
        let mut word = || {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            *seed
        };
        F192::new(word(), word(), word())
    }

    #[test]
    fn two_round_tensor_matches_normal_messages() {
        let mut seed = 314159265;
        for len in [0usize, 1, 4, 7, 31, 63, 64, 65, 100, 127, 128, 257, 512] {
            for kind in 0..3 {
                let values: Vec<_> = (0..len)
                    .map(|_| match kind {
                        0 => random(&mut seed),
                        1 => F192::ONE,
                        _ => F192::ZERO,
                    })
                    .collect();
                let pairs = len.div_ceil(128).max(1);
                let input: Vec<_> = (0..2 * pairs)
                    .map(|b| F192::block(values.get(64 * b..(64 * b + 64).min(len)).unwrap_or_default()))
                    .collect();
                let weights: Vec<_> = (0..pairs)
                    .map(|_| F192x8::from_array(&std::array::from_fn(|_| random(&mut seed))))
                    .collect();
                let (eta, r1, r2) = (random(&mut seed), random(&mut seed), random(&mut seed));
                let mut half = vec![[F192x8::ZERO; 8]; pairs];
                let mut q0 = vec![[F192x8::ZERO; 4]; pairs];
                let mut q1 = vec![[F192x8::ZERO; 4]; pairs];
                assert_eq!(
                    baseline(
                        &input,
                        &mut half,
                        &mut q0,
                        &weights,
                        &first_weights(&weights, eta),
                        r1,
                        r2
                    ),
                    fused(&input, &mut q1, &weights, eta, r1, r2)
                );
                for (a, b) in q0.iter().flatten().zip(q1.iter().flatten()) {
                    assert_eq!(a.to_array(), b.to_array());
                }
                let mut coefficients = [[Prod8::ZERO; 5]; 6];
                for (pair, weight) in input.chunks_exact(2).zip(&weights) {
                    bivariate_summand(&pair[0], &pair[1], *weight, &mut coefficients);
                }
                let coefficients = bivariate_coefficients(&coefficients);
                for (x, y) in [(F192::ZERO, F192::ONE), (r1, r2)] {
                    let actual = coefficients.iter().rev().fold(F192::ZERO, |sum, row| {
                        sum * x + row.iter().rev().fold(F192::ZERO, |v, c| v * y + *c)
                    });
                    let mut expected = F192::ZERO;
                    for g in 0..8 * pairs {
                        let child = |c: usize| {
                            let v = |r: usize| values.get(16 * g + 4 * r + c).copied().unwrap_or(F192::ONE);
                            interp(interp(v(0), v(1), x), interp(v(2), v(3), x), y)
                        };
                        expected += weights[g / 8].lane(g % 8) * child(0) * child(1) * child(2) * child(3);
                    }
                    assert_eq!(actual, expected);
                }
            }
        }
    }

    #[test]
    fn deferred_column_folds_match_dense_proofs() {
        struct TestColumns {
            blocks: [Vec<Block>; 2],
            runs: [Vec<Range<usize>>; 2],
        }
        impl LeafColumns for TestColumns {
            fn runs(&self, side: usize) -> &[Range<usize>] {
                &self.runs[side]
            }
            fn tasks(&self) -> usize {
                self.blocks[0].len() / 4
            }
            fn task(&self, t: usize, f: &mut dyn FnMut(usize, usize, &Block)) {
                // The sides interleave block by block, as a twin's leaves do.
                for b in 4 * t..4 * t + 4 {
                    for side in (0..2).filter(|&side| in_runs(&self.runs[side], 4 * t)) {
                        f(side, b, &self.blocks[side][b]);
                    }
                }
            }
        }
        for (mu, short_pull) in (9..=14).flat_map(|mu| [(mu, false), (mu, true)]) {
            let len = (1 << mu) - 5;
            let tree = |side: usize| -> Vec<F192> {
                (0..if short_pull && side == 1 { 13 } else { len })
                    .map(|i| F192::new((i + 73 * side + 1) as u64, (i * i + 1) as u64, (i * 17 + side) as u64))
                    .collect()
            };
            let columns = TestColumns {
                blocks: [blocks_from_values(&tree(0)), blocks_from_values(&tree(1))],
                runs: std::array::from_fn(|side| {
                    if short_pull && side == 1 {
                        Vec::new()
                    } else {
                        vec![0..4, 8..(len / BLOCK_VALUES / 4 * 4).max(8)]
                    }
                }),
            };
            let sparse = |side| {
                let mut leaves = Leaves::new(tree(side));
                let LeafData::Blocks(blocks) = &mut leaves.leaves else {
                    unreachable!()
                };
                for b in columns.runs[side].iter().cloned().flatten() {
                    blocks[b] = [F192x8::ZERO; 8];
                }
                leaves
            };
            let mut ps = ProverState::from_label(b"two-round-columns");
            let proof = prove_product_triple_from(
                sparse(0),
                sparse(1),
                Leaves::new(tree(2)),
                Some(&columns),
                &mut ps,
                RootShape::Distinct,
            );
            let mut dense_ps = ProverState::from_label(b"two-round-columns");
            let dense = prove_product_triple([tree(0), tree(1), tree(2)], &mut dense_ps, RootShape::Distinct);
            assert_eq!(proof.point, dense.point, "mu={mu}");
            assert_eq!(proof.values, dense.values, "mu={mu}");
            assert_eq!(ps.into_proof().stream, dense_ps.into_proof().stream, "mu={mu}");
        }
    }
}
