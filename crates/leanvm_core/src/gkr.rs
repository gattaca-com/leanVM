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
use primitives::field::{F64, F64x8, F192, F192x8, Prod8, mul2};
use primitives::multilinear::{eq_table, interp};
use primitives::stream::Stream;
use std::ops::{Add, Deref, DerefMut, Mul, Range};

/// Stored leaf blocks: on the heap, or in a mapping of their own when almost none are written.
///
/// With several workers, GKR recomputes almost every bus leaf block or skips it as identity.
/// On the heap, the unwritten extent would host later buffers and become resident.
/// A mapping faults in only the written pages, and goes back to the system after the first fold.
/// Leaves written in full stay on the heap, where the first fold reuses their buffer.
pub enum Store<T> {
    Heap(Vec<T>),
    Mapped(Mapping<T>),
}

impl<T> Deref for Store<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match self {
            Self::Heap(v) => v,
            Self::Mapped(m) => m,
        }
    }
}

impl<T> DerefMut for Store<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            Self::Heap(v) => v,
            Self::Mapped(m) => m,
        }
    }
}

/// `len` plain values, zero until written, in an anonymous mapping of their own.
pub struct Mapping<T> {
    ptr: std::ptr::NonNull<T>,
    len: usize,
    bytes: usize,
}

// SAFETY: the mapping is owned like a `Vec<T>`, and shared access only reads.
unsafe impl<T: Send> Send for Mapping<T> {}
// SAFETY: as for `Send`.
unsafe impl<T: Sync> Sync for Mapping<T> {}

impl<T: primitives::Plain> Mapping<T> {
    /// `len` zero values. Writes are scattered, so huge pages stay off: each would fault in 2 MiB.
    pub fn new(len: usize) -> Self {
        let bytes = (len * size_of::<T>()).max(1);
        // SAFETY: an anonymous private mapping, checked for failure; zero bytes are plain values.
        unsafe {
            let p = libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            );
            assert!(p != libc::MAP_FAILED, "no mapping of {bytes} bytes for the leaves");
            #[cfg(target_os = "linux")]
            libc::madvise(p, bytes, libc::MADV_NOHUGEPAGE);
            Self {
                ptr: std::ptr::NonNull::new_unchecked(p.cast()),
                len,
                bytes,
            }
        }
    }
}

impl<T> Deref for Mapping<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: the mapping holds `len` plain values.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> DerefMut for Mapping<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as in `deref`; the mapping is owned, so the borrow is unique.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> Drop for Mapping<T> {
    fn drop(&mut self) {
        // SAFETY: the mapping `new` made, unmapped once.
        unsafe { libc::munmap(self.ptr.as_ptr().cast(), self.bytes) };
    }
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

#[cfg(test)]
impl<F: TreeField> Leaves<'_, F> {
    pub fn new(values: Vec<F>) -> Self {
        let leaves = blocks_from_values(&values);
        let quads = (0..values.len().div_ceil(4)).map(|k| quad_at(&leaves, k)).collect();
        Self {
            leaves: LeafData::Blocks(Store::Heap(leaves)),
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
    Blocks(Store<F::Block>),
    Columns(Columns<'a, F>),
}

impl<F: TreeField> Default for LeafData<'_, F> {
    fn default() -> Self {
        Self::Blocks(Store::Heap(Vec::new()))
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
                QuaternaryLayerState::small((0..len).map(|l| columns.read::<1>(l)[0].embed()), width)
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
/// block or less has none.
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
    /// The quartic summand of the block's eight row pairs, one pair per lane, weighted by `equality` and
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

/// Eight values in one register: `F192x8`, or `F64x8` for the values of a tree in `K`.
pub trait Lanes: primitives::Plain + Copy + Send + Sync + Add<Output = Self> + Mul<Output = Self> {
    type Elem: primitives::Plain;
    const ZERO: Self;
    const ONE: Self;
    fn from_array(values: &[Self::Elem; 8]) -> Self;
    fn to_array(self) -> [Self::Elem; 8];
    fn lane(&self, i: usize) -> Self::Elem;
    /// The 8x8 transpose: lane `j` of `out[p]` is lane `p` of `g[j]`.
    fn transpose(g: [Self; 8]) -> [Self; 8];
    /// `([a0, a2, a4, a6, b0, b2, b4, b6], [a1, a3, a5, a7, b1, b3, b5, b7])`.
    fn even_odd(a: Self, b: Self) -> (Self, Self);
    /// Every lane times `x`, the generator of `K`: a shift per base-field word.
    fn mul_x(self) -> Self;
}

macro_rules! impl_lanes {
    ($lanes:ty, $elem:ty) => {
        impl Lanes for $lanes {
            type Elem = $elem;
            const ZERO: Self = <$lanes>::ZERO;
            const ONE: Self = <$lanes>::ONE;

            #[inline(always)]
            fn from_array(values: &[$elem; 8]) -> Self {
                <$lanes>::from_array(values)
            }

            #[inline(always)]
            fn to_array(self) -> [$elem; 8] {
                <$lanes>::to_array(self)
            }

            #[inline(always)]
            fn lane(&self, i: usize) -> $elem {
                <$lanes>::lane(self, i)
            }

            #[inline(always)]
            fn transpose(g: [Self; 8]) -> [Self; 8] {
                <$lanes>::transpose(g)
            }

            #[inline(always)]
            fn even_odd(a: Self, b: Self) -> (Self, Self) {
                <$lanes>::even_odd(a, b)
            }

            #[inline(always)]
            fn mul_x(self) -> Self {
                <$lanes>::mul_x(self)
            }
        }
    };
}

impl_lanes!(F192x8, F192);
impl_lanes!(F64x8, F64);

/// The field a product tree's values live in: `E`, or `K` for a tree that the first fold lifts to
/// `E`. The two share every level's shape, so a tree is built and proved once, generically, and
/// only the arithmetic differs.
pub trait TreeField: primitives::Plain + Send + Sync + Mul<Output = Self> + 'static {
    type Lanes: Lanes<Elem = Self>;
    /// Sixty-four values as eight [`Self::Lanes`], in the layout of a [`Block`].
    type Block: TableBlock + From<[Self::Lanes; 8]> + AsRef<[Self::Lanes]>;
    const ONE: Self;

    fn embed(self) -> F192;

    /// `c[0] · c[1] · c[2] · c[3]`.
    fn product4(c: [Self; 4]) -> Self;

    /// The layer state over a table of `len` values held as blocks, whose tail is the identity.
    fn state_from_blocks(blocks: Store<Self::Block>, len: usize, width: usize) -> QuaternaryLayerState<'static>;

    /// The products of eight consecutive rows of four, from their 32 children: an even-odd split
    /// pairs neighbours, so two rounds of it leave four-fold products in order.
    #[inline(always)]
    fn eight_products(children: &[Self]) -> [Self; 8] {
        let eight = |i: usize| Self::Lanes::from_array(children[8 * i..8 * i + 8].try_into().unwrap());
        let pairs = |x, y| {
            let (even, odd) = Self::Lanes::even_odd(x, y);
            even * odd
        };
        let (even, odd) = Self::Lanes::even_odd(pairs(eight(0), eight(1)), pairs(eight(2), eight(3)));
        (even * odd).to_array()
    }

    /// The block of up to 64 values, the missing tail being the identity.
    #[inline(always)]
    fn block(values: &[Self]) -> Self::Block {
        // Group `j` of eight values is lane `j` of the eight vectors.
        let block = |chunk: &[Self; BLOCK_VALUES]| {
            Self::Lanes::transpose(std::array::from_fn(|j| {
                Self::Lanes::from_array(chunk[8 * j..8 * j + 8].try_into().unwrap())
            }))
            .into()
        };
        match <&[Self; BLOCK_VALUES]>::try_from(values) {
            Ok(chunk) => block(chunk),
            Err(_) => {
                let mut padded = [Self::ONE; BLOCK_VALUES];
                padded[..values.len()].copy_from_slice(values);
                block(&padded)
            }
        }
    }

    /// Leaf `l` of a block table.
    #[inline(always)]
    fn leaf(blocks: &[Self::Block], l: usize) -> Self {
        blocks[l / BLOCK_VALUES].as_ref()[l % 8].lane(l / 8 % 8)
    }
}

impl TreeField for F192 {
    type Lanes = F192x8;
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

    fn state_from_blocks(blocks: Store<Block>, _len: usize, width: usize) -> QuaternaryLayerState<'static> {
        match blocks {
            Store::Heap(blocks) => QuaternaryLayerState::from_blocks(blocks, width),
            Store::Mapped(leaves) => QuaternaryLayerState::from_leaves(leaves, width),
        }
    }
}

impl TreeField for F64 {
    type Lanes = F64x8;
    type Block = KBlock;
    const ONE: Self = F64::ONE;

    fn embed(self) -> F192 {
        self.into()
    }

    #[inline(always)]
    fn product4([c0, c1, c2, c3]: [Self; 4]) -> Self {
        (c0 * c1) * (c2 * c3)
    }

    fn state_from_blocks(blocks: Store<KBlock>, len: usize, width: usize) -> QuaternaryLayerState<'static> {
        if blocks.len() < 2 {
            return QuaternaryLayerState::small((0..len).map(|l| Self::leaf(&blocks, l).embed()), width);
        }
        QuaternaryLayerState::from_table(Box::new(blocks), width)
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
                for (i, slots) in out.as_chunks_mut::<8>().0.iter_mut().enumerate() {
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
                // SAFETY: the fill writes every element before it reads any.
                None => unsafe { primitives::filled_vec(full_rows, fill_rows) },
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
#[cfg(test)]
fn blocks_from_values<F: TreeField>(values: &[F]) -> Vec<F::Block> {
    values.chunks(BLOCK_VALUES).map(F::block).collect()
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

/// A quadratic's values at `0, 1, ∞`, extended to the five points `0, 1, ∞, x, x + 1` that
/// [`bivariate_summand`] evaluates both variables at. `∞` stands for the leading coefficient.
#[inline(always)]
fn extend<T: Lanes>([v0, v1, top]: [T; 3]) -> [T; 5] {
    // q = v0 + (v0 + v1 + top) X + top X^2, and q(x + 1) = q(x) + q(1) + q(0).
    let at_x = v0 + (v0 + v1 + top + top.mul_x()).mul_x();
    [v0, v1, top, at_x, at_x + v0 + v1]
}

/// The product of two multilinear factors at the points `0, 1, ∞` of X and the five points of Y,
/// `[X][Y]`. Each factor is given by its corners `[[f(0, 0), f(1, 0)], [f(0, 1), f(1, 1)]]`.
#[inline(always)]
fn bilinear_columns<T: Lanes>(
    [[f00, f10], [f01, f11]]: [[T; 2]; 2],
    [[g00, g10], [g01, g11]]: [[T; 2]; 2],
) -> [[T; 5]; 3] {
    // The column at X = ∞ is the slope in X, and Y = ∞ is the slope in Y.
    let column = |f0: T, f1: T, g0: T, g1: T| extend([f0 * g0, f1 * g1, (f0 + f1) * (g0 + g1)]);
    [
        column(f00, f01, g00, g01),
        column(f10, f11, g10, g11),
        column(f00 + f10, f01 + f11, g00 + g10, g01 + g11),
    ]
}

/// Adds `product(a(x, y), b(x, y))` at the 5 × 5 points to `sum`, one row of Y at a time.
#[inline(always)]
fn add_products<T: Lanes>(sum: &mut BivariateSum, a: [[T; 5]; 3], b: [[T; 5]; 3], product: impl Fn(T, T) -> Prod8) {
    for (y, row) in sum.iter_mut().enumerate() {
        let a = extend([a[0][y], a[1][y], a[2][y]]);
        let b = extend([b[0][y], b[1][y], b[2][y]]);
        for x in 0..5 {
            row[x] ^= product(a[x], b[x]);
        }
    }
}

/// The pair's four factors by their corners, `[Y][X]`: X folds the low row bit, and Y the next.
#[inline(always)]
fn pair_corners<T: Copy>(first: &[T; 8], second: &[T; 8], even_odd: impl Fn(T, T) -> (T, T)) -> [[[T; 2]; 2]; 4] {
    std::array::from_fn(|c| {
        let (v0, v2) = even_odd(first[c], second[c]);
        let (v1, v3) = even_odd(first[c + 4], second[c + 4]);
        [[v0, v2], [v1, v3]]
    })
}

/// The summand of [`bivariate_summand`] at the 5 × 5 points, `[Y][X]`, summed over pairs.
type BivariateSum = [[Prod8; 5]; 5];

/// The coefficients of `X^j Y^i` of a [`BivariateSum`], at `[i][j]`.
fn bivariate_coefficients(sum: &BivariateSum) -> [[F192; 5]; 5] {
    let [over_x, over_x_squared, over_x_one] = [2, 6, 3].map(|k| F64(k).inv());
    // A quartic's coefficients from its values at the five points.
    let interpolate = |[v0, v1, top, at_x, at_x_one]: [F192; 5]| {
        // c3 (x^2 + x) = q(0) + q(1) + q(x) + q(x + 1).
        let c3 = (v0 + v1 + at_x + at_x_one).mul_base(over_x_squared);
        let ones = v0 + v1 + top + c3;
        // (q(x) + q(0) + top x^4 + c3 x^3) / x = c1 + c2 x, and c1 + c2 = ones.
        let line = (at_x + v0 + top.mul_base(F64(16)) + c3.mul_base(F64(8))).mul_base(over_x);
        let c2 = (line + ones).mul_base(over_x_one);
        [v0, ones + c2, c2, c3, top]
    };
    let rows = sum.map(|row| interpolate(row.map(Prod8::reduce)));
    let columns: [[F192; 5]; 5] = std::array::from_fn(|j| interpolate(rows.map(|row| row[j])));
    std::array::from_fn(|i| std::array::from_fn(|j| columns[j][i]))
}

/// Adds the pair's summand, the weighted product of its four factors, to `sum` at the 5 × 5 points.
/// Building the two factor pairs takes 4 + 18 products and their product 25, where Karatsuba in both
/// variables takes 36. The extension to the points costs shifts.
#[inline(never)]
fn bivariate_summand(first: &Block, second: &Block, weight: F192x8, sum: &mut BivariateSum) {
    let factors = pair_corners(first, second, F192x8::even_odd);
    let weighted = factors[0].map(|row| row.map(|v| weight * v));
    let (a, b) = (
        bilinear_columns(weighted, factors[1]),
        bilinear_columns(factors[2], factors[3]),
    );
    add_products(sum, a, b, |a, b| a.mul_unreduced(b).merge_parities());
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
    /// the 25 values as mixed products.
    #[inline(never)]
    fn bivariate(&self, second: &Self, weight: F192x8, sum: &mut BivariateSum) {
        let factors = pair_corners(self, second, F64x8::even_odd);
        let (a, b) = (
            bilinear_columns(factors[0], factors[1]),
            bilinear_columns(factors[2], factors[3]),
        );
        add_products(sum, a, b, |a, b| weight.mul_base_unreduced(a * b).merge_parities());
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
        pairs: Option<&Equality>,
    ) -> ([F192; 4], Option<[[F192; 5]; 5]>);
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
        pairs: Option<&Equality>,
    ) -> ([F192; 4], Option<[[F192; 5]; 5]>) {
        fold_block_pairs(self, next_blocks, ones, challenge, second, equality, pairs)
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
fn columns_polynomial(columns: &dyn LeafColumns, equality: &Equality) -> [[[F192; 5]; 5]; 2] {
    let task = |t| {
        let mut sums = [[[Prod8::ZERO; 5]; 5]; 2];
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
        let mut sum = [[Prod8::ZERO; 5]; 5];
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
        parallel::map_reduce(windows, || [[Prod8::ZERO; 5]; 5], window, xor)
    } else {
        (0..windows).map(window).fold([[Prod8::ZERO; 5]; 5], xor)
    };
    bivariate_coefficients(&sum)
}

/// Fold every pair of blocks of `source` by `challenge`, and with a `second` challenge every pair of
/// the folded blocks again, into `E` blocks, and with an `equality` weigh the folded table's round
/// message as it is produced, or with `pairs` its next two rounds' polynomial ([`blocks_polynomial`]).
/// Folded blocks that are wholly identity are not written.
fn fold_block_pairs<S: Source + ?Sized>(
    source: &S,
    next_blocks: &mut [Block],
    ones: &[Range<usize>],
    challenge: F192,
    second: Option<F192>,
    equality: Option<&Equality>,
    pairs: Option<&Equality>,
) -> ([F192; 4], Option<[[F192; 5]; 5]>) {
    let count = next_blocks.len();
    let next_ones = match second {
        Some(_) => halve(&halve(ones, source.count().div_ceil(2)), count),
        None => halve(ones, count),
    };
    let span = if second.is_some() { 4 } else { 2 };
    let second = second.map(F192x8::splat);
    let challenge = F192x8::splat(challenge);
    // Even, so that no pair of folded blocks straddles two windows.
    let window = window_blocks(count).next_multiple_of(2);
    let task = |index: usize, out: &mut [Block]| {
        // The next round reads the destination; this round reads only the stage.
        let stream = Stream::new();
        let first = index * window;
        let (mut message, mut sum) = ([Prod8::ZERO; 4], [[Prod8::ZERO; 5]; 5]);
        // An even folded block waiting for its odd neighbour; an identity neighbour is never computed.
        let mut even: Option<(usize, Block)> = None;
        let one = <Block as TableBlock>::ONE;
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
            if let Some(pairs) = pairs {
                if let Some((e, left)) = even.take_if(|(e, _)| *e + 1 != k) {
                    left.bivariate(&one, pairs.block(e / 2), &mut sum);
                }
                if k % 2 == 0 {
                    even = Some((k, block));
                } else {
                    let left = even.take().map_or(one, |(_, left)| left);
                    left.bivariate(&block, pairs.block(k / 2), &mut sum);
                }
            }
        }
        if let (Some(pairs), Some((e, left))) = (pairs, even) {
            left.bivariate(&one, pairs.block(e / 2), &mut sum);
        }
        (message, sum)
    };
    type Sums = ([Prod8; 4], BivariateSum);
    let add = |(m, s): Sums, (n, t): Sums| -> Sums {
        (
            xor_coefficients(m, n),
            std::array::from_fn(|k| xor_coefficients(s[k], t[k])),
        )
    };
    let zero = || ([Prod8::ZERO; 4], [[Prod8::ZERO; 5]; 5]);
    let (message, sum) = if BLOCK_PAIRS * count >= PAR_THRESHOLD {
        parallel::map_reduce_chunks_mut(
            [next_blocks],
            [window],
            || (),
            zero,
            |(), acc, index, [out]| *acc = add(*acc, task(index, out)),
            add,
        )
    } else {
        next_blocks
            .chunks_mut(window)
            .enumerate()
            .map(|(index, out)| task(index, out))
            .fold(zero(), add)
    };
    (message.map(Prod8::reduce), pairs.map(|_| bivariate_coefficients(&sum)))
}

/// Two binary product levels contracted into one degree-four layer.
pub struct QuaternaryLayerState<'a> {
    /// The table as blocks, one block for a table of at most 64 values.
    /// Empty while the table is held as `first` or `leaves`.
    blocks: Vec<Block>,
    next_blocks: Vec<Block>,
    /// A leaf table of at least two blocks, read as `blocks` until the first fold unmaps it.
    leaves: Option<Mapping<Block>>,
    /// A table of at least two blocks that is not yet `E` blocks (base-field blocks, or rows), until
    /// the first fold: it then replaces `blocks`.
    first: Option<Box<dyn Table + 'a>>,
    /// Logical row count after identity padding. The blocks store an arbitrary
    /// prefix; every omitted row is the constant four-tuple one.
    logical_rows: usize,
    /// Runs of blocks that are wholly identity, which the table does not store.
    ones: Vec<Range<usize>>,
    /// Runs of blocks supplied by a [`LeafColumns`] pass rather than stored in the table.
    columns: Vec<Range<usize>>,
    /// The polynomial of the next two rounds, when a pass over the table produced it.
    pending: Option<[[F192; 5]; 5]>,
}

impl<'a> QuaternaryLayerState<'a> {
    fn new<F: TreeField>(values: Vec<F>, width: usize) -> Self {
        if values.len() <= BLOCK_VALUES {
            return Self::small(values.into_iter().map(F::embed), width);
        }
        Self::from_table(Box::new(Rows(values)), width)
    }

    /// The state over a table of at most one block.
    fn small(values: impl Iterator<Item = F192>, width: usize) -> Self {
        let values: Vec<F192> = values.collect();
        Self::from_blocks(vec![F192::block(&values)], width)
    }

    /// The state over a table held as blocks, whose tail is the identity.
    fn from_blocks(blocks: Vec<Block>, width: usize) -> Self {
        Self {
            // SAFETY: the first fold writes every slot of `next_blocks` before any read.
            next_blocks: unsafe { primitives::uninit_vec(blocks.len().div_ceil(2)) },
            blocks,
            leaves: None,
            first: None,
            logical_rows: width,
            ones: Vec::new(),
            columns: Vec::new(),
            pending: None,
        }
    }

    /// The state over a leaf table held as blocks, whose tail is the identity.
    fn from_leaves(leaves: Mapping<Block>, width: usize) -> Self {
        if leaves.len() < 2 {
            return Self::from_blocks(leaves.to_vec(), width);
        }
        Self {
            // SAFETY: the first fold writes every slot of `next_blocks` before any read.
            next_blocks: unsafe { primitives::uninit_vec(leaves.len().div_ceil(2)) },
            leaves: Some(leaves),
            ..Self::from_blocks(Vec::new(), width)
        }
    }

    /// The state over a table of at least two blocks that the first fold turns into `E` blocks.
    fn from_table(table: Box<dyn Table + 'a>, width: usize) -> Self {
        Self {
            // SAFETY: the first fold writes every slot of `next_blocks` before any read.
            next_blocks: unsafe { primitives::uninit_vec(table.count().div_ceil(2)) },
            blocks: Vec::new(),
            leaves: None,
            first: Some(table),
            logical_rows: width,
            ones: Vec::new(),
            columns: Vec::new(),
            pending: None,
        }
    }

    /// The runs of blocks that the stored table does not hold.
    fn unstored(&self) -> Vec<Range<usize>> {
        let mut runs = [self.ones.as_slice(), self.columns.as_slice()].concat();
        runs.sort_unstable_by_key(|run| run.start);
        runs
    }

    /// `(q(0)+q(1), [X²]q, [X³]q, [X⁴]q)`.
    fn round_message(&self, equality: &Equality) -> [F192; 4] {
        match &self.first {
            Some(first) => first.message(&self.ones, equality),
            None => blocks_message(&self.stored(), &self.unstored(), equality),
        }
    }

    /// The table as blocks, when it is not held as `first`.
    fn stored(&self) -> &[Block] {
        self.leaves.as_deref().unwrap_or(&self.blocks)
    }

    /// Fold every pair of blocks by `challenge` into `next_blocks`, and with an `equality` weigh the
    /// folded table's round message as it is produced.
    /// With a `second` challenge, the folded table is folded again by it in the same pass. With
    /// `pairs`, `(pair weights, η)`, the folded table's next two rounds go to `pending` and the message
    /// comes from them.
    fn fold_blocks(
        &mut self,
        challenge: F192,
        second: Option<F192>,
        equality: Option<&Equality>,
        pairs: Option<(&Equality, F192)>,
        keep_columns: bool,
    ) -> [F192; 4] {
        let equality = if pairs.is_some() { None } else { equality };
        let weights = pairs.map(|(weights, _)| weights);
        let blocks = self.first.as_ref().map_or(self.stored().len(), |first| first.count());
        let half = blocks.div_ceil(2);
        let count = if second.is_some() { half.div_ceil(2) } else { half };
        self.next_blocks.truncate(count);
        let first = self.first.take();
        let unstored = self.unstored();
        let (mut message, polynomial) = match &first {
            Some(first) => first.fold(&mut self.next_blocks, &unstored, challenge, second, equality, weights),
            None => fold_block_pairs(
                &self.leaves.as_deref().unwrap_or(&self.blocks),
                &mut self.next_blocks,
                &unstored,
                challenge,
                second,
                equality,
                weights,
            ),
        };
        if let (Some(polynomial), Some((_, eta))) = (polynomial, pairs) {
            message = bivariate_messages(polynomial, eta, F192::ZERO)[0];
            self.pending = Some(polynomial);
        }
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
        let leaves = self.leaves.take();
        if first.is_some() || leaves.is_some() {
            // The leaf table dies first, so the allocator can hand its pages to the next buffer.
            // Mapped leaves go back to the system.
            drop((first, leaves));
            // SAFETY: the next fold writes every slot of `next_blocks` before any read.
            self.next_blocks = unsafe { primitives::uninit_vec(count.div_ceil(2)) };
        }
        self.logical_rows /= 2;
        message
    }

    fn fold(&mut self, challenge: F192) {
        self.fold_blocks(challenge, None, None, None, false);
    }

    fn children(&self) -> [F192; 4] {
        debug_assert_eq!(self.logical_rows, 1);
        std::array::from_fn(|child| self.blocks[0][child].lane(0))
    }
}

/// The result of a batched grand-product proof: the three leaf evaluations, all reduced to one shared point.
pub struct ProductTriple {
    pub point: Vec<F192>,
    pub values: [F192; 3],
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
/// The first two trees share a product by construction, as the bus's two sides do.
/// One root is sent for both, then the third, so no verifier can be handed an unbalanced pair.
#[cfg(test)]
pub(crate) fn prove_product_triple(leaves: [Vec<F192>; 3], ps: &mut ProverState) -> ProductTriple {
    let [push, pull, count] = leaves.map(Leaves::new);
    prove_product_triple_from(push, pull, count, None, ps)
}

/// [`prove_product_triple`] over leaves whose first radix-four layer is already built, the third
/// tree's in `C`. The proof is the same whichever field holds the third tree's values.
///
/// The `columns` are blocks of the first two trees' leaves that `push` and `pull` do not store.
pub fn prove_product_triple_from<C: TreeField>(
    push: Leaves,
    pull: Leaves,
    count: Leaves<C>,
    columns: Option<&dyn LeafColumns>,
    ps: &mut ProverState,
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
    assert_eq!(roots[0], roots[1], "the first two products must agree");
    ps.add_scalar(roots[0]);
    ps.add_scalar(roots[2]);
    let mut point = Vec::new();
    let mut values = roots;

    let mut layer = mu;
    while layer > 0 {
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
                target_feature = "avx2"
            ));
        let pairs = two_rounds.then(|| Equality::new(&point[2..]));
        // A first table's double fold carries the third round's message, so it needs three rounds.
        for tree in &mut trees {
            if let (Some(first), Some(pairs)) = (tree.first.as_ref().filter(|_| round_count >= 3), &pairs) {
                tree.pending = Some(first.polynomial(&tree.ones, pairs));
            }
        }
        // Per tree, the first challenge of a pair of rounds whose double fold is still to come.
        let mut held: [Option<F192>; 3] = [None; 3];
        let mut messages = if round_count > 0 {
            std::array::from_fn(|s| match trees[s].pending {
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
                // A fold whose table has two more rounds before the last weighs their polynomial.
                // `two_rounds` asks for four workers, and from four on the halved fold traffic pays for the products.
                let next = (two_rounds && round + 3 < round_count)
                    .then(|| (Equality::new(&point[round + 3..]), point[round + 2]));
                messages = std::array::from_fn(|s| {
                    let tree = &mut trees[s];
                    if let Some(polynomial) = tree.pending.take() {
                        held[s] = Some(challenge);
                        return bivariate_messages(polynomial, F192::ZERO, challenge)[1];
                    }
                    // Trees with column runs fold singly until the columns are folded in.
                    let early = columns.is_some() && s < 2 && round < 2;
                    let next = next.as_ref().filter(|_| !early).map(|(w, eta)| (w, *eta));
                    if let Some(first) = held[s].take() {
                        tree.fold_blocks(first, Some(challenge), Some(&equality), next, false)
                    } else if round == 0 && polynomial.is_some() && s < 2 {
                        tree.fold_blocks(challenge, None, Some(&equality), None, true)
                    } else {
                        tree.fold_blocks(challenge, None, Some(&equality), next, false)
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

    ProductTriple { point, values }
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
                    sum + equality.lo[row] * values[0] * values[1] * values[2] * values[3]
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

    /// Distinct nonzero base-field values.
    fn base_values(len: usize, seed: u64) -> Vec<F64> {
        (0..len as u64)
            .map(|i| F64((i + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed.wrapping_mul(0xD6E8_FEB8_6659_FD93) | 1))
            .collect()
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
                // The bus's two sides balance: the same leaves in another order.
                let pull: Vec<F192> = push.iter().rev().copied().collect();
                let count = base_values(count_len, mu as u64);
                let mut dense: Vec<F192> = count.iter().map(|&k| F192::from(k)).collect();
                let mut extension_ps = ProverState::from_label(b"base-field-gkr-test");
                let extension = prove_product_triple([push.clone(), pull.clone(), dense.clone()], &mut extension_ps);
                let mut base_ps = ProverState::from_label(b"base-field-gkr-test");
                let base = prove_product_triple_from(
                    Leaves::new(push),
                    Leaves::new(pull),
                    Leaves::<F64>::new(count),
                    None,
                    &mut base_ps,
                );
                dense.resize(1 << mu, F192::ONE);
                assert_eq!(
                    base.values[2],
                    mle_eval_e(&dense, &base.point),
                    "mu={mu}, len={count_len}"
                );
                assert_eq!(base.point, extension.point, "mu={mu}, len={count_len}");
                assert_eq!(base.values, extension.values, "mu={mu}, len={count_len}");
                assert_eq!(base_ps.into_proof().stream, extension_ps.into_proof().stream, "mu={mu}");
            }
        }
    }

    fn product(lane: &[F192]) -> F192 {
        lane.iter().product()
    }

    /// Rescale lane 1's last leaf so its product matches lane 0's, as the bus's two sides do.
    fn balance(leaves: &mut [Vec<F192>; 3]) {
        let last = leaves[1].len() - 1;
        leaves[1][last] = F192::ONE;
        leaves[1][last] = product(&leaves[0]) * product(&leaves[1]).inv();
    }

    #[test]
    fn radix_four_claims_are_true_at_even_and_odd_depths() {
        for mu in 0..=12 {
            let mut leaves: [Vec<F192>; 3] = [0, 1, 2].map(|lane| {
                (0..1usize << mu)
                    .map(|row| F192::new((1 + row + lane * 100_003) as u64, row as u64, lane as u64))
                    .collect()
            });
            balance(&mut leaves);
            let mut ps = ProverState::from_label(b"radix-four-gkr-test");
            let proved = prove_product_triple(leaves.each_ref().map(|l| l.to_vec()), &mut ps);
            let stream = ps.into_proof().stream;
            assert_eq!(stream[..2], [product(&leaves[0]), product(&leaves[2])]);
            for lane in 0..3 {
                assert_eq!(proved.values[lane], mle_eval_e(&leaves[lane], &proved.point));
            }
        }
    }

    #[test]
    fn implicit_identity_suffix_matches_dense_padding() {
        for mu in 3..=10 {
            let lengths = [(1usize << mu) - 3, (1usize << (mu - 1)) + 1, (1usize << (mu - 2)) + 3];
            let mut leaves: [Vec<F192>; 3] = std::array::from_fn(|lane| {
                (0..lengths[lane])
                    .map(|row| F192::new((3 + row + lane * 10_007) as u64, row as u64, lane as u64))
                    .collect()
            });
            balance(&mut leaves);
            let dense = leaves.each_ref().map(|lane| {
                let mut padded = lane.clone();
                padded.resize(1 << mu, F192::ONE);
                padded
            });
            let mut sparse_ps = ProverState::from_label(b"sparse-radix-four-gkr-test");
            let proved = prove_product_triple(leaves.each_ref().map(|l| l.to_vec()), &mut sparse_ps);
            for lane in 0..3 {
                assert_eq!(proved.values[lane], mle_eval_e(&dense[lane], &proved.point));
            }
            let proof = sparse_ps.into_proof();
            assert_eq!(proof.stream[..2], [product(&dense[0]), product(&dense[2])]);
            let mut dense_ps = ProverState::from_label(b"sparse-radix-four-gkr-test");
            let dense_proved = prove_product_triple(dense.each_ref().map(|l| l.to_vec()), &mut dense_ps);
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
            let mut trees: [Vec<F192>; 3] = std::array::from_fn(|lane| {
                (0..len)
                    .map(|row| {
                        if runs.iter().any(|run| run.contains(&row)) {
                            F192::ONE
                        } else {
                            F192::new((3 + row + lane * 10_007) as u64, row as u64, lane as u64)
                        }
                    })
                    .collect()
            });
            balance(&mut trees);
            let tree = |lane: usize| trees[lane].clone();
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
            let proved = prove_product_triple_from(sparse(0), sparse(1), Leaves::new(tree(2)), None, &mut sparse_ps);
            let mut dense_ps = ProverState::from_label(b"interior-identity-gkr-test");
            let dense = prove_product_triple([tree(0), tree(1), tree(2)], &mut dense_ps);
            assert_eq!(dense.point, proved.point, "mu={mu}");
            assert_eq!(dense.values, proved.values, "mu={mu}");
            assert_eq!(dense_ps.into_proof().stream, sparse_ps.into_proof().stream, "mu={mu}");
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
            let mut trees: [Vec<F192>; 3] = std::array::from_fn(|side| {
                (0..if short_pull && side == 1 { 13 } else { len })
                    .map(|i| F192::new((i + 73 * side + 1) as u64, (i * i + 1) as u64, (i * 17 + side) as u64))
                    .collect()
            });
            super::tests::balance(&mut trees);
            let tree = |side: usize| trees[side].clone();
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
            let proof = prove_product_triple_from(sparse(0), sparse(1), Leaves::new(tree(2)), Some(&columns), &mut ps);
            let mut dense_ps = ProverState::from_label(b"two-round-columns");
            let dense = prove_product_triple([tree(0), tree(1), tree(2)], &mut dense_ps);
            assert_eq!(proof.point, dense.point, "mu={mu}");
            assert_eq!(proof.values, dense.values, "mu={mu}");
            assert_eq!(ps.into_proof().stream, dense_ps.into_proof().stream, "mu={mu}");
        }
    }
}
