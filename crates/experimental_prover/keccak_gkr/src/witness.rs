//! The witness: every round's state before `chi`, sliced by bit position.
//!
//! A word holds one position of 64 instances, and a block is 512 instances: eight such words
//! per position, laid out position after position. Only every `depth`-th state is committed,
//! as lanes.

use std::marker::PhantomData;

use parallel::SendPtr;
use primitives::field::F64;
use primitives::hash::positions::linear;
use primitives::hash::{RC, ROUNDS, STATE_LANES};
use primitives::stream::Stream;
use verifier::keccak::{BITS, along};

/// Instances a block holds, and the words one position of a block takes.
pub const BLOCK: usize = 512;
pub(crate) const BLOCK_WORDS: usize = 8;
/// Words one block of one state takes.
pub(crate) const BLOCK_LEN: usize = BITS * BLOCK_WORDS;

/// The eight words of position `j` of a sliced block.
#[inline(always)]
pub(crate) fn words(block: &[u64], j: usize) -> &[u64; BLOCK_WORDS] {
    block[j * BLOCK_WORDS..][..BLOCK_WORDS].try_into().expect("eight words")
}

/// Built only by [`Shape::new`], whose bounds the witness's raw writes rely on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// `log2` of the number of instances of one table, at least 9.
    pub(crate) n_log: usize,
    /// Rounds between two committed states. Divides 24.
    pub(crate) depth: usize,
    /// `log2` of the number of tables. A table is its own batch of `2^n_log` instances with its own
    /// padding instance; the committed block holds them side by side, slot after slot.
    pub(crate) table_bits: usize,
}

impl Shape {
    /// One table of `n_instances`.
    pub const fn new(n_instances: usize, depth: usize) -> Self {
        Self::with_tables(n_instances, depth, 0)
    }

    /// `2^table_bits` tables of `n_instances` each, which the reduction proves together.
    pub const fn with_tables(n_instances: usize, depth: usize, table_bits: usize) -> Self {
        assert!(ROUNDS.is_multiple_of(depth), "depth must divide the round count");
        assert!(
            table_bits == 0 || (ROUNDS / depth).is_power_of_two(),
            "several tables need a power of two of round segments"
        );
        assert!(table_bits <= 2, "at most four tables");
        let n = if n_instances < BLOCK { BLOCK } else { n_instances };
        let n_log = n.next_power_of_two().ilog2() as usize;
        let shape = Self {
            n_log,
            depth,
            table_bits,
        };
        // The bound the raw writes rely on. Every size this crate derives is at most the
        // `2^stack_log` words of the committed block (`slots <= 2^slot_log`), so bounding that
        // one bounds them all; without it `slots * n` and `1 << stack_log` wrap, and a buffer
        // too small for the writes passes the checks that guard them.
        assert!(
            shape.stack_log() < usize::BITS as usize,
            "too many instances for one batch"
        );
        shape
    }

    pub const fn n_log(&self) -> usize {
        self.n_log
    }
    pub const fn depth(&self) -> usize {
        self.depth
    }
    pub const fn n(&self) -> usize {
        1 << self.n_log
    }
    pub const fn n_blocks(&self) -> usize {
        self.n() / BLOCK
    }
    pub const fn table_bits(&self) -> usize {
        self.table_bits
    }
    /// The number of tables.
    pub const fn tables(&self) -> usize {
        1 << self.table_bits
    }
    /// The runs of `depth` rounds a permutation splits into.
    pub const fn round_segments(&self) -> usize {
        ROUNDS / self.depth
    }
    /// `log2` of the round segments, rounded up.
    pub const fn round_seg_log(&self) -> usize {
        self.round_segments().next_power_of_two().ilog2() as usize
    }
    /// The segments a layer walks: each run of rounds of each table. Segment `m + round_segments * t`
    /// is run `m` of table `t`.
    pub const fn segments(&self) -> usize {
        self.round_segments() << self.table_bits
    }
    /// Variables of a segment index: the run, then the table.
    pub const fn seg_log(&self) -> usize {
        self.round_seg_log() + self.table_bits
    }
    /// Variables of a pack index: the instance index without its low four bits.
    pub(crate) const fn pack_vars(&self) -> usize {
        self.n_log - 4
    }
    /// Variables the layer sumcheck binds: a pack, then a segment.
    pub(crate) const fn y_vars(&self) -> usize {
        self.pack_vars() + self.seg_log()
    }
    /// Committed lanes per instance: the state each run starts from.
    pub const fn slots(&self) -> usize {
        STATE_LANES * self.round_segments()
    }
    pub const fn slot_log(&self) -> usize {
        self.slots().next_power_of_two().ilog2() as usize
    }
    /// `log2` of the committed stack, in words: the instance, the table, then the slot.
    pub const fn stack_log(&self) -> usize {
        self.n_log + self.table_bits + self.slot_log()
    }
    /// The round whose output segment `segment` claims after `step` layers.
    pub fn round(&self, segment: usize, step: usize) -> usize {
        self.depth * (segment % self.round_segments() + 1) - step
    }
    /// The table segment `segment` belongs to.
    pub fn table_of(&self, segment: usize) -> usize {
        segment / self.round_segments()
    }
}

/// A run of blocks, round after round: block `first + i` of round `r` (the input state being
/// round 0) is at word `(r * blocks + i) * BLOCK_LEN`.
pub struct Extent {
    first: usize,
    blocks: usize,
    words: Vec<u64>,
}

/// Where one block of an [`Extent`] goes.
#[derive(Clone, Copy)]
pub struct Slot {
    base: SendPtr<u64>,
    stride: usize,
}

impl Extent {
    /// An extent for blocks `first .. first + blocks`, none of them written yet. A block is
    /// read only through a [`Witness`], whose builder vouches for the ones [`block`] wrote.
    pub fn new(first: usize, blocks: usize) -> Self {
        // A wrapped length would leave `slot`'s offsets past the buffer.
        assert!(first.checked_add(blocks).is_some(), "an extent's blocks are indexable");
        let len = blocks
            .checked_mul((ROUNDS + 1) * BLOCK_LEN)
            .expect("an extent's words fit in memory");
        // SAFETY: nothing reads a block that `Witness::from_extents` was not told is written.
        let words = unsafe { primitives::uninit_vec(len) };
        Self { first, blocks, words }
    }

    pub fn end(&self) -> usize {
        self.first + self.blocks
    }

    /// The slot of block `g`, which must be one of this extent's. Slots of distinct blocks
    /// are disjoint.
    pub fn slot(&mut self, g: usize) -> Slot {
        assert!((self.first..self.end()).contains(&g));
        Slot {
            // SAFETY: the block's first round lies in the buffer.
            base: SendPtr(unsafe { self.words.as_mut_ptr().add((g - self.first) * BLOCK_LEN) }),
            stride: self.blocks * BLOCK_LEN,
        }
    }

    fn round(&self, round: usize, g: usize) -> &[u64] {
        &self.words[(round * self.blocks + g - self.first) * BLOCK_LEN..][..BLOCK_LEN]
    }
}

/// The witness of every table: a [`TableWitness`] each.
pub struct Witness {
    pub shape: Shape,
    tables: Vec<TableWitness>,
}

/// The witness of one table's batch.
pub struct TableWitness {
    pub shape: Shape,
    /// The first `written` blocks of the batch, in order. The others are all `padding`.
    extents: Vec<Extent>,
    written: usize,
    padding: Option<Extent>,
}

/// Transpose a 64 by 64 bit matrix: bit `c` of row `r` becomes bit `r` of row `c`.
#[inline]
pub(crate) fn transpose64(a: &mut [u64; 64]) {
    cfg_select! {
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512vbmi",
            target_feature = "gfni"
        ) => {
            // SAFETY: the features are enabled at compile time.
            unsafe { x86::transpose64(a) }
        }
        all(target_arch = "x86_64", target_feature = "avx2") => {
            // SAFETY: the feature is enabled at compile time.
            unsafe { avx2::transpose64(a) }
        }
        target_arch = "aarch64" => {
            // SAFETY: aarch64 always has NEON.
            unsafe { neon::transpose64(a) }
        }
        _ => transpose64_portable(a),
    }
}

pub fn transpose64_portable(a: &mut [u64; 64]) {
    let mut j = 32;
    let mut mask = 0x0000_0000_ffff_ffffu64;
    while j != 0 {
        for k in (0..64).filter(|k| k & j == 0) {
            let t = ((a[k] >> j) ^ a[k + j]) & mask;
            a[k + j] ^= t;
            a[k] ^= t << j;
        }
        j >>= 1;
        mask ^= mask << j;
    }
}

type Word = [u64; BLOCK_WORDS];

fn xor(a: Word, b: Word) -> Word {
    std::array::from_fn(|i| a[i] ^ b[i])
}

/// The input a table's padding instances hold, all zero but for a table that says otherwise.
pub const ZERO_PAD: [u64; STATE_LANES] = [0; STATE_LANES];

/// Run one block of instances, padded with `pad` inputs, and write its input state and the
/// state each round applies `chi` to, all sliced, into `slot` if there is one.
///
/// `commit(slot, w, lanes)` receives lane `slot % 25` of the state after `depth * (slot / 25)`
/// rounds, for instances `64 * w ..` of the block: every state a segment starts from, the output
/// not being committed. With no slot, the run ends at the last committed state.
///
/// # Safety
/// `slot`, if any, is a slot of a live [`Extent`] that no other call is writing.
pub unsafe fn block(
    depth: usize,
    pad: &[u64; STATE_LANES],
    inputs: &[[u64; STATE_LANES]],
    slot: Option<Slot>,
    mut commit: impl FnMut(usize, usize, &[u64; 64]),
) {
    assert!(inputs.len() <= BLOCK);
    let mut state = vec![[0u64; BLOCK_WORDS]; BITS];
    let mut u = vec![[0u64; BLOCK_WORDS]; BITS];
    let mut rows = [0u64; 64];
    for lane in 0..STATE_LANES {
        for w in 0..BLOCK_WORDS {
            for (i, row) in rows.iter_mut().enumerate() {
                *row = inputs.get(64 * w + i).map_or(pad[lane], |x| x[lane]);
            }
            transpose64(&mut rows);
            for z in 0..64 {
                state[64 * lane + z][w] = rows[z];
            }
        }
    }
    let mut lanes = |m: usize, state: &[Word]| {
        let mut rows = [0u64; 64];
        for lane in 0..STATE_LANES {
            for w in 0..BLOCK_WORDS {
                for z in 0..64 {
                    rows[z] = state[64 * lane + z][w];
                }
                transpose64(&mut rows);
                commit(STATE_LANES * m + lane, w, &rows);
            }
        }
    };
    // A round's state is next read by the layer that proves it, long after this pass.
    let stream = Stream::new();
    let store = |round: usize, words: &[Word]| {
        if let Some(slot) = slot {
            // SAFETY: the slot holds a block of every round.
            unsafe { stream.copy_uninit(slot.base.add(round * slot.stride), words.as_flattened()) };
        }
    };
    let last = if slot.is_some() { ROUNDS } else { ROUNDS - depth };
    store(0, &state);
    lanes(0, &state);
    for round in 1..=last {
        linear(&state, &mut u, xor);
        store(round, &u);
        for lane in 0..STATE_LANES {
            let (a, b, c) = (64 * lane, along(64 * lane, 1), along(64 * lane, 2));
            // A lane's 64 positions are 512 consecutive words in each table: `chi` word by word
            // over them, which vectorizes with plain loads (by position, it took gathers).
            let flat = |at: usize| -> &[u64] { u[at..at + 64].as_flattened() };
            let (ua, ub, uc) = (flat(a), flat(b), flat(c));
            for (k, s) in state[a..a + 64].as_flattened_mut().iter_mut().enumerate() {
                *s = ua[k] ^ (!ub[k] & uc[k]);
            }
        }
        for z in (0..64).filter(|z| RC[round - 1] >> z & 1 == 1) {
            state[z] = state[z].map(|w| !w);
        }
        if round.is_multiple_of(depth) && round < ROUNDS {
            lanes(round / depth, &state);
        }
    }
}

/// [`block`]'s committed lanes, written into `slot` if there is one: slot `s` at `s * BLOCK ..`,
/// a word per instance.
///
/// # Safety
/// As [`block`].
unsafe fn block_lanes_into(
    depth: usize,
    pad: &[u64; STATE_LANES],
    inputs: &[[u64; STATE_LANES]],
    slot: Option<Slot>,
) -> Vec<u64> {
    // SAFETY: `block` writes every instance of every slot.
    let mut lanes = unsafe { primitives::uninit_vec(STATE_LANES * (ROUNDS / depth) * BLOCK) };
    // SAFETY: forwarded to the caller.
    unsafe {
        block(depth, pad, inputs, slot, |slot, w, words| {
            lanes[slot * BLOCK + 64 * w..][..64].copy_from_slice(words)
        })
    };
    lanes
}

/// One block's committed lanes alone: slot `s` at `s * BLOCK ..`, a word per instance.
///
/// Its rounds are not kept: the reduction rebuilds them from the committed column.
pub fn block_lanes(depth: usize, pad: &[u64; STATE_LANES], inputs: &[[u64; STATE_LANES]]) -> Vec<u64> {
    // SAFETY: no slot is written.
    unsafe { block_lanes_into(depth, pad, inputs, None) }
}

/// The padding block of a batch whose first `written` blocks hold its inputs, if blocks follow
/// them: its one-block extent and its lanes.
fn padding(shape: Shape, pad: &[u64; STATE_LANES], written: usize) -> Option<(Extent, Vec<u64>)> {
    (written < shape.n_blocks()).then(|| {
        let mut extent = Extent::new(written, 1);
        // SAFETY: the extent's one slot, written here alone.
        let lanes = unsafe { block_lanes_into(shape.depth, pad, &[], Some(extent.slot(written))) };
        (extent, lanes)
    })
}

/// The committed column, written a block at a time: slot `s` of instance `i` of table `t` at word
/// `(s * tables + t) * n + i`.
#[derive(Clone, Copy)]
struct Column<'a> {
    words: SendPtr<u64>,
    n: usize,
    slots: usize,
    tables: usize,
    _committed: PhantomData<&'a mut [F64]>,
}

impl<'a> Column<'a> {
    /// The writer for `committed`, which holds at least the states' `slots * n * tables` words and at
    /// most the whole block: anything it has past the last slot is zeroed, so a caller whose
    /// spare slots carry other columns passes the states' prefix alone.
    fn new(shape: Shape, committed: &'a mut [F64]) -> Self {
        let (n, slots, tables) = (shape.n(), shape.slots(), shape.tables());
        assert!(committed.len() >= slots * n * tables && committed.len() <= 1 << shape.stack_log());
        committed[slots * n * tables..].fill(F64(0));
        Self {
            words: SendPtr(committed.as_mut_ptr().cast()),
            n,
            slots,
            tables,
            _committed: PhantomData,
        }
    }

    /// Instances `at ..` of block `g` of table `table` in slot `slot`.
    ///
    /// # Safety
    /// The words lie in the column, and no other call writes them.
    unsafe fn write(self, stream: &Stream, table: usize, slot: usize, g: usize, at: usize, words: &[u64]) {
        // SAFETY: forwarded to the caller, the column outliving `self`.
        unsafe {
            stream.copy_uninit(
                self.words.add((slot * self.tables + table) * self.n + g * BLOCK + at),
                words,
            )
        }
    }

    /// Block `g` of table `table` in every slot, its lanes as [`block_lanes`] keeps them.
    ///
    /// # Safety
    /// Block `g` is one of the table's, and no other call writes it.
    unsafe fn write_block(self, table: usize, g: usize, lanes: &[u64]) {
        let stream = Stream::new();
        for slot in 0..self.slots {
            // SAFETY: forwarded to the caller.
            unsafe { self.write(&stream, table, slot, g, 0, &lanes[slot * BLOCK..][..BLOCK]) };
        }
    }
}

/// What a table holds: the inputs of its instances, and the input its padding instances hold.
pub struct TableInputs<'a> {
    pub inputs: &'a [[u64; STATE_LANES]],
    pub pad: [u64; STATE_LANES],
}

/// What `generate_tables` and `from_committed` keep of one table while its blocks run.
struct Run {
    written: usize,
    padding: Option<(Extent, Vec<u64>)>,
    extent: Extent,
    slots: Vec<Slot>,
}

impl Run {
    fn new(shape: Shape, pad: &[u64; STATE_LANES], written: usize) -> Self {
        let padding = padding(shape, pad, written);
        let mut extent = Extent::new(0, written);
        let slots = (0..written).map(|g| extent.slot(g)).collect();
        Self {
            written,
            padding,
            extent,
            slots,
        }
    }

    /// The table's witness, its blocks below `written` having run into the extent.
    ///
    /// # Safety
    /// Every block below `written` was run into the extent, and the padding block into its own.
    unsafe fn into_witness(self, shape: Shape) -> TableWitness {
        // SAFETY: forwarded to the caller.
        unsafe {
            TableWitness::from_extents(
                shape,
                vec![self.extent],
                self.written,
                self.padding.map(|(extent, _)| extent),
            )
        }
    }
}

impl Witness {
    /// One table of `inputs`, padded with all-zero ones: [`Witness::generate_tables`] of a
    /// one-table shape.
    pub fn generate(shape: Shape, inputs: &[[u64; STATE_LANES]], committed: &mut [F64]) -> Self {
        assert_eq!(shape.tables(), 1);
        Self::generate_tables(shape, &[TableInputs { inputs, pad: ZERO_PAD }], committed)
    }

    /// Run every instance of every table, padding each batch with its table's pad input, and write
    /// the committed lanes into `committed`: lane `l` of the state after `depth * m` rounds, for `m`
    /// below the run count, of instance `i` of table `t`, is word `(25 * m + l) * tables * n + t * n + i`.
    /// Whatever `committed` has past the last slot is zeroed, so a caller whose spare slots carry
    /// other columns (§cpu) passes the states' prefix alone. The blocks after a table's last input
    /// are run once, as its padding block.
    pub fn generate_tables(shape: Shape, tables: &[TableInputs], committed: &mut [F64]) -> Self {
        assert_eq!(tables.len(), shape.tables());
        let column = Column::new(shape, committed);
        let nb = shape.n_blocks();
        let runs: Vec<Run> = tables
            .iter()
            .map(|table| {
                assert!(table.inputs.len() <= shape.n());
                Run::new(shape, &table.pad, table.inputs.len().div_ceil(BLOCK))
            })
            .collect();
        parallel::for_each(nb * tables.len(), |task| {
            let (t, g) = (task / nb, task % nb);
            let (table, run) = (&tables[t], &runs[t]);
            // SAFETY: block `g` of table `t` owns its slot and its instances' words of every committed lane.
            unsafe {
                if g < run.written {
                    let stream = Stream::new();
                    let inputs = &table.inputs[g * BLOCK..((g + 1) * BLOCK).min(table.inputs.len())];
                    block(shape.depth, &table.pad, inputs, Some(run.slots[g]), |slot, w, lanes| {
                        column.write(&stream, t, slot, g, 64 * w, lanes)
                    });
                } else {
                    column.write_block(t, g, &run.padding.as_ref().expect("a padding block").1);
                }
            }
        });
        let tables = runs
            .into_iter()
            // SAFETY: the blocks below `written` were run into the extent just above.
            .map(|run| unsafe { run.into_witness(shape) })
            .collect();
        Self { shape, tables }
    }

    /// Write the committed lanes of every block into `committed`, laid out as `generate_tables` does.
    ///
    /// - Block `g` of table `t` has the lanes `lanes[t][g]`.
    /// - The blocks past them are the table's padding block.
    pub fn write_committed(shape: Shape, pads: &[[u64; STATE_LANES]], lanes: &[&[Vec<u64>]], committed: &mut [F64]) {
        assert!(lanes.len() == shape.tables() && pads.len() == shape.tables());
        let column = Column::new(shape, committed);
        let nb = shape.n_blocks();
        let paddings: Vec<Option<Vec<u64>>> = lanes
            .iter()
            .zip(pads)
            .map(|(lanes, pad)| (lanes.len() < nb).then(|| block_lanes(shape.depth, pad, &[])))
            .collect();
        parallel::for_each(nb * lanes.len(), |task| {
            let (t, g) = (task / nb, task % nb);
            let lanes = lanes[t]
                .get(g)
                .unwrap_or_else(|| paddings[t].as_ref().expect("a padding block"));
            // SAFETY: block `g` of table `t` owns its instances' words of every committed lane.
            unsafe { column.write_block(t, g, lanes) };
        });
    }

    /// The witness of the first `instances[t]` instances of each table, rebuilt from their committed inputs.
    ///
    /// - `committed` is laid out as `generate_tables` writes it.
    /// - The blocks past a table's last input are its padding block.
    pub fn from_committed(shape: Shape, instances: &[usize], pads: &[[u64; STATE_LANES]], committed: &[F64]) -> Self {
        let (n, count) = (shape.n(), shape.tables());
        assert!(instances.len() == count && pads.len() == count);
        assert!(committed.len() >= STATE_LANES * n * count);
        let runs: Vec<Run> = (0..count)
            .map(|t| {
                assert!(instances[t] <= n);
                Run::new(shape, &pads[t], instances[t].div_ceil(BLOCK))
            })
            .collect();
        let starts: Vec<usize> = runs
            .iter()
            .scan(0, |sum, run| {
                *sum += run.written;
                Some(*sum - run.written)
            })
            .collect();
        let total: usize = runs.iter().map(|run| run.written).sum();
        parallel::for_each(total, |task| {
            let t = starts.partition_point(|&start| start <= task) - 1;
            let g = task - starts[t];
            // Slot `l` holds lane `l` of every input, a word per instance. A step past the last
            // input reads the input the block pads with.
            let inputs: Vec<[u64; STATE_LANES]> = (g * BLOCK..(g + 1) * BLOCK)
                .map(|i| std::array::from_fn(|lane| committed[(lane * count + t) * n + i].0))
                .collect();
            // SAFETY: block `g` of table `t` owns its slot.
            unsafe { block(shape.depth, &pads[t], &inputs, Some(runs[t].slots[g]), |_, _, _| {}) };
        });
        let tables = runs
            .into_iter()
            // SAFETY: the blocks below `written` were run into the extent just above, and the padding
            // block into its own.
            .map(|run| unsafe { run.into_witness(shape) })
            .collect();
        Self { shape, tables }
    }

    /// One table's witness, built from [`TableWitness::from_extents`]: the whole witness of a one-table shape.
    ///
    /// # Safety
    /// As [`TableWitness::from_extents`].
    #[cfg(test)]
    pub(crate) unsafe fn from_extents(
        shape: Shape,
        extents: Vec<Extent>,
        written: usize,
        padding: Option<Extent>,
    ) -> Self {
        assert_eq!(shape.tables(), 1);
        Self {
            shape,
            // SAFETY: forwarded to the caller.
            tables: vec![unsafe { TableWitness::from_extents(shape, extents, written, padding) }],
        }
    }

    pub(crate) fn table(&self, table: usize) -> &TableWitness {
        &self.tables[table]
    }
}

impl TableWitness {
    /// The witness of extents whose first `written` blocks [`block`] wrote: consecutive from
    /// block 0, the batch's remaining blocks all being `padding`, a one-block extent of no
    /// instance.
    ///
    /// # Safety
    /// [`block`] has written blocks `0 .. written` of `extents`, and `padding`'s one block.
    pub(crate) unsafe fn from_extents(
        shape: Shape,
        extents: Vec<Extent>,
        written: usize,
        padding: Option<Extent>,
    ) -> Self {
        let mut end = 0;
        for extent in &extents {
            assert_eq!(extent.first, end, "extents are consecutive");
            end = extent.end();
        }
        assert!(written <= end && written <= shape.n_blocks());
        assert!(written == shape.n_blocks() || padding.as_ref().is_some_and(|p| p.blocks == 1));
        Self {
            shape,
            extents,
            written,
            padding,
        }
    }

    /// Blocks `written()..` of the batch are all the padding block, every instance of which is
    /// the all-zero input.
    pub fn written(&self) -> usize {
        self.written
    }

    /// The blocks a segment's tasks walk: those written, then the padding block once, standing
    /// for every block after them.
    pub(crate) fn held(&self) -> usize {
        self.written + usize::from(self.written < self.shape.n_blocks())
    }

    /// Block `g` of the state round `round` applies `chi` to, rounds counted from 1, or of the
    /// input state for `round` 0, sliced.
    pub fn layer(&self, round: usize, g: usize) -> &[u64] {
        if g >= self.written {
            let padding = self.padding.as_ref().expect("a block of the batch");
            return padding.round(round, padding.first);
        }
        let extent = self.extents.iter().find(|e| g < e.end()).expect("a written block");
        extent.round(round, g)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
mod x86 {
    use core::arch::x86_64::*;

    use primitives::bits::bit_transpose_64bytes_m512;

    /// The register of group `g` after the word transpose: groups 0, 1, 4, 5, 2, 3, 6, 7.
    const ORDER: [usize; 8] = [0, 1, 4, 5, 2, 3, 6, 7];

    /// Byte `8 * c + g` of eight output rows: byte `c` of group `g`'s word.
    const SCATTER: [u8; 64] = {
        let mut idx = [0u8; 64];
        let mut i = 0;
        while i < 64 {
            idx[i] = (8 * ORDER[i % 8] + i / 8) as u8;
            i += 1;
        }
        idx
    };

    /// An 8 by 8 grid of 8 by 8 blocks: a block's bytes gathered into a word, its bits
    /// transposed there by an affine map, then the grid transposed, words across registers and
    /// bytes within them.
    #[inline(always)]
    pub unsafe fn transpose64(a: &mut [u64; 64]) {
        unsafe {
            // Word `b` of register `g`: the transposed block of rows `8 * g ..` and byte `b`.
            let w: [__m512i; 8] =
                std::array::from_fn(|g| bit_transpose_64bytes_m512(_mm512_loadu_si512(a.as_ptr().add(8 * g).cast())));
            // Word `b` of every register into register `b`: halves, quarters, then words.
            let mut r = w;
            for g in 0..4 {
                r[g] = _mm512_shuffle_i64x2::<0b01_00_01_00>(w[g], w[g + 4]);
                r[g + 4] = _mm512_shuffle_i64x2::<0b11_10_11_10>(w[g], w[g + 4]);
            }
            let mut q = r;
            for g in [0, 1, 4, 5] {
                q[g] = _mm512_shuffle_i64x2::<0b10_00_10_00>(r[g], r[g + 2]);
                q[g + 2] = _mm512_shuffle_i64x2::<0b11_01_11_01>(r[g], r[g + 2]);
            }
            let scatter = _mm512_loadu_si512(SCATTER.as_ptr().cast());
            for b in (0..8).step_by(2) {
                let (lo, hi) = (
                    _mm512_unpacklo_epi64(q[b], q[b + 1]),
                    _mm512_unpackhi_epi64(q[b], q[b + 1]),
                );
                _mm512_storeu_si512(a.as_mut_ptr().add(8 * b).cast(), _mm512_permutexvar_epi8(scatter, lo));
                _mm512_storeu_si512(
                    a.as_mut_ptr().add(8 * b + 8).cast(),
                    _mm512_permutexvar_epi8(scatter, hi),
                );
            }
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", target_feature = "avx512vbmi", target_feature = "gfni"))
))]
mod avx2 {
    use core::arch::x86_64::*;

    /// A 4 by 4 transpose of words.
    #[inline(always)]
    unsafe fn transpose4([a, b, c, d]: [__m256i; 4]) -> [__m256i; 4] {
        unsafe {
            let (ab0, ab1) = (_mm256_unpacklo_epi64(a, b), _mm256_unpackhi_epi64(a, b));
            let (cd0, cd1) = (_mm256_unpacklo_epi64(c, d), _mm256_unpackhi_epi64(c, d));
            [
                _mm256_permute2x128_si256::<0x20>(ab0, cd0),
                _mm256_permute2x128_si256::<0x20>(ab1, cd1),
                _mm256_permute2x128_si256::<0x31>(ab0, cd0),
                _mm256_permute2x128_si256::<0x31>(ab1, cd1),
            ]
        }
    }

    /// The portable form's swaps, which commute, with rows `i + 16 l` in lane `l` of register
    /// `i`: those of `j` below 16 are across registers, a masked swap by shifts, and those of 16
    /// and 32 within one, a permutation of its 16- and 32-bit fields.
    #[inline(always)]
    pub unsafe fn transpose64(a: &mut [u64; 64]) {
        unsafe {
            let mut r = [_mm256_setzero_si256(); 16];
            for m in 0..4 {
                let rows = std::array::from_fn(|l| _mm256_loadu_si256(a.as_ptr().add(4 * m + 16 * l).cast()));
                r[4 * m..][..4].copy_from_slice(&transpose4(rows));
            }
            macro_rules! swap {
                ($j:expr, $mask:expr) => {
                    let mask = _mm256_set1_epi64x($mask);
                    for i in (0..16).filter(|i| i & $j == 0) {
                        let t = _mm256_and_si256(
                            _mm256_xor_si256(_mm256_srli_epi64::<$j>(r[i]), r[i + $j]),
                            mask,
                        );
                        r[i + $j] = _mm256_xor_si256(r[i + $j], t);
                        r[i] = _mm256_xor_si256(r[i], _mm256_slli_epi64::<$j>(t));
                    }
                };
            }
            swap!(8, 0x00ff_00ff_00ff_00ff);
            swap!(4, 0x0f0f_0f0f_0f0f_0f0f);
            swap!(2, 0x3333_3333_3333_3333);
            swap!(1, 0x5555_5555_5555_5555);
            // Rows `k` and `k + 32` trade the high half of the one for the low half of the other,
            // and rows `k` and `k + 16` the same of each 32-bit field.
            let halves = _mm256_setr_epi32(0, 4, 2, 6, 1, 5, 3, 7);
            let fields =
                _mm256_broadcastsi128_si256(_mm_setr_epi8(0, 1, 8, 9, 4, 5, 12, 13, 2, 3, 10, 11, 6, 7, 14, 15));
            for x in &mut r {
                *x = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*x, halves), fields);
            }
            for m in 0..4 {
                let rows = transpose4(r[4 * m..][..4].try_into().unwrap());
                for (l, row) in rows.into_iter().enumerate() {
                    _mm256_storeu_si256(a.as_mut_ptr().add(4 * m + 16 * l).cast(), row);
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    /// The portable form's swaps, two rows to a register: rows `k` and `k + j` exchange the
    /// `j`-bit halves of each `2j`-bit field. For `j` of 8 and more that is a lane transpose of
    /// the two registers at width `j`; below, a masked swap by shifts.
    #[inline(always)]
    pub unsafe fn transpose64(a: &mut [u64; 64]) {
        unsafe {
            let mut v: [uint64x2_t; 32] = std::array::from_fn(|i| vld1q_u64(a.as_ptr().add(2 * i)));
            // Registers `i` and `i + d` hold rows `2i, 2i + 1` and `2i + 2d, 2i + 2d + 1`.
            macro_rules! lanes {
                ($d:expr, $trn1:ident, $trn2:ident, $as:ident, $back:ident) => {
                    for i in (0..32).filter(|i| i & $d == 0) {
                        let (x, y) = ($as(v[i]), $as(v[i + $d]));
                        (v[i], v[i + $d]) = ($back($trn1(x, y)), $back($trn2(x, y)));
                    }
                };
            }
            lanes!(16, vtrn1q_u32, vtrn2q_u32, vreinterpretq_u32_u64, vreinterpretq_u64_u32);
            lanes!(8, vtrn1q_u16, vtrn2q_u16, vreinterpretq_u16_u64, vreinterpretq_u64_u16);
            lanes!(4, vtrn1q_u8, vtrn2q_u8, vreinterpretq_u8_u64, vreinterpretq_u64_u8);
            macro_rules! swap {
                ($d:expr, $j:expr, $mask:expr) => {
                    let mask = vdupq_n_u64($mask);
                    for i in (0..32).filter(|i| i & $d == 0) {
                        let t = vandq_u64(veorq_u64(vshrq_n_u64::<$j>(v[i]), v[i + $d]), mask);
                        v[i + $d] = veorq_u64(v[i + $d], t);
                        v[i] = veorq_u64(v[i], vshlq_n_u64::<$j>(t));
                    }
                };
            }
            swap!(2, 4, 0x0f0f_0f0f_0f0f_0f0f);
            swap!(1, 2, 0x3333_3333_3333_3333);
            // Rows `2i` and `2i + 1` share a register: swap its halves to pair them across.
            let mask = vdupq_n_u64(0x5555_5555_5555_5555);
            for x in &mut v {
                let partner = vextq_u64::<1>(*x, *x);
                let t = vandq_u64(veorq_u64(vshrq_n_u64::<1>(*x), partner), mask);
                // Lane 0 is row `k`, taking `t << 1`; lane 1 is row `k + 1`, taking `t`.
                let t = vcombine_u64(vget_low_u64(vshlq_n_u64::<1>(t)), vget_low_u64(t));
                *x = veorq_u64(*x, t);
            }
            for (i, x) in v.iter().enumerate() {
                vst1q_u64(a.as_mut_ptr().add(2 * i), *x);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use primitives::hash::positions::iota_bit;
    use primitives::test_rng::Rng;

    #[test]
    fn transpose_matches_the_portable_one() {
        let mut rng = Rng::new(8);
        let a: [u64; 64] = std::array::from_fn(|_| rng.next_u64());
        let (mut fast, mut slow) = (a, a);
        transpose64(&mut fast);
        transpose64_portable(&mut slow);
        assert_eq!(fast, slow);
        for r in 0..64 {
            for c in 0..64 {
                assert_eq!(slow[c] >> r & 1, a[r] >> c & 1);
            }
        }
    }

    #[test]
    fn committed_states_follow_the_permutation() {
        let mut rng = Rng::new(9);
        let inputs: Vec<[u64; STATE_LANES]> = (0..1100).map(|_| std::array::from_fn(|_| rng.next_u64())).collect();
        let shape = Shape::new(inputs.len(), 8);
        let mut committed = vec![F64(1); 1 << shape.stack_log()];
        Witness::generate(shape, &inputs, &mut committed);
        let n = shape.n();
        for x in (0..n).step_by(37) {
            let state = inputs.get(x).copied().unwrap_or([0; STATE_LANES]);
            for lane in 0..STATE_LANES {
                assert_eq!(committed[lane * n + x].0, state[lane]);
            }
            // The state after 16 rounds, bit by bit.
            let mut bits: Vec<bool> = (0..BITS).map(|j| state[j / 64] >> (j % 64) & 1 == 1).collect();
            for round in 1..=16 {
                let mut u = vec![false; BITS];
                linear(&bits, &mut u, |a, b| a ^ b);
                for j in 0..BITS {
                    bits[j] = u[j] ^ (!u[along(j, 1)] & u[along(j, 2)]) ^ (j < 64 && iota_bit(round, j));
                }
            }
            for lane in 0..STATE_LANES {
                let word = (0..64).fold(0u64, |w, z| w | (bits[64 * lane + z] as u64) << z);
                assert_eq!(committed[(2 * STATE_LANES + lane) * n + x].0, word, "instance {x}");
            }
        }
        assert!(committed[shape.slots() * n..].iter().all(|w| w.0 == 0));
    }

    /// A shape whose committed block would not fit a `usize` is refused, rather than wrapping
    /// into sizes that let an undersized buffer pass the checks the raw writes rely on.
    #[test]
    #[should_panic(expected = "too many instances")]
    fn an_unrepresentable_batch_is_refused() {
        Shape::new(1usize << 63, 12);
    }
}
