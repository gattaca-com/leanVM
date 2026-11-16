//! The witness: every round's state before `chi`, sliced by bit position.
//!
//! A word holds one position of 64 instances, and a block is 512 instances: eight such words
//! per position, laid out position after position. Only every `depth`-th state is committed,
//! as lanes.

use primitives::field::F64;
use primitives::hash::positions::{iota_bit, linear};
use primitives::hash::{ROUNDS, STATE_LANES};
use verifier::keccak::{BITS, along};

use crate::kernels::xor;

/// Instances a block holds, and the words one position of a block takes.
pub const BLOCK: usize = 512;
pub(crate) const BLOCK_WORDS: usize = 8;
/// Words one block of one state takes.
pub(crate) const BLOCK_LEN: usize = BITS * BLOCK_WORDS;

/// Built only by [`Shape::with_tables`], which bounds its sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// `log2` of the number of instances of one table, at least 9.
    pub(crate) n_log: usize,
    /// Rounds between two committed states. Divides 24.
    pub(crate) depth: usize,
    /// `log2` of the number of tables. A table is its own batch of `2^n_log` instances with its own padding
    /// instance; the committed block holds them side by side, slot after slot.
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
        // Every size this crate derives is at most the `2^stack_log` words of the committed
        // block (`slots <= 2^slot_log`), so bounding that one keeps them all from wrapping.
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
    /// The segments a layer walks: each run of rounds of each table. Segment `m + round_segments * t` is run `m` of
    /// table `t`.
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

/// The witness of every table: its blocks' rounds, per table.
pub struct Witness {
    pub shape: Shape,
    pub(crate) tables: Vec<TableWitness>,
}

/// Every round's state before `chi`, sliced, block by block, for one table's batch.
pub(crate) struct TableWitness {
    /// The blocks holding inputs, then the padding block if any block follows them, which every later one equals:
    /// per block, every round's sliced state, round `r` at `r * BLOCK_LEN`, the input state being round 0.
    pub(crate) blocks: Vec<Vec<u64>>,
    /// The blocks holding inputs: every later one is the padding block, each instance of which holds the table's
    /// pad input.
    pub(crate) written: usize,
}

/// Transpose a 64 by 64 bit matrix: bit `c` of row `r` becomes bit `r` of row `c`.
pub(crate) fn transpose64(a: &mut [u64; 64]) {
    // Swap the off-diagonal `j` by `j` blocks of every `2j` by `2j` block, `j` halving: rows `i` and `i + j` of a
    // block, which the compiler vectorizes over `i`.
    const MASKS: [u64; 6] = [
        0x0000_0000_ffff_ffff,
        0x0000_ffff_0000_ffff,
        0x00ff_00ff_00ff_00ff,
        0x0f0f_0f0f_0f0f_0f0f,
        0x3333_3333_3333_3333,
        0x5555_5555_5555_5555,
    ];
    for (level, mask) in MASKS.into_iter().enumerate() {
        let j = 32 >> level;
        for block in a.chunks_exact_mut(2 * j) {
            let (low, high) = block.split_at_mut(j);
            for (a, b) in low.iter_mut().zip(high) {
                let t = ((*a >> j) ^ *b) & mask;
                *b ^= t;
                *a ^= t << j;
            }
        }
    }
}

/// One position of a block.
type Word = [u64; BLOCK_WORDS];

/// The input a table's padding instances hold, all zero but for a table that says otherwise.
pub const ZERO_PAD: [u64; STATE_LANES] = [0; STATE_LANES];

/// What a table holds: the inputs of its instances, and the input its padding instances hold.
pub struct TableInputs<'a> {
    pub inputs: &'a [[u64; STATE_LANES]],
    pub pad: [u64; STATE_LANES],
}

/// Run one block of instances, padded with `pad` inputs. Returns its committed lanes, slot `s`
/// at `s * BLOCK ..` and a word per instance, slot `s` being lane `s % 25` of the state after
/// `depth * (s / 25)` rounds: every state a run starts from, the output not being committed.
/// With `keep`, it also returns its input state and the state each round applies `chi` to, all
/// sliced, round after round; without, the run ends at the last committed state.
pub(crate) fn run_block(
    depth: usize,
    pad: &[u64; STATE_LANES],
    inputs: &[[u64; STATE_LANES]],
    keep: bool,
) -> (Vec<u64>, Vec<u64>) {
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
    let mut lanes = vec![0u64; STATE_LANES * (ROUNDS / depth) * BLOCK];
    let mut commit = |m: usize, state: &[Word]| {
        for lane in 0..STATE_LANES {
            for w in 0..BLOCK_WORDS {
                for z in 0..64 {
                    rows[z] = state[64 * lane + z][w];
                }
                transpose64(&mut rows);
                lanes[(STATE_LANES * m + lane) * BLOCK + 64 * w..][..64].copy_from_slice(&rows);
            }
        }
    };
    let mut rounds = Vec::with_capacity(if keep { (ROUNDS + 1) * BLOCK_LEN } else { 0 });
    let last = if keep { ROUNDS } else { ROUNDS - depth };
    if keep {
        rounds.extend_from_slice(state.as_flattened());
    }
    commit(0, &state);
    for round in 1..=last {
        linear(&state, &mut u, xor);
        if keep {
            rounds.extend_from_slice(u.as_flattened());
        }
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
        for z in (0..64).filter(|&z| iota_bit(round, z)) {
            state[z] = state[z].map(|w| !w);
        }
        if round.is_multiple_of(depth) && round < ROUNDS {
            commit(round / depth, &state);
        }
    }
    (lanes, rounds)
}

/// Run every instance of every table, padding each batch with its table's pad input, and write the committed states
/// into `states`, the first `slots * tables * n` words of the committed block: lane `l` of the state after `depth * m`
/// rounds, for `m` below the run count, of instance `i` of table `t`, is word `((25 * m + l) * tables + t) * n + i`.
/// The slots past them are the caller's: in the VM other columns ride them (`lean_vm_core::cpu::keccak_value`). The
/// blocks after a table's last input are run once, as its padding block.
pub fn commit_states(shape: Shape, tables: &[TableInputs], states: &mut [F64]) {
    let (n, count) = (shape.n(), shape.tables());
    assert_eq!(tables.len(), count);
    assert!(tables.iter().all(|table| table.inputs.len() <= n));
    assert_eq!(
        states.len(),
        shape.slots() * count * n,
        "the committed states are the wrong size"
    );
    let block_lanes =
        |table: &TableInputs, inputs: &[[u64; STATE_LANES]]| run_block(shape.depth, &table.pad, inputs, false).0;
    let lanes: Vec<Vec<Vec<u64>>> = tables
        .iter()
        .map(|table| {
            parallel::map_collect(table.inputs.len().div_ceil(BLOCK), |g| {
                block_lanes(
                    table,
                    &table.inputs[g * BLOCK..((g + 1) * BLOCK).min(table.inputs.len())],
                )
            })
        })
        .collect();
    let padding: Vec<Option<Vec<u64>>> = tables
        .iter()
        .zip(&lanes)
        .map(|(table, lanes)| (lanes.len() < shape.n_blocks()).then(|| block_lanes(table, &[])))
        .collect();
    // Column `slot * tables + t` of `n` words is slot `slot` of table `t`.
    parallel::chunks_mut(states, n, |column, out| {
        let (slot, t) = (column / count, column % count);
        for (g, out) in out.chunks_mut(BLOCK).enumerate() {
            let lanes = lanes[t].get(g).or(padding[t].as_ref()).expect("a padding block");
            for (out, &word) in out.iter_mut().zip(&lanes[slot * BLOCK..][..BLOCK]) {
                *out = F64(word);
            }
        }
    });
}

impl Witness {
    /// One table of `inputs`, padded with all-zero ones: [`Witness::generate_tables`] of a one-table shape.
    pub fn generate(shape: Shape, inputs: &[[u64; STATE_LANES]], committed: &mut [F64]) -> Self {
        assert_eq!(shape.tables(), 1);
        let table = TableInputs { inputs, pad: ZERO_PAD };
        Self::generate_tables(shape, &[table], committed)
    }

    /// [`commit_states`] over the first `slots * tables * n` words of `committed`, then the witness rebuilt from them.
    pub fn generate_tables(shape: Shape, tables: &[TableInputs], committed: &mut [F64]) -> Self {
        let states = &mut committed[..shape.slots() * shape.tables() * shape.n()];
        commit_states(shape, tables, states);
        let instances: Vec<usize> = tables.iter().map(|table| table.inputs.len()).collect();
        let pads: Vec<[u64; STATE_LANES]> = tables.iter().map(|table| table.pad).collect();
        Self::from_committed(shape, &instances, &pads, states)
    }

    /// The witness of the first `instances[t]` instances of each table, rebuilt from their committed inputs.
    ///
    /// - `committed` is laid out as [`commit_states`] writes it.
    /// - The blocks past a table's last input are its padding block.
    pub fn from_committed(shape: Shape, instances: &[usize], pads: &[[u64; STATE_LANES]], committed: &[F64]) -> Self {
        let (n, count) = (shape.n(), shape.tables());
        assert!(instances.len() == count && pads.len() == count);
        assert!(committed.len() >= STATE_LANES * count * n);
        let tables = (0..count)
            .map(|t| {
                assert!(instances[t] <= n);
                let written = instances[t].div_ceil(BLOCK);
                let held = written + usize::from(written < shape.n_blocks());
                let blocks = parallel::map_collect(held, |g| {
                    // Slot `l` holds lane `l` of every input, a word per instance. A step past the last
                    // input reads the input the block pads with.
                    let inputs: Vec<[u64; STATE_LANES]> = if g < written {
                        (g * BLOCK..(g + 1) * BLOCK)
                            .map(|i| std::array::from_fn(|lane| committed[(lane * count + t) * n + i].0))
                            .collect()
                    } else {
                        Vec::new()
                    };
                    run_block(shape.depth, &pads[t], &inputs, true).1
                });
                TableWitness { blocks, written }
            })
            .collect();
        Self { shape, tables }
    }

    /// The blocks of `table` held: those written, then the padding block if any block follows them.
    pub(crate) fn held(&self, table: usize) -> usize {
        self.tables[table].blocks.len()
    }

    /// The blocks of `table` holding inputs: every later one is its padding block.
    pub(crate) fn written(&self, table: usize) -> usize {
        self.tables[table].written
    }

    /// Block `g` of `table`'s state round `round` applies `chi` to, rounds counted from 1, or of the
    /// input state for `round` 0, sliced.
    pub(crate) fn layer(&self, table: usize, round: usize, g: usize) -> &[u64] {
        assert!(g < self.shape.n_blocks(), "a block of the batch");
        let table = &self.tables[table];
        &table.blocks[g.min(table.written)][round * BLOCK_LEN..][..BLOCK_LEN]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use primitives::test_rng::Rng;

    #[test]
    fn transpose_swaps_rows_and_columns() {
        let mut rng = Rng::new(8);
        let a: [u64; 64] = std::array::from_fn(|_| rng.next_u64());
        let mut t = a;
        transpose64(&mut t);
        for r in 0..64 {
            for c in 0..64 {
                assert_eq!(t[c] >> r & 1, a[r] >> c & 1);
            }
        }
    }

    #[test]
    fn committed_states_follow_the_permutation() {
        let mut rng = Rng::new(9);
        let inputs: Vec<[u64; STATE_LANES]> = (0..1100).map(|_| std::array::from_fn(|_| rng.next_u64())).collect();
        let shape = Shape::new(inputs.len(), 8);
        let mut committed = vec![F64(0); 1 << shape.stack_log()];
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
    }

    /// The tables' input states sit side by side in each slot, a padding instance holding its table's pad input.
    #[test]
    fn tables_commit_side_by_side() {
        let mut rng = Rng::new(10);
        let mut inputs =
            |n: usize| -> Vec<[u64; STATE_LANES]> { (0..n).map(|_| std::array::from_fn(|_| rng.next_u64())).collect() };
        let (a, b) = (inputs(700), inputs(1100));
        let pad = std::array::from_fn(|l| 0x0303 * (l as u64 + 1));
        let tables = [
            TableInputs {
                inputs: &a,
                pad: ZERO_PAD,
            },
            TableInputs { inputs: &b, pad },
        ];
        let shape = Shape::with_tables(1100, 12, 1);
        let mut committed = vec![F64(0); 1 << shape.stack_log()];
        Witness::generate_tables(shape, &tables, &mut committed);
        let n = shape.n();
        for (t, table) in tables.iter().enumerate() {
            for x in (0..n).step_by(37) {
                let state = table.inputs.get(x).copied().unwrap_or(table.pad);
                for lane in 0..STATE_LANES {
                    assert_eq!(
                        committed[(lane * 2 + t) * n + x].0,
                        state[lane],
                        "table {t}, instance {x}"
                    );
                }
            }
        }
    }

    /// A shape whose committed block would not fit a `usize` is refused, rather than wrapping.
    #[test]
    #[should_panic(expected = "too many instances")]
    fn an_unrepresentable_batch_is_refused() {
        Shape::new(1usize << 63, 12);
    }
}
