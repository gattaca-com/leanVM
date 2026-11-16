//! Bridge to the Keccak prover ([`keccak_gkr`]), single-PCS.
//!
//! The committed states of every `KECCAK` step are one column of leanVM's ONE stacked `F64`
//! witness (§sec:stacking), a lane per slot and a word per step, with no separate commitment:
//! the input and the state after every `DEPTH` rounds before the output. The output is not
//! committed. The VM's `KECCAK` table commits no value column of its own: a claim on an input lane is
//! a claim on its slot of that column, and the claims on the output lanes are the reduction's
//! claim on the permutation's output ([`keccak_gkr::OutputClaim`]). The GKR reduction proves the
//! rounds between, and its openings join the batch-mixed opening ([`::pcs::stack_open`]): the
//! column's bit slices as the ring-switched claim, the committed states after the input as a
//! point claim.
//!
//! ## The mapping
//!
//! The VM's `KECCAK` is Keccak-f\[1600\] of the 25 lanes thirteen cells hold, written to one, two
//! or all thirteen cells, each side in one of two layouts ([`keccak_cell_lanes`]). The committed input is
//! the lanes the VM reads, and the output is the permutation's, written as it is.

use crate::transcript::ProverState;
use keccak_gkr::witness::{BLOCK, ZERO_PAD, block_lanes};
use keccak_gkr::{Openings, OutputClaim, Shape, Witness};
use primitives::field::{F64, F128};
use primitives::hash::STATE_LANES;
use verifier::layout::keccak_cell_lanes;

/// Rounds between two committed states.
pub const DEPTH: usize = 12;
/// `log2` of the tables the reduction proves together: `KECCAK` is half 0 of the committed column, `KECCAK64` half 1.
pub const TABLE_BITS: usize = 1;
/// The tables the reduction proves together.
pub const TABLES: usize = 1 << TABLE_BITS;
/// The half of the committed column, and the table bit, of `KECCAK` and of `KECCAK64`.
pub const KECCAK_HALF: usize = 0;
pub const KECCAK64_HALF: usize = 1;
/// The reduction's shape at its fewest steps.
const MIN_SHAPE: Shape = shape(0);
/// `log2` of the slots the committed column holds per step: one per lane of each state but the
/// output, rounded up to a power of two.
pub const SLOT_LOG: usize = MIN_SHAPE.slot_log();
/// Slots of a step the committed states occupy, one per lane of each: the ones from here to
/// `2^SLOT_LOG` carry no state, and the tables' own columns ride them (`cpu::keccak_value`).
pub const SLOTS: usize = MIN_SHAPE.slots();
/// The fewest steps the reduction is sized for, one of its blocks. Each table is sized for at least this.
pub const MIN_STEPS: usize = MIN_SHAPE.n();

/// `log2` of the steps each table is completed to, for tables of `rows` executed steps: the tallest's.
pub(crate) fn kappa(rows: [usize; TABLES]) -> usize {
    rows.into_iter()
        .map(|n| Shape::new(n, DEPTH).n_log())
        .max()
        .unwrap_or(0)
}

/// The reduction's shape for tables of `2^kappa` steps each.
const fn shape(n_steps: usize) -> Shape {
    Shape::with_tables(n_steps, DEPTH, TABLE_BITS)
}

/// `log2` of the committed column, in words: the step, the table, then the slot.
pub fn qkeccak_log(kappa: usize) -> usize {
    shape(1 << kappa).stack_log()
}

/// Cells one `KECCAK64` row flushes: four read, then the one or two written (the second cancelled when it is not).
pub const KECCAK64_ROW_CELLS: usize = 6;

/// The state a 64-byte message leaves, but for the message's own eight lanes: the padding's first byte in lane 8
/// and its last in lane 16, the end of the rate. A `KECCAK64` row holds these in lanes 8 to 24.
pub fn keccak64_pad() -> [u64; STATE_LANES] {
    let mut pad = [0; STATE_LANES];
    pad[8] = u64::from(primitives::hash::PAD_FIRST);
    pad[primitives::hash::RATE / 8 - 1] = primitives::hash::END_BIT;
    pad
}

/// The slots of `q_keccak` a `KECCAK64` row does not commit: lanes 8 to 23 of its input state hold the padding on every row,
/// executed or not. They are public constants, [`keccak64_pad`]'s.
pub const KECCAK64_PUBLIC_SLOTS: std::ops::Range<usize> = 8..24;

/// The slot blocks a half of `q_keccak` commits, `(first slot, log slots)`, each an aligned cube of the slot index. A
/// piece of a half is a row piece times a block. `KECCAK` commits all 64 slots in one block. `KECCAK64` leaves out the public
/// slots 8 to 23: slots 0 to 7, 24 to 31 (lane 24 and the middle state's first seven lanes), and 32 to 63.
pub const fn slot_blocks(half: usize) -> &'static [(usize, usize)] {
    if half == KECCAK_HALF {
        &[(0, 6)]
    } else {
        &[(0, 3), (24, 3), (32, 5)]
    }
}

/// The constant slot `slot` of a half holds on every row, if it is public.
pub fn public_slot(half: usize, slot: usize) -> Option<u64> {
    (half == KECCAK64_HALF && KECCAK64_PUBLIC_SLOTS.contains(&slot)).then(|| keccak64_pad()[slot])
}

/// The input every padding step of each table holds: zeros for `KECCAK`, a message of zeros for `KECCAK64`.
pub fn pad_inputs() -> [[u64; STATE_LANES]; TABLES] {
    [ZERO_PAD, keccak64_pad()]
}

/// The 25 lanes of a `KECCAK64` row's state, from the four cells `cell(0..4)` it reads.
pub(crate) fn keccak64_state(cell: impl Fn(usize) -> F128) -> [u64; STATE_LANES] {
    let mut state = keccak64_pad();
    for c in 0..4 {
        let w = cell(c);
        state[2 * c] = w.c0;
        state[2 * c + 1] = w.c1;
    }
    state
}

pub use isa::STATE_CELLS;
/// Cells one `KECCAK` row flushes: thirteen read, thirteen written or, past what it writes, cancelled.
pub(crate) const ROW_CELLS: usize = 2 * STATE_CELLS;

/// The 25 lanes the thirteen state cells `cell(0..13)` hold in a layout.
pub(crate) fn state_of_cells(odd: bool, cell: impl Fn(usize) -> F128) -> [u64; STATE_LANES] {
    // The layout is a constant, so the loop compiles to straight-line moves.
    fn read<const ODD: bool>(cell: impl Fn(usize) -> F128) -> [u64; STATE_LANES] {
        let mut s = [0; STATE_LANES];
        for c in 0..STATE_CELLS {
            let (lo, hi) = const { keccak_cell_lanes(ODD) }[c];
            let w = cell(c);
            if let Some(l) = lo {
                s[l] = w.c0;
            }
            if let Some(h) = hi {
                s[h] = w.c1;
            }
        }
        s
    }
    if odd { read::<true>(cell) } else { read::<false>(cell) }
}

/// The thirteen state cells holding 25 lanes in a layout, a free lane zero.
pub(crate) fn cells_of_state(s: &[u64; STATE_LANES], odd: bool) -> [F128; STATE_CELLS] {
    let lanes = keccak_cell_lanes(odd);
    std::array::from_fn(|c| {
        let (lo, hi) = lanes[c];
        F128::new(lo.map_or(0, |l| s[l]), hi.map_or(0, |h| s[h]))
    })
}

/// What a `KECCAK` instruction writes, given the cells it reads.
#[cfg(test)]
pub(crate) fn permute_cells(cells: &[F128; STATE_CELLS], in_odd: bool, out_odd: bool) -> [F128; STATE_CELLS] {
    let mut s = state_of_cells(in_odd, |c| cells[c]);
    primitives::hash::keccak_f(&mut s);
    cells_of_state(&s, out_odd)
}

/// What the reduction needs beside the committed column.
///
/// Its witness, every round's state of every step, is rebuilt from the column's input states.
/// Held from the run instead, it would sit beside the bus's trees, at the prover's peak.
pub(crate) struct Reduction {
    shape: Shape,
    steps: [usize; TABLES],
}

impl Reduction {
    /// Rebuild the witness from `q_keccak`, the whole committed block, and run the reduction.
    pub(crate) fn prove(self, ps: &mut ProverState, q_keccak: &[F64], output: &OutputClaim) -> Openings {
        let witness = tracing::info_span!("Witness")
            .in_scope(|| Witness::from_committed(self.shape, &self.steps, &pad_inputs(), q_keccak));
        witness.prove(ps, q_keccak, output)
    }
}

/// A step the interpreter reports to the reduction: the table that ran it (the index of the half of `q_keccak`) and
/// the lanes it reads.
pub(crate) type Step = (usize, [u64; STATE_LANES]);

/// The committed lanes of the steps a program ran, built while it ran: every block of steps of every table,
/// slot after slot.
pub(crate) struct Blocks {
    lanes: [Vec<Vec<u64>>; TABLES],
    steps: [usize; TABLES],
}

/// Where an execution hands the steps of the Keccak tables, in step order.
pub(crate) trait Steps {
    fn step(&mut self, step: &Step);
    /// The steps of `runs`, end to end.
    fn steps(&mut self, runs: &[&[Step]]) {
        runs.iter().copied().flatten().for_each(|step| self.step(step));
    }
}

impl<F: FnMut(&Step)> Steps for F {
    fn step(&mut self, step: &Step) {
        self(step)
    }
}

/// A block of one table's steps for a worker: the table, the block's index in it, and the lanes it reads.
type Job = (usize, usize, Vec<[u64; STATE_LANES]>);

/// The steps of a run as [`while_executing`] takes them: whole blocks to its workers, the last
/// ones' lanes on the executing thread.
struct Sink<'a> {
    jobs: Option<&'a std::sync::mpsc::Sender<Job>>,
    /// Each table's steps of its block so far.
    pending: [Vec<[u64; STATE_LANES]>; TABLES],
    /// Blocks done here, by their table and index.
    inline: Vec<(usize, usize, Vec<u64>)>,
    blocks: [usize; TABLES],
    steps: [usize; TABLES],
}

impl Sink<'_> {
    fn dispatch(&mut self, table: usize) {
        let block = std::mem::replace(&mut self.pending[table], Vec::with_capacity(BLOCK));
        match self.jobs {
            Some(jobs) => jobs
                .send((table, self.blocks[table], block))
                .expect("the workers outlive the run"),
            None => self.inline.push((
                table,
                self.blocks[table],
                block_lanes(DEPTH, &pad_inputs()[table], &block),
            )),
        }
        self.blocks[table] += 1;
    }
}

impl Steps for Sink<'_> {
    fn step(&mut self, &(table, lanes): &Step) {
        self.pending[table].push(lanes);
        self.steps[table] += 1;
        if self.pending[table].len() == BLOCK {
            self.dispatch(table);
        }
    }

    /// The whole blocks of each table among `runs` on the pool, which their sender leaves idle.
    fn steps(&mut self, runs: &[&[Step]]) {
        let pads = pad_inputs();
        for table in 0..TABLES {
            let mut all = runs
                .iter()
                .copied()
                .flatten()
                .filter(|step| step.0 == table)
                .map(|step| &step.1);
            // Fill the pending block, so that the rest starts one.
            while !self.pending[table].is_empty()
                && let Some(lanes) = all.next()
            {
                self.step(&(table, *lanes));
            }
            let rest: Vec<_> = all.collect();
            let whole = rest.len() / BLOCK;
            let done = parallel::map_collect(whole, |b| {
                let block: Vec<_> = rest[b * BLOCK..(b + 1) * BLOCK].iter().map(|&&lanes| lanes).collect();
                block_lanes(DEPTH, &pads[table], &block)
            });
            for lanes in done {
                self.inline.push((table, self.blocks[table], lanes));
                self.blocks[table] += 1;
            }
            self.steps[table] += whole * BLOCK;
            rest[whole * BLOCK..]
                .iter()
                .for_each(|&&lanes| self.step(&(table, lanes)));
        }
    }
}

/// Run `run`, which reports each step it executes, and compute the committed lanes
/// behind it: executing is one thread's work, so the others take each block of steps as it
/// completes instead of waiting for the last.
pub(crate) fn while_executing<R>(run: impl FnOnce(&mut dyn Steps) -> R) -> (R, Blocks) {
    let pads = pad_inputs();
    let (jobs, queue) = std::sync::mpsc::channel::<Job>();
    let queue = std::sync::Mutex::new(queue);
    // More workers than this slow the executing thread by more than they finish early.
    let workers = parallel::num_threads() / 2;
    let (result, lanes, steps) = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let job = queue.lock().expect("a worker panicked").recv();
                        let Ok((table, g, steps)) = job else { break };
                        done.push((table, g, block_lanes(DEPTH, &pads[table], &steps)));
                    }
                    done
                })
            })
            .collect();
        let mut sink = Sink {
            jobs: (workers > 0).then_some(&jobs),
            pending: std::array::from_fn(|_| Vec::with_capacity(BLOCK)),
            inline: Vec::new(),
            blocks: [0; TABLES],
            steps: [0; TABLES],
        };
        let result = run(&mut sink);
        sink.jobs = None;
        for table in 0..TABLES {
            if !sink.pending[table].is_empty() {
                sink.dispatch(table);
            }
        }
        let Sink { inline, steps, .. } = sink;
        drop(jobs);
        let mut lanes = inline;
        for handle in handles {
            lanes.extend(handle.join().expect("a witness worker panicked"));
        }
        lanes.sort_unstable_by_key(|&(table, g, _)| (table, g));
        let mut by_table: [Vec<Vec<u64>>; TABLES] = std::array::from_fn(|_| Vec::new());
        for (table, _, lanes) in lanes {
            by_table[table].push(lanes);
        }
        (result, by_table, steps)
    });
    (result, Blocks { lanes, steps })
}

impl Blocks {
    /// Write the committed states of the steps these blocks hold, each table completed with its padding steps up to
    /// `2^kappa`. `q_keccak` is the states' part of the committed block, its first [`SLOTS`] slots: the
    /// spare slots past them belong to the columns that ride there and are written by their own fill.
    pub(crate) fn place(self, steps: [usize; TABLES], kappa: usize, q_keccak: &mut [F64]) -> Reduction {
        assert_eq!(self.steps, steps, "the blocks are of another run");
        let shape = shape(1 << kappa);
        assert_eq!(
            q_keccak.len(),
            shape.slots() * shape.n() * TABLES,
            "the committed states are the wrong size"
        );
        let lanes: Vec<&[Vec<u64>]> = self.lanes.iter().map(Vec::as_slice).collect();
        Witness::write_committed(shape, &pad_inputs(), &lanes, q_keccak);
        Reduction { shape, steps }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: u64) -> F64 {
        F64(x)
    }

    /// The committed column for the tables' `instances`, each padded with its padding steps, and the reduction's
    /// witness.
    fn build_qkeccak(instances: [&[[u64; STATE_LANES]]; TABLES]) -> (Vec<F64>, Witness) {
        let kappa = kappa(instances.map(<[_]>::len));
        let mut q_keccak = vec![F64::ZERO; 1 << qkeccak_log(kappa)];
        let states = SLOTS * (1 << kappa) * TABLES;
        let pads = pad_inputs();
        let tables: [keccak_gkr::TableInputs; TABLES] = std::array::from_fn(|t| keccak_gkr::TableInputs {
            inputs: instances[t],
            pad: pads[t],
        });
        let witness = Witness::generate_tables(shape(1 << kappa), &tables, &mut q_keccak[..states]);
        (q_keccak, witness)
    }

    fn sample_instances(n: usize) -> Vec<[u64; STATE_LANES]> {
        (0..n as u64)
            .map(|i| std::array::from_fn(|l| 0x0101_0101_0101_0101u64.wrapping_mul(i + 1) ^ l as u64))
            .collect()
    }

    /// The lanes committed behind a run, handed over a step or a run of steps at a time, and the
    /// witness rebuilt from them, are what a single pass over its steps builds: the same committed column and the same reduction, whether the steps fill whole blocks, leave a
    /// partial one, or leave blocks of the batch with no step at all.
    #[test]
    fn blocks_built_while_executing_match_one_pass() {
        for (n, m) in [
            (0, 0),
            (1, 700),
            (511, 3),
            (512, 0),
            (513, 513),
            (1500, 2),
            (4099, 1500),
        ] {
            let (k_steps, t_steps) = (sample_instances(n), sample_instances(m));
            let (expected, reference) = build_qkeccak([&k_steps, &t_steps]);
            // The two tables' steps interleave, as the interpreter reports them.
            let mut steps: Vec<Step> = Vec::new();
            for i in 0..n.max(m) {
                if i < n {
                    steps.push((0, k_steps[i]));
                }
                if i < m {
                    steps.push((1, t_steps[i]));
                }
            }
            // One at a time, then in runs that start and end anywhere in a block.
            let ((), blocks) = while_executing(|sink| {
                let (first, rest) = steps.split_at(steps.len().min(7));
                first.iter().for_each(|step| sink.step(step));
                let runs: Vec<_> = rest
                    .chunks(700)
                    .flat_map(|c| [&c[..c.len() / 3], &c[c.len() / 3..]])
                    .collect();
                sink.steps(&runs);
            });
            let kappa = kappa([n, m]);
            let mut q_keccak = vec![f(0xdead_beef); expected.len()];
            let states = SLOTS * (1 << kappa) * TABLES;
            let witness = blocks.place([n, m], kappa, &mut q_keccak[..states]);
            assert_eq!(q_keccak[..states], expected[..states], "{n} and {m} steps");
            // What the columns riding the spare slots left there is none of the reduction's
            // business, but both openings fold it, so the two runs have to see the same.
            q_keccak[states..].fill(f(0x5151_5151));
            let point = (0..kappa).map(|i| F128::new(3 + i as u64, 5)).collect();
            let pads = pad_inputs();
            let tables: [keccak_gkr::TableInputs; TABLES] = std::array::from_fn(|t| keccak_gkr::TableInputs {
                inputs: [&k_steps[..], &t_steps[..]][t],
                pad: pads[t],
            });
            let output = OutputClaim::of_tables(shape(1 << kappa), &tables, point);
            let mut ps = ProverState::from_label(b"blocks");
            let deferred = (witness.prove(&mut ps, &q_keccak, &output), ps.into_proof());
            let mut ps = ProverState::from_label(b"blocks");
            let one_pass = (reference.prove(&mut ps, &q_keccak, &output), ps.into_proof());
            assert!(deferred == one_pass, "{n} and {m} steps");
        }
    }

    /// Both layouts round-trip, and a padded message laid out in cells, block `k`'s nine cells
    /// from cell `17k / 2` XORed into the previous output, read and written in the layout of each
    /// block's parity, the last written even, hashes to its Keccak-256.
    #[test]
    fn layouts_absorb_a_message() {
        use primitives::hash::{PAD_FIRST, PAD_LAST, RATE, permutations};
        let s: [u64; STATE_LANES] = std::array::from_fn(|i| 0x1111 * (i as u64 + 1));
        for odd in [false, true] {
            let cells = cells_of_state(&s, odd);
            assert_eq!(state_of_cells(odd, |c| cells[c]), s);
        }
        for n in [0, 1, 64, 127, 135, 136, 137, 200, 271, 272, 300, 543, 1000] {
            let msg: Vec<u8> = (0..n).map(|i| (i * 7 + 3) as u8).collect();
            let blocks = permutations(n);
            let mut padded = msg.clone();
            padded.resize(blocks * RATE + 8, 0);
            padded[n] ^= PAD_FIRST;
            padded[blocks * RATE - 1] ^= PAD_LAST;
            let cells: Vec<F128> = padded
                .as_chunks::<16>()
                .0
                .iter()
                .map(|c| {
                    let w = |k: usize| u64::from_le_bytes(c[8 * k..8 * k + 8].try_into().unwrap());
                    F128::new(w(0), w(1))
                })
                .collect();
            let mut state = [F128::ZERO; STATE_CELLS];
            for k in 0..blocks {
                for (j, cell) in state.iter_mut().take(9).enumerate() {
                    *cell += cells[17 * k / 2 + j];
                }
                state = permute_cells(&state, k % 2 == 1, k + 1 < blocks && k % 2 == 0);
            }
            let digest: Vec<u8> = [state[0].c0, state[0].c1, state[1].c0, state[1].c1]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            assert_eq!(digest, primitives::hash::hash(&msg), "{n} bytes");
        }
    }

    /// The committed column holds every step's input lanes, a word per step. The output lanes
    /// have no slot.
    #[test]
    fn qkeccak_words_match_layout() {
        let (k_steps, t_steps) = (sample_instances(5), sample_instances(9));
        let (q_keccak, _) = build_qkeccak([&k_steps, &t_steps]);
        let n = 1usize << kappa([5, 9]);
        for (t, instances) in [&k_steps, &t_steps].into_iter().enumerate() {
            for (j, input) in instances.iter().enumerate() {
                for i in 0..STATE_LANES {
                    assert_eq!(q_keccak[(i * TABLES + t) * n + j], f(input[i]));
                }
            }
        }
        // The rows past a table's own hold its padding step.
        let pad = pad_inputs();
        for i in 0..STATE_LANES {
            assert_eq!(q_keccak[i * TABLES * n + 5], f(pad[0][i]));
            assert_eq!(q_keccak[(i * TABLES + 1) * n + 9], f(pad[1][i]));
        }
    }

    /// `KECCAK64`'s state is the state of the `KECCAK` with the same four cells and the padding constants in the others.
    #[test]
    fn keccak64_state_is_the_padded_message() {
        let cells: [F128; 4] = std::array::from_fn(|i| F128::new(0x1234 * (i as u64 + 1), 0x9abc * (i as u64 + 7)));
        let keccak64 = keccak64_state(|c| cells[c]);
        let mut input = [F128::ZERO; STATE_CELLS];
        input[..4].copy_from_slice(&cells);
        input[4] = F128::new(primitives::hash::PAD_FIRST as u64, 0);
        input[8] = F128::new(primitives::hash::END_BIT, 0xdead);
        assert_eq!(state_of_cells(false, |c| input[c]), keccak64);
        let bytes: Vec<u8> = cells
            .iter()
            .flat_map(|c| [c.c0, c.c1])
            .flat_map(u64::to_le_bytes)
            .collect();
        let mut state = keccak64;
        primitives::hash::keccak_f(&mut state);
        assert_eq!(primitives::hash::digest_of(&state), primitives::hash::hash(&bytes));
    }
}
