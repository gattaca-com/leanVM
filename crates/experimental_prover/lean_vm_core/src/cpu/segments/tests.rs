//! A hand-assembled loop whose iterations take their `DEREF` pointers, write pointers and values from witness streams,
//! run serially and as segments.

use std::collections::HashMap;

use super::super::execute::{Execution, Lanes, execute_as};
use super::super::{DerefMode, ExecError, Op, OutCells, Program, to_scalar};
use super::{HEAVY_ITERATION as HEAVY, LOOPS, LOW};
use isa::RHint;
use primitives::field::{F128, g_pow};
use verifier::K;
use verifier::layout::{OP_KECCAK, OP_KECCAK64};

/// The first iteration's frame, above `LOW` so that the segments own their frames.
const FRAME0: usize = 70_000;
/// First frames below `LOW`, where every cell is an overlay cell, and straddling it, where a segment's own cells start
/// at `LOW`.
const LOWER: [usize; 2] = [6_000, LOW as usize - 20 * STRIDE];
const STRIDE: usize = 64;
const ITERATIONS: usize = 40;
/// The loop's first pc, whose hints pop the stream `loop`.
const MARKER: u32 = 8;
/// Cells an iteration allocates, in two buffers.
const HEAP: usize = 16;

type Run = (Program, Vec<(String, Vec<Vec<F128>>)>);

fn address(i: usize) -> F128 {
    F128::from(g_pow(i))
}

fn word(v: u64) -> F128 {
    F128::new(v, 0)
}

fn set(o: usize, k: F128) -> Op {
    Op::Set {
        o: o as u32,
        k: to_scalar(&k),
    }
}

fn deref(o1: u32, o3: u32) -> Op {
    Op::Deref {
        o1,
        o2: 0,
        o3,
        mode: DerefMode::Cell,
    }
}

/// One iteration's witness: the cells its four `DEREF`s equate with frame cells 24..28, the cells three heap hints
/// write, and their values.
#[derive(Clone)]
struct Iteration {
    ptrs: [usize; 4],
    wptrs: [usize; 3],
    values: [u64; 3],
}

/// Main's frame, past every iteration's: the loop's frames are reserved before it starts.
fn main(frame0: usize) -> usize {
    frame0 + (ITERATIONS + 2) * STRIDE
}

/// Instructions a hashing iteration runs before its jump back, in frame cells no other instruction uses: five
/// constants, then a `KECCAK64` and a `KECCAK` over the iteration's own words (cells 1 and 2) and the first two.
/// The `KECCAK`'s `tail` and `cap` are the same four cells.
const HASH_OPS: usize = 7;

fn hash_ops() -> [Op; HASH_OPS] {
    [
        set(52, word(11)),
        set(53, word(13)),
        set(54, word(17)),
        set(55, word(19)),
        set(58, word(23)),
        Op::keccak64([1, 2, 52, 53], 56, true, K(1), K(0), K(0)).expect("a nonzero scale"),
        Op::Keccak {
            m: [1, 2, 52, 53],
            tail: 52,
            b8: 58,
            cap: 52,
            out: 59,
            in_odd: false,
            out_odd: false,
            out_cells: OutCells::One,
        },
    ]
}

/// The loop over `iterations`, its first frame at `frame0`, with `prefix` run in main's frame before it. With `heap`,
/// each iteration allocates two buffers of `HEAP / 2` cells, at its first instruction and after its `DEREF`s. With
/// `hashes`, it also runs a `KECCAK64` and a `KECCAK` ([`hash_ops`]).
fn program(frame0: usize, iterations: &[Iteration], heap: bool, hashes: bool, prefix: [Op; 2]) -> Run {
    padded(frame0, iterations, heap, hashes, 0, prefix)
}

/// [`program`], with `pad` more instructions before each iteration's jump back, which rewrite a frame cell with its own
/// value: a loop of iterations that run past [`HEAVY`] steps.
fn padded(frame0: usize, iterations: &[Iteration], heap: bool, hashes: bool, pad: usize, prefix: [Op; 2]) -> Run {
    let tail = 17 + pad + if hashes { HASH_OPS } else { 0 };
    let len = (tail + 5).next_power_of_two().max(64);
    let mut code = vec![set(7, F128::ZERO); len];
    code[0] = set(frame0, word(7));
    code[1] = set(2, F128::ONE);
    code[2] = set(3, address(MARKER as usize));
    code[3] = set(4, address(frame0));
    code[4] = set(310, word(7));
    code[5..7].copy_from_slice(&prefix);
    code[7] = Op::Jump { oc: 2, od: 3, of: 4 };
    code[8] = set(4, word(7));
    code[9] = set(5, F128::ONE);
    // Back-solve the next frame's first cell: the loop's one argument.
    code[10] = Op::Mul {
        a: STRIDE as u32,
        b: 5,
        c: 4,
    };
    code[11] = deref(1, 0);
    for k in 0..4 {
        code[12 + k as usize] = deref(8 + k, 24 + k);
    }
    code[16] = set(6, address(MARKER as usize));
    // Cell 50 holds 7 in every iteration: the iterations' pointers and writes never reach it.
    code[17..tail].fill(set(50, word(7)));
    if hashes {
        code[17..17 + HASH_OPS].copy_from_slice(&hash_ops());
    }
    code[tail] = Op::Jump { oc: 3, od: 6, of: 2 };
    // The exit path, in the last iteration's frame, ending at the last slot.
    code[tail + 1] = set(7, address(len - 1));
    code[tail + 2] = set(30, address(5000));
    code[tail + 3] = deref(30, 31);
    code[tail + 4] = Op::Jump { oc: 5, od: 7, of: 5 };
    let stack = |name: &str, base, len| RHint::WitnessStack {
        name: name.into(),
        base,
        len,
    };
    let alloc = |ptr| RHint::Alloc {
        ptr,
        size: HEAP as u32 / 2,
    };
    let write = |q: u32| RHint::WitnessHeap {
        name: format!("v{q}"),
        ptr: 16 + q,
        lo: 0,
        len: 1,
    };
    let marker = [stack("loop", 1, 3)].into_iter().chain(heap.then(|| alloc(20)));
    let marker = marker.chain([stack("ptrs", 8, 4), stack("wptrs", 16, 3)]).collect();
    let mut hints = HashMap::from([
        (MARKER, marker),
        (12, vec![write(0)]),
        (14, vec![write(1)]),
        (15, vec![write(2)]),
    ]);
    if heap {
        hints.insert(16, vec![alloc(21)]);
    }
    let program = Program::from(isa::CompiledProgram {
        prog: code,
        hints,
        main_frame: main(frame0) as u32,
        fn_ranges: Vec::new(),
        src_lines: Vec::new(),
    });
    let mut streams: Vec<_> = ["loop", "ptrs", "wptrs", "v0", "v1", "v2"]
        .map(|s| (s.to_string(), Vec::new()))
        .into();
    for (i, it) in iterations.iter().enumerate() {
        let more = F128::from(u128::from(i + 1 < iterations.len()));
        streams[0]
            .1
            .push(vec![address(1000 + i), address(frame0 + (i + 1) * STRIDE), more]);
        streams[1].1.push(it.ptrs.iter().map(|&p| address(p)).collect());
        streams[2].1.push(it.wptrs.iter().map(|&p| address(p)).collect());
        for q in 0..3 {
            streams[3 + q].1.push(vec![word(it.values[q])]);
        }
    }
    (program, streams)
}

fn plain(frame0: usize, iterations: &[Iteration], heap: bool, hashes: bool) -> Run {
    program(frame0, iterations, heap, hashes, [set(311, word(9)), set(312, word(7))])
}

/// The run, and the steps it handed to the Keccak reduction, in order.
fn execute((program, witness): &Run, segments: bool) -> Result<(Execution, Vec<Lanes>), ExecError> {
    let mut steps = Vec::new();
    let e = execute_as(
        program,
        witness,
        [F128::ZERO; 2],
        &mut |step: &Lanes| steps.push(*step),
        segments,
    )?;
    Ok((e, steps))
}

/// How the run with segments differs from the serial one (memory image, counts, write states, rows, Keccak steps, or
/// error), if it does, and the loops it dispatched and merged.
fn divergence(run: &Run) -> (Option<String>, (usize, usize)) {
    let serial = execute(run, false);
    LOOPS.set((0, 0));
    let d = match (serial, execute(run, true)) {
        (Ok((s, serial_steps)), Ok((e, steps))) => {
            let cells = (0..s.mem_used).all(|i| s.mem.cell(i) == e.mem.cell(i));
            let same = (s.cycles, s.mem_used, s.mem.len(), &s.unconstrained_reads, &s.trace)
                == (e.cycles, e.mem_used, e.mem.len(), &e.unconstrained_reads, &e.trace);
            (!cells || !same || steps != serial_steps).then(|| "differs from serial".into())
        }
        (Err(s), Err(e)) => (s != e).then(|| format!("error `{e}`, serial error `{s}`")),
        (Ok(_), Err(e)) => Some(format!("error `{e}`, the serial run succeeds")),
        (Err(s), Ok(_)) => Some(format!("succeeds, serial error `{s}`")),
    };
    (d, LOOPS.get())
}

/// With one worker every loop runs serially, so a positive control holds only with two or more.
fn merges() -> usize {
    usize::from(parallel::num_threads() >= 2)
}

/// Pointers and writes to cells of the iteration's own, in a low range no other iteration reaches.
fn benign(i: usize) -> Iteration {
    Iteration {
        ptrs: std::array::from_fn(|k| 2000 + 16 * i + k),
        wptrs: std::array::from_fn(|k| 2008 + 16 * i + k),
        values: [7; 3],
    }
}

/// Positive control: the plain loop runs as segments and matches, with and without allocations or hashes, wherever its
/// frames.
#[test]
fn plain_loop_runs_as_segments() {
    let iterations: Vec<_> = (0..ITERATIONS).map(benign).collect();
    for frame0 in [FRAME0, LOWER[0], LOWER[1]] {
        for (heap, hashes) in [(false, false), (true, false), (false, true), (true, true)] {
            assert_eq!(
                divergence(&plain(frame0, &iterations, heap, hashes)),
                (None, (merges(), merges())),
                "frame0 {frame0}, heap {heap}, hashes {hashes}"
            );
        }
    }
}

/// Invariant: a loop whose iterations run a `KECCAK` and a `KECCAK64` gives, as segments, the serial run's rows of
/// both tables and the steps it passes on, in the serial order and tagged with their table.
#[test]
fn keccak_rows_of_both_tables_merge_as_serial() {
    let iterations: Vec<_> = (0..ITERATIONS).map(benign).collect();
    let run = plain(FRAME0, &iterations, false, true);
    let (serial, steps) = execute(&run, false).expect("the serial run succeeds");
    let rows = serial.rows();
    assert_eq!((rows[OP_KECCAK], rows[OP_KECCAK64]), (ITERATIONS, ITERATIONS));
    let tagged = |table| steps.iter().filter(|(t, _)| *t == table).count();
    assert_eq!(
        (tagged(crate::keccak::KECCAK_HALF), tagged(crate::keccak::KECCAK64_HALF)),
        (ITERATIONS, ITERATIONS),
        "a step per row, by table"
    );
    assert_eq!(divergence(&run), (None, (merges(), merges())));
}

/// Invariant: a loop of light iterations runs as segments of at least eight iterations, so a short one runs serially,
/// with the serial result.
#[test]
fn short_loop_of_light_iterations_is_serial() {
    for n in [5, 12, 18] {
        let iterations: Vec<_> = (0..n).map(benign).collect();
        assert_eq!(
            divergence(&plain(FRAME0, &iterations, true, false)),
            (None, (0, 0)),
            "{n} iterations"
        );
    }
}

/// Invariant: a loop of a few heavy iterations, like the aggregation guest's child loop, runs as segments of one
/// iteration each, with the serial result, whatever its frames; with fewer than five iterations it runs serially.
#[test]
fn few_heavy_iterations_run_as_segments() {
    let prefix = || [set(311, word(9)), set(312, word(7))];
    for n in [3, 4, 5, 6, 9] {
        let iterations: Vec<_> = (0..n).map(benign).collect();
        for frame0 in [FRAME0, LOWER[0], LOWER[1]] {
            let run = padded(frame0, &iterations, true, true, 2 * HEAVY, prefix());
            let loops = if n >= 5 { merges() } else { 0 };
            assert_eq!(
                divergence(&run),
                (None, (loops, loops)),
                "{n} heavy iterations, frame0 {frame0}"
            );
        }
    }
    let iterations: Vec<_> = (0..5).map(benign).collect();
    let run = padded(FRAME0, &iterations, true, false, 2 * HEAVY, prefix());
    let (serial, _) = execute(&run, false).expect("the serial run succeeds");
    assert!(serial.cycles > 5 * HEAVY, "the iterations run past the heavy mark");
}

/// A segment sees its own cells as memory holds them when the loop starts, so a run whose prefix writes or links a
/// later iteration's frame gives the serial result, for every frame a segment may start at. Writing a frame's argument
/// cell makes the iteration before it conflict at its `MUL`; linking an own cell with a low cell carries the
/// iteration's later write to the low cell.
#[test]
fn prefix_touching_own_cells_is_serial() {
    let iterations: Vec<_> = (0..ITERATIONS).map(benign).collect();
    for b in 3..ITERATIONS - 1 {
        let fp = FRAME0 + b * STRIDE;
        let argument = program(FRAME0, &iterations, false, false, [set(311, word(9)), set(fp, word(9))]);
        assert_eq!(execute(&argument, false).err().map(|e| e.pc), Some(10), "frame {b}");
        let mut linked = iterations.clone();
        linked[b].wptrs[0] = fp + 25;
        let link = program(
            FRAME0,
            &linked,
            false,
            false,
            [set(311, address(fp + 25)), deref(311, 305)],
        );
        assert_eq!(execute(&link, false).unwrap().0.mem.word(305), word(7), "frame {b}");
        for (name, run) in [("argument", argument), ("link", link)] {
            assert_eq!(divergence(&run).0, None, "{name} in frame {b}");
        }
    }
}

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n) as usize
    }
}

/// A run with segments gives the serial execution or error, whatever links and stores join the segments' own cells,
/// the low cells every iteration shares, a cell written before the loop and the next segment's frame. The pointers go
/// to the iteration's own low cells, shared ones, its frame, `310` and the next frame; with `heap`, a quarter of the
/// writes land in its first buffer; a value is rarely 9 rather than 7, for a write-once conflict. Two cases in three
/// put the frames below `LOW` or across it. Positive controls: some cases merge, some fall back to a serial rerun after
/// dispatching, and some fail serially.
#[test]
fn random_links_and_stores_merge_as_serial() {
    let (mut merged, mut fell_back, mut failing) = (0, 0, 0);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for case in 0..1000 {
        let heap = case % 2 == 0;
        let frame0 = [FRAME0, LOWER[0], LOWER[1]][case % 3];
        let iterations: Vec<_> = (0..ITERATIONS)
            .map(|i| {
                let fp = frame0 + i * STRIDE;
                let pick = |rng: &mut Rng, slot: usize| match rng.below(10) {
                    0..=2 => 2000 + 16 * i + slot,
                    3..=5 => 300 + rng.below(4),
                    6..=7 => fp + 24 + rng.below(6),
                    8 => 310,
                    // Away from the loop's ends, whose frames nothing predicts.
                    _ if i >= 2 && i + 2 < ITERATIONS => fp + STRIDE + 40 + rng.below(2),
                    _ => 2000 + 16 * i + slot,
                };
                let ptrs = std::array::from_fn(|k| pick(&mut rng, k));
                let wptrs = std::array::from_fn(|k| match heap && rng.below(4) == 0 {
                    true => main(frame0) + i * HEAP + rng.below(HEAP as u64 / 2),
                    false => pick(&mut rng, 8 + k),
                });
                let values = std::array::from_fn(|_| if rng.below(400) == 0 { 9 } else { 7 });
                Iteration { ptrs, wptrs, values }
            })
            .collect();
        let run = plain(frame0, &iterations, heap, case % 4 == 1);
        failing += usize::from(execute(&run, false).is_err());
        let (d, (dispatched, ok)) = divergence(&run);
        assert_eq!(d, None, "case {case}");
        (merged, fell_back) = (merged + ok, fell_back + dispatched - ok);
    }
    assert!(
        merges() == 0 || (merged > 0 && fell_back > 0),
        "{merged} cases merged, {fell_back} fell back"
    );
    assert!(failing > 0, "no case fails serially");
}
