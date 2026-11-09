//! A hand-assembled loop whose iterations take their `DEREF` pointers, write pointers and
//! values from witness streams, run serially and as segments.

use std::collections::HashMap;

use super::super::Program;
use super::super::hints::RHint;
use super::super::isa::{DerefMode, Op};
use primitives::field::{F128, g_pow};

/// The first iteration's frame, above `LOW` so that the segments own their frames.
const FRAME0: usize = 70_000;
const STRIDE: usize = 64;
const ITERATIONS: usize = 40;
/// Main's frame, past every iteration's: the run's allocations start there.
const MAIN: usize = FRAME0 + (ITERATIONS + 2) * STRIDE;
/// The loop's first pc, whose hints pop the stream `loop`.
const MARKER: u32 = 8;
/// Cells an iteration allocates, in two buffers.
const HEAP: usize = 16;

fn address(i: usize) -> F128 {
    F128::from(g_pow(i))
}

fn word(v: u64) -> F128 {
    F128::new(v, 0)
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// One iteration's witness: the cells its four `DEREF`s equate with frame cells 24..28, the
/// cells three heap hints write, and their values.
#[derive(Clone)]
struct Iteration {
    ptrs: [usize; 4],
    wptrs: [usize; 3],
    values: [u64; 3],
}

/// The loop over `iterations`. With `heap`, each iteration allocates two buffers of
/// `HEAP / 2` cells, at its first instruction and after its `DEREF`s.
fn program(iterations: &[Iteration], heap: bool) -> Program {
    let prefix = [Op::Set { o: 311, k: word(9) }, Op::Set { o: 312, k: word(7) }];
    program_with(iterations, heap, prefix)
}

/// [`program`], with `prefix` run in main's frame before the loop.
fn program_with(iterations: &[Iteration], heap: bool, prefix: [Op; 2]) -> Program {
    let seven = word(7);
    let mut code = vec![Op::Set { o: 7, k: F128::ZERO }; 64];
    code[0] = Op::Set {
        o: FRAME0 as u32,
        k: seven,
    };
    code[1] = Op::Set { o: 2, k: F128::ONE };
    code[2] = Op::Set {
        o: 3,
        k: address(MARKER as usize),
    };
    code[3] = Op::Set {
        o: 4,
        k: address(FRAME0),
    };
    code[4] = Op::Set { o: 310, k: seven };
    code[5..7].copy_from_slice(&prefix);
    code[7] = Op::Jump { oc: 2, od: 3, of: 4 };
    code[8] = Op::Set { o: 4, k: seven };
    code[9] = Op::Set { o: 5, k: F128::ONE };
    // Back-solve the next frame's first cell: the loop's one argument.
    code[10] = Op::Mul {
        a: STRIDE as u32,
        b: 5,
        c: 4,
    };
    code[11] = Op::Deref {
        o1: 1,
        o2: 0,
        o3: 0,
        mode: DerefMode::Cell,
    };
    for k in 0..4u32 {
        code[12 + k as usize] = Op::Deref {
            o1: 8 + k,
            o2: 0,
            o3: 24 + k,
            mode: DerefMode::Cell,
        };
    }
    code[16] = Op::Set {
        o: 6,
        k: address(MARKER as usize),
    };
    code[17] = Op::Jump { oc: 3, od: 6, of: 2 };
    // The exit path, in the last iteration's frame.
    code[18] = Op::Set { o: 7, k: address(63) };
    code[19] = Op::Set {
        o: 30,
        k: address(5000),
    };
    code[20] = Op::Deref {
        o1: 30,
        o2: 0,
        o3: 31,
        mode: DerefMode::Cell,
    };
    code[21] = Op::Jump { oc: 5, od: 7, of: 5 };
    let mut marker = vec![RHint::WitnessStack {
        name: "loop".into(),
        base: 1,
        len: 3,
    }];
    if heap {
        marker.push(RHint::Alloc {
            ptr: 20,
            size: HEAP as u32 / 2,
        });
    }
    marker.push(RHint::WitnessStack {
        name: "ptrs".into(),
        base: 8,
        len: 4,
    });
    marker.push(RHint::WitnessStack {
        name: "wptrs".into(),
        base: 16,
        len: 3,
    });
    let write = |q: u32| RHint::WitnessHeap {
        name: format!("v{q}"),
        ptr: 16 + q,
        lo: 0,
        len: 1,
    };
    let mut hints = HashMap::from([
        (MARKER, marker),
        (12, vec![write(0)]),
        (14, vec![write(1)]),
        (15, vec![write(2)]),
    ]);
    if heap {
        hints.insert(
            16,
            vec![RHint::Alloc {
                ptr: 21,
                size: HEAP as u32 / 2,
            }],
        );
    }
    let mut program = Program::assemble(code, hints, MAIN as u32);
    let n = iterations.len();
    let mut lanes = Vec::new();
    let (mut ptrs, mut wptrs) = (Vec::new(), Vec::new());
    let mut values = [Vec::new(), Vec::new(), Vec::new()];
    for (i, it) in iterations.iter().enumerate() {
        lanes.push(vec![
            address(1000 + i),
            address(FRAME0 + (i + 1) * STRIDE),
            if i + 1 < n { F128::ONE } else { F128::ZERO },
        ]);
        ptrs.push(it.ptrs.iter().map(|&p| address(p)).collect());
        wptrs.push(it.wptrs.iter().map(|&p| address(p)).collect());
        for (q, v) in values.iter_mut().enumerate() {
            v.push(vec![word(it.values[q])]);
        }
    }
    program.set_witness("loop", lanes);
    program.set_witness("ptrs", ptrs);
    program.set_witness("wptrs", wptrs);
    for (q, v) in values.into_iter().enumerate() {
        program.set_witness(format!("v{q}"), v);
    }
    program
}

/// How the segmented run differs from the serial one, if it does, and whether it ran the
/// loop as segments.
fn divergence(program: &Program) -> (Option<String>, bool) {
    let input = [F128::ZERO; 2];
    let serial = program.execute(input);
    let segmented = program.execute_in_segments(input, &["loop"], 0);
    match (serial, segmented) {
        (Ok(s), Ok((e, _, loops))) => (s.mismatch(&e).map(|d| format!("differs from serial: {d}")), loops > 0),
        (Err(s), Err(e)) => ((s != e).then(|| format!("error `{e}`, serial error `{s}`")), false),
        (Ok(_), Err(e)) => (Some(format!("error `{e}`, the serial run succeeds")), false),
        (Err(s), Ok((_, _, loops))) => (Some(format!("succeeds, serial error `{s}`")), loops > 0),
    }
}

/// Whether the pool can run a loop as segments: with one worker every loop runs serially, so
/// a positive control that a loop ran as segments holds only with two or more.
fn segmenting() -> bool {
    parallel::num_threads() >= 2
}

/// Pointers and writes to cells of the iteration's own, in a low range no other iteration
/// reaches.
fn benign(i: usize) -> Iteration {
    Iteration {
        ptrs: std::array::from_fn(|k| 2000 + 16 * i + k),
        wptrs: std::array::from_fn(|k| 2008 + 16 * i + k),
        values: [7; 3],
    }
}

/// Invariant: a value the merge stores into an own cell reaches the own cells the segment
/// linked to it. Iteration 3 writes 7 to low cell 300; iteration 30 links `fp+25` with
/// `fp+24`, then cell 300 with `fp+25`, so serially `fp+24` holds 7. Iterations 3 and 30
/// fall in different segments whenever the loop runs as segments.
#[test]
fn own_link_carries_a_merged_value() {
    let mut iterations: Vec<_> = (0..ITERATIONS).map(benign).collect();
    iterations[3].wptrs[0] = 300;
    let fp = FRAME0 + 30 * STRIDE;
    iterations[30].ptrs[..2].copy_from_slice(&[fp + 25, 300]);
    let program = program(&iterations, false);
    let serial = program.execute([F128::ZERO; 2]).expect("the serial run succeeds");
    assert_eq!(serial.mem.word(fp + 24), word(7));
    assert_eq!(divergence(&program), (None, segmenting()));
}

/// Invariant: a segmented run stops at the step limit where the serial run stops, or completes
/// as it does, wherever the limit falls: every limit from the first step to one past the
/// last, with and without allocations. Positive control: some limits leave the loop running
/// as segments.
#[test]
fn every_step_limit_is_serial() {
    for heap in [false, true] {
        let iterations: Vec<_> = (0..ITERATIONS).map(benign).collect();
        let mut program = program(&iterations, heap);
        let total = program
            .execute([F128::ZERO; 2])
            .expect("the serial run succeeds")
            .cycles;
        let mut segmented = 0;
        for limit in 1..=total + 1 {
            program.set_step_limit(limit);
            let (d, seg) = divergence(&program);
            segmented += usize::from(seg);
            assert_eq!(d, None, "heap {heap}, limit {limit} of {total}");
        }
        assert!(segmented > 0 || !segmenting(), "heap {heap}: no limit ran as segments");
    }
}

/// Random pointers into the iteration's own low cells, low cells every iteration shares, its
/// own frame, `310` (written before the loop) and the next iteration's frame; with `heap`, a
/// quarter of the writes land in its first buffer. A value is rarely 9 rather than 7, for a
/// write-once conflict.
fn random_iterations(rng: &mut Rng, heap: bool) -> Vec<Iteration> {
    (0..ITERATIONS)
        .map(|i| {
            let fp = FRAME0 + i * STRIDE;
            let pick = |rng: &mut Rng, slot: usize| -> usize {
                match rng.below(10) {
                    0..=2 => 2000 + 16 * i + slot,
                    3..=5 => 300 + rng.below(4) as usize,
                    6..=7 => fp + 24 + rng.below(6) as usize,
                    8 => 310,
                    // Away from the loop's ends, whose frames nothing predicts.
                    _ if i >= 2 && i + 2 < ITERATIONS => fp + STRIDE + 40 + rng.below(2) as usize,
                    _ => 2000 + 16 * i + slot,
                }
            };
            Iteration {
                ptrs: std::array::from_fn(|k| pick(rng, k)),
                wptrs: std::array::from_fn(|k| {
                    if heap && rng.below(4) == 0 {
                        MAIN + i * HEAP + rng.below(HEAP as u64 / 2) as usize
                    } else {
                        pick(rng, 8 + k)
                    }
                }),
                values: std::array::from_fn(|_| if rng.below(400) == 0 { 9 } else { 7 }),
            }
        })
        .collect()
}

/// Invariant: a segmented run's execution (memory image, counts, write states, rows) or error
/// is the serial run's, whatever links and stores join the segments' own cells, the cells
/// every iteration shares and the next segment's frame. Positive control: some cases run as
/// segments and some fail serially.
#[test]
fn random_links_and_stores_merge_as_serial() {
    const CASES: u64 = 2000;
    let (mut segmented, mut failing, mut diverging) = (0, 0, Vec::new());
    for seed in 1..=CASES {
        let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
        let heap = seed % 2 == 0;
        let program = program(&random_iterations(&mut rng, heap), heap);
        failing += usize::from(program.execute([F128::ZERO; 2]).is_err());
        let (d, seg) = divergence(&program);
        segmented += usize::from(seg);
        if let Some(d) = d {
            diverging.push(format!("seed {seed}: {d}"));
        }
    }
    assert!(
        diverging.is_empty(),
        "{} of {CASES} cases diverge: {diverging:#?}",
        diverging.len()
    );
    assert!(
        segmented > CASES as usize / 20 || !segmenting(),
        "{segmented} of {CASES} cases ran as segments"
    );
    assert!(failing > 0, "no case fails serially");
}

/// Invariant: a pointer past what the serial run's g-power table covers at that instruction
/// gives the serial run's result: a segment covers nothing, so the run's coverage stays the
/// serial run's. Iteration 20 points three cells into its second buffer before allocating it:
/// through a `DEREF` and through a hint's heap pointer.
#[test]
fn pointer_past_serial_coverage_matches_serial() {
    let target = MAIN + 20 * HEAP + HEAP / 2 + 3;
    let found = [true, false].map(|deref| {
        let mut iterations: Vec<_> = (0..ITERATIONS).map(benign).collect();
        if deref {
            iterations[20].ptrs[2] = target;
        } else {
            iterations[20].wptrs[1] = target;
        }
        let program = program(&iterations, true);
        divergence(&program)
    });
    assert_eq!(found, [(None, false), (None, false)]);
}

/// Positive control for the above: the plain loop runs as segments and matches, with and
/// without allocations.
#[test]
fn plain_loop_runs_as_segments() {
    for heap in [false, true] {
        let iterations: Vec<_> = (0..ITERATIONS).map(benign).collect();
        assert_eq!(
            divergence(&program(&iterations, heap)),
            (None, segmenting()),
            "heap {heap}"
        );
    }
}

/// Invariant: a segment sees its own cells as memory holds them when the loop starts, links
/// included, so a run whose prefix writes or links a later iteration's frame still gives the
/// serial result. Each case runs every placement in the segmented range, so that some fall
/// on a segment's first frame whatever the pool's size. Writing the argument cell of a frame
/// makes the iteration before it conflict serially at its `MUL`; linking an own cell with a
/// low cell carries the iteration's later write to the low cell. Both are checked on the
/// serial run, so that the prefix cannot stop mattering unnoticed.
#[test]
fn prefix_touching_own_cells_is_serial() {
    let iterations: Vec<_> = (0..ITERATIONS).map(benign).collect();
    let mut found = Vec::new();
    for b in 3..ITERATIONS - 1 {
        let fp = FRAME0 + b * STRIDE;
        let argument = [
            Op::Set { o: 311, k: word(9) },
            Op::Set {
                o: fp as u32,
                k: word(9),
            },
        ];
        let mut linked = iterations.clone();
        linked[b].wptrs[0] = fp + 25;
        let link = [
            Op::Set {
                o: 311,
                k: address(fp + 25),
            },
            Op::Deref {
                o1: 311,
                o2: 0,
                o3: 305,
                mode: DerefMode::Cell,
            },
        ];
        let argument = program_with(&iterations, false, argument);
        let serial = argument
            .execute([F128::ZERO; 2])
            .err()
            .expect("the argument case fails");
        assert_eq!(serial.pc, 10, "{serial}");
        let link = program_with(&linked, false, link);
        let serial = link.execute([F128::ZERO; 2]).expect("the link case runs");
        assert_eq!(serial.mem.word(305), word(7), "frame {b}");
        for (name, program) in [("argument", argument), ("link", link)] {
            if let (Some(d), _) = divergence(&program) {
                found.push(format!("{name} in frame {b}: {d}"));
            }
        }
    }
    assert!(found.is_empty(), "{found:#?}");
}
