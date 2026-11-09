//! Parallel execution of a loop's independent iterations, as segments whose merged result is
//! the serial run's, row for row and cell for cell.
//!
//! The caller names witness streams that a loop pops once per iteration, at the iteration's
//! first instruction ([`Program::segment_streams`]). The run goes serially to the start of
//! iteration 2, predicts every later iteration's start (pc, frame, `next_free`, hint
//! cursors, and the arguments the previous iteration wrote into its frame) as linear in the
//! iteration index from iterations 1 and 2, runs iterations `2..n-1` as one segment per pool
//! thread, merges them in serial order, and goes on serially from iteration `n-1`.
//!
//! A segment owns its iterations' frames and heap at or above [`LOW`], and writes
//! them in place. Every other cell it reaches goes through its own overlay, which starts
//! from memory as the dispatch found it (nothing writes those cells during the dispatch),
//! from the predicted arguments, or empty for the next segment's first frame. Below
//! [`LOW`] a range check can reach any cell, so no segment owns one there. A
//! loop whose own cells memory holds links from, or whose later segments' first frames
//! memory holds anything in, runs serially: a segment would not see either.
//!
//! The merge replays each segment against memory in serial order. It checks that the
//! segment ended where the next one was predicted to start; turns the counts it read on
//! overlay cells into serial ones by the counts memory held at its turn; adds the links
//! its own cells still hold to memory's, then replays its `DEREF` links and stores on
//! overlay cells through the serial interpreter's own `link` and `put`, which spread a value
//! through linked cells, its own included, and report a write-once conflict as the serial
//! run would; and checks that no cell the segment saw unwritten had been written before its
//! turn, and that every pointer it resolved past its own g-power coverage lies within the
//! coverage the serial run had at its turn. A segment can link where the serial run would
//! have copied a value an earlier iteration wrote, as a range check's touch cells do
//! through the low cells. The final image agrees either way: once every link is in place, a
//! class of linked cells holds the one value any of its cells was given, whatever order the
//! links and stores came in, and two values conflict in any order. Nothing the segment
//! decided can depend on the difference: reading a cell it did not see written abandons it,
//! as does testing or hinting from a linked cell of its own, and testing or hinting from
//! any other records it unwritten, which the merge checks. Any failed check, a fault, or a
//! segment touching another's cells sends the whole run back to a serial rerun.

use std::hash::{BuildHasherDefault, Hasher};
use std::ops::Range;

use super::Program;
use super::execute::{Cell, ExecError, Fault, Machine, Mem, PAGE, PAGE_LOG, Plan, RowSet, Run, State, Stop, Table};
use super::hints::{GPow, MAX_CELLS};
use primitives::field::{F64, F128};
use primitives::hash::STATE_LANES;

/// The segments' maps, keyed by cells and g-powers: a segment probes its overlay on most
/// accesses to a cell it does not own, and the merge probes them per row, where SipHash's
/// resistance to chosen keys buys nothing.
type HashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<Fold>>;

/// One folded multiply per key.
#[derive(Default)]
struct Fold(u64);

impl Hasher for Fold {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b.into());
        }
    }
    #[inline(always)]
    fn write_u32(&mut self, k: u32) {
        self.write_u64(k.into());
    }
    #[inline(always)]
    fn write_u64(&mut self, k: u64) {
        let p = u128::from(self.0 ^ k) * 0x9e37_79b9_7f4a_7c15;
        self.0 = p as u64 ^ (p >> 64) as u64;
    }
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }
}

/// Cells a range check of a bound up to `2^16` reaches, and so no segment owns. A range check
/// past it that lands on another segment's cell abandons the segment, as any such touch does, so
/// `LOW` decides how often segments succeed, never what a run computes.
const LOW: u32 = 1 << 16;
/// Fewest iterations a segment runs: fewer cost more to merge than they save.
const MIN_ITERATIONS: usize = 8;
/// Fewest iterations of a loop that runs as segments: the two that predict the rest, the last,
/// and two segments.
const MIN_LOOP: usize = 3 + 2 * MIN_ITERATIONS;

impl Program {
    /// Whether a run of the program may execute a loop as segments: one of its segment streams
    /// holds enough iterations. What the run itself decides is final.
    pub(crate) fn may_segment(&self) -> bool {
        self.segment_streams
            .iter()
            .any(|name| self.witness.get(name).is_some_and(|entries| entries.len() >= MIN_LOOP))
    }
}

type Lanes = [u64; STATE_LANES];

fn same(a: &Cell, b: &Cell) -> bool {
    a.word == b.word && a.count == b.count && a.state == b.state
}

fn pow(mut b: F128, mut e: usize) -> F128 {
    let mut r = F128::ONE;
    while e != 0 {
        if e & 1 == 1 {
            r *= b;
        }
        b = b * b;
        e >>= 1;
    }
    r
}

/// An iteration's start: the pc is the loop's.
#[derive(Clone, PartialEq, Eq)]
struct Start {
    fp: u32,
    next_free: u32,
    cursors: Vec<usize>,
}

/// Every iteration's start, linear in its index from iteration 2's.
struct Prediction {
    at2: Start,
    stride: u32,
    dnf: u32,
    dcur: Vec<usize>,
    /// The frame cells an iteration finds written when it starts, at their offsets: iteration
    /// 2's cell, and the ratio of its word to iteration 1's.
    args: Vec<(u32, Cell, F128)>,
    /// A test's misprediction: added to one cursor of the last segment's start.
    skew: (usize, usize, usize),
    /// `KECCAK` rows iteration 1 ran, to size a segment's buffer of their lanes.
    keccaks: usize,
}

impl Prediction {
    fn start(&self, k: usize) -> Start {
        let j = k - 2;
        let mut cursors: Vec<usize> = self
            .at2
            .cursors
            .iter()
            .zip(&self.dcur)
            .map(|(c, d)| c + j * d)
            .collect();
        let (at, stream, by) = self.skew;
        if k == at {
            cursors[stream] += by;
        }
        Start {
            fp: self.at2.fp + j as u32 * self.stride,
            next_free: self.at2.next_free + j as u32 * self.dnf,
            cursors,
        }
    }

    /// Iteration `k`'s argument cells.
    fn args(&self, k: usize) -> HashMap<u32, Cell> {
        let fp = self.start(k).fp;
        self.args
            .iter()
            .map(|&(o, cell, ratio)| {
                let word = if ratio == F128::ONE {
                    cell.word
                } else {
                    cell.word * pow(ratio, k - 2)
                };
                (fp + o, Cell { word, ..cell })
            })
            .collect()
    }

    /// The cells iterations `a..b` own: their frames and heap, at and above `LOW`.
    fn own(&self, a: usize, b: usize) -> [Range<u32>; 2] {
        let (start, end) = (self.start(a), self.start(b));
        [
            start.fp.max(LOW)..end.fp.max(LOW),
            start.next_free.max(LOW)..end.next_free.max(LOW),
        ]
    }
}

fn start_of(run: &Run<'_>) -> Start {
    Start {
        fp: run.fp,
        next_free: run.next_free,
        cursors: run.streams.iter().map(|s| s.next).collect(),
    }
}

fn touched(c: &Cell) -> bool {
    c.state != State::Unwritten || c.count != 0
}

fn predict(m: &Mem<'_>, at1: Start, frame1: &[Cell], at2: Start, s: usize) -> Option<Prediction> {
    let stride = at2.fp.checked_sub(at1.fp).filter(|&d| d != 0)?;
    let dnf = at2.next_free.checked_sub(at1.next_free)?;
    let dcur: Vec<usize> = at2
        .cursors
        .iter()
        .zip(&at1.cursors)
        .map(|(b, a)| b.checked_sub(*a))
        .collect::<Option<_>>()?;
    if dcur[s] != 1 || stride as usize > frame1.len() {
        return None;
    }
    let mut args = Vec::new();
    for o in 0..stride {
        let (c1, c2) = (frame1[o as usize], m.cell(at2.fp + o));
        let arg = c1.state == State::Written && c2.state == State::Written && c1.count == c2.count;
        if at1.fp + o < LOW {
            // A range check may have touched the cell; what is not an argument is memory's.
            if !arg {
                continue;
            }
        } else if touched(&c1) != touched(&c2) || (touched(&c2) && !arg) {
            return None;
        } else if !touched(&c2) {
            continue;
        }
        let ratio = if c1.word == c2.word {
            F128::ONE
        } else if c1.word.is_zero() {
            return None;
        } else {
            c2.word * c1.word.inv()
        };
        args.push((o, c2, ratio));
    }
    Some(Prediction {
        at2,
        stride,
        dnf,
        dcur,
        args,
        skew: (0, 0, 0),
        keccaks: 0,
    })
}

/// Run the iterations `streams` mark as segments, from `run`'s start, leaving `run` where
/// serial execution goes on: `Some` of the number of loops that ran as segments, `None` if a
/// segment's check failed and the run has to start over serially. An error is the serial
/// run's own, from a part it ran serially.
#[allow(clippy::too_many_arguments)]
pub(super) fn run<'w>(
    program: &Program,
    plan: &Plan<'_>,
    m: &mut Mem<'_>,
    run: &mut Run<'w>,
    step: &mut dyn FnMut(&Lanes),
    streams: &[&str],
    ending_pc: u32,
    skew: usize,
) -> Result<Option<usize>, ExecError> {
    let mut loops = 0;
    let span = program
        .prog
        .iter()
        .map(|op| op.max_offset() as usize)
        .max()
        .unwrap_or(0)
        + 1;
    for name in streams {
        let Some(s) = run.streams.iter().position(|st| st.name() == *name) else {
            continue;
        };
        let n = run.streams[s].len();
        if run.streams[s].next != 0 || n < MIN_LOOP {
            continue;
        }
        let Some(pc) = plan.marker(s as u32) else {
            continue;
        };
        let stop = |pops| Stop {
            pc,
            stream: s as u32,
            pops,
        };
        if !program.walk(plan, m, run, step, ending_pc, stop(1))? {
            return Ok(Some(loops));
        }
        let at1 = start_of(run);
        let keccaks = run.rows.keccak.len();
        let frame1: Vec<Cell> = (0..span as u32).map(|o| m.cell(at1.fp + o)).collect();
        if !program.walk(plan, m, run, step, ending_pc, stop(2))? {
            return Ok(Some(loops));
        }
        let at2 = start_of(run);
        let Some(mut pred) = predict(m, at1, &frame1, at2.clone(), s) else {
            continue;
        };
        pred.keccaks = run.rows.keccak.len() - keccaks;
        let iterations = n - 3;
        let segments = (iterations / MIN_ITERATIONS).min(parallel::num_threads());
        if segments < 2 {
            continue;
        }
        let bounds: Vec<usize> = (0..=segments).map(|c| 2 + c * iterations / segments).collect();
        if skew != 0
            && let Some(other) = (0..pred.dcur.len()).find(|&i| i != s && pred.dcur[i] != 0)
        {
            pred.skew = (bounds[segments - 1], other, skew);
        }
        // The segments write their own cells in place and index the shared g-power table:
        // both have to reach the last iteration's frame and heap before they start, the
        // table without covering more for the run. A prediction past the address space is
        // no prediction.
        let j = (n - 3) as u64;
        let reach = (at2.next_free as u64 + j * pred.dnf as u64).max(at2.fp as u64 + (j + 1) * pred.stride as u64);
        if reach >= MAX_CELLS as u64 {
            continue;
        }
        if !unseen_is_empty(m, &pred, &bounds) {
            continue;
        }
        m.reserve(reach as usize);
        m.g.reserve(reach as usize);
        if !dispatch(program, plan, m, run, step, &pred, &bounds, pc, s as u32, ending_pc) {
            return Ok(None);
        }
        loops += 1;
    }
    Ok(Some(loops))
}

/// Whether memory holds nothing about the segments' own cells that a segment would not see:
/// links from them, which live in memory's map and not the segment's, or anything in a
/// later segment's first frame, which the segment before it takes as unwritten and which
/// its own predicted arguments overwrite.
fn unseen_is_empty(m: &Mem<'_>, pred: &Prediction, bounds: &[usize]) -> bool {
    let own = pred.own(bounds[0], bounds[bounds.len() - 1]);
    if m.links.keys().any(|k| own.iter().any(|r| r.contains(k))) {
        return false;
    }
    bounds[1..bounds.len() - 1].iter().all(|&b| {
        let fp = pred.start(b).fp;
        (fp.max(LOW)..fp + pred.stride).all(|c| !touched(&m.cell(c)))
    })
}

/// The pages of memory, shared by the segments: each writes only its own cells.
#[derive(Clone, Copy)]
struct Base<'a>(&'a [*mut Cell]);
// SAFETY: segments write disjoint cells through it, and read only cells no segment writes
// (`Segment::over`).
unsafe impl Sync for Base<'_> {}
unsafe impl Send for Base<'_> {}

impl Base<'_> {
    /// Cell `c`, if its page exists.
    #[inline(always)]
    fn ptr(&self, c: u32) -> Option<*mut Cell> {
        let c = c as usize;
        self.0.get(c >> PAGE_LOG).map(|p| p.wrapping_add(c & (PAGE - 1)))
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch<'w>(
    program: &Program,
    plan: &Plan<'_>,
    m: &mut Mem<'_>,
    run: &mut Run<'w>,
    step: &mut dyn FnMut(&Lanes),
    pred: &Prediction,
    bounds: &[usize],
    pc: u32,
    s: u32,
    ending_pc: u32,
) -> bool {
    let segments = bounds.len() - 1;
    // Where the segments' own cells begin: below it, every cell is the run's from before the dispatch.
    let par_lo = (0..segments)
        .flat_map(|c| pred.own(bounds[c], bounds[c + 1]))
        .filter(|r| !r.is_empty())
        .map(|r| r.start)
        .min()
        .unwrap_or(u32::MAX);
    let pages = m.page_ptrs();
    let base = Base(&pages);
    let g = &m.g;
    let streams = &run.streams;
    let len = run.bytecode_count.len();
    let outs: Vec<Option<Out<'w>>> = crate::stage!("Run segments", || {
        parallel::map_collect(segments, |c| {
            let (a, b) = (bounds[c], bounds[c + 1]);
            let (start, end) = (pred.start(a), pred.start(b));
            let mut seeds = pred.args(a);
            let own = pred.own(a, b).map(|r| (r.start, r.end));
            // The segment's first frame, at and above `LOW`, is its own: write its arguments
            // there. The first segment's are there already.
            seeds.retain(|&cell, seed| {
                if cell < LOW {
                    return true;
                }
                if c != 0 {
                    // SAFETY: the cell is this segment's own, in a reserved page.
                    unsafe { *base.ptr(cell).expect("a reserved page") = *seed };
                }
                false
            });
            let mut seg = Segment {
                base,
                own,
                next: (end.fp, end.fp + pred.stride),
                par_lo,
                seeds: &seeds,
                overlay: HashMap::default(),
                links: HashMap::default(),
                events: Vec::new(),
                fixups: Vec::new(),
                pending: Vec::new(),
                nbump: 0,
                used: 0,
                counts: 0,
                abandoned: false,
                g,
                cover: g.covered(),
                need: 0,
                cache: HashMap::default(),
                program,
                dbg_pc: pc,
                dbg_hint: None,
                scratch: Cell::UNWRITTEN,
            };
            let mut r = Run {
                pc,
                fp: start.fp,
                next_free: start.next_free,
                steps: 0,
                streams: streams
                    .iter()
                    .zip(&start.cursors)
                    .map(|(st, &next)| {
                        let mut st = st.clone();
                        st.next = next;
                        st
                    })
                    .collect(),
                bytecode_count: vec![0; len],
                rows: RowSet::new(),
                prof: None,
            };
            let mut lanes: Vec<Lanes> = Vec::with_capacity((b - a) * pred.keccaks);
            let stop = Stop { pc, stream: s, pops: b };
            let ok = matches!(
                program.walk(plan, &mut seg, &mut r, &mut |l| lanes.push(*l), ending_pc, stop),
                Ok(true)
            );
            let Segment {
                overlay,
                links,
                events,
                fixups,
                used,
                counts,
                abandoned,
                cover,
                need,
                ..
            } = seg;
            (ok && !abandoned).then_some(Out {
                run: r,
                overlay,
                links,
                events,
                fixups,
                used,
                counts,
                cover,
                need,
                lanes,
                seeds,
                own,
            })
        })
    });
    crate::stage!("Merge segments", || {
        for (c, out) in outs.into_iter().enumerate() {
            let Some(mut out) = out else {
                return false;
            };
            let lanes = std::mem::take(&mut out.lanes);
            let next = (c + 1 < segments).then(|| pred.args(bounds[c + 1]));
            if !merge(m, run, out, &pred.start(bounds[c + 1]), next, pred.stride) {
                return false;
            }
            // A merged segment ran as the serial run would have: the Keccak witness workers
            // take its steps while the later segments merge.
            for l in &lanes {
                step(l);
            }
        }
        true
    })
}

/// A segment's record, for the merge.
struct Out<'w> {
    run: Run<'w>,
    overlay: HashMap<u32, Over>,
    links: HashMap<u32, Vec<u32>>,
    events: Vec<Event>,
    fixups: Vec<Fixup>,
    used: usize,
    counts: u32,
    /// The segment's `cover` and `need`.
    cover: usize,
    need: usize,
    lanes: Vec<Lanes>,
    /// Its predicted arguments below `LOW`, which live in the overlay.
    seeds: HashMap<u32, Cell>,
    own: [(u32, u32); 2],
}

/// An overlay cell and the count it started from.
struct Over {
    cell: Cell,
    init: u32,
}

/// What a segment did to an overlay cell that the merge replays, in the segment's order.
enum Event {
    /// A store wrote it.
    Put(u32, F128),
    /// A `DEREF` equated these two cells, one an overlay cell.
    Link(u32, u32),
    /// A hint or an instruction found it unwritten.
    Unwritten(u32),
}

/// A count a row read on an overlay cell: the row's table and index, the count's place
/// among the counts the row read, and the cell.
struct Fixup {
    table: Table,
    row: u32,
    field: u8,
    cell: u32,
}

fn merge(
    m: &mut Mem<'_>,
    run: &mut Run<'_>,
    out: Out<'_>,
    end: &Start,
    next: Option<HashMap<u32, Cell>>,
    stride: u32,
) -> bool {
    if start_of(&out.run) != *end || out.need > m.g.covered() {
        return false;
    }
    for (&cell, seed) in &out.seeds {
        if !same(&m.cell(cell), seed) {
            return false;
        }
    }
    // The next segment's first frame at and above `LOW` is that segment's own, already in
    // memory: what this one wrote there has to be the arguments it assumed.
    let mut held = 0..0;
    if let Some(next) = next {
        held = end.fp.max(LOW)..end.fp + stride;
        for cell in held.clone() {
            let mine = out.overlay.get(&cell).map_or(Cell::UNWRITTEN, |o| o.cell);
            if !same(&mine, next.get(&cell).unwrap_or(&Cell::UNWRITTEN)) {
                return false;
            }
        }
    }
    let skip = |c: u32| held.contains(&c);
    let mut delta: HashMap<u32, u32> = HashMap::with_capacity_and_hasher(out.overlay.len(), Default::default());
    for (&cell, o) in &out.overlay {
        if skip(cell) {
            continue;
        }
        let Some(d) = m.cell(cell).count.checked_sub(o.init) else {
            return false;
        };
        delta.insert(cell, d);
    }
    // The segment's live links from its own cells go in before its events: a value an event
    // stores into an own cell spreads through them, as it would have serially.
    let own = |c: u32| out.own.iter().any(|&(lo, hi)| (lo..hi).contains(&c));
    for (k, v) in out.links {
        if own(k) {
            m.links.entry(k).or_default().extend(v);
        }
    }
    for e in &out.events {
        let ok = match *e {
            Event::Put(c, v) => skip(c) || m.put(c, v).is_ok(),
            Event::Link(a2, a3) => !skip(a2) && !skip(a3) && m.deref_cell(a2, a3).is_ok(),
            Event::Unwritten(c) => skip(c) || !m.is_written(c),
        };
        if !ok {
            return false;
        }
    }
    for (&cell, o) in &out.overlay {
        // An instruction's access, which covers the cell as the serial run's would.
        if o.cell.count != o.init
            && let Some(&d) = delta.get(&cell)
        {
            m.ensure(cell as usize);
            let c = m.at(cell);
            c.count = o.cell.count + d;
            m.counts = m.counts.max(o.cell.count + d);
        }
    }
    m.counts = m.counts.max(out.counts);
    m.g.grow_to(out.cover - 1);
    if out.used > m.used {
        m.ensure(out.used - 1);
    }

    let Run {
        steps,
        bytecode_count,
        mut rows,
        streams,
        ..
    } = out.run;
    for f in &out.fixups {
        let d = delta.get(&f.cell).copied().unwrap_or(0);
        let (row, field) = (f.row as usize, f.field as usize);
        match f.table {
            Table::Xor | Table::Mul => {
                let r = if matches!(f.table, Table::Xor) {
                    rows.xor.get_mut(row)
                } else {
                    rows.mul.get_mut(row)
                };
                *[&mut r.ra, &mut r.rb, &mut r.rc][field] += d;
            }
            Table::Set => rows.set.get_mut(row).r += d,
            Table::Deref => {
                let r = rows.deref.get_mut(row);
                *[&mut r.r1, &mut r.r2, &mut r.r3][field] += d;
            }
            Table::Jump => {
                let r = rows.jump.get_mut(row);
                *[&mut r.rc, &mut r.rd, &mut r.rf][field] += d;
            }
            Table::Keccak => rows.keccak.get_mut(row).r[field] += d,
        }
    }
    let bc = &run.bytecode_count;
    let to = &mut run.rows;
    for mut r in rows.xor.into_iter() {
        r.bytecode_read += bc[r.pc as usize];
        to.xor.push(r);
    }
    for mut r in rows.mul.into_iter() {
        r.bytecode_read += bc[r.pc as usize];
        to.mul.push(r);
    }
    for mut r in rows.set.into_iter() {
        r.bytecode_read += bc[r.pc as usize];
        to.set.push(r);
    }
    for mut r in rows.deref.into_iter() {
        r.bytecode_read += bc[r.pc as usize];
        to.deref.push(r);
    }
    for mut r in rows.jump.into_iter() {
        r.bytecode_read += bc[r.pc as usize];
        to.jump.push(r);
    }
    for mut r in rows.keccak.into_iter() {
        r.bytecode_read += bc[r.pc as usize];
        to.keccak.push(r);
    }
    for (total, add) in run.bytecode_count.iter_mut().zip(&bytecode_count) {
        *total += add;
    }
    run.steps += steps;
    run.fp = end.fp;
    run.next_free = end.next_free;
    for (to, from) in run.streams.iter_mut().zip(&streams) {
        to.next = from.next;
    }
    true
}

/// A segment's machine: its own cells in place, every other one in its overlay.
struct Segment<'a> {
    base: Base<'a>,
    /// Its frames and its heap, at and above `LOW`.
    own: [(u32, u32); 2],
    /// The next segment's first frame, which that segment writes.
    next: (u32, u32),
    /// Where the segments' own cells begin: nothing at or above it is the run's before the dispatch.
    par_lo: u32,
    seeds: &'a HashMap<u32, Cell>,
    overlay: HashMap<u32, Over>,
    links: HashMap<u32, Vec<u32>>,
    events: Vec<Event>,
    fixups: Vec<Fixup>,
    /// The overlay counts the current row read, by their place in it.
    pending: Vec<(u8, u32)>,
    nbump: u8,
    used: usize,
    /// One past the highest count on an own cell.
    counts: u32,
    abandoned: bool,
    g: &'a GPow,
    /// Exponents the serial run covers at this point for certain: what it covered at the
    /// dispatch, grown as this segment grew it.
    cover: usize,
    /// One past the largest exponent this segment resolved beyond `cover`: the serial run
    /// has to have covered it before the segment's first iteration.
    need: usize,
    /// Exponents this segment resolved, which the shared table does not index.
    cache: HashMap<u64, u32>,
    program: &'a Program,
    dbg_pc: u32,
    dbg_hint: Option<&'static str>,
    scratch: Cell,
}

impl Segment<'_> {
    #[inline(always)]
    fn own(&self, c: u32) -> bool {
        (c >= self.own[0].0 && c < self.own[0].1) || (c >= self.own[1].0 && c < self.own[1].1)
    }

    #[inline(always)]
    fn slot(&mut self, c: u32) -> &mut Cell {
        if self.own(c) {
            // SAFETY: an own cell, in a reserved page, which no other segment reads or writes.
            unsafe { &mut *self.base.ptr(c).expect("a reserved page") }
        } else {
            self.over(c)
        }
    }

    fn over(&mut self, c: u32) -> &mut Cell {
        let next = c >= LOW && c >= self.next.0 && c < self.next.1;
        if !next && c >= LOW && c >= self.par_lo {
            // Another segment's cell, or past every segment's.
            self.abandoned = true;
            self.scratch = Cell::UNWRITTEN;
            return &mut self.scratch;
        }
        let (seeds, base) = (self.seeds, self.base);
        &mut self
            .overlay
            .entry(c)
            .or_insert_with(|| {
                let cell = match seeds.get(&c) {
                    Some(seed) => *seed,
                    None if next => Cell::UNWRITTEN,
                    // SAFETY: below `par_lo` or `LOW` and in no segment's frame: no segment
                    // writes it.
                    None => base.ptr(c).map_or(Cell::UNWRITTEN, |p| unsafe { *p }),
                };
                Over { cell, init: cell.count }
            })
            .cell
    }

    /// A hint or an instruction decides on cell `c`'s state.
    #[inline(always)]
    fn observe(&mut self, c: u32) -> Cell {
        let own = self.own(c);
        let cell = *self.slot(c);
        match (cell.state, own) {
            (State::Written, _) | (State::Unwritten, true) => {}
            // Whether the serial run would find a value here depends on other iterations.
            (State::Linked, true) => self.abandoned = true,
            (_, false) => self.events.push(Event::Unwritten(c)),
        }
        cell
    }

    #[inline(always)]
    fn store(&mut self, cell: u32, v: F128) -> Result<bool, ExecError> {
        let own = self.own(cell);
        let c = self.slot(cell);
        let s = c.state;
        if s == State::Written && c.word != v {
            let fault = Fault::Conflict {
                cell,
                had: c.word,
                new: v,
                hint: self.dbg_hint,
            };
            return Err(ExecError::new(self.program, self.dbg_pc, fault));
        }
        c.word = v;
        c.state = State::Written;
        if !own && s != State::Written {
            self.events.push(Event::Put(cell, v));
        }
        Ok(s == State::Linked)
    }

    #[cold]
    fn spread(&mut self, cell: u32, v: F128) -> Result<(), ExecError> {
        let mut todo = vec![cell];
        while let Some(c) = todo.pop() {
            for p in self.links.remove(&c).unwrap_or_default() {
                if self.store(p, v)? {
                    todo.push(p);
                }
            }
        }
        Ok(())
    }

    fn link(&mut self, a: u32, b: u32) {
        for (x, y) in [(a, b), (b, a)] {
            self.ensure(x as usize);
            let c = self.slot(x);
            if c.state == State::Unwritten {
                c.state = State::Linked;
            }
            self.links.entry(x).or_default().push(y);
        }
    }

    #[inline(always)]
    fn g_lookup(&self, x: F64) -> Option<u32> {
        self.g.lookup(x).or_else(|| self.cache.get(&x.0).copied())
    }
}

impl Machine for Segment<'_> {
    #[inline(always)]
    fn ensure(&mut self, idx: usize) {
        self.used = self.used.max(idx + 1);
    }
    #[inline(always)]
    fn written(&mut self, cell: u32) -> bool {
        self.observe(cell).state == State::Written
    }
    #[inline(always)]
    fn get(&mut self, cell: u32) -> F128 {
        self.observe(cell).word
    }
    #[inline(always)]
    fn read(&mut self, cell: u32) -> F128 {
        self.ensure(cell as usize);
        let c = *self.slot(cell);
        if c.state != State::Written {
            // An unwritten read is refused by every proof; the serial rerun reports it.
            self.abandoned = true;
        }
        c.word
    }
    #[inline(always)]
    fn put(&mut self, cell: u32, v: F128) -> Result<(), ExecError> {
        self.ensure(cell as usize);
        if self.store(cell, v)? {
            self.spread(cell, v)?;
        }
        Ok(())
    }
    #[inline(always)]
    fn bump(&mut self, cell: u32) -> u32 {
        self.ensure(cell as usize);
        let own = self.own(cell);
        let c = self.slot(cell);
        let count = c.count;
        c.count = count + 1;
        if own {
            self.counts = self.counts.max(count + 1);
        } else {
            self.pending.push((self.nbump, cell));
        }
        self.nbump += 1;
        count
    }
    #[inline(always)]
    fn put_counted(&mut self, cell: u32, v: F128) -> Result<u32, ExecError> {
        self.ensure(cell as usize);
        let linked = self.store(cell, v)?;
        let count = self.bump(cell);
        if linked {
            self.spread(cell, v)?;
        }
        Ok(count)
    }
    #[inline(always)]
    fn deref_cell(&mut self, a2: u32, a3: u32) -> Result<(), ExecError> {
        if !self.own(a2) || !self.own(a3) {
            self.events.push(Event::Link(a2, a3));
        }
        let c2 = *self.slot(a2);
        if c2.state == State::Written {
            return self.put(a3, c2.word);
        }
        let c3 = *self.slot(a3);
        if c3.state == State::Written {
            return self.put(a2, c3.word);
        }
        self.link(a2, a3);
        Ok(())
    }
    #[inline(always)]
    fn row(&mut self, table: Table, row: usize) {
        for (field, cell) in self.pending.drain(..) {
            self.fixups.push(Fixup {
                table,
                row: row as u32,
                field,
                cell,
            });
        }
        self.nbump = 0;
    }
    #[inline(always)]
    fn at_pc(&mut self, pc: u32) {
        self.dbg_pc = pc;
    }
    #[inline(always)]
    fn hint(&mut self, hint: Option<&'static str>) {
        self.dbg_hint = hint;
    }
    #[inline(always)]
    fn abandoned(&self) -> bool {
        self.abandoned
    }
    fn g_note(&mut self, k: usize) {
        self.g_grow(k);
        if !self.abandoned {
            self.cache.insert(self.g.pow(k).0, k as u32);
        }
    }
    #[inline(always)]
    fn g_grow(&mut self, upto: usize) {
        if upto >= self.g.reserved() {
            self.abandoned = true;
        }
        self.cover = self.cover.max(upto + 1);
    }
    #[inline(always)]
    fn g_pow(&mut self, k: usize) -> F64 {
        if k >= self.g.reserved() {
            self.abandoned = true;
            return F64::ONE;
        }
        self.g.pow(k)
    }
    #[inline(always)]
    fn g_log_below(&mut self, x: F64, bound: usize) -> Option<u32> {
        let k = match self.g_lookup(x) {
            Some(k) => ((k as usize) < bound).then_some(k)?,
            None => {
                let k = self.g.log_below_ro(x, bound)?;
                self.cache.insert(x.0, k);
                k
            }
        };
        // The serial run covers what it resolves. This segment covers nothing, so one it resolves past its own
        // growth, or past the run's before the segment's first iteration, is recorded, and the merge fails unless the
        // run had covered it by then: the run's coverage, and the g-power table an execution carries, stay the
        // serial run's.
        if k as usize >= self.cover {
            self.need = self.need.max(k as usize + 1);
        }
        Some(k)
    }
}

#[cfg(test)]
mod tests;
