//! The write-once execution interpreter: run the compiled program to produce
//! the final memory image and the per-opcode [`Trace`].

use std::collections::HashMap;

use super::hints::{BitsDest, GPow, MAX_CELLS, MAX_INDEX, RHint, bounded_dlog};
use super::*;
use primitives::{
    field::{F64, F128},
    pretty_f64, pretty_integer,
};

type Lanes = [u64; primitives::hash::STATE_LANES];

pub struct Execution {
    pub mem: Memory,     // data memory after the run, write-once (size cells, power of two)
    pub cycles: usize,   // number of instructions the run executed (trace length)
    pub mem_used: usize, // cells actually touched, before the power-of-two pad of `mem`
    /// Cells an instruction read before anything wrote them, so the value it read was
    /// ZERO here and prover-chosen in a proof: memory is a committed array and the bus
    /// only forces accesses to one address to *agree*, never that the address was
    /// written. `zkDSL.md` says don't; this is what says whether the
    /// emitted code did. A non-empty list means a live value came from outside the
    /// constraint system, so an `assert` on it is vacuous and a published value is
    /// free: the source read a cell nothing stores, or the lowering dropped or
    /// misplaced a store the source asked for.
    ///
    /// Legitimate unconstrained cells are absent by construction, not by
    /// exemption: a range-check touch only links its two cells, reading neither,
    /// and an arithmetic back-solve writes its operand before reading it.
    pub unconstrained_reads: Vec<u32>,
    pub(crate) trace: Trace, // rows + final access-count columns, emitted in the same walk
    /// `g^0, g^1, ..` as far as the run indexed them: every pc, frame base and cell it reached.
    pub(crate) gpow: Vec<F64>,
    /// Each witness stream's cursor when the run ended, in the order [`Plan`] resolved them.
    pub(crate) cursors: Vec<usize>,
    /// Loops the run executed as parallel segments ([`super::segments`]).
    pub(crate) segmented: usize,
}

impl Execution {
    /// Rows per table: the instructions of each opcode the run executed.
    pub fn rows(&self) -> [usize; crate::tables::N_TABLES] {
        self.trace.row_counts()
    }

    /// The first difference between two runs' traces, memory images (each cell's word, access
    /// count and write state), counts, g-power tables and hint cursors, if any.
    #[doc(hidden)]
    pub fn mismatch(&self, other: &Self) -> Option<String> {
        fn rows<T: PartialEq + std::fmt::Debug>(name: &str, a: &Rows<T>, b: &Rows<T>) -> Option<String> {
            if a.len() != b.len() {
                return Some(format!("{name}: {} rows against {}", a.len(), b.len()));
            }
            let (i, (x, y)) = a.iter().zip(b.iter()).enumerate().find(|(_, (x, y))| x != y)?;
            Some(format!("{name} row {i}: {x:?} against {y:?}"))
        }
        let (a, b) = (&self.trace, &other.trace);
        let scalars = [
            ("cycles", self.cycles, other.cycles),
            ("mem_used", self.mem_used, other.mem_used),
            ("memory cells", self.mem.len, other.mem.len),
            ("counts", a.counts, b.counts),
            ("g-powers", self.gpow.len(), other.gpow.len()),
        ];
        if let Some((name, x, y)) = scalars.iter().find(|(_, x, y)| x != y) {
            return Some(format!("{name}: {x} against {y}"));
        }
        if self.unconstrained_reads != other.unconstrained_reads {
            return Some("unconstrained reads differ".into());
        }
        if self.cursors != other.cursors {
            return Some(format!("hint cursors: {:?} against {:?}", self.cursors, other.cursors));
        }
        if let Some(pc) = (0..a.bytecode_count.len()).find(|&pc| a.bytecode_count[pc] != b.bytecode_count[pc]) {
            return Some(format!("bytecode count at pc {pc}"));
        }
        if let Some(i) = (0..self.mem_used).find(|&i| {
            let (x, y) = (self.mem.cell(i), other.mem.cell(i));
            x.word != y.word || x.count != y.count || x.state != y.state
        }) {
            let (x, y) = (self.mem.cell(i), other.mem.cell(i));
            return Some(format!(
                "cell {i}: {:?} count {} {:?} against {:?} count {} {:?}",
                x.word, x.count, x.state, y.word, y.count, y.state
            ));
        }
        rows("xor", &a.xor, &b.xor)
            .or_else(|| rows("mul", &a.mul, &b.mul))
            .or_else(|| rows("set", &a.set, &b.set))
            .or_else(|| rows("deref", &a.deref, &b.deref))
            .or_else(|| rows("jump", &a.jump, &b.jump))
            .or_else(|| rows("keccak", &a.keccak, &b.keccak))
    }
}

/// `log2` of the cells a [`Memory`] page holds.
pub(super) const PAGE_LOG: usize = 16;
pub(super) const PAGE: usize = 1 << PAGE_LOG;

/// A cell's write state. A cell not `Written` holds ZERO; a `DEREF` between two
/// unwritten cells links them, and a write to either reaches the other at once.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub(super) enum State {
    Unwritten = 0,
    Linked,
    Written,
}

/// One cell: its word, its access count as an exponent (the column holds `g^count`),
/// and its write state, side by side so one access touches one line. All zero is an
/// unwritten cell holding ZERO, never accessed.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct Cell {
    pub(crate) word: F128,
    pub(crate) count: u32,
    pub(super) state: State,
}

impl Cell {
    pub(crate) const UNWRITTEN: Self = Self {
        word: F128::ZERO,
        count: 0,
        state: State::Unwritten,
    };
}

/// The write-once data memory: `len` cells, a power of two, in pages, so that the
/// image grows without moving what it holds. Cells past the last page are unwritten
/// and never accessed.
pub struct Memory {
    pages: Vec<Box<[Cell]>>,
    len: usize,
}

impl Memory {
    /// Cells in the image, its power-of-two pad included.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The word in cell `i`.
    pub fn word(&self, i: usize) -> F128 {
        self.cell(i).word
    }

    /// Cell `i`, unwritten past the last page.
    fn cell(&self, i: usize) -> Cell {
        self.page(i >> PAGE_LOG).map_or(Cell::UNWRITTEN, |p| p[i & (PAGE - 1)])
    }

    /// Page `p`'s cells, `None` past the last one.
    pub(crate) fn page(&self, p: usize) -> Option<&[Cell]> {
        self.pages.get(p).map(|p| &p[..])
    }

    pub(crate) fn page_log() -> usize {
        PAGE_LOG
    }
}

/// Why a run has no execution the interpreter can find: the program, on this public
/// input and advice, asks for something the machine cannot do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecError {
    /// The instruction the run failed at, or whose hints it failed in.
    pub pc: u32,
    /// Its function and source line ([`Program::site_at`]).
    pub site: String,
    pub fault: Fault,
}

/// What the run asked for that the machine cannot do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fault {
    /// A second value for a written cell: a failed `assert`, two writes that disagree,
    /// or a write to a cell an instruction already read, unwritten, as ZERO. `hint`
    /// names the hint that wrote it, if one did.
    Conflict {
        cell: u32,
        had: F128,
        new: F128,
        hint: Option<&'static str>,
    },
    /// A `DEREF` through a word that is no g-power below `MAX_CELLS`: a wild pointer, a
    /// failed range check, or an address past any memory a stack can hold.
    WildPointer { value: F128 },
    /// A word that has to be in `K` is not: a `JUMP` operand, or one a hint reads.
    NotInK { what: &'static str, value: F128 },
    /// A word that has to be an address or a count is not a g-power in its range.
    NotAGPower { what: &'static str, value: F64 },
    /// A `MUL` asked to solve `a·x = c` for `x` with `a = 0`.
    BackSolveThroughZero,
    /// The advice does not fit the program: a witness stream is missing or
    /// exhausted, or an entry has the wrong length.
    Witness(String),
    /// The program allocates or accesses a cell past any memory a stack can hold (`MAX_CELLS`).
    OutOfMemory,
    /// The halt `pc` reached in a frame other than `main`'s.
    HaltOutsideMain { fp: u32 },
    /// Too many instructions: runaway recursion, or a loop that never ends.
    StepLimit,
}

/// The longest run the interpreter executes, the longest a stack can commit: every step is a row
/// of at least `MIN_COMMITTED_COLUMNS` committed columns.
pub(super) const MAX_STEPS: usize = (1 << verifier::MAX_STACKED_LOG) / crate::tables::MIN_COMMITTED_COLUMNS;
// A row touches at most a `KECCAK` row's cells, so every access count fits the `u32` exponent.
const _: () = assert!(crate::keccak::ROW_CELLS * MAX_STEPS < u32::MAX as usize);

impl ExecError {
    #[cold]
    pub(super) fn new(program: &Program, pc: u32, fault: Fault) -> Self {
        Self {
            pc,
            site: program.site_at(pc),
            fault,
        }
    }
}

impl std::fmt::Display for ExecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at pc {} (in {})", self.fault, self.pc, self.site)
    }
}

impl std::error::Error for ExecError {}

/// A word as its limbs, most significant first.
fn word(w: F128) -> String {
    format!("{:x}:{:x}", w.c1, w.c0)
}

impl std::fmt::Display for Fault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Conflict { cell, had, new, hint } => {
                write!(
                    f,
                    "write-once conflict at cell {cell}: had {}, new {}",
                    word(*had),
                    word(*new)
                )?;
                match hint {
                    Some(hint) => write!(f, " (hint {hint})"),
                    None => Ok(()),
                }
            }
            Self::WildPointer { value } => write!(
                f,
                "DEREF pointer is not a g-power below 2^{}: a wild pointer, a failed range check, or past any memory a stack can hold (value {})",
                MAX_CELLS.ilog2(),
                word(*value)
            ),
            Self::NotInK { what, value } => write!(f, "{what} is not a K-valued word ({})", word(*value)),
            Self::NotAGPower { what, value } => write!(f, "{what} is not a g-power in range (0x{:016x})", value.0),
            Self::BackSolveThroughZero => write!(f, "cannot back-solve MUL through a zero operand"),
            Self::Witness(problem) => write!(f, "{problem}"),
            Self::OutOfMemory => write!(
                f,
                "the program reaches past 2^{} cells, more memory than a stack can hold",
                MAX_CELLS.ilog2()
            ),
            Self::HaltOutsideMain { fp } => write!(f, "the halt pc reached in frame {fp}, not main's"),
            Self::StepLimit => write!(f, "step limit of {MAX_STEPS} exceeded (runaway recursion?)"),
        }
    }
}

/// A memory word that has to lie in K, or why it does not.
fn in_k(what: &'static str, v: F128) -> Result<F64, Fault> {
    v.as_base().ok_or(Fault::NotInK { what, value: v })
}

/// A named witness stream and the next entry `hint_witness` pops from it. Resolved
/// once per run, so a pop indexes it rather than hashing its name.
#[derive(Clone)]
pub(super) struct Stream<'a> {
    name: &'a str,
    entries: Option<&'a [Vec<F128>]>,
    pub(super) next: usize,
}

impl<'a> Stream<'a> {
    fn pop(&mut self, len: u32) -> Result<&'a [F128], Fault> {
        let name = self.name;
        let entries = self
            .entries
            .ok_or_else(|| Fault::Witness(format!("no witness stream `{name}` (Program::set_witness)")))?;
        let entry = entries.get(self.next).ok_or_else(|| {
            Fault::Witness(format!(
                "witness stream `{name}` exhausted (needs entry {}, has {})",
                self.next + 1,
                entries.len()
            ))
        })?;
        if entry.len() != len as usize {
            return Err(Fault::Witness(format!(
                "witness `{name}` entry {} holds {} values, the destination {len}",
                self.next,
                entry.len()
            )));
        }
        self.next += 1;
        Ok(entry)
    }

    pub(super) fn name(&self) -> &'a str {
        self.name
    }

    /// Entries in the stream.
    pub(super) fn len(&self) -> usize {
        self.entries.map_or(0, <[_]>::len)
    }
}

/// The hints of a program as per-pc tables. `Program::hints` is keyed by pc, so probing
/// it each step costs a hash of the counter for what is almost always a miss; the
/// program has one bytecode slot per pc, so flatten the map into a dense table: 0 = no
/// hints, otherwise the 1-based index into `lists`, whose entries pair a pc's hints with
/// where their streams start in `streams`.
pub(super) struct Plan<'a> {
    at: Vec<u32>,
    lists: Vec<(&'a [RHint], usize)>,
    streams: Vec<u32>,
}

impl<'a> Plan<'a> {
    /// The plan, and the witness streams it indexes, resolved by name once so a pop indexes one.
    fn new(program: &'a Program) -> (Self, Vec<Stream<'a>>) {
        let mut streams: Vec<Stream> = Vec::new();
        let mut stream_of: HashMap<&str, u32> = HashMap::new();
        let mut at = vec![0u32; program.prog.len()];
        let mut lists: Vec<(&[RHint], usize)> = Vec::with_capacity(program.hints.len());
        let mut hint_streams: Vec<u32> = Vec::new();
        for (&hpc, hs) in &program.hints {
            lists.push((hs.as_slice(), hint_streams.len()));
            at[hpc as usize] = lists.len() as u32;
            for h in hs {
                let (RHint::WitnessStack { name, .. } | RHint::WitnessHeap { name, .. }) = h else {
                    hint_streams.push(u32::MAX);
                    continue;
                };
                let s = *stream_of.entry(name).or_insert_with(|| {
                    streams.push(Stream {
                        name,
                        entries: program.witness.get(name).map(Vec::as_slice),
                        next: 0,
                    });
                    streams.len() as u32 - 1
                });
                hint_streams.push(s);
            }
        }
        let plan = Self {
            at,
            lists,
            streams: hint_streams,
        };
        (plan, streams)
    }

    /// The hints of `pc`, each with the stream it pops.
    #[inline(always)]
    fn list(&self, pc: u32) -> impl Iterator<Item = (&'a RHint, u32)> {
        let (hs, first) = self.lists[self.at[pc as usize] as usize - 1];
        hs.iter().zip(self.streams[first..].iter().copied())
    }

    /// The one pc whose hints pop stream `s`, if exactly one does.
    pub(super) fn marker(&self, s: u32) -> Option<u32> {
        let mut found = None;
        for pc in 0..self.at.len() as u32 {
            if self.at[pc as usize] != 0 && self.list(pc).any(|(_, st)| st == s) {
                if found.is_some() {
                    return None;
                }
                found = Some(pc);
            }
        }
        found
    }
}

/// The trace table a row went to, so a segment can find a row's counts again.
#[derive(Clone, Copy)]
pub(super) enum Table {
    Xor,
    Mul,
    Set,
    Deref,
    Jump,
    Keccak,
}

/// The rows a walk emits.
pub(super) struct RowSet {
    pub(super) xor: Rows<Xrow>,
    pub(super) mul: Rows<Xrow>,
    pub(super) set: Rows<Srow>,
    pub(super) deref: Rows<Drow>,
    pub(super) jump: Rows<Jrow>,
    pub(super) keccak: Rows<Krow>,
}

impl RowSet {
    pub(super) fn new() -> Self {
        Self {
            xor: Rows::new(),
            mul: Rows::new(),
            set: Rows::new(),
            deref: Rows::new(),
            jump: Rows::new(),
            keccak: Rows::new(),
        }
    }
}

/// The interpreter's registers, hint cursors and output, which a walk advances.
pub(super) struct Run<'a> {
    pub(super) pc: u32,
    pub(super) fp: u32,
    pub(super) next_free: u32,
    pub(super) steps: usize,
    pub(super) streams: Vec<Stream<'a>>,
    /// Per-pc execution count, as an exponent of `g`.
    pub(super) bytecode_count: Vec<u32>,
    pub(super) rows: RowSet,
    /// `DBG_PROF=1`: per-pc step counts.
    pub(super) prof: Option<Vec<u64>>,
}

/// Where a walk stops early: before the hints of `pc`, once stream `stream` has popped
/// `pops` entries.
#[derive(Clone, Copy)]
pub(super) struct Stop {
    pub(super) pc: u32,
    pub(super) stream: u32,
    pub(super) pops: usize,
}

impl Stop {
    const NEVER: Self = Self {
        pc: u32::MAX,
        stream: 0,
        pops: 0,
    };
}

/// What the interpreter loop reads and writes: memory and the g-power table. The run's own
/// ([`Mem`]) and a segment's (`segments::Segment`) differ in where a cell lives and in what
/// they record about it.
pub(super) trait Machine {
    /// Cover cell `idx` for a store.
    fn ensure(&mut self, idx: usize);
    /// Whether `cell` is written, when a hint or an instruction decides on it.
    fn written(&mut self, cell: u32) -> bool;
    /// A hint's read, which writes nothing: what a hint computes is advice the
    /// instructions go on to check.
    fn get(&mut self, cell: u32) -> F128;
    /// An instruction's read.
    fn read(&mut self, cell: u32) -> F128;
    /// Write-once store, carried to every cell linked to this one.
    fn put(&mut self, cell: u32, v: F128) -> Result<(), ExecError>;
    /// Read the running access count and advance it.
    fn bump(&mut self, cell: u32) -> u32;
    /// `put`, then the cell's access count.
    fn put_counted(&mut self, cell: u32, v: F128) -> Result<u32, ExecError>;
    /// `DEREF` in cell mode: the equality `m[a2] == m[a3]`.
    fn deref_cell(&mut self, a2: u32, a3: u32) -> Result<(), ExecError>;
    /// Row `row` of `table` holds the counts read since the previous row.
    #[inline(always)]
    fn row(&mut self, _table: Table, _row: usize) {}
    fn at_pc(&mut self, pc: u32);
    fn hint(&mut self, hint: Option<&'static str>);
    /// Whether the walk has to stop: a segment met what it cannot run alone.
    #[inline(always)]
    fn abandoned(&self) -> bool {
        false
    }
    fn g_note(&mut self, k: usize);
    fn g_grow(&mut self, upto: usize);
    fn g_pow(&mut self, k: usize) -> F64;
    /// The discrete log of `x`, if it is a g-power below `bound` (at most `MAX_INDEX`).
    fn g_log_below(&mut self, x: F64, bound: usize) -> Option<u32>;
    /// The discrete log of `x`, if it is a g-power below `MAX_CELLS`: an address or a count.
    #[inline(always)]
    fn g_log(&mut self, x: F64) -> Option<u32> {
        self.g_log_below(x, MAX_CELLS as usize)
    }

    /// `read`, then the cell's access count: the word and the count before it.
    #[inline(always)]
    fn read_counted(&mut self, cell: u32) -> (F128, u32) {
        let word = self.read(cell);
        (word, self.bump(cell))
    }
}

/// The memory's pages, grown as the run reaches them, each cell's access count and write
/// state with its word. Every method on the hot path is `#[inline(always)]`: they sit in
/// the interpreter's opcode loop.
pub(super) struct Mem<'a> {
    pages: Vec<Box<[Cell]>>,
    /// One past the highest cell accessed or reserved: every cell below it is
    /// in a page.
    pub(super) used: usize,
    /// The highest access count any cell reached.
    pub(super) counts: u32,
    /// The cells a `DEREF` linked each cell to.
    pub(super) links: HashMap<u32, Vec<u32>>,
    /// Cells an instruction read before anything wrote them.
    unwritten_reads: Vec<u32>,
    /// To name where a conflict happened.
    program: &'a Program,
    /// The pc of the currently executing instruction, and the name of the
    /// computed-advice hint if the write comes from one, so a write-once
    /// conflict can report where it happened. Plain fields rather than
    /// thread-locals: this is written on every step, and a thread-local
    /// costs a lazy-init check each time.
    dbg_pc: u32,
    dbg_hint: Option<&'static str>,
    pub(super) g: GPow,
}

impl Mem<'_> {
    #[cold]
    #[inline(never)]
    fn grow(&mut self, idx: usize) {
        self.used = idx + 1;
        self.reserve(idx);
    }
    /// Pages up to the one holding cell `upto`, leaving `used` alone.
    pub(super) fn reserve(&mut self, upto: usize) {
        while self.pages.len() <= upto >> PAGE_LOG {
            // SAFETY: all zero is a valid `Cell`, the unwritten one.
            self.pages
                .push(unsafe { Box::<[Cell]>::new_zeroed_slice(PAGE).assume_init() });
        }
    }
    /// The pages, for the segments to write their own cells into.
    pub(super) fn page_ptrs(&mut self) -> Vec<*mut Cell> {
        self.pages.iter_mut().map(|p| p.as_mut_ptr()).collect()
    }
    /// Cell `c`, which `ensure` has covered.
    #[inline(always)]
    pub(super) fn at(&mut self, c: u32) -> &mut Cell {
        let c = c as usize;
        &mut self.pages[c >> PAGE_LOG][c & (PAGE - 1)]
    }
    #[inline(always)]
    fn covered(&self, c: u32) -> Option<&Cell> {
        let c = c as usize;
        (c < self.used).then(|| &self.pages[c >> PAGE_LOG][c & (PAGE - 1)])
    }
    /// Cell `c` as a hint sees it: unwritten and unaccessed past what `ensure` covered.
    #[inline(always)]
    pub(super) fn cell(&self, c: u32) -> Cell {
        self.covered(c).copied().unwrap_or(Cell::UNWRITTEN)
    }
    #[inline(always)]
    pub(super) fn is_written(&self, cell: u32) -> bool {
        self.covered(cell).is_some_and(|c| c.state == State::Written)
    }
    #[inline(always)]
    fn word(&self, cell: u32) -> F128 {
        self.covered(cell).map_or(F128::ZERO, |c| c.word)
    }
    // `put` on one cell; whether it was linked.
    #[inline(always)]
    fn store(&mut self, cell: u32, v: F128) -> Result<bool, ExecError> {
        self.ensure(cell as usize);
        let c = self.at(cell);
        let s = c.state;
        if s == State::Written && c.word != v {
            return Err(self.conflict(cell, v));
        }
        c.word = v;
        c.state = State::Written;
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
            let c = self.at(x);
            if c.state == State::Unwritten {
                c.state = State::Linked;
            }
            self.links.entry(x).or_default().push(y);
        }
    }
    #[cold]
    fn conflict(&self, cell: u32, v: F128) -> ExecError {
        let fault = Fault::Conflict {
            cell,
            had: self.word(cell),
            new: v,
            hint: self.dbg_hint,
        };
        ExecError::new(self.program, self.dbg_pc, fault)
    }
}

impl Machine for Mem<'_> {
    // All accessed cells satisfy cell < next_free after their frame's allocation, so
    // this only ever extends.
    #[inline(always)]
    fn ensure(&mut self, idx: usize) {
        if idx >= self.used {
            self.grow(idx);
        }
    }
    #[inline(always)]
    fn written(&mut self, cell: u32) -> bool {
        self.is_written(cell)
    }
    #[inline(always)]
    fn get(&mut self, cell: u32) -> F128 {
        self.word(cell)
    }
    // An unwritten cell reads as ZERO, which is then written there: the row is filled
    // from the final image, which has to agree.
    #[inline(always)]
    fn read(&mut self, cell: u32) -> F128 {
        self.ensure(cell as usize);
        let c = self.at(cell);
        let word = c.word;
        if c.state != State::Written {
            c.state = State::Written;
            self.unwritten_reads.push(cell);
        }
        word
    }
    #[inline(always)]
    fn put(&mut self, cell: u32, v: F128) -> Result<(), ExecError> {
        if self.store(cell, v)? {
            self.spread(cell, v)?;
        }
        Ok(())
    }
    // The column's `×g`, on the exponent. A run makes fewer than `2^32` accesses in all,
    // so it cannot wrap.
    #[inline(always)]
    fn bump(&mut self, cell: u32) -> u32 {
        self.ensure(cell as usize);
        let c = self.at(cell);
        let count = c.count;
        c.count = count + 1;
        self.counts = self.counts.max(count + 1);
        count
    }
    #[inline(always)]
    fn put_counted(&mut self, cell: u32, v: F128) -> Result<u32, ExecError> {
        let linked = self.store(cell, v)?;
        let count = self.bump(cell);
        if linked {
            self.spread(cell, v)?;
        }
        Ok(count)
    }
    // Carry a written side to the other (`put` checks it when both are), or link two
    // unwritten sides until either is written. A range-check touch may leave both
    // unwritten for good: only the validity of `a2` matters there.
    #[inline(always)]
    fn deref_cell(&mut self, a2: u32, a3: u32) -> Result<(), ExecError> {
        if self.is_written(a2) {
            self.put(a3, self.word(a2))
        } else if self.is_written(a3) {
            self.put(a2, self.word(a3))
        } else {
            self.link(a2, a3);
            Ok(())
        }
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
    fn g_note(&mut self, k: usize) {
        self.g.note(k);
    }
    #[inline(always)]
    fn g_grow(&mut self, upto: usize) {
        self.g.grow_to(upto);
    }
    #[inline(always)]
    fn g_pow(&mut self, k: usize) -> F64 {
        self.g.pow(k)
    }
    #[inline(always)]
    fn g_log_below(&mut self, x: F64, bound: usize) -> Option<u32> {
        self.g.log_below(x, bound)
    }
}

// The cell a heap run starts at: read the pointer back out of memory and
// invert it. Shared by every hint that writes through one.
fn heap_base<M: Machine>(m: &mut M, cell: u32, what: &'static str) -> Result<u32, Fault> {
    let p = in_k(what, m.get(cell))?;
    m.g_log(p).ok_or(Fault::NotAGPower { what, value: p })
}

// Where a computed-advice bit buffer starts: a frame run needs no lookup
// at all, which is the point of having one.
fn bits_base<M: Machine>(m: &mut M, fp: u32, dest: BitsDest, what: &'static str) -> Result<u32, Fault> {
    match dest {
        BitsDest::Stack(base) => Ok(fp + base),
        BitsDest::Heap(ptr) => heap_base(m, fp + ptr, what),
    }
}

impl Program {
    /// Run the program in write-once *fill* mode to produce its [`Execution`]:
    /// the final memory image and the step count, or why it has none. The public
    /// input seeds the first two memory cells `m[0], m[1]` (§sec:e2e-pi).
    /// Compilation yields the `Program`; executing it (here) and proving it are
    /// separate later phases.
    pub fn execute(&self, public_input: [F128; 2]) -> Result<Execution, ExecError> {
        self.execute_with(public_input, &mut |_| {})
    }

    /// [`Self::execute`], handing `keccak_step` the lanes each `KECCAK` row reads, in row order
    /// and as the row runs. With [`Program::segment_streams`] set, the iterations those streams
    /// mark run in parallel ([`super::segments`]); the result is the serial one either way.
    pub(crate) fn execute_with(
        &self,
        public_input: [F128; 2],
        keccak_step: &mut dyn FnMut(&Lanes),
    ) -> Result<Execution, ExecError> {
        let streams: Vec<&str> = self.segment_streams.iter().map(String::as_str).collect();
        self.execute_segmented(public_input, keccak_step, &streams, 0)
    }

    /// [`Self::execute`], running the iterations `segment_streams` mark as parallel segments,
    /// with the lanes it passed on for the Keccak witness, in order, and how many loops it ran
    /// so: none when it ran serially throughout, because nothing qualified or because a
    /// segment's check failed. `skew` adds to a hint cursor predicted for the last segment,
    /// which a test uses to force a misprediction after earlier segments merged.
    #[doc(hidden)]
    pub fn execute_in_segments(
        &self,
        public_input: [F128; 2],
        segment_streams: &[&str],
        skew: usize,
    ) -> Result<(Execution, Vec<Lanes>, usize), ExecError> {
        let mut lanes = Vec::new();
        let e = self.execute_segmented(public_input, &mut |l| lanes.push(*l), segment_streams, skew)?;
        let segmented = e.segmented;
        Ok((e, lanes, segmented))
    }

    fn execute_segmented(
        &self,
        public_input: [F128; 2],
        keccak_step: &mut dyn FnMut(&Lanes),
        segment_streams: &[&str],
        skew: usize,
    ) -> Result<Execution, ExecError> {
        let ending_pc = (self.prog.len() - 1) as u32; // last bytecode slot, g^{B-1}
        if self.main_frame >= MAX_CELLS {
            return Err(ExecError::new(self, 0, Fault::OutOfMemory));
        }

        // g^j, covering the pcs and return targets and the main frame, then grown by each
        // reservation and each lookup past it, and its reverse index g^j ↦ j.
        let seed = (self.prog.len() + 2).max(self.main_frame as usize);

        // The write-once data memory, each cell's access count and write state with its word.
        let n0 = self.main_frame.max(2) as usize;
        let mut m = Mem {
            pages: Vec::new(),
            used: 0,
            counts: 0,
            links: HashMap::new(),
            unwritten_reads: Vec::new(),
            program: self,
            dbg_pc: 0,
            dbg_hint: None,
            g: GPow::new(seed),
        };
        m.ensure(n0 - 1);
        // Seed the public input into m[0], m[1] (addresses g^0, g^1, §sec:e2e-pi).
        m.put(0, public_input[0])?;
        m.put(1, public_input[1])?;

        let (plan, streams) = Plan::new(self);
        let mut run = Run {
            pc: 0,
            fp: 0,
            next_free: self.main_frame,
            steps: 0,
            streams,
            bytecode_count: vec![0; self.prog.len()],
            rows: RowSet::new(),
            // `DBG_PROF=1`: per-pc step counts, printed as a per-function cycle
            // profile after the run (needs `fn_ranges`, i.e. a compiled program).
            prof: std::env::var("DBG_PROF").is_ok().then(|| vec![0u64; self.prog.len()]),
        };

        // Count the `KECCAK` rows `keccak_step` has seen, so a serial rerun after a failed
        // segmented attempt passes it only the rows it has not.
        let mut emitted = 0usize;
        let mut step = |lanes: &Lanes| {
            emitted += 1;
            keccak_step(lanes)
        };
        let outcome: Result<Option<usize>, ExecError> = (|| {
            let mut segmented = 0;
            if !segment_streams.is_empty() && run.prof.is_none() {
                match super::segments::run(
                    self,
                    &plan,
                    &mut m,
                    &mut run,
                    &mut step,
                    segment_streams,
                    ending_pc,
                    skew,
                )? {
                    Some(done) => segmented = done,
                    None => return Ok(None),
                }
            }
            self.walk(&plan, &mut m, &mut run, &mut step, ending_pc, Stop::NEVER)?;
            Ok(Some(segmented))
        })();
        // A segment counts its steps from zero, so a run that passes the step limit inside its segments stops later
        // than the serial run would, at another instruction: rerun it serially. The second arm is a defence: a merge
        // leaves the run inside its loop, so the walk that follows meets the limit before the run can halt.
        let outcome = match outcome {
            Err(e) if !segment_streams.is_empty() && e.fault == Fault::StepLimit => Ok(None),
            Ok(Some(_)) if run.steps > self.step_limit => Ok(None),
            outcome => outcome,
        };
        let segmented = match outcome? {
            Some(done) => done,
            None => {
                // A segment's check failed, or the run reached the step limit: rerun serially
                // from the start, which is the result by definition.
                let skip = emitted;
                let mut seen = 0usize;
                return self.execute_segmented(
                    public_input,
                    &mut |lanes| {
                        seen += 1;
                        if seen > skip {
                            keccak_step(lanes)
                        }
                    },
                    &[],
                    0,
                );
            }
        };
        let Run {
            steps,
            bytecode_count,
            rows,
            prof,
            streams,
            ..
        } = run;

        if let Some(p) = &prof {
            let mut rows: Vec<(String, u64)> = self
                .fn_ranges
                .iter()
                .map(|(name, entry, len)| {
                    let total: u64 = p[*entry as usize..(*entry + *len) as usize].iter().sum();
                    (name.clone(), total)
                })
                .collect();
            rows.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
            // `DBG_PROF_DUMP=path`: also write the raw per-pc counts plus the
            // function table, so an offline pass can attribute the straight-line
            // cycles of one big function to its source regions (the call sites of
            // the lowered `for` helpers are the landmarks).
            if let Ok(path) = std::env::var("DBG_PROF_DUMP") {
                let mut out = format!("# steps {steps}\n");
                for (name, entry, len) in &self.fn_ranges {
                    out += &format!("F {name} {entry} {len}\n");
                }
                for (pc, c) in p.iter().enumerate() {
                    if *c > 0 {
                        out += &format!("{pc} {c}\n");
                    }
                }
                let path = if std::path::Path::new(&path).exists() {
                    format!("{path}.{}", std::process::id())
                } else {
                    path
                };
                std::fs::write(&path, out).expect("write DBG_PROF_DUMP");
                eprintln!("== DBG_PROF: per-pc counts written to {path}");
            }
            eprintln!("== DBG_PROF: cycles by function ({} total) ==", pretty_integer(steps));
            for (name, c) in rows.iter().filter(|(_, c)| *c > 0) {
                eprintln!(
                    "  {:>13}  {:>7}%  {name}",
                    pretty_integer(c),
                    pretty_f64(100.0 * *c as f64 / steps as f64)
                );
            }
        }

        let counts = bytecode_count.iter().copied().max().unwrap_or(0).max(m.counts) as usize + 1;
        let RowSet {
            xor,
            mul,
            set,
            deref,
            jump,
            keccak,
        } = rows;
        let trace = Trace {
            xor,
            mul,
            set,
            deref,
            jump,
            keccak,
            bytecode_count,
            counts,
        };
        // The memory announced: the least power of two covering every cell the run touched
        // (the boundary tables read a dense image).
        let mem_used = m.used;
        let cells = mem_used.next_power_of_two();
        Ok(Execution {
            mem: Memory {
                pages: m.pages,
                len: cells,
            },
            cycles: steps,
            mem_used,
            unconstrained_reads: m.unwritten_reads,
            trace,
            gpow: m.g.into_powers(),
            cursors: streams.iter().map(|s| s.next).collect(),
            segmented,
        })
    }

    /// The interpreter loop: run from `r`'s registers until the program halts (`Ok(false)`)
    /// or reaches `stop` (`Ok(true)`), or a segment's machine gives up (an error).
    pub(super) fn walk<M: Machine>(
        &self,
        plan: &Plan<'_>,
        m: &mut M,
        r: &mut Run<'_>,
        keccak_step: &mut dyn FnMut(&Lanes),
        ending_pc: u32,
        stop: Stop,
    ) -> Result<bool, ExecError> {
        let Run {
            pc: pc_out,
            fp: fp_out,
            next_free,
            steps,
            streams,
            bytecode_count,
            rows,
            prof,
        } = r;
        let RowSet {
            xor,
            mul,
            set,
            deref,
            jump,
            keccak,
        } = rows;
        let (mut pc, mut fp) = (*pc_out, *fp_out);
        loop {
            if pc >= ending_pc {
                // The machine halts in the final STATE `(g^{B-1}, g^0)`, the one the state bus
                // pulls as its boundary (doc/leanvm sec:state).
                if fp == 0 && pc == ending_pc {
                    (*pc_out, *fp_out) = (pc, fp);
                    return Ok(false);
                }
                if pc > ending_pc {
                    // The last slot ran in another frame and fell through to `g^B`, no address.
                    return Err(ExecError::new(self, ending_pc, Fault::HaltOutsideMain { fp }));
                }
            }
            if pc == stop.pc && streams[stop.stream as usize].next == stop.pops {
                (*pc_out, *fp_out) = (pc, fp);
                return Ok(true);
            }
            if *steps >= self.step_limit {
                return Err(ExecError::new(self, pc, Fault::StepLimit));
            }
            if m.abandoned() {
                return Err(ExecError::new(self, pc, Fault::Witness("segment abandoned".into())));
            }
            m.at_pc(pc);
            let fail = move |fault| ExecError::new(self, pc, fault);
            // Cell `base + offset`, if it lies in the address space.
            let at = move |base: u32, offset: u32| {
                let cell = u64::from(base) + u64::from(offset);
                if cell < u64::from(MAX_CELLS) {
                    Ok(cell as u32)
                } else {
                    Err(fail(Fault::OutOfMemory))
                }
            };
            if let Some(p) = prof.as_mut() {
                p[pc as usize] += 1;
            }

            // Apply the hints scheduled before this instruction.
            if plan.at[pc as usize] != 0 {
                for (h, stream) in plan.list(pc) {
                    m.hint(Some(match h {
                        RHint::FrameAddress { .. } => "FrameAddress",
                        RHint::AllocFrames { .. } => "AllocFrames",
                        RHint::Alloc { .. } => "Alloc",
                        RHint::AllocDyn { .. } => "AllocDyn",
                        RHint::WitnessStack { .. } => "WitnessStack",
                        RHint::WitnessHeap { .. } => "WitnessHeap",
                        RHint::Log2Ceil { .. } => "Log2Ceil",
                        RHint::BitDecompose { .. } => "BitDecompose",
                        RHint::BitDecomposeExp { .. } => "BitDecomposeExp",
                        RHint::DigitDecomposeExp { .. } => "DigitDecomposeExp",
                        RHint::WindowsExp { .. } => "WindowsExp",
                        RHint::FieldLimbs { .. } => "FieldLimbs",
                        RHint::Inverse { .. } => "Inverse",
                        RHint::Print { .. } => "Print",
                    }));
                    match h {
                        RHint::FrameAddress { offset } => {
                            m.g_note(at(fp, *offset)? as usize);
                        }
                        // A fresh region: write its base `g^{next_free}` into the
                        // pointer cell (once) and reserve `size` cells. `AllocDyn`
                        // reads the size from a cell at runtime.
                        RHint::Alloc { .. } | RHint::AllocDyn { .. } | RHint::AllocFrames { .. } => {
                            let (ptr, size) = match *h {
                                RHint::Alloc { ptr, size } => (ptr, size),
                                RHint::AllocFrames {
                                    ptr,
                                    size,
                                    end,
                                    start_inverse,
                                    tail,
                                } => {
                                    let span = in_k("loop bound", m.get(fp + end)).map_err(fail)? * start_inverse;
                                    if span.is_zero() {
                                        return Err(fail(Fault::NotAGPower {
                                            what: "loop bound",
                                            value: span,
                                        }));
                                    }
                                    let n = m.g_log(span).ok_or_else(|| {
                                        fail(Fault::NotAGPower {
                                            what: "loop bound",
                                            value: span,
                                        })
                                    })?;
                                    let cells = size.checked_mul(n).and_then(|frames| frames.checked_add(tail));
                                    (ptr, cells.ok_or_else(|| fail(Fault::OutOfMemory))?)
                                }
                                // A runtime size is carried in the exponent:
                                // the cell holds g^k, allocate k cells.
                                RHint::AllocDyn { ptr, size } => {
                                    let sz = in_k("HeapBuf size", m.get(fp + size)).map_err(fail)?;
                                    let cells = m.g_log(sz).ok_or_else(|| {
                                        fail(Fault::NotAGPower {
                                            what: "HeapBuf size",
                                            value: sz,
                                        })
                                    })?;
                                    (ptr, cells)
                                }
                                _ => unreachable!(),
                            };
                            let cell = fp + ptr;
                            if !m.written(cell) {
                                let base = *next_free;
                                *next_free = at(base, size)?;
                                m.g_grow(*next_free as usize);
                                // The base is about to become a pointer in memory.
                                m.g_note(base as usize);
                                if *next_free > base {
                                    m.ensure(*next_free as usize - 1);
                                }
                                let p = m.g_pow(base as usize);
                                m.put(cell, F128::from(p))?;
                            }
                        }
                        RHint::Print { label, cell } => {
                            let c = fp + cell;
                            if m.written(c) {
                                let v = m.get(c);
                                // Small integers and small g-powers overlap (8 = x^3
                                // = g^3): show every reading that applies. Only a
                                // K-valued word (extension limb 0) can be a g-power. Read
                                // without growing the address table.
                                let k = v.as_base().and_then(|lo| bounded_dlog(lo, MAX_INDEX.ilog2()));
                                let small = v.as_base().is_some_and(|lo| lo.0 < 1 << 32);
                                match (k, small) {
                                    (Some(k), true) => eprintln!(
                                        "[print] {label} = {} (g^{})",
                                        pretty_integer(v.c0),
                                        pretty_integer(k)
                                    ),
                                    (Some(k), false) => {
                                        eprintln!("[print] {label} = g^{}", pretty_integer(k))
                                    }
                                    (None, true) => {
                                        eprintln!("[print] {label} = {}", pretty_integer(v.c0))
                                    }
                                    (None, false) => {
                                        eprintln!("[print] {label} = {:#x}:{:#x}", v.c1, v.c0)
                                    }
                                }
                            } else {
                                eprintln!("[print] {label} = <unwritten>");
                            }
                        }
                        RHint::WitnessStack { base, len, .. } => {
                            let values = streams[stream as usize].pop(*len).map_err(fail)?;
                            for (k, &value) in values.iter().enumerate() {
                                m.put(fp + base + k as u32, value)?;
                            }
                        }
                        RHint::WitnessHeap { ptr, lo, len, .. } => {
                            let b = heap_base(m, fp + ptr, "hint_witness heap pointer").map_err(fail)?;
                            let values = streams[stream as usize].pop(*len).map_err(fail)?;
                            for (k, &value) in values.iter().enumerate() {
                                m.put(b + lo + k as u32, value)?;
                            }
                        }
                        RHint::Log2Ceil {
                            bits,
                            dst,
                            nbits,
                            floor,
                        } => {
                            let b = bits_base(m, fp, *bits, "log2_ceil pointer").map_err(fail)?;
                            let mut word: u128 = 0;
                            for j in 0..*nbits {
                                if !m.get(b + j).is_zero() {
                                    word |= 1u128 << j;
                                }
                            }
                            let cl = if word <= 1 {
                                0
                            } else {
                                u128::BITS - (word - 1).leading_zeros()
                            };
                            let mu = cl.max(*floor);
                            m.put(fp + dst, F128::from(primitives::field::g_pow(mu as usize)))?;
                        }
                        RHint::BitDecompose { value, bits, nbits } => {
                            assert!(*nbits <= 128, "a machine word has 128 bits");
                            let v = u128::from(m.get(fp + value));
                            let bb = bits_base(m, fp, *bits, "decompose pointer").map_err(fail)?;
                            for j in 0..*nbits {
                                m.put(bb + j, F128::from((v >> j) & 1))?;
                            }
                        }
                        RHint::BitDecomposeExp { value, bits, nbits } => {
                            const WHAT: &str = "hint_decompose_bits_exponent value";
                            let x = in_k(WHAT, m.get(fp + value)).map_err(fail)?;
                            let n = bounded_dlog(x, *nbits)
                                .ok_or_else(|| fail(Fault::NotAGPower { what: WHAT, value: x }))?;
                            let bb = bits_base(m, fp, *bits, "hint_decompose_bits_exponent pointer").map_err(fail)?;
                            for j in 0..*nbits {
                                let bit = ((n >> j) & 1) as u64;
                                m.put(bb + j, F128::new(bit, 0))?;
                            }
                        }
                        RHint::WindowsExp {
                            value,
                            dest,
                            at,
                            start,
                            width,
                            count,
                        } => {
                            assert!(start + width * count <= 128, "a machine word has 128 bits");
                            let v = u128::from(m.get(fp + value));
                            let bb = bits_base(m, fp, *dest, "windows pointer").map_err(fail)?;
                            for i in 0..*count {
                                let mut n = 0usize;
                                for j in 0..*width {
                                    n |= (((v >> (start + i * width + j)) & 1) as usize) << j;
                                }
                                m.put(bb + at + i, F128::from(primitives::field::g_pow(n)))?;
                            }
                        }
                        RHint::DigitDecomposeExp {
                            value,
                            dst,
                            nbits,
                            ndigits,
                            digit_bits,
                            start,
                            start_cell,
                        } => {
                            const WHAT: &str = "hint_decompose_digits_exponent value";
                            let x = in_k(WHAT, m.get(fp + value)).map_err(fail)?;
                            let n = bounded_dlog(x, *nbits)
                                .ok_or_else(|| fail(Fault::NotAGPower { what: WHAT, value: x }))?;
                            let s = match start_cell {
                                None => *start,
                                Some(c) => {
                                    const WHAT: &str = "hint_decompose_digits_exponent start";
                                    let y = in_k(WHAT, m.get(fp + c)).map_err(fail)?;
                                    bounded_dlog(y, 7)
                                        .ok_or_else(|| fail(Fault::NotAGPower { what: WHAT, value: y }))?
                                        as u32
                                }
                            };
                            let bb = bits_base(m, fp, *dst, "hint_decompose_digits_exponent pointer").map_err(fail)?;
                            let mask = (1u128 << digit_bits) - 1;
                            for j in 0..*ndigits {
                                let at = s + digit_bits * j;
                                let d = if at >= 128 { 0 } else { (n >> at) & mask };
                                m.put(bb + j, F128::from(primitives::field::g_pow(d as usize)))?;
                            }
                        }
                        RHint::FieldLimbs { value, base, len } => {
                            assert!((1..=2).contains(len), "an F128 value has two K limbs");
                            let v = m.get(fp + value);
                            let limbs = v.to_limbs();
                            for j in 0..*len {
                                m.put(fp + base + j, F128::new(limbs[j as usize], 0))?;
                            }
                        }
                        RHint::Inverse { value, dst } => {
                            let v = m.get(fp + value);
                            m.put(fp + dst, if v.is_zero() { F128::ZERO } else { v.inv() })?;
                        }
                    }
                    m.hint(None);
                }
            }
            // Cover the g-powers this step may index (g²·pc return target, g^fp).
            m.g_grow((pc as usize + 2).max(fp as usize));

            let bytecode_read = {
                let v = bytecode_count[pc as usize];
                bytecode_count[pc as usize] = v + 1;
                v
            };

            // Loaded once: the shared Xor/Mul arm needs the discriminant again,
            // and `Op` is wide enough that re-reading it costs a second load.
            let op = self.prog[pc as usize];
            match op {
                Op::Xor { a, b, c } | Op::Mul { a, b, c } => {
                    let is_xor = matches!(op, Op::Xor { .. });
                    let (aa, ab, ac) = (at(fp, a)?, at(fp, b)?, at(fp, c)?);
                    // The row is the equality `m[c] = m[a] op m[b]` over write-once
                    // memory. Normally the operands are known and the result is
                    // computed forward; for a `MUL` whose result is already written
                    // and exactly one of whose operands is not, the runner
                    // back-solves that operand, which is what produces the
                    // range-check complement `y = g^{k-1}·x^{-1}` from
                    // `MUL x·y = g^{k-1}`, and the quotient of `a / b`, with no
                    // dedicated hint.
                    //
                    // `XOR` deliberately does NOT deduce. Nothing asks it to (both
                    // users are `MUL`), and an `XOR` into an already-written cell is
                    // how `assert a == b` is spelled, so deducing there would define
                    // the operand the assert exists to check instead of failing on it.
                    if !is_xor && m.written(ac) {
                        let (ha, hb) = (m.written(aa), m.written(ab));
                        if ha ^ hb {
                            let vk = m.get(if ha { aa } else { ab });
                            if vk.is_zero() {
                                return Err(fail(Fault::BackSolveThroughZero));
                            }
                            let vc = m.get(ac);
                            m.put(if ha { ab } else { aa }, vc * vk.inv())?;
                        }
                    }
                    let (va, ra) = m.read_counted(aa);
                    let (vb, rb) = m.read_counted(ab);
                    let vc = if is_xor { va + vb } else { va * vb };
                    let rc = m.put_counted(ac, vc)?;
                    let row = Xrow {
                        pc,
                        fp,
                        ra,
                        rb,
                        rc,
                        bytecode_read,
                    };
                    if is_xor {
                        m.row(Table::Xor, xor.len());
                        xor.push(row);
                    } else {
                        m.row(Table::Mul, mul.len());
                        mul.push(row);
                    }
                    pc += 1;
                }
                Op::Set { o, k } => {
                    let a = at(fp, o)?;
                    let r = m.put_counted(a, k)?;
                    m.row(Table::Set, set.len());
                    set.push(Srow {
                        pc,
                        fp,
                        r,
                        bytecode_read,
                    });
                    pc += 1;
                }
                Op::Deref { o1, o2, o3, mode } => {
                    let a1 = at(fp, o1)?;
                    let p = m.read(a1);
                    let p_addr = p.as_base().ok_or_else(|| fail(Fault::WildPointer { value: p }))?;
                    let base = m.g_log(p_addr).ok_or_else(|| fail(Fault::WildPointer { value: p }))?;
                    let a2 = at(base, o2)?;
                    let a3 = at(fp, o3)?;
                    match mode {
                        DerefMode::Cell => m.deref_cell(a2, a3)?,
                        DerefMode::Pc => {
                            // The return target and the frame base are stored as
                            // addresses, and JUMP reads them back.
                            m.g_note(pc as usize + 2);
                            let v = F128::from(m.g_pow(pc as usize + 2));
                            m.put(a2, v)?;
                        }
                        DerefMode::Fp => {
                            m.g_note(fp as usize);
                            let v = F128::from(m.g_pow(fp as usize));
                            m.put(a2, v)?;
                        }
                    }
                    let r1 = m.bump(a1);
                    let r2 = m.bump(a2);
                    let r3 = m.bump(a3);
                    m.row(Table::Deref, deref.len());
                    deref.push(Drow {
                        pc,
                        fp,
                        r1,
                        r2,
                        r3,
                        bytecode_read,
                    });
                    pc += 1;
                }
                Op::Jump { oc, od, of } => {
                    let (ac, ad, af) = (at(fp, oc)?, at(fp, od)?, at(fp, of)?);
                    // All three cells are K-valued on EVERY row, taken or not: the
                    // table commits one lane each and their memory flushes carry
                    // literal zeros above it (§sec:tab-jump), which the bus would
                    // otherwise not balance. A guest branches on g-powers; the one
                    // idiom that once branched on a word, `assert a != b`, takes an
                    // inverse hint instead (§sec:prog-div-ne).
                    let (c, rc) = m.read_counted(ac);
                    let (d, rd) = m.read_counted(ad);
                    let (f, rf) = m.read_counted(af);
                    let c = in_k("JUMP condition", c).map_err(fail)?;
                    let d = in_k("JUMP target", d).map_err(fail)?;
                    let f = in_k("JUMP fp", f).map_err(fail)?;
                    // The is-nonzero witness `w = c⁻¹` is never used for control
                    // flow, only recorded as a witness column, so it is not
                    // computed here at all: `JumpTable::fill` batch-inverts every
                    // row's condition at once (§the trace rows in `cpu::trace`).
                    let taken = !c.is_zero();
                    m.row(Table::Jump, jump.len());
                    jump.push(Jrow {
                        pc,
                        fp,
                        rc,
                        rd,
                        rf,
                        bytecode_read,
                    });
                    if taken {
                        pc = m.g_log_below(d, self.prog.len()).ok_or_else(|| {
                            fail(Fault::NotAGPower {
                                what: "JUMP target",
                                value: d,
                            })
                        })?;
                        fp = m.g_log(f).ok_or_else(|| {
                            fail(Fault::NotAGPower {
                                what: "JUMP fp",
                                value: f,
                            })
                        })?;
                    } else {
                        pc += 1;
                    }
                }
                Op::Keccak { .. } => {
                    use crate::keccak::{ROW_CELLS, STATE_CELLS};
                    let k = crate::tables::KeccakOp::of(&op);
                    let mut cells = [0; ROW_CELLS];
                    for (c, cell) in cells.iter_mut().enumerate() {
                        let (offset, step) = k.cell_parts(c);
                        *cell = at(at(fp, offset)?, step)?;
                    }
                    let input: [F128; STATE_CELLS] = std::array::from_fn(|c| m.read(cells[c]));
                    // No table constraint covers the output (the relation is proven
                    // by the Keccak reduction, §keccak); the interpreter still computes it so
                    // the output cells are consistent for any later read.
                    let mut lanes = crate::keccak::state_of_cells(k.in_odd, |c| input[c]);
                    keccak_step(&lanes);
                    primitives::hash::keccak_f(&mut lanes);
                    let written = STATE_CELLS + k.out_cells.cells();
                    let output = crate::keccak::cells_of_state(&lanes, k.out_odd);
                    for (&cell, &v) in cells[STATE_CELLS..written].iter().zip(&output) {
                        m.put(cell, v)?;
                    }
                    // In flush order, so a cell two operands alias still pairs each
                    // read with its own count. A cell not written keeps count 0.
                    let mut r = [0; ROW_CELLS];
                    for (count, &cell) in r.iter_mut().zip(&cells[..written]) {
                        *count = m.bump(cell);
                    }
                    m.row(Table::Keccak, keccak.len());
                    keccak.push(Krow {
                        pc,
                        fp,
                        r,
                        bytecode_read,
                    });
                    pc += 1;
                }
            }
            *steps += 1;
        }
    }
}
