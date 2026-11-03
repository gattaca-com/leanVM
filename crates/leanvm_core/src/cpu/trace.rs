//! Per-opcode trace rows, emitted during execution and assembled into a [`Trace`].
//!
//! A row carries only what the witness fill cannot recover: the step's
//! `(pc, fp)`, the access counts read at access time, and `DEREF`'s resolved
//! store index. Everything else is a function of those:
//! operands and immediates come from `prog[pc]`, addresses from `fp` plus those
//! operands, and values from the final memory image, which is write-once and so
//! still holds what each accessed cell held at the time it was accessed. The
//! rows are the interpreter's largest write stream, so what they do not carry
//! they do not pay for, twice: once writing them and once reading them back.
//!
//! An access count is held as its exponent: the column holds `g^k`, the row `k`,
//! and the fill looks the power up.

/// `XOR` or `MUL` row: the three cells are `fp·g^{a,b,c}`.
pub(crate) struct Xrow {
    pub(crate) pc: u32,
    pub(crate) fp: u32, // frame base: address = fp + offset, operand = g^offset
    pub(crate) ra: u32,
    pub(crate) rb: u32,
    pub(crate) rc: u32,
    pub(crate) bytecode_read: u32,
}
pub(crate) struct Srow {
    pub(crate) pc: u32,
    pub(crate) fp: u32,
    pub(crate) r: u32,
    pub(crate) bytecode_read: u32,
}
pub(crate) struct Drow {
    pub(crate) pc: u32,
    pub(crate) fp: u32,
    pub(crate) r1: u32,
    pub(crate) r2: u32,
    pub(crate) r3: u32,
    pub(crate) bytecode_read: u32,
}
pub(crate) struct Jrow {
    pub(crate) pc: u32,
    pub(crate) fp: u32,
    pub(crate) rc: u32,
    pub(crate) rd: u32,
    pub(crate) rf: u32,
    pub(crate) bytecode_read: u32,
}

/// `KECCAK` row: the per-cell memory access counts, in the order the table flushes
/// them (`tables::KeccakOp::cell`): the four `m` cells, the four `tail` cells, `b8`, the
/// four `cap` cells, then the thirteen output cells. The fifty value lanes are those
/// cells' lanes, read back from the final memory image.
#[derive(Clone, Copy)]
pub(crate) struct Krow {
    pub(crate) pc: u32,
    pub(crate) fp: u32,
    pub(crate) r: [u32; crate::keccak::ROW_CELLS],
    pub(crate) bytecode_read: u32,
}

/// Rows in fixed-size chunks, so the trace grows without ever moving what it holds:
/// a `Vec` would copy every row written so far each time it doubled.
pub(crate) struct Rows<T> {
    chunks: Vec<Vec<T>>,
    len: usize,
    /// `log2` of a chunk's rows.
    log: u32,
}

const CHUNK_LOG: u32 = 14;

impl<T> Rows<T> {
    pub(crate) fn new() -> Self {
        Self {
            chunks: Vec::new(),
            len: 0,
            log: CHUNK_LOG,
        }
    }

    /// The rows of `chunks`, each of `2^log` rows but the last.
    pub(crate) fn from_chunks(chunks: Vec<Vec<T>>, log: u32) -> Self {
        let n = chunks.len();
        assert!(
            chunks.iter().enumerate().all(|(i, c)| c.len() == 1 << log || (i + 1 == n && c.len() < 1 << log)),
            "every chunk is full but the last"
        );
        Self {
            len: chunks.iter().map(Vec::len).sum(),
            chunks,
            log,
        }
    }

    #[inline(always)]
    pub(crate) fn push(&mut self, row: T) {
        let chunk = 1 << self.log;
        match self.chunks.last_mut() {
            Some(last) if last.len() < chunk => last.push(row),
            _ => {
                let mut last = Vec::with_capacity(chunk);
                last.push(row);
                self.chunks.push(last);
            }
        }
        self.len += 1;
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub(crate) fn get(&self, i: usize) -> &T {
        &self.chunks[i >> self.log][i & ((1 << self.log) - 1)]
    }

    pub(crate) fn iter(&self) -> impl DoubleEndedIterator<Item = &T> {
        self.chunks.iter().flatten()
    }
}

pub(crate) struct Trace {
    pub(crate) xor: Rows<Xrow>,
    pub(crate) mul: Rows<Xrow>,
    pub(crate) set: Rows<Srow>,
    pub(crate) deref: Rows<Drow>,
    pub(crate) jump: Rows<Jrow>,
    /// In the order of [`crate::keccak::ShapeSort`], so `keccak_live` bounds the rows of each output flag.
    pub(crate) keccak: Rows<Krow>,
    /// One past the last `KECCAK` row that writes output cell 1, and one past the last that writes cells `2..13`.
    pub(crate) keccak_live: [usize; 2],
    pub(crate) bytecode_count: Vec<u32>, // per-pc execution count; the column holds g^{A[pc]}
    /// One past the highest access count, of a cell or a pc: the `g`-powers a fill looks up.
    pub(crate) counts: usize,
}

impl Trace {
    /// Rows per instruction table, in [`crate::cpu::Stats::TABLES`] order.
    pub(crate) fn row_counts(&self) -> [usize; crate::tables::N_TABLES] {
        [
            self.xor.len(),
            self.mul.len(),
            self.set.len(),
            self.deref.len(),
            self.jump.len(),
            self.keccak.len(),
        ]
    }
}
