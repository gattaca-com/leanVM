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

use primitives::field::F64;

/// `XOR` or `MUL` row: the three cells are `fp·g^{a,b,c}`.
pub(crate) struct Xrow {
    pub(crate) pc: u32,
    pub(crate) fp: u32, // frame base: address = fp + offset, operand = g^offset
    pub(crate) ra: F64,
    pub(crate) rb: F64,
    pub(crate) rc: F64,
    pub(crate) bytecode_read: F64,
}
pub(crate) struct Srow {
    pub(crate) pc: u32,
    pub(crate) fp: u32,
    pub(crate) r: F64,
    pub(crate) bytecode_read: F64,
}
pub(crate) struct Drow {
    pub(crate) pc: u32,
    pub(crate) fp: u32,
    pub(crate) r1: F64,
    pub(crate) r2: F64,
    pub(crate) r3: F64,
    pub(crate) bytecode_read: F64,
}
pub(crate) struct Jrow {
    pub(crate) pc: u32,
    pub(crate) fp: u32,
    pub(crate) rc: F64,
    pub(crate) rd: F64,
    pub(crate) rf: F64,
    pub(crate) bytecode_read: F64,
}

/// `Haraka` row: the six per-cell memory access counts of the four message-chunk
/// cells and the output's two. The addresses are `fp·g^{ins[i]}`, `fp·g^{out}` and its
/// successor; the twelve flock words are those cells' lanes.
pub(crate) struct Hrow {
    pub(crate) pc: u32,
    pub(crate) fp: u32,
    pub(crate) r_in: [F64; 4],
    pub(crate) r_out: [F64; 2],
    pub(crate) bytecode_read: F64,
}

pub(crate) struct Trace {
    pub(crate) xor: Vec<Xrow>,
    pub(crate) mul: Vec<Xrow>,
    pub(crate) set: Vec<Srow>,
    pub(crate) deref: Vec<Drow>,
    pub(crate) jump: Vec<Jrow>,
    pub(crate) haraka: Vec<Hrow>,
    pub(crate) mem_count: Vec<F64>, // per-cell running access count g^{count}; final = g^{A[i]}
    pub(crate) bytecode_count: Vec<F64>, // per-pc running execution count g^{count}; final = g^{A[pc]}
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
            self.haraka.len(),
        ]
    }
}
