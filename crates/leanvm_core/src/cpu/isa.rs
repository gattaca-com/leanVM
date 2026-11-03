//! The ISA and the `DEREF` store modes.

use primitives::field::{F64, F192};

#[derive(Clone, Copy, Debug)]
pub enum Op {
    Xor {
        a: u32,
        b: u32,
        c: u32,
    },
    Mul {
        a: u32,
        b: u32,
        c: u32,
    },
    Set {
        o: u32,
        /// The immediate stored into `mem[fp·o]`. A full 192-bit machine word
        /// (`E = F192`); K-valued constants (addresses, small ints) ride the
        /// low lane with `c1 = c2 = 0`.
        k: F192,
    },
    Deref {
        o1: u32,
        o2: u32,
        o3: u32,
        mode: DerefMode,
    },
    Jump {
        oc: u32,
        od: u32,
        of: u32,
    },
    /// `KECCAK`: the Keccak-f\[1600\] permutation of a 25-lane state held in thirteen cells,
    /// read in the layout `in_odd` names and written to consecutive cells from `out` in
    /// the layout `out_odd` names ([`crate::keccak::cell_lanes`]). Only the first `out_cells`
    /// of the thirteen output cells are written; the others are not touched. The cells read are four
    /// independently addressed cells `m`, four consecutive cells from `tail`, the cell `b8`, and
    /// four consecutive cells from `cap` (the capacity lanes 17..25), each canonical (top limb
    /// zero).
    ///
    /// A program absorbs a 136-byte block by XORing its nine message cells into the previous
    /// output: block `k` of a message laid out in cells starts at cell `17k / 2`, halfway
    /// through it when `k` is odd, and the odd layout pairs the lanes as such a block's cells
    /// do. The relation is proven by the Keccak reduction ([`crate::keccak`]).
    Keccak {
        m: [u32; 4],
        tail: u32,
        b8: u32,
        cap: u32,
        out: u32,
        in_odd: bool,
        out_odd: bool,
        out_cells: OutCells,
    },
    /// `DOT2`: `mem[fp·c] = mem[fp·acc] + a0·w0 + a1·w1`, where `mem[fp·a] = a0 + y·a1` is read
    /// canonical (top limb zero), so both lanes are in `K`. The two weights are the consecutive
    /// cells `w0 = mem[p·w]` and `w1 = mem[p·w·g]` behind the pointer `p = mem[fp·ptr]`, as a
    /// `DEREF` addresses its target.
    Dot2 {
        a: u32,
        ptr: u32,
        w: u32,
        acc: u32,
        c: u32,
    },
}

/// How many of a `KECCAK`'s thirteen output cells it writes, from `out` on: the first one, the
/// first two (a 256-bit digest), or all of them (a state a later block reads). Two bytecode
/// flags carry the choice ([`Self::flags`]), so the program fixes it, never the prover.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutCells {
    One,
    Two,
    All,
}

impl OutCells {
    /// The cells written.
    pub const fn cells(self) -> usize {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::All => crate::keccak::STATE_CELLS,
        }
    }

    /// The bytecode flags: whether output cell 1 is written, and whether cells `2..13` are.
    pub(crate) const fn flags(self) -> (bool, bool) {
        (!matches!(self, Self::One), matches!(self, Self::All))
    }
}

impl Op {
    /// The largest memory offset among the instruction's operands.
    pub(crate) fn max_offset(&self) -> u32 {
        match *self {
            Self::Xor { a, b, c } | Self::Mul { a, b, c } => a.max(b).max(c),
            Self::Set { o, .. } => o,
            Self::Deref { o1, o2, o3, .. } => o1.max(o2).max(o3),
            Self::Jump { oc, od, of } => oc.max(od).max(of),
            // `w + 1` is the second weight's offset from the pointer, bounded as a `DEREF`'s `o2`.
            Self::Dot2 { a, ptr, w, acc, c } => a.max(ptr).max(w + 1).max(acc).max(c),
            Self::Keccak {
                m,
                tail,
                b8,
                cap,
                out,
                out_cells,
                ..
            } => {
                // The last cell of each run of consecutive cells, as `KeccakOp::cell` steps.
                let last_out = out + (out_cells.cells() as u32 - 1);
                m.into_iter().chain([tail + 3, b8, cap + 3, last_out]).max().unwrap_or(0)
            }
        }
    }
}

/// The source `DEREF` stores at `mem[loc_o1·o2]`: a local cell, the return
/// address `g²·pc`, or the frame pointer. Encoded as two boolean flags `(f_pc,
/// f_fp)`: `Cell=(0,0)`, `Pc=(1,0)`, `Fp=(0,1)`, keeping the store constraint degree 2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DerefMode {
    Cell,
    Pc,
    Fp,
}

impl DerefMode {
    pub(crate) fn f_pc(self) -> F64 {
        if self == DerefMode::Pc { F64::ONE } else { F64::ZERO }
    }
    pub(crate) fn f_fp(self) -> F64 {
        if self == DerefMode::Fp { F64::ONE } else { F64::ZERO }
    }
}
