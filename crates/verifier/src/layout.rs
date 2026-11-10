//! The instruction tables, their bus flushes, and where every committed column sits in the stack.

use crate::bytecode::Bytecode;
use crate::field::{E, GEN, K, ONE, ZERO};
use crate::hash::{STATE_LANES, keccak_f, keccak_rounds};
use crate::keccak::KeccakShape;
use crate::multilinear::{Placement, eq_kernel, log2_ceil, log2_strict, stack_offsets};
use crate::{VerifierError, require};
use std::collections::BTreeMap;
use std::ops::Add;

pub const BUS_BITS: usize = 4; // bus communicates tuples of 2^BUS_BITS field elements

/// A polynomial of degree at most 2 in a table's columns.
#[derive(Clone, Debug, Default)]
pub struct Form {
    terms: BTreeMap<Vec<usize>, E>, // monomial -> coefficient; [] is 1, [i] is x_i, [i, j] is x_i*x_j
}

impl Form {
    fn term(monomial: Vec<usize>, coefficient: E) -> Self {
        Self {
            terms: BTreeMap::from([(monomial, coefficient)]),
        }
    }

    pub fn add_scaled(&mut self, other: &Form, weight: E) {
        for (monomial, &coefficient) in &other.terms {
            *self.terms.entry(monomial.clone()).or_default() += weight * coefficient;
        }
    }

    /// `Σ forms`, the empty sum being the zero polynomial.
    fn sum(forms: impl IntoIterator<Item = Form>) -> Form {
        forms.into_iter().fold(Form::default(), |acc, form| acc + form)
    }

    pub fn evaluate(&self, column: &[E]) -> E {
        E::sum(
            self.terms
                .iter()
                .map(|(monomial, &c)| monomial.iter().fold(c, |acc, &index| acc * column[index])),
        )
    }
}

impl Add for Form {
    type Output = Form;
    fn add(mut self, other: Form) -> Form {
        self.add_scaled(&other, ONE);
        self
    }
}

/// One block of a bus side, always owned by a table. The three blocks no table owns are the framework, on `BusLayout`.
#[derive(Clone, Debug)]
pub struct BusBlock {
    pub log_rows: usize,        // the owner's height: the block spans 2^log_rows rows
    pub rows: u64,              // ...of which the first `rows` flush, the rest being padding rows, whose leaves are 1
    pub coordinates: Vec<Form>, // the tuple flushed, over the owner's OWN local column indices; stays symbolic until its table sumcheck
    pub owner: usize,           // whose block it is, by opcode
}

/// The proof system's version, leading the Fiat-Shamir IV's preimage so that no two versions share a
/// transcript. Its length is fixed and Keccak's padding binds the preimage's, so the bytecode that
/// follows needs no length field of its own. Nothing derives it from the constants below: any change
/// to the constraint system bumps it by hand. Mirrored in `python-verifier/verifier.py`.
pub const TAG: &[u8] = b"leanvm-v1";

// The columns no instruction table owns. They come first in the global column numbering, the tables after.
pub const NUM_GLOBAL_COLUMNS: usize = 6;
pub const MEMORY_0: usize = 0;
pub const MEMORY_1: usize = 1;
pub const MEMORY_2: usize = 2;
pub const MEMORY_FINAL_COUNTERS: usize = 3;
pub const BYTECODE_FINAL_COUNTERS: usize = 4;
pub const QKECCAK: usize = 5;

pub const K_BITS: usize = 64;

// The Keccak reduction commits a step's state every KECCAK_DEPTH rounds but the output, a lane per
// slot of q_keccak and a word per step: lane l of the state after KECCAK_DEPTH * m rounds is slot 25 m + l.
pub const KECCAK_DEPTH: usize = 12;
pub const KECCAK_MIN_STEPS_LOG: usize = 9; // the reduction is sized for at least one block of 512 steps

pub const MAX_TABLE_LOG: usize = 32;
pub const MIN_STACKED_LOG: usize = 15;
pub const MAX_STACKED_LOG: usize = 28;

const TRUNC_D: usize = 4;
const MIN_PIECE_LOG: usize = 10;

/// The pieces, `(offset in the column, log size)` by increasing offset, that commit the first `real` rows of a
/// column of `2^k` rows: the prefix rounded up to a multiple of `2^max(k - TRUNC_D, MIN_PIECE_LOG)`, cut by its binary digits.
fn pieces(k: usize, real: u64) -> Vec<(u64, usize)> {
    if k <= MIN_PIECE_LOG {
        return vec![(0, k)];
    }
    let g_log = (k - TRUNC_D).max(MIN_PIECE_LOG);
    let covered = (real.max(1).div_ceil(1 << g_log) << g_log).min(1 << k);
    if covered == 1 << k {
        return vec![(0, k)];
    }
    let mut off = 0;
    let mut out = Vec::new();
    for b in (g_log..k).rev().filter(|&b| covered >> b & 1 == 1) {
        out.push((off, b));
        off += 1 << b;
    }
    out
}

/// One piece of a truncated column, laid in the stack at `placement`.
pub struct Piece {
    pub column: usize,
    pub off: u64, // the piece's first row in its column
    pub placement: Placement,
}

/// The truncated q_keccak ("QT"): only its first `c` chunks of `2^(g_log)` steps are committed, as a region of
/// `2^(6 + tau)` words at `offset`, laid out `(steplo, slot, chunk)`. The rest of each slot is the public `pad`.
pub struct Qt {
    pub offset: u64,
    pub tau: usize,
    pub g_log: usize,
    pub c: u64,
    pub pad: [u64; 64],
}

/// What a padding step holds in each slot: zero but for slots `25..50`, lane `l` of the state after `KECCAK_DEPTH`
/// rounds on the zero state, at slot `25 + l`.
fn qt_pad() -> [u64; 64] {
    let mut pad = [0u64; 64];
    let mut state = [0u64; STATE_LANES];
    keccak_rounds(&mut state, KECCAK_DEPTH);
    pad[STATE_LANES..2 * STATE_LANES].copy_from_slice(&state);
    pad
}

fn bits(value: u64, len: usize) -> Vec<E> {
    (0..len).map(|bit| E::new(value >> bit & 1, 0, 0)).collect()
}

impl Qt {
    pub fn placement(&self) -> Placement {
        Placement {
            variables: 6 + self.tau,
            index: self.offset,
        }
    }

    /// `1 + sum_{i < c} eq(bits(i), y)`: the weight of the uncovered chunks at `y`.
    fn uncovered(&self, y: &[E]) -> E {
        ONE + E::sum(eq_kernel(y)[..self.c as usize].iter().copied())
    }

    /// `sum_s pad[s] eq(bits(s), x)`.
    fn pad_mle(&self, x: &[E]) -> E {
        E::sum(eq_kernel(x).into_iter().zip(self.pad).map(|(eq, p)| eq * K(p)))
    }

    /// The region point of a point `(steplo, chunk, slot)` of the full layout: `(steplo, slot, chunk)`.
    fn rotate(&self, point: &[E]) -> Vec<E> {
        let (steplo, rest) = point.split_at(self.g_log);
        let (chunk, slot) = rest.split_at(self.tau - self.g_log);
        [steplo, slot, chunk].concat()
    }

    /// The stack point and value of a claim on q_keccak's output (`QKECCAK`) or on one of its slot columns.
    fn claim(&self, column: usize, point: &[E], value: E, stack_log: usize) -> Option<(Vec<E>, E)> {
        let (region_point, value) = if column == QKECCAK {
            let (chunk, slot) = point[self.g_log..].split_at(self.tau - self.g_log);
            (self.rotate(point), value + self.pad_mle(slot) * self.uncovered(chunk))
        } else {
            let slot = qkeccak_slot(column)?;
            let (low, chunk) = point.split_at(self.g_log);
            let region = [low, &bits(slot as u64, 6), chunk].concat();
            (region, value + self.uncovered(chunk) * K(self.pad[slot]))
        };
        Some((self.placement().stack_point(&region_point, stack_log), value))
    }

    /// The ring switch's inputs for the committed region: the point rotated, the slices less the pad's share.
    pub fn ring_inputs(&self, point: Vec<E>, mut slices: Vec<E>) -> (Vec<E>, Vec<E>) {
        let (chunk, slot) = point[self.g_log..].split_at(self.tau - self.g_log);
        let (u, eq) = (self.uncovered(chunk), eq_kernel(slot));
        for (i, slice) in slices.iter_mut().enumerate() {
            let share = E::sum(
                eq.iter()
                    .zip(self.pad)
                    .filter(|(_, p)| p >> i & 1 == 1)
                    .map(|(e, _)| *e),
            );
            *slice += u * share;
        }
        (self.rotate(&point), slices)
    }
}

pub struct Layout<'a> {
    pub log_memory: usize,
    pub log_bytecode: usize,
    pub bytecode: &'a Bytecode,
    pub tables: Vec<Table>,
    pub push: Vec<BusBlock>,
    pub pull: Vec<BusBlock>,
    pub count: Vec<BusBlock>,
    /// Where each column sits, `None` for a column nothing commits: the `KECCAK` output lanes.
    pub placements: Vec<Option<Placement>>,
    /// The pieces of the truncated columns, by column then offset; empty unless the proof is truncated. A truncated
    /// column's placement keeps its `variables`, its `index` meaning nothing.
    pub pieces: Vec<Piece>,
    pub stack_log: usize,
    pub table_log_heights: Vec<usize>,
    pub keccak: KeccakShape,
    /// Set when a truncated proof commits q_keccak as a region (`placements[QKECCAK]` is then the region's).
    pub qt: Option<Qt>,
}

impl Layout<'_> {
    /// A claim on q_keccak or one of its slot columns under QT, as a stack point and value; `None` otherwise.
    pub fn qt_claim(&self, column: usize, point: &[E], value: E) -> Option<(Vec<E>, E)> {
        self.qt.as_ref()?.claim(column, point, value, self.stack_log)
    }
}

fn cols<const N: usize>(columns: &[&str], names: [&str; N]) -> [usize; N] {
    names.map(|name| {
        columns
            .iter()
            .position(|&column| column == name)
            .expect("unknown column")
    })
}

pub fn gpow(index: u64) -> E {
    GEN.pow(index)
}

pub fn constant(value: E) -> Form {
    Form::term(vec![], value)
}

fn col(index: usize, exponent: u64) -> Form {
    Form::term(vec![index], gpow(exponent))
}

fn prod(a: usize, b: usize, exponent: u64) -> Form {
    Form::term(vec![a.min(b), a.max(b)], gpow(exponent))
}

pub const SEP_STATE: E = ONE;
pub const SEP_MEM: E = GEN;
pub const SEP_BYTECODE: E = E::new(4, 0, 0); // GEN^2

#[derive(Clone, Debug, Default)]
pub struct Flushes {
    push: Vec<Vec<Form>>,
    pull: Vec<Vec<Form>>,
}

impl Flushes {
    fn pair(&mut self, push: Vec<Form>, pull: Vec<Form>) {
        self.push.push(push);
        self.pull.push(pull);
    }

    fn state_derived(&mut self, pc: usize, fp: usize, npc: Form, nfp: Form) {
        self.pair(
            vec![constant(SEP_STATE), npc, nfp],
            vec![constant(SEP_STATE), col(pc, 0), col(fp, 0)],
        );
    }

    fn state_step(&mut self, pc: usize, fp: usize) {
        self.state_derived(pc, fp, col(pc, 1), col(fp, 0));
    }

    /// A pull at `count` and a push at `counter`, the same otherwise.
    fn counted(&mut self, prefix: &[Form; 2], counter: Form, count: usize, suffix: &[Form]) {
        let side = |counter: Form| [&prefix[..], &[counter], suffix].concat();
        self.pair(side(counter), side(col(count, 0)));
    }

    fn bytecode(&mut self, pc: usize, count: usize, opcode: usize, operands: Vec<Form>) {
        let suffix = [vec![constant(gpow(opcode as u64))], operands].concat();
        self.counted(&[constant(SEP_BYTECODE), col(pc, 0)], col(count, 1), count, &suffix);
    }

    fn memory(&mut self, address: Form, count: usize, values: &[Form; 3]) {
        self.counted(&[constant(SEP_MEM), address], col(count, 1), count, values);
    }

    /// `memory` when the column `flag` is 1. The push's counter is `count·(1 + flag·(g + 1))`: `g·count` at 1, and
    /// the pull's `count` at 0, so the pair cancels and touches no cell.
    fn memory_if(&mut self, flag: usize, address: Form, count: usize, values: &[Form; 3]) {
        let counter = col(count, 0) + prod(flag, count, 1) + prod(flag, count, 0);
        self.counted(&[constant(SEP_MEM), address], counter, count, values);
    }

    /// The cell's limbs are the named columns, the missing top ones zero.
    fn memory_cols(&mut self, address: Form, count: usize, columns: &[usize]) {
        let values = std::array::from_fn(|limb| columns.get(limb).map_or(constant(ZERO), |&column| col(column, 0)));
        self.memory(address, count, &values);
    }
}

/// One instruction's table: its columns, its bus flushes, its constraints.
pub struct Table {
    pub opcode: usize, // also its index in the table list, so g^opcode is its bytecode tag
    pub columns: Vec<String>,
    pub padding: Vec<E>, // the row a padding row holds
    pub flushes: Flushes,
    pub constraints: fn(&[E]) -> Vec<E>,
}

impl Table {
    pub fn n_constraints(&self) -> usize {
        (self.constraints)(&vec![ZERO; self.width()]).len()
    }

    pub fn width(&self) -> usize {
        self.columns.len()
    }

    fn count_columns(&self) -> impl Iterator<Item = usize> {
        (0..self.width()).filter(|&i| self.columns[i].starts_with("cnt"))
    }
}

fn flushes_arith(opcode: usize, multiply: bool) -> Flushes {
    let [pc, fp, o_a, o_b, o_c, cnt_a, cnt_b, cnt_c, cnt_bc] = cols(
        ARITH_COLUMNS,
        ["pc", "fp", "o_a", "o_b", "o_c", "cnt_a", "cnt_b", "cnt_c", "cnt_bc"],
    );
    let va = cols(ARITH_COLUMNS, ["va_0", "va_1", "va_2"]);
    let vb = cols(ARITH_COLUMNS, ["vb_0", "vb_1", "vb_2"]);
    let mut flushes = Flushes::default();
    flushes.state_step(pc, fp);
    let operands = vec![col(o_a, 0), col(o_b, 0), col(o_c, 0), constant(ZERO), constant(ZERO)];
    flushes.bytecode(pc, cnt_bc, opcode, operands);
    flushes.memory_cols(prod(fp, o_a, 0), cnt_a, &va);
    flushes.memory_cols(prod(fp, o_b, 0), cnt_b, &vb);
    const TOWER_LANES: [&[(usize, usize)]; 3] = [
        &[(0, 0), (1, 2), (2, 1)],
        &[(0, 1), (1, 0), (1, 2), (2, 1), (2, 2)],
        &[(0, 2), (1, 1), (2, 0), (2, 2)],
    ];
    let result = if multiply {
        TOWER_LANES.map(|lane| Form::sum(lane.iter().map(|&(j, k)| prod(va[j], vb[k], 0))))
    } else {
        std::array::from_fn(|i| col(va[i], 0) + col(vb[i], 0))
    };
    flushes.memory(prod(fp, o_c, 0), cnt_c, &result);
    flushes
}

fn flushes_set() -> Flushes {
    let [pc, fp, o, cnt, cnt_bc] = cols(SET_COLUMNS, ["pc", "fp", "o", "cnt", "cnt_bc"]);
    let k = cols(SET_COLUMNS, ["k_0", "k_1", "k_2"]);
    let mut flushes = Flushes::default();
    flushes.state_step(pc, fp);
    let operands = vec![col(o, 0), col(k[0], 0), col(k[1], 0), col(k[2], 0), constant(ZERO)];
    flushes.bytecode(pc, cnt_bc, OP_SET, operands);
    flushes.memory_cols(prod(fp, o, 0), cnt, &k);
    flushes
}

fn flushes_deref() -> Flushes {
    let [pc, fp, o1, o2, o3, f_pc, f_fp, ptr] =
        cols(DEREF_COLUMNS, ["pc", "fp", "o1", "o2", "o3", "f_pc", "f_fp", "ptr"]);
    let [cnt_ptr, cnt_target, cnt_local, cnt_bc] =
        cols(DEREF_COLUMNS, ["cnt_ptr", "cnt_target", "cnt_local", "cnt_bc"]);
    let v3 = cols(DEREF_COLUMNS, ["v3_0", "v3_1", "v3_2"]);

    let gated = |lane: usize| [col(lane, 0), prod(f_pc, lane, 0), prod(f_fp, lane, 0)];

    // v2 = (1 + f_pc + f_fp)*v3 + f_pc*(g^2*pc) + f_fp*fp, lane-wise: only the low lane takes the two K-valued sources.
    let store = [
        Form::sum(gated(v3[0]).into_iter().chain([prod(f_pc, pc, 2), prod(f_fp, fp, 0)])),
        Form::sum(gated(v3[1])),
        Form::sum(gated(v3[2])),
    ];
    let mut flushes = Flushes::default();
    flushes.state_step(pc, fp);
    let operands = vec![col(o1, 0), col(o2, 0), col(o3, 0), col(f_pc, 0), col(f_fp, 0)];
    flushes.bytecode(pc, cnt_bc, OP_DEREF, operands);
    flushes.memory_cols(prod(fp, o1, 0), cnt_ptr, &[ptr]);
    flushes.memory(prod(ptr, o2, 0), cnt_target, &store);
    flushes.memory_cols(prod(fp, o3, 0), cnt_local, &v3);
    flushes
}

fn flushes_jump() -> Flushes {
    let [pc, fp, o_c, o_d, o_f, cond, dest, frame, b] = cols(
        JUMP_COLUMNS,
        ["pc", "fp", "o_c", "o_d", "o_f", "v_cond", "v_pc", "v_fp", "b"],
    );
    let [cnt_c, cnt_d, cnt_f, cnt_bc] = cols(JUMP_COLUMNS, ["cnt_c", "cnt_d", "cnt_f", "cnt_bc"]);
    let mut flushes = Flushes::default();
    // next_pc = b*dest + (b+1)*g*pc, next_fp = b*frame + (b+1)*fp, both derived.
    flushes.state_derived(
        pc,
        fp,
        prod(b, dest, 0) + prod(b, pc, 1) + col(pc, 1),
        prod(b, frame, 0) + prod(b, fp, 0) + col(fp, 0),
    );
    let operands = vec![col(o_c, 0), col(o_d, 0), col(o_f, 0), constant(ZERO), constant(ZERO)];
    flushes.bytecode(pc, cnt_bc, OP_JUMP, operands);
    flushes.memory_cols(prod(fp, o_c, 0), cnt_c, &[cond]);
    flushes.memory_cols(prod(fp, o_d, 0), cnt_d, &[dest]);
    flushes.memory_cols(prod(fp, o_f, 0), cnt_f, &[frame]);
    flushes
}

fn jump_constraints(columns: &[E]) -> Vec<E> {
    let [condition, inverse, flag] = cols(JUMP_COLUMNS, ["v_cond", "w", "b"]).map(|index| columns[index]);
    vec![flag + condition * inverse, condition * (flag + ONE)]
}

/// The lanes each of a sponge state's thirteen cells holds, `(lo, hi)`, in the even layout or the odd one, `None`
/// marking a limb that holds no lane: nine rate cells, lanes 0..16 two to a cell then lane 16 in a low limb (even), or
/// lane 0 in a high limb then lanes 1..17 two to a cell (odd), then the capacity lanes 17..25.
fn keccak_cell_lanes(odd: bool) -> [(Option<usize>, Option<usize>); KECCAK_CELLS] {
    std::array::from_fn(|cell| match (odd, cell) {
        (_, 9..) | (true, 1..) => (Some(2 * cell - 1), Some(2 * cell)),
        (true, _) => (None, Some(0)),
        (false, 8) => (Some(16), None),
        (false, _) => (Some(2 * cell), Some(2 * cell + 1)),
    })
}

/// A cell limb holding lane `even` in the even layout and `odd` in the odd one, as `flag` selects: `a + flag·(a + b)`,
/// a missing lane being the column `missing`, or zero.
fn layout_limb(even: Option<usize>, odd: Option<usize>, flag: usize, missing: Option<usize>) -> Form {
    let (a, b) = (even.or(missing), odd.or(missing));
    if a == b {
        return a.map_or(constant(ZERO), |a| col(a, 0));
    }
    Form::sum(
        [
            a.map(|a| col(a, 0)),
            a.map(|a| prod(flag, a, 0)),
            b.map(|b| prod(flag, b, 0)),
        ]
        .into_iter()
        .flatten(),
    )
}

fn flushes_keccak() -> Flushes {
    let [pc, fp, free] = cols(KECCAK_HEAD, ["pc", "fp", "free"]);
    let o_m = cols(KECCAK_HEAD, ["o_m0", "o_m1", "o_m2", "o_m3"]);
    let [o_tail, o_b8, o_cap, o_out] = cols(KECCAK_HEAD, ["o_tail", "o_b8", "o_cap", "o_out"]);
    let [in_odd, out_odd, out_two, out_all] = cols(KECCAK_HEAD, ["odd_in", "odd_out", "out_two", "out_all"]);
    let mut flushes = Flushes::default();
    flushes.state_step(pc, fp);
    let operands = o_m
        .into_iter()
        .chain([o_tail, o_b8, o_cap, o_out, in_odd, out_odd, out_two, out_all]);
    flushes.bytecode(pc, KECCAK_CNT_BC, OP_KECCAK, operands.map(|i| col(i, 0)).collect());
    // The thirteen cells read (four addressed ones, the tail run, b8, the cap run), then the thirteen written, each
    // holding its lanes in the layout its flag names: a read cell's free lane is the column `free`, a written one's zero.
    // Output cell 0 is always written, cell 1 when `out_two` is 1, and cells 2..13 when `out_all` is 1.
    let address = |cell: usize| match cell {
        0..4 => prod(fp, o_m[cell], 0),
        4..8 => prod(fp, o_tail, cell as u64 - 4),
        8 => prod(fp, o_b8, 0),
        _ => prod(fp, o_cap, cell as u64 - 9),
    };
    let (even, odd) = (keccak_cell_lanes(false), keccak_cell_lanes(true));
    for (side, lanes, flag, missing) in [(0, KECCAK_IN, in_odd, Some(free)), (1, KECCAK_OUT, out_odd, None)] {
        let lane = |lane: Option<usize>| lane.map(|lane| lanes + lane);
        for cell in 0..KECCAK_CELLS {
            let limb = |e: Option<usize>, o: Option<usize>| layout_limb(lane(e), lane(o), flag, missing);
            let values = [
                limb(even[cell].0, odd[cell].0),
                limb(even[cell].1, odd[cell].1),
                constant(ZERO),
            ];
            let count = KECCAK_CNT + KECCAK_CELLS * side + cell;
            match (side, cell) {
                (0, _) => flushes.memory(address(cell), count, &values),
                (_, 0) => flushes.memory(prod(fp, o_out, 0), count, &values),
                _ => {
                    let flag = if cell == 1 { out_two } else { out_all };
                    flushes.memory_if(flag, prod(fp, o_out, cell as u64), count, &values);
                }
            }
        }
    }
    flushes
}

pub const OP_XOR: usize = 0;
pub const OP_MUL: usize = 1;
pub const OP_SET: usize = 2;
pub const OP_DEREF: usize = 3;
pub const OP_JUMP: usize = 4;
pub const OP_KECCAK: usize = 5;
pub const N_TABLES: usize = 6;

#[rustfmt::skip]
const ARITH_COLUMNS: &[&str] = &["pc", "fp", "o_a", "o_b", "o_c", "va_0", "va_1", "va_2", "vb_0", "vb_1", "vb_2", "cnt_a", "cnt_b", "cnt_c", "cnt_bc"];
const SET_COLUMNS: &[&str] = &["pc", "fp", "o", "k_0", "k_1", "k_2", "cnt", "cnt_bc"];
#[rustfmt::skip]
const DEREF_COLUMNS: &[&str] = &["pc", "fp", "o1", "o2", "o3", "f_pc", "f_fp", "ptr", "v3_0", "v3_1", "v3_2", "cnt_ptr", "cnt_target", "cnt_local", "cnt_bc"];
#[rustfmt::skip]
const JUMP_COLUMNS: &[&str] = &["pc", "fp", "o_c", "o_d", "o_f", "v_cond", "v_pc", "v_fp", "cnt_c", "cnt_d", "cnt_f", "cnt_bc", "w", "b"];
// KECCAK's columns: these, then the input lanes `in_l`, the output lanes `out_l`, a read count `cnt_c` for each cell
// read then each written, and `cnt_bc`. The fifty value lanes are not committed in the table: an input lane is one of
// q_keccak's slots, and the output lanes' claims are the Keccak reduction's claim on the output.
#[rustfmt::skip]
const KECCAK_HEAD: &[&str] = &["pc", "fp", "o_m0", "o_m1", "o_m2", "o_m3", "o_tail", "o_b8", "o_cap", "o_out", "odd_in", "odd_out", "out_two", "out_all", "free"];
const KECCAK_CELLS: usize = 13; // a sponge state, read then written
pub const KECCAK_IN: usize = KECCAK_HEAD.len();
pub const KECCAK_OUT: usize = KECCAK_IN + STATE_LANES;
const KECCAK_CNT: usize = KECCAK_OUT + STATE_LANES;
const KECCAK_CNT_BC: usize = KECCAK_CNT + 2 * KECCAK_CELLS;

/// The local KECCAK column riding spare slot `k` (`cpu::keccak_value` in the prover): its
/// committed columns in local order, the ones before the value lanes then the read counts
/// after them.
const fn spare_local(k: usize) -> usize {
    if k < KECCAK_IN { k } else { KECCAK_CNT + (k - KECCAK_IN) }
}

/// Slots of the run the committed states occupy, and the ones they leave over.
const QKECCAK_SLOTS: usize = STATE_LANES * (crate::hash::ROUNDS / KECCAK_DEPTH);
const QKECCAK_SPARE: usize = (1 << log2_ceil(QKECCAK_SLOTS as u64)) - QKECCAK_SLOTS;
// The same guard the prover carries (`cpu::layout`): every spare slot needs a committed column
// of the table to put in it, distinct from the others and not one of the value lanes. It holds
// at this depth, not at every divisor of 24, and the two sides must refuse together.
const _: () = {
    let mut k = 0;
    while k < QKECCAK_SPARE {
        let local = spare_local(k);
        assert!(k == 0 || spare_local(k - 1) < local);
        assert!(local < KECCAK_IN || (local >= KECCAK_CNT && local < TABLE_WIDTHS[OP_KECCAK]));
        k += 1;
    }
};

/// The slot of `q_keccak` the global column `column` IS, if it is one: an input lane (slot =
/// lane), or one of the KECCAK table's own columns riding a slot past the committed states.
/// The reduction weighs nothing on those spare slots, so the stack carries those columns
/// there rather than a block of zeros and a block of its own.
fn qkeccak_slot(column: usize) -> Option<usize> {
    let local = column.checked_sub(GLOBAL_COLUMN_BASES[OP_KECCAK])?;
    if let Some(k) = (0..QKECCAK_SPARE).find(|&k| spare_local(k) == local) {
        return Some(QKECCAK_SLOTS + k);
    }
    (KECCAK_IN..KECCAK_OUT).contains(&local).then(|| local - KECCAK_IN)
}

fn keccak_columns() -> Vec<String> {
    let lanes = |prefix: &'static str| (0..STATE_LANES).map(move |lane| format!("{prefix}_{lane}"));
    let counts = (0..2 * KECCAK_CELLS).map(|cell| format!("cnt_{cell}"));
    let head = KECCAK_HEAD.iter().map(|name| name.to_string());
    head.chain(lanes("in"))
        .chain(lanes("out"))
        .chain(counts)
        .chain(["cnt_bc".to_string()])
        .collect()
}

pub fn tables() -> Vec<Table> {
    let table = |opcode, columns: Vec<String>, flushes, constraints| Table {
        opcode,
        padding: vec![ZERO; columns.len()],
        columns,
        flushes,
        constraints,
    };
    let named = |columns: &[&str]| columns.iter().map(|name| name.to_string()).collect();
    let mut keccak = table(OP_KECCAK, keccak_columns(), flushes_keccak(), |_| vec![]);
    // A padding row is zero, but for KECCAK's output lanes: a padding row permutes zeros.
    let mut output = [0; STATE_LANES];
    keccak_f(&mut output);
    for (padding, lane) in keccak.padding[KECCAK_OUT..].iter_mut().zip(output) {
        *padding = E::new(lane, 0, 0);
    }
    vec![
        table(OP_XOR, named(ARITH_COLUMNS), flushes_arith(OP_XOR, false), |_| vec![]),
        table(OP_MUL, named(ARITH_COLUMNS), flushes_arith(OP_MUL, true), |_| vec![]),
        table(OP_SET, named(SET_COLUMNS), flushes_set(), |_| vec![]),
        table(OP_DEREF, named(DEREF_COLUMNS), flushes_deref(), |_| vec![]),
        table(OP_JUMP, named(JUMP_COLUMNS), flushes_jump(), jump_constraints),
        keccak,
    ]
}

const TABLE_WIDTHS: [usize; N_TABLES] = [
    ARITH_COLUMNS.len(),
    ARITH_COLUMNS.len(),
    SET_COLUMNS.len(),
    DEREF_COLUMNS.len(),
    JUMP_COLUMNS.len(),
    KECCAK_CNT_BC + 1,
];
pub const GLOBAL_COLUMN_BASES: [usize; N_TABLES] = {
    let mut bases = [NUM_GLOBAL_COLUMNS; N_TABLES];
    let mut table = 1;
    while table < N_TABLES {
        bases[table] = bases[table - 1] + TABLE_WIDTHS[table - 1];
        table += 1;
    }
    bases
};

/// Each table's least log height: 0, but the Keccak reduction's block of steps for KECCAK.
const TABLE_LOG_FLOORS: [usize; N_TABLES] = [0, 0, 0, 0, 0, KECCAK_MIN_STEPS_LOG];

pub fn build_layout<'a>(
    bytecode: &'a Bytecode,
    log_memory: u64,
    table_rows: &[u64],
    mem_count: Option<u64>, // the announced memory cells of a truncated proof
) -> Result<Layout<'a>, VerifierError> {
    let log_bytecode = log2_strict(bytecode.len() as u64)?;
    require(
        (16..=32).contains(&log_memory)
            && table_rows.iter().all(|&rows| rows <= 1 << MAX_TABLE_LOG)
            && (BUS_BITS..=32 + BUS_BITS).contains(&log_bytecode)
            && mem_count.is_none_or(|count| (1..=1 << log_memory).contains(&count)),
        VerifierError::AnnouncedTableSizes,
    )?;
    let log_bytecode = log_bytecode - BUS_BITS;
    let log_memory = log_memory as usize;
    // A table's height is the least power of two holding its rows.
    let table_log_heights: Vec<usize> = table_rows
        .iter()
        .zip(TABLE_LOG_FLOORS)
        .map(|(&rows, floor)| log2_ceil(rows).max(floor))
        .collect();

    let tables = tables();
    let mut push = Vec::new();
    let mut pull = Vec::new();
    let mut count = Vec::new();
    for ((table, &height), &rows) in tables.iter().zip(&table_log_heights).zip(table_rows) {
        let block = |coordinates: &Vec<Form>| BusBlock {
            log_rows: height,
            rows,
            coordinates: coordinates.clone(),
            owner: table.opcode,
        };
        push.extend(table.flushes.push.iter().map(block));
        pull.extend(table.flushes.pull.iter().map(block));
        count.extend(table.count_columns().map(|local| block(&vec![col(local, 0)])));
    }

    // Every column's log size, in global order: the framework's, q_keccak's, then each table's block.
    let keccak = KeccakShape {
        log_steps: table_log_heights[OP_KECCAK],
        depth: KECCAK_DEPTH,
    };
    let qkeccak_kappa = keccak.log_steps + keccak.slot_bits();
    let mut kappas = vec![
        log_memory,
        log_memory,
        log_memory,
        log_memory,
        log_bytecode,
        qkeccak_kappa,
    ];
    for table in &tables {
        kappas.extend(vec![table_log_heights[table.opcode]; table.width()]);
    }

    // A column q_keccak carries gets no block of its own: it is one of its slots, 2^height words,
    // whether an input lane or a KECCAK column riding a spare slot (`qkeccak_slot`). An output
    // lane is not committed at all, nothing ever opening it.
    let base = GLOBAL_COLUMN_BASES[OP_KECCAK];
    let outputs = base + KECCAK_OUT..base + KECCAK_OUT + STATE_LANES;
    let blocks: Vec<usize> = (0..kappas.len())
        .filter(|column| qkeccak_slot(*column).is_none() && !outputs.contains(column))
        .collect();
    // The stacked items: a block each, or its pieces for a truncated column (the memory and the table columns).
    let mut items: Vec<(usize, u64, usize)> = Vec::new(); // (column, offset in the column, kappa)
    for &column in &blocks {
        let real = match (mem_count, column) {
            (Some(count), 0..=MEMORY_FINAL_COUNTERS) => Some(count),
            (Some(_), NUM_GLOBAL_COLUMNS..) => {
                let table = GLOBAL_COLUMN_BASES.iter().rposition(|&base| base <= column);
                table.map(|table| table_rows[table])
            }
            _ => None,
        };
        match real {
            Some(real) => items.extend(
                pieces(kappas[column], real)
                    .into_iter()
                    .map(|(off, b)| (column, off, b)),
            ),
            None => items.push((column, 0, kappas[column])),
        }
    }
    let stack = |items: &[(usize, u64, usize)]| {
        let item_kappas: Vec<usize> = items.iter().map(|&(.., kappa)| kappa).collect();
        stack_offsets(&item_kappas)
    };
    let (mut item_offsets, total_log) = stack(&items);
    let mut stack_log = MIN_STACKED_LOG.max(total_log); // Floor at the PCS minimum
    // A truncated proof may commit q_keccak as a region instead (QT), when that makes the stack smaller.
    let committed = |placed: u64, mu: usize| placed.div_ceil(1 << (mu - 6)) << (mu - 6);
    let mut qt = None;
    if mem_count.is_some() {
        let tau = table_log_heights[OP_KECCAK];
        let g_log = tau.min(tau.saturating_sub(TRUNC_D).max(MIN_PIECE_LOG));
        let g_rows = 1u64 << g_log;
        let c = table_rows[OP_KECCAK].max(1).div_ceil(g_rows).min(1 << (tau - g_log));
        let others: Vec<_> = items.iter().copied().filter(|item| item.0 != QKECCAK).collect();
        let (other_offsets, _) = stack(&others);
        let other_words: u64 = others.iter().map(|&(.., kappa)| 1u64 << kappa).sum();
        let offset = other_words.next_multiple_of(1 << (6 + tau));
        let qt_log = MIN_STACKED_LOG.max(log2_ceil(offset + (1 << (6 + tau))));
        let whole: u64 = items.iter().map(|&(.., kappa)| 1u64 << kappa).sum();
        if committed(offset + (c << (6 + g_log)), qt_log) < committed(whole, stack_log) {
            qt = Some(Qt {
                offset,
                tau,
                g_log,
                c,
                pad: qt_pad(),
            });
            (items, item_offsets, stack_log) = (others, other_offsets, qt_log);
        }
    }
    let mut offsets: BTreeMap<usize, u64> = BTreeMap::new();
    let mut piece_list = Vec::new();
    for (&(column, off, kappa), index) in items.iter().zip(item_offsets) {
        offsets.entry(column).or_insert(index);
        if mem_count.is_some() && column != BYTECODE_FINAL_COUNTERS && column != QKECCAK {
            piece_list.push(Piece {
                column,
                off,
                placement: Placement {
                    variables: kappa,
                    index,
                },
            });
        }
    }
    if let Some(qt) = &qt {
        offsets.insert(QKECCAK, qt.offset);
    }
    let placement = |(column, &kappa): (usize, &usize)| {
        let index = match qkeccak_slot(column) {
            Some(slot) => offsets[&QKECCAK] + ((slot as u64) << kappa),
            None => *offsets.get(&column)?,
        };
        Some(Placement {
            variables: kappa,
            index,
        })
    };
    let placements = kappas.iter().enumerate().map(placement).collect();
    Ok(Layout {
        log_memory,
        log_bytecode,
        bytecode,
        tables,
        push,
        pull,
        count,
        placements,
        pieces: piece_list,
        stack_log,
        table_log_heights,
        keccak,
        qt,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multilinear::multilinear_eval;

    #[test]
    fn qt_pad_is_the_state_after_the_first_rounds() {
        let pad = qt_pad();
        assert!(pad[..25].iter().chain(&pad[50..]).all(|&word| word == 0));
        assert_ne!(pad[25], 0);
        let mut state = [0u64; STATE_LANES];
        keccak_rounds(&mut state, crate::hash::ROUNDS);
        let mut full = [0u64; STATE_LANES];
        keccak_f(&mut full);
        assert_eq!(state, full);
    }

    #[test]
    fn qt_claims_equal_the_full_layout() {
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let (tau, g_log, c) = (12, 10, 2u64);
        let mut pad = [0u64; 64];
        pad.iter_mut().for_each(|p| *p = next());
        let qt = Qt {
            offset: 0,
            tau,
            g_log,
            c,
            pad,
        };
        let g = 1usize << g_log;
        // The full layout, index step + 2^tau * slot, and the region, index steplo + g * (slot + 64 * chunk).
        let mut full = vec![0u64; 64 << tau];
        let mut region = vec![0u64; 64 << tau];
        for slot in 0..64 {
            for step in 0..1usize << tau {
                let chunk = step / g;
                let word = if (chunk as u64) < c { next() } else { pad[slot] };
                full[step + (slot << tau)] = word;
                region[step % g + g * (slot + 64 * chunk)] = if (chunk as u64) < c { word } else { 0 };
            }
        }
        let mut random = || E::new(next(), next(), next());
        let rho: Vec<E> = (0..6 + tau).map(|_| random()).collect();
        let (chunk, slot) = rho[g_log..].split_at(tau - g_log);
        let at = |words: &[u64], point: &[E]| {
            let words: Vec<K> = words.iter().map(|&w| K(w)).collect();
            multilinear_eval(&words, point).unwrap()
        };
        let rotated = qt.rotate(&rho);
        let expected = at(&full, &rho);
        assert_eq!(expected, at(&region, &rotated) + qt.pad_mle(slot) * qt.uncovered(chunk));
        // A slot column at zeta: the same identity on one slot's `2^tau` words.
        let zeta = &rho[..tau];
        let slot_id = 7;
        let region_point = [&zeta[..g_log], &bits(slot_id as u64, 6), &zeta[g_log..]].concat();
        let column = at(&full[slot_id << tau..(slot_id + 1) << tau], zeta);
        let shift = qt.uncovered(&zeta[g_log..]) * K(pad[slot_id]);
        assert_eq!(column, at(&region, &region_point) + shift);
        // The ring switch: bit `i` of the full layout's words at `rho` is the region's at `rotated`, less the pad's share.
        let (_, corrected) = qt.ring_inputs(rho.clone(), (0..64).map(|i| at_bit(&full, &rho, i)).collect());
        for (i, slice) in corrected.into_iter().enumerate() {
            assert_eq!(slice, at_bit(&region, &rotated, i));
        }
    }

    fn at_bit(words: &[u64], point: &[E], i: usize) -> E {
        let bits: Vec<K> = words.iter().map(|&w| K(w >> i & 1)).collect();
        multilinear_eval(&bits, point).unwrap()
    }

    #[test]
    fn pieces_cover_the_prefix() {
        assert_eq!(pieces(19, 280940), [(0, 18), (262144, 15)]);
        assert_eq!(pieces(22, 2374000), [(0, 21), (2097152, 19)]);
        assert_eq!(pieces(10, 5), [(0, 10)]);
        assert_eq!(pieces(19, 1 << 19), [(0, 19)]);
        assert_eq!(pieces(19, 0), [(0, 15)]);
    }

    #[test]
    fn pieces_and_pad_give_the_column() {
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut random = || E::new(next(), next(), next());
        let k = 12;
        let zeta: Vec<E> = (0..k).map(|_| random()).collect();
        for pad in [ZERO, ONE] {
            let layout = pieces(k, 2500);
            assert_eq!(layout, [(0, 11), (2048, 10)]);
            let covered = 3072;
            let column: Vec<E> = (0..1 << k)
                .map(|row| if row < covered { random() } else { pad })
                .collect();
            let (mut e_sum, mut sum) = (ZERO, ZERO);
            for &(off, b) in &layout {
                let e = (b..k).fold(ONE, |e, i| e * if off >> i & 1 == 1 { zeta[i] } else { ONE + zeta[i] });
                let piece = &column[off as usize..off as usize + (1 << b)];
                sum += e * multilinear_eval(piece, &zeta[..b]).unwrap();
                e_sum += e;
            }
            let expected = multilinear_eval(&column, &zeta).unwrap();
            assert_eq!(expected, sum + pad * (ONE + e_sum));
        }
    }
}
