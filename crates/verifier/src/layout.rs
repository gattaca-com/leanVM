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
/// to the constraint system bumps it by hand. Mirrored in `formal/Statement/LeanVmVerifier.lean`.
pub const TAG: &[u8] = b"leanvm-v1";

// The columns no instruction table owns. They come first in the global column numbering, the tables after.
pub const NUM_GLOBAL_COLUMNS: usize = 5;
pub const MEMORY_0: usize = 0;
pub const MEMORY_1: usize = 1;
pub const MEMORY_FINAL_COUNTERS: usize = 2;
pub const BYTECODE_FINAL_COUNTERS: usize = 3;
pub const QKECCAK: usize = 4;

pub const K_BITS: usize = 64;

// The Keccak reduction commits a step's state every KECCAK_DEPTH rounds but the output, a lane per
// slot of q_keccak and a word per step: lane l of the state after KECCAK_DEPTH * m rounds is slot 25 m + l.
pub const KECCAK_DEPTH: usize = 12;
pub const KECCAK_MIN_STEPS_LOG: usize = 9; // the reduction is sized for at least one block of 512 steps

/// Every size a proof announces or commits is at most `2^30`: a table's rows, the memory, the bytecode and the stack.
pub const MAX_TABLE_LOG: usize = 30;
/// The least memory a proof announces, as a log: the public input's two cells.
pub const MIN_LOG_MEMORY: usize = 1;
pub const MAX_LOG_MEMORY: usize = 30;
/// The largest bytecode, as a log of its instructions.
pub const MAX_LOG_BYTECODE: usize = 30;
/// The PCS's least stack, as a log.
pub const MIN_STACKED_LOG: usize = 16;
pub const MAX_STACKED_LOG: usize = 30;

/// A truncated column keeps its rows to within `2^TRUNC_D` of its real count, in pieces of at least `2^MIN_PIECE_LOG` rows.
pub const TRUNC_D: usize = 4;
pub const MIN_PIECE_LOG: usize = 10;

/// What the rate word announces of a truncated proof: its memory cells, and whether q_keccak is a region (QT).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Truncation {
    pub mem_count: u64,
    pub qt: bool,
}

/// The pieces, `(offset in the column, log size)` by increasing offset, that commit the first `real` rows of a
/// column of `2^k` rows: the prefix rounded up to a multiple of `2^max(k - TRUNC_D, MIN_PIECE_LOG)`, cut by its binary digits.
pub fn pieces(k: usize, real: u64) -> Vec<(u64, usize)> {
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

/// One piece of the truncated q_keccak: rows `off..off + 2^b` of every slot of one table half, laid out `(row, slot)`
/// in `2^(b + 6)` words at `placement`.
#[derive(Clone, Debug)]
pub struct QtPiece {
    pub half: usize,
    pub off: u64,
    pub placement: Placement,
}

impl QtPiece {
    /// The piece's log rows.
    pub fn b(&self) -> usize {
        self.placement.variables - 6
    }
}

/// The truncated q_keccak ("QT"): each table half is committed as the pieces of its covered rows, cut as a truncated
/// column is, each piece all 64 slots of its rows. Past its covered rows a half is its public `pad`, a row of 64 words.
pub struct Qt {
    /// `log2` of the steps each table is completed to.
    pub log_steps: usize,
    pub pieces: Vec<QtPiece>,
    pub pads: [[u64; 64]; 2],
}

/// What a padding step of table half `half` holds in each slot: its input state in slots `0..25`, and slot `25 + l`
/// the lane `l` of the state after `KECCAK_DEPTH` rounds on it. Zero in the spare slots.
pub fn qt_pad(half: usize) -> [u64; 64] {
    let mut pad = [0u64; 64];
    let mut state = if half == 0 { [0u64; STATE_LANES] } else { keccak64_pad() };
    pad[..STATE_LANES].copy_from_slice(&state);
    keccak_rounds(&mut state, KECCAK_DEPTH);
    pad[STATE_LANES..2 * STATE_LANES].copy_from_slice(&state);
    pad
}

fn bits(value: u64, len: usize) -> Vec<E> {
    (0..len).map(|bit| E::new(value >> bit & 1, 0)).collect()
}

impl Qt {
    /// The pieces of a half.
    fn half_pieces(&self, half: usize) -> impl Iterator<Item = &QtPiece> {
        self.pieces.iter().filter(move |piece| piece.half == half)
    }

    /// `e_j` of a piece at the row point `zeta`: the weight of its rows in the column.
    fn weight(&self, piece: &QtPiece, zeta: &[E]) -> E {
        (piece.b()..self.log_steps).fold(ONE, |e, i| {
            e * if piece.off >> i & 1 == 1 {
                zeta[i]
            } else {
                ONE + zeta[i]
            }
        })
    }

    /// `1 + sum_j e_j`, the weight of the rows past a half's pieces.
    fn uncovered(&self, half: usize, zeta: &[E]) -> E {
        ONE + E::sum(self.half_pieces(half).map(|piece| self.weight(piece, zeta)))
    }

    /// `sum_s pad[half][s] eq(bits(s), slot)`.
    fn pad_mle(&self, half: usize, slot: &[E]) -> E {
        E::sum(
            eq_kernel(slot)
                .into_iter()
                .zip(self.pads[half])
                .map(|(eq, p)| eq * K(p)),
        )
    }

    /// `zeta` padded with zeros to the rows of a half: the point of the same column in the completed table.
    fn extend(&self, zeta: &[E]) -> Vec<E> {
        let mut point = zeta.to_vec();
        point.resize(self.log_steps, ZERO);
        point
    }

    /// The stack claim and value of a claim on q_keccak's output (`QKECCAK`, at `(row, table, slot)`) or on one of its
    /// slot columns (at the rows of its table).
    fn claim(&self, column: usize, point: &[E], value: E, stack_log: usize) -> Option<StackClaimParts> {
        let mut parts = Vec::new();
        let value = if column == QKECCAK {
            let (row, rest) = point.split_at(self.log_steps);
            let (table, slot) = rest.split_at(TABLE_BITS);
            let mut value = value;
            for half in 0..HASH_TABLES.len() {
                // The tables' weights are the table coordinates' eq.
                let eq = table
                    .iter()
                    .enumerate()
                    .fold(ONE, |eq, (bit, &x)| eq * if half >> bit & 1 == 1 { x } else { ONE + x });
                value += eq * self.pad_mle(half, slot) * self.uncovered(half, row);
                for piece in self.half_pieces(half) {
                    let at = [&row[..piece.b()], slot].concat();
                    parts.push((
                        eq * self.weight(piece, row),
                        piece.placement.stack_point(&at, stack_log),
                    ));
                }
            }
            value
        } else {
            let (half, slot) = qkeccak_slot(column)?;
            let row = self.extend(point);
            for piece in self.half_pieces(half) {
                let at = [&row[..piece.b()], &bits(slot as u64, 6)].concat();
                parts.push((self.weight(piece, &row), piece.placement.stack_point(&at, stack_log)));
            }
            value + self.uncovered(half, &row) * K(self.pads[half][slot])
        };
        Some((parts, value))
    }

    /// The ring claims the reduction's input opening splits into, one per piece: the piece's point, and the weight `c_j` the
    /// piece's slices have in the opening's `slices`. With them, `slices_z = sum_j c_j s_{j,z} + pad share`, where
    /// the pad share is returned.
    pub fn ring_split(&self, point: &[E]) -> Vec<(Vec<E>, E)> {
        let (row, rest) = point.split_at(self.log_steps);
        let (table, slot) = rest.split_at(TABLE_BITS);
        self.pieces
            .iter()
            .map(|piece| {
                let eq = table.iter().enumerate().fold(ONE, |eq, (bit, &x)| {
                    eq * if piece.half >> bit & 1 == 1 { x } else { ONE + x }
                });
                ([&row[..piece.b()], slot].concat(), eq * self.weight(piece, row))
            })
            .collect()
    }

    /// The bit `i` of the pad's share of the opening's slice: `sum_t eq_t sum_s pad[t][s]_i eq(bits(s), slot) (1 + sum_j e_tj)`.
    pub fn pad_share(&self, point: &[E]) -> Vec<E> {
        let (row, rest) = point.split_at(self.log_steps);
        let (table, slot) = rest.split_at(TABLE_BITS);
        let eq_slot = eq_kernel(slot);
        let mut share = vec![ZERO; K_BITS];
        for half in 0..HASH_TABLES.len() {
            let eq = table
                .iter()
                .enumerate()
                .fold(ONE, |eq, (bit, &x)| eq * if half >> bit & 1 == 1 { x } else { ONE + x });
            let u = eq * self.uncovered(half, row);
            for (i, slice) in share.iter_mut().enumerate() {
                let mass = E::sum(
                    eq_slot
                        .iter()
                        .zip(self.pads[half])
                        .filter(|(_, p)| p >> i & 1 == 1)
                        .map(|(e, _)| *e),
                );
                *slice += u * mass;
            }
        }
        share
    }
}

/// A claim on the stack as parts `(weight, point)`, and its value.
pub type StackClaimParts = (Vec<(E, Vec<E>)>, E);

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
    /// A claim on q_keccak or one of its slot columns under QT, as weighted stack points and a value; `None` otherwise.
    pub fn qt_claim(&self, column: usize, point: &[E], value: E) -> Option<StackClaimParts> {
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
pub const SEP_BYTECODE: E = E::new(4, 0); // GEN^2

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

    fn memory(&mut self, address: Form, count: usize, values: &[Form; 2]) {
        self.counted(&[constant(SEP_MEM), address], col(count, 1), count, values);
    }

    /// `memory` when the column `flag` is 1. The push's counter is `count·(1 + flag·(g + 1))`: `g·count` at 1, and
    /// the pull's `count` at 0, so the pair cancels and touches no cell.
    fn memory_if(&mut self, flag: usize, address: Form, count: usize, values: &[Form; 2]) {
        let counter = col(count, 0) + prod(flag, count, 1) + prod(flag, count, 0);
        self.counted(&[constant(SEP_MEM), address], counter, count, values);
    }

    /// The cell's limbs are the named columns, a missing high one zero.
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
    let va = cols(ARITH_COLUMNS, ["va_0", "va_1"]);
    let vb = cols(ARITH_COLUMNS, ["vb_0", "vb_1"]);
    let mut flushes = Flushes::default();
    flushes.state_step(pc, fp);
    let operands = vec![col(o_a, 0), col(o_b, 0), col(o_c, 0), constant(ZERO), constant(ZERO)];
    flushes.bytecode(pc, cnt_bc, opcode, operands);
    flushes.memory_cols(prod(fp, o_a, 0), cnt_a, &va);
    flushes.memory_cols(prod(fp, o_b, 0), cnt_b, &vb);
    // Each product lane's terms `g^e va_j vb_k`, as (j, k, e): y^2 = x y + 1, and x = g.
    const TOWER_LANES: [&[(usize, usize, u64)]; 2] = [&[(0, 0, 0), (1, 1, 0)], &[(0, 1, 0), (1, 0, 0), (1, 1, 1)]];
    let result = if multiply {
        TOWER_LANES.map(|lane| Form::sum(lane.iter().map(|&(j, k, e)| prod(va[j], vb[k], e))))
    } else {
        std::array::from_fn(|i| col(va[i], 0) + col(vb[i], 0))
    };
    flushes.memory(prod(fp, o_c, 0), cnt_c, &result);
    flushes
}

fn flushes_set() -> Flushes {
    let [pc, fp, o, cnt, cnt_bc] = cols(SET_COLUMNS, ["pc", "fp", "o", "cnt", "cnt_bc"]);
    let k = cols(SET_COLUMNS, ["k_0", "k_1"]);
    let mut flushes = Flushes::default();
    flushes.state_step(pc, fp);
    let operands = vec![col(o, 0), col(k[0], 0), col(k[1], 0), constant(ZERO), constant(ZERO)];
    flushes.bytecode(pc, cnt_bc, OP_SET, operands);
    flushes.memory_cols(prod(fp, o, 0), cnt, &k);
    flushes
}

fn flushes_deref() -> Flushes {
    let [pc, fp, o1, o2, o3, f_pc, f_fp, ptr] =
        cols(DEREF_COLUMNS, ["pc", "fp", "o1", "o2", "o3", "f_pc", "f_fp", "ptr"]);
    let [cnt_ptr, cnt_target, cnt_local, cnt_bc] =
        cols(DEREF_COLUMNS, ["cnt_ptr", "cnt_target", "cnt_local", "cnt_bc"]);
    let v3 = cols(DEREF_COLUMNS, ["v3_0", "v3_1"]);

    let gated = |lane: usize| [col(lane, 0), prod(f_pc, lane, 0), prod(f_fp, lane, 0)];

    // v2 = (1 + f_pc + f_fp)*v3 + f_pc*(g^2*pc) + f_fp*fp, lane-wise: only the low lane takes the two K-valued sources.
    let store = [
        Form::sum(gated(v3[0]).into_iter().chain([prod(f_pc, pc, 2), prod(f_fp, fp, 0)])),
        Form::sum(gated(v3[1])),
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
pub const fn keccak_cell_lanes(odd: bool) -> [(Option<usize>, Option<usize>); KECCAK_CELLS] {
    let mut lanes = [(None, None); KECCAK_CELLS];
    let mut cell = 0;
    while cell < KECCAK_CELLS {
        lanes[cell] = match (odd, cell) {
            (_, 9..) | (true, 1..) => (Some(2 * cell - 1), Some(2 * cell)),
            (true, _) => (None, Some(0)),
            (false, 8) => (Some(16), None),
            (false, _) => (Some(2 * cell), Some(2 * cell + 1)),
        };
        cell += 1;
    }
    lanes
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
            let values = [limb(even[cell].0, odd[cell].0), limb(even[cell].1, odd[cell].1)];
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
pub const OP_KECCAK64: usize = 6;
pub const N_TABLES: usize = 7;

/// The tables the Keccak reduction proves, in the order of the table bit: half `h` of q_keccak is table `HASH_TABLES[h]`.
pub const HASH_TABLES: [usize; 2] = [OP_KECCAK, OP_KECCAK64];
/// `log2` of the number of hash tables.
pub const TABLE_BITS: usize = 1;

/// `KECCAK64`'s columns (the table is named `KECCAK64`): these, then the input lanes `in_l`, its output lanes 0 to 3 `out_l`, a
/// count `cnt_c` for each of the four cells read and the one or two written, `cnt_bc`, and the three immediates
/// `s_a`, `k_a` and `k_y`. The twenty-nine value lanes are not committed in the table: an input lane is one of
/// q_keccak's slots, and the output lanes' claims are the Keccak reduction's claim on the output.
#[rustfmt::skip]
const KECCAK64_HEAD: &[&str] = &["pc", "fp", "o_a", "o_b", "o_c", "o_d", "o_out", "out_two"];
pub const KECCAK64_OUT_LANES: usize = 4;
const KECCAK64_CELLS: usize = 6; // four read, then the one or two written
pub const KECCAK64_IN: usize = KECCAK64_HEAD.len();
pub const KECCAK64_OUT: usize = KECCAK64_IN + STATE_LANES;
const KECCAK64_CNT: usize = KECCAK64_OUT + KECCAK64_OUT_LANES;
const KECCAK64_CNT_BC: usize = KECCAK64_CNT + KECCAK64_CELLS;
const KECCAK64_S_A: usize = KECCAK64_CNT_BC + 1;
const KECCAK64_K_A: usize = KECCAK64_S_A + 1;
const KECCAK64_K_Y: usize = KECCAK64_K_A + 1;
const KECCAK64_WIDTH: usize = KECCAK64_K_Y + 1;
/// The lanes of the message: the four cells' eight.
const KECCAK64_MESSAGE_LANES: usize = 8;

/// The state a 64-byte message leaves but for its own lanes: the padding's first byte in lane 8, its last at the end of
/// the rate.
pub fn keccak64_pad() -> [u64; STATE_LANES] {
    const PAD_FIRST: u64 = 0x01;
    const END_BIT: u64 = 0x80 << 56;
    const RATE_LANES: usize = 17;
    let mut pad = [0u64; STATE_LANES];
    pad[KECCAK64_MESSAGE_LANES] = PAD_FIRST;
    pad[RATE_LANES - 1] = END_BIT;
    pad
}

/// What a padding step of each hash table outputs: the permutation of its input, zeros for KECCAK and a message of
/// zeros for KECCAK64.
pub fn hash_padding_outputs() -> [[u64; STATE_LANES]; 2] {
    let mut outputs = [[0u64; STATE_LANES], keccak64_pad()];
    outputs.iter_mut().for_each(keccak_f);
    outputs
}

fn flushes_keccak64() -> Flushes {
    let [pc, fp, o_a, o_b, o_c, o_d, o_out, out_two] = std::array::from_fn(|i| i);
    let mut flushes = Flushes::default();
    flushes.state_step(pc, fp);
    let operands = vec![
        col(o_a, 0),
        col(o_b, 0),
        col(o_c, 0),
        col(o_d, 0),
        col(o_out, 0),
        col(out_two, 0),
        col(KECCAK64_S_A, 0),
        col(KECCAK64_K_A, 0),
        col(KECCAK64_K_Y, 0),
    ];
    flushes.bytecode(pc, KECCAK64_CNT_BC, OP_KECCAK64, operands);
    // The four cells read, two lanes each in order, then output cell 0, then output cell 1 when `out_two` is 1.
    // The second cell holds `s_a * in_2 + k_a` in its low limb, and the fourth `in_6 + k_y`.
    for (cell, o) in [o_a, o_b, o_c, o_d].into_iter().enumerate() {
        let lo = col(KECCAK64_IN + 2 * cell, 0);
        let lo = match cell {
            1 => prod(KECCAK64_IN + 2, KECCAK64_S_A, 0) + col(KECCAK64_K_A, 0),
            3 => lo + col(KECCAK64_K_Y, 0),
            _ => lo,
        };
        let values = [lo, col(KECCAK64_IN + 2 * cell + 1, 0)];
        flushes.memory(prod(fp, o, 0), KECCAK64_CNT + cell, &values);
    }
    flushes.memory_cols(prod(fp, o_out, 0), KECCAK64_CNT + 4, &[KECCAK64_OUT, KECCAK64_OUT + 1]);
    let values = [col(KECCAK64_OUT + 2, 0), col(KECCAK64_OUT + 3, 0)];
    flushes.memory_if(out_two, prod(fp, o_out, 1), KECCAK64_CNT + 5, &values);
    flushes
}

/// `in_l = PAD_l` for the lanes 8 to 24 the message leaves to the padding.
fn keccak64_constraints(columns: &[E]) -> Vec<E> {
    let pad = keccak64_pad();
    (KECCAK64_MESSAGE_LANES..STATE_LANES)
        .map(|lane| columns[KECCAK64_IN + lane] + E::new(pad[lane], 0))
        .collect()
}

#[rustfmt::skip]
const ARITH_COLUMNS: &[&str] = &["pc", "fp", "o_a", "o_b", "o_c", "va_0", "va_1", "vb_0", "vb_1", "cnt_a", "cnt_b", "cnt_c", "cnt_bc"];
const SET_COLUMNS: &[&str] = &["pc", "fp", "o", "k_0", "k_1", "cnt", "cnt_bc"];
#[rustfmt::skip]
const DEREF_COLUMNS: &[&str] = &["pc", "fp", "o1", "o2", "o3", "f_pc", "f_fp", "ptr", "v3_0", "v3_1", "cnt_ptr", "cnt_target", "cnt_local", "cnt_bc"];
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
pub const KECCAK_CNT: usize = KECCAK_OUT + STATE_LANES;
const KECCAK_CNT_BC: usize = KECCAK_CNT + 2 * KECCAK_CELLS;

/// The local column of hash table `half` riding spare slot `k` (`cpu::keccak_value` in the prover): the table's
/// committed columns in local order, the ones before the value lanes then the read counts after them.
pub const fn spare_local(half: usize, k: usize) -> usize {
    let (head, counts) = if half == 0 {
        (KECCAK_IN, KECCAK_CNT)
    } else {
        (KECCAK64_IN, KECCAK64_CNT)
    };
    if k < head { k } else { counts + (k - head) }
}

/// Slots of the run the committed states occupy, and the ones they leave over.
pub const QKECCAK_SLOTS: usize = STATE_LANES * (crate::hash::ROUNDS / KECCAK_DEPTH);
pub const QKECCAK_SPARE: usize = (1 << log2_ceil(QKECCAK_SLOTS as u64)) - QKECCAK_SLOTS;
// The same guard the prover carries (`cpu::layout`): every spare slot needs a committed column
// of the table to put in it, distinct from the others and not one of the value lanes. It holds
// at this depth, not at every divisor of 24, and the two sides must refuse together.
const _: () = {
    let mut half = 0;
    while half < 2 {
        let (head, counts, end) = if half == 0 {
            (KECCAK_IN, KECCAK_CNT, TABLE_WIDTHS[OP_KECCAK])
        } else {
            (KECCAK64_IN, KECCAK64_CNT, TABLE_WIDTHS[OP_KECCAK64])
        };
        let mut k = 0;
        while k < QKECCAK_SPARE {
            let local = spare_local(half, k);
            assert!(k == 0 || spare_local(half, k - 1) < local);
            assert!(local < head || (local >= counts && local < end));
            k += 1;
        }
        half += 1;
    }
};

/// The half of q_keccak (the table bit) and slot of it the global column `column` IS, if it is one: an input lane
/// (slot = lane), or one of a hash table's own columns riding a slot past the committed states. The reduction
/// weighs nothing on those spare slots, so the stack carries those columns there rather than a block of zeros and a
/// block of its own.
pub fn qkeccak_slot(column: usize) -> Option<(usize, usize)> {
    let half = HASH_TABLES
        .iter()
        .position(|&t| column >= GLOBAL_COLUMN_BASES[t] && column < GLOBAL_COLUMN_BASES[t] + TABLE_WIDTHS[t])?;
    let local = column - GLOBAL_COLUMN_BASES[HASH_TABLES[half]];
    if let Some(k) = (0..QKECCAK_SPARE).find(|&k| spare_local(half, k) == local) {
        return Some((half, QKECCAK_SLOTS + k));
    }
    let in_lanes = if half == 0 { KECCAK_IN } else { KECCAK64_IN };
    (in_lanes..in_lanes + STATE_LANES)
        .contains(&local)
        .then(|| (half, local - in_lanes))
}

/// Where slot `slot` of table half `half` starts in q_keccak, laid out `(step, table, slot)` over `2^log_steps` steps a
/// table.
pub fn qkeccak_slot_start(log_steps: usize, half: usize, slot: usize) -> u64 {
    (((slot << TABLE_BITS) + half) as u64) << log_steps
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

fn keccak64_columns() -> Vec<String> {
    let lanes = |prefix: &'static str, n: usize| (0..n).map(move |lane| format!("{prefix}_{lane}"));
    let counts = (0..KECCAK64_CELLS).map(|cell| format!("cnt_{cell}"));
    let head = KECCAK64_HEAD.iter().map(|name| name.to_string());
    head.chain(lanes("in", STATE_LANES))
        .chain(lanes("out", KECCAK64_OUT_LANES))
        .chain(counts)
        .chain(["cnt_bc", "s_a", "k_a", "k_y"].map(|name| name.to_string()))
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
        *padding = E::new(lane, 0);
    }
    // A padding row of KECCAK64 holds the padding's lanes and permutes the state they make, with a message of zeros.
    let mut keccak64 = table(
        OP_KECCAK64,
        keccak64_columns(),
        flushes_keccak64(),
        keccak64_constraints,
    );
    let pad = keccak64_pad();
    let mut output = pad;
    keccak_f(&mut output);
    for (padding, lane) in keccak64.padding[KECCAK64_IN..KECCAK64_IN + STATE_LANES]
        .iter_mut()
        .zip(pad)
    {
        *padding = E::new(lane, 0);
    }
    for (padding, lane) in keccak64.padding[KECCAK64_OUT..KECCAK64_OUT + KECCAK64_OUT_LANES]
        .iter_mut()
        .zip(output)
    {
        *padding = E::new(lane, 0);
    }
    vec![
        table(OP_XOR, named(ARITH_COLUMNS), flushes_arith(OP_XOR, false), |_| vec![]),
        table(OP_MUL, named(ARITH_COLUMNS), flushes_arith(OP_MUL, true), |_| vec![]),
        table(OP_SET, named(SET_COLUMNS), flushes_set(), |_| vec![]),
        table(OP_DEREF, named(DEREF_COLUMNS), flushes_deref(), |_| vec![]),
        table(OP_JUMP, named(JUMP_COLUMNS), flushes_jump(), jump_constraints),
        keccak,
        keccak64,
    ]
}

pub const TABLE_WIDTHS: [usize; N_TABLES] = [
    ARITH_COLUMNS.len(),
    ARITH_COLUMNS.len(),
    SET_COLUMNS.len(),
    DEREF_COLUMNS.len(),
    JUMP_COLUMNS.len(),
    KECCAK_CNT_BC + 1,
    KECCAK64_WIDTH,
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
pub const TABLE_LOG_FLOORS: [usize; N_TABLES] = [0, 0, 0, 0, 0, KECCAK_MIN_STEPS_LOG, KECCAK_MIN_STEPS_LOG];

/// The reduction's shape: each hash table is completed with its padding step to the tallest's height.
fn keccak_shape(table_log_heights: &[usize]) -> KeccakShape {
    KeccakShape {
        log_steps: HASH_TABLES
            .iter()
            .map(|&t| table_log_heights[t])
            .max()
            .expect("hash tables"),
        depth: KECCAK_DEPTH,
        table_bits: TABLE_BITS,
    }
}

/// Every column's log size, in global order: the framework's, q_keccak's, then each table's block.
fn column_kappas(tables: &[Table], log_memory: usize, log_bytecode: usize, table_log_heights: &[usize]) -> Vec<usize> {
    let keccak = keccak_shape(table_log_heights);
    let mut kappas = vec![
        log_memory,
        log_memory,
        log_memory,
        log_bytecode,
        keccak.log_steps + keccak.table_bits + keccak.slot_bits(),
    ];
    for table in tables {
        kappas.extend(vec![table_log_heights[table.opcode]; table.width()]);
    }
    kappas
}

/// The columns with a block of their own in the stack, out of `columns`. A column q_keccak carries gets none: it is
/// one of its slots, 2^height words, whether an input lane or a KECCAK column riding a spare slot (`qkeccak_slot`). An
/// output lane is not committed at all, nothing ever opening it.
fn stacked_blocks(columns: usize) -> Vec<usize> {
    let (keccak, keccak64) = (GLOBAL_COLUMN_BASES[OP_KECCAK], GLOBAL_COLUMN_BASES[OP_KECCAK64]);
    let outputs = [
        keccak + KECCAK_OUT..keccak + KECCAK_OUT + STATE_LANES,
        keccak64 + KECCAK64_OUT..keccak64 + KECCAK64_OUT + KECCAK64_OUT_LANES,
    ];
    (0..columns)
        .filter(|column| qkeccak_slot(*column).is_none() && !outputs.iter().any(|range| range.contains(column)))
        .collect()
}

/// The stack of columns of these log sizes: each stacked column's offset, and its log size, floored at the PCS minimum.
fn stack(kappas: &[usize]) -> (BTreeMap<usize, u64>, usize) {
    let blocks = stacked_blocks(kappas.len());
    let block_kappas: Vec<usize> = blocks.iter().map(|&column| kappas[column]).collect();
    let (block_offsets, total_log) = stack_offsets(&block_kappas);
    (
        blocks.into_iter().zip(block_offsets).collect(),
        MIN_STACKED_LOG.max(total_log),
    )
}

/// The stack of the smallest execution of a bytecode of `2^log_bytecode` instructions: the least memory, every table at
/// its floor. A recursion guest, whose children are proofs of its own bytecode, need open no smaller one.
pub fn min_execution_stacked_log(log_bytecode: usize) -> usize {
    stack(&column_kappas(
        &tables(),
        MIN_LOG_MEMORY,
        log_bytecode,
        &TABLE_LOG_FLOORS,
    ))
    .1
}

pub fn build_layout<'a>(
    bytecode: &'a Bytecode,
    log_memory: u64,
    table_rows: &[u64],
    truncation: Option<Truncation>,
) -> Result<Layout<'a>, VerifierError> {
    let log_bytecode = log2_strict(bytecode.len() as u64)?;
    let mem_count = truncation.map(|t| t.mem_count);
    require(
        (MIN_LOG_MEMORY as u64..=MAX_LOG_MEMORY as u64).contains(&log_memory)
            && table_rows.iter().all(|&rows| rows <= 1 << MAX_TABLE_LOG)
            && (BUS_BITS..=MAX_LOG_BYTECODE + BUS_BITS).contains(&log_bytecode)
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

    let keccak = keccak_shape(&table_log_heights);
    let kappas = column_kappas(&tables, log_memory, log_bytecode, &table_log_heights);
    let blocks = stacked_blocks(kappas.len());
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
    // A truncated proof may announce q_keccak as the pieces of its halves instead (QT): they are stacked like any other
    // item, in place of the column.
    let mut qt_pieces: Vec<(usize, u64, usize)> = Vec::new(); // (half, first row, log rows)
    if truncation.is_some_and(|t| t.qt) {
        items.retain(|item| item.0 != QKECCAK);
        for (half, &table) in HASH_TABLES.iter().enumerate() {
            for (off, b) in pieces(keccak.log_steps, table_rows[table]) {
                qt_pieces.push((half, off, b));
            }
        }
    }
    let item_count = items.len();
    let all_kappas: Vec<usize> = items
        .iter()
        .map(|&(.., kappa)| kappa)
        .chain(qt_pieces.iter().map(|&(_, _, b)| b + 6))
        .collect();
    let (mut item_offsets, total_log) = stack_offsets(&all_kappas);
    let stack_log = MIN_STACKED_LOG.max(total_log); // Floor at the PCS minimum
    let qt_offsets = item_offsets.split_off(item_count);
    let qt = truncation.is_some_and(|t| t.qt).then(|| Qt {
        log_steps: keccak.log_steps,
        pieces: qt_pieces
            .iter()
            .zip(qt_offsets)
            .map(|(&(half, off, b), index)| QtPiece {
                half,
                off,
                placement: Placement {
                    variables: b + 6,
                    index,
                },
            })
            .collect(),
        pads: [qt_pad(0), qt_pad(1)],
    });
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
    if qt.is_some() {
        // The column has no placement of its own: its pieces carry it.
        offsets.insert(QKECCAK, 0);
    }
    let placement = |(column, &kappa): (usize, &usize)| {
        let index = match qkeccak_slot(column) {
            Some((half, slot)) => offsets[&QKECCAK] + qkeccak_slot_start(keccak.log_steps, half, slot),
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
        let (zero, keccak64) = (qt_pad(0), qt_pad(1));
        assert!(zero[..25].iter().chain(&zero[50..]).all(|&word| word == 0));
        assert_ne!(zero[25], 0);
        assert_eq!(keccak64[..25], keccak64_pad());
        assert!(keccak64[50..].iter().all(|&word| word == 0));
        for (half, mut state) in [[0u64; STATE_LANES], keccak64_pad()].into_iter().enumerate() {
            keccak_rounds(&mut state, crate::hash::ROUNDS);
            let mut full = if half == 0 { [0u64; STATE_LANES] } else { keccak64_pad() };
            keccak_f(&mut full);
            assert_eq!(state, full);
        }
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
        // Two tables of 2^12 steps, the first with 2500 covered rows and the second with 1500: three pieces and one.
        let (kappa, rows) = (12usize, [2500u64, 1500]);
        let mut pads = [[0u64; 64]; 2];
        pads.iter_mut().flatten().for_each(|p| *p = next());
        let mut stack_pieces = Vec::new();
        for (half, &real) in rows.iter().enumerate() {
            for (off, b) in pieces(kappa, real) {
                stack_pieces.push((half, off, b));
            }
        }
        assert_eq!(stack_pieces.len(), 3);
        let sizes: Vec<usize> = stack_pieces.iter().map(|&(_, _, b)| b + 6).collect();
        let (offsets, total_log) = stack_offsets(&sizes);
        let qt = Qt {
            log_steps: kappa,
            pieces: stack_pieces
                .iter()
                .zip(&offsets)
                .map(|(&(half, off, b), &index)| QtPiece {
                    half,
                    off,
                    placement: Placement {
                        variables: b + 6,
                        index,
                    },
                })
                .collect(),
            pads,
        };
        // The full layout, index `row + 2^kappa (table + 2 slot)`, and the stack of pieces.
        let covered = |half: usize| qt.half_pieces(half).map(|p| p.off + (1 << p.b())).max().unwrap() as usize;
        let mut full = vec![0u64; 128 << kappa];
        let mut stack = vec![0u64; 1 << total_log];
        for half in 0..2 {
            for slot in 0..64 {
                for row in 0..1usize << kappa {
                    let word = if row < covered(half) { next() } else { pads[half][slot] };
                    full[row + (((slot << 1) + half) << kappa)] = word;
                    if let Some(p) = qt
                        .half_pieces(half)
                        .find(|p| (p.off as usize..p.off as usize + (1 << p.b())).contains(&row))
                    {
                        stack[p.placement.index as usize + (row - p.off as usize) + (slot << p.b())] = word;
                    }
                }
            }
        }
        let mut random = || E::new(next(), next());
        let at = |words: &[u64], point: &[E]| {
            let words: Vec<K> = words.iter().map(|&w| K(w)).collect();
            multilinear_eval(&words, point).unwrap()
        };
        let on_stack = |parts: &[(E, Vec<E>)]| E::sum(parts.iter().map(|(scale, point)| *scale * at(&stack, point)));
        // The reduction's output claim, at `(row, table, slot)`.
        let rho: Vec<E> = (0..kappa + 1 + 6).map(|_| random()).collect();
        let (parts, value) = qt.claim(QKECCAK, &rho, at(&full, &rho), total_log).unwrap();
        assert_eq!(on_stack(&parts), value);
        // A slot column of each table, at the rows of its own height.
        for (half, tau, slot) in [(0, 11, 5), (1, 9, 20), (1, 12, 3)] {
            let zeta: Vec<E> = (0..tau).map(|_| random()).collect();
            let start = ((slot << 1) + half) << kappa;
            let column = at(&full[start..][..1 << tau], &zeta);
            let global =
                GLOBAL_COLUMN_BASES[HASH_TABLES[half]] + if half == 0 { KECCAK_IN } else { KECCAK64_IN } + slot;
            assert_eq!(qkeccak_slot(global), Some((half, slot)));
            let (parts, value) = qt.claim(global, &zeta, column, total_log).unwrap();
            assert_eq!(on_stack(&parts), value, "half {half} slot {slot}");
        }
        // The ring switch: bit `i` of the full layout's words at `rho` is the pieces' weighted slices, and the pad's share.
        let share = qt.pad_share(&rho);
        let split = qt.ring_split(&rho);
        for i in 0..64 {
            let mut sum = share[i];
            for (piece, (point, coefficient)) in qt.pieces.iter().zip(&split) {
                let words = &stack[piece.placement.index as usize..][..1 << piece.placement.variables];
                sum += *coefficient * at_bit(words, point, i);
            }
            assert_eq!(sum, at_bit(&full, &rho, i), "bit {i}");
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
        let mut random = || E::new(next(), next());
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
