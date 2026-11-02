//! The public column schema and bus layout: the committed-column indices, and
//! the flush/count blocks the verifier reconstructs from the program + announced
//! sizes and public input. Plus the prover-side witness build.

use super::*;
use crate::witness::Column;
use flux_profiler::timed;

// ---- column schema -----------------------------------------------------------

// Shared committed columns (indices `0..N_SHARED`). The program (opcode +
// operands) is PUBLIC, not committed: it rides the bytecode seed/finalize blocks
// as `Coord::Public`; only the witness-dependent finalize counts are committed.
// The data-memory image, a 192-bit word per cell committed as three K-lane columns.
pub const MEM_LO: usize = 0;
pub const MEM_HI: usize = 1;
pub const MEM_TOP: usize = 2;
pub const MFCNT: usize = 3; // per-cell memory access count, g^{A[i]}
pub const BFCNT: usize = 4; // per-pc bytecode execution count, g^{A[pc]}
// The Keccak reduction's committed states `q_keccak`, in the SAME stack as every
// other column (single PCS). Size `2^(SLOT_LOG + tau)` F64 words for `2^tau`
// KECCAK steps, never fewer than `MIN_STEPS`. It is the
// SOLE copy of the input words: the VM's KECCAK value columns are virtual, the
// input lanes' memory-bus claims routing to `q_keccak` slots (§keccak) and the
// output lanes' becoming the reduction's claim on the output, which is not
// committed. The reduction's two claims are discharged by the single stacked
// WHIR opening over this commitment.
pub const QKECCAK: usize = 5;
pub const N_SHARED: usize = 6;

/// Global column indexing: the shared columns occupy `0..N_SHARED`, then each
/// table `t` (in [`tables::tables`] order) owns the contiguous block `[base[t],
/// base[t] + n_committed_columns_t)`. Both prover and verifier derive this identically
/// from the table set, so every column claim lines up.
pub struct Schema {
    pub base: [usize; tables::N_TABLES],
    pub n: usize,
}

/// The schema is a pure function of the fixed table set, so compute it once.
pub fn schema() -> &'static Schema {
    static SCHEMA: std::sync::OnceLock<Schema> = std::sync::OnceLock::new();
    SCHEMA.get_or_init(|| {
        let mut base = [0usize; tables::N_TABLES];
        let mut next = N_SHARED;
        for (t, table) in tables::tables().iter().enumerate() {
            base[t] = next;
            next += table.n_committed_columns();
        }
        Schema { base, n: next }
    })
}

/// A KECCAK column that `q_keccak` carries. Both kinds are VIRTUAL, with no placement of
/// their own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeccakLane {
    /// A column whose words ARE `q_keccak`'s slot `s`, so a claim on it is a claim on that
    /// slot: an input lane (slot = lane), or one of the columns riding the spare slots.
    Slot(usize),
    /// Output lane `l`, whose claim is the Keccak reduction's claim on the permutation's output.
    Output(usize),
}

/// Slots of `q_keccak` no committed state occupies: the lanes of the states a step commits,
/// `crate::keccak::SLOTS`, rounded up to the power of two the slot index needs. The reduction
/// weighs nothing there, so they carry the KECCAK table's own first columns instead of zeros,
/// which is `QKECCAK_SPARE` columns the stack does not have to hold twice.
pub const QKECCAK_SPARE: usize = (1 << crate::keccak::SLOT_LOG) - crate::keccak::SLOTS;

/// The local KECCAK column riding spare slot `k`: its committed columns in local order, which
/// are the ones before the value lanes, then the read counts after them.
const fn spare_local(k: usize) -> usize {
    if k < tables::keccakt::V_IN {
        k
    } else {
        tables::keccakt::R0 + (k - tables::keccakt::V_IN)
    }
}
const _: () = {
    let mut k = 0;
    while k < QKECCAK_SPARE {
        let local = spare_local(k);
        // Increasing, so distinct: two spare slots sharing a column would leave the other's
        // words unwritten. And never a value lane, which q_keccak already holds.
        assert!(
            k == 0 || spare_local(k - 1) < local,
            "the spare columns must be distinct"
        );
        assert!(
            local < tables::keccakt::V_IN || (local >= tables::keccakt::R0 && local < tables::keccakt::N),
            "the KECCAK table has fewer committed columns than q_keccak has spare slots"
        );
        k += 1;
    }
};

/// What `q_keccak` holds for the global column `col`, if it holds anything: a value lane of
/// the KECCAK table, or one of its columns riding a spare slot.
pub fn keccak_value(col: usize) -> Option<KeccakLane> {
    use primitives::hash::STATE_LANES;
    let local = col.checked_sub(schema().base[tables::KECCAK_TABLE])?;
    if local >= tables::keccakt::N {
        return None;
    }
    if let Some(k) = (0..QKECCAK_SPARE).find(|&k| spare_local(k) == local) {
        return Some(KeccakLane::Slot(crate::keccak::SLOTS + k));
    }
    let lane = local.checked_sub(tables::keccakt::V_IN)?;
    match lane {
        0..STATE_LANES => Some(KeccakLane::Slot(lane)),
        _ if lane < 2 * STATE_LANES => Some(KeccakLane::Output(lane - STATE_LANES)),
        _ => None,
    }
}

/// Offset a table's local flush coordinates to global column indices.
fn offset_coords(base: usize, coords: Vec<Coord>) -> Vec<Coord> {
    coords.into_iter().map(|c| offset_coord(base, c)).collect()
}

fn offset_coord(base: usize, c: Coord) -> Coord {
    match c {
        Coord::Col(i) => Coord::Col(base + i),
        Coord::GCol(i, k) => Coord::GCol(base + i, k),
        Coord::Prod(i, j, k) => Coord::Prod(base + i, base + j, k),
        Coord::Sum(cs) => Coord::Sum(offset_coords(base, cs)),
        other => other,
    }
}

/// The public proof structure: everything the verifier reconstructs from the
/// program, the announced sizes, and the public input, with no witness values. The
/// flush blocks reference columns by INDEX (see [`crate::leaf::Coord`]), so they
/// are pure public structure.
pub struct Layout {
    pub push: Vec<Block>,
    pub pull: Vec<Block>,
    /// Count channel: read-count columns whose product must be nonzero (§sec:memchan).
    pub count: Vec<Block>,
    /// Per-column placement (offset + n_vars) in the padded stack: each column's padded height `2^n_vars`. The
    /// prover holds no padded stack, so the offsets are unused.
    pub placements: Vec<witness::Placement>,
    /// The padded stack's shape.
    pub shape: witness::StackShape,
    /// The committed stack: the same columns, the jagged ones cut to their tables' chunks. From the announced
    /// sizes alone, so reconstructable by the verifier.
    pub dense: witness::Dense,
    /// The memory rows the committed stack holds, a multiple of a chunk: the cells past them are unwritten.
    pub mem_rows: usize,
    /// Public input: the first two memory cells `m[0], m[1]` (each a 192-bit
    /// word), bound to the committed memory at verification (§sec:e2e-pi).
    pub pi: [F192; 2],
    pub taus: [usize; tables::N_TABLES],
    /// Rows per table the program executed, at most `2^taus[t]`: the rest are padding
    /// rows (§sec:e2e-pad).
    pub rows: [usize; tables::N_TABLES],
}

/// The prover's witness: the committed stack, which holds every committed column at its
/// [`Layout::dense`] slot, plus the public [`Layout`] (and the sizes needed to announce it).
pub(crate) struct Witness {
    pub(crate) dense: Vec<F64>,
    /// Which blocks of `dense` equal their predecessor, as far as the fills know ([`::pcs::stack_open::equal_blocks`]
    /// is the exact map).
    pub(crate) repeats: Vec<bool>,
    /// The KECCAK output lanes' values as `(global column index, values)`. They carry
    /// data for the bus but are not committed, so they are not in `dense`.
    pub(crate) virt: Vec<(usize, Vec<F64>)>,
    pub(crate) layout: Layout,
    pub(crate) log_mem: usize,
    /// Freed immediately after reduction, before the mixed PCS opening.
    pub(crate) keccak_reduction: crate::keccak::PreparedReductionWitness,
}

impl Witness {
    /// One read-only view per column, in global column order: a committed column's runs in the
    /// stack, a KECCAK input lane's slot of `q_keccak`, an output lane's private buffer.
    ///
    /// A jagged column reads its padding past its chunks: zero, or `g^0 = 1` for the memory's final counts.
    pub(crate) fn columns(&self) -> Vec<Column<'_>> {
        let l = &self.layout;
        let q_keccak = &self.dense[dense_qkeccak_offset(l)..];
        let steps = 1 << l.taus[tables::KECCAK_TABLE];
        let mut cols: Vec<Column> = l
            .placements
            .iter()
            .enumerate()
            .map(|(i, p)| match keccak_value(i) {
                Some(KeccakLane::Slot(slot)) => Column::flat(&q_keccak[slot * steps..][..steps]),
                _ if p.is_virtual() => Column::flat(&[]),
                _ => {
                    let pad = if i == MFCNT { F64::ONE } else { F64::ZERO };
                    l.dense.column(&self.dense, i, p.n_vars, pad)
                }
            })
            .collect();
        for (i, buf) in &self.virt {
            cols[*i] = Column::flat(buf);
        }
        cols
    }

    /// Committed data before the zero-pad to `2^m`: the words of the committed stack the columns fill.
    pub(crate) fn committed_size(&self) -> usize {
        self.layout.dense.placed
    }
}

/// The committed columns' kappa SOURCES, for the recursion guest's
/// in-circuit certification of the stacked size m = max(log2_ceil(sum of
/// 2^kappa), MIN_MU). Per committed column: `Some((source, adj))` with
/// kappa = value(source) + adj, where source 0 is the constant 0 (kappa =
/// adj; used for the fixed-size columns and the program bytecode length,
/// which the caller passes as `log_bytecode`), source 1 is log_mem, and
/// source 2 + t is tau_t. `None` = virtual (never committed). `col_kappas`
/// is derived from this, so the two cannot drift apart.
pub fn col_kappa_sources(log_bytecode: usize) -> Vec<Option<(usize, usize)>> {
    let sch = schema();
    let mut k = vec![Some((0usize, 0usize)); sch.n];
    k[MEM_LO] = Some((1, 0));
    k[MEM_HI] = Some((1, 0));
    k[MEM_TOP] = Some((1, 0));
    k[MFCNT] = Some((1, 0));
    k[BFCNT] = Some((0, log_bytecode));
    // q_keccak is `2^(SLOT_LOG + n_steps_log)` F64 words, and tau_5 IS
    // n_steps_log (the announced-size certification uses the same floor), so this
    // reproduces `qkeccak_kappa`.
    k[QKECCAK] = Some((2 + tables::KECCAK_TABLE, crate::keccak::SLOT_LOG));
    for (t, table) in tables::tables().iter().enumerate() {
        let base = sch.base[t];
        k[base..base + table.n_committed_columns()].fill(Some((2 + t, 0)));
    }
    // The columns `q_keccak` carries are ALWAYS virtual: it already holds the input words at
    // fixed slots and the KECCAK columns riding its spare ones, and the output words are the
    // reduction's to prove. A slot column's bus claims route directly to the matching
    // `q_keccak` slot evaluation (`slot_claims`), the output lanes' to the reduction's output
    // claim, which binds both to the proven witness with no separate value-binding
    // sub-protocol.
    for (col, k) in k.iter_mut().enumerate() {
        if keccak_value(col).is_some() {
            *k = None;
        }
    }
    k
}

/// The bus flush blocks' kappa SOURCES, flattened in side order (push, pull,
/// count) exactly as the blocks are constructed below: per block
/// `(source, adj)` with kappa = value(source) + adj, source 0 = the constant
/// 0, 1 = log_mem, 2 + t = tau_t. For the recursion guest's in-circuit pin
/// of every hinted block kappa. Keep in lockstep with the block
/// construction in [`fn@layout`].
pub fn block_kappa_sources(log_bytecode: usize) -> Vec<(usize, usize)> {
    let mut push = vec![(0, 0), (1, 0), (0, log_bytecode)];
    let mut pull = vec![(0, 0), (1, 0), (0, log_bytecode)];
    let mut count = Vec::new();
    for (t, table) in tables::tables().iter().enumerate() {
        let mut fb = tables::FlushBuilder::new();
        table.flushes(&mut fb);
        push.extend(std::iter::repeat_n((2 + t, 0), fb.push.len()));
        pull.extend(std::iter::repeat_n((2 + t, 0), fb.pull.len()));
        count.extend(std::iter::repeat_n((2 + t, 0), table.count_columns().len()));
    }
    push.extend(pull);
    push.extend(count);
    push
}

/// Column → log-size (`kappa`) map, derived from [`col_kappa_sources`] by
/// substituting the announced sizes: source 0 is the constant 0, source 1 is
/// `log_mem`, source `2 + t` is `tau_t`. `None` marks a **virtual**
/// (uncommitted) column. Depends only on the public sizes, so the verifier can
/// reconstruct the placements.
fn col_kappas(log_mem: usize, log_bytecode: usize, taus: [usize; tables::N_TABLES]) -> Vec<Option<usize>> {
    let mut values = vec![0usize, log_mem];
    values.extend(taus);
    col_kappa_sources(log_bytecode)
        .iter()
        .map(|s| s.map(|(source, adj)| values[source] + adj))
        .collect()
}

/// The jagged groups' global columns: each table's committed columns but the ones `q_keccak` carries, in table
/// order, then the memory's four.
pub fn jagged_columns() -> Vec<Vec<usize>> {
    let sch = schema();
    let mut groups: Vec<Vec<usize>> = tables::tables()
        .iter()
        .enumerate()
        .map(|(t, table)| {
            (sch.base[t]..sch.base[t] + table.n_committed_columns())
                .filter(|&col| keccak_value(col).is_none())
                .collect()
        })
        .collect();
    groups.push(vec![MEM_LO, MEM_HI, MEM_TOP, MFCNT]);
    groups
}

/// The columns the committed stack keeps as whole padded blocks: their padding is not zero.
pub const PADDED_COLUMNS: [usize; 2] = [BFCNT, QKECCAK];

/// The committed stack the announced sizes imply ([`witness::Dense`]).
pub fn dense_layout(
    log_mem: usize,
    log_bytecode: usize,
    rows: [usize; tables::N_TABLES],
    mem_rows: usize,
) -> witness::Dense {
    let kappas = col_kappas(log_mem, log_bytecode, table_logs(rows));
    let groups = jagged_columns()
        .into_iter()
        .zip(rows.into_iter().chain([mem_rows]))
        .collect();
    witness::dense_of(&kappas, &PADDED_COLUMNS, groups)
}

/// `log2` of the committed stack the announced sizes imply. The one part of
/// [`layout`] a size floor needs, and far cheaper than the rest of it.
pub(crate) fn committed_log(
    log_mem: usize,
    mem_rows: usize,
    log_bytecode: usize,
    rows: [usize; tables::N_TABLES],
) -> usize {
    dense_layout(log_mem, log_bytecode, rows, mem_rows).shape.mu
}

/// Operand slots of a bytecode tuple, after the opcode.
pub const BYTECODE_OPERANDS: usize = 12;

/// The public bytecode columns, shared by the seed and finalize blocks.
pub type BytecodeColumns = [std::sync::Arc<Vec<F64>>; 1 + BYTECODE_OPERANDS];

/// The PUBLIC bytecode columns over the program cube, in bytecode-slot order: the opcode, then
/// the operand slots. The program is not committed, so these ride the seed/finalize blocks as
/// `Coord::Public` and stack into the polynomial [`bytecode_table`] returns.
pub fn bytecode_columns(prog: &[Op]) -> BytecodeColumns {
    let g_at = primitives::field::g_pow32_by_lookup;

    // The opcode, then the operand slots: offsets as g-powers, a `SET` immediate as its three
    // limbs, the `DEREF` store-mode flags and `KECCAK`'s layout and output flags as 0 or 1. Unused slots
    // are zero.
    let flag = |b: bool| F64(u64::from(b));
    let row = |op: &Op| -> [F64; 1 + BYTECODE_OPERANDS] {
        let mut row = [F64::ZERO; 1 + BYTECODE_OPERANDS];
        let (opcode, slots) = row.split_first_mut().expect("an opcode");
        let mut put = |values: &[F64]| slots[..values.len()].copy_from_slice(values);
        *opcode = match *op {
            Op::Xor { a, b, c } | Op::Mul { a, b, c } => {
                put(&[g_at(a), g_at(b), g_at(c)]);
                if matches!(op, Op::Xor { .. }) { OP_XOR } else { OP_MUL }
            }
            Op::Set { o, k } => {
                put(&[g_at(o), F64(k.c0), F64(k.c1), F64(k.c2)]);
                OP_SET
            }
            Op::Deref { o1, o2, o3, mode } => {
                put(&[g_at(o1), g_at(o2), g_at(o3), mode.f_pc(), mode.f_fp()]);
                OP_DEREF
            }
            Op::Jump { oc, od, of } => {
                put(&[g_at(oc), g_at(od), g_at(of)]);
                OP_JUMP
            }
            Op::Keccak {
                m,
                tail,
                b8,
                cap,
                out,
                in_odd,
                out_odd,
                out_cells,
            } => {
                // The gated output cells' flushes read `out_odd·gate` as `out_odd` ([`crate::tables`]).
                assert!(!out_odd || out_cells == OutCells::All, "an odd output layout writes every cell");
                let [m0, m1, m2, m3] = m.map(g_at);
                let (out_two, out_all) = out_cells.flags();
                put(&[
                    m0,
                    m1,
                    m2,
                    m3,
                    g_at(tail),
                    g_at(b8),
                    g_at(cap),
                    g_at(out),
                    flag(in_odd),
                    flag(out_odd),
                    flag(out_two),
                    flag(out_all),
                ]);
                OP_KECCAK
            }
            Op::Dot2 { a, ptr, w, acc, c } => {
                put(&[g_at(a), g_at(ptr), g_at(w), g_at(acc), g_at(c)]);
                OP_DOT2
            }
        };
        row
    };
    // One pass writes all thirteen columns.
    let chunk = parallel::recommended_chunk_size(prog.len());
    let columns = primitives::filled_vecs([prog.len(); 1 + BYTECODE_OPERANDS], |columns| {
        parallel::for_each_chunks_mut(columns, [chunk; 1 + BYTECODE_OPERANDS], |t, mut columns| {
            for (i, op) in prog[t * chunk..].iter().take(chunk).enumerate() {
                for (column, value) in columns.iter_mut().zip(row(op)) {
                    column[i] = value;
                }
            }
        });
    });
    columns.map(std::sync::Arc::new)
}

/// The stacked bytecode polynomial: the thirteen columns at their bus tuple
/// coordinates, which is what makes the program's whole share of a bus leaf one
/// evaluation at `(ζ, α⃗)` (see [`crate::leaf::stacked_bytecode_table`]).
///
/// This is the multilinear an outermost verifier is handed in place of a
/// structured program, and what the transcript seed binds ([`super::fs_seed`]).
#[timed]
pub fn bytecode_table(prog: &[Op]) -> Vec<F64> {
    stacked_columns(&bytecode_columns(prog))
}

/// [`bytecode_table`] over columns [`bytecode_columns`] already built.
pub(crate) fn stacked_columns(cols: &BytecodeColumns) -> Vec<F64> {
    let coords = cols.each_ref().map(|c| Coord::Public(c.clone())).into();
    let block = Block::full(crate::log2_strict_usize(cols[0].len()), coords);
    crate::leaf::stacked_bytecode_table(std::slice::from_ref(&block))
}

/// Build the public [`Layout`] from the program, the memory log-size `log_mem`, the
/// instruction tables' row counts `rows`, and the public input `pi`. The flush blocks
/// reference columns only by INDEX and the program only through its public columns, so
/// this needs no committed witness: both prover and verifier reconstruct exactly the
/// same structure.
#[timed]
pub fn layout(
    prog_cols: &BytecodeColumns,
    log_mem: usize,
    rows: [usize; tables::N_TABLES],
    mem_rows: usize,
    pi: [F192; 2],
) -> Layout {
    let taus = table_logs(rows);
    let bytecode_size = prog_cols[0].len();
    let log_bytecode = crate::log2_strict_usize(bytecode_size);

    // Derived boundary: the run starts at (pc,fp) = (0,0) and, by convention, the
    // final pc is the bytecode's last cell g^{B-1} (the compiler emits a halt jump
    // there), with fp returned to 0. All public, no trace needed.
    let final_pc = (bytecode_size - 1) as u32;

    let one = F64::ONE;

    // ---- bus blocks ----
    use Coord::{Col, Const, Index, Public};

    let mut push: Vec<Block> = Vec::new();
    let mut pull: Vec<Block> = Vec::new();

    // Shared blocks (cross-instruction infra, not owned by any single table).
    // boundary state.
    push.push(Block::full(0, vec![Const(SEP_STATE), Const(one), Const(one)]));
    pull.push(Block::full(
        0,
        vec![Const(SEP_STATE), Const(g_pow(final_pc as usize)), Const(one)],
    ));
    // memory seed + finalize (every address real, no padding). The value is the
    // full three-limb 192-bit word.
    push.push(Block::full(
        log_mem,
        vec![
            Const(SEP_MEM),
            Index,
            Const(one),
            Col(MEM_LO),
            Col(MEM_HI),
            Col(MEM_TOP),
        ],
    ));
    pull.push(Block::full(
        log_mem,
        vec![
            Const(SEP_MEM),
            Index,
            Col(MFCNT),
            Col(MEM_LO),
            Col(MEM_HI),
            Col(MEM_TOP),
        ],
    ));
    // bytecode seed + finalize (program columns are public; padding entries
    // self-cancel at count 1, so the whole 2^log_bytecode is "real").
    let bytecode_block = |count: Coord| {
        Block::full(
            log_bytecode,
            [Const(SEP_BYTECODE), Index, count]
                .into_iter()
                .chain(prog_cols.iter().cloned().map(Public))
                .collect(),
        )
    };
    push.push(bytecode_block(Const(one)));
    pull.push(bytecode_block(Col(BFCNT)));

    // Per-table blocks: each table declares its flushes and read-count columns in
    // local indices; offset them to the table's global columns.
    let sch = schema();
    let mut count_blocks: Vec<Block> = Vec::new();
    for (t, table) in tables::tables().iter().enumerate() {
        let base = sch.base[t];
        let padding = table.padding();
        let table_block = |coords: Vec<Coord>, gate: Option<usize>, follows: Option<usize>| Block {
            kappa: taus[t],
            rows: rows[t],
            padding: coords.iter().map(|c| crate::leaf::coord_at(c, &padding)).collect(),
            coords: offset_coords(base, coords),
            gate: gate.map(|g| base + g),
            follows: follows.map(|g| base + g),
            live: rows[t],
        };
        let mut fb = FlushBuilder::new();
        table.flushes(&mut fb);
        for (side, coords) in [(&mut push, fb.push), (&mut pull, fb.pull)] {
            side.extend(coords.into_iter().zip(&fb.gates).map(|(coords, &gate)| {
                let block = table_block(coords, gate, gate);
                // A padding row's gated leaf is 1: its gate and every coordinate are 0 there.
                assert!(
                    gate.is_none_or(|g| padding[g] == F64::ZERO && block.padding.iter().all(|v| *v == F64::ZERO)),
                    "a gated block's padding row does not flush"
                );
                block
            }));
        }
        count_blocks.extend(table.count_columns().iter().map(|&c| {
            let follows = fb.gated_counts.iter().find(|&&(count, _)| count == c).map(|&(_, gate)| gate);
            table_block(vec![Col(c)], None, follows)
        }));
    }

    let (placements, shape) = witness::placements_of(&col_kappas(log_mem, log_bytecode, taus));
    let dense = dense_layout(log_mem, log_bytecode, rows, mem_rows);
    Layout {
        push,
        pull,
        count: count_blocks,
        placements,
        shape,
        dense,
        mem_rows,
        pi,
        taus,
        rows,
    }
}

impl Layout {
    /// Bound the prover's leaves of every block that follows a gate by the rows where the gate can be 1:
    /// one past the last `KECCAK` row writing output cell 1, and cells `2..13` ([`Trace::keccak_live`]).
    pub(crate) fn set_live(&mut self, keccak_live: [usize; 2]) {
        let base = schema().base[tables::KECCAK_TABLE];
        for blk in self.push.iter_mut().chain(&mut self.pull).chain(&mut self.count) {
            if let Some(gate) = blk.follows {
                blk.live = match gate - base {
                    tables::keccakt::OUT_TWO => keccak_live[0],
                    tables::keccakt::OUT_ALL => keccak_live[1],
                    _ => unreachable!("a gate is a KECCAK output flag"),
                };
                debug_assert!(blk.live <= blk.rows);
            }
        }
    }
}

/// Every table's log height: the least that holds its rows, at least its floor.
pub(crate) fn table_logs(rows: [usize; tables::N_TABLES]) -> [usize; tables::N_TABLES] {
    let taus = std::array::from_fn(|t| crate::log2_ceil_usize(rows[t]).max(tables::TABLE_LOG_FLOORS[t]));
    debug_assert_eq!(
        taus[tables::KECCAK_TABLE],
        crate::keccak::n_steps_log(rows[tables::KECCAK_TABLE]),
        "the KECCAK table is the Keccak reduction's batch"
    );
    taus
}

impl Program {
    /// `keccak_blocks` is what [`crate::keccak::while_executing`] built behind `exec`'s run.
    #[timed]
    pub(crate) fn build(&self, exec: &Execution, keccak_blocks: crate::keccak::Blocks) -> Witness {
        assert!(self.prog.len().is_power_of_two());
        assert!(exec.mem.len().is_power_of_two());
        // The trace was emitted in the same walk as the memory image (no re-walk).
        let tr = &exec.trace;
        let cells = exec.mem.len();
        let log_mem = crate::log2_strict_usize(cells);

        let sch = schema();
        // g^0.., so every pc/frame/operand/count fill is an O(1) lookup instead of an
        // O(log) power: the run's own table covers every one.
        let gpow = exec.gpow.powers();

        // The public layout (flush/count blocks, placements, boundary, taus) is a pure
        // function of the program + announced sizes + public input, with no committed
        // witness; reconstruct it here so the prover and verifier share exactly the same
        // structure. It comes before the fill because it fixes each table's height
        // `2^tau`, which lets every column be allocated at its final length in one pass.
        let row_counts = exec.trace.row_counts();
        assert!(
            row_counts.iter().all(|&r| r <= 1 << MAX_LOG_ROWS),
            "a table exceeds 2^{MAX_LOG_ROWS} rows"
        );
        let pi = [exec.mem.word(0), exec.mem.word(1)];
        // The committed memory rows: the used cells, but the whole memory where only that clears the floor
        // `execute` sized the memory for.
        let log_bytecode = crate::log2_strict_usize(self.prog.len());
        let used = exec.mem_used.next_multiple_of(1 << witness::CHUNK_LOG).min(cells);
        let mem_rows = if committed_log(log_mem, used, log_bytecode, row_counts) < self.min_log_committed {
            cells
        } else {
            used
        };
        let l = layout(&self.bytecode_cols, log_mem, row_counts, mem_rows, pi);

        // The committed stack is written exactly ONCE: allocate it, carve each committed column's runs out of
        // it, and have every fill write its column straight into place. The prover keeps no padded copy: a
        // column's rows past its chunks are padding, which its readers know ([`Witness::columns`]).
        //
        // The fill below writes every word: `Dense::split` zeroes the words no column holds; `fill_table`
        // checks that each table wrote every window it was given, and the shared columns below write theirs.
        let mut repeats = l.dense.alloc_repeats();
        let mut virt = Vec::new();
        let mut keccak_reduction = None;
        let dense = primitives::filled_vec(l.dense.shape.committed_len(), |dense| {
            // An output lane is neither in the stack nor in `q_keccak`, so its values need
            // storage of their own: it carries data for the bus. Every other column `q_keccak`
            // carries ([`Witness::columns`]) is filled through the slot window handed out below.
            let steps = 1 << l.taus[tables::KECCAK_TABLE];
            virt = (0..sch.n)
                .filter(|&i| matches!(keccak_value(i), Some(KeccakLane::Output(_))))
                // SAFETY: `FillCtx::cols_at` writes every row of every window it is given, and
                // `fill_table` asserts each table wrote all of its windows.
                .map(|i| (i, unsafe { primitives::uninit_vec::<F64>(steps) }))
                .collect();
            let (mut windows, padded) = l.dense.split(dense, &mut repeats, &l.placements);
            for (i, buf) in virt.iter_mut() {
                windows[*i] = tables::ColumnOut::flat(buf, &mut []);
            }
            let (mut bytecode_counts, mut q_keccak) = (None, None);
            for (column, words, tags) in padded {
                match column {
                    BFCNT => bytecode_counts = Some(words),
                    QKECCAK => q_keccak = Some((words, tags)),
                    _ => unreachable!("only BFCNT and q_keccak are padded blocks"),
                }
            }
            let bytecode_counts = bytecode_counts.expect("BFCNT is a padded block");
            // q_keccak's block is the committed states and then the spare slots, a slot per
            // KECCAK column riding them: hand each its slot, so its fill writes it where it
            // lives and the stack never holds it twice.
            // The states stay untagged in the repeat map: a missing repeat is only a read. A spare slot
            // takes its own tags, which line up whenever a slot spans whole blocks.
            let (words, tags) = q_keccak.expect("q_keccak is a padded block");
            let (states, spare) = words.split_at_mut(crate::keccak::SLOTS * steps);
            let per_slot = steps >> ::pcs::stack_open::EQUAL_BLOCK_LOG;
            let (_, spare_tags) = tags.split_at_mut(crate::keccak::SLOTS * per_slot);
            let mut spare_tags = spare_tags.chunks_mut(per_slot.max(1));
            for (k, words) in spare.chunks_exact_mut(steps).enumerate() {
                let repeats = if per_slot > 0 {
                    spare_tags.next().unwrap_or_default()
                } else {
                    &mut []
                };
                windows[sch.base[tables::KECCAK_TABLE] + spare_local(k)] = tables::ColumnOut::flat(words, repeats);
            }

            // Each table fills its own columns from the trace (local indices, offset
            // into its global block).
            crate::stage!("Fill columns", || {
                for (t, table) in tables::tables().iter().enumerate() {
                    let (base, n) = (sch.base[t], table.n_committed_columns());
                    let ctx = FillCtx::new(tr, &exec.mem, gpow, &self.prog, 1 << l.taus[t], table.padding());
                    tables::fill_table(*table, &ctx, &mut windows[base..base + n]);
                }
                // Shared columns. The 192-bit memory image splits into three K-limbs.
                // These five plus `QKECCAK` below are every shared column, and each has to
                // be written: the stack is uninitialized, so one left out would be read
                // as indeterminate bytes rather than caught by a length mismatch.
                const _: () = assert!(N_SHARED == 6, "a new shared column needs a fill here");
                // One pass over the image fills all four of its columns, streamed out a
                // cache line of each at a time: nothing reads them before the commitment.
                // Only the committed rows are written: past them the cells are unwritten.
                const LINE: usize = 8;
                const BLOCK: usize = crate::witness::RUN;
                let written = exec.mem.cells(BLOCK);
                let image = windows
                    .get_disjoint_mut([MEM_LO, MEM_HI, MEM_TOP, MFCNT])
                    .expect("the image columns are distinct")
                    .map(|window| {
                        assert_eq!(window.len, cells, "a memory column spans the image");
                        std::mem::take(&mut window.runs)
                    });
                let [lo, hi, top, counts] = image;
                assert!(written.len() <= lo.len() * BLOCK, "the committed rows hold every written cell");
                let mut runs: Vec<[&mut [F64]; 4]> = (lo.into_iter().zip(hi).zip(top).zip(counts))
                    .map(|(((lo, hi), top), counts)| [lo, hi, top, counts])
                    .collect();
                parallel::chunks_mut(&mut runs, 1, |b, image| {
                    let image = &mut image[0];
                    let stream = primitives::stream::Stream::new();
                    // Cells past the written ones are unwritten and never accessed.
                    let unwritten = [F64::ZERO, F64::ZERO, F64::ZERO, gpow[0]].map(|v| [v; LINE]);
                    let at = b * BLOCK;
                    for line in (at..at + BLOCK).step_by(LINE) {
                        // A chunk past a memory under one chunk holds zeros.
                        let mut buf = if line < cells { unwritten } else { [[F64::ZERO; LINE]; 4] };
                        if let Some(cells) = written.get(line..line + LINE) {
                            for (j, cell) in cells.iter().enumerate() {
                                // counts ended at g^{A[i]}
                                buf[0][j] = F64(cell.word.c0);
                                buf[1][j] = F64(cell.word.c1);
                                buf[2][j] = F64(cell.word.c2);
                                buf[3][j] = gpow[cell.count as usize];
                            }
                        }
                        for (column, values) in image.iter_mut().zip(&buf) {
                            stream.copy(&mut column[line - at..][..LINE], values);
                        }
                    }
                });
                parallel::fill(bytecode_counts, |i| gpow[tr.bytecode_count[i] as usize]); // … at g^{A[pc]}
            });
            // The committed Keccak states q_keccak, ALWAYS in this same stack: built from
            // the executed KECCAK rows in order (row j = step j).
            keccak_reduction = Some(crate::stage!("Build q_keccak", || {
                keccak_blocks.place(tr.keccak.len(), states)
            }));

            // (`execute` already asserts the run halts at the sentinel (pc, fp) =
            // (g^{B-1}, 0), exactly the boundary the public layout derives.)
            drop(windows); // release the borrow of the stack, its repeat map and the output lanes' buffers
        });
        let keccak_reduction = keccak_reduction.expect("the fill builds q_keccak");
        Witness {
            dense,
            repeats,
            virt,
            layout: l,
            log_mem,
            keccak_reduction,
        }
    }
}
