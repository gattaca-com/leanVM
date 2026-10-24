//! The tables' local constraints (§sec:air), proven by one sumcheck for all tables.
//!
//! Each table folds its identities with a DISJOINT range of one `η`'s powers, so
//! the batch is a polynomial in `η` whose coefficients are the individual sums and
//! matching the batch's target still pins each one. The three bus forms are the
//! exception: they SHARE their three powers across tables
//! ([`crate::cpu::xi_form_base`]), so those coefficients are per-side totals and
//! the target pins the total, which is all the bus needs. The identities vanish on a
//! valid row, but a table also attaches its three bus forms, whose sums are the
//! values the bus is owed, so the target is those rather than zero. It is
//! derived from the bus's leaf claims, never read off the stream.
//!
//! Tables of different heights are combined by back-loaded batching: table `t`'s
//! summand is lifted onto the common `n`-cube by `∏_{i ≥ τ_t} X_i`, which leaves
//! its hypercube sum alone. Rounds bind `X_{n-1}` first, so table `t` sits out the
//! first `n − τ_t` and joins at round `n − τ_t` weighted by the challenges it sat
//! out. Two payoffs: every table active in a round binds the same variable, so one
//! eq table serves the round; and the claims land on nested points `ρ[..τ_t]`.
//!
//! With nonzero sums the waiting tables stop dropping out: in a round it sits out, a
//! table's variable reaches its summand once, through the padding product, so its
//! contribution is degree 1 in that variable and vanishes at 0, and all of them
//! share the same challenge product. The round polynomial is the cubic
//! `eq(ζ_m, Y)·p(Y) + Y·u`. Its constant, quadratic and cubic coefficients are
//! sent; the running claim fixes the linear coefficient. The verifier evaluates
//! it at the challenge. Heights and `ζ` enter only the per-table `weights`, which
//! may be accumulated along the way or deferred to the end, as the recursion guest does.
//!
//! The eq point is the caller's, not a fresh one (the bus's GKR point `ζ`), which
//! is what lets the forms' sums settle the bus. Batching derived in `doc/leanvm/main.tex`
//! §sec:air. Both sides take `n = max τ_t` from the tables' heights; a recursive
//! verifier certifies that maximum with one hinted `g`-power (§recursion), so there
//! are no rounds in which no table has joined.

use crate::gkr::prefetch_lines;
use crate::transcript::ProverState;
use flux_profiler::timed;
use primitives::field::{F64, F64x8, F192, F192Unreduced, F192x8, Prod8, Weights8, dot_base, powers};
use primitives::multilinear::{eq_table, interp, poly_eval, shrink_eq_high};

/// One table's involved columns' evaluations at its table-sumcheck point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claims {
    pub chi: Vec<F192>,
    pub evals: Vec<F192>,
}

/// One table's row constraint: identity `i` is weighted by `pows[i]`; `true` selects its homogeneous quadratic part.
pub type Constraint<'a> = Box<dyn Fn(&[F192], &[F192], bool) -> F192 + Sync + 'a>;

/// One table's place in the shared batch.
pub struct Air<'a> {
    pub tau: usize,
    /// Rows before the table's padding, at most `2^tau`.
    pub rows: usize,
    pub n_constraints: usize,
    pub eval: Constraint<'a>,
}

/// Start of each table's disjoint range of `η`-powers.
pub fn xi_offsets(n_constraints: impl Iterator<Item = usize>) -> Vec<usize> {
    n_constraints
        .scan(0usize, |off, n| {
            let start = *off;
            *off += n;
            Some(start)
        })
        .collect()
}

/// A linear form over a table's columns, as `(column, coefficient)` terms.
type Lin = Vec<(usize, F192)>;

/// Blocks ahead that the passes over the `K` columns prefetch: they read many columns at once,
/// more streams than the hardware prefetcher follows.
const PREFETCH: usize = 4;

/// Blocks of eight rows per task of the round passes.
const TASK_BLOCKS: usize = 128;

/// Variables of a block of rows: the eq tables of the bus and of the constraints split here.
pub const BLOCK_VARS: usize = 10;
pub const BLOCK: usize = 1 << BLOCK_VARS;

/// What the bus computed for one table while summing its columns at `ζ`: for every column and every
/// product pair its forms use, the dot of each block of `2^BLOCK_VARS` rows against `eq(ζ[..BLOCK_VARS])`.
/// Since `eq` factors by block, these give the round sums of everything linear in the columns with no pass
/// over rows. A table of at most one block carries none (`n_blocks == 0`).
#[derive(Default)]
pub struct TableBlocks {
    pub n_blocks: usize,
    pub n_cols: usize,
    /// The pairs whose products' dots follow the columns', in LOCAL column indices.
    pub pairs: Vec<(usize, usize)>,
    /// Column runs, then pair runs, each `n_blocks` dots.
    pub dots: Vec<F192>,
}

impl TableBlocks {
    fn run(&self, i: usize) -> &[F192] {
        &self.dots[i * self.n_blocks..][..self.n_blocks]
    }

    fn prod(&self, a: usize, b: usize) -> Option<&[F192]> {
        let i = self.pairs.iter().position(|&p| p == (a, b) || p == (b, a))?;
        Some(self.run(self.n_cols + i))
    }
}

/// `Σ w·v` over the `(w, v)` terms.
fn dot_e(terms: impl Iterator<Item = (F192, F192)>) -> F192 {
    terms
        .fold(F192Unreduced::ZERO, |acc, (w, v)| acc ^ w.mul_unreduced(v))
        .reduce()
}

/// A block-dotted table's linear part, per block: `lb[h] = Σ_c lin_c·D_c(h)`, folded with the
/// challenges like a state column, and `end[h] = lb[h] + Σ q_ab·PD_ab(h)`, the whole endpoint
/// before the table joins.
struct Dots {
    lb: Vec<F192>,
    end: Vec<F192>,
}

impl Dots {
    /// `None` when the table has no blocks, or its form multiplies a pair the bus did not sum.
    fn new(form: &Form, blocks: &TableBlocks) -> Option<Self> {
        if blocks.n_blocks == 0 {
            return None;
        }
        let prods = form
            .groups
            .iter()
            .flat_map(|(left, right)| right.iter().map(move |&(c, q)| Some((q, blocks.prod(*left, c)?))))
            .collect::<Option<Vec<_>>>()?;
        let at = |terms: &[(F192, &[F192])], h: usize| dot_e(terms.iter().map(|&(w, run)| (w, run[h])));
        let lin: Vec<_> = form.lin.iter().map(|&(c, w)| (w, blocks.run(c))).collect();
        let lb: Vec<F192> = (0..blocks.n_blocks).map(|h| at(&lin, h)).collect();
        let end = (0..blocks.n_blocks).map(|h| lb[h] + at(&prods, h)).collect();
        Some(Self { lb, end })
    }
}

/// `Σ_{h < eqh.len()} eqh[h]·v[h + off]`: the endpoint sum of per-block values, `off` the upper half's.
fn block_endpoint(v: &[F192], eqh: &[F192], at_one: bool) -> F192 {
    let off = if at_one { eqh.len() } else { 0 };
    dot_e(eqh.iter().copied().zip(v[off..].iter().copied()))
}

/// Every column's `F_c(x) = Σ_h eqh[h]·col_c[x + BLOCK·h]`, `BLOCK` values each, column after column:
/// the column folded by the challenges above the block, which leave `BLOCK` rows.
///
/// Only the first `real` blocks are read. Every block past them holds the same value in each of a
/// column's rows (a table's padding), so together they add that value times their weights to every `x`.
pub fn partial_evals(cols: &[&[F64]], eqh: &[F192], real: usize) -> Vec<F192> {
    /// Vectors per set of accumulators.
    const VECS: usize = 4;
    /// Rows per set of accumulators.
    const STEP: usize = 8 * VECS;
    /// Sets of accumulators per column.
    const STEPS: usize = BLOCK / STEP;
    /// Blocks ahead that the reads are prefetched: a stride of `BLOCK` rows is beyond the hardware prefetcher.
    const AHEAD: usize = 16;
    let real = real.min(eqh.len());
    let mass = eqh[real..].iter().fold(F192::ZERO, |acc, &e| acc + e);
    // SAFETY: the tasks below write every entry.
    let mut out = unsafe { primitives::uninit_vec::<F192>(cols.len() * BLOCK) };
    parallel::chunks_mut(&mut out, BLOCK, |c, out| {
        let col = cols[c];
        let padding = (real < eqh.len()).then(|| {
            F192x8::splat(mass).mul_base_unreduced(F64x8::from_array(&[col[real * BLOCK]; 8]))
        });
        for x in (0..BLOCK).step_by(STEP) {
            let mut acc = [Prod8::ZERO; VECS];
            // The reads of all columns in order: by column, then set of accumulators, then block.
            let ahead = (c * STEPS + x / STEP) * real + AHEAD;
            let (mut set, mut block) = (ahead / real.max(1), ahead % real.max(1));
            for (h, &e) in eqh[..real].iter().enumerate() {
                if let Some(next_col) = cols.get(set / STEPS) {
                    let p = next_col.as_ptr().wrapping_add(block * BLOCK + set % STEPS * STEP);
                    prefetch_lines(p.cast(), 64 * VECS);
                }
                block += 1;
                if block == real {
                    (set, block) = (set + 1, 0);
                }
                let w = F192x8::splat(e);
                for (v, a) in acc.iter_mut().enumerate() {
                    *a ^= w.mul_base_unreduced(load_k(col, h * BLOCK + x + 8 * v));
                }
            }
            for (v, a) in acc.iter_mut().enumerate() {
                if let Some(padding) = padding {
                    *a ^= padding;
                }
                out[x + 8 * v..][..8].copy_from_slice(&a.reduce_lanes().to_array());
            }
        }
    });
    out
}

/// Each column's evaluation at `chi`, for a table of at most `BLOCK` rows.
fn evals_at(cols: &[&[F64]], chi: &[F192]) -> Vec<F192> {
    debug_assert!(chi.len() <= BLOCK_VARS);
    let mut eq = eq_table(chi);
    // A table below one vector of rows is zero-padded.
    eq.resize(eq.len().max(8), F192::ZERO);
    let eq: Vec<Weights8> = eq.as_chunks::<8>().0.iter().map(Weights8::new).collect();
    cols.iter()
        .map(|col| {
            let mut padded = [F64::ZERO; 8];
            let rows = if col.len() < 8 {
                padded[..col.len()].copy_from_slice(col);
                &padded[..]
            } else {
                col
            };
            dot_base(&eq, rows).reduce()
        })
        .collect()
}

/// A table's summand `P(v) = constant + lin(v) + Σ_g v[left_g] · right_g(v)`, read off its
/// degree-2 `eval` once. Its homogeneous quadratic part is `Q(v) = Σ_g v[left_g] · right_g(v)`.
///
/// The `right_g` and `lin` are linear, so a joined table folds them as single `E` columns
/// instead of folding every column: its state is `[lin, left_0, right_0, left_1, right_1, ..]`.
struct Form {
    constant: F192,
    lin: Lin,
    groups: Vec<(usize, Lin)>,
    /// Every column a term reads, ascending.
    used: Vec<usize>,
    /// The columns the products read, ascending.
    quad_used: Vec<usize>,
}

impl Form {
    /// Probe `eval` on unit vectors: in characteristic 2, `Q(e_a + e_b) + Q(e_a) + Q(e_b)` is the
    /// coefficient of `v_a v_b`. Products are then grouped by a greedy vertex cover: the column
    /// with the most remaining products becomes a group's left factor.
    fn extract(eval: &Constraint<'_>, pows: &[F192], n_cols: usize) -> Self {
        let mut v = vec![F192::ZERO; n_cols];
        let mut probe = |cols: &[usize], quadratic| {
            cols.iter().for_each(|&c| v[c] = F192::ONE);
            let out = eval(pows, &v, quadratic);
            cols.iter().for_each(|&c| v[c] = F192::ZERO);
            out
        };
        let constant = probe(&[], false);
        let squares: Vec<F192> = (0..n_cols).map(|a| probe(&[a], true)).collect();
        let lin: Lin = (0..n_cols)
            .map(|a| (a, probe(&[a], false) + constant + squares[a]))
            .filter(|(_, c)| !c.is_zero())
            .collect();
        let mut edges: Vec<(usize, usize, F192)> = (0..n_cols).map(|a| (a, a, squares[a])).collect();
        for a in 0..n_cols {
            for b in a + 1..n_cols {
                edges.push((a, b, probe(&[a, b], true) + squares[a] + squares[b]));
            }
        }
        edges.retain(|e| !e.2.is_zero());
        let mut groups = Vec::new();
        while !edges.is_empty() {
            let mut degree = vec![0usize; n_cols];
            for &(a, b, _) in &edges {
                degree[a] += 1;
                degree[b] += usize::from(a != b);
            }
            let left = (0..n_cols).max_by_key(|&c| degree[c]).unwrap();
            let mut right = Vec::new();
            edges.retain(|&(a, b, q)| {
                let partner = if a == left { b } else { a };
                let mine = a == left || b == left;
                if mine {
                    right.push((partner, q));
                }
                !mine
            });
            groups.push((left, right));
        }
        let mut quad_used = Vec::new();
        for (left, right) in &groups {
            quad_used.push(*left);
            quad_used.extend(right.iter().map(|t| t.0));
        }
        let mut used: Vec<usize> = lin.iter().map(|t| t.0).chain(quad_used.iter().copied()).collect();
        for cols in [&mut used, &mut quad_used] {
            cols.sort_unstable();
            cols.dedup();
        }
        Self {
            constant,
            lin,
            groups,
            used,
            quad_used,
        }
    }

    /// Columns of the folded state: `lin`, then `left_g` and `right_g` per group.
    fn width(&self) -> usize {
        1 + 2 * self.groups.len()
    }

    /// The columns a pass reads: those of `lin` and the products, or only the products without `with_lin`.
    fn read(&self, with_lin: bool) -> &[usize] {
        if with_lin { &self.used } else { &self.quad_used }
    }

    /// The state of the first `rows` (at most eight) rows, unfolded, from the `K` columns.
    fn small_state(&self, cols: &[&[F64]], rows: usize) -> Vec<Col> {
        let dot = |terms: &Lin, i: usize| {
            terms
                .iter()
                .fold(F192::ZERO, |acc, &(c, w)| acc + w.mul_base(cols[c][i]))
        };
        let mut state = vec![vec![F192x8::ZERO]; self.width()];
        for i in 0..rows {
            state[0][0].set_lane(i, dot(&self.lin, i));
            for (g, (left, right)) in self.groups.iter().enumerate() {
                state[1 + 2 * g][0].set_lane(i, F192::from(cols[*left][i]));
                state[2 + 2 * g][0].set_lane(i, dot(right, i));
            }
        }
        state
    }
}

fn load_k(col: &[F64], i: usize) -> F64x8 {
    F64x8::from_array(col[i..i + 8].try_into().unwrap())
}

fn load_e(col: &[F192], i: usize) -> F192x8 {
    F192x8::from_array(col[i..i + 8].try_into().unwrap())
}

fn splat(terms: &Lin) -> Vec<(usize, F192x8)> {
    terms.iter().map(|&(c, w)| (c, F192x8::splat(w))).collect()
}

/// A worker's scratch for [`load_rows`] over `n` columns: `end` and `sum`.
fn rows_scratch(n: usize) -> (Vec<F64x8>, Vec<F64x8>) {
    (vec![F64x8::ZERO; n], vec![F64x8::ZERO; n])
}

/// Loads the vectors of rows `i` and `i + half` of every column in `read`, after prefetching the ones
/// `PREFETCH` blocks on: `end` takes the upper vector with `at_one` and the lower without, `sum` their sum.
#[inline(always)]
fn load_rows(
    cols: &[&[F64]],
    read: &[usize],
    i: usize,
    half: usize,
    at_one: bool,
    (end, sum): &mut (Vec<F64x8>, Vec<F64x8>),
) {
    for &c in read {
        let ahead = cols[c].as_ptr().wrapping_add(i + 8 * PREFETCH);
        prefetch_lines(ahead.cast(), 64);
        prefetch_lines(ahead.wrapping_add(half).cast(), 64);
        let (lo, hi) = (load_k(cols[c], i), load_k(cols[c], i + half));
        (end[c], sum[c]) = (if at_one { hi } else { lo }, lo + hi);
    }
}

/// A joined state column: one `F192x8` per eight rows, below eight rows one vector holding them all.
type Col = Vec<F192x8>;

/// The eq table and `at_one` of the round after a fold or join. With them the pass also returns
/// that round's message, from the rows it has just built.
type Next<'a> = Option<(&'a [F192], bool)>;

/// The blocks task `t` covers out of `n`.
fn task_blocks(t: usize, n: usize) -> std::ops::Range<usize> {
    t * TASK_BLOCKS..((t + 1) * TASK_BLOCKS).min(n)
}

/// A task's `range` of the first `n` blocks, then of the next `n` when a pass is fused (`passes == 2`).
fn over_passes(range: std::ops::Range<usize>, n: usize, passes: usize) -> impl Iterator<Item = usize> {
    (0..passes).flat_map(move |p| range.clone().map(move |b| b + p * n))
}

fn xor2(a: [Prod8; 2], b: [Prod8; 2]) -> [Prod8; 2] {
    [a[0] ^ b[0], a[1] ^ b[1]]
}

fn finish([endpoint, slope]: [Prod8; 2], constant: F192) -> [F192; 2] {
    [endpoint.reduce() + constant, slope.reduce()]
}

/// An active round of a table that has not joined: its message from the `K` columns, in place.
/// Per vector of rows, the linear forms are `E x K` mixed dots, and every product of the
/// summand is one mixed product against a reduced `right_g`. Without `with_end` the endpoint is
/// left at the constant: the caller has the rest from the bus's block dots.
fn message_k(form: &Form, cols: &[&[F64]], eqr: &[F192], half: usize, at_one: bool, with_end: bool) -> [F192; 2] {
    if half < 8 {
        debug_assert!(with_end, "a table with block dots is taller than a vector");
        return message_e(form, &form.small_state(cols, 2 * half), eqr, half, at_one);
    }
    let lin = if with_end { splat(&form.lin) } else { Vec::new() };
    let rights: Vec<_> = form.groups.iter().map(|(_, right)| splat(right)).collect();
    let dot = |terms: &[(usize, F192x8)], vals: &[F64x8]| {
        terms
            .iter()
            .fold(Prod8::ZERO, |acc, &(c, w)| acc ^ w.mul_base_unreduced(vals[c]))
    };
    let blocks = half / 8;
    let acc = parallel::map_reduce_with_state(
        blocks.div_ceil(TASK_BLOCKS),
        || rows_scratch(cols.len()),
        || [Prod8::ZERO; 2],
        |rows, acc, t| {
            for b in task_blocks(t, blocks) {
                let i = 8 * b;
                load_rows(cols, form.read(with_end), i, half, at_one, rows);
                let (end, sum) = &*rows;
                let (mut endpoint, mut slope) = (dot(&lin, end), Prod8::ZERO);
                for ((left, _), right) in form.groups.iter().zip(&rights) {
                    if with_end {
                        endpoint ^= dot(right, end).reduce_lanes().mul_base_unreduced(end[*left]);
                    }
                    slope ^= dot(right, sum).reduce_lanes().mul_base_unreduced(sum[*left]);
                }
                let eq = load_e(eqr, i);
                acc[0] ^= endpoint.reduce_lanes().mul_unreduced(eq).merge_parities();
                acc[1] ^= slope.reduce_lanes().mul_unreduced(eq).merge_parities();
            }
        },
        xor2,
    );
    finish(acc, form.constant)
}

/// A joined table's round message over the block pairs of `range`: `eq·endpoint` and `eq·slope`, unreduced.
/// `col(c, b)` gives the `[lo, hi]` blocks of pair `b` in state column `c`. Without `with_lin` the
/// endpoint leaves out `lin`: the caller adds it from the bus's block dots.
#[inline(always)]
fn message_blocks(
    form: &Form,
    with_lin: bool,
    at_one: bool,
    eqr: &[F192],
    range: std::ops::Range<usize>,
    col: impl Fn(usize, usize) -> [F192x8; 2],
) -> [Prod8; 2] {
    let end = |p: [F192x8; 2]| if at_one { p[1] } else { p[0] };
    let mut acc = [Prod8::ZERO; 2];
    for b in range {
        let lin = if with_lin { end(col(0, b)) } else { F192x8::ZERO };
        let (mut endpoint, mut slope) = (Prod8::ZERO, Prod8::ZERO);
        for g in 0..form.groups.len() {
            let (left, right) = (col(1 + 2 * g, b), col(2 + 2 * g, b));
            endpoint ^= end(left).mul_unreduced(end(right));
            slope ^= (left[0] + left[1]).mul_unreduced(right[0] + right[1]);
        }
        let eq = load_e(eqr, 8 * b);
        acc[0] ^= (endpoint.reduce_lanes() + lin).mul_unreduced(eq).merge_parities();
        acc[1] ^= slope.reduce_lanes().mul_unreduced(eq).merge_parities();
    }
    acc
}

/// An active round of a joined table, from its folded state.
fn message_e(form: &Form, state: &[Col], eqr: &[F192], half: usize, at_one: bool) -> [F192; 2] {
    if half < 8 {
        // Fewer row pairs than a vector: `[Σ_i eqr[i]·end_i, Σ_i eqr[i]·Q(lo_i + hi_i)]` by lanes.
        // `eqr` sums to one, so the form's constant seeds the endpoint.
        return (0..half).fold([form.constant, F192::ZERO], |acc, i| {
            let (lo, hi) = (|c: usize| state[c][0].lane(i), |c: usize| state[c][0].lane(i + half));
            let end = |c| if at_one { hi(c) } else { lo(c) };
            let (mut endpoint, mut slope) = (end(0), F192::ZERO);
            for g in (1..state.len()).step_by(2) {
                endpoint += end(g) * end(g + 1);
                slope += (lo(g) + hi(g)) * (lo(g + 1) + hi(g + 1));
            }
            [acc[0] + eqr[i] * endpoint, acc[1] + eqr[i] * slope]
        });
    }
    let (blocks, with_lin) = (half / 8, !state[0].is_empty());
    let acc = parallel::fold_reduce(
        blocks.div_ceil(TASK_BLOCKS),
        || [Prod8::ZERO; 2],
        |acc, t| {
            let range = task_blocks(t, blocks);
            let m = message_blocks(form, with_lin, at_one, eqr, range, |c, b| [state[c][b], state[c][b + blocks]]);
            *acc = xor2(*acc, m);
        },
        xor2,
    );
    finish(acc, form.constant)
}

/// The round a table joins, after its challenge `r`: the state `lo + r·(lo + hi)` of every
/// entry, built straight from the `K` columns. By linearity `lin` and `right_g` take the
/// terms `w·lo + (w·r)·(lo + hi)`, so no row pays an `E x E` product. Without `with_lin`,
/// `lin` is left an empty column. With `next`, a task builds the block pairs `(b, b + h/2)` the next
/// round pairs, which give its message.
fn join(
    form: &Form,
    cols: &[&[F64]],
    r: F192,
    half: usize,
    with_lin: bool,
    next: Next,
) -> (Vec<Col>, Option<[F192; 2]>) {
    if half < 8 {
        debug_assert!(with_lin, "a table with block dots is taller than a vector");
        let mut state = form.small_state(cols, 2 * half);
        fold_state(form, &mut state, r, half, None);
        return (state, None);
    }
    let (blocks, width) = (half / 8, form.width());
    // SAFETY: the pass below writes every block of every entry.
    let mut state: Vec<Col> = (0..width)
        .map(|c| {
            if c == 0 && !with_lin {
                Vec::new()
            } else {
                unsafe { primitives::uninit_vec(blocks) }
            }
        })
        .collect();
    let out: Vec<_> = state.iter_mut().map(|c| parallel::SendPtr(c.as_mut_ptr())).collect();
    let pair = |terms: &Lin| -> Vec<_> {
        terms
            .iter()
            .map(|&(c, w)| (c, F192x8::splat(w), F192x8::splat(w * r)))
            .collect()
    };
    let (lin, rights) = (
        if with_lin { pair(&form.lin) } else { Vec::new() },
        form.groups.iter().map(|(_, right)| pair(right)).collect::<Vec<_>>(),
    );
    let r8 = F192x8::splat(r);
    let (n, passes) = if next.is_some() { (blocks / 2, 2) } else { (blocks, 1) };
    let acc = parallel::map_reduce_with_state(
        n.div_ceil(TASK_BLOCKS),
        || rows_scratch(cols.len()),
        || [Prod8::ZERO; 2],
        |rows, acc, t| {
            let range = task_blocks(t, n);
            for j in over_passes(range.clone(), n, passes) {
                load_rows(cols, form.read(with_lin), 8 * j, half, false, rows);
                let (lo, sum) = &*rows;
                let dot = |terms: &[(usize, F192x8, F192x8)]| {
                    terms
                        .iter()
                        .fold(Prod8::ZERO, |acc, &(c, w, wr)| {
                            acc ^ w.mul_base_unreduced(lo[c]) ^ wr.mul_base_unreduced(sum[c])
                        })
                        .reduce_lanes()
                };
                // SAFETY: block `j` of an entry is written by the task that owns it, and by no other.
                let put = |c: usize, v: F192x8| unsafe { *out[c].add(j) = v };
                if with_lin {
                    put(0, dot(&lin));
                }
                for (g, ((left, _), right)) in form.groups.iter().zip(&rights).enumerate() {
                    put(1 + 2 * g, F192x8::from(lo[*left]) + r8.mul_base_unreduced(sum[*left]).reduce_lanes());
                    put(2 + 2 * g, dot(right));
                }
            }
            if let Some((eqr, at_one)) = next {
                // SAFETY: both blocks of each pair were written above, by this task.
                let col = |c: usize, b: usize| unsafe { [*out[c].add(b), *out[c].add(b + n)] };
                *acc = xor2(*acc, message_blocks(form, with_lin, at_one, eqr, range, col));
            }
        },
        xor2,
    );
    (state, next.map(|_| finish(acc, form.constant)))
}

/// Every non-empty column of `state` folded by `r`, `col[i] = col[i] + r·(col[i] + col[i + half])`, the upper
/// half dropped. With `next`, a task folds the block pairs `(b, b + h/2)` the next round pairs, which give its message.
fn fold_state(form: &Form, state: &mut [Col], r: F192, half: usize, next: Next) -> Option<[F192; 2]> {
    if half < 8 {
        for col in state {
            let v = &mut col[0];
            (0..half).for_each(|i| v.set_lane(i, interp(v.lane(i), v.lane(i + half), r)));
        }
        return None;
    }
    let (blocks, r8, with_lin) = (half / 8, F192x8::splat(r), !state[0].is_empty());
    let ptrs: Vec<_> = state.iter_mut().map(|c| parallel::SendPtr(c.as_mut_ptr())).collect();
    let (n, passes) = if next.is_some() { (blocks / 2, 2) } else { (blocks, 1) };
    let acc = parallel::fold_reduce(
        n.div_ceil(TASK_BLOCKS),
        || [Prod8::ZERO; 2],
        |acc, t| {
            let range = task_blocks(t, n);
            for col in ptrs.iter().skip(usize::from(!with_lin)) {
                for j in over_passes(range.clone(), n, passes) {
                    // SAFETY: the task folds the blocks `j` of its range, no other task touches them,
                    // and the upper half is only read.
                    unsafe {
                        let (lo, hi) = (*col.add(j), *col.add(j + blocks));
                        *col.add(j) = lo + (lo + hi) * r8;
                    }
                }
            }
            if let Some((eqr, at_one)) = next {
                // SAFETY: the blocks were folded above, by this task.
                let col = |c: usize, b: usize| unsafe { [*ptrs[c].add(b), *ptrs[c].add(b + n)] };
                *acc = xor2(*acc, message_blocks(form, with_lin, at_one, eqr, range, col));
            }
        },
        xor2,
    );
    state.iter_mut().for_each(|c| c.truncate(blocks));
    next.map(|_| finish(acc, form.constant))
}

/// A block dot column folded by `r`: `lb[h] = lb[h] + r·(lb[h] + lb[h + half])`, the upper half dropped.
fn fold_lb(lb: &mut Vec<F192>, r: F192) {
    let half = lb.len() / 2;
    for i in 0..half {
        lb[i] = interp(lb[i], lb[i + half], r);
    }
    lb.truncate(half);
}

/// One table of the batch, and what the rounds have made of it.
struct Table<'a> {
    tau: usize,
    rows: usize,
    sigma: F192,
    /// Its summand's factor: the challenges it sat out, then the eq factors of the rounds it was active.
    weight: F192,
    form: Form,
    cols: &'a [&'a [F64]],
    dots: Option<Dots>,
    /// The folded state, once the table has joined: a few `E` columns, the batch's largest transients.
    state: Option<Vec<Col>>,
    /// The message a fold or join already made for the next round.
    pending: Option<[F192; 2]>,
    /// [`partial_evals`] of the columns, once the challenges above a block are drawn.
    partial: Option<Vec<F192>>,
}

impl Table<'_> {
    /// The message of the round that binds variable `m`, while the table is active.
    fn message(&mut self, eqr: &[F192], eqh: &[F192], m: usize, at_one: bool) -> [F192; 2] {
        let (form, half) = (&self.form, 1 << m);
        let mut p = match (self.pending.take(), &self.state) {
            (Some(p), _) => p,
            (None, Some(state)) => message_e(form, state, eqr, half, at_one),
            (None, None) => message_k(form, self.cols, eqr, half, at_one, self.dots.is_none()),
        };
        // The block dots stand in for the sum over rows of `lin`, and before the join of the whole endpoint.
        if let Some(dots) = &self.dots {
            p[0] += block_endpoint(if self.state.is_some() { &dots.lb } else { &dots.end }, eqh, at_one);
        }
        p
    }

    /// Binds variable `m` to `r`: the join of a table that has not joined, the fold of one that has.
    fn bind(&mut self, r: F192, m: usize, next: Next) {
        // A dotted table's state has no `lin` until the partial evals, which come after round `BLOCK_VARS`.
        let next = next.filter(|_| m != BLOCK_VARS || self.dots.is_none());
        self.pending = match &mut self.state {
            Some(state) => fold_state(&self.form, state, r, 1 << m, next),
            None => {
                let (state, message) = join(&self.form, self.cols, r, 1 << m, self.dots.is_none(), next);
                self.state = Some(state);
                message
            }
        };
        if let Some(dots) = &mut self.dots {
            fold_lb(&mut dots.lb, r);
        }
    }

    /// With the challenges above a block drawn (`chi`), leaves every column as `BLOCK` values. A dotted
    /// table, which is now a block tall, gets its `lin` column from them and leaves the dots.
    fn fold_to_block(&mut self, chi: &[F192]) {
        let f = partial_evals(self.cols, &eq_table(&chi[BLOCK_VARS..self.tau]), self.rows.div_ceil(BLOCK));
        if self.dots.take().is_some() {
            let lin: Vec<F192> = (0..BLOCK)
                .map(|x| dot_e(self.form.lin.iter().map(|&(c, w)| (w, f[c * BLOCK + x]))))
                .collect();
            self.state.as_mut().expect("the table joined")[0] =
                lin.as_chunks::<8>().0.iter().map(F192x8::from_array).collect();
        }
        self.partial = Some(f);
    }
}

fn round_polynomial([endpoint, quadratic]: [F192; 2], zeta: F192, claim: F192, waiting: F192) -> [F192; 4] {
    let eq_z = F192::ONE + zeta;
    // claim = (1 + zeta) p(0) + zeta p(1) + waiting.
    let (p0, p1) = if zeta.is_zero() {
        (claim + waiting, endpoint)
    } else {
        (endpoint, (claim + waiting + eq_z * endpoint) * zeta.inv())
    };
    let h2 = eq_z * quadratic + p0 + p1 + quadratic;
    [eq_z * p0, claim + h2 + quadratic, h2, quadratic]
}

/// Prove that every table's batched constraint vanishes on all of its rows, as ONE
/// sumcheck over `max τ_t` variables. `cols[t]` holds table `t`'s involved columns
/// (`2^{τ_t}` values each, read in place). Returns the per-table claims, in input
/// order, on the nested points `ρ[..τ_t]`.
///
/// `blocks` are the bus's block dots at the same `zeta` (empty when there are none, as in unit tests).
/// A table that has them, and whose products are all among the bus's pairs, never sums its linear part
/// over rows while it is taller than a block: its endpoint sums come from the dots. Once the challenges
/// above a block are drawn, one pass leaves every column as `BLOCK` values, from which the linear part
/// of the rest and the final claims follow.
#[timed]
pub fn prove(
    airs: &[Air<'_>],
    cols: &[Vec<&[F64]>],
    xi: F192,
    zeta: &[F192],
    sigma: &[F192],
    blocks: &[TableBlocks],
    ps: &mut ProverState,
) -> Vec<Claims> {
    let n = airs.iter().map(|a| a.tau).max().unwrap_or(0);
    debug_assert!(zeta.len() >= n, "the eq point must cover the tallest table");
    let offsets = xi_offsets(airs.iter().map(|a| a.n_constraints));
    let pows = powers(xi, airs.iter().map(|a| a.n_constraints).sum());
    let mut tables: Vec<Table> = airs
        .iter()
        .zip(cols)
        .zip(sigma)
        .enumerate()
        .map(|(t, ((air, cols), &sigma))| {
            let form = Form::extract(&air.eval, &pows[offsets[t]..][..air.n_constraints], cols.len());
            Table {
                tau: air.tau,
                rows: air.rows,
                sigma,
                weight: F192::ONE,
                dots: blocks.get(t).and_then(|b| Dots::new(&form, b)),
                form,
                cols,
                state: None,
                pending: None,
                partial: None,
            }
        })
        .collect();
    // ONE eq table over the low (still free) variables serves every active table.
    let mut eqr = eq_table(&zeta[..n.saturating_sub(1)]);
    // The same table over the variables above a block, for the block dots.
    let top = n.saturating_sub(1);
    let mut eqh = eq_table(&zeta[BLOCK_VARS.min(top)..top]);
    let mut chi = vec![F192::ZERO; n];
    // `k`, the challenges drawn so far, common to every table that is still waiting.
    let mut k = F192::ONE;
    let mut claim = sigma.iter().copied().fold(F192::ZERO, |a, b| a + b);
    for j in 0..n {
        let m = n - 1 - j; // the variable this round binds
        let at_one = zeta[m].is_zero();
        // The waiting tables contribute the line `Y·k·Σσ`, whose slope `u` is all there
        // is to it. It is NOT sent on its own: it folds into `h` below, and only `h`
        // travels. `msg` is the active tables' degree-2 cofactor, `h`'s multiplicand.
        let waiting = tables
            .iter()
            .filter(|t| t.tau <= m)
            .fold(F192::ZERO, |acc, t| acc + t.sigma);
        let u = k * waiting;
        let mut msg = [F192::ZERO; 2];
        for t in tables.iter_mut().filter(|t| t.tau > m) {
            let p = t.message(&eqr, &eqh, m, at_one);
            msg[0] += t.weight * p[0];
            msg[1] += t.weight * p[1];
        }
        shrink_eq_high(&mut eqr);
        if eqh.len() > 1 {
            shrink_eq_high(&mut eqh);
        }
        // The running claim recovers the missing endpoint of the quadratic cofactor.
        // The fold stays separate: its challenge only exists after this message is bound.
        let h = round_polynomial(msg, zeta[m], claim, u);
        ps.add_round_poly(&h, false);
        let rk = ps.sample();
        claim = poly_eval(&h, rk);
        chi[m] = rk;
        k *= rk;
        let eq_k = F192::ONE + zeta[m] + rk;
        // `eqr` now serves round `m - 1`, whose message a fold or join can make once its rows are built,
        // unless it is shorter than a vector.
        let next = (m >= 4).then(|| (&eqr[..], zeta[m - 1].is_zero()));
        for t in &mut tables {
            if t.tau > m {
                t.weight *= eq_k;
                t.bind(rk, m, next);
            } else {
                t.weight *= rk;
            }
        }
        // Every state is now one block tall: the tables above a block fold their columns once more.
        if m == BLOCK_VARS {
            crate::stage!("Constraint partial evals", || {
                for t in tables.iter_mut().filter(|t| t.tau > BLOCK_VARS) {
                    t.fold_to_block(&chi);
                }
            });
        }
    }

    // Each column's evaluation at its table's point: the columns were never folded one by one, so
    // this is one dot per column, over a block's values if the pass above left them.
    let all_evals = crate::stage!("Constraint evals", || {
        let low = eq_table(&chi[..BLOCK_VARS.min(n)]);
        tables
            .iter()
            .map(|t| match &t.partial {
                Some(f) => {
                    let dot = |col: &[F192]| dot_e(low.iter().copied().zip(col.iter().copied()));
                    f.chunks(BLOCK).map(dot).collect()
                }
                None => evals_at(t.cols, &chi[..t.tau]),
            })
            .collect::<Vec<_>>()
    });
    tables
        .iter()
        .zip(all_evals)
        .map(|(t, evals)| {
            ps.add_scalars(&evals);
            Claims {
                chi: chi[..t.tau].to_vec(),
                evals,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const COLS: usize = 6;

    /// Products sharing column 0, a square, linear terms and a constant.
    fn synth_eval(pows: &[F192], v: &[F192], quadratic: bool) -> F192 {
        let products = v[0] * v[1] * pows[0] + (v[0] * v[3] + v[0] * v[4]) * pows[1] + v[5] * v[5] * pows[2];
        if quadratic {
            return products;
        }
        products + v[2] * pows[0] + v[1] * pows[1] + v[4] * pows[2] + F192::ONE
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn random_e(state: &mut u64) -> F192 {
        F192::new(splitmix(state), splitmix(state), splitmix(state))
    }

    fn random_columns(state: &mut u64, rows: usize) -> Vec<Vec<F64>> {
        (0..COLS)
            .map(|_| (0..rows).map(|_| F64(splitmix(state))).collect())
            .collect()
    }

    fn form_of(pows: &[F192]) -> Form {
        let eval: Constraint<'static> = Box::new(synth_eval);
        Form::extract(&eval, pows, COLS)
    }

    #[test]
    fn extracted_form_reproduces_eval() {
        let mut s = 1;
        let pows: Vec<F192> = (0..3).map(|_| random_e(&mut s)).collect();
        let form = form_of(&pows);
        assert_eq!(
            form.groups.len(),
            2,
            "column 0 covers three products, the square is the other group"
        );
        for _ in 0..4 {
            let v: Vec<F192> = (0..COLS).map(|_| random_e(&mut s)).collect();
            let dot = |terms: &Lin| terms.iter().fold(F192::ZERO, |acc, &(c, w)| acc + w * v[c]);
            let quad = form
                .groups
                .iter()
                .fold(F192::ZERO, |acc, (l, right)| acc + v[*l] * dot(right));
            assert_eq!(quad, synth_eval(&pows, &v, true));
            assert_eq!(form.constant + dot(&form.lin) + quad, synth_eval(&pows, &v, false));
        }
    }

    /// `[Σ eqr·P(end), Σ eqr·Q(lo + hi)]` by evaluating the identity row by row.
    fn naive_message(pows: &[F192], cols: &[Vec<F192>], eqr: &[F192], half: usize, at_one: bool) -> [F192; 2] {
        (0..half).fold([F192::ZERO; 2], |acc, i| {
            let row = |off: usize| -> Vec<F192> { cols.iter().map(|c| c[i + off]).collect() };
            let (lo, hi) = (row(0), row(half));
            let sum: Vec<F192> = lo.iter().zip(&hi).map(|(&a, &b)| a + b).collect();
            let end = if at_one { &hi } else { &lo };
            [
                acc[0] + eqr[i] * synth_eval(pows, end, false),
                acc[1] + eqr[i] * synth_eval(pows, &sum, true),
            ]
        })
    }

    /// Every round of a table's life, at row counts on both sides of one vector: the join's
    /// message and state, then each fold and joined message, against the naive formulas. The
    /// message a join or fold makes for the next round is checked too, for both `at_one`.
    #[test]
    fn round_messages_match_naive_evaluation() {
        let mut s = 7;
        let pows: Vec<F192> = (0..3).map(|_| random_e(&mut s)).collect();
        let form = form_of(&pows);
        for m in [1usize, 3, 4, 5, 11] {
            let kcols = random_columns(&mut s, 2 << m);
            let views: Vec<&[F64]> = kcols.iter().map(|c| &c[..]).collect();
            let mut naive_cols: Vec<Vec<F192>> = kcols
                .iter()
                .map(|c| c.iter().map(|&x| F192::from(x)).collect())
                .collect();
            let mut eqr = eq_table(&(0..m).map(|_| random_e(&mut s)).collect::<Vec<_>>());
            let mut state: Option<Vec<Col>> = None;
            let mut pending = None;
            let mut half = 1usize << m;
            while half >= 1 {
                if let Some((message, at_one)) = pending.take() {
                    let want = naive_message(&pows, &naive_cols, &eqr, half, at_one);
                    assert_eq!(message, want, "fused m {m} half {half}");
                }
                for at_one in [false, true] {
                    let got = match &state {
                        None => message_k(&form, &views, &eqr, half, at_one, true),
                        Some(st) => message_e(&form, st, &eqr, half, at_one),
                    };
                    assert_eq!(
                        got,
                        naive_message(&pows, &naive_cols, &eqr, half, at_one),
                        "m {m} half {half}"
                    );
                }
                let r = random_e(&mut s);
                shrink_eq_high(&mut eqr);
                let at_one = half.trailing_zeros() % 2 == 1;
                let next = (half >= 16).then_some((&eqr[..], at_one));
                let message = match &mut state {
                    None => {
                        let (st, message) = join(&form, &views, r, half, true, next);
                        state = Some(st);
                        message
                    }
                    Some(st) => fold_state(&form, st, r, half, next),
                };
                pending = message.map(|message| (message, at_one));
                for c in &mut naive_cols {
                    *c = (0..half).map(|i| interp(c[i], c[i + half], r)).collect();
                }
                half /= 2;
            }
        }
    }

    /// The bus's dots of a table: per block of rows, each column's and each product's against `eq(zeta[..BLOCK_VARS])`.
    fn blocks_of(cols: &[Vec<F64>], zeta: &[F192], pairs: &[(usize, usize)]) -> TableBlocks {
        let n_blocks = cols[0].len() >> BLOCK_VARS;
        let low = eq_table(&zeta[..BLOCK_VARS]);
        let products = pairs
            .iter()
            .map(|&(a, b)| cols[a].iter().zip(&cols[b]).map(|(&x, &y)| x * y).collect::<Vec<F64>>());
        let dots = cols
            .iter()
            .cloned()
            .chain(products)
            .flat_map(|run| {
                let low = &low;
                (0..n_blocks).map(move |h| {
                    run[h * BLOCK..][..BLOCK]
                        .iter()
                        .zip(low)
                        .fold(F192::ZERO, |acc, (&v, &w)| acc + w.mul_base(v))
                })
            })
            .collect();
        TableBlocks {
            n_blocks,
            n_cols: cols.len(),
            pairs: pairs.to_vec(),
            dots,
        }
    }

    /// Tables of different heights in one batch, from empty to a few blocks: each table's claims are its
    /// columns' true evaluations at a prefix of the tallest table's point, and the bus's block dots change
    /// no byte of the transcript. Tables above a block join at, above and below its height, one of them
    /// lacks a product's dots, and two zero coordinates of `zeta` make those rounds read the upper halves.
    #[test]
    fn ragged_batch_claims_are_true_evaluations() {
        let taus = [13usize, 12, 11, 12, 5, 9, 6, 0, 1, 4];
        let mut s = 5;
        let cols: Vec<Vec<Vec<F64>>> = taus.iter().map(|&tau| random_columns(&mut s, 1 << tau)).collect();
        let airs: Vec<Air<'static>> = taus
            .iter()
            .map(|&tau| Air {
                tau,
                rows: 1 << tau,
                n_constraints: 3,
                eval: Box::new(synth_eval),
            })
            .collect();
        let xi = random_e(&mut s);
        let mut zeta: Vec<F192> = (0..13).map(|_| random_e(&mut s)).collect();
        zeta[10] = F192::ZERO;
        zeta[11] = F192::ZERO;
        let sigma: Vec<F192> = (0..taus.len()).map(|_| random_e(&mut s)).collect();
        let views: Vec<Vec<&[F64]>> = cols.iter().map(|t| t.iter().map(|c| &c[..]).collect()).collect();
        // The form multiplies (0,1), (0,3), (0,4) and (5,5); the bus names a pair either way round.
        let pairs = [(1, 0), (0, 3), (4, 0), (5, 5)];
        let blocks: Vec<TableBlocks> = taus
            .iter()
            .zip(&cols)
            .enumerate()
            .map(|(t, (&tau, table))| match tau > BLOCK_VARS {
                true => blocks_of(table, &zeta, if t == 3 { &pairs[..2] } else { &pairs }),
                false => TableBlocks::default(),
            })
            .collect();
        let pows: Vec<F192> = (0..3).map(|_| random_e(&mut s)).collect();
        let form = form_of(&pows);
        assert!(
            Dots::new(&form, &blocks[0]).is_some(),
            "all of the form's pairs are dotted"
        );
        assert!(Dots::new(&form, &blocks[3]).is_none(), "a pair is missing");
        let run = |blocks: &[TableBlocks]| {
            let mut ps = ProverState::from_label(b"zc-test");
            let claims = prove(&airs, &views, xi, &zeta, &sigma, blocks, &mut ps);
            (claims, ps.sample())
        };
        let (plain, dotted) = (run(&[]), run(&blocks));
        assert_eq!(plain, dotted);
        let tallest = &dotted.0[0].chi;
        for ((claim, &tau), table) in dotted.0.iter().zip(&taus).zip(&cols) {
            assert_eq!(claim.chi, tallest[..tau]);
            for (eval, col) in claim.evals.iter().zip(table) {
                assert_eq!(*eval, primitives::multilinear::mle_eval(col, &claim.chi));
            }
        }
    }
}
