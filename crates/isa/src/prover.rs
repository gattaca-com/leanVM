//! What the crates both provers share (`aggregation`, `lean_vm`) need of a prover. Each prover tree implements
//! [`Prover`] once, beside its `cpu::prove`.

use crate::Program;
use verifier::{E, layout::N_TABLES};

/// A prover of ISA programs, and the thread pool and field kernels it brings for the work around a proof.
pub trait Prover: 'static {
    /// The prover's name: `stable` or `experimental`.
    const NAME: &'static str;

    /// Warm up the prover's thread pool, so that no timed work pays for starting it. Idempotent.
    fn init();

    /// Fixes the thread pool's size before it starts; `0` keeps the default. Panics if it is already another.
    fn set_num_threads(n: usize);

    /// Prove an execution of `program` on `public_input` at the code rate `2^-log_inv_rate`, or say why there is none.
    /// `witness` names the streams of its `hint_witness` calls, each with one entry per call, in the order they are
    /// popped, and no two with one name.
    fn prove(
        program: &Program,
        witness: &[(String, Vec<Vec<E>>)],
        public_input: [E; 2],
        log_inv_rate: usize,
    ) -> Result<(verifier::Proof, Stats), String>;

    /// `f(i)` for every `i < n`, on the thread pool. `f` must not dispatch to the pool itself.
    fn map_collect<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T>;

    /// The least `i < n` with `pred(i)`, or `None`: on the thread pool when it can start at once (started in this
    /// process, no dispatch running, not called from one of its tasks), else on the calling thread. It never starts,
    /// sizes or waits for the pool. `pred` must not dispatch to the pool itself.
    fn find_first_if_idle(n: usize, pred: impl Fn(usize) -> bool + Sync) -> Option<usize>;

    /// `f(i)` for every `i < n`, in order: on the thread pool when it can start at once, else on the calling thread, as
    /// [`Self::find_first_if_idle`] decides. `f` must not dispatch to the pool itself.
    fn map_collect_if_idle<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T>;

    // The deferred claims' batching (`aggregation`): its two passes over the stacked bytecode, in the prover's
    // fastest form.

    /// `Σ_r weights[r] · rows[r]`, over the `weights.len()` rows of `K` words that `rows` holds end to end. A prover
    /// may require the row count and the row width to be multiples of 8.
    fn combine_rows(rows: &[u64], weights: &[E]) -> Vec<E>;

    /// Each row's dot product with `weights`, over the rows of `weights.len()` `K` words that `rows` holds end to end.
    /// A prover may require the row width to be a multiple of 8.
    fn dot_rows(rows: &[u64], weights: &[E]) -> Vec<E>;
}

/// Run statistics returned with a proof.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    /// Executed instructions.
    pub cycles: usize,
    /// Rows per table the program executed, in the order of [`Self::TABLES`].
    pub table_heights: [usize; N_TABLES],
    /// Committed words: the columns' lengths, before the stack's pad to a power of two.
    pub total_committed: usize,
    /// Cells actually touched, before the pad to a power of two.
    pub memory_used: usize,
}

impl Stats {
    /// Table names in `table_heights` order.
    pub const TABLES: [&'static str; N_TABLES] = ["XOR", "MUL", "SET", "DEREF", "JUMP", "KECCAK"];
}
