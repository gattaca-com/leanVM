//! The thread pool that leanXMSS signing searches its randomizers on and, with `aggregation`, the
//! prover proves on: one pool per process. With `aggregation`, signing and `setup` go through
//! `lean_vm`, so they reach the prover's own pool; without, this crate's dependency on the same pool
//! crate, the experimental prover's.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::OnceLock;

use lean_xmss::{LeanXmssExpandedSecretKey, LeanXmssSignature};

use crate::Error;

/// The thread count `setup` asked for, `0` for one per physical core, once the pool is set up.
static POOL: OnceLock<usize> = OnceLock::new();

/// The process that set up the pool, set before it starts: a child forked since inherits it but
/// not the threads running the pool.
pub(crate) static POOL_OWNER: AtomicU32 = AtomicU32::new(0);
/// An owner that every process may use.
pub(crate) const BUILT: u32 = u32::MAX;

/// Starts the thread pool with `threads` threads (0: one per physical core), which leanXMSS signing
/// and aggregation run on, and with `aggregation` builds the aggregation guest, once per process.
///
/// - Signing never starts the pool: before `setup` or a proof, it runs on the calling thread.
/// - Signing uses the pool when it is free, and the calling thread while a proof or another
///   signature holds it, or in a process forked from one that started it.
/// - `setup(1)` keeps signing and proving on the calling thread.
/// - The signatures are the same on any number of threads.
///
/// # Errors
///
/// - Fails when `threads` exceeds the machine's parallelism.
/// - Fails when the pool already runs with another thread count.
/// - Fails in a process forked while or after its parent started the pool.
pub fn setup(threads: usize) -> Result<(), Error> {
    let max = std::thread::available_parallelism().map_or(1, usize::from);
    if threads > max {
        return Err(Error::TooManyThreads { threads, max });
    }
    let running = start(threads)?;
    if running != threads {
        return Err(Error::SetupThreads { running, threads });
    }
    Ok(())
}

/// Sets up the pool with `threads` threads unless it is already, and returns the count it was set
/// up with, `0` for one per physical core.
pub(crate) fn start(threads: usize) -> Result<usize, Error> {
    claim(&POOL_OWNER)?;
    #[cfg(feature = "aggregation")]
    crate::aggregate::build_guest()?;
    Ok(*POOL.get_or_init(|| {
        #[cfg(feature = "aggregation")]
        lean_vm::setup(threads);
        #[cfg(not(feature = "aggregation"))]
        {
            parallel::set_num_threads(threads);
            parallel::init();
        }
        threads
    }))
}

/// Signs on the pool when it is set up and free, else on the calling thread.
pub(crate) fn sign(
    sk: &LeanXmssExpandedSecretKey,
    message: &lean_xmss::Message,
    leaf_index: lean_xmss::LeafIndex,
) -> LeanXmssSignature {
    #[cfg(feature = "aggregation")]
    return lean_vm::lean_xmss::sign(sk, message, leaf_index);
    #[cfg(not(feature = "aggregation"))]
    return lean_xmss::sign_with::<Pool>(sk, message, leaf_index);
}

/// The pool, as a randomizer search.
#[cfg(not(feature = "aggregation"))]
struct Pool;

#[cfg(not(feature = "aggregation"))]
impl lean_xmss::FindFirst for Pool {
    fn find_first(n: usize, pred: impl Fn(usize) -> bool + Sync) -> Option<usize> {
        parallel::find_first_if_idle(n, pred)
    }
}

/// Claims `owner` for this process, or fails in a process forked from the one that did.
pub(crate) fn claim(owner: &AtomicU32) -> Result<(), Error> {
    let pid = std::process::id();
    match owner.compare_exchange(0, pid, Ordering::AcqRel, Ordering::Acquire) {
        Err(other) if other != pid && other != BUILT => Err(Error::ForkedAfterSetup),
        _ => Ok(()),
    }
}
