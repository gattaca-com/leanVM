//! Proving and verifying, and LeanDA's encoding.

use crate::{AggregateClaims, OutputClaims};
use ::lean_sphincs::{LeanSphincsPublicKey, LeanSphincsSignature};
use ::lean_xmss::{LeanXmssPublicKey, LeanXmssSignature};
pub use aggregation::{AggregateError, AggregateProof};
use isa::Prover;

#[cfg(feature = "experimental_prover")]
type P = experimental_dispatch::ExperimentalProver;
#[cfg(not(feature = "experimental_prover"))]
type P = stable_lean_vm_core::StableProver;

// TODO: make leanDA a self-contained dependency, so that it can move to lib.rs
pub mod lean_da {
    pub use ::lean_da::{
        BLOB_SYMBOLS, Blob, BlobCountError, BlobMatrix, CELL_SYMBOLS, CELLS_PER_ROW, Codeword, DA_MAX_ROWS,
        MERKLE_PATH_LEN, Symbol, verify_cell,
    };
}

/// The prover's thread pool, as the schemes' strategy: on the pool once [`setup`] or a proof has started it in this
/// process, and on the calling thread while the pool is busy or before it starts, which key generation and signing
/// never do. Keys and signatures are the same either way, on any number of threads.
struct Pool;

impl ::lean_xmss::Parallel for Pool {
    fn find_first(n: usize, pred: impl Fn(usize) -> bool + Sync) -> Option<usize> {
        P::find_first_if_idle(n, pred)
    }

    fn map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
        P::map_collect_if_idle(n, f)
    }
}

impl ::lean_sphincs::Parallel for Pool {
    fn map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
        P::map_collect_if_idle(n, f)
    }
}

/// The leanXMSS key of `key_seed`, the top tree built on the prover's thread pool when [`setup`] or a proof has
/// started it and it is free, else on the calling thread.
pub fn xmss_key_gen(key_seed: &::lean_xmss::SecretKey) -> (::lean_xmss::LeanXmssExpandedSecretKey, LeanXmssPublicKey) {
    ::lean_xmss::key_gen_with::<Pool>(key_seed)
}

/// Builds and caches the bottom tree of `leaf_index`, so that signing there is faster, on the prover's thread pool
/// when [`setup`] or a proof has started it and it is free, else on the calling thread.
pub fn xmss_prepare(sk: &::lean_xmss::LeanXmssExpandedSecretKey, leaf_index: ::lean_xmss::LeafIndex) {
    sk.prepare_with::<Pool>(leaf_index);
}

/// Signs `message` at `leaf_index`, the randomizer searches and the trees signing builds on the prover's thread pool
/// when [`setup`] or a proof has started it and it is free, else on the calling thread. Never use the same key and leaf
/// index to sign two different messages. The first signature in a bottom tree builds it.
///
/// # Panics
///
/// When none of a layer's `2^32` randomizer trials gives a codeword, with probability below `2^-2954`: impossible in
/// practice.
pub fn xmss_sign(
    sk: &::lean_xmss::LeanXmssExpandedSecretKey,
    message: &::lean_xmss::Message,
    leaf_index: ::lean_xmss::LeafIndex,
) -> LeanXmssSignature {
    ::lean_xmss::sign_with::<Pool>(sk, message, leaf_index)
}

/// The leanSPHINCS public key of `secret_key`, its top tree built on the prover's thread pool when [`setup`] or a proof
/// has started it and it is free, else on the calling thread.
pub fn sphincs_key_gen(secret_key: &::lean_sphincs::SecretKey) -> LeanSphincsPublicKey {
    ::lean_sphincs::key_gen_with::<Pool>(secret_key)
}

/// Signs `message`, deterministically and statelessly, the trees signing builds on the prover's thread pool when
/// [`setup`] or a proof has started it and it is free, else on the calling thread. A key signs at most
/// `2^LOG_LIFETIME` messages.
///
/// # Panics
///
/// When all encoding counters of a layer miss a codeword, with probability below `2^-378193`: impossible in practice.
pub fn sphincs_sign(secret_key: &::lean_sphincs::SecretKey, message: &::lean_sphincs::Message) -> LeanSphincsSignature {
    ::lean_sphincs::sign_with::<Pool>(secret_key, message)
}

/// Call this function before generating / verifying proofs, and before generating keys and signing to do so on the
/// prover's thread pool: it starts the pool, with `threads` threads (0: one per physical core), picks the experimental
/// prover's CPU tier, and builds the aggregation program.
///
/// # Panics
/// If the pool already runs with another thread count, or `LEANVM_TIER` names a tier this build does not run on this CPU.
pub fn setup(threads: usize) {
    P::set_num_threads(threads);
    P::init();
    aggregation::unified_guest();
}

/// Aggregate leanXMSS and leanSPHINCS signatures, blobs and the claims of child proofs.
///
/// - `children`: at most [`MAX_RECURSIONS`](crate::MAX_RECURSIONS), each with its claims.
/// - `raw_lean_xmss`, `raw_lean_sphincs`: `(public_key, message, signature)`, in any order.
/// - `blobs`: the encoded blobs, if any ([`lean_da::BlobMatrix::build`]).
/// - `output`: which covered claims the proof attests.
/// - `log_inv_rate`: the code rate's log, in [`MIN_LOG_INV_RATE`](crate::MIN_LOG_INV_RATE)..=
///   [`MAX_LOG_INV_RATE`](crate::MAX_LOG_INV_RATE); higher gives a smaller proof but slower proving.
///
/// # Errors
///
/// [`AggregateError`] says why there is no proof.
///
/// Calls run one at a time: a second waits for the first.
///
/// # WARNING
/// - Signatures and children must be valid (children checked with [`AggregateProof::verify`]): otherwise proving fails.
pub fn aggregate(
    children: &[(AggregateClaims, AggregateProof)],
    raw_lean_xmss: Vec<(LeanXmssPublicKey, ::lean_xmss::Message, LeanXmssSignature)>,
    raw_lean_sphincs: Vec<(LeanSphincsPublicKey, ::lean_sphincs::Message, LeanSphincsSignature)>,
    blobs: Option<&lean_da::BlobMatrix>,
    output: OutputClaims,
    log_inv_rate: usize,
) -> Result<(AggregateClaims, AggregateProof), AggregateError> {
    aggregation::aggregate::<P>(children, raw_lean_xmss, raw_lean_sphincs, blobs, output, log_inv_rate)
        .map(|(claims, proof, _)| (claims, proof))
}
