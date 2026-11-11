pub mod aggregation;
/// The Keccak hash chain, proven end to end. A `src` module rather than its own
/// test binary so it shares the process with the other workloads.
#[cfg(test)]
mod hash_chain;
#[cfg(any(test, feature = "signers-cache"))]
pub mod signers_cache;

pub use aggregation::{
    AggregateVerifyError, AggregationError, ClaimTree, EthereumClaims, EthereumProof, LeanSphincsClaim, LeanXmssClaim,
    MAX_RECURSIONS, MAX_TREE_DEPTH, MAX_TREE_NODES, SnarkProveError, SnarkVerifyError, aggregate, aggregate_with_stats,
    warm_up,
};
