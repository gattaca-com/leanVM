//! Recursive proofs of leanXMSS and leanSPHINCS signature claims and LeanDA blob well-formedness.
//! One bytecode (`guests/lean_ethereum.py`) serves every node of an aggregation tree.
//!
//! A node verifies `n_raw_lean_xmss` leanXMSS signatures, `n_raw_lean_sphincs` leanSPHINCS
//! signatures and `n_children` sub-proofs **of this same bytecode**, and commits to
//! the claims in the shape of that recursion ([`ClaimTree`]): its signer commitment
//! hashes each child's commitment, then its own raw claim lists. A claim of either
//! scheme is a `(pubkey, message)` pair; a leanXMSS signature carries its own leaf
//! index, which the statement does not record. A child's claims never enter the
//! parent's tables: the parent hashes the child's commitment, which the child's
//! statement pins, and verifies the child's proof against that statement. The native
//! verifier rebuilds the commitment from the tree and exposes the flattened, sorted,
//! deduplicated set.
//! Coverage carries the security claim for the raw claims a node checks itself: a
//! write-once slot per declared claim, written once by each raw signature, plus a
//! final count, so every declared raw claim is backed by a real signature.
//!
//! Those slots are one contiguous region per scheme, so the one range check a write
//! already needs also keeps a signature off the other scheme's declared claims: that
//! is what makes the published split mean which scheme verified which claim.
//!
//! A duplicate slot sits outside the prefix the digest hashes, so a raw claim in one is
//! covered and not published. `aggregate`'s `declare` rests on that. A slot holds
//! four cells, the pubkey and its message, so the guest reads each signature's message
//! out of the slot it verifies.
//!
//! The bytecode is compiled to a fixed point on its own size
//! ([`unified_guest`]): the recursion placeholders depend on the inner bytecode
//! size, and here the inner bytecode is this one. Its digest does not need a
//! fixed point, riding the statement instead of the code.
//!
//! One fixed polynomial, the stacked bytecode, is too
//! big to evaluate in-circuit, so each node exports one deferred claim on it
//! and batches its children's carried claims with the fresh ones its
//! verifications raise (`doc/leanvm/main.tex` §Deferred evaluation claims). Only
//! the root's is discharged natively, by [`EthereumProof::verify`].
//!
//! `gen_verify` derives the guest's whole witness for a child from the real
//! `cpu::layout` of the inner program and the summary the `verifier` crate
//! reads out of the child's proof, so there is no hand-mirrored copy of the
//! protocol to drift.

mod cells;
mod child;
mod claim_pool;
mod coverage;
mod deferred;
mod error;
mod guest;
mod hints;
mod node;
mod placeholders;
mod proof;
mod statement;
#[cfg(test)]
mod tests;

pub use error::{AggregateVerifyError, AggregationError, SnarkProveError, SnarkVerifyError};
pub use guest::{unified_guest, warm_up};
pub use node::{aggregate, aggregate_with_stats};
pub use proof::{ClaimTree, EthereumClaims, EthereumProof};
use signatures::lean_sphincs::LeanSphincsPublicKey;
use signatures::lean_xmss::LeanXmssPublicKey;
use signatures::{lean_sphincs, lean_xmss};

/// One leanXMSS claim: a pubkey, and the message it signed at whatever leaf index its
/// signature carries.
pub type LeanXmssClaim = (LeanXmssPublicKey, lean_xmss::Message);

/// One leanSPHINCS claim: a pubkey, and the message it signed.
pub type LeanSphincsClaim = (LeanSphincsPublicKey, lean_sphincs::Message);

/// The most earlier aggregates one [`aggregate`] call can take, so the arity of
/// an aggregation tree.
pub const MAX_RECURSIONS: usize = 16;

/// The most levels a [`ClaimTree`] may have, the root being level 1.
///
/// An aggregation tree is a few levels deep: leaves, a node over them, a root. 32 leaves room
/// for any realistic pipeline, and keeps every recursion over a checked tree far from a stack limit.
pub const MAX_TREE_DEPTH: usize = 32;

/// The most nodes a [`ClaimTree`] may have in all. A tree of 4096 leaves under one root is 4352
/// nodes, so this is generous, and it bounds the hashing and flattening a hostile tree can force.
pub const MAX_TREE_NODES: usize = 1 << 16;
