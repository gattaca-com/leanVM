//! The aggregate, its wire format and its native verification.

use super::deferred::{DeferredBytecodeClaim, bytecode_vars};
use super::guest::unified_guest;
use super::statement::{statement_digest, tree_commitment};
use super::{LeanSphincsClaim, LeanXmssClaim, MAX_RECURSIONS, MAX_TREE_DEPTH, MAX_TREE_NODES};
use lean_vm_core::cpu::public_input_bytes;
use primitives::field::F128;

/// The claims a node commits to, in the shape of its recursion: each child's tree in the order
/// the node verified them, then the raw claims the node verified itself, each list strictly
/// sorted. A node publishes everything it verified; [`AggregateClaims::flatten`] is the sorted,
/// deduplicated flattening of the tree.
///
/// A verified proof guarantees that each `(pubkey, message)` in a node's lists has a verified
/// signature in that node. It does NOT guarantee distinct signers, uniqueness across nodes, or
/// any bound on the total: the same claim may repeat in several nodes. A consumer applying a
/// threshold must count distinct public keys over [`AggregateClaims::flatten`], and must not sum
/// per-child counts. The tree is at most [`MAX_TREE_DEPTH`] levels and [`MAX_TREE_NODES`] nodes.
///
/// [`AggregateProof::verify`] and [`aggregate`](super::aggregate) check a tree before they recurse
/// over it. `Clone`, `Debug`, `==` and the drop of a tree built over those bounds recurse once a
/// level, so they remain the builder's risk.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClaimTree {
    pub children: Vec<ClaimTree>,
    pub lean_xmss: Vec<LeanXmssClaim>,
    pub lean_sphincs: Vec<LeanSphincsClaim>,
}

/// What an [`AggregateProof`] proves: a [`ClaimTree`], every `(pubkey, message)` in whose
/// leanXMSS lists has a valid leanXMSS signature, at a leaf index the statement does not record, and
/// every one in whose leanSPHINCS lists has a valid leanSPHINCS signature; and, optionally, a
/// commitment that is the root of a well-formed matrix of blobs (see [`lean_da::BlobMatrix`]).
///
/// The two schemes' lists remain separate because the proof says which scheme verified which claim
/// (the crate docs give the coverage argument). Each node's lists are strictly sorted and may be
/// empty. Until [`AggregateProof::verify`] returns `true` these are claims, not attestation: a receiver
/// builds the ones it expects, and a caller of [`aggregate`](super::aggregate) compares its output
/// against what it expected.
///
/// **No length counts signers, only claims.** One pubkey may appear under several messages, so a
/// committee threshold has to count distinct pubkeys itself.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AggregateClaims {
    pub tree: ClaimTree,
    /// `None` for no blobs.
    pub da_commitment: Option<[u8; 32]>,
}

/// Which covered claims [`aggregate`](super::aggregate) proves.
#[derive(Clone, Copy, Debug)]
pub enum OutputClaims<'a> {
    /// Every raw signature's claim. A node publishes its children's claims whole. The DA root is `blobs`' if any,
    /// otherwise the first child's that has one, otherwise none.
    All,
    /// Exactly the raw claims in this tree's own lists, which has no children, and this DA root: each raw claim
    /// covered by a raw signature. A child's claims cannot be narrowed, since the node's commitment hashes the
    /// child's whole.
    Only(&'a AggregateClaims),
}

/// A proof of some [`AggregateClaims`], which travel separately: [`Self::verify`] checks it against
/// the claims the receiver expects.
#[derive(Clone, Debug)]
pub struct AggregateProof {
    /// The point of the deferred bytecode claim, `bytecode_vars()` long; its value is computed from it.
    pub(super) bytecode_point: Vec<F128>,
    pub(super) proof: verifier::Proof,
}

// Wire format: the deferred point (a u64 count, then each element as two u64 limbs, all little-endian),
// then the VM proof. The encoding is canonical: the VM proof's decoder rejects trailing bytes. The claim tree
// is not sent: the receiver knows what it expects a proof to attest, and supplies it to `verify`.

pub(super) fn encode(point: &[F128], proof: &[u8]) -> Vec<u8> {
    let mut out = (point.len() as u64).to_le_bytes().to_vec();
    out.extend(point.iter().flat_map(|x| x.to_limbs()).flat_map(u64::to_le_bytes));
    out.extend(proof);
    out
}

/// The deferred point and the VM proof's bytes. A forged count allocates nothing.
pub(super) fn decode(bytes: &[u8]) -> Option<(Vec<F128>, &[u8])> {
    let (count, rest) = bytes.split_first_chunk::<8>()?;
    let count = usize::try_from(u64::from_le_bytes(*count)).ok()?;
    let (limbs, proof) = rest.split_at_checked(count.checked_mul(16)?)?;
    let limb = |bytes: &[u8; 8]| u64::from_le_bytes(*bytes);
    let pairs = limbs.as_chunks::<8>().0.as_chunks::<2>().0;
    let point = pairs.iter().map(|[c0, c1]| F128::new(limb(c0), limb(c1))).collect();
    Some((point, proof))
}

/// Whether `tree` is within the bounds on arity, depth and node count, and every raw list is strictly sorted, so
/// that no claim counts twice in its node. Either list may be empty, and both may be in a blob-only tree. Hashing
/// or flattening the tree comes after this.
pub(super) fn is_well_formed(tree: &ClaimTree) -> bool {
    // An explicit stack, not recursion: a hostile tree must not reach the native stack. Depth and
    // node count are bounded before a node's children are pushed, so the stack holds at most
    // `MAX_TREE_NODES` entries. Every recursion over a tree that passes is at most `MAX_TREE_DEPTH` deep.
    let mut nodes = 1;
    let mut stack = vec![(tree, 1)];
    while let Some((node, depth)) = stack.pop() {
        nodes += node.children.len();
        if depth > MAX_TREE_DEPTH
            || nodes > MAX_TREE_NODES
            || node.children.len() > MAX_RECURSIONS
            || !node.lean_xmss.windows(2).all(|w| w[0] < w[1])
            || !node.lean_sphincs.windows(2).all(|w| w[0] < w[1])
        {
            return false;
        }
        stack.extend(node.children.iter().map(|child| (child, depth + 1)));
    }
    true
}

impl AggregateClaims {
    /// Every claim under the tree, sorted and deduplicated per scheme on the whole `(pubkey, message)`
    /// pair: what a consumer reads, whatever shape the recursion had. Each claim has a verified signature in
    /// some node of the tree, once [`AggregateProof::verify`] holds.
    pub fn flatten(&self) -> (Vec<LeanXmssClaim>, Vec<LeanSphincsClaim>) {
        let (mut lean_xmss, mut lean_sphincs) = (Vec::new(), Vec::new());
        let mut stack = vec![&self.tree];
        while let Some(node) = stack.pop() {
            lean_xmss.extend_from_slice(&node.lean_xmss);
            lean_sphincs.extend_from_slice(&node.lean_sphincs);
            stack.extend(&node.children);
        }
        lean_xmss.sort();
        lean_xmss.dedup();
        lean_sphincs.sort();
        lean_sphincs.dedup();
        (lean_xmss, lean_sphincs)
    }

    /// The number of distinct `(pubkey, message)` claims in [`Self::flatten`], not of distinct signers: a
    /// pubkey may sign several messages.
    pub fn num_signature_claims(&self) -> usize {
        let (lean_xmss, lean_sphincs) = self.flatten();
        lean_xmss.len() + lean_sphincs.len()
    }
}

impl AggregateProof {
    pub fn to_bytes(&self) -> Vec<u8> {
        encode(&self.bytecode_point, &self.proof.to_bytes())
    }

    /// Decode a proof, `None` if the bytes are not one. This does not verify: call [`Self::verify`].
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let (bytecode_point, proof) = decode(bytes)?;
        if bytecode_point.len() != bytecode_vars() {
            return None;
        }
        let proof = verifier::Proof::from_bytes(proof)?;
        Some(Self { bytecode_point, proof })
    }

    /// Whether this proves `claims`: the tree is within its bounds and its lists are sorted, the deferred claim
    /// holds, and the stark proof is valid.
    pub fn verify(&self, claims: &AggregateClaims) -> bool {
        // The cheap check first: the rest evaluates the bytecode at a point the sender chose.
        if !is_well_formed(&claims.tree) {
            return false;
        }
        let defer = DeferredBytecodeClaim::recompute(self.bytecode_point.clone());
        let da = verifier::da::digest(claims.da_commitment);
        let pi = statement_digest(tree_commitment(&claims.tree), da, &defer);
        let verdict = verifier::verify_execution(unified_guest().bytecode(), &public_input_bytes(&pi), &self.proof);
        if let Err(error) = &verdict {
            tracing::debug!(%error, "the proof does not verify");
        }
        verdict.is_ok()
    }
}
