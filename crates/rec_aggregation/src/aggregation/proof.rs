//! The aggregate, its wire format and its native verification.

use super::deferred::{DeferredClaim, scalar_bytecode};
use super::error::{AggregateVerifyError, SnarkVerifyError};
use super::statement::{statement_digest, tree_commitment};
use super::{LeanSphincsClaim, LeanXmssClaim, MAX_RECURSIONS};
use leanvm_core::cpu::{public_input_bytes, scalar_proof};
use primitives::field::F128;

/// What an aggregate attests: every `(pubkey, message)` in [`Self::lean_xmss`] is backed by a valid
/// leanXMSS signature, at a leaf index the statement does not record, every one in
/// [`Self::lean_sphincs`] by a valid leanSPHINCS signature, and the [`Self::da_commitment`], if any,
/// commits to blob rows that are valid Reed-Solomon codewords.
///
/// The two lists remain separate because the proof says which scheme verified which claim (the
/// module docs give the coverage argument). A proof's lists are each strictly sorted and deduplicated
/// on the whole `(pubkey, message)` pair by flattening its [`ClaimTree`].
/// Each node holds strictly sorted raw lists. Either list may be empty.
///
/// **Neither length counts signers, only claims.** One pubkey may appear under several messages, so a
/// committee threshold has to count distinct pubkeys itself.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EthereumClaims {
    pub lean_xmss: Vec<LeanXmssClaim>,
    pub lean_sphincs: Vec<LeanSphincsClaim>,
    /// `None` makes no blob claim. The statement carries its [`verifier::da::digest`].
    pub da_commitment: Option<[u8; 32]>,
}

/// The claims a node commits to, in the shape of its recursion: each child's tree in the order
/// the node verified them, then the raw claims the node verified itself, each list strictly
/// sorted. A node publishes everything it verified; [`EthereumProof::claims`] is the sorted,
/// deduplicated flattening of the tree.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClaimTree {
    pub children: Vec<ClaimTree>,
    pub lean_xmss: Vec<LeanXmssClaim>,
    pub lean_sphincs: Vec<LeanSphincsClaim>,
}

/// A proof of its [`ClaimTree`], established directly or carried from verified child proofs.
///
/// Until [`Self::verify`] returns `Ok` the tree and its claims are claims, not attestation. A receiver
/// that decoded the proof with [`Self::from_proof_bytes`] supplied the tree itself; otherwise, as for
/// [`super::aggregate`]'s output, they are the prover's choice, and a caller has to compare them
/// against what it expected.
#[derive(Clone, Debug)]
pub struct EthereumProof {
    /// What the statement commits to: the claims in the shape of the recursion.
    pub(super) tree: ClaimTree,
    /// The flattening of [`Self::tree`], and the DA root.
    pub(super) claims: EthereumClaims,
    /// What this aggregate defers to whoever discharges it: its parent, in
    /// circuit, or [`Self::verify`], natively.
    pub(super) defer: DeferredClaim,
    pub(super) proof: leanvm_core::cpu::Proof,
}

// The wire format: the deferred point as a little-endian u64 count of limb pairs `c0, c1`,
// each a little-endian u64, then the VM proof in its own encoding, to the end. The claim tree is not
// sent: the receiver knows what it expects a proof to attest, and supplies it.
//
// Given the tree, every byte string decodes to at most one aggregate and back to itself: the VM
// proof's decoder rejects trailing bytes. Otherwise anything downstream that dedupes or indexes on the
// serialized bytes could be made to see one aggregate as many.

pub(super) fn encode(point: &[F128], proof: &[u8]) -> Vec<u8> {
    let mut out = (point.len() as u64).to_le_bytes().to_vec();
    out.extend(point.iter().flat_map(|x| x.to_limbs()).flat_map(u64::to_le_bytes));
    out.extend(proof);
    out
}

/// The deferred point and the VM proof's bytes. The point's limbs are cut from the input before
/// any is read, so a forged count allocates nothing.
pub(super) fn decode(bytes: &[u8]) -> Option<(Vec<F128>, &[u8])> {
    let (count, rest) = bytes.split_first_chunk::<8>()?;
    let count = usize::try_from(u64::from_le_bytes(*count)).ok()?;
    let (limbs, proof) = rest.split_at_checked(count.checked_mul(16)?)?;
    let limb = |bytes: &[u8; 8]| u64::from_le_bytes(*bytes);
    let pairs = limbs.as_chunks::<8>().0.as_chunks::<2>().0;
    let point = pairs.iter().map(|[c0, c1]| F128::new(limb(c0), limb(c1))).collect();
    Some((point, proof))
}

/// Reject unsupported arities and raw lists that are not strictly sorted.
pub(super) fn check_claim_tree(tree: &ClaimTree) -> Result<(), AggregateVerifyError> {
    if tree.children.len() > MAX_RECURSIONS
        || !tree.lean_xmss.windows(2).all(|w| w[0] < w[1])
        || !tree.lean_sphincs.windows(2).all(|w| w[0] < w[1])
    {
        return Err(AggregateVerifyError::MalformedSignerSet);
    }
    tree.children.iter().try_for_each(check_claim_tree)
}

/// Every claim under `tree`, sorted and deduplicated per scheme.
pub(super) fn flatten_claims(tree: &ClaimTree) -> (Vec<LeanXmssClaim>, Vec<LeanSphincsClaim>) {
    fn walk(tree: &ClaimTree, lean_xmss: &mut Vec<LeanXmssClaim>, lean_sphincs: &mut Vec<LeanSphincsClaim>) {
        for child in &tree.children {
            walk(child, lean_xmss, lean_sphincs);
        }
        lean_xmss.extend_from_slice(&tree.lean_xmss);
        lean_sphincs.extend_from_slice(&tree.lean_sphincs);
    }
    let (mut lean_xmss, mut lean_sphincs) = (Vec::new(), Vec::new());
    walk(tree, &mut lean_xmss, &mut lean_sphincs);
    lean_xmss.sort();
    lean_xmss.dedup();
    lean_sphincs.sort();
    lean_sphincs.dedup();
    (lean_xmss, lean_sphincs)
}

impl EthereumClaims {
    /// The signature claims, as many as the coverage table's declared slots.
    ///
    /// NOT a count of distinct signers: a pubkey of either scheme may hold several
    /// claims, one per message it signed. A caller that wants signers has to
    /// deduplicate by pubkey itself.
    pub fn num_signature_claims(&self) -> usize {
        self.lean_xmss.len() + self.lean_sphincs.len()
    }
}

impl EthereumProof {
    /// The VM's public input for this aggregate: the commitment to its claim tree, its DA digest
    /// and its deferred claim.
    pub(super) fn vm_public_input(&self) -> [F128; 2] {
        let da = verifier::da::digest(self.claims.da_commitment);
        statement_digest(tree_commitment(&self.tree), da, &self.defer)
    }

    /// What this aggregate claims, which [`Self::verify`] checks: the sorted, deduplicated
    /// flattening of [`Self::claim_tree`].
    pub fn claims(&self) -> &EthereumClaims {
        &self.claims
    }

    /// The claims in the shape of the recursion that verified them, which the statement commits to.
    pub fn claim_tree(&self) -> &ClaimTree {
        &self.tree
    }

    /// What goes on the wire: the deferred point and the VM proof. Neither the claim tree nor the
    /// deferred claim's value is transmitted: the receiver supplies the tree to
    /// [`Self::from_proof_bytes`], which recomputes the value, so there is nothing to lie about.
    pub fn proof_bytes(&self) -> Vec<u8> {
        encode(&self.defer.bytecode_point, &self.proof.to_bytes())
    }

    /// The aggregate `bytes` encode, as a proof of `tree` and `da_commitment`: what the receiver
    /// expects it to attest. A tree or root other than the aggregated ones fails [`Self::verify`].
    ///
    /// WARNING: Parsing does NOT verify: the proof is untouched, only shapes are checked.
    /// Call [`Self::verify`] before believing any of it.
    pub fn from_proof_bytes(
        bytes: &[u8],
        tree: ClaimTree,
        da_commitment: Option<[u8; 32]>,
    ) -> Result<Self, AggregateVerifyError> {
        let (bytecode_point, proof) = decode(bytes).ok_or(AggregateVerifyError::MalformedEncoding)?;
        // Cheap rejections first. `recompute_scalar` below is a pass over the whole stacked
        // bytecode, on a point a peer chose, so anything decidable without it has to be
        // decided before it.
        check_claim_tree(&tree)?;
        let proof = leanvm_core::cpu::Proof::from_bytes(proof).ok_or(AggregateVerifyError::MalformedEncoding)?;
        let (lean_xmss, lean_sphincs) = flatten_claims(&tree);
        Ok(Self {
            tree,
            claims: EthereumClaims {
                lean_xmss,
                lean_sphincs,
                da_commitment,
            },
            // The wire carries no value, only the point to derive it from.
            defer: DeferredClaim::recompute_scalar(bytecode_point)?,
            proof,
        })
    }

    /// Verify the aggregate's internal consistency: the signer set is well
    /// formed, the deferred bytecode claim holds at its
    /// transmitted point, and the VM proof satisfies the statement built from
    /// all of it.
    ///
    /// What `formal/` proves of it, in the random-permutation model, is one node: the guest ran on these claims, each
    /// a signature it verified itself or a claim of a child proof it verified. That the children's claims are valid
    /// signatures in turn, and that the `da_commitment`, if any, commits to a well-formed blob matrix, are not proved
    /// (`formal/statement.tex`, "What is not proved"). The claims are the receiver's own after
    /// [`Self::from_proof_bytes`]; which leaf index each leanXMSS signature used is not part of the claim.
    ///
    /// Every field operation, the proof check and its Merkle hashing run in the `verifier` crate: one thread, no explicit SIMD.
    pub fn verify(&self) -> Result<(), AggregateVerifyError> {
        check_claim_tree(&self.tree)?;
        // Discharging the deferred claim IS recomputing it: its value has
        // to be the true evaluation, or the recursion below proves nothing about
        // the bytecode.
        let _s = tracing::info_span!("Recompute deferred claims").entered();
        let defer = DeferredClaim::recompute_scalar(self.defer.bytecode_point.clone())?;
        drop(_s);
        if defer != self.defer {
            return Err(AggregateVerifyError::MalformedClaim);
        }
        let _s = tracing::info_span!("Statement digest").entered();
        let pi = self.vm_public_input();
        drop(_s);
        verifier::verify_execution(scalar_bytecode(), &public_input_bytes(&pi), &scalar_proof(&self.proof))
            .map(|_| ())
            .map_err(|e| AggregateVerifyError::Snark(SnarkVerifyError(e)))
    }
}
