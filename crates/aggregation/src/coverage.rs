//! The coverage table: which slot each raw signature writes.

use super::error::AggregateError;
use super::proof::AggregateClaims;
use super::{LeanSphincsClaim, LeanXmssClaim};
use std::collections::BTreeSet;

/// The coverage slot each raw signature's write in the guest's coverage walk targets.
/// A child's claims never enter the table: the node's commitment hashes the child's.
///
/// The table is one contiguous region per scheme, each its declared claims followed
/// by its own duplicate slots:
///
/// | slots | holds |
/// | --- | --- |
/// | `[0, n_lean_xmss)` | the declared leanXMSS claims |
/// | `[n_lean_xmss, X)` | leanXMSS duplicate slots |
/// | `[X, X + n_lean_sphincs)` | the declared leanSPHINCS claims |
/// | `[X + n_lean_sphincs, n_total)` | leanSPHINCS duplicate slots |
///
/// A claim the declared set holds takes its slot there, one it does not takes a
/// fresh duplicate slot in the same region, so the walk hits every one of the
/// `n_total` slots exactly once. That bijection, enforced in-circuit by write-once
/// memory plus the final count, is what makes every declared claim covered by a real
/// signature.
///
/// Keeping each region contiguous is what binds the scheme: the guest addresses
/// every writer as an offset into one region, bounded by that region's size, one
/// range check per write, so no signature can reach a claim of the other scheme.
pub(super) struct Coverage {
    pub(super) lean_xmss: Region<LeanXmssClaim>,
    pub(super) lean_sphincs: Region<LeanSphincsClaim>,
}

/// One scheme's region of the coverage table.
pub(super) struct Region<K> {
    /// The declared claims, strictly sorted.
    pub(super) signers: Vec<K>,
    pub(super) dups: Vec<K>,
    /// The slot of each raw signature, in the order given.
    pub(super) raw: Vec<usize>,
}

impl<K: Ord + Clone> Region<K> {
    /// Declares the raw claims, or exactly `declare`, which they must cover.
    fn plan(raw: &[K], declare: Option<&[K]>) -> Result<Self, AggregateError> {
        let mut signers = raw.to_vec();
        signers.sort();
        signers.dedup();
        if let Some(declare) = declare {
            let mut wanted: BTreeSet<&K> = declare.iter().collect();
            signers.retain(|claim| wanted.remove(claim));
            if !wanted.is_empty() {
                return Err(AggregateError::InvalidInput(
                    "a requested claim is not covered by the inputs",
                ));
            }
        }
        let mut claimed = vec![false; signers.len()];
        let mut dups = Vec::new();
        let raw = raw
            .iter()
            .map(|claim| take_slot(&signers, &mut claimed, &mut dups, claim))
            .collect();
        Ok(Self { signers, dups, raw })
    }
}

/// A claim takes its declared slot once; subsequent or omitted occurrences take
/// fresh slots outside the hashed prefix.
fn take_slot<K: Ord + Clone>(claims: &[K], claimed: &mut [bool], duplicates: &mut Vec<K>, claim: &K) -> usize {
    match claims.binary_search(claim) {
        Ok(pos) if !claimed[pos] => {
            claimed[pos] = true;
            pos
        }
        _ => {
            duplicates.push(claim.clone());
            claims.len() + duplicates.len() - 1
        }
    }
}

/// The table for a node's raw signatures. `declare` names the raw claims to publish: its tree's own lists, which
/// [`aggregate`](super::aggregate) has checked to have no children.
pub(super) fn plan_coverage(
    raw_lean_xmss: &[LeanXmssClaim],
    raw_lean_sphincs: &[LeanSphincsClaim],
    declare: Option<&AggregateClaims>,
) -> Result<Coverage, AggregateError> {
    Ok(Coverage {
        lean_xmss: Region::plan(raw_lean_xmss, declare.map(|claims| claims.tree.lean_xmss.as_slice()))?,
        lean_sphincs: Region::plan(
            raw_lean_sphincs,
            declare.map(|claims| claims.tree.lean_sphincs.as_slice()),
        )?,
    })
}
