//! The verification walk over many signatures at once, each hash one batched call.

use crate::*;

/// `Th` of each `(P, A, payload)`, through the batched hash, the inputs padded to whole batches so that none is hashed
/// alone.
fn th_each<'a, const K: usize>(inputs: impl IntoIterator<Item = (&'a PublicParam, Tweak, [Digest; K])>) -> Vec<Digest> {
    let len = PUBLIC_PARAM_LEN + ADDRESS_LEN + K * N;
    let mut data = Vec::new();
    for (pp, tw, payload) in inputs {
        data.extend_from_slice(pp);
        data.extend_from_slice(&address(&tw));
        data.extend_from_slice(payload.as_flattened());
    }
    let n = data.len() / len;
    data.resize(n.next_multiple_of(keccak256::lanes()) * len, 0);
    let mut out = vec![[0; keccak256::OUT_LEN]; data.len() / len];
    keccak256::hash_many(&data, len, out.as_flattened_mut());
    out[..n].iter().map(|h| keccak256::truncate(*h)).collect()
}

/// Each signature's codewords, bottom layer first, or `None` where a layer's digest is not a codeword: the signatures
/// walked in lockstep, every step one batched hash over all of them. The walk takes each bottom digest, the bottom
/// chains, leaf and path, then the top digest over the root they reach. A signature whose bottom digest is refused
/// walks no chain, its chain values taken as the ends; one refused on its top digest has walked its bottom chains.
///
/// This does not verify. The walk stops short of the top layer's chains, leaf and path, and compares nothing with the
/// key, so `Some` does not mean a signature is valid: [`verify`](crate::verify) says that.
pub fn walk(signers: &[(LeanXmssPublicKey, Message, LeanXmssSignature)]) -> Vec<Option<[[u8; V]; 2]>> {
    let e = |sig: &LeanXmssSignature, layer| layer_index(sig.leaf_index, layer);
    let digest_tweak = |sig, layer| tweak(TWEAK_MSG, layer_hi(layer, 0), e(sig, layer));

    let bottom: Vec<_> = th_each(signers.iter().map(|(pk, message, sig)| {
        let [m0, m1] = message.as_chunks().0.try_into().unwrap();
        (&pk.public_param, digest_tweak(sig, 0), [m0, m1, sig.bottom.randomizer])
    }))
    .iter()
    .map(codeword)
    .collect();
    let starts: Vec<[u8; V]> = bottom.iter().map(|x| x.unwrap_or([CHAIN_LEN as u8 - 1; V])).collect();
    let mut ends: Vec<[Digest; V]> = signers.iter().map(|(_, _, sig)| sig.bottom.chain_values).collect();
    for from in 0..CHAIN_LEN - 1 {
        let steps: Vec<(usize, usize)> = (starts.iter().enumerate())
            .flat_map(|(k, start)| (0..V).filter(|&i| usize::from(start[i]) <= from).map(move |i| (k, i)))
            .collect();
        let digests = th_each(steps.iter().map(|&(k, i)| {
            let (pk, _, sig) = &signers[k];
            let (tw, payload) = chain_input(0, e(sig, 0), i, from, ends[k][i]);
            (&pk.public_param, tw, payload)
        }));
        for (&(k, i), digest) in steps.iter().zip(digests) {
            ends[k][i] = digest;
        }
    }

    let mut nodes = th_each(
        (signers.iter().zip(&ends)).map(|((pk, _, sig), &ends)| (&pk.public_param, leaf_tweak(0, e(sig, 0)), ends)),
    );
    for z in 0..LAYER_HEIGHT {
        nodes = th_each(signers.iter().zip(&nodes).map(|((pk, _, sig), &node)| {
            let (e, sibling) = (e(sig, 0), sig.bottom.path[z]);
            let children = if (e >> z) & 1 == 0 {
                [node, sibling]
            } else {
                [sibling, node]
            };
            (&pk.public_param, node_tweak(0, z + 1, e >> (z + 1)), children)
        }));
    }
    let top = th_each(
        (signers.iter().zip(&nodes))
            .map(|((pk, _, sig), &root)| (&pk.public_param, digest_tweak(sig, 1), [root, sig.top.randomizer])),
    );
    (bottom.into_iter().zip(top))
        .map(|(bottom, top)| Some([bottom?, codeword(&top)?]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The walk one signature at a time, through the scheme's own layer walk.
    fn one_by_one((pk, message, sig): &(LeanXmssPublicKey, Message, LeanXmssSignature)) -> Option<[[u8; V]; 2]> {
        let e = |layer| layer_index(sig.leaf_index, layer);
        let (bottom, root) = layer_root(&pk.public_param, 0, e(0), message, &sig.bottom, |_| {})?;
        let (top, _) = layer_root(&pk.public_param, 1, e(1), &root, &sig.top, |_| {})?;
        Some([bottom, top])
    }

    /// Invariant: the lockstep walk gives each signature what the scheme's one-signature layer walk gives, and refuses
    /// exactly the signatures it refuses, on either layer, at any place among any number of others, duplicates and
    /// signers of other keys included.
    #[test]
    fn walk_matches_the_layer_walk() {
        let (sk, pk) = key_gen(&[3; SECRET_KEY_LEN]);
        // Both layers' indices at the bit patterns' edges, and two leaves sharing a bottom tree.
        let mut signers: Vec<_> = [0, 1, u32::MAX, 0x5555_5555, 0xaaaa_aaaa]
            .into_iter()
            .map(|leaf_index| {
                let message = [leaf_index as u8; MESSAGE_LEN];
                (pk, message, sign(&sk, &message, leaf_index))
            })
            .collect();
        signers.push(signers[3].clone());
        let (other_sk, other_pk) = key_gen(&[4; SECRET_KEY_LEN]);
        let message = [7; MESSAGE_LEN];
        signers.insert(2, (other_pk, message, sign(&other_sk, &message, 7)));
        assert!(signers.iter().all(|signer| one_by_one(signer).is_some()));

        // Refused on the bottom digest alone: the chain values advanced to their ends under the signed digits, then a
        // randomizer whose digest is no codeword. Walking no chain, they reach the signed root, over which the top
        // digest is still a codeword, so only the bottom check refuses it.
        let bottom_refused = {
            let (pk, message, mut sig) = signers[3].clone();
            let (pp, e) = (&pk.public_param, sig.leaf_index);
            let digest = |randomizer: &Randomizer| codeword(&message_digest(pp, 0, e, randomizer, &message));
            let digits = digest(&sig.bottom.randomizer).unwrap();
            sig.bottom.chain_values = core::array::from_fn(|i| {
                let start = usize::from(digits[i]);
                chain(pp, 0, e, i, start, CHAIN_LEN - 1 - start, sig.bottom.chain_values[i])
            });
            sig.bottom.randomizer = (0..=u8::MAX)
                .map(|byte| [byte; RANDOMIZER_LEN])
                .find(|randomizer| digest(randomizer).is_none())
                .unwrap();
            (pk, message, sig)
        };
        // Refused on the top digest: its randomizer changed.
        let mut top_refused = signers[4].clone();
        top_refused.2.top.randomizer[0] ^= 1;
        let refused = [bottom_refused, top_refused];
        assert!(refused.iter().all(|signer| one_by_one(signer).is_none()));

        let check = |list: &[_]| assert_eq!(walk(list), list.iter().map(one_by_one).collect::<Vec<_>>());
        for n in 0..=signers.len() {
            check(&signers[..n]);
        }
        for bad in &refused {
            for at in 0..=signers.len() {
                let mut list = signers.clone();
                list.insert(at, bad.clone());
                check(&list);
            }
            check(core::slice::from_ref(bad));
        }
    }
}
