use lean_vm::*;

const MSG_0: lean_xmss::Message = [0; lean_xmss::MESSAGE_LEN];
const MSG_1: lean_xmss::Message = [1; lean_xmss::MESSAGE_LEN];
const MSG_2: lean_xmss::Message = [2; lean_xmss::MESSAGE_LEN];

#[test]
#[ignore = "the recursion guest verifies one hash table's reduction, not the two a node's children prove"]
fn public_api_end_to_end() {
    setup();

    // 1. Eight leanXMSS signatures over three messages, three of the first, four of the second, one of the third,
    //    by four keys. A key signs each message at its own leaf index, which the statement does not record.
    let keys: Vec<_> = (0..4u8)
        .map(|k| lean_xmss::key_gen(&[k; lean_xmss::SECRET_KEY_LEN]))
        .collect();
    // The first signature in a bottom tree builds it, so a signer prepares the tree ahead of time. Leaves 7 to 9
    // share one tree.
    for (secret_key, _) in &keys {
        secret_key.prepare(7);
    }
    let mut lean_xmss_input = Vec::new();
    for (leaf_index, (message, count)) in (7..).zip([(MSG_0, 3), (MSG_1, 4), (MSG_2, 1)]) {
        for (secret_key, pub_key) in &keys[..count] {
            let signature = lean_xmss::sign(secret_key, &message, leaf_index);
            lean_xmss_input.push((*pub_key, message, signature));
        }
    }

    // 2. Three leanSPHINCS signatures, each on its own message
    let mut lean_sphincs_input = Vec::new();
    for signer in 0..3u8 {
        let secret_key = [signer; lean_sphincs::SECRET_KEY_LEN];
        let pub_key = lean_sphincs::key_gen(&secret_key);
        let message = [signer; lean_sphincs::MESSAGE_LEN];
        let signature = lean_sphincs::sign(&secret_key, &message);
        lean_sphincs_input.push((pub_key, message, signature));
    }

    // 3. Two leaves, then a root over both. Both leaves hold claims on the second message; the root's claims are the union.
    let blobs: Vec<lean_da::Blob> = vec![std::array::from_fn(|i| i as u128)];
    let matrix = lean_da::BlobMatrix::build(&blobs).unwrap();
    let commitment = matrix.commitment();
    let left = aggregate(
        &[],
        lean_xmss_input[..4].to_vec(),
        lean_sphincs_input[..1].to_vec(),
        Some(&matrix),
        OutputClaims::All,
        2,
    )
    .unwrap();
    assert!(left.1.verify(&left.0));
    assert_eq!(left.0.da_commitment, Some(commitment));
    let other_blobs: Vec<lean_da::Blob> = blobs.iter().map(|blob| blob.map(|x| x ^ 42)).collect();
    let other_matrix = lean_da::BlobMatrix::build(&other_blobs).unwrap();
    let other_commitment = other_matrix.commitment();
    let right = aggregate(
        &[],
        lean_xmss_input[4..7].to_vec(),
        lean_sphincs_input[1..].to_vec(),
        Some(&other_matrix),
        OutputClaims::All,
        2,
    )
    .unwrap();
    assert_eq!(right.0.da_commitment, Some(other_commitment));
    // A proof carries one blob matrix: the root passes on one child's, by default the first's.
    let (claims, proof) = aggregate(
        &[left, right],
        lean_xmss_input[7..].to_vec(),
        vec![],
        None,
        OutputClaims::All,
        2,
    )
    .unwrap();
    assert_eq!(claims.num_signature_claims(), 11);
    assert_eq!(claims.da_commitment, Some(commitment));

    // 4. Onto the wire, which carries only the proof: the receiver rebuilds the claim tree it expects, and
    //    verifying says exactly these pubkeys signed exactly these messages.
    let received = AggregateProof::from_bytes(&proof.to_bytes()).unwrap();
    assert!(received.verify(&claims));
    let (flat_xmss, _) = claims.flatten();
    let on = |message| flat_xmss.iter().filter(|(_, m)| *m == message).count();
    assert_eq!((on(MSG_0), on(MSG_1), on(MSG_2)), (3, 4, 1));
    // A tree the proof does not prove fails, here one signature claim short.
    let mut fewer = claims.clone();
    fewer.tree.lean_xmss.pop();
    assert!(!received.verify(&fewer));
    // So does one a level deeper than the bound.
    let mut deep = ClaimTree::default();
    for _ in 0..MAX_TREE_DEPTH {
        deep = ClaimTree {
            children: vec![deep],
            ..Default::default()
        };
    }
    assert!(!received.verify(&AggregateClaims {
        tree: deep,
        da_commitment: Some(commitment),
    }));

    // 5. Selecting what to publish: a node publishes its children's claims whole, since its commitment hashes
    //    each child's. `OutputClaims::Only` selects the node's own raw signatures and the DA root. Here every
    //    claim and the DA root stay.
    let child = (claims, received);
    let keep_root = AggregateClaims {
        da_commitment: Some(commitment),
        ..Default::default()
    };
    let narrowed = aggregate(
        std::slice::from_ref(&child),
        vec![],
        vec![],
        None,
        OutputClaims::Only(&keep_root),
        2,
    )
    .unwrap();
    assert!(narrowed.1.verify(&narrowed.0));
    assert_eq!(narrowed.0.flatten(), child.0.flatten());
    // Naming a child's claim as the node's own is refused: nothing at this node covers it.
    let subset = AggregateClaims {
        tree: ClaimTree {
            lean_xmss: flat_xmss.iter().filter(|(_, m)| *m != MSG_1).cloned().collect(),
            ..Default::default()
        },
        da_commitment: Some(commitment),
    };
    let refused = aggregate(
        std::slice::from_ref(&child),
        vec![],
        vec![],
        None,
        OutputClaims::Only(&subset),
        2,
    );
    assert!(matches!(refused, Err(AggregateError::InvalidInput(_))));
}
