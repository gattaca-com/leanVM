use lean_vm::*;

const MSG_0: lean_xmss::Message = [0; lean_xmss::MESSAGE_LEN];
const MSG_1: lean_xmss::Message = [1; lean_xmss::MESSAGE_LEN];
const MSG_2: lean_xmss::Message = [2; lean_xmss::MESSAGE_LEN];

#[test]
fn public_api_end_to_end() {
    setup(0);

    // 1. Eight leanXMSS signatures over three messages, three of the first, four of the second, one of the third,
    //    by four keys. A key signs each message at its own leaf index, which the statement does not record.
    let keys: Vec<_> = (0..4u8)
        .map(|k| lean_xmss::key_gen(&[k; lean_xmss::SECRET_KEY_LEN]))
        .collect();
    // The first signature in a bottom tree builds it, so a signer prepares the tree ahead of time. Leaves 7 to 9
    // share one tree.
    for (secret_key, _) in &keys {
        secret_key.set_bottom_tree_cache_size(4);
        lean_xmss::prepare(secret_key, 7);
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

    // 4. Onto the wire, which carries only the proof: the receiver rebuilds the claims it expects, and
    //    verifying says exactly these pubkeys signed exactly these messages.
    let received = AggregateProof::from_bytes(&proof.to_bytes()).unwrap();
    assert!(received.verify(&claims));
    let on = |message| claims.lean_xmss.iter().filter(|(_, m)| *m == message).count();
    assert_eq!((on(MSG_0), on(MSG_1), on(MSG_2)), (3, 4, 1));
    // Claims the proof does not prove fail, here one signature claim short.
    let mut fewer = claims.clone();
    fewer.lean_xmss.pop();
    assert!(!received.verify(&fewer));

    // 5. Removing some signatures from the aggregate: `OutputClaims::Only` is what we keep. Here every claim on MSG_1 goes,
    //    and one leanSPHINCS claim.
    let retained = AggregateClaims {
        lean_xmss: claims.lean_xmss.iter().filter(|(_, m)| *m != MSG_1).cloned().collect(),
        lean_sphincs: claims.lean_sphincs[1..].to_vec(),
        da_commitment: Some(commitment),
    };
    let narrowed = aggregate(
        &[(claims, received)],
        vec![],
        vec![],
        None,
        OutputClaims::Only(&retained),
        2,
    )
    .unwrap();
    assert!(narrowed.1.verify(&narrowed.0));
    assert_eq!(narrowed.0, retained);
    assert_eq!(narrowed.0.num_signature_claims(), 11 - 4 - 1);

    // And the blob claim.
    let without_blobs = AggregateClaims {
        da_commitment: None,
        ..retained
    };
    let dropped = aggregate(&[narrowed], vec![], vec![], None, OutputClaims::Only(&without_blobs), 2).unwrap();
    assert!(dropped.1.verify(&dropped.0));
    assert_eq!(dropped.0, without_blobs);
}

#[test]
fn vrf_reveal() {
    let key = lean_vrf::VrfKey::new([7; 32], 100..200).unwrap();
    assert!(lean_vrf::verify_vrf_proof(&key.root(), 150, &key.proof(150).unwrap()));
}

/// Key generation and signing on the prover's pool, by threads signing at once, some of them while the pool is busy
/// with another, give the keys and signatures the calling thread gives.
#[test]
fn signing_on_the_pool_matches_the_calling_thread() {
    setup(0);
    let (secret_key, pub_key) = lean_xmss::key_gen(&[9; lean_xmss::SECRET_KEY_LEN]);
    let (reference, reference_pub_key) = ::lean_xmss::key_gen(&[9; lean_xmss::SECRET_KEY_LEN]);
    assert_eq!(pub_key, reference_pub_key);
    assert_eq!(secret_key.cache_to_bytes(), reference.cache_to_bytes());
    let sphincs_key = [9; lean_sphincs::SECRET_KEY_LEN];
    let sphincs_pub_key = lean_sphincs::key_gen(&sphincs_key);
    assert_eq!(sphincs_pub_key, ::lean_sphincs::key_gen(&sphincs_key));
    let signature = lean_sphincs::sign(&sphincs_key, &MSG_1);
    assert_eq!(signature, ::lean_sphincs::sign(&sphincs_key, &MSG_1));
    assert!(lean_sphincs::verify(&sphincs_pub_key, &MSG_1, &signature));
    let leaves: [lean_xmss::LeafIndex; 4] = [5, 6, 300, (1 << 16) + 5];
    let secret_key = &secret_key;
    let on_pool: Vec<_> = std::thread::scope(|s| {
        let handles: Vec<_> = leaves
            .iter()
            .map(|&leaf_index| s.spawn(move || lean_xmss::sign(secret_key, &MSG_2, leaf_index)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (&leaf_index, signature) in leaves.iter().zip(&on_pool) {
        assert!(lean_xmss::verify(&pub_key, &MSG_2, signature));
        assert_eq!(*signature, ::lean_xmss::sign(secret_key, &MSG_2, leaf_index));
    }
}
