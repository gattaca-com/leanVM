use leanvm::*;

const MSG_0: lean_xmss::Message = [0; lean_xmss::MESSAGE_LEN];
const MSG_1: lean_xmss::Message = [1; lean_xmss::MESSAGE_LEN];
const MSG_2: lean_xmss::Message = [2; lean_xmss::MESSAGE_LEN];

#[test]
fn public_api_end_to_end() {
    setup();

    // 1. Eight leanXMSS signatures over three messages, three of the first, four of the second, one of the third,
    //    by four keys. A key signs each message at its own leaf index, which the statement does not record.
    let keys: Vec<_> = (0..4).map(|_| lean_xmss::key_gen()).collect();
    // The first signature in a bottom tree builds it, so a signer prepares the tree ahead of time. Leaves 7 to 9
    // share one tree.
    for (secret_key, _) in &keys {
        secret_key.prepare(7).unwrap();
    }
    let mut lean_xmss_input = Vec::new();
    for (leaf_index, (message, count)) in (7..).zip([(MSG_0, 3), (MSG_1, 4), (MSG_2, 1)]) {
        for (secret_key, pub_key) in &keys[..count] {
            let signature = lean_xmss::sign(secret_key, &message, leaf_index).unwrap();
            lean_xmss_input.push((*pub_key, message, signature));
        }
    }

    // 2. Three leanSPHINCS signatures, each on its own message
    let mut lean_sphincs_input = Vec::new();
    for signer in 0..3u8 {
        let (secret_key, pub_key) = lean_sphincs::key_gen();
        let message = [signer; lean_sphincs::MESSAGE_LEN];
        let signature = lean_sphincs::sign(&secret_key, &message).unwrap();
        lean_sphincs_input.push((pub_key, message, signature));
    }

    // 3. Two leaves, then a root over both. Both leaves hold claims on the second message; the root's claims are the union.
    let blobs: Vec<_> = (0..lean_da::BLOB_SYMBOLS).map(|i| i as u64).collect();
    let (commitment, _) = lean_da::commit(&blobs).unwrap();
    let left = aggregate(
        &[],
        lean_xmss_input[..4].to_vec(),
        lean_sphincs_input[..1].to_vec(),
        &blobs,
        None,
        2,
    )
    .unwrap();
    let left_claims = left.claims().da_commitment;
    let left = EthereumProof::from_proof_bytes(&left.proof_bytes(), left.claim_tree().clone(), left_claims).unwrap();
    left.verify().unwrap();
    assert_eq!(left.claims().da_commitment, Some(commitment));
    let other_blobs: Vec<_> = blobs.iter().map(|x| x ^ 42).collect();
    let (other_commitment, _) = lean_da::commit(&other_blobs).unwrap();
    let right = aggregate(
        &[],
        lean_xmss_input[4..7].to_vec(),
        lean_sphincs_input[1..].to_vec(),
        &other_blobs,
        None,
        2,
    )
    .unwrap();
    assert_eq!(right.claims().da_commitment, Some(other_commitment));
    // A proof carries one blob matrix: the root passes on one child's, by default the first's.
    let root = aggregate(&[left, right], lean_xmss_input[7..].to_vec(), vec![], &[], None, 2).unwrap();
    assert_eq!(root.claims().num_signature_claims(), 11);
    assert_eq!(root.claims().da_commitment, Some(commitment));

    // 4. Onto the wire, which carries no claims: the receiver supplies the claim tree it expects, and
    //    verifying says exactly these pubkeys signed exactly these messages.
    let bytes = root.proof_bytes();
    let received =
        EthereumProof::from_proof_bytes(&bytes, root.claim_tree().clone(), root.claims().da_commitment).unwrap();
    received.verify().unwrap();
    let on = |message| {
        received
            .claims()
            .lean_xmss
            .iter()
            .filter(|(_, m)| *m == message)
            .count()
    };
    assert_eq!((on(MSG_0), on(MSG_1), on(MSG_2)), (3, 4, 1));
    // A tree the proof does not prove fails, here one signature claim short.
    let mut fewer = root.claim_tree().clone();
    fewer.lean_xmss.pop();
    let wrong = EthereumProof::from_proof_bytes(&bytes, fewer, root.claims().da_commitment).unwrap();
    assert!(matches!(wrong.verify(), Err(AggregateVerifyError::Snark(_))));

    // 5. Selecting what to publish: a node publishes its children's claims whole, since its
    //    commitment hashes each child's. `declare` selects the node's own raw signatures and
    //    the DA root. Here every claim and the DA root stay.
    let keep_root = EthereumClaims {
        da_commitment: Some(commitment),
        ..Default::default()
    };
    let narrowed = aggregate(
        std::slice::from_ref(&received),
        vec![],
        vec![],
        &[],
        Some(&keep_root),
        2,
    )
    .unwrap();
    narrowed.verify().unwrap();
    assert_eq!(narrowed.claims(), received.claims());
    // Naming a child's claim as the node's own is refused: nothing at this node covers it.
    let subset = EthereumClaims {
        lean_xmss: received
            .claims()
            .lean_xmss
            .iter()
            .filter(|(_, m)| *m != MSG_1)
            .cloned()
            .collect(),
        ..Default::default()
    };
    let refused = aggregate(&[received], vec![], vec![], &[], Some(&subset), 2);
    assert!(matches!(refused, Err(AggregationError::NotCovered)));
}
