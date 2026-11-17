use sidecar_streaming::types::*;
use sidecar_streaming::wire::*;

/// A signature table of `n` records: 4 bytes of public key, 2 of message, 3 of signature.
fn table(n: u32) -> SigTable {
    let mut data = Vec::new();
    for i in 0..n {
        data.extend(i.to_be_bytes());
        data.extend([0xAA, i as u8]);
        data.extend([1, 2, 3]);
    }
    SigTable::new(data, 4, 2, 3, n).unwrap()
}

#[test]
fn payloads_follow_the_protocol() {
    let t = table(10);
    assert_eq!(t.len(), 10);
    let sigs = signatures_payload(&t, &[SigId(3), SigId(1)]);
    assert_eq!(&sigs[..4], &2u32.to_le_bytes());
    assert_eq!(&sigs[4..13], t.record(3));
    assert_eq!(&sigs[13..], t.record(1));
    // A child is `claims | proof`, each with its prefix, as the server returned them.
    let claims = [1u8, 0, 0, 0, 0, 0, 0, 3, 0xAA, 3];
    let proof = [2u8, 0, 0, 0, 0, 0, 0, 0, 7, 8];
    let child = Child { claims: &claims, proof: &proof };
    let rec = recurse_payload(&[Child { claims: &claims, proof: &proof }, Child { claims: &claims, proof: &proof }]);
    assert_eq!(&rec[..4], &2u32.to_le_bytes());
    assert_eq!(rec.len(), 4 + 2 * (claims.len() + proof.len()));
    let ext = extend_payload(&[Child { claims: &claims, proof: &proof }], &t, &[SigId(5)]);
    assert_eq!(ext.len(), 4 + claims.len() + proof.len() + 4 + 9);
    assert_eq!(&ext[ext.len() - 9..], t.record(5));
    assert_eq!(verify_payload(&child), [claims.as_slice(), proof.as_slice()].concat());
}

#[test]
fn bodies_split_and_claims_map_back() {
    let t = table(10);
    let mut cov = Coverage::empty();
    [7u32, 2, 9].iter().for_each(|&c| cov.insert(c));
    let claims = claims_blob(&t, &cov);
    // Strictly sorted on the bytes.
    let rest = &claims[4..];
    let chunks: Vec<&[u8]> = rest.chunks(6).collect();
    assert!(chunks.windows(2).all(|w| w[0] < w[1]));
    let proof = [vec![3u8, 0, 0, 0, 0, 0, 0, 0], vec![9, 9, 9]].concat();
    let body = [proof.clone(), claims.clone()].concat();
    let b = split_body(&body).unwrap();
    assert_eq!((b.proof, b.claims), (proof.as_slice(), claims.as_slice()));
    assert_eq!(claims_coverage(&t, b.claims).unwrap(), cov);
    assert!(split_body(&body[..5]).is_err());
    assert!(split_body(&proof).is_err());
    assert!(claims_coverage(&t, &claims[..claims.len() - 1]).is_err());
}
