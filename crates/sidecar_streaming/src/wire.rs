//! Payload builders and body parsers for `bins/leanvm-sidecar/PROTOCOL.md`, as in `examples/e2e.rs`.
//!
//! - `claims`: `n u32 | n x (pubkey | message)`
//! - `proof`: `len u64 | len bytes`
//! - `signatures`: `n u32 | n x (pubkey | message | signature)`

use crate::types::*;
use std::collections::HashMap;

/// The signature file: records of `pubkey | message | signature`. Record `i` signs claim `i`.
pub struct SigTable {
    data: Vec<u8>,
    record: usize,
    claim_len: usize,
    index: HashMap<Vec<u8>, u32>,
}

impl SigTable {
    pub fn new(data: Vec<u8>, pub_key: usize, message: usize, sig: usize) -> Result<SigTable, String> {
        let (claim_len, record) = (pub_key + message, pub_key + message + sig);
        if record == 0 || !data.len().is_multiple_of(record) {
            return Err("signature file is not a whole number of records".into());
        }
        let n = data.len() / record;
        if n > MAX_CLAIMS {
            return Err(format!("signature file has {n} records, at most {MAX_CLAIMS} are used"));
        }
        let index = (0..n).map(|i| (data[i * record..i * record + claim_len].to_vec(), i as u32)).collect();
        Ok(SigTable { data, record, claim_len, index })
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.record
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn claim(&self, i: u32) -> &[u8] {
        let at = i as usize * self.record;
        &self.data[at..at + self.claim_len]
    }

    pub fn record(&self, i: u32) -> &[u8] {
        let at = i as usize * self.record;
        &self.data[at..at + self.record]
    }

    pub fn claim_index(&self, claim: &[u8]) -> Option<u32> {
        self.index.get(claim).copied()
    }
}

/// `n u32 | n x record`, the AGGREGATE payload, and the tail of an EXTEND payload.
pub fn signatures_payload(t: &SigTable, ids: &[SigId]) -> Vec<u8> {
    let mut p = (ids.len() as u32).to_le_bytes().to_vec();
    for id in ids {
        p.extend_from_slice(t.record(id.0));
    }
    p
}

/// A child as the server returned it: its claims and its proof, each with its length prefix.
pub struct Child<'a> {
    pub claims: &'a [u8],
    pub proof: &'a [u8],
}

fn children_payload(children: &[Child]) -> Vec<u8> {
    let mut p = (children.len() as u32).to_le_bytes().to_vec();
    for c in children {
        p.extend_from_slice(c.claims);
        p.extend_from_slice(c.proof);
    }
    p
}

/// `n u32 | n x (claims | proof)`.
pub fn recurse_payload(children: &[Child]) -> Vec<u8> {
    children_payload(children)
}

/// `n u32 | n x (claims | proof) | signatures`.
pub fn extend_payload(children: &[Child], t: &SigTable, ids: &[SigId]) -> Vec<u8> {
    let mut p = children_payload(children);
    p.extend(signatures_payload(t, ids));
    p
}

/// `claims | proof`.
pub fn verify_payload(c: &Child) -> Vec<u8> {
    [c.claims, c.proof].concat()
}

/// A response body `proof | claims`, split.
pub struct Body<'a> {
    /// With its length prefix.
    pub proof: &'a [u8],
    /// With its count prefix.
    pub claims: &'a [u8],
}

pub fn split_body(body: &[u8]) -> Result<Body<'_>, String> {
    let short = || "response body too short".to_string();
    let len = u64::from_le_bytes(body.get(..8).ok_or_else(short)?.try_into().unwrap()) as usize;
    let end = 8usize.checked_add(len).filter(|&e| e <= body.len()).ok_or_else(short)?;
    let (proof, claims) = body.split_at(end);
    if claims.len() < 4 {
        return Err(short());
    }
    Ok(Body { proof, claims })
}

/// The claim indices of a `claims` blob, through the signature table.
pub fn claims_coverage(t: &SigTable, claims: &[u8]) -> Result<Coverage, String> {
    let n = u32::from_le_bytes(claims.get(..4).ok_or("claims too short")?.try_into().unwrap()) as usize;
    if claims.len() != 4 + n * t.claim_len {
        return Err("claims blob has the wrong length".into());
    }
    let mut c = Coverage::empty();
    for chunk in claims[4..].chunks_exact(t.claim_len) {
        c.insert(t.claim_index(chunk).ok_or("a claim is not in the signature file")?);
    }
    Ok(c)
}

/// A claims blob for the given claim indices, strictly sorted on the bytes of `pubkey | message`.
pub fn claims_blob(t: &SigTable, cov: &Coverage) -> Vec<u8> {
    let mut claims: Vec<&[u8]> = cov.iter().map(|i| t.claim(i)).collect();
    claims.sort_unstable();
    let mut p = (claims.len() as u32).to_le_bytes().to_vec();
    claims.iter().for_each(|c| p.extend_from_slice(c));
    p
}
