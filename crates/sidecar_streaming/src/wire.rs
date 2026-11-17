//! Payload builders and body parsers for `bins/leanvm-sidecar/PROTOCOL.md`, as in `examples/e2e.rs`.
//!
//! - `claims`: `n u32 | n x (pubkey | message)`
//! - `proof`: `len u64 | len bytes`
//! - `signatures`: `n u32 | n x (pubkey | message | signature)`

use crate::types::*;
use std::collections::HashMap;

/// The signature file: records of `pubkey | message | signature`. A subnet of `subnet_size` signers owns `subnet_size`
/// consecutive records; claim `SUBNET_SIZE * s + j` (the stride of the harness) is record `subnet_size * s + j`.
/// With the file of 24000 records, `subnet_size` can be at most 1500 for 16 subnets.
pub struct SigTable {
    data: Vec<u8>,
    record: usize,
    claim_len: usize,
    subnet_size: u32,
    index: HashMap<Vec<u8>, u32>,
}

impl SigTable {
    pub fn new(data: Vec<u8>, pub_key: usize, message: usize, sig: usize, subnet_size: u32) -> Result<SigTable, String> {
        let (claim_len, record) = (pub_key + message, pub_key + message + sig);
        if record == 0 || !data.len().is_multiple_of(record) {
            return Err("signature file is not a whole number of records".into());
        }
        let mut t = SigTable { data, record, claim_len, subnet_size, index: HashMap::new() };
        t.rebuild()?;
        Ok(t)
    }

    fn rebuild(&mut self) -> Result<(), String> {
        if self.subnet_size == 0 || self.subnet_size > SUBNET_SIZE {
            return Err(format!("subnet size {} is outside 1..={SUBNET_SIZE}", self.subnet_size));
        }
        self.index = (0..self.len())
            .map(|rec| (self.data[rec * self.record..rec * self.record + self.claim_len].to_vec(), self.claim_of(rec as u32)))
            .collect();
        Ok(())
    }

    /// Use `size` records per subnet. A set whose subnets need more records than the file has is refused.
    pub fn set_subnet_size(&mut self, size: u32, subnets: u32) -> Result<(), String> {
        let needed = size as usize * subnets as usize;
        if needed > self.len() {
            return Err(format!("{subnets} subnets of {size} signers need {needed} signature records; the file has {}", self.len()));
        }
        if self.subnet_size == size {
            return Ok(());
        }
        self.subnet_size = size;
        self.rebuild()
    }

    fn rec_of(&self, claim: u32) -> usize {
        ((claim / SUBNET_SIZE) * self.subnet_size + claim % SUBNET_SIZE) as usize
    }

    fn claim_of(&self, rec: u32) -> u32 {
        (rec / self.subnet_size) * SUBNET_SIZE + rec % self.subnet_size
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.record
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn claim(&self, i: u32) -> &[u8] {
        let at = self.rec_of(i) * self.record;
        &self.data[at..at + self.claim_len]
    }

    pub fn record(&self, i: u32) -> &[u8] {
        let at = self.rec_of(i) * self.record;
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
