//! Aggregates derived from leaf runs: one candidate per (slot, subnet, view), at the time its
//! aggregator published, plus a network delay.
//!
//! The file is CSV: `slot,subnet,view,published_us,rate,signers_hex`. `streamsim` writes it from its leaf runs.

use crate::schedule::signers_from_hex;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct ComposeEntry {
    pub slot: u32,
    pub subnet: u8,
    pub view: u8,
    pub published_us: u64,
    pub rate: u8,
    pub signers: Vec<u64>,
}

pub fn read_compose_file(path: &Path) -> Result<Vec<ComposeEntry>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for line in text.lines().skip(1).filter(|l| !l.is_empty()) {
        let f: Vec<&str> = line.split(',').collect();
        let bad = || format!("{}: bad line {line}", path.display());
        if f.len() != 6 {
            return Err(bad());
        }
        out.push(ComposeEntry {
            slot: f[0].parse().map_err(|_| bad())?,
            subnet: f[1].parse().map_err(|_| bad())?,
            view: f[2].parse().map_err(|_| bad())?,
            published_us: f[3].parse().map_err(|_| bad())?,
            rate: f[4].parse().map_err(|_| bad())?,
            signers: signers_from_hex(f[5])?,
        });
    }
    Ok(out)
}
