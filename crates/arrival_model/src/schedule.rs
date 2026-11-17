//! The one file format schedules leave this crate in: a directory with
//! `meta.txt`, `params.txt` (generated sets), `pool.csv` (sets with aggregates) and `slots.csv`.

use std::fmt::Write as _;
use std::fs;

/// The schedule file format. Bump it when a file written by one build cannot be read by another.
pub const FORMAT: u32 = 1;

/// The line `arrivalgen --version` prints, and `streamsim` expects from the binary it runs.
pub fn version_line() -> String {
    format!("arrival_model {} schedule-format {FORMAT}", env!("CARGO_PKG_VERSION"))
}
use std::io;
use std::path::Path;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Vote,
    Aggregate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arrival {
    /// Microseconds from the slot start.
    pub t_us: u64,
    pub kind: Kind,
    pub subnet: u8,
    /// Which aggregator of the subnet sees this vote; for an aggregate, which aggregator made it.
    pub view: u8,
    /// Vote: signer index in the subnet. Aggregate: index into the pool.
    pub id: u32,
    /// Aggregate only: false fails verification.
    pub valid: bool,
}

/// Sorted by `t_us`, then subnet, id and view.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SlotSchedule {
    pub arrivals: Vec<Arrival>,
}

/// What an aggregate proves: a set of signers of its subnet, and its rate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolEntry {
    pub subnet: u8,
    pub id: u32,
    /// Bit `j` is signer `j`.
    pub signers: Vec<u64>,
    pub rate: u8,
    /// Set for entries made from one slot's leaf runs (simulator only).
    pub slot: Option<u32>,
}

impl PoolEntry {
    pub fn signer_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.signers
            .iter()
            .enumerate()
            .flat_map(|(w, &bits)| (0..64u32).filter(move |b| bits >> b & 1 == 1).map(move |b| w as u32 * 64 + b))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Meta {
    pub shape: String,
    pub slots: u32,
    pub slot_us: u64,
    pub att_due_us: u64,
    pub agg_due_us: u64,
    pub ptc_due_us: u64,
    pub deadline_us: u64,
    pub horizon_us: u64,
    pub final_rate: u8,
    pub expected_inputs: u32,
    pub seed: u64,
    pub source: String,
    pub subnet_size: u32,
    pub subnets: u32,
    pub views: u32,
}

impl Meta {
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "shape {}", self.shape);
        let _ = writeln!(s, "slots {}", self.slots);
        let _ = writeln!(s, "slot_us {}", self.slot_us);
        let _ = writeln!(s, "att_due_us {}", self.att_due_us);
        let _ = writeln!(s, "agg_due_us {}", self.agg_due_us);
        let _ = writeln!(s, "ptc_due_us {}", self.ptc_due_us);
        let _ = writeln!(s, "deadline_us {}", self.deadline_us);
        let _ = writeln!(s, "horizon_us {}", self.horizon_us);
        let _ = writeln!(s, "final_rate {}", self.final_rate);
        let _ = writeln!(s, "expected_inputs {}", self.expected_inputs);
        let _ = writeln!(s, "seed {}", self.seed);
        let _ = writeln!(s, "source {}", self.source);
        let _ = writeln!(s, "subnet_size {}", self.subnet_size);
        let _ = writeln!(s, "subnets {}", self.subnets);
        let _ = writeln!(s, "views {}", self.views);
        s
    }

    pub fn parse(text: &str) -> Result<Meta, String> {
        let map = parse_key_values(text);
        let get = |k: &str| map.iter().find(|(key, _)| key == k).map(|(_, v)| v.as_str()).ok_or(format!("meta.txt: missing {k}"));
        let num = |k: &str| -> Result<u64, String> { get(k)?.parse().map_err(|_| format!("meta.txt: bad {k}")) };
        Ok(Meta {
            shape: get("shape")?.to_string(),
            slots: num("slots")? as u32,
            slot_us: num("slot_us")?,
            att_due_us: num("att_due_us")?,
            agg_due_us: num("agg_due_us")?,
            ptc_due_us: num("ptc_due_us")?,
            deadline_us: num("deadline_us")?,
            horizon_us: num("horizon_us")?,
            final_rate: num("final_rate")? as u8,
            expected_inputs: num("expected_inputs")? as u32,
            seed: num("seed")?,
            source: get("source")?.to_string(),
            subnet_size: num("subnet_size")? as u32,
            subnets: num("subnets")? as u32,
            views: num("views")? as u32,
        })
    }
}

/// `key value` lines, `#` comments (also after a value), blank lines ignored.
pub fn parse_key_values(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        match line.split_once(char::is_whitespace) {
            Some((k, v)) => out.push((k.to_string(), v.trim().to_string())),
            None => out.push((line.to_string(), String::new())),
        }
    }
    out
}

pub struct ScheduleSet {
    pub meta: Meta,
    pub params_txt: Option<String>,
    pub pool: Vec<PoolEntry>,
    pub slots: Vec<SlotSchedule>,
}

/// Hex of a signer bitmap, `digits` digits, the last digit holding bits 0 to 3.
pub fn signers_to_hex(signers: &[u64], digits: usize) -> String {
    let mut s = String::with_capacity(digits);
    for k in (0..digits).rev() {
        let word = signers.get(k / 16).copied().unwrap_or(0);
        let nibble = (word >> (4 * (k % 16))) & 0xF;
        s.push(char::from_digit(nibble as u32, 16).unwrap());
    }
    s
}

pub fn signers_from_hex(hex: &str) -> Result<Vec<u64>, String> {
    let digits = hex.len();
    let mut words = vec![0u64; (digits * 4).div_ceil(64)];
    for (i, c) in hex.chars().enumerate() {
        let k = digits - 1 - i;
        let nibble = c.to_digit(16).ok_or("bad hex digit in signers")? as u64;
        words[k / 16] |= nibble << (4 * (k % 16));
    }
    Ok(words)
}

impl ScheduleSet {
    pub fn write(&self, dir: &Path) -> io::Result<()> {
        fs::create_dir_all(dir)?;
        fs::write(dir.join("meta.txt"), self.meta.to_text())?;
        if let Some(p) = &self.params_txt {
            fs::write(dir.join("params.txt"), p)?;
        }
        if !self.pool.is_empty() {
            let digits = (self.meta.subnet_size as usize).div_ceil(4);
            let with_slot = self.pool.iter().any(|e| e.slot.is_some());
            let mut s = String::new();
            s.push_str(if with_slot { "slot,subnet,id,signers_hex,rate\n" } else { "subnet,id,signers_hex,rate\n" });
            for e in &self.pool {
                if with_slot {
                    let _ = write!(s, "{},", e.slot.unwrap_or(0));
                }
                let _ = writeln!(s, "{},{},{},{}", e.subnet, e.id, signers_to_hex(&e.signers, digits), e.rate);
            }
            fs::write(dir.join("pool.csv"), s)?;
        }
        let mut s = String::from("slot,t_us,kind,subnet,view,id,valid\n");
        for (slot, sched) in self.slots.iter().enumerate() {
            for a in &sched.arrivals {
                let kind = if a.kind == Kind::Vote { "vote" } else { "agg" };
                let _ = writeln!(s, "{slot},{},{kind},{},{},{},{}", a.t_us, a.subnet, a.view, a.id, a.valid as u8);
            }
        }
        fs::write(dir.join("slots.csv"), s)
    }

    pub fn read(dir: &Path) -> Result<ScheduleSet, String> {
        let err = |what: &str, e: io::Error| format!("{}: {what}: {e}", dir.display());
        let meta = Meta::parse(&fs::read_to_string(dir.join("meta.txt")).map_err(|e| err("meta.txt", e))?)?;
        let params_txt = fs::read_to_string(dir.join("params.txt")).ok();
        let mut pool = Vec::new();
        if let Ok(text) = fs::read_to_string(dir.join("pool.csv")) {
            let mut lines = text.lines();
            let with_slot = lines.next().is_some_and(|h| h.starts_with("slot,"));
            for line in lines.filter(|l| !l.is_empty()) {
                let f: Vec<&str> = line.split(',').collect();
                let o = usize::from(with_slot);
                if f.len() != 4 + o {
                    return Err(format!("pool.csv: bad line {line}"));
                }
                let bad = |_| format!("pool.csv: bad line {line}");
                pool.push(PoolEntry {
                    slot: if with_slot { Some(f[0].parse().map_err(bad)?) } else { None },
                    subnet: f[o].parse().map_err(bad)?,
                    id: f[o + 1].parse().map_err(bad)?,
                    signers: signers_from_hex(f[o + 2])?,
                    rate: f[o + 3].parse().map_err(bad)?,
                });
            }
        }
        let mut slots = vec![SlotSchedule::default(); meta.slots as usize];
        let text = fs::read_to_string(dir.join("slots.csv")).map_err(|e| err("slots.csv", e))?;
        for line in text.lines().skip(1).filter(|l| !l.is_empty()) {
            let f: Vec<&str> = line.split(',').collect();
            if f.len() != 7 {
                return Err(format!("slots.csv: bad line {line}"));
            }
            let bad = |_| format!("slots.csv: bad line {line}");
            let slot: usize = f[0].parse().map_err(bad)?;
            let kind = match f[2] {
                "vote" => Kind::Vote,
                "agg" => Kind::Aggregate,
                _ => return Err(format!("slots.csv: bad kind in {line}")),
            };
            let arrival = Arrival {
                t_us: f[1].parse().map_err(bad)?,
                kind,
                subnet: f[3].parse().map_err(bad)?,
                view: f[4].parse().map_err(bad)?,
                id: f[5].parse().map_err(bad)?,
                valid: f[6] != "0",
            };
            slots.get_mut(slot).ok_or(format!("slots.csv: slot {slot} beyond meta.slots"))?.arrivals.push(arrival);
        }
        Ok(ScheduleSet { meta, params_txt, pool, slots })
    }
}

/// The key under which arrivals are ordered in a slot.
pub fn sort_arrivals(a: &mut [Arrival]) {
    a.sort_by_key(|x| (x.t_us, x.subnet, x.id, x.view, x.kind == Kind::Aggregate));
}

/// A leaf set with `n` votes per subnet in place of the votes it has: each (slot, subnet, view) keeps its arrival
/// distribution. The `n` offsets are the `(i + 0.5) / n` quantiles of the sorted times, interpolated linearly between
/// order statistics, so the shape of the spread is unchanged; the signer ids are a seeded shuffle of `0..n`.
/// Aggregates are not resampled: a set with candidates is refused.
pub fn resample_votes(set: &ScheduleSet, n: u32, seed: u64) -> Result<ScheduleSet, String> {
    use crate::rng::SplitMix64;
    if set.meta.shape != "leaf" || !set.pool.is_empty() {
        return Err("only a leaf set of votes can be resampled".into());
    }
    let mut out = ScheduleSet { meta: set.meta.clone(), params_txt: set.params_txt.clone(), pool: Vec::new(), slots: Vec::new() };
    out.meta.subnet_size = n;
    out.meta.expected_inputs = n;
    for (slot, sched) in set.slots.iter().enumerate() {
        let mut arrivals = Vec::new();
        for subnet in 0..set.meta.subnets as u8 {
            for view in 0..set.meta.views as u8 {
                let mut times: Vec<u64> = sched.arrivals.iter().filter(|a| a.kind == Kind::Vote && a.subnet == subnet && a.view == view).map(|a| a.t_us).collect();
                if times.is_empty() {
                    continue;
                }
                times.sort_unstable();
                let m = times.len();
                let mut rng = SplitMix64::derive(seed, slot as u64, subnet as u64 * 256 + view as u64);
                let mut ids: Vec<u32> = (0..n).collect();
                for i in (1..ids.len()).rev() {
                    ids.swap(i, rng.below(i as u64 + 1) as usize);
                }
                for (i, &id) in ids.iter().enumerate() {
                    let pos = ((i as f64 + 0.5) / n as f64 * m as f64 - 0.5).clamp(0.0, (m - 1) as f64);
                    let (lo, frac) = (pos.floor() as usize, pos - pos.floor());
                    let hi = (lo + 1).min(m - 1);
                    let t = times[lo] as f64 * (1.0 - frac) + times[hi] as f64 * frac;
                    arrivals.push(Arrival { t_us: t.round() as u64, kind: Kind::Vote, subnet, view, id, valid: true });
                }
            }
        }
        sort_arrivals(&mut arrivals);
        out.slots.push(SlotSchedule { arrivals });
    }
    Ok(out)
}
