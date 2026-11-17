//! arrivalgen --params FILE --slots N --seed S --out DIR [--set key=value]... [--explain]
//! arrivalgen --import FILE.csv --deadline-ms D --out DIR [--pool FILE]

use arrival_model::model::{Params, build, generate};
use arrival_model::schedule::{Kind, Meta, PoolEntry, ScheduleSet, SlotSchedule, signers_from_hex};
use std::path::{Path, PathBuf};

fn usage() -> ! {
    eprintln!("usage: arrivalgen --params FILE --slots N --seed S --out DIR [--set key=value].. [--explain]");
    eprintln!("       arrivalgen --import FILE.csv --deadline-ms D --out DIR [--pool FILE]");
    std::process::exit(2);
}

fn import(csv: &Path, deadline_ms: u64, pool: Option<&Path>) -> Result<ScheduleSet, String> {
    let text = std::fs::read_to_string(csv).map_err(|e| format!("{}: {e}", csv.display()))?;
    let mut slots: Vec<SlotSchedule> = Vec::new();
    for line in text.lines().skip(1).filter(|l| !l.is_empty()) {
        let f: Vec<&str> = line.split(',').collect();
        let bad = || format!("bad line {line}");
        if f.len() != 7 {
            return Err(bad());
        }
        let slot: usize = f[0].parse().map_err(|_| bad())?;
        if slots.len() <= slot {
            slots.resize(slot + 1, SlotSchedule::default());
        }
        slots[slot].arrivals.push(arrival_model::schedule::Arrival {
            t_us: f[1].parse().map_err(|_| bad())?,
            kind: if f[2] == "vote" { Kind::Vote } else { Kind::Aggregate },
            subnet: f[3].parse().map_err(|_| bad())?,
            view: f[4].parse().map_err(|_| bad())?,
            id: f[5].parse().map_err(|_| bad())?,
            valid: f[6] != "0",
        });
    }
    let all = || slots.iter().flat_map(|s| s.arrivals.iter());
    let merge = all().any(|a| a.kind == Kind::Aggregate);
    let subnets = all().map(|a| a.subnet as u32 + 1).max().unwrap_or(1);
    let views = all().map(|a| a.view as u32 + 1).max().unwrap_or(1);
    let mut pool_entries = Vec::new();
    if let Some(p) = pool {
        let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
        for line in text.lines().skip(1).filter(|l| !l.is_empty()) {
            let f: Vec<&str> = line.split(',').collect();
            if f.len() != 4 {
                return Err(format!("pool: bad line {line}"));
            }
            pool_entries.push(PoolEntry {
                subnet: f[0].parse().map_err(|_| "pool subnet")?,
                id: f[1].parse().map_err(|_| "pool id")?,
                signers: signers_from_hex(f[2])?,
                rate: f[3].parse().map_err(|_| "pool rate")?,
                slot: None,
            });
        }
    }
    let meta = Meta {
        shape: if merge { "merge" } else { "leaf" }.into(),
        slots: slots.len() as u32,
        slot_us: 12_000_000,
        att_due_us: 3_000_000,
        agg_due_us: 6_000_000,
        ptc_due_us: 9_000_000,
        deadline_us: deadline_ms * 1000,
        horizon_us: 16_000_000,
        final_rate: if merge { 1 } else { 2 },
        expected_inputs: if merge { subnets } else { 1500 },
        seed: 0,
        source: "recorded".into(),
        subnet_size: 1500,
        subnets,
        views,
    };
    Ok(ScheduleSet { meta, params_txt: None, pool: pool_entries, slots })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut params, mut slots, mut seed, mut out, mut explain) = (None, 100u32, 1u64, None, false);
    let (mut imp, mut deadline, mut pool) = (None, 0u64, None);
    let mut sets: Vec<(String, String)> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].clone();
        let mut value = || {
            i += 1;
            args.get(i).cloned().unwrap_or_else(|| usage())
        };
        match flag.as_str() {
            "--params" => params = Some(PathBuf::from(value())),
            "--slots" => slots = value().parse().unwrap_or_else(|_| usage()),
            "--seed" => seed = value().parse().unwrap_or_else(|_| usage()),
            "--out" => out = Some(PathBuf::from(value())),
            "--explain" => explain = true,
            "--import" => imp = Some(PathBuf::from(value())),
            "--deadline-ms" => deadline = value().parse().unwrap_or_else(|_| usage()),
            "--pool" => pool = Some(PathBuf::from(value())),
            "--set" => {
                let v = value();
                let (k, x) = v.split_once('=').unwrap_or_else(|| usage());
                sets.push((k.to_string(), x.to_string()));
            }
            _ => usage(),
        }
        i += 1;
    }
    let fail = |e: String| -> ! {
        eprintln!("arrivalgen: {e}");
        std::process::exit(1);
    };
    if let Some(csv) = imp {
        let out = out.unwrap_or_else(|| usage());
        let set = import(&csv, deadline, pool.as_deref()).unwrap_or_else(|e| fail(e));
        set.write(&out).unwrap_or_else(|e| fail(e.to_string()));
        return;
    }
    let mut p = Params::from_file(&params.unwrap_or_else(|| usage())).unwrap_or_else(|e| fail(e));
    for (k, v) in &sets {
        p.set(k, v);
    }
    let model = build(&p).unwrap_or_else(|e| fail(e));
    if explain {
        for (name, list) in arrival_model::model::explain(&p).unwrap_or_else(|e| fail(e)) {
            println!("{name}: {}", if list.is_empty() { "(none)".to_string() } else { list.join("; ") });
        }
        return;
    }
    let set = generate(model.as_ref(), &p, slots, seed);
    set.write(&out.unwrap_or_else(|| usage())).unwrap_or_else(|e| fail(e.to_string()));
}
