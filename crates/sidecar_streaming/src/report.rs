//! `streamsim report`: the SPEC 4.3 table per deadline, `D_99`, paired differences, and sim against live.

use crate::metric::*;
use std::collections::BTreeMap;
use std::fmt::Write;

type Key = (String, String, u64);

fn pct(x: f64) -> String {
    format!("{x:.3}")
}

fn ms(x: f64) -> String {
    if x.is_nan() { "-".into() } else { format!("{x:.0}") }
}

type Groups<'a> = BTreeMap<(String, String), BTreeMap<u64, BTreeMap<String, Vec<&'a Row>>>>;

pub fn report(rows: &[Row], baseline: Option<&str>, by_arrival: bool) -> String {
    let mut out = String::new();
    // (set, backend) -> deadline -> strategy -> rows by slot.
    let mut groups: Groups = BTreeMap::new();
    for r in rows {
        groups.entry((r.set.clone(), r.backend.clone())).or_default().entry(r.deadline_us).or_default().entry(r.strategy.clone()).or_default().push(r);
    }
    for v in groups.values_mut().flat_map(|d| d.values_mut()).flat_map(|s| s.values_mut()) {
        v.sort_by_key(|r| r.slot);
    }
    if by_arrival {
        let _ = writeln!(out, "| set | backend | D ms | best strategy | mean coverage | margin to second | p10 |\n| --- | --- | --- | --- | --- | --- | --- |");
        for ((set, backend), by_d) in &groups {
            for (d, by_s) in by_d {
                let mut ranked: Vec<(f64, f64, &String)> = by_s.iter().map(|(s, v)| (summarize(v).mean, summarize(v).p10, s)).collect();
                ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
                let margin = if ranked.len() > 1 { ranked[0].0 - ranked[1].0 } else { 0.0 };
                let _ = writeln!(out, "| {set} | {backend} | {} | {} | {} | {} | {} |", d / 1000, ranked[0].2, pct(ranked[0].0), pct(margin), pct(ranked[0].1));
            }
        }
        return out;
    }
    for ((set, backend), by_d) in &groups {
        let _ = writeln!(out, "## {set}, {backend}\n");
        let mut d99s: BTreeMap<&String, Vec<(u64, f64)>> = BTreeMap::new();
        for (d, by_s) in by_d {
            let _ = writeln!(out, "### D = {} ms\n", d / 1000);
            let _ = write!(out, "| strategy | n | empty | p50 | p10 | mean | sent p50 | sent p10 | stale ms | slack ms | jobs | busy ms | wasted ms | verify ms |");
            let _ = writeln!(out, "{}", if baseline.is_some() { " d mean vs baseline |" } else { "" });
            let _ = writeln!(out, "|{}", "---|".repeat(14 + usize::from(baseline.is_some())));
            for (s, v) in by_s {
                let sm = summarize(v);
                d99s.entry(s).or_default().push((*d, sm.p10));
                let _ = write!(
                    out,
                    "| {s} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.1} | {} | {} | {} |",
                    sm.n,
                    sm.empty,
                    pct(sm.p50),
                    pct(sm.p10),
                    pct(sm.mean),
                    pct(sm.sent_p50),
                    pct(sm.sent_p10),
                    ms(sm.staleness_ms),
                    ms(sm.slack_ms),
                    sm.jobs,
                    ms(sm.busy_ms),
                    ms(sm.wasted_ms),
                    ms(sm.verify_ms)
                );
                if let Some(b) = baseline {
                    match by_s.iter().find(|(name, _)| name.starts_with(b)) {
                        Some((_, base)) if base.len() == v.len() => {
                            let (m, se) = paired_diff(&v.iter().map(|r| r.m.coverage).collect::<Vec<_>>(), &base.iter().map(|r| r.m.coverage).collect::<Vec<_>>());
                            let _ = write!(out, " {m:+.4} +- {se:.4} |");
                        }
                        _ => {
                            let _ = write!(out, " - |");
                        }
                    }
                }
                let _ = writeln!(out);
            }
            let _ = writeln!(out);
        }
        if by_d.len() > 1 {
            let _ = writeln!(out, "### D_99 (smallest swept D with p10 coverage >= 0.99)\n\n| strategy | D_99 ms |\n| --- | --- |");
            for (s, pts) in &d99s {
                let pts: Vec<(u64, f64)> = pts.clone();
                let _ = writeln!(out, "| {s} | {} |", d99(&pts).map_or("-".into(), |d| (d / 1000).to_string()));
            }
            let _ = writeln!(out);
        }
    }
    out
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Sim against live, same schedule, strategy and `D`.
pub fn compare(sim: &[Row], live: &[Row]) -> String {
    let index: BTreeMap<(Key, u32), &Row> = sim.iter().map(|r| (((r.set.clone(), r.strategy.clone(), r.deadline_us), r.slot), r)).collect();
    let (mut dcov, mut dt) = (Vec::new(), Vec::new());
    for l in live {
        if let Some(s) = index.get(&((l.set.clone(), l.strategy.clone(), l.deadline_us), l.slot)) {
            if !(l.m.coverage.is_nan() || s.m.coverage.is_nan()) {
                dcov.push((l.m.coverage - s.m.coverage).abs());
            }
            if let (Some(a), Some(b)) = (l.m.published_us, s.m.published_us) {
                dt.push((a as f64 - b as f64).abs() / b.max(1) as f64);
            }
        }
    }
    let (mc, mt) = (median(dcov.clone()), median(dt.clone()));
    format!(
        "matched slots: {}\nmedian |coverage live - sim|: {:.4} (accept <= 0.01): {}\nmedian |t_P live - sim| / sim: {:.4} (accept <= 0.05): {}\n",
        dcov.len(),
        mc,
        if mc <= 0.01 { "ok" } else { "NOT ok" },
        mt,
        if mt <= 0.05 { "ok" } else { "NOT ok" }
    )
}

/// Per job: measured over predicted duration, by op and size bucket. `lines` are job file lines.
pub fn compare_jobs(lines: &[String]) -> String {
    let mut buckets: BTreeMap<(String, &'static str), Vec<f64>> = BTreeMap::new();
    for l in lines {
        let f: Vec<&str> = l.split(',').collect();
        if f.len() < 12 {
            continue;
        }
        let n = |i: usize| f[i].parse::<f64>().unwrap_or(0.0);
        let (start, end, pred) = (n(8), n(9), n(10));
        if pred <= 0.0 {
            continue;
        }
        let size = n(5) + n(6);
        let bucket = if size < 150.0 { "<150" } else if size < 700.0 { "150-700" } else if size < 1600.0 { "700-1600" } else { ">=1600" };
        buckets.entry((f[3].to_string(), bucket)).or_default().push((end - start) / pred);
    }
    let mut out = String::from("| op | size | n | median measured / predicted |\n| --- | --- | --- | --- |\n");
    for ((op, b), v) in buckets {
        let _ = writeln!(out, "| {op} | {b} | {} | {:.3} |", v.len(), median(v.clone()));
    }
    out
}
