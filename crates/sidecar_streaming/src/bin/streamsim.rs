//! streamsim fit | sim | e2e | live | bound | report. See `design/SPEC.md` section 8.

use arrival_model::schedule::{Kind, ScheduleSet};
use sidecar_streaming::bound::{clairvoyant_leaf, merge_bounds, universal_leaf};
use sidecar_streaming::cli::*;
use sidecar_streaming::metric::{ROW_HEADER, Row};
use sidecar_streaming::report::{compare, compare_jobs, report};
use sidecar_streaming::runner::{JOBS_HEADER, Job, job_line, run_set};
use sidecar_streaming::sim::SimBackend;
use sidecar_streaming::strategies::{Spec, registry};
use sidecar_streaming::types::*;
use sidecar_streaming::{e2e, fit, live};
use std::io::Write;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some(cmd @ ("fit" | "sim" | "e2e" | "live" | "bound" | "report")) => {
            let mut rest: Vec<String> = args[1..].to_vec();
            if cmd == "report" && rest.first().is_some_and(|f| !f.starts_with("--")) {
                rest.insert(0, "--files".into());
            }
            Args::parse(&rest).and_then(|a| match cmd {
                "fit" => cmd_fit(&a),
                "sim" => cmd_sim(&a),
                "e2e" => e2e::run(&a),
                "live" => live::run(&a),
                "bound" => cmd_bound(&a),
                _ => cmd_report(&a),
            })
        }
        _ => Err("usage: streamsim fit|sim|e2e|live|bound|report [--option value]..  (see design/SPEC.md section 8)".into()),
    };
    if let Err(e) = result {
        eprintln!("streamsim: {e}");
        std::process::exit(1);
    }
}

fn cmd_fit(a: &Args) -> Result<(), String> {
    let csv = std::fs::read_to_string(a.need("csv")?).map_err(|e| e.to_string())?;
    let r = fit::fit(&csv, a.get("backend").unwrap_or("unknown"), a.num("threads", 8)?, a.get("isa").unwrap_or("avx2"), a.get("form"))?;
    std::fs::write(a.need("out")?, &r.model_text).map_err(|e| e.to_string())?;
    let mut text = format!("form {}\nmedian |residual| {:.2} %, max {:.2} %\n", r.form, r.median_res * 100.0, r.max_res * 100.0);
    for l in &r.lines {
        text += l;
        text.push('\n');
    }
    for l in fit::verify_cross_check(&csv)? {
        text += &format!("cross-check {l}\n");
    }
    match a.get("report") {
        Some(p) => std::fs::write(p, &text).map_err(|e| e.to_string())?,
        None => print!("{text}"),
    }
    println!("form {}: median |residual| {:.2} %, max {:.2} %", r.form, r.median_res * 100.0, r.max_res * 100.0);
    Ok(())
}

fn caps_of(a: &Args) -> Result<Caps, String> {
    Ok(Caps { lanes: a.num("lanes", 1)?, abort: a.flag("abort"), verify_lane: a.flag("verify-lane"), ..Caps::default() })
}

fn cmd_sim(a: &Args) -> Result<(), String> {
    let set_dir = a.need("set")?;
    let set = ScheduleSet::read(Path::new(set_dir))?;
    let (model, backend_name) = load(Path::new(a.need("model")?))?;
    let shape = if set.meta.shape == "merge" { Shape::Merge } else { Shape::Leaf };
    let deadlines = match a.get("deadline-ms") {
        Some(t) => parse_deadlines(t)?,
        None => vec![set.meta.deadline_us],
    };
    let slots = parse_range(a.get("slots").unwrap_or("100000"), set.meta.slots)?;
    let forecast = load_forecast(a)?;
    let policy = a.get("verify-policy").unwrap_or("all_idle");
    let names: Vec<String> = match a.need("strategy")? {
        "all" => registry(shape).iter().map(|s| s.to_string()).collect(),
        s => vec![s.to_string()],
    };
    let mut backend = SimBackend::new(model, caps_of(a)?, a.num("seed", 1)?, !a.flag("no-noise"));
    let set_name = Path::new(set_dir).file_name().map_or(set_dir.to_string(), |s| s.to_string_lossy().into());
    let mut rows: Vec<Row> = Vec::new();
    let mut jobs = vec![JOBS_HEADER.to_string()];
    let want_jobs = a.get("jobs").is_some();
    for name in names {
        let mut spec = Spec::parse(&name)?;
        if spec.shape() != shape {
            return Err(format!("strategy {} is for {:?} sets", spec.name, spec.shape()));
        }
        if shape == Shape::Merge {
            spec.default_param("verify", policy);
        }
        if forecast.is_none() && (spec.name == "plan_cover" || spec.name == "plan_merge") && a.need("strategy")? == "all" {
            eprintln!("skipping {}: no --train set", spec.name);
            continue;
        }
        let job = Job { spec, forecast: forecast.clone() };
        let results = run_set(&mut backend, &set, &set_name, &backend_name, &job, &deadlines, slots.clone(), |_| {})?;
        for r in results {
            if want_jobs {
                jobs.extend(r.record.jobs.iter().map(|j| job_line(r.row.slot, j)));
            }
            rows.push(r.row);
        }
    }
    let mut text = format!("{ROW_HEADER}\n");
    for r in &rows {
        text += &r.to_csv();
        text.push('\n');
    }
    std::fs::write(a.need("out")?, text).map_err(|e| e.to_string())?;
    if let Some(p) = a.get("jobs") {
        std::fs::write(p, jobs.join("\n") + "\n").map_err(|e| e.to_string())?;
    }
    print!("{}", report(&rows, a.get("baseline"), false));
    Ok(())
}

fn cmd_bound(a: &Args) -> Result<(), String> {
    let set = ScheduleSet::read(Path::new(a.need("set")?))?;
    let (model, _) = load(Path::new(a.need("model")?))?;
    let deadlines = match a.get("deadline-ms") {
        Some(t) => parse_deadlines(t)?,
        None => vec![set.meta.deadline_us],
    };
    let slots = parse_range(a.get("slots").unwrap_or("100000"), set.meta.slots)?;
    let grid = a.num("grid", 10usize)?;
    let cheap = a.num("cheap", 1u8)?;
    let mut out = String::from("slot,deadline_us,arrived,clairvoyant,universal,clairvoyant_coverage,universal_coverage\n");
    for &d in &deadlines {
        for slot in slots.clone() {
            for (id, subnet, view) in sidecar_streaming::runner::runs_of(&set, slot) {
                let sched = &set.slots[slot as usize];
                let (clair, univ, arrived) = if set.meta.shape == "merge" {
                    let covs: Vec<(u64, u8, Coverage, bool)> = sched
                        .arrivals
                        .iter()
                        .filter(|x| x.kind == Kind::Aggregate)
                        .map(|x| {
                            let e = &set.pool[x.id as usize];
                            let mut c = Coverage::empty();
                            e.signer_ids().for_each(|s| c.insert(SUBNET_SIZE * e.subnet as u32 + s));
                            (x.t_us, e.subnet, c, x.valid)
                        })
                        .collect();
                    let cands: Vec<(u64, u8, &Coverage, bool)> = covs.iter().map(|c| (c.0, c.1, &c.2, c.3)).collect();
                    let mut by_d = Coverage::empty();
                    covs.iter().filter(|c| c.3 && c.0 <= d).for_each(|c| by_d.union_with(&c.2));
                    let (c, u) = merge_bounds(&cands, model.as_ref(), d, set.meta.final_rate, 16);
                    (c, u, by_d.len())
                } else {
                    let arr: Vec<u64> = sched.arrivals.iter().filter(|x| x.kind == Kind::Vote && x.subnet == subnet && x.view == view).map(|x| x.t_us).collect();
                    let arrived = arr.partition_point(|&t| t <= d) as u32;
                    (clairvoyant_leaf(&arr, model.as_ref(), d, set.meta.final_rate, cheap, grid), universal_leaf(&arr, model.as_ref(), d, set.meta.final_rate, cheap), arrived)
                };
                let cov = |x: u32| if arrived == 0 { 1.0 } else { x as f64 / arrived as f64 };
                out += &format!("{id},{d},{arrived},{clair},{univ},{:.6},{:.6}\n", cov(clair), cov(univ));
            }
        }
    }
    std::fs::write(a.need("out")?, out).map_err(|e| e.to_string())?;
    Ok(())
}

fn read_rows(path: &str) -> Result<Vec<Row>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    text.lines().skip(1).filter(|l| !l.is_empty()).map(Row::from_csv).collect()
}

fn cmd_report(a: &Args) -> Result<(), String> {
    if let Some(files) = a.get("compare") {
        let f: Vec<&str> = files.split_whitespace().collect();
        if f.len() != 2 {
            return Err("--compare takes two files: sim.csv live.csv".into());
        }
        print!("{}", compare(&read_rows(f[0])?, &read_rows(f[1])?));
        if let Some(j) = a.get("jobs-live") {
            let text = std::fs::read_to_string(j).map_err(|e| e.to_string())?;
            let lines: Vec<String> = text.lines().skip(1).map(str::to_string).collect();
            print!("\n{}", compare_jobs(&lines));
        }
        return Ok(());
    }
    let files = a.need("files")?;
    let mut rows = Vec::new();
    for f in files.split_whitespace() {
        rows.extend(read_rows(f)?);
    }
    let text = report(&rows, a.get("baseline"), a.flag("by-arrival"));
    match a.get("out") {
        Some(p) => std::fs::write(p, text).map_err(|e| e.to_string())?,
        None => {
            let _ = std::io::stdout().write_all(text.as_bytes());
        }
    }
    Ok(())
}
