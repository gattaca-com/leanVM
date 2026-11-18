//! `streamsim e2e` (SPEC 4.5, step 2): leaf runs for every subnet and aggregator view, the proposer's
//! candidates composed from their publish times, then the merge run. `end_to_end` is the claims in the
//! root by `D_merge` over the votes sent in all subnets, for each `D_leaf` of a sweep.
//!
//! The composed candidate schedule is made by the `arrivalgen` binary (the arrival model stays its own
//! program): this module writes a compose file and a parameter file, runs it, and reads the set back.

use crate::backend::Backend;
use crate::calib;
use crate::cli::*;
use crate::driver::{RunSpec, run_slot, run_slot_with};
use crate::forecast::Forecast;
use crate::metric::*;
use crate::runner::runs_of;
use crate::sim::SimBackend;
use crate::strategies::{Spec, merge::Policy};
use crate::types::*;
use arrival_model::schedule::{Kind, ScheduleSet, signers_to_hex};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::rc::Rc;

fn backend(a: &Args, path: &Path, caps: Caps, seed: u64, noise: bool) -> Result<(SimBackend, String), String> {
    let (model, name) = load_args(a, path)?;
    Ok((SimBackend::new(model, caps, seed, noise), name))
}

fn arrivalgen_path(a: &Args) -> Result<PathBuf, String> {
    match a.get("arrivalgen") {
        Some(p) => Ok(p.into()),
        None => {
            let exe = std::env::current_exe().map_err(|e| e.to_string())?;
            Ok(exe.with_file_name("arrivalgen"))
        }
    }
}

/// Refuse an `arrivalgen` that was not built from the same arrival crate as this binary.
pub fn check_arrivalgen(path: &Path) -> Result<(), String> {
    let out = Command::new(path).arg("--version").output().map_err(|e| format!("{}: {e} (build arrivalgen, or pass --arrivalgen)", path.display()))?;
    let got = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let want = arrival_model::schedule::version_line();
    if got != want {
        return Err(format!("{} is a different build: it says `{got}`, this binary expects `{want}`", path.display()));
    }
    Ok(())
}

struct Stats {
    end_to_end: Vec<f64>,
    merge_coverage: Vec<f64>,
    leaf_coverage: Vec<f64>,
    published: usize,
    runs: usize,
}

pub fn run(a: &Args) -> Result<(), String> {
    let leaf_dir = a.need("leaf-set")?;
    let leaf = ScheduleSet::read(Path::new(leaf_dir))?;
    if leaf.meta.shape != "leaf" {
        return Err("--leaf-set must be a leaf set".into());
    }
    let model_path = Path::new(a.need("model")?);
    let merge_model_path = a.get("merge-model").map_or(model_path, Path::new);
    let d_leaf = parse_deadlines(a.need("d-leaf-ms")?)?;
    let d_merge = *parse_deadlines(a.need("d-merge-ms")?)?.first().ok_or("--d-merge-ms is empty")?;
    let slots = parse_range(a.get("slots").unwrap_or("100000"), leaf.meta.slots)?;
    let seed = a.num("seed", 1u64)?;
    let noise = !a.flag("no-noise");
    let mut leaf_spec = Spec::parse(a.need("leaf-strategy")?)?;
    let mut merge_spec = Spec::parse(a.need("merge-strategy")?)?;
    if leaf_spec.shape() != Shape::Leaf || merge_spec.shape() != Shape::Merge {
        return Err("--leaf-strategy must be a leaf strategy and --merge-strategy a merge strategy".into());
    }
    leaf_spec.default_param("verify", "all_idle");
    merge_spec.default_param("verify", a.get("verify-policy").unwrap_or("all_idle"));
    Policy::parse(merge_spec.params.0.get("verify").unwrap())?;
    let leaf_fc = a.get("train-leaf").map(|d| Forecast::from_set(Path::new(d)).map(Rc::new)).transpose()?;
    let merge_fc = a.get("train-merge").map(|d| Forecast::from_set(Path::new(d)).map(Rc::new)).transpose()?;
    let out = a.need("out")?;
    let work = PathBuf::from(a.get("work").map_or(format!("{out}.work"), str::to_string));
    std::fs::create_dir_all(&work).map_err(|e| e.to_string())?;
    let gen_bin = arrivalgen_path(a)?;
    check_arrivalgen(&gen_bin)?;
    let delay = a.get("delay").unwrap_or("const(0)");
    let set_name = Path::new(leaf_dir).file_name().map_or(leaf_dir.to_string(), |s| s.to_string_lossy().into());

    // Votes sent per slot, over all subnets: a vote is one claim, however many aggregators saw it.
    let sent: Vec<u32> = leaf
        .slots
        .iter()
        .map(|s| s.arrivals.iter().filter(|x| x.kind == Kind::Vote).map(|x| (x.subnet, x.id)).collect::<HashSet<_>>().len() as u32)
        .collect();

    let mut text = format!("{ROW_HEADER}\n");
    let mut summary = String::from("| D_leaf ms | leaf runs | published | leaf coverage p50 | end_to_end p50 | end_to_end p10 | end_to_end mean | merge coverage p50 |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n");
    let mut best: Option<(f64, u64)> = None;
    for &dl in &d_leaf {
        let (mut lb, backend_name) = backend(a, model_path, Caps::default(), seed, noise)?;
        let mut est = cost_kind(a)?.estimator(Shape::Leaf);
        let mut st = Stats { end_to_end: Vec::new(), merge_coverage: Vec::new(), leaf_coverage: Vec::new(), published: 0, runs: 0 };
        let mut compose = String::from("slot,subnet,view,published_us,rate,signers_hex\n");
        for slot in slots.clone() {
            for (_, subnet, view) in runs_of(&leaf, slot) {
                let mut s = leaf_spec.make(leaf_fc.as_ref())?;
                let spec = RunSpec { set: &leaf, slot, subnet, view, deadline: dl, caps: lb.caps() };
                let rec = run_slot_with(&mut lb, s.as_mut(), &spec, calib::lend(&mut est));
                st.runs += 1;
                st.leaf_coverage.push(slot_metrics(&rec).coverage);
                if let Some((t, cov)) = rec.published.as_ref().filter(|(t, _)| *t <= dl) {
                    st.published += 1;
                    let mut signers = vec![0u64; (leaf.meta.subnet_size as usize).div_ceil(64)];
                    for c in cov.iter() {
                        let id = (c - SUBNET_SIZE * subnet as u32) as usize;
                        signers[id / 64] |= 1 << (id % 64);
                    }
                    let digits = (leaf.meta.subnet_size as usize).div_ceil(4);
                    compose.push_str(&format!("{slot},{subnet},{view},{t},{},{}\n", leaf.meta.final_rate, signers_to_hex(&signers, digits)));
                }
            }
        }
        let dir = work.join(format!("d{}", dl / 1000));
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        std::fs::write(dir.join("compose.csv"), compose).map_err(|e| e.to_string())?;
        let params = format!(
            "shape merge\nslot_ms {}\nsubnets {}\nsubnet_size {}\nagg.per_subnet {}\nagg.compose compose.csv\nagg.delay {delay}\nfinal_rate {}\ndeadline {}\nagg.rate {}\n",
            leaf.meta.slot_us / 1000,
            leaf.meta.subnets,
            leaf.meta.subnet_size,
            leaf.meta.views,
            a.num("merge-rate", 1u8)?,
            d_merge / 1000,
            leaf.meta.final_rate
        );
        std::fs::write(dir.join("merge.params"), params).map_err(|e| e.to_string())?;
        let status = Command::new(&gen_bin)
            .args(["--params", &dir.join("merge.params").to_string_lossy(), "--slots", &leaf.meta.slots.to_string(), "--seed", &seed.to_string(), "--out", &dir.join("set").to_string_lossy()])
            .status()
            .map_err(|e| format!("{}: {e} (build arrivalgen, or pass --arrivalgen)", gen_bin.display()))?;
        if !status.success() {
            return Err("arrivalgen failed".into());
        }
        let merge_set = ScheduleSet::read(&dir.join("set"))?;
        let (mut mb, _) = backend(a, merge_model_path, Caps::default(), seed, noise)?;
        for slot in slots.clone() {
            let mut s = merge_spec.make(merge_fc.as_ref())?;
            let spec = RunSpec { set: &merge_set, slot, subnet: 0, view: 0, deadline: d_merge, caps: mb.caps() };
            let mut rec = run_slot(&mut mb, s.as_mut(), &spec);
            rec.sent_override = Some(sent[slot as usize]);
            let m = slot_metrics(&rec);
            st.end_to_end.push(m.coverage_sent);
            st.merge_coverage.push(m.coverage);
            let row = Row { set: set_name.clone(), backend: backend_name.clone(), strategy: format!("{}+{}", s.name().replace(',', ";"), "e2e"), deadline_us: dl, slot, m };
            text += &row.to_csv();
            text.push('\n');
        }
        let (e50, e10, em) = (quantile(&st.end_to_end, 50), quantile(&st.end_to_end, 10), mean(&st.end_to_end));
        if best.is_none_or(|b| em > b.0) {
            best = Some((em, dl));
        }
        summary += &format!(
            "| {} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |\n",
            dl / 1000,
            st.runs,
            st.published,
            quantile(&st.leaf_coverage, 50),
            e50,
            e10,
            em,
            quantile(&st.merge_coverage, 50)
        );
    }
    std::fs::write(out, text).map_err(|e| e.to_string())?;
    print!("{summary}");
    if let Some((m, d)) = best {
        println!("\nbest D_leaf by mean end_to_end: {} ms ({m:.3})", d / 1000);
    }
    Ok(())
}
