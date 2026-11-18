//! The strategy's cost estimate is learned, not read from a table (`calib.rs`, AUDIT.md section 11).
mod common;
use common::*;
use sidecar_streaming::backend::Backend;
use sidecar_streaming::calib::{Calibrated, CostEstimator, CostKind};
use sidecar_streaming::driver::{RunSpec, run_slot_with};
use sidecar_streaming::metric::{Metrics, SlotRecord, slot_metrics};
use sidecar_streaming::strategies::Spec;
use sidecar_streaming::types::*;

/// SMOOTH with every job twice as slow: another build on another machine.
const SLOW: &str = "form rows\nbackend slow\nthreads 8\nisa avx2\nsigma 0\nstep 170\nw0 0\n\
t0.1 60000\nlin.1 1000\npad.1 0\nt0.2 62000\nlin.2 1040\npad.2 0\nwc.1 200\nwc.2 200\nwk 0\nv.1 6000\nv.2 6000\nbytes.1 300000\nbytes.2 220000\n\
merge.step 400\nmerge.w0 0\nmerge.t0.1 60000\nmerge.lin.1 600\nmerge.pad.1 0\nmerge.t0.2 62000\nmerge.lin.2 640\nmerge.pad.2 0\nmerge.wc.2 300\nmerge.wc.1 300\nmerge.wk 0.02\n";

/// One slot with the strategy pricing jobs by `est` (`None`: the backend's table).
pub fn run_est(backend: &mut dyn Backend, set: &arrival_model::schedule::ScheduleSet, slot: u32, d: Micros, spec: &str, est: Option<&mut dyn CostEstimator>) -> (SlotRecord, Metrics) {
    let mut s = Spec::parse(spec).unwrap().make(None).unwrap();
    let rs = RunSpec { set, slot, subnet: 0, view: 0, deadline: d, caps: backend.caps() };
    let rec = run_slot_with(backend, s.as_mut(), &rs, est);
    let m = slot_metrics(&rec);
    (rec, m)
}

#[test]
fn cost_kind_parses() {
    assert_eq!(CostKind::parse("calibrated").unwrap(), CostKind::Calibrated);
    assert_eq!(CostKind::parse("table").unwrap(), CostKind::Table);
    assert!(CostKind::parse("perf").is_err());
    assert_eq!(CostKind::default(), CostKind::Calibrated);
    assert!(CostKind::Table.estimator(Shape::Leaf).is_none());
    assert!(CostKind::Calibrated.estimator(Shape::Leaf).is_some());
    // The lines cover leaf jobs; a merge job keeps the table.
    assert!(CostKind::Calibrated.estimator(Shape::Merge).is_none());
}

/// A fresh estimator knows nothing of the backend: two builds with different tables give the same first job.
/// With the table as the estimate they would not.
#[test]
fn the_strategy_has_no_path_to_the_table() {
    let set = make_set(&leaf_params(), 2, 5);
    let d = 3_000_000;
    let start_of_last_job = |text: &str, calibrated: bool| {
        let mut b = sim(text);
        let mut est = Calibrated::new();
        let e: Option<&mut dyn CostEstimator> = if calibrated { Some(&mut est) } else { None };
        let (rec, _) = run_est(&mut b, &set, 0, d, "one_shot_late", e);
        rec.jobs[0].start
    };
    assert_eq!(start_of_last_job(SMOOTH, true), start_of_last_job(SLOW, true));
    assert_ne!(start_of_last_job(SMOOTH, false), start_of_last_job(SLOW, false));
}

/// Within one slot of a chain the line covers the table of the build it runs on, without a large excess.
#[test]
fn the_model_converges_within_one_slot_for_a_chain() {
    // Votes spread over 2 s, so that a chain makes a dozen jobs.
    let p = arrival_model::model::Params::parse("shape leaf\nslot_ms 4000\nsubnets 1\nagg.per_subnet 1\nvotes.base uniform(300, 2300)\n", std::path::Path::new("."));
    let set = make_set(&p, 2, 5);
    for text in [SMOOTH, SLOW] {
        let mut b = sim(text);
        let mut est = Calibrated::new();
        let (rec, m) = run_est(&mut b, &set, 0, 3_000_000, "greedy_fit", Some(&mut est));
        assert!(rec.jobs.len() >= 4, "the chain made {} jobs", rec.jobs.len());
        assert!(m.got > 0, "the conservative first slot still publishes");
        for n in [30u32, 100, 200, 400] {
            let shape = JobShape::extend(600, 2, n, 2);
            let truth = b.model().prove_us(&shape, 8) as f64;
            let e = est.estimate(&shape) as f64;
            assert!(e >= truth && e <= truth * 1.4, "{text:.30}: n {n}: estimate {e} for truth {truth}");
        }
    }
}

/// A strategy calibrated on one build, run against another, and back: every slot publishes and takes every vote
/// once the model has seen a slot of the new build. No noise.
#[test]
fn a_model_calibrated_on_one_build_runs_on_another() {
    let set = make_set(&leaf_params(), 6, 9);
    let d = 3_000_000;
    for (from, to) in [(SMOOTH, SLOW), (SLOW, SMOOTH)] {
        for strategy in ["greedy_fit", "one_shot_late"] {
            let mut est = Calibrated::new();
            let mut b = sim(from);
            for slot in 0..3 {
                run_est(&mut b, &set, slot, d, strategy, Some(&mut est));
            }
            let mut b = sim(to);
            for slot in 0..6 {
                let (_, m) = run_est(&mut b, &set, slot, d, strategy, Some(&mut est));
                // The first slot may be priced by the other build's numbers; a late job is learned from.
                if slot >= 1 {
                    assert!(m.published_us.is_some_and(|t| t <= d), "{strategy}: slot {slot} of the new build published nothing");
                    // The reference: the same slot with the strategy reading the new build's table.
                    let (_, table) = run_est(&mut b, &set, slot, d, strategy, None);
                    assert!(m.coverage >= table.coverage - 0.3, "{strategy}: slot {slot} coverage {} against {} by table", m.coverage, table.coverage);
                }
            }
        }
    }
}
