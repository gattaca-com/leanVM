//! A slot that finished a proof publishes it at the deadline, whatever the strategy did (AUDIT.md findings 1 and 3).
mod common;
use arrival_model::model::Params;
use common::*;
use sidecar_streaming::backend::Backend;
use sidecar_streaming::calib::{CostEstimator, Observation};
use sidecar_streaming::driver::{RunSpec, run_slot, run_slot_with};
use sidecar_streaming::metric::slot_metrics;
use sidecar_streaming::strategies::Spec;
use sidecar_streaming::strategy::{Ctx, Strategy};
use sidecar_streaming::types::*;
use std::path::Path;

/// Thinks every job takes 1 ms.
struct Optimist;

impl CostEstimator for Optimist {
    fn estimate(&self, _: &JobShape) -> Micros {
        1_000
    }
    fn observe(&mut self, _: &Observation) {}
}

/// Half of the votes at 0.5 s, half at the end of the window.
fn two_bursts() -> Params {
    Params::parse("shape leaf\nslot_ms 4000\nsubnets 1\nagg.per_subnet 1\nvotes.base mix(0.5: uniform(500, 600), 0.5: uniform(2470, 2490))\n", Path::new("."))
}

/// The last job is believed to take 1 ms and takes much longer. The proof finished before it is published at D.
#[test]
fn the_best_finished_proof_is_published_when_the_last_job_overruns() {
    let set = make_set(&two_bursts(), 3, 4);
    let d = 2_500_000;
    for slot in 0..3 {
        let mut b = sim(SMOOTH);
        let mut s = Spec::parse("greedy_fit").unwrap().make(None).unwrap();
        let rs = RunSpec { set: &set, slot, subnet: 0, view: 0, deadline: d, caps: b.caps(), grace: 0 };
        let rec = run_slot_with(&mut b, s.as_mut(), &rs, Some(&mut Optimist));
        let m = slot_metrics(&rec);
        let last = rec.jobs.last().unwrap();
        assert!(last.end > d && last.result == "ok", "slot {slot}: the last job ends at {} after D", last.end);
        assert_eq!(m.published_us, Some(d), "slot {slot}");
        assert!(m.got > 0 && m.got < m.arrived, "slot {slot}: got {} of {}", m.got, m.arrived);
    }
}

/// The real leaf strategies tick on a 5 ms grid. Wherever `D` falls between two ticks, a slot whose votes are all in
/// and proved publishes everything by `D`, with or without the driver's fallback.
#[test]
fn the_leaf_strategies_publish_by_d_wherever_d_falls_on_the_tick_grid() {
    let set = make_set(&leaf_params(), 2, 3);
    for offset in [0u64, 1, 1_000, 2_500, 4_000, 4_999] {
        let d = 2_500_000 + offset;
        for name in ["greedy_fit", "cheap_chain", "min_batch_fit:m=300", "one_shot_late"] {
            for slot in 0..2 {
                let mut b = sim(SMOOTH);
                let (rec, m) = run(&mut b, &set, slot, d, name, None);
                assert_eq!(m.got, m.arrived, "{name} slot {slot} D {d}");
                assert!(m.published_us.is_some_and(|t| t <= d), "{name} slot {slot} D {d}");
                // The strategy itself published, not the driver at the end of the window: its last job was its own.
                assert!(rec.jobs.last().is_some_and(|j| j.useful), "{name} slot {slot} D {d}");
            }
        }
    }
}

/// Nothing is published when no proof finished: the fallback invents nothing.
#[test]
fn no_finished_proof_no_publish() {
    let set = make_set(&leaf_params(), 1, 3);
    let mut b = sim(SMOOTH);
    // D before the first vote: nothing to prove.
    let (_, m) = run(&mut b, &set, 0, 100_000, "greedy_fit", None);
    assert_eq!(m.published_us, None);
}
