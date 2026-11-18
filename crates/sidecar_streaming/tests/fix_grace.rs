//! The grace window: a proof finished by `D + grace` counts. Strategies still plan against `D`.
mod common;
use arrival_model::model::Params;
use common::*;
use sidecar_streaming::backend::Backend;
use sidecar_streaming::calib::{CostEstimator, Observation};
use sidecar_streaming::cli::{Args, grace_us};
use sidecar_streaming::driver::{RunSpec, run_slot, run_slot_with};
use sidecar_streaming::metric::{Row, SlotRecord, slot_metrics};
use sidecar_streaming::strategies::Spec;
use sidecar_streaming::strategy::{Ctx, Strategy};
use sidecar_streaming::types::*;
use std::path::Path;

/// One AGGREGATE of every vote that has come, started at `start_at`; published when it finishes.
struct Late {
    start_at: Micros,
    sigs: Vec<SigId>,
}

impl Strategy for Late {
    fn name(&self) -> String {
        "late".into()
    }
    fn on_event(&mut self, event: Event, ctx: &mut dyn Ctx) {
        match event {
            Event::SlotStart => ctx.set_timer(self.start_at, 1),
            Event::Signature(s) => self.sigs.push(s),
            Event::Timer(_) => {
                let job = Job { children: vec![], signatures: std::mem::take(&mut self.sigs), rate: ctx.slot().final_rate };
                ctx.prove(0, job);
            }
            Event::Proved { result: Ok(id), .. } => ctx.publish(id),
            _ => {}
        }
    }
}

/// `D` such that the job of every vote ends `over` microseconds after it.
fn run_late(grace: Micros, over: Micros) -> (SlotRecord, Micros) {
    let set = make_set(&leaf_params(), 1, 3);
    let mut b = sim(SMOOTH);
    let n = set.slots[0].arrivals.iter().filter(|a| a.subnet == 0 && a.view == 0).count() as u32;
    let cost = b.model().prove_us(&JobShape::aggregate(n, set.meta.final_rate), 8);
    let start_at = 1_500_000;
    let d = start_at + cost - over;
    let rs = RunSpec { set: &set, slot: 0, subnet: 0, view: 0, deadline: d, caps: b.caps(), grace };
    (run_slot(&mut b, &mut Late { start_at, sigs: Vec::new() }, &rs), d)
}

#[test]
fn a_proof_finished_inside_the_grace_window_counts() {
    let (rec, d) = run_late(25_000, 10_000);
    let m = slot_metrics(&rec);
    assert_eq!(m.got, m.arrived);
    assert!(m.arrived > 1000);
    assert_eq!(m.published_us, Some(d + 10_000));
    assert_eq!(m.slack_us, Some(0));
}

#[test]
fn a_proof_finished_after_the_grace_window_does_not() {
    for grace in [0, 5_000] {
        let (rec, _) = run_late(grace, 10_000);
        let m = slot_metrics(&rec);
        assert_eq!(m.published_us, None, "grace {grace}");
        assert_eq!(m.got, 0);
        // The job is still written down.
        assert_eq!(rec.jobs.len(), 1);
    }
}

/// Thinks every job takes 1 ms.
struct Optimist;

impl CostEstimator for Optimist {
    fn estimate(&self, _: &JobShape) -> Micros {
        1_000
    }
    fn observe(&mut self, _: &Observation) {}
}

/// The fallback waits for the end of the window while a job may still finish, then publishes the best finished proof.
#[test]
fn the_fallback_publishes_at_the_end_of_the_window() {
    let p = Params::parse("shape leaf\nslot_ms 4000\nsubnets 1\nagg.per_subnet 1\nvotes.base mix(0.5: uniform(500, 600), 0.5: uniform(2470, 2490))\n", Path::new("."));
    let set = make_set(&p, 1, 4);
    let d = 2_500_000;
    for grace in [0, 25_000] {
        let mut b = sim(SMOOTH);
        let mut s = Spec::parse("greedy_fit").unwrap().make(None).unwrap();
        let rs = RunSpec { set: &set, slot: 0, subnet: 0, view: 0, deadline: d, caps: b.caps(), grace };
        let rec = run_slot_with(&mut b, s.as_mut(), &rs, Some(&mut Optimist));
        let m = slot_metrics(&rec);
        assert_eq!(m.published_us, Some(d + grace), "grace {grace}");
        assert!(m.got > 0 && m.got < m.arrived);
        // Votes that come after D are not delivered, grace or not.
        assert!(rec.jobs.iter().all(|j| j.new_sigs <= m.arrived));
    }
}

#[test]
fn a_row_can_be_rescored_with_a_shorter_window() {
    let (rec, d) = run_late(25_000, 10_000);
    let row = Row { set: "s".into(), backend: "b".into(), strategy: "late".into(), deadline_us: d, slot: 0, m: slot_metrics(&rec) };
    let mut same = row.clone();
    same.rescore_grace(25_000);
    assert_eq!(same, row);
    let mut shorter = row.clone();
    shorter.rescore_grace(5_000);
    assert_eq!(shorter.m.got, 0);
    assert_eq!(shorter.m.coverage, 0.0);
    assert_eq!(shorter.m.published_us, None);
    assert!(shorter.m.arrived > 0);
}

#[test]
fn grace_ms_parses() {
    let args = |v: &[&str]| Args::parse(&v.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap();
    assert_eq!(grace_us(&args(&[])).unwrap(), 0);
    assert_eq!(grace_us(&args(&["--grace-ms", "25"])).unwrap(), 25_000);
    assert_eq!(grace_us(&args(&["--grace-ms", "0.5"])).unwrap(), 500);
    assert!(grace_us(&args(&["--grace-ms", "-1"])).is_err());
    assert!(grace_us(&args(&["--grace-ms", "x"])).is_err());
}
