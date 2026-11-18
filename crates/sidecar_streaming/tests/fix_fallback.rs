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

/// A strategy whose own clock never lands in the last 5 ms before `D`.
struct Drift {
    sigs: Vec<SigId>,
    kept: Option<ProofId>,
    busy: bool,
}

impl Strategy for Drift {
    fn name(&self) -> String {
        "drift".into()
    }
    fn on_event(&mut self, event: Event, ctx: &mut dyn Ctx) {
        match event {
            Event::SlotStart => ctx.set_timer(7_000, 1),
            Event::Signature(s) => self.sigs.push(s),
            Event::Proved { result: Ok(id), .. } => {
                self.kept = Some(id);
                self.busy = false;
            }
            Event::Timer(_) => {
                let (now, d) = (ctx.now(), ctx.slot().deadline);
                if !self.busy && !self.sigs.is_empty() {
                    let job = Job { children: self.kept.into_iter().collect(), signatures: std::mem::take(&mut self.sigs), rate: ctx.slot().final_rate };
                    ctx.prove(0, job);
                    self.busy = true;
                }
                if let Some(k) = self.kept.filter(|_| now + 5_000 >= d && now < d) {
                    ctx.publish(k);
                } else if now < d {
                    ctx.set_timer(now + 7_000, 1);
                }
            }
            _ => {}
        }
    }
}

/// Finding 3: ticks 6 ms and 1 ms before `D` never come, so the strategy never publishes. The driver does, at `D`.
#[test]
fn a_finished_proof_is_published_without_a_tick_near_d() {
    let set = make_set(&leaf_params(), 2, 3);
    // Ticks fall on multiples of 7 ms: the last one before D is 6 ms ahead of it.
    let d = 2_505_000;
    for slot in 0..2 {
        let mut b = sim(SMOOTH);
        let rs = RunSpec { set: &set, slot, subnet: 0, view: 0, deadline: d, caps: b.caps(), grace: 0 };
        let rec = run_slot(&mut b, &mut Drift { sigs: Vec::new(), kept: None, busy: false }, &rs);
        let m = slot_metrics(&rec);
        assert_eq!(m.got, m.arrived, "slot {slot}");
        assert!(m.published_us.is_some_and(|t| t <= d), "slot {slot}");
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
