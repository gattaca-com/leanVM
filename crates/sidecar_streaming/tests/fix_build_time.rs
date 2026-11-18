//! The estimator learns the time from the start of the payload build, not from after the send.
mod common;
use arrival_model::schedule::{PoolEntry, ScheduleSet};
use common::*;
use sidecar_streaming::backend::*;
use sidecar_streaming::calib::{CostEstimator, Observation};
use sidecar_streaming::cost::CostModel;
use sidecar_streaming::driver::{RunSpec, run_slot_with};
use sidecar_streaming::sim::SimBackend;
use sidecar_streaming::strategy::{Ctx, Strategy};
use sidecar_streaming::types::*;

const BUILD: Micros = 30_000;

/// A simulator whose `start_prove` takes `BUILD` of the clock, as building a payload does.
struct SlowBuild(SimBackend);

impl Backend for SlowBuild {
    fn caps(&self) -> Caps {
        self.0.caps()
    }
    fn model(&self) -> &dyn CostModel {
        self.0.model()
    }
    fn now(&self) -> Micros {
        self.0.now()
    }
    fn begin_slot(&mut self, set: &ScheduleSet, run: u32) {
        self.0.begin_slot(set, run)
    }
    fn receive(&mut self, id: ProofId, e: &PoolEntry, valid: bool) {
        self.0.receive(id, e, valid)
    }
    fn start_prove(&mut self, lane: usize, job: &Job, store: &ProofStore, made: ProofId) -> Started {
        let until = self.0.now() + BUILD;
        assert!(self.0.wait(until).is_none());
        self.0.start_prove(lane, job, store, made)
    }
    fn start_verify(&mut self, lane: usize, proof: ProofId, store: &ProofStore) -> Started {
        self.0.start_verify(lane, proof, store)
    }
    fn abort(&mut self, lane: usize) -> bool {
        self.0.abort(lane)
    }
    fn wait(&mut self, until: Micros) -> Option<Done> {
        self.0.wait(until)
    }
}

#[derive(Default)]
struct Recorder(Vec<Observation>);

impl CostEstimator for Recorder {
    fn estimate(&self, _: &JobShape) -> Micros {
        1_000
    }
    fn observe(&mut self, o: &Observation) {
        self.0.push(*o);
    }
}

struct OneJob(Vec<SigId>);

impl Strategy for OneJob {
    fn name(&self) -> String {
        "one_job".into()
    }
    fn on_event(&mut self, event: Event, ctx: &mut dyn Ctx) {
        match event {
            Event::SlotStart => ctx.set_timer(1_500_000, 1),
            Event::Signature(s) => self.0.push(s),
            Event::Timer(_) => {
                let job = Job { children: vec![], signatures: std::mem::take(&mut self.0), rate: ctx.slot().final_rate };
                ctx.prove(0, job);
            }
            Event::Proved { result: Ok(id), .. } => ctx.publish(id),
            _ => {}
        }
    }
}

#[test]
fn the_observed_time_includes_the_payload_build() {
    let set = make_set(&leaf_params(), 1, 3);
    let mut b = SlowBuild(sim(SMOOTH));
    let mut est = Recorder::default();
    let rs = RunSpec { set: &set, slot: 0, subnet: 0, view: 0, deadline: 3_900_000, caps: b.caps(), grace: 0 };
    let rec = run_slot_with(&mut b, &mut OneJob(Vec::new()), &rs, Some(&mut est));
    let j = &rec.jobs[0];
    assert_eq!(est.0.len(), 1);
    // No noise: the sidecar's own time is the prediction.
    assert_eq!(est.0[0].took_us, j.predicted + BUILD);
    assert_eq!(j.end - j.start, j.predicted + BUILD);
    assert_eq!(j.start, 1_500_000);
}
