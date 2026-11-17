//! Regression tests for the live-backend findings of `design/REVIEW.md` that a simulator can reach.
mod common;
use arrival_model::schedule::{PoolEntry, ScheduleSet};
use common::*;
use sidecar_streaming::backend::*;
use sidecar_streaming::cost::CostModel;
use sidecar_streaming::driver::{RunSpec, run_slot};
use sidecar_streaming::metric::slot_metrics;
use sidecar_streaming::sim::SimBackend;
use sidecar_streaming::strategies::Spec;
use sidecar_streaming::types::*;

/// A simulator whose first prove job is rejected, as a live sidecar may do.
struct RejectFirst {
    inner: SimBackend,
    rejected: bool,
    pending: Option<Done>,
}

impl Backend for RejectFirst {
    fn caps(&self) -> Caps {
        self.inner.caps()
    }
    fn model(&self) -> &dyn CostModel {
        self.inner.model()
    }
    fn now(&self) -> Micros {
        self.inner.now()
    }
    fn begin_slot(&mut self, set: &ScheduleSet, run: u32) {
        self.inner.begin_slot(set, run)
    }
    fn receive(&mut self, id: ProofId, e: &PoolEntry, valid: bool) {
        self.inner.receive(id, e, valid)
    }
    fn start_prove(&mut self, lane: usize, job: &Job, store: &ProofStore, made: ProofId) -> Started {
        if self.rejected {
            return self.inner.start_prove(lane, job, store, made);
        }
        // Reject after 1 ms of lane time, leaving the inner simulator idle.
        self.rejected = true;
        let at = self.inner.now() + 1000;
        self.pending = Some(Done { lane, at, outcome: Outcome::Failed(Failure::Rejected("rejected".into())) });
        Started::default()
    }
    fn start_verify(&mut self, lane: usize, proof: ProofId, store: &ProofStore) -> Started {
        self.inner.start_verify(lane, proof, store)
    }
    fn abort(&mut self, lane: usize) -> bool {
        self.inner.abort(lane)
    }
    fn wait(&mut self, until: Micros) -> Option<Done> {
        if let Some(p) = &self.pending
            && p.at <= until
        {
            let p = self.pending.take().unwrap();
            self.inner.wait(p.at);
            return Some(p);
        }
        self.inner.wait(until)
    }
}

/// Finding 13: a rejected leaf job keeps the kept proof and the signatures it took.
#[test]
fn rejected_leaf_job_loses_nothing() {
    let set = make_set(&leaf_params(), 5, 6);
    for slot in 0..5 {
        let mut backend = RejectFirst { inner: sim(SMOOTH), rejected: false, pending: None };
        let mut s = Spec::parse("greedy_fit").unwrap().make(None).unwrap();
        let rs = RunSpec { set: &set, slot, subnet: 0, view: 0, deadline: 2_500_000, caps: backend.caps() };
        let rec = run_slot(&mut backend, s.as_mut(), &rs);
        let m = slot_metrics(&rec);
        assert!(rec.jobs.iter().any(|j| j.result.starts_with("Rejected")), "the rejection happened");
        // Every vote that arrived is in the proof.
        assert_eq!(m.got, m.arrived, "slot {slot}");
    }
}
