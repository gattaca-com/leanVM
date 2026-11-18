//! The fallback stamps a proof at most at `D + grace`. A proof that ended after that never counts.
mod common;
use arrival_model::schedule::{PoolEntry, ScheduleSet};
use common::*;
use sidecar_streaming::backend::{Backend, Done, ProofStore, Started};
use sidecar_streaming::cost::CostModel;
use sidecar_streaming::driver::{RunSpec, run_slot};
use sidecar_streaming::metric::slot_metrics;
use sidecar_streaming::sim::SimBackend;
use sidecar_streaming::strategies::Spec;
use sidecar_streaming::types::*;

/// A simulator whose every wait that ends without a completion returns `over` microseconds late.
struct Late {
    inner: SimBackend,
    over: Micros,
    skew: Micros,
}

impl Backend for Late {
    fn caps(&self) -> Caps {
        self.inner.caps()
    }
    fn model(&self) -> &dyn CostModel {
        self.inner.model()
    }
    fn now(&self) -> Micros {
        self.inner.now() + self.skew
    }
    fn begin_slot(&mut self, set: &ScheduleSet, run: u32) {
        self.skew = 0;
        self.inner.begin_slot(set, run)
    }
    fn receive(&mut self, id: ProofId, e: &PoolEntry, valid: bool) {
        self.inner.receive(id, e, valid)
    }
    fn start_prove(&mut self, lane: usize, job: &Job, store: &ProofStore, made: ProofId) -> Started {
        self.inner.start_prove(lane, job, store, made)
    }
    fn start_verify(&mut self, lane: usize, proof: ProofId, store: &ProofStore) -> Started {
        self.inner.start_verify(lane, proof, store)
    }
    fn abort(&mut self, lane: usize) -> bool {
        self.inner.abort(lane)
    }
    fn wait(&mut self, until: Micros) -> Option<Done> {
        let d = self.inner.wait(until);
        if d.is_none() {
            self.skew = self.over;
        }
        d
    }
}

/// Run `greedy_simple`, which publishes only through the fallback, with grace `grace`.
fn run_simple(over: Micros, d: Micros, grace: Micros, slot: u32) -> (sidecar_streaming::metric::SlotRecord, sidecar_streaming::metric::Metrics) {
    let set = make_set(&leaf_params(), 3, 5);
    let mut b = Late { inner: sim(SMOOTH), over, skew: 0 };
    let mut s = Spec::parse("greedy_simple").unwrap().make(None).unwrap();
    let rs = RunSpec { set: &set, slot, subnet: 0, view: 0, deadline: d, caps: b.caps(), grace };
    let rec = run_slot(&mut b, s.as_mut(), &rs);
    let m = slot_metrics(&rec);
    (rec, m)
}

/// The last job runs past `D` and the window ends on a wake-up 40 us late: the earlier proof still counts.
#[test]
fn a_wake_up_that_overshoots_the_window_does_not_make_a_proof_late() {
    let grace = 25_000;
    // The votes end about 1.2 s into the slot and the jobs are short: pick a D inside the work so a job straddles it.
    let mut found = 0;
    for d in (1_300_000..1_700_000).step_by(10_000) {
        let (rec, m) = run_simple(40, d, grace, 0);
        let end = d + grace;
        let last = rec.jobs.last().unwrap();
        if last.end > end && rec.jobs.len() > 1 {
            found += 1;
            let t = m.published_us.expect("an earlier proof was finished and must be published");
            assert!(t <= end, "stamped {t}, window ends {end}");
            assert!(m.got > 0, "D {d}");
        }
    }
    assert!(found > 0, "no D had a job straddle the window; the test checks nothing");
}

/// A proof whose job ended after `D + grace` is never published, however the stamp is clamped.
#[test]
fn a_proof_that_ended_after_the_window_never_counts() {
    // One job only, and it ends after the window: nothing is published.
    let (rec, m) = run_simple(40, 575_000, 5_000, 0);
    assert!(rec.jobs.iter().all(|j| j.end > 580_000) || rec.jobs.is_empty());
    assert_eq!(m.published_us, None);
    assert_eq!(m.got, 0);
}
