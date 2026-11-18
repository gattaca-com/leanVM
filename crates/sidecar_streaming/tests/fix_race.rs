//! A completion that arrives after the end of the window, because `wait` overshoots, is not the strategy's to see
//! and not the fallback's to publish. The live backend polls every 50 us, so this happens there.
mod common;
use arrival_model::schedule::{PoolEntry, ScheduleSet};
use common::*;
use sidecar_streaming::backend::*;
use sidecar_streaming::cost::CostModel;
use sidecar_streaming::driver::{RunSpec, run_slot};
use sidecar_streaming::metric::slot_metrics;
use sidecar_streaming::sim::SimBackend;
use sidecar_streaming::strategy::{Ctx, Strategy};
use sidecar_streaming::types::*;

/// A simulator whose `wait` overshoots `until` by `slip`.
struct Slip {
    inner: SimBackend,
    slip: Micros,
}

impl Backend for Slip {
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
        self.inner.start_prove(lane, job, store, made)
    }
    fn start_verify(&mut self, lane: usize, proof: ProofId, store: &ProofStore) -> Started {
        self.inner.start_verify(lane, proof, store)
    }
    fn abort(&mut self, lane: usize) -> bool {
        self.inner.abort(lane)
    }
    fn wait(&mut self, until: Micros) -> Option<Done> {
        let u = if until == Micros::MAX { until } else { until + self.slip };
        self.inner.wait(u)
    }
}

/// Proves every vote in one job at `start_at`. Publishes on completion, or not at all if `publish` is false.
/// Records the clock of every completion it is shown.
struct OneJob {
    start_at: Micros,
    publish: bool,
    sigs: Vec<SigId>,
    seen_at: Vec<Micros>,
}

impl Strategy for OneJob {
    fn name(&self) -> String {
        "one_job".into()
    }
    fn on_event(&mut self, event: Event, ctx: &mut dyn Ctx) {
        match event {
            Event::SlotStart => ctx.set_timer(self.start_at, 1),
            Event::Signature(s) => self.sigs.push(s),
            Event::Timer(_) => {
                let job = Job { children: vec![], signatures: std::mem::take(&mut self.sigs), rate: ctx.slot().final_rate };
                ctx.prove(0, job);
            }
            Event::Proved { result: Ok(id), .. } => {
                self.seen_at.push(ctx.now());
                if self.publish {
                    ctx.publish(id);
                }
            }
            _ => {}
        }
    }
}

const SLIP: Micros = 40;
const START_AT: Micros = 1_500_000;

/// The end of the job when the window is wide, under this slip.
fn real_end(set: &ScheduleSet, slip: Micros) -> Micros {
    let mut b = Slip { inner: sim(SMOOTH), slip };
    let rs = RunSpec { set, slot: 0, subnet: 0, view: 0, deadline: 3_900_000, caps: b.caps(), grace: 0 };
    let mut s = OneJob { start_at: START_AT, publish: false, sigs: Vec::new(), seen_at: Vec::new() };
    run_slot(&mut b, &mut s, &rs).jobs[0].end
}

/// A window that closes `over` microseconds before the job ends, `grace` of it being grace.
fn run_over(slip: Micros, grace: Micros, over: Micros, publish: bool) -> (sidecar_streaming::metric::SlotRecord, Micros, Vec<Micros>) {
    let set = make_set(&leaf_params(), 1, 3);
    let d = real_end(&set, slip) - over - grace;
    let mut b = Slip { inner: sim(SMOOTH), slip };
    let rs = RunSpec { set: &set, slot: 0, subnet: 0, view: 0, deadline: d, caps: b.caps(), grace };
    let mut s = OneJob { start_at: START_AT, publish, sigs: Vec::new(), seen_at: Vec::new() };
    let rec = run_slot(&mut b, &mut s, &rs);
    (rec, d, s.seen_at)
}

#[test]
fn the_fallback_does_not_publish_a_proof_that_finished_after_the_window() {
    for grace in [0, 25_000] {
        let (rec, d, _) = run_over(SLIP, grace, 20, false);
        assert!(rec.jobs[0].end > d + grace, "the job must end after the window");
        let m = slot_metrics(&rec);
        assert_eq!(m.got, 0, "grace {grace}: a proof that ended after the window counted");
        assert!(m.published_us.is_none_or(|t| t > d + grace), "grace {grace}: published at {:?}", m.published_us);
    }
}

#[test]
fn the_strategy_is_not_shown_a_completion_after_the_window() {
    for grace in [0, 25_000] {
        let (rec, d, seen) = run_over(SLIP, grace, 20, true);
        assert!(rec.jobs[0].end > d + grace);
        assert!(seen.is_empty(), "grace {grace}: shown a completion at {seen:?}, window ends {}", d + grace);
        // The job is still written down, and what ended late is not useful.
        assert_eq!(rec.jobs[0].result, "ok");
        assert_eq!(slot_metrics(&rec).got, 0);
    }
}

#[test]
fn a_completion_inside_the_window_is_still_delivered() {
    for grace in [0, 25_000] {
        let (rec, d, seen) = run_over(SLIP, grace, 0, true);
        // The job ends at the very end of the window, which counts.
        assert!(rec.jobs[0].end <= d + grace);
        assert_eq!(seen.len(), 1);
        let m = slot_metrics(&rec);
        assert_eq!(m.got, m.arrived);
    }
}
