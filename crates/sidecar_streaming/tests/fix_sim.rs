//! Regression tests for the simulator findings of `design/REVIEW.md`.
mod common;
use arrival_model::schedule::{Meta, ScheduleSet, SlotSchedule};
use common::*;
use sidecar_streaming::driver::{RunSpec, run_slot};
use sidecar_streaming::strategy::{Ctx, Strategy};
use sidecar_streaming::types::*;

struct OneJob;

impl Strategy for OneJob {
    fn name(&self) -> String {
        "one_job".into()
    }
    fn on_event(&mut self, e: Event, ctx: &mut dyn Ctx) {
        if matches!(e, Event::SlotStart) {
            ctx.prove(0, Job { children: vec![], signatures: vec![SigId(0)], rate: 2 });
        }
    }
}

fn two_subnet_set() -> ScheduleSet {
    ScheduleSet {
        meta: Meta {
            shape: "leaf".into(),
            slots: 1,
            slot_us: 4_000_000,
            att_due_us: 1_000_000,
            agg_due_us: 2_000_000,
            ptc_due_us: 3_000_000,
            deadline_us: 3_000_000,
            horizon_us: 8_000_000,
            final_rate: 2,
            expected_inputs: 1500,
            seed: 0,
            source: "recorded".into(),
            subnet_size: 1500,
            subnets: 2,
            views: 1,
        },
        params_txt: None,
        pool: Vec::new(),
        slots: vec![SlotSchedule::default()],
    }
}

/// Finding 12: the runs of one slot (subnets, views) draw independent noise, and each run is reproducible.
#[test]
fn noise_is_independent_per_run() {
    let text = SMOOTH.replace("sigma 0", "sigma 0.05");
    let set = two_subnet_set();
    let duration = |subnet: u8| {
        let mut backend = sidecar_streaming::sim::SimBackend::new(model(&text), Caps::default(), 1, true);
        let rec = run_slot(&mut backend, &mut OneJob, &RunSpec { set: &set, slot: 0, subnet, view: 0, deadline: 3_000_000, caps: Caps::default() });
        rec.jobs[0].end - rec.jobs[0].start
    };
    assert_ne!(duration(0), duration(1));
    assert_eq!(duration(0), duration(0));
}
