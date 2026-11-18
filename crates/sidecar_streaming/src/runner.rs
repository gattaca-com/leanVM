//! Runs of one strategy over a schedule set at several deadlines, with any backend.

use crate::backend::Backend;
use crate::calib::{self, CostKind};
use crate::driver::{RunSpec, run_slot_with};
use crate::forecast::Forecast;
use crate::metric::{JobRec, Row, SlotRecord, slot_metrics};
use crate::strategies::Spec;
use crate::types::*;
use arrival_model::schedule::ScheduleSet;
use std::rc::Rc;

/// One executed run: the metric row and the record it came from.
pub struct RunResult {
    pub row: Row,
    pub record: SlotRecord,
    /// Jobs the estimator had learned from when this run started; 0 with no estimator.
    pub est_jobs: usize,
}

/// The runs of a slot: one per (subnet, view) for `Leaf`, one for `Merge`. A run's id is `(slot * subnets + subnet) * views + view`.
pub fn runs_of(set: &ScheduleSet, slot: u32) -> Vec<(u32, u8, u8)> {
    if set.meta.shape == "merge" {
        return vec![(slot, 0, 0)];
    }
    let (s, v) = (set.meta.subnets, set.meta.views);
    (0..s).flat_map(|sn| (0..v).map(move |vw| ((slot * s + sn) * v + vw, sn as u8, vw as u8))).collect()
}

pub struct Study {
    /// Where the strategy gets its job costs. State carries over all slots and deadlines of one `run_set`.
    pub cost: CostKind,
    /// A proof finished by `D + grace` counts.
    pub grace: Micros,
    pub spec: Spec,
    pub forecast: Option<Rc<Forecast>>,
    /// Slots run and discarded before each deadline's scored slots; see `warm_up`.
    pub warmup: u32,
}

/// Run the first `n` slots of `slots` at deadline `d` for `est` to learn from, and discard the results. Does nothing
/// when there is no estimator. The backend sees the same runs again in the scored pass: a simulator draws its noise
/// per run, so the scored slots do not change, and a live backend simply proves them twice.
#[allow(clippy::too_many_arguments)]
pub fn warm_up(
    backend: &mut dyn Backend,
    set: &ScheduleSet,
    spec: &Spec,
    forecast: Option<&Rc<Forecast>>,
    grace: Micros,
    d: Micros,
    slots: std::ops::Range<u32>,
    n: u32,
    est: &mut Option<Box<dyn calib::CostEstimator>>,
) -> Result<(), String> {
    if est.is_none() {
        return Ok(());
    }
    for slot in slots.take(n as usize) {
        for (_, subnet, view) in runs_of(set, slot) {
            let mut strategy = spec.make(forecast)?;
            let rs = RunSpec { set, slot, subnet, view, deadline: d, caps: backend.caps(), grace };
            run_slot_with(backend, strategy.as_mut(), &rs, calib::lend(est));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn run_set(
    backend: &mut dyn Backend,
    set: &ScheduleSet,
    set_name: &str,
    backend_name: &str,
    job: &Study,
    deadlines: &[Micros],
    slots: std::ops::Range<u32>,
    mut progress: impl FnMut(&RunResult),
) -> Result<Vec<RunResult>, String> {
    let mut out = Vec::new();
    for &d in deadlines {
        // One estimator per deadline, warmed on discarded slots: a row does not depend on the other deadlines.
        let mut est = job.cost.estimator(if set.meta.shape == "merge" { Shape::Merge } else { Shape::Leaf });
        warm_up(backend, set, &job.spec, job.forecast.as_ref(), job.grace, d, slots.clone(), job.warmup, &mut est)?;
        for slot in slots.clone() {
            for (id, subnet, view) in runs_of(set, slot) {
                let est_jobs = est.as_ref().map_or(0, |e| e.jobs_seen());
                let mut strategy = job.spec.make(job.forecast.as_ref())?;
                let spec = RunSpec { set, slot, subnet, view, deadline: d, caps: backend.caps(), grace: job.grace };
                let record = run_slot_with(backend, strategy.as_mut(), &spec, calib::lend(&mut est));
                let row = Row { set: set_name.into(), backend: backend_name.into(), strategy: strategy.name(), deadline_us: d, slot: id, m: slot_metrics(&record) };
                let r = RunResult { row, record, est_jobs };
                progress(&r);
                out.push(r);
            }
        }
    }
    Ok(out)
}

pub const JOBS_HEADER: &str = "slot,job,lane,op,children,child_claims,new_sigs,rate,start_us,end_us,predicted_us,result";

pub fn job_line(slot: u32, j: &JobRec) -> String {
    format!("{slot},{},{},{},{},{},{},{},{},{},{},{}", j.job, j.lane, j.op, j.children, j.child_claims, j.new_sigs, j.rate, j.start, j.end, j.predicted, j.result.replace(',', ";"))
}
