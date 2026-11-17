//! Runs of one strategy over a schedule set at several deadlines, with any backend.

use crate::backend::Backend;
use crate::driver::{RunSpec, run_slot};
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
}

/// The runs of a slot: one per (subnet, view) for `Leaf`, one for `Merge`. A run's id is `(slot * subnets + subnet) * views + view`.
pub fn runs_of(set: &ScheduleSet, slot: u32) -> Vec<(u32, u8, u8)> {
    if set.meta.shape == "merge" {
        return vec![(slot, 0, 0)];
    }
    let (s, v) = (set.meta.subnets, set.meta.views);
    (0..s).flat_map(|sn| (0..v).map(move |vw| ((slot * s + sn) * v + vw, sn as u8, vw as u8))).collect()
}

pub struct Job {
    pub spec: Spec,
    pub forecast: Option<Rc<Forecast>>,
}

#[allow(clippy::too_many_arguments)]
pub fn run_set(
    backend: &mut dyn Backend,
    set: &ScheduleSet,
    set_name: &str,
    backend_name: &str,
    job: &Job,
    deadlines: &[Micros],
    slots: std::ops::Range<u32>,
    mut progress: impl FnMut(&RunResult),
) -> Result<Vec<RunResult>, String> {
    let mut out = Vec::new();
    for &d in deadlines {
        for slot in slots.clone() {
            for (id, subnet, view) in runs_of(set, slot) {
                let mut strategy = job.spec.make(job.forecast.as_ref())?;
                let spec = RunSpec { set, slot, subnet, view, deadline: d, caps: backend.caps() };
                let record = run_slot(backend, strategy.as_mut(), &spec);
                let row = Row { set: set_name.into(), backend: backend_name.into(), strategy: strategy.name(), deadline_us: d, slot: id, m: slot_metrics(&record) };
                let r = RunResult { row, record };
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
