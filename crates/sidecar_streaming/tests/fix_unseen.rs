//! A class the estimator has not seen is derived from one it has, so a calibrated plan_cover can use its cheap rate.
mod common;
use common::*;
use sidecar_streaming::backend::Backend;
use sidecar_streaming::calib::{Calibrated, CostEstimator};
use sidecar_streaming::driver::{RunSpec, run_slot_with};
use sidecar_streaming::forecast::Forecast;
use sidecar_streaming::strategies::Spec;
use sidecar_streaming::types::*;
use std::rc::Rc;

fn rate_one_jobs(calibrated: bool) -> (u32, u32) {
    let train = make_set(&leaf_params(), 4, 11);
    let set = make_set(&leaf_params(), 30, 12);
    let fc = Rc::new(Forecast::from_schedule_set(&train));
    let mut b = sim(SMOOTH);
    let mut est = Calibrated::new();
    let (mut rate1, mut jobs) = (0, 0);
    for slot in 0..30 {
        let mut s = Spec::parse("plan_cover").unwrap().make(Some(&fc)).unwrap();
        let rs = RunSpec { set: &set, slot, subnet: 0, view: 0, deadline: 1_600_000, caps: b.caps(), grace: 0 };
        let e: Option<&mut dyn CostEstimator> = if calibrated { Some(&mut est) } else { None };
        let rec = run_slot_with(&mut b, s.as_mut(), &rs, e);
        jobs += rec.jobs.len() as u32;
        rate1 += rec.jobs.iter().filter(|j| j.rate == 1).count() as u32;
    }
    (rate1, jobs)
}

#[test]
fn plan_cover_uses_rate_one_under_calibration() {
    let (table, _) = rate_one_jobs(false);
    let (cal, jobs) = rate_one_jobs(true);
    assert!(table > 0, "the table run is the reference");
    // The first slot learns rate 2 only. Rate 1 must be priced from it afterwards.
    assert!(cal >= table / 4, "{cal} rate-1 jobs of {jobs} under calibration, {table} with the table");
}
