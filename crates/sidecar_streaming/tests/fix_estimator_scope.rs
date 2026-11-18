//! One estimator per deadline, warmed on discarded slots: a row does not depend on the deadline list's order.
mod common;
use common::*;
use sidecar_streaming::calib::CostKind;
use sidecar_streaming::cli::{Args, DEFAULT_WARMUP, warmup_slots};
use sidecar_streaming::metric::Row;
use sidecar_streaming::runner::{Study, run_set};
use sidecar_streaming::strategies::Spec;
use sidecar_streaming::types::*;

fn rows(deadlines: &[Micros], spec: &str, warmup: u32) -> Vec<Row> {
    let set = make_set(&leaf_params(), 6, 12);
    let mut b = sim(STEPPED);
    let job = Study { cost: CostKind::Calibrated, grace: 0, spec: Spec::parse(spec).unwrap(), forecast: None, warmup };
    run_set(&mut b, &set, "set", "test", &job, deadlines, 0..6, |_| {}).unwrap().into_iter().map(|r| r.row).collect()
}

fn of(rows: &[Row], d: Micros) -> Vec<Row> {
    rows.iter().filter(|r| r.deadline_us == d).cloned().collect()
}

#[test]
fn results_do_not_depend_on_the_order_of_the_deadlines() {
    let (d1, d2, d3) = (1_500_000, 1_800_000, 2_500_000);
    for spec in ["greedy_fit", "one_shot_late"] {
        for warmup in [0, 2] {
            let fwd = rows(&[d1, d2, d3], spec, warmup);
            let rev = rows(&[d3, d2, d1], spec, warmup);
            assert_eq!(fwd.len(), 18);
            for d in [d1, d2, d3] {
                assert_eq!(of(&fwd, d), of(&rev, d), "{spec} warmup {warmup} D {d}");
            }
        }
    }
}

#[test]
fn the_warm_up_slots_are_not_in_the_rows() {
    let r = rows(&[1_800_000], "greedy_fit", 2);
    assert_eq!(r.len(), 6);
    assert_eq!(r.iter().map(|x| x.slot).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn warmup_slots_parses() {
    let args = |v: &[&str]| Args::parse(&v.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap();
    assert_eq!(warmup_slots(&args(&[])).unwrap(), DEFAULT_WARMUP);
    assert_eq!(DEFAULT_WARMUP, 2);
    assert_eq!(warmup_slots(&args(&["--warmup-slots", "0"])).unwrap(), 0);
    assert_eq!(warmup_slots(&args(&["--warmup-slots", "5"])).unwrap(), 5);
    assert!(warmup_slots(&args(&["--warmup-slots", "-1"])).is_err());
}
