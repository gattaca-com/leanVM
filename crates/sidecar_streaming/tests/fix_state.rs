//! `--estimator-state`: two calls of one slot each, joined by a state file, give the rows of one call over both slots.
mod common;
use common::*;
use sidecar_streaming::calib::{Calibrated, CostKind};
use sidecar_streaming::metric::Row;
use sidecar_streaming::runner::{RunResult, StateSpec, Study, read_state, run_set};
use sidecar_streaming::strategies::Spec;
use std::path::PathBuf;

fn run(spec: &str, slots: std::ops::Range<u32>, file: &PathBuf) -> Vec<RunResult> {
    let set = make_set(&leaf_params(), 6, 12);
    let warm = make_set(&leaf_params(), 6, 99);
    let mut b = sim(STEPPED);
    let state = Some(StateSpec { path: file.clone(), warmup_set: Some(warm) });
    let job = Study { cost: CostKind::Calibrated, grace: 0, spec: Spec::parse(spec).unwrap(), forecast: None, warmup: 2, state };
    run_set(&mut b, &set, "set", "test", &job, &[1_500_000, 2_500_000], slots, |_| {}).unwrap()
}

fn tmp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("fix_state-{}-{name}", std::process::id()));
    let _ = std::fs::remove_file(&p);
    p
}

fn rows(r: &[RunResult], slot_of: impl Fn(&Row) -> bool) -> Vec<Row> {
    r.iter().map(|x| x.row.clone()).filter(|x| slot_of(x)).collect()
}

#[test]
fn two_calls_with_a_state_file_match_one_call() {
    for spec in ["greedy_fit", "one_shot_late"] {
        let one = run(spec, 0..2, &tmp(&format!("one-{spec}")));
        let f = tmp(&format!("two-{spec}"));
        let a = run(spec, 0..1, &f);
        let b = run(spec, 1..2, &f);
        for d in [1_500_000, 2_500_000] {
            let want = rows(&one, |r| r.deadline_us == d);
            let mut got = rows(&a, |r| r.deadline_us == d);
            got.extend(rows(&b, |r| r.deadline_us == d));
            assert_eq!(want, got, "{spec} D {d}");
        }
    }
}

#[test]
fn the_estimator_grows_across_calls() {
    let f = tmp("grow");
    let a = run("greedy_fit", 0..1, &f);
    let b = run("greedy_fit", 1..2, &f);
    for d in [1_500_000, 2_500_000] {
        let first = a.iter().find(|r| r.row.deadline_us == d).unwrap().est_jobs;
        let second = b.iter().find(|r| r.row.deadline_us == d).unwrap().est_jobs;
        assert!(first > 0, "warm-up set trained the estimator at D {d}");
        assert!(second > first, "{second} not above {first} at D {d}");
    }
    let state = read_state(&f).unwrap();
    assert_eq!(state.len(), 2);
}

#[test]
fn a_bad_state_line_is_an_error() {
    assert!(Calibrated::restore(&["A 2 10 5000".to_string()]).is_ok());
    for l in ["X 2 10 5000", "A two 10 5000", "A 2 10", "A 2 -1 5000", "A 2 10 0", "A 2 nan 5000"] {
        assert!(Calibrated::restore(&[l.to_string()]).is_err(), "{l}");
    }
}
