//! The simulator can replay measured live error ratios, so that it shows empty slots (AUDIT.md finding 7).
mod common;
use common::*;
use sidecar_streaming::backend::Backend;
use sidecar_streaming::driver::{RunSpec, run_slot};
use sidecar_streaming::sim::{Replay, SimBackend};
use sidecar_streaming::strategies::Spec;
use sidecar_streaming::types::*;

const HEADER: &str = "slot,job,lane,op,children,child_claims,new_sigs,rate,start_us,end_us,predicted_us,result";

/// A live job line whose duration is `ratio` times its prediction of 1 s.
fn job(op: &str, claims: u32, new: u32, ratio: f64, result: &str) -> String {
    format!("0,0,0,{op},0,{claims},{new},2,1000,{},1000000,{result}", 1000 + (ratio * 1_000_000.0) as u64)
}

fn file(lines: &[String]) -> String {
    format!("{HEADER}\n{}\n", lines.join("\n"))
}

#[test]
fn only_finished_jobs_count_and_an_empty_file_is_refused() {
    let t = file(&[job("AGGREGATE", 0, 900, 1.1, "ok"), job("AGGREGATE", 0, 900, 5.0, "Rejected(x)")]);
    assert_eq!(Replay::from_job_files(&[t]).unwrap().jobs(), 1);
    assert!(Replay::from_job_files(&[file(&[job("AGGREGATE", 0, 900, 5.0, "Rejected(x)")])]).is_err());
    assert!(Replay::from_job_files(&[format!("{HEADER}\n1,2,3\n")]).is_err());
}

fn backend_with(texts: &[String]) -> SimBackend {
    SimBackend::new(model(SMOOTH), Caps::default(), 1, false).with_replay(Replay::from_job_files(texts).unwrap())
}

#[test]
fn a_job_takes_its_prediction_times_the_replayed_ratio() {
    let set = make_set(&leaf_params(), 1, 3);
    let t = file(&(0..10).map(|_| job("AGGREGATE", 0, 900, 1.25, "ok")).collect::<Vec<_>>());
    let mut b = backend_with(&[t]);
    let mut s = Spec::parse("one_shot_late").unwrap().make(None).unwrap();
    let rs = RunSpec { set: &set, slot: 0, subnet: 0, view: 0, deadline: 3_000_000, caps: b.caps(), grace: 0 };
    let rec = run_slot(&mut b, s.as_mut(), &rs);
    let j = &rec.jobs[0];
    assert_eq!(j.end - j.start, (j.predicted as f64 * 1.25).round() as u64);
}

#[test]
fn ratios_are_drawn_per_op_and_size() {
    // Large AGGREGATEs are slow, small ones are exact: a one-shot of 1500 votes draws from the large pool.
    let mut lines: Vec<String> = (0..10).map(|_| job("AGGREGATE", 0, 900, 1.3, "ok")).collect();
    lines.extend((0..10).map(|_| job("AGGREGATE", 0, 100, 1.0, "ok")));
    let set = make_set(&leaf_params(), 1, 3);
    let mut b = backend_with(&[file(&lines)]);
    let mut s = Spec::parse("one_shot_late").unwrap().make(None).unwrap();
    let rs = RunSpec { set: &set, slot: 0, subnet: 0, view: 0, deadline: 3_000_000, caps: b.caps(), grace: 0 };
    let j = run_slot(&mut b, s.as_mut(), &rs).jobs[0].clone();
    assert!(j.new_sigs >= 700);
    assert_eq!(j.end - j.start, (j.predicted as f64 * 1.3).round() as u64);
}

#[test]
fn an_op_with_no_replayed_ratio_keeps_the_model_noise() {
    // Only EXTENDs in the file: an AGGREGATE takes its table time.
    let t = file(&(0..10).map(|_| job("EXTEND", 100, 100, 2.0, "ok")).collect::<Vec<_>>());
    let set = make_set(&leaf_params(), 1, 3);
    let mut b = backend_with(&[t]);
    let mut s = Spec::parse("one_shot_late").unwrap().make(None).unwrap();
    let rs = RunSpec { set: &set, slot: 0, subnet: 0, view: 0, deadline: 3_000_000, caps: b.caps(), grace: 0 };
    let j = run_slot(&mut b, s.as_mut(), &rs).jobs[0].clone();
    assert_eq!(j.end - j.start, j.predicted);
}

/// Without noise the one-shot, planning at a 1 % margin, never misses. With the replay of a prover that is 0 % or 40 %
/// slower than predicted, about half of the slots publish nothing.
#[test]
fn the_replay_shows_empty_slots_that_the_simulator_did_not() {
    let set = make_set(&leaf_params(), 40, 5);
    let lines: Vec<String> = (0..20).map(|i| job("AGGREGATE", 0, 900, if i % 2 == 0 { 1.0 } else { 1.4 }, "ok")).collect();
    let d = 2_500_000;
    let empties = |mut b: SimBackend| {
        (0..40)
            .filter(|&slot| {
                let mut s = Spec::parse("one_shot_late").unwrap().make(None).unwrap();
                let rs = RunSpec { set: &set, slot, subnet: 0, view: 0, deadline: d, caps: b.caps(), grace: 0 };
                run_slot(&mut b, s.as_mut(), &rs).published.is_none()
            })
            .count()
    };
    assert_eq!(empties(sim(SMOOTH)), 0);
    let e = empties(backend_with(&[file(&lines)]));
    assert!((8..=32).contains(&e), "{e} empty slots of 40");
}

#[test]
fn the_replay_is_reproducible() {
    let set = make_set(&leaf_params(), 3, 5);
    let lines: Vec<String> = (0..20).map(|i| job("AGGREGATE", 0, 900, 0.9 + 0.02 * i as f64, "ok")).collect();
    let run = || {
        let mut b = backend_with(&[file(&lines)]);
        (0..3)
            .map(|slot| {
                let mut s = Spec::parse("one_shot_late").unwrap().make(None).unwrap();
                let rs = RunSpec { set: &set, slot, subnet: 0, view: 0, deadline: 3_000_000, caps: b.caps(), grace: 0 };
                let j = run_slot(&mut b, s.as_mut(), &rs).jobs[0].clone();
                j.end - j.start
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(run(), run());
    // And it varies from slot to slot.
    let r = run();
    assert!(r.iter().any(|x| *x != r[0]));
}

