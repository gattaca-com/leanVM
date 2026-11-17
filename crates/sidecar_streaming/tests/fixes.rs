//! Regression tests for the review findings of `design/REVIEW.md`.
mod common;
use arrival_model::model::Params;
use common::*;
use std::path::Path;

fn merge_invalid_params() -> Params {
    Params::parse(
        "shape merge\nslot_ms 4000\nsubnets 16\nagg.per_subnet 3\nagg.base uniform(1500, 2900)\nagg.invalid 0.1\n",
        Path::new("."),
    )
}

/// Finding 3: a failed merge job must not lose the kept root.
#[test]
fn failed_merge_job_keeps_the_root() {
    let set = make_set(&merge_invalid_params(), 40, 11);
    let mut backend = sim(SMOOTH);
    let d = 4_000_000;
    let mut checked = 0;
    for slot in 0..40 {
        let (rec, m) = run(&mut backend, &set, slot, d, "two_round:verify=inline", None);
        let best_ok = rec.jobs.iter().filter(|j| j.result == "ok" && j.end + 100_000 <= d).map(|j| j.result_claims).max();
        if let Some(b) = best_ok {
            assert!(m.got >= b, "slot {slot}: published {} claims after a job made {b}", m.got);
            checked += 1;
        }
    }
    assert!(checked > 3, "only {checked} slots had a successful job");
}
