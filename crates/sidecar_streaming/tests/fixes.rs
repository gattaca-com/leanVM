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

/// Finding 4: a slot with nothing arrived by `D` has no coverage, is left out of the aggregates and counted apart.
#[test]
fn empty_slots_are_not_full_coverage() {
    use sidecar_streaming::metric::*;
    let set = make_set(&leaf_params(), 10, 3);
    let mut backend = sim(SMOOTH);
    // The votes start at 567 ms: at D = 300 ms nothing has arrived.
    let (_, m) = run(&mut backend, &set, 0, 300_000, "one_shot_late", None);
    assert_eq!((m.arrived, m.got), (0, 0));
    assert!(m.coverage.is_nan());
    let row = |slot: u32, d: u64| {
        let (_, m) = run(&mut sim(SMOOTH), &set, slot, d, "one_shot_late", None);
        Row { set: "s".into(), backend: "b".into(), strategy: "x".into(), deadline_us: d, slot, m }
    };
    let early: Vec<Row> = (0..10).map(|s| row(s, 300_000)).collect();
    let sm = summarize(&early.iter().collect::<Vec<_>>());
    assert_eq!((sm.n, sm.empty), (0, 10));
    assert!(sm.p10.is_nan() && sm.p50.is_nan());
    // D_99 skips a deadline with no valid slot.
    let late: Vec<Row> = (0..10).map(|s| row(s, 3_000_000)).collect();
    let p10 = summarize(&late.iter().collect::<Vec<_>>()).p10;
    assert_eq!(d99(&[(300_000, sm.p10), (3_000_000, p10)]), Some(3_000_000));
    // The CSV keeps the N/A.
    assert!(Row::from_csv(&early[0].to_csv()).unwrap().m.coverage.is_nan());
}

/// Finding 6: the merge bound is the union of the valid candidates in time, and no strategy exceeds it.
#[test]
fn merge_bound_is_the_arrived_union() {
    use arrival_model::schedule::Kind;
    use sidecar_streaming::bound::merge_upper_bound;
    use sidecar_streaming::types::*;
    let set = make_set(&merge_invalid_params(), 20, 4);
    let mut backend = sim(SMOOTH);
    for slot in 0..20 {
        let covs: Vec<(u64, Coverage, bool)> = set.slots[slot as usize]
            .arrivals
            .iter()
            .filter(|a| a.kind == Kind::Aggregate)
            .map(|a| {
                let e = &set.pool[a.id as usize];
                let mut c = Coverage::empty();
                e.signer_ids().for_each(|s| c.insert(SUBNET_SIZE * e.subnet as u32 + s));
                (a.t_us, c, a.valid)
            })
            .collect();
        let refs: Vec<(u64, &Coverage, bool)> = covs.iter().map(|c| (c.0, &c.1, c.2)).collect();
        let d = 4_000_000;
        let bound = merge_upper_bound(&refs, d);
        for s in ["best_per_subnet_late", "fill_16", "two_round", "union_chain"] {
            let (_, m) = run(&mut backend, &set, slot, d, s, None);
            assert!(m.got <= bound, "{s}: got {} above the union {bound}", m.got);
        }
        // It is the metric's `arrived`.
        assert_eq!(run(&mut backend, &set, slot, d, "fill_16", None).1.arrived, bound);
    }
}
