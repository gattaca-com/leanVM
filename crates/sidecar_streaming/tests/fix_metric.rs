//! Regression tests for the diagnostics of finding 15 in `design/REVIEW.md`.
mod common;
use common::*;
use sidecar_streaming::metric::*;
use sidecar_streaming::types::*;

fn cov(claims: std::ops::Range<u32>) -> Coverage {
    let mut c = Coverage::empty();
    claims.for_each(|x| c.insert(x));
    c
}

#[test]
fn merge_cut_ignores_a_later_subset_candidate() {
    let rec = SlotRecord {
        shape: Shape::Merge,
        deadline: 1000,
        inputs: vec![
            InputRec { t: 100, claims: (0..10).collect(), valid: true },
            InputRec { t: 200, claims: (5..15).collect(), valid: true },
            // Arrives later and adds nothing: all its claims are in the proof.
            InputRec { t: 500, claims: (0..5).collect(), valid: true },
        ],
        published: Some((900, cov(0..15))),
        jobs: Vec::new(),
        sent_override: None,
    };
    let m = slot_metrics(&rec);
    assert_eq!((m.cut_us, m.staleness_us), (Some(200), Some(700)));
}

#[test]
fn verifying_a_used_candidate_is_not_waste() {
    let set = make_set(&merge_params(), 10, 2);
    let mut backend = sim(SMOOTH);
    for slot in 0..10 {
        let (rec, m) = run(&mut backend, &set, slot, 4_000_000, "fill_16:verify=all_idle", None);
        assert!(m.verify_us > 0);
        // Candidates that were verified and used are useful work; the others are waste.
        assert!(m.wasted_us < m.verify_us, "slot {slot}");
        let used = rec.jobs.iter().filter(|j| j.op == "VERIFY" && j.useful).count();
        assert!(used >= 10, "slot {slot}: {used} verifications of used candidates");
    }
}
