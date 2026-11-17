//! Regression tests for the leaf findings of `design/REVIEW.md`.
mod common;
use common::*;

/// Finding 7: with no vote missing, a strategy must not learn that the window has closed.
/// The last job starts as late as it fits, so the proof is published within a tick and the margin of `D`.
#[test]
fn leaf_strategies_do_not_know_the_total() {
    const TICK: u64 = 5_000;
    let set = make_set(&leaf_params(), 10, 8);
    let mut backend = sim(SMOOTH);
    // All 1500 votes are in by 1.2 s; D = 2.5 s leaves a long quiet stretch.
    let d = 2_500_000;
    for s in ["one_shot_late", "greedy_fit", "min_batch_fit:m=300", "cheap_chain"] {
        for slot in 0..10 {
            let (rec, m) = run(&mut backend, &set, slot, d, s, None);
            assert_eq!(m.got, 1500, "{s}");
            let t = m.published_us.unwrap();
            assert!(rec.inputs.iter().all(|i| i.t < t));
            assert!(t + TICK + 12_000 > d, "{s}: published at {t} us, long before D {d}");
        }
    }
}
