//! `plan_cover` plans again after a job, instead of repeating the step of the plan that started it (AUDIT.md finding 6).
mod common;
use common::*;

#[test]
fn plan_cover_does_not_repeat_the_step_it_just_ran() {
    let train = make_set(&leaf_params(), 4, 11);
    let set = make_set(&leaf_params(), 12, 12);
    let mut backend = sim(SMOOTH);
    let (mut repeats, mut jobs) = (0, 0);
    for d in [2_000_000, 2_500_000] {
        for slot in 0..12 {
            let (rec, _) = run(&mut backend, &set, slot, d, "plan_cover", Some(&train));
            let sizes: Vec<u32> = rec.jobs.iter().filter(|j| j.op == "EXTEND").map(|j| j.new_sigs).collect();
            jobs += sizes.len();
            // A size off the planner's grid of 30 that comes twice in a row is a stale step.
            repeats += sizes.windows(2).filter(|w| w[0] == w[1] && w[0] % 30 != 0).count();
        }
    }
    eprintln!("{repeats} repeats in {jobs} extends");
    assert_eq!(repeats, 0, "{repeats} repeated steps in {jobs} extends");
}
