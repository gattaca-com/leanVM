//! "The most that fits" must be right for any cost function the strategy sees (AUDIT.md finding 2).
mod common;
use common::*;
use sidecar_streaming::backend::Backend;
use sidecar_streaming::calib::{CostEstimator, Observation};
use sidecar_streaming::driver::{RunSpec, run_slot_with};
use sidecar_streaming::metric::slot_metrics;
use sidecar_streaming::strategies::{Spec, with_margin};
use sidecar_streaming::strategies::leaf::{most_that_fits, steps};
use sidecar_streaming::types::*;

/// Perf `agg.2` as measured: 1560 votes take 1181 ms, 1564 take 1211 ms, 1596 take 1195 ms.
fn perf_agg2(n: usize) -> u64 {
    match n {
        0..=1560 => 1_181_000 - (1560 - n as u64) * 700,
        1561..=1564 => 1_211_000,
        _ => 1_195_000 - (1596u64.saturating_sub(n as u64)) * 100,
    }
}

/// The search the audit found, as it was: a bisection.
fn bisection(max: usize, fits: impl Fn(usize) -> bool) -> usize {
    let (mut lo, mut hi) = (0usize, max);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if fits(mid) { lo = mid } else { hi = mid - 1 }
    }
    lo
}

#[test]
fn the_1560_1564_1596_case_takes_every_vote() {
    // 1212 ms left: 1561..=1564 do not fit, 1565..=1596 do.
    let fits = |n: usize| with_margin(perf_agg2(n)) <= 1_212_000;
    assert!(!fits(1562) && fits(1560) && fits(1565) && fits(1596));
    // With 1586 waiting, the bisection probes 1562, fails, and settles at 1560.
    assert_eq!(bisection(1586, fits), 1560);
    assert_eq!(most_that_fits(1586, fits), 1586);
    assert_eq!(most_that_fits(1596, fits), 1596);
    // Nothing fits, and nothing waits.
    assert_eq!(most_that_fits(1596, |_| false), 0);
    assert_eq!(most_that_fits(0, |_| true), 0);
}

/// A cost with a hump of four sizes that a bisection probes.
struct Hump {
    from: usize,
}

impl CostEstimator for Hump {
    fn estimate(&self, shape: &JobShape) -> Micros {
        let n = shape.new_sigs as usize;
        20_000 + 800 * n as u64 + if (self.from..self.from + 4).contains(&n) { 30_000 } else { 0 }
    }
    fn observe(&mut self, _: &Observation) {}
}

/// The one-shot, whose last job takes "the most that fits", must take every vote when only a hump is in the way.
#[test]
fn the_one_shot_takes_every_vote_past_a_hump() {
    let set = make_set(&leaf_params(), 1, 3);
    let w = set.slots[0].arrivals.iter().filter(|a| a.subnet == 0 && a.view == 0).count();
    // The probes of a bisection when everything fits: the hump sits on the fifth from last.
    let mut probes = Vec::new();
    let (mut lo, hi) = (0usize, w);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        probes.push(mid);
        lo = mid;
    }
    let from = probes[probes.len() - 5];
    assert!(from + 4 <= w);
    let mut b = sim(SMOOTH);
    let mut s = Spec::parse("one_shot_late").unwrap().make(None).unwrap();
    let rs = RunSpec { set: &set, slot: 0, subnet: 0, view: 0, deadline: 3_000_000, caps: b.caps(), grace: 0 };
    let rec = run_slot_with(&mut b, s.as_mut(), &rs, Some(&mut Hump { from }));
    let m = slot_metrics(&rec);
    assert_eq!(rec.jobs.len(), 1);
    assert_eq!(m.got, w as u32, "the hump at {from}..{} cost the strategy votes", from + 4);
}

#[test]
fn a_hump_is_not_a_step_and_a_step_is() {
    // A step of 30 % at 100 that stays; a hump at 40.
    let cost = |n: u32| -> f64 {
        let base = 1000.0 + 10.0 * n as f64;
        base * if n >= 100 { 1.3 } else { 1.0 } * if n == 40 { 1.5 } else { 1.0 }
    };
    assert_eq!(steps(cost, 2, 200), vec![100]);
    // Looking only at sizes below the step, the step is not there.
    assert_eq!(steps(cost, 2, 99), Vec::<u32>::new());
    // A smooth cost has no step.
    assert!(steps(|n| 1000.0 + 10.0 * n as f64, 2, 2048).is_empty());
}
