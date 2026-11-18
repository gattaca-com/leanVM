//! What a strategy believes about job durations on hardware it does not know.
//!
//! A backend's cost table is the simulator's truth (`cost.rs`). A strategy never reads it. It asks a
//! [`CostEstimator`], which learns from the durations of the jobs this sidecar finished: one line
//! `fixed + slope * new_signatures` per op class and rate, fitted over the last [`KEEP`] jobs.
//! The line is monotone by construction, so "the most that fits" can be found by bisection.
//!
//! The estimate is the line times the worst under-estimate seen in the kept jobs, plus a floor.
//! Before any job has finished it is a conservative prior.

use crate::types::*;
use std::collections::{HashMap, VecDeque};

/// Where the strategy's cost numbers come from (`--strategy-cost`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CostKind {
    /// The online-fitted lines of this module. The default.
    #[default]
    Calibrated,
    /// The backend's own table, as before the audit. Kept to compare against.
    Table,
}

impl CostKind {
    pub fn parse(text: &str) -> Result<CostKind, String> {
        match text {
            "calibrated" => Ok(CostKind::Calibrated),
            "table" => Ok(CostKind::Table),
            _ => Err(format!("--strategy-cost: `{text}` is not calibrated or table")),
        }
    }

    /// The estimator for one run of `shape` slots. `None` means the table: the driver then asks the backend's model.
    /// The lines cover leaf jobs only; a merge job has many children and keeps the table.
    pub fn estimator(self, shape: Shape) -> Option<Box<dyn CostEstimator>> {
        match (self, shape) {
            (CostKind::Calibrated, Shape::Leaf) => Some(Box::new(Calibrated::new())),
            _ => None,
        }
    }
}

impl CostKind {
    /// The estimator of `estimator` with the state of a `dump`. `None` for the table.
    pub fn restore(self, shape: Shape, lines: &[String]) -> Result<Option<Box<dyn CostEstimator>>, String> {
        match (self, shape) {
            (CostKind::Calibrated, Shape::Leaf) => Ok(Some(Box::new(Calibrated::restore(lines)?))),
            _ => Ok(None),
        }
    }
}

/// The estimator of `CostKind::estimator`, lent to one slot.
pub fn lend(est: &mut Option<Box<dyn CostEstimator>>) -> Option<&mut dyn CostEstimator> {
    est.as_mut().map(|b| -> &mut dyn CostEstimator { b.as_mut() })
}

/// A finished prove job, as the strategy may know it.
#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub children: usize,
    pub new_sigs: u32,
    pub rate: Rate,
    pub took_us: Micros,
}

pub trait CostEstimator {
    /// Predicted lane time of a job, for the strategy's planning.
    fn estimate(&self, shape: &JobShape) -> Micros;
    /// A prove job finished. Called for every completion, also one that ended after the deadline.
    fn observe(&mut self, obs: &Observation);
    /// Jobs learned from so far, for traces.
    fn jobs_seen(&self) -> usize {
        0
    }
    /// The learned state as text lines, for `--estimator-state`. `Calibrated::restore` reads it back.
    fn dump(&self) -> Vec<String> {
        Vec::new()
    }
}

/// The numbers a fresh estimator starts from, in one place.
pub struct Prior {
    /// AGGREGATE: `fixed + per_sig * n`, in microseconds, for a perf-like build.
    pub agg_fixed_us: f64,
    pub agg_per_sig_us: f64,
    pub ext_fixed_us: f64,
    pub ext_per_sig_us: f64,
    /// A conversion: no new signatures.
    pub conv_us: f64,
    /// The prior is this many times the perf-like line, so that it sits above every build measured.
    pub scale: f64,
    /// Rate 1 costs this much of rate 2 for the same job. Measured 0.77 to 0.91 over the perf and main tables;
    /// the top is taken so that a class never seen at rate 1 is not under-priced.
    pub rate1_over_rate2: f64,
}

pub const PRIOR: Prior = Prior { agg_fixed_us: 40_000.0, agg_per_sig_us: 750.0, ext_fixed_us: 120_000.0, ext_per_sig_us: 750.0, conv_us: 125_000.0, scale: 2.5, rate1_over_rate2: 0.9 };

/// What a job costs at `rate`, relative to rate 2, on the prior's scale.
fn rate_weight(rate: Rate) -> f64 {
    if rate == 1 { PRIOR.rate1_over_rate2 } else { 1.0 }
}

/// Jobs kept per (class, rate).
pub const KEEP: usize = 64;
/// A slope needs this many jobs, the largest at least [`SPREAD`] times the smallest plus [`SPREAD_ADD`].
pub const MIN_FOR_SLOPE: usize = 6;
const SPREAD: f64 = 2.0;
const SPREAD_ADD: f64 = 50.0;
/// The floor on the margin: this much above the worst ratio, plus [`FEW`] divided by the job count.
pub const PAD: f64 = 0.03;
pub const FEW: f64 = 0.25;
/// Added to every estimate.
pub const ADD_US: f64 = 2_000.0;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Class {
    Aggregate,
    Extend,
    Convert,
}

fn class(children: usize, new_sigs: u32) -> Class {
    if children == 0 {
        Class::Aggregate
    } else if new_sigs > 0 {
        Class::Extend
    } else {
        Class::Convert
    }
}

fn prior(c: Class, n: f64) -> f64 {
    let p = &PRIOR;
    let unit = match c {
        Class::Aggregate => p.agg_fixed_us + p.agg_per_sig_us * n,
        Class::Extend => p.ext_fixed_us + p.ext_per_sig_us * n,
        Class::Convert => p.conv_us,
    };
    unit * p.scale
}

/// The line of one (class, rate) and its margin.
#[derive(Clone, Copy, Debug)]
struct Fit {
    fixed: f64,
    slope: f64,
    /// The worst measured over fitted among the kept jobs, at least 1.
    worst: f64,
    jobs: usize,
}

/// The line through `obs` (pairs of `n` and microseconds): least squares with both terms at or above zero, or the
/// prior's shape scaled by the median ratio when the jobs do not span enough sizes.
fn line(c: Class, obs: &VecDeque<(f64, f64)>) -> (f64, f64) {
    let (lo, hi) = obs.iter().fold((f64::MAX, 0.0f64), |(l, h), o| (l.min(o.0), h.max(o.0)));
    let k = obs.len() as f64;
    let (sx, sy) = (obs.iter().map(|o| o.0).sum::<f64>(), obs.iter().map(|o| o.1).sum::<f64>());
    let (sxx, sxy) = (obs.iter().map(|o| o.0 * o.0).sum::<f64>(), obs.iter().map(|o| o.0 * o.1).sum::<f64>());
    let spread = k * sxx - sx * sx;
    if c == Class::Convert || obs.len() < MIN_FOR_SLOPE || hi < SPREAD * lo.max(1.0) + SPREAD_ADD || spread <= 0.0 {
        let mut r: Vec<f64> = obs.iter().map(|o| o.1 / prior(c, o.0)).collect();
        r.sort_by(f64::total_cmp);
        let m = r[r.len() / 2];
        return (prior(c, 0.0) * m, (prior(c, 1.0) - prior(c, 0.0)) * m);
    }
    let mut slope = (k * sxy - sx * sy) / spread;
    let mut fixed = (sy - slope * sx) / k;
    if fixed < 0.0 {
        fixed = 0.0;
        slope = sxy / sxx;
    }
    if slope < 0.0 {
        slope = 0.0;
        fixed = sy / k;
    }
    (fixed, slope)
}

/// The strategy's own cost model. State lives as long as the value: keep one across slots.
#[derive(Clone, Debug, Default)]
pub struct Calibrated {
    obs: HashMap<(Class, Rate), VecDeque<(f64, f64)>>,
    fit: HashMap<(Class, Rate), Fit>,
}

impl Calibrated {
    pub fn new() -> Self {
        Self::default()
    }

    /// Finished jobs seen, over all classes (capped per class by [`KEEP`]).
    pub fn jobs_kept(&self) -> usize {
        self.obs.values().map(VecDeque::len).sum()
    }

    /// The base time of a job of class `c` and `n` new signatures at `rate`, with the fit whose margin it carries.
    /// A class seen at this rate uses its own line. Otherwise it is derived from a line that was seen:
    ///
    /// - a conversion is the EXTEND line at `n = 0`, its fixed term;
    /// - another rate of the same class is scaled by the prior's rate ratio, taking the fit with most jobs.
    ///
    /// `None`: nothing related was seen, the prior applies.
    fn derive(&self, c: Class, rate: Rate, n: f64) -> Option<(f64, Fit)> {
        if let Some(f) = self.fit.get(&(c, rate)) {
            return Some((f.fixed + f.slope * n, *f));
        }
        if c == Class::Convert
            && let Some(d) = self.derive(Class::Extend, rate, 0.0)
        {
            return Some(d);
        }
        let (other, f) = self.fit.iter().filter(|((k, r), _)| *k == c && *r != rate).map(|((_, r), f)| (*r, f)).max_by_key(|(r, f)| (f.jobs, *r))?;
        Some(((f.fixed + f.slope * n) * rate_weight(rate) / rate_weight(other), *f))
    }
}

impl CostEstimator for Calibrated {
    fn jobs_seen(&self) -> usize {
        self.jobs_kept()
    }

    fn estimate(&self, shape: &JobShape) -> Micros {
        let c = class(shape.children.len(), shape.new_sigs);
        let n = shape.new_sigs as f64;
        let (base, worst, pad) = match self.derive(c, shape.rate, n) {
            Some((base, f)) => (base, f.worst, PAD + FEW / f.jobs as f64),
            None => (prior(c, n), 1.0, PAD),
        };
        (base * worst * (1.0 + pad) + ADD_US).round() as Micros
    }

    fn observe(&mut self, o: &Observation) {
        let c = class(o.children, o.new_sigs);
        let v = self.obs.entry((c, o.rate)).or_default();
        v.push_back((o.new_sigs as f64, o.took_us as f64));
        if v.len() > KEEP {
            v.pop_front();
        }
        self.refit(c, o.rate);
    }

    fn dump(&self) -> Vec<String> {
        let mut keys: Vec<(Class, Rate)> = self.obs.keys().copied().collect();
        keys.sort_by_key(|(c, r)| (c.letter(), *r));
        keys.iter().flat_map(|k| self.obs[k].iter().map(move |(n, us)| format!("{} {} {} {}", k.0.letter(), k.1, n, us))).collect()
    }
}

impl Class {
    fn letter(self) -> char {
        match self {
            Class::Aggregate => 'A',
            Class::Extend => 'E',
            Class::Convert => 'C',
        }
    }

    fn from_letter(s: &str) -> Option<Class> {
        match s {
            "A" => Some(Class::Aggregate),
            "E" => Some(Class::Extend),
            "C" => Some(Class::Convert),
            _ => None,
        }
    }
}

impl Calibrated {
    /// Fit the line and margin of one (class, rate) from its kept jobs.
    fn refit(&mut self, c: Class, rate: Rate) {
        let v = &self.obs[&(c, rate)];
        let (fixed, slope) = line(c, v);
        let worst = v.iter().map(|ob| ob.1 / (fixed + slope * ob.0).max(1.0)).fold(1.0, f64::max);
        self.fit.insert((c, rate), Fit { fixed, slope, worst, jobs: v.len() });
    }

    /// The estimator a `dump` came from. Every line is checked: a bad line is an error, not a skipped job.
    pub fn restore(lines: &[String]) -> Result<Calibrated, String> {
        let mut c = Calibrated::new();
        for l in lines {
            let f: Vec<&str> = l.split_whitespace().collect();
            let bad = || format!("estimator state: bad line `{l}`");
            if f.len() != 4 {
                return Err(bad());
            }
            let class = Class::from_letter(f[0]).ok_or_else(bad)?;
            let rate: Rate = f[1].parse().map_err(|_| bad())?;
            let (n, us): (f64, f64) = (f[2].parse().map_err(|_| bad())?, f[3].parse().map_err(|_| bad())?);
            if !(n.is_finite() && us.is_finite() && n >= 0.0 && us > 0.0) {
                return Err(bad());
            }
            let v = c.obs.entry((class, rate)).or_default();
            v.push_back((n, us));
            if v.len() > KEEP {
                return Err(format!("estimator state: more than {KEEP} jobs of one class and rate"));
            }
        }
        let keys: Vec<(Class, Rate)> = c.obs.keys().copied().collect();
        keys.into_iter().for_each(|(k, r)| c.refit(k, r));
        Ok(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(children: usize, n: u32, us: Micros) -> Observation {
        Observation { children, new_sigs: n, rate: 2, took_us: us }
    }

    #[test]
    fn prior_is_conservative_and_monotone() {
        let c = Calibrated::new();
        let mut prev = 0;
        for n in 1..=2048 {
            let t = c.estimate(&JobShape::aggregate(n, 2));
            assert!(t >= prev);
            prev = t;
        }
        // About 2.5 times 100 ms + 1.9 ms per signature.
        let t = c.estimate(&JobShape::aggregate(1000, 2));
        assert!((2_000_000..2_900_000).contains(&t), "{t}");
    }

    #[test]
    fn fit_recovers_a_line_and_covers_the_worst_case() {
        let mut c = Calibrated::new();
        for (i, n) in [20u32, 60, 150, 300, 500, 800, 1100, 1500].into_iter().enumerate() {
            let truth = 30_000.0 + 900.0 * n as f64;
            // Up to 8 % jitter.
            let jitter = 1.0 + 0.08 * ((i % 3) as f64 - 1.0).abs();
            c.observe(&obs(0, n, (truth * jitter) as Micros));
        }
        for n in [100u32, 400, 1000, 1500] {
            let truth = 30_000.0 + 900.0 * n as f64;
            let e = c.estimate(&JobShape::aggregate(n, 2)) as f64;
            assert!(e >= truth && e <= truth * 1.35, "n {n}: estimate {e}, truth {truth}");
        }
    }

    #[test]
    fn estimate_is_monotone_after_noisy_observations() {
        let mut c = Calibrated::new();
        // A cost that is not monotone: a hump at 1564.
        for n in [100u32, 400, 900, 1300, 1560, 1564, 1596, 200, 700] {
            let us = if n == 1564 { 1_211_000 } else { 100_000 + 700 * n as u64 };
            c.observe(&obs(0, n, us));
        }
        let mut prev = 0;
        for n in 1..=2048 {
            let t = c.estimate(&JobShape::aggregate(n, 2));
            assert!(t >= prev, "n {n}");
            prev = t;
        }
    }

    #[test]
    fn classes_and_rates_do_not_mix() {
        let mut c = Calibrated::new();
        for n in [10u32, 50, 100, 300, 600, 900, 1200] {
            c.observe(&obs(0, n, 10_000 + 100 * n as u64));
        }
        let a = c.estimate(&JobShape::aggregate(500, 2));
        assert!(a < 100_000, "{a}");
        // No EXTEND seen: still the prior, far above.
        assert!(c.estimate(&JobShape::extend(300, 2, 500, 2)) > 1_000_000);
        // Another rate of a seen class: the seen line times the prior's rate ratio, not the prior.
        let r1 = c.estimate(&JobShape::aggregate(500, 1));
        assert!(r1 < a && r1 > a * 8 / 10, "{r1} against {a}");
    }

    #[test]
    fn an_unseen_class_is_derived_from_a_seen_one() {
        let mut c = Calibrated::new();
        // EXTEND at rate 2: 100 ms fixed, 600 us per signature.
        for n in [10u32, 50, 100, 300, 600, 900, 1200] {
            c.observe(&obs(1, n, 100_000 + 600 * n as u64));
        }
        // A conversion costs the EXTEND's fixed term, not the 2.5 times prior.
        let conv = c.estimate(&JobShape::extend(800, 2, 0, 2));
        assert!((100_000..140_000).contains(&conv), "{conv}");
        // The same at rate 1: the rate ratio of the prior applies.
        let conv1 = c.estimate(&JobShape::extend(800, 1, 0, 1));
        assert!(conv1 < conv && conv1 > conv * 8 / 10, "{conv1} against {conv}");
        // EXTEND at rate 1 follows rate 2 the same way.
        let e2 = c.estimate(&JobShape::extend(800, 2, 400, 2));
        let e1 = c.estimate(&JobShape::extend(800, 1, 400, 1));
        assert!(e1 < e2 && e1 > e2 * 8 / 10, "{e1} against {e2}");
        // Nothing of the AGGREGATE class was seen, and an EXTEND does not stand for it.
        assert!(c.estimate(&JobShape::aggregate(500, 2)) > 1_000_000);
    }

    #[test]
    fn window_forgets_old_jobs() {
        let mut c = Calibrated::new();
        for i in 0..200u32 {
            c.observe(&obs(0, 10 + i, 1_000_000));
        }
        assert_eq!(c.jobs_kept(), KEEP);
    }
}
