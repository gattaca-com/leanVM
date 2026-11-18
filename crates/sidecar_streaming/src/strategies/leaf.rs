//! Leaf strategies (STRATEGIES.md 2.2). Every one runs under `fit`, the deadline rule in `Core`:
//!
//! - the last job starts at the latest moment it still ends by `D`, with all that waits, or the largest part that fits;
//! - an earlier job is allowed only if a last job still fits after it.
//!
//! Time moves in ticks of 5 ms, and on every completion.

use super::{Params, with_margin};
use std::collections::VecDeque;
use crate::forecast::Forecast;
use crate::strategy::{Ctx, Strategy};
use crate::types::*;
use std::rc::Rc;

pub const TICK: Micros = 5_000;
const TICK_TOKEN: u64 = 1;

/// What the inner rule sees.
pub struct View {
    pub now: Micros,
    pub waiting: usize,
    pub seen: u32,
    pub kept_claims: u32,
    pub kept_rate: Rate,
}

pub enum Act {
    Wait,
    /// Prove the `n` oldest waiting signatures with the kept proof, at `rate`. `n = 0` converts the kept proof.
    Go { n: usize, rate: Rate },
}

pub trait Inner {
    fn name(&self) -> String;
    fn start(&mut self, _ctx: &dyn Ctx) {}
    fn act(&mut self, v: &View, ctx: &dyn Ctx) -> Act;
    /// True if waiting with votes in hand is the strategy itself, as in the one-shot and two-stage baselines.
    /// Every other strategy is kept from idling by `LeafStrategy::fill`.
    fn waits_by_design(&self) -> bool {
        false
    }
}

pub struct LeafStrategy<I: Inner> {
    inner: I,
    waiting: VecDeque<SigId>,
    seen: u32,
    known_total: Option<u32>,
    /// The kept proof and the signatures of the job in flight, to restore if it fails.
    in_flight: Option<(Option<ProofId>, Vec<SigId>)>,
    kept: Option<ProofId>,
    busy: bool,
    last_started: bool,
    published: bool,
}

impl<I: Inner> LeafStrategy<I> {
    pub fn new(inner: I) -> Self {
        Self { inner, waiting: VecDeque::new(), seen: 0, known_total: None, in_flight: None, kept: None, busy: false, last_started: false, published: false }
    }

    /// The strategy is told how many inputs will come in all, and may stop waiting when they have. A strategy does not
    /// know this by default: even with no vote missing, the aggregator cannot tell that the window has closed.
    pub fn with_known_total(mut self, total: u32) -> Self {
        self.known_total = Some(total);
        self
    }

    fn kept_info(&self, ctx: &dyn Ctx) -> (u32, Rate) {
        self.kept.map_or((0, ctx.slot().final_rate), |k| {
            let p = ctx.proof(k);
            (p.coverage.len(), p.rate)
        })
    }

    fn launch(&mut self, ctx: &mut dyn Ctx, n: usize, rate: Rate, last: bool) {
        let signatures: Vec<SigId> = self.waiting.drain(..n).collect();
        let kept = self.kept.take();
        self.in_flight = Some((kept, signatures.clone()));
        ctx.prove(0, Job { children: kept.into_iter().collect(), signatures, rate });
        self.busy = true;
        self.last_started = last;
    }

    fn publish(&mut self, ctx: &mut dyn Ctx, id: ProofId) {
        ctx.publish(id);
        self.published = true;
    }

    fn evaluate(&mut self, ctx: &mut dyn Ctx) {
        if self.published || self.busy {
            return;
        }
        let (s, d, f) = (ctx.now(), ctx.slot().deadline, ctx.slot().final_rate);
        if s >= d {
            return;
        }
        let w = self.waiting.len();
        let (kc, kr) = self.kept_info(ctx);
        let kept_final = self.kept.is_some() && kr >= f;
        let all_present = self.known_total.is_some_and(|n| self.seen >= n);
        if w == 0 && (self.kept.is_none() || kept_final) {
            if let Some(k) = self.kept.filter(|_| kept_final && (all_present || s + TICK >= d)) {
                self.publish(ctx, k);
            }
            return;
        }
        let cap = ctx.slot().caps.max_new_sigs as usize;
        let last_cost = |ctx: &dyn Ctx, n: usize| with_margin(ctx.cost(&JobShape::extend(kc, kr, n as u32, f)));
        if all_present || s + last_cost(ctx, w.min(cap)) + TICK > d {
            // The last job. If it cannot take everything, take the most that fits.
            let (mut lo, mut hi) = (0usize, w.min(cap));
            while lo < hi {
                let mid = (lo + hi).div_ceil(2);
                if s + last_cost(ctx, mid) <= d { lo = mid } else { hi = mid - 1 }
            }
            let n = lo;
            let convert = n == 0 && self.kept.is_some() && kr < f && s + last_cost(ctx, 0) <= d;
            if n > 0 || convert {
                self.launch(ctx, n, f, true);
            } else if let Some(k) = self.kept.filter(|_| kept_final) {
                self.publish(ctx, k);
            }
            return;
        }
        let view = View { now: s, waiting: w, seen: self.seen, kept_claims: kc, kept_rate: kr };
        if let Act::Go { n, rate } = self.inner.act(&view, ctx) {
            let n = n.min(w).min(cap);
            if n > 0 || self.kept.is_some() {
                if self.fits_before_last(ctx, (kc, kr), n, rate) {
                    self.launch(ctx, n, rate, false);
                    return;
                }
            }
        }
        self.fill(ctx);
    }

    /// True if an EXTEND of `n` waiting votes at `rate` and then a last job still end by `D`. The last job must fit
    /// with one more signature or, if this job is cheaper than the final rate, with a plain conversion. Take the dearer.
    fn fits_before_last(&self, ctx: &dyn Ctx, (kc, kr): (u32, Rate), n: usize, rate: Rate) -> bool {
        let (s, d, f) = (ctx.now(), ctx.slot().deadline, ctx.slot().final_rate);
        let last = ctx.cost(&JobShape::extend(kc + n as u32, rate, 1, f)).max(ctx.cost(&JobShape::extend(kc + n as u32, rate, 0, f)));
        s + with_margin(ctx.cost(&JobShape::extend(kc, kr, n as u32, rate))) + with_margin(last) <= d
    }

    /// No lane idles while votes wait and an EXTEND of at least one of them, then the last job, still fits before `D`.
    /// If the inner rule did not start a job, start the largest such EXTEND at the final rate. Strategies that wait by
    /// design are left alone.
    fn fill(&mut self, ctx: &mut dyn Ctx) {
        if self.inner.waits_by_design() || self.waiting.is_empty() {
            return;
        }
        let f = ctx.slot().final_rate;
        let (kc, kr) = self.kept_info(ctx);
        let (mut lo, mut hi) = (0usize, self.waiting.len().min(ctx.slot().caps.max_new_sigs as usize));
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if self.fits_before_last(ctx, (kc, kr), mid, f) { lo = mid } else { hi = mid - 1 }
        }
        if lo > 0 {
            self.launch(ctx, lo, f, false);
        }
    }
}

impl<I: Inner> Strategy for LeafStrategy<I> {
    fn name(&self) -> String {
        self.inner.name()
    }

    fn on_event(&mut self, event: Event, ctx: &mut dyn Ctx) {
        match event {
            Event::SlotStart => {
                self.inner.start(ctx);
                ctx.set_timer(TICK, TICK_TOKEN);
            }
            Event::Signature(s) => {
                self.waiting.push_back(s);
                self.seen += 1;
                if self.known_total.is_some_and(|n| self.seen >= n) {
                    self.evaluate(ctx);
                }
            }
            Event::Aggregate(_) | Event::Verified { .. } => {}
            Event::Proved { result, .. } => {
                self.busy = false;
                let launched = self.in_flight.take();
                match result {
                    Ok(id) => {
                        self.kept = Some(id);
                        if self.last_started {
                            self.publish(ctx, id);
                            return;
                        }
                    }
                    Err(_) => {
                        // A failed job changes nothing: the kept proof and the signatures it took are back.
                        if let Some((kept, signatures)) = launched {
                            self.kept = kept;
                            signatures.into_iter().rev().for_each(|s| self.waiting.push_front(s));
                        }
                        self.last_started = false;
                    }
                }
                self.evaluate(ctx);
            }
            Event::Timer(_) => {
                self.evaluate(ctx);
                if !self.published && ctx.now() < ctx.slot().deadline {
                    // The next point of the grid `k * TICK`, so that a late wake-up does not shift every later tick.
                    ctx.set_timer((ctx.now() / TICK + 1) * TICK, TICK_TOKEN);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------

/// A0: nothing before the last job.
pub struct OneShotLate;

impl Inner for OneShotLate {
    fn name(&self) -> String {
        "one_shot_late".into()
    }
    fn waits_by_design(&self) -> bool {
        true
    }
    fn act(&mut self, _: &View, _: &dyn Ctx) -> Act {
        Act::Wait
    }
}

/// A1: one AGGREGATE of all that waits at `t1`, then only the last job.
pub struct TwoStage {
    t1_frac: f64,
    t1: Micros,
    fired: bool,
}

impl TwoStage {
    pub fn new(p: &Params) -> Self {
        Self { t1_frac: p.f("t1_frac", 0.4), t1: 0, fired: false }
    }
}

impl Inner for TwoStage {
    fn name(&self) -> String {
        format!("two_stage:t1_frac={}", self.t1_frac)
    }
    fn waits_by_design(&self) -> bool {
        true
    }
    fn start(&mut self, ctx: &dyn Ctx) {
        self.t1 = (ctx.slot().deadline as f64 * self.t1_frac) as Micros;
    }
    fn act(&mut self, v: &View, ctx: &dyn Ctx) -> Act {
        if !self.fired && v.kept_claims == 0 && v.waiting > 0 && v.now >= self.t1 {
            self.fired = true;
            return Act::Go { n: v.waiting, rate: ctx.slot().final_rate };
        }
        Act::Wait
    }
}

/// A2: lane idle and `w >= m`: EXTEND all `w`.
pub struct MinBatch {
    m: usize,
}

impl MinBatch {
    pub fn new(p: &Params) -> Self {
        Self { m: p.f("m", 300.0) as usize }
    }
}

impl Inner for MinBatch {
    fn name(&self) -> String {
        format!("min_batch_fit:m={}", self.m)
    }
    fn act(&mut self, v: &View, ctx: &dyn Ctx) -> Act {
        if v.waiting >= self.m.max(1) { Act::Go { n: v.waiting, rate: ctx.slot().final_rate } } else { Act::Wait }
    }
}

/// A3: lane idle and `w >= b`: EXTEND exactly `b`.
pub struct FixedFit {
    b: usize,
}

impl FixedFit {
    pub fn new(p: &Params) -> Self {
        Self { b: p.f("b", 600.0) as usize }
    }
}

impl Inner for FixedFit {
    fn name(&self) -> String {
        format!("fixed_fit:b={}", self.b)
    }
    fn act(&mut self, v: &View, ctx: &dyn Ctx) -> Act {
        if v.waiting >= self.b.max(1) { Act::Go { n: self.b, rate: ctx.slot().final_rate } } else { Act::Wait }
    }
}

/// A4: lane idle and `w >= 1`: EXTEND all `w`.
pub struct Greedy;

impl Inner for Greedy {
    fn name(&self) -> String {
        "greedy_fit".into()
    }
    fn act(&mut self, v: &View, ctx: &dyn Ctx) -> Act {
        if v.waiting >= 1 { Act::Go { n: v.waiting, rate: ctx.slot().final_rate } } else { Act::Wait }
    }
}

/// A5: batches at the cheapest rate; the last job converts to the final rate.
pub struct CheapChain {
    m: usize,
    cheap: Rate,
}

impl CheapChain {
    pub fn new(p: &Params) -> Self {
        Self { m: p.f("m", 1.0) as usize, cheap: p.f("cheap", 1.0) as Rate }
    }
}

impl Inner for CheapChain {
    fn name(&self) -> String {
        format!("cheap_chain:m={};cheap={}", self.m, self.cheap)
    }
    fn act(&mut self, v: &View, _: &dyn Ctx) -> Act {
        if v.waiting >= self.m.max(1) { Act::Go { n: v.waiting, rate: self.cheap } } else { Act::Wait }
    }
}

/// A6: like A2, but a batch is cut just under the largest jump of the cost it crosses.
pub struct StepCutFit {
    m: usize,
}

impl StepCutFit {
    pub fn new(p: &Params) -> Self {
        Self { m: p.f("m", 150.0) as usize }
    }
}

impl Inner for StepCutFit {
    fn name(&self) -> String {
        format!("step_cut_fit:m={}", self.m)
    }
    fn act(&mut self, v: &View, ctx: &dyn Ctx) -> Act {
        let f = ctx.slot().final_rate;
        if v.waiting < self.m.max(1) {
            return Act::Wait;
        }
        let cost = |n: usize| ctx.cost(&JobShape::extend(v.kept_claims, v.kept_rate, n as u32, f)) as f64;
        // The largest n <= w at which the cost jumps by more than 12 % over n - 1.
        let jump = (self.m.max(2)..=v.waiting).rev().find(|&n| cost(n) > 1.12 * cost(n - 1));
        match jump {
            Some(j) if j > self.m.max(1) => Act::Go { n: j - 1, rate: f },
            _ => Act::Go { n: v.waiting, rate: f },
        }
    }
}

// ---------------------------------------------------------------------------------------------

/// A7: forecast the remaining arrivals, solve the chain DP for the largest prefix whose proof is ready
/// by `D`, run that plan's first job, replan every 10 ms.
pub struct PlanCover {
    forecast: Rc<Forecast>,
    grid: u32,
    cheap: Rate,
    jumps: Vec<u32>,
    ticks: u32,
    plan: Option<Act>,
    /// True while the last plan chose to wait for votes that have not arrived yet.
    deliberate: std::cell::Cell<bool>,
}

impl PlanCover {
    pub fn new(p: &Params, forecast: Rc<Forecast>) -> Self {
        Self { forecast, grid: p.f("grid", 30.0) as u32, cheap: p.f("cheap", 1.0) as Rate, jumps: Vec::new(), ticks: 0, plan: None, deliberate: Default::default() }
    }
}

#[derive(Clone, Copy)]
struct Node {
    t: f64,
    from: Option<(usize, usize)>,
    /// True when this node is reached by converting the proof of the same prefix.
    conv: bool,
}

impl Inner for PlanCover {
    fn name(&self) -> String {
        format!("plan_cover:grid={};cheap={}", self.grid, self.cheap)
    }

    /// The planner waits on purpose when its first job needs votes that have not come. It does not when no plan
    /// reaches a prefix by `D`: then the wrapper keeps the lane busy.
    fn waits_by_design(&self) -> bool {
        self.deliberate.get()
    }

    /// Put the model's jump positions on the planner's grid.
    fn start(&mut self, ctx: &dyn Ctx) {
        let f = ctx.slot().final_rate;
        let cap = ctx.slot().caps.max_new_sigs;
        let mut prev = ctx.cost(&JobShape::aggregate(1, f)) as f64;
        for n in 2..=cap {
            let c = ctx.cost(&JobShape::aggregate(n, f)) as f64;
            if c > 1.12 * prev {
                self.jumps.push(n - 1);
                self.jumps.push(n);
            }
            prev = c;
        }
    }

    fn act(&mut self, v: &View, ctx: &dyn Ctx) -> Act {
        self.ticks += 1;
        self.deliberate.set(false);
        // Between replans, repeat the last plan's first step if it is still possible.
        if self.ticks.is_multiple_of(2)
            && let Some(Act::Go { n, rate }) = self.plan.take()
            && n <= v.waiting
        {
            return Act::Go { n, rate };
        }
        let a = self.replan(v, ctx);
        self.plan = Some(match &a {
            Act::Wait => Act::Wait,
            Act::Go { n, rate } => Act::Go { n: *n, rate: *rate },
        });
        a
    }
}

impl PlanCover {
    fn replan(&self, v: &View, ctx: &dyn Ctx) -> Act {
        let (f, d) = (ctx.slot().final_rate, ctx.slot().deadline as f64);
        let cap = ctx.slot().caps.max_new_sigs;
        let kc = v.kept_claims;
        let present = kc + v.waiting as u32;
        let rest = self.forecast.remaining(v.now, v.seen);
        let total = present + rest.len() as u32;
        // Arrival time of the j-th claim: now for those that are here, forecast after.
        let arrival = |j: u32| if j <= present { v.now as f64 } else { rest[(j - present - 1) as usize] as f64 };
        let mut idx: Vec<u32> = (kc + 1..=total).filter(|j| (j - kc).is_multiple_of(self.grid)).collect();
        idx.extend(self.jumps.iter().copied().filter(|j| *j > kc && *j <= total));
        idx.extend([present, total].into_iter().filter(|j| *j > kc));
        idx.sort_unstable();
        idx.dedup();
        let rates: Vec<Rate> = if self.cheap < f { vec![self.cheap, f] } else { vec![f] };
        // Node 0 is the start: the kept proof (or nothing) at `now`.
        let mut pos = vec![kc];
        pos.extend(idx.iter().copied());
        let mut nodes: Vec<Vec<Node>> = vec![vec![Node { t: f64::INFINITY, from: None, conv: false }; rates.len()]; pos.len()];
        let start_rate = rates.iter().position(|&r| r == v.kept_rate);
        for (ri, _) in rates.iter().enumerate() {
            if kc > 0 && Some(ri) == start_rate {
                nodes[0][ri].t = v.now as f64;
            }
        }
        if kc == 0 {
            // No kept proof: the first job is an AGGREGATE, modelled as a step from a virtual start of every rate.
            for n in nodes[0].iter_mut() {
                n.t = v.now as f64;
            }
        }
        for j in 1..pos.len() {
            for (ri, &r) in rates.iter().enumerate() {
                let mut best = Node { t: f64::INFINITY, from: None, conv: false };
                for i in 0..j {
                    if pos[j] - pos[i] > cap {
                        continue;
                    }
                    for (rk_i, &rk) in rates.iter().enumerate() {
                        let ft = nodes[i][rk_i].t;
                        if !ft.is_finite() {
                            continue;
                        }
                        let shape = if pos[i] == 0 { JobShape::aggregate(pos[j] - pos[i], r) } else { JobShape::extend(pos[i], rk, pos[j] - pos[i], r) };
                        let t = ft.max(arrival(pos[j])) + with_margin(ctx.cost(&shape)) as f64;
                        if t < best.t {
                            best = Node { t, from: Some((i, rk_i)), conv: false };
                        }
                    }
                }
                nodes[j][ri] = best;
            }
            // Conversion to the final rate.
            let fi = rates.len() - 1;
            for ri in 0..fi {
                let t = nodes[j][ri].t + with_margin(ctx.cost(&JobShape::extend(pos[j], rates[ri], 0, f))) as f64;
                if t < nodes[j][fi].t {
                    nodes[j][fi] = Node { t, from: Some((j, ri)), conv: true };
                }
            }
        }
        let fi = rates.len() - 1;
        let Some(target) = (1..pos.len()).rev().find(|&j| nodes[j][fi].t <= d) else {
            return Act::Wait;
        };
        // Walk back to the first job: the first step that leaves the start node.
        let (mut j, mut r) = (target, fi);
        loop {
            let n = nodes[j][r];
            let (pj, pr) = n.from.expect("a reachable node has a predecessor");
            if !n.conv && pj == 0 {
                break;
            }
            (j, r) = (pj, pr);
        }
        if pos[j] > present {
            self.deliberate.set(true);
            return Act::Wait;
        }
        Act::Go { n: (pos[j] - kc) as usize, rate: rates[r] }
    }
}
