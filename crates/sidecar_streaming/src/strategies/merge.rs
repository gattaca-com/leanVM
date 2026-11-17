//! Merge strategies (STRATEGIES.md 3.3): one root from overlapping candidate aggregates by `D_merge`.
//!
//! `Core` runs the deadline rule and the verify policy; a `Plan` says which children go in a job.
//! The last job starts at the latest moment it still ends by `D`. If it cannot take every chosen child,
//! the child with the smallest gain is left out until it fits. A root made by an earlier job is kept
//! as the fallback and as a child of the last job.

use super::{Params, with_margin};
use crate::forecast::Forecast;
use crate::select::select_exact;
use crate::strategy::{Ctx, Strategy};
use crate::types::*;
use std::rc::Rc;

/// Time kept free before `D`.
const MARGIN: Micros = 2_000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Policy {
    /// Verify every candidate on arrival, while the lane is idle.
    AllIdle,
    /// Verify only the children a job will take, just before it.
    Selected,
    /// Verify inside the job (`VERIFY_CHILDREN`); a bad child fails the job.
    Inline,
}

impl Policy {
    pub fn parse(s: &str) -> Result<Policy, String> {
        match s {
            "all_idle" => Ok(Policy::AllIdle),
            "selected" => Ok(Policy::Selected),
            "inline" => Ok(Policy::Inline),
            _ => Err(format!("unknown verify policy `{s}`")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Policy::AllIdle => "all_idle",
            Policy::Selected => "selected",
            Policy::Inline => "inline",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum St {
    Unverified,
    Verifying,
    Good,
    Bad,
}

pub struct Cand {
    pub id: ProofId,
    pub subnet: u8,
    pub claims: u32,
    st: St,
}

/// What a plan sees.
pub struct State<'a> {
    pub now: Micros,
    pub avail: &'a [&'a Cand],
    /// The kept root, if an earlier job made one: its coverage.
    pub kept: Option<&'a Coverage>,
    pub seen: u32,
}

pub trait Plan {
    fn name(&self) -> String;
    /// The children of a job, at most `budget`. For the last job these are what the job will take.
    fn pick(&self, st: &State, ctx: &dyn Ctx, budget: usize) -> Vec<ProofId>;
    /// An earlier job to run now, or None.
    fn early(&mut self, _st: &State, _ctx: &dyn Ctx, _budget: usize) -> Option<Vec<ProofId>> {
        None
    }
    /// A launch time of the last job other than the latest one that fits.
    fn go_time(&mut self, _st: &State, _ctx: &dyn Ctx, _latest: Micros) -> Option<Micros> {
        None
    }
}

pub struct MergeStrategy<P: Plan> {
    plan: P,
    policy: Policy,
    cands: Vec<Cand>,
    kept: Option<ProofId>,
    busy: bool,
    last_started: bool,
    published: bool,
    seen: u32,
}

impl<P: Plan> MergeStrategy<P> {
    pub fn new(plan: P, policy: Policy) -> Self {
        Self { plan, policy, cands: Vec::new(), kept: None, busy: false, last_started: false, published: false, seen: 0 }
    }

    fn set_state(&mut self, id: ProofId, st: St) {
        if let Some(c) = self.cands.iter_mut().find(|c| c.id == id) {
            c.st = st;
        }
    }

    /// The job's shape if its children are `children` plus the kept root, charging verification of unverified ones.
    fn shape(&self, ctx: &dyn Ctx, children: &[ProofId]) -> JobShape {
        let f = ctx.slot().final_rate;
        let mut shape = JobShape { children: Vec::new(), new_sigs: 0, rate: f };
        for &c in self.kept.iter().chain(children) {
            let p = ctx.proof(c);
            shape.children.push(ChildShape { claims: p.coverage.len(), rate: p.rate, verify: !p.verified, leaves: p.leaves });
        }
        shape
    }

    fn launch(&mut self, ctx: &mut dyn Ctx, children: Vec<ProofId>, last: bool) {
        // Under `Selected`, verify the unverified children first, one per step.
        if self.policy == Policy::Selected
            && let Some(&c) = children.iter().find(|&&c| !ctx.proof(c).verified)
        {
            self.set_state(c, St::Verifying);
            ctx.verify(0, c);
            self.busy = true;
            return;
        }
        // The kept root stays until a job succeeds: a failed job must not lose the fallback.
        let all: Vec<ProofId> = self.kept.into_iter().chain(children).collect();
        ctx.prove(0, Job { children: all, signatures: Vec::new(), rate: ctx.slot().final_rate });
        self.busy = true;
        self.last_started = last;
    }

    fn step(&mut self, ctx: &mut dyn Ctx) {
        if self.published || self.busy {
            return;
        }
        let (now, d, f) = (ctx.now(), ctx.slot().deadline, ctx.slot().final_rate);
        if now >= d {
            return;
        }
        let max_children = ctx.slot().caps.max_children;
        let budget = max_children - usize::from(self.kept.is_some());
        let avail: Vec<&Cand> = self
            .cands
            .iter()
            .filter(|c| match self.policy {
                Policy::AllIdle => c.st == St::Good,
                _ => c.st != St::Bad && c.st != St::Verifying,
            })
            .collect();
        let kept_cov = self.kept.map(|k| ctx.proof(k).coverage.clone());
        let st = State { now, avail: &avail, kept: kept_cov.as_ref(), seen: self.seen };
        let picked = self.plan.pick(&st, ctx, budget);
        let mut chosen = picked.clone();
        // Fit: leave out the child with the smallest gain until the job ends by `D`.
        let base = kept_cov.clone().unwrap_or_default();
        let est = |s: &Self, chosen: &[ProofId]| with_margin(ctx.cost(&s.shape(ctx, chosen)));
        while !chosen.is_empty() && now + est(self, &chosen) + MARGIN > d {
            let worst = (0..chosen.len())
                .min_by_key(|&i| ctx.proof(chosen[i]).coverage.union_len(&base) - base.len())
                .unwrap();
            chosen.remove(worst);
        }
        let latest = d.saturating_sub(est(self, &chosen) + MARGIN);
        let go = self.plan.go_time(&st, ctx, latest).unwrap_or(latest).min(latest);
        let nothing_new = chosen.is_empty();
        if now >= go {
            if nothing_new {
                if let Some(k) = self.kept.filter(|&k| ctx.proof(k).rate >= f) {
                    ctx.publish(k);
                    self.published = true;
                }
                return;
            }
            self.launch(ctx, chosen, true);
            return;
        }
        ctx.set_timer(go, go);
        // Before that, an earlier job, or verification while the lane waits.
        if let Some(children) = self.plan.early(&st, ctx, budget) {
            // A last job must still fit after this round, with the new root and every other child the plan would take.
            let after = {
                let round = self.shape(ctx, &children);
                let mut merged = kept_cov.clone().unwrap_or_default();
                children.iter().for_each(|&c| merged.union_with(&ctx.proof(c).coverage));
                let mut next = JobShape { children: vec![ChildShape { claims: merged.len(), rate: f, verify: false, leaves: 1 }], new_sigs: 0, rate: f };
                for &c in picked.iter().filter(|c| !children.contains(c)) {
                    let p = ctx.proof(c);
                    next.children.push(ChildShape { claims: p.coverage.len(), rate: p.rate, verify: !p.verified, leaves: p.leaves });
                }
                with_margin(ctx.cost(&round)) + with_margin(ctx.cost(&next))
            };
            if !children.is_empty() && now + after + MARGIN <= d {
                self.launch(ctx, children, false);
                return;
            }
        }
        if self.policy == Policy::AllIdle
            && let Some(c) = self.cands.iter().find(|c| c.st == St::Unverified).map(|c| c.id)
        {
            self.set_state(c, St::Verifying);
            ctx.verify(0, c);
            self.busy = true;
        }
    }
}

impl<P: Plan> Strategy for MergeStrategy<P> {
    fn name(&self) -> String {
        format!("{};verify={}", self.plan.name(), self.policy.name())
    }

    fn on_event(&mut self, event: Event, ctx: &mut dyn Ctx) {
        match event {
            Event::SlotStart | Event::Signature(_) => {}
            Event::Aggregate(id) => {
                let p = ctx.proof(id);
                self.cands.push(Cand { id, subnet: p.subnet.unwrap_or(0), claims: p.coverage.len(), st: St::Unverified });
                self.seen += 1;
                self.step(ctx);
            }
            Event::Verified { proof, ok, .. } => {
                self.busy = false;
                self.set_state(proof, if ok { St::Good } else { St::Bad });
                self.step(ctx);
            }
            Event::Proved { result, .. } => {
                self.busy = false;
                match result {
                    Ok(id) => {
                        self.kept = Some(id);
                        if self.last_started {
                            ctx.publish(id);
                            self.published = true;
                            return;
                        }
                    }
                    Err(Failure::InvalidChild(c)) => {
                        self.set_state(c, St::Bad);
                    }
                    Err(_) => {}
                }
                self.step(ctx);
            }
            Event::Timer(_) => self.step(ctx),
        }
    }
}

// ---------------------------------------------------------------------------------------------

fn pairs<'a>(st: &'a State, ctx: &'a dyn Ctx) -> Vec<(ProofId, u8, &'a Coverage)> {
    st.avail.iter().map(|c| (c.id, c.subnet, &ctx.proof(c.id).coverage)).collect()
}

/// The largest candidate of each subnet.
fn best_per_subnet(st: &State, budget: usize, exclude: Option<&Coverage>) -> Vec<ProofId> {
    let mut best: Vec<&Cand> = Vec::new();
    for c in st.avail {
        match best.iter_mut().find(|b| b.subnet == c.subnet) {
            Some(b) if b.claims >= c.claims => {}
            Some(b) => *b = c,
            None => best.push(c),
        }
    }
    let _ = exclude;
    best.sort_by_key(|c| std::cmp::Reverse(c.claims));
    best.into_iter().take(budget).map(|c| c.id).collect()
}

/// M0: at the last moment, one RECURSE of the largest candidate of each subnet present.
pub struct BestPerSubnetLate;

impl Plan for BestPerSubnetLate {
    fn name(&self) -> String {
        "best_per_subnet_late".into()
    }
    fn pick(&self, st: &State, _: &dyn Ctx, budget: usize) -> Vec<ProofId> {
        best_per_subnet(st, budget, None)
    }
}

/// M1: M0 with the selection of 3.2 at the full child budget: slots of missing subnets go to overlapping candidates.
pub struct Fill16;

impl Plan for Fill16 {
    fn name(&self) -> String {
        "fill_16".into()
    }
    fn pick(&self, st: &State, ctx: &dyn Ctx, budget: usize) -> Vec<ProofId> {
        let base = st.kept.cloned().unwrap_or_default();
        select_exact(&pairs(st, ctx), &base, budget).0
    }
}

/// M2: round 1 at `t1` (the best per subnet present); round 2 at the last moment (the kept root plus the best extras).
pub struct TwoRound {
    t1_frac: f64,
    done: bool,
}

impl TwoRound {
    pub fn new(p: &Params) -> Self {
        Self { t1_frac: p.f("t1_frac", 0.6), done: false }
    }
}

impl Plan for TwoRound {
    fn name(&self) -> String {
        format!("two_round:t1_frac={}", self.t1_frac)
    }
    fn pick(&self, st: &State, ctx: &dyn Ctx, budget: usize) -> Vec<ProofId> {
        let base = st.kept.cloned().unwrap_or_default();
        select_exact(&pairs(st, ctx), &base, budget).0
    }
    fn early(&mut self, st: &State, ctx: &dyn Ctx, budget: usize) -> Option<Vec<ProofId>> {
        let t1 = (ctx.slot().deadline as f64 * self.t1_frac) as Micros;
        if self.done || st.now < t1 || st.kept.is_some() || st.avail.is_empty() {
            return None;
        }
        self.done = true;
        Some(best_per_subnet(st, budget, None))
    }
}

/// M3: a round with up to 15 children when, after them, other candidates still add at least `g_min` claims (more than a job can take). The last job takes the kept root and the rest.
pub struct UnionChain {
    g_min: u32,
}

impl UnionChain {
    pub fn new(p: &Params) -> Self {
        Self { g_min: p.f("g_min", 100.0) as u32 }
    }
}

impl Plan for UnionChain {
    fn name(&self) -> String {
        format!("union_chain:g_min={}", self.g_min)
    }
    fn pick(&self, st: &State, ctx: &dyn Ctx, budget: usize) -> Vec<ProofId> {
        let base = st.kept.cloned().unwrap_or_default();
        select_exact(&pairs(st, ctx), &base, budget).0
    }
    fn early(&mut self, st: &State, ctx: &dyn Ctx, budget: usize) -> Option<Vec<ProofId>> {
        // A round is worth it only if what the last job could take is not all that is worth taking:
        // after this round's children, other candidates still add `g_min` claims.
        let base = st.kept.cloned().unwrap_or_default();
        let all = pairs(st, ctx);
        let (ids, _) = select_exact(&all, &base, budget);
        let mut merged = base;
        ids.iter().for_each(|&c| merged.union_with(&ctx.proof(c).coverage));
        let rest: Vec<_> = all.into_iter().filter(|c| !ids.contains(&c.0)).collect();
        let (_, leftover) = select_exact(&rest, &merged, budget);
        (leftover >= self.g_min && !ids.is_empty()).then_some(ids)
    }
}

/// M4: forecast the candidates still to come; wait for one only while the expected claims it brings
/// still fit a job that ends by `D`.
pub struct PlanMerge {
    forecast: Rc<Forecast>,
}

impl PlanMerge {
    pub fn new(forecast: Rc<Forecast>) -> Self {
        Self { forecast }
    }
}

impl Plan for PlanMerge {
    fn name(&self) -> String {
        "plan_merge".into()
    }
    fn pick(&self, st: &State, ctx: &dyn Ctx, budget: usize) -> Vec<ProofId> {
        let base = st.kept.cloned().unwrap_or_default();
        select_exact(&pairs(st, ctx), &base, budget).0
    }
    fn go_time(&mut self, st: &State, ctx: &dyn Ctx, latest: Micros) -> Option<Micros> {
        let (f, d) = (ctx.slot().final_rate, ctx.slot().deadline);
        let rest = self.forecast.remaining(st.now, st.seen);
        let have = best_per_subnet(st, 16, None);
        let mean_claims = if st.avail.is_empty() { 1500.0 } else { st.avail.iter().map(|c| c.claims as f64).sum::<f64>() / st.avail.len() as f64 };
        let present: f64 = have.iter().map(|&c| ctx.proof(c).coverage.len() as f64).sum();
        let job = |n: usize| with_margin(ctx.cost(&JobShape { children: vec![ChildShape { claims: mean_claims as u32, rate: f, verify: true, leaves: 1 }; n], new_sigs: 0, rate: f }));
        // Candidate launch times: now, and each forecast arrival. Launching at `t` takes what has arrived by `t`.
        let mut best = (present, st.now);
        for (i, &t) in rest.iter().enumerate() {
            let n = (have.len() + i + 1).min(16);
            if t + job(n) + MARGIN > d {
                break;
            }
            let value = present + (i + 1) as f64 * mean_claims;
            if value > best.0 {
                best = (value, t);
            }
        }
        Some(best.1.min(latest))
    }
}
