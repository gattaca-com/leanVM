//! The event loop shared by both backends (SPEC 7.2). Ties are delivered arrivals first, then
//! completions, then timers. The driver, not the strategy, writes the slot record.

use crate::backend::*;
use crate::calib::{CostEstimator, Observation};
use crate::metric::{InputRec, JobRec, SlotRecord};
use crate::strategy::{Ctx, Strategy};
use crate::types::*;
use arrival_model::schedule::{Kind, ScheduleSet};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

/// One run: a slot of a schedule set, for one aggregator view of one subnet (`Leaf`) or for the proposer (`Merge`).
pub struct RunSpec<'a> {
    pub set: &'a ScheduleSet,
    pub slot: u32,
    /// `Leaf` only.
    pub subnet: u8,
    pub view: u8,
    pub deadline: Micros,
    pub caps: Caps,
    /// A proof finished by `deadline + grace` counts. A strategy still plans against `deadline`.
    pub grace: Micros,
}

struct Arrival {
    t: Micros,
    kind: Kind,
    id: u32,
    valid: bool,
}

#[derive(Clone, Copy)]
enum Target {
    Prove(ProofId),
    Verify(ProofId),
}

pub struct DriverCtx<'a, 'e> {
    backend: &'a mut dyn Backend,
    /// What the strategy prices jobs with. `None`: the backend's own table, as the old harness did.
    /// The backend's model is the simulator's truth and is otherwise out of the strategy's reach.
    est: Option<&'e mut dyn CostEstimator>,
    info: SlotInfo,
    store: ProofStore,
    timers: BinaryHeap<Reverse<(Micros, u64, u64)>>,
    timer_seq: u64,
    lane_job: Vec<Option<JobId>>,
    jobs: Vec<JobRec>,
    target: Vec<Target>,
    published: Option<(Micros, ProofId)>,
}

impl Ctx for DriverCtx<'_, '_> {
    fn now(&self) -> Micros {
        self.backend.now()
    }

    fn slot(&self) -> &SlotInfo {
        &self.info
    }

    fn proof(&self, id: ProofId) -> &ProofInfo {
        &self.store.get(id).info
    }

    fn lane_idle(&self, lane: usize) -> bool {
        self.lane_job[lane].is_none()
    }

    fn prove(&mut self, lane: usize, job: Job) -> JobId {
        assert!(self.lane_idle(lane), "prove on busy lane {lane}");
        assert!(!job.children.is_empty() || !job.signatures.is_empty(), "empty job");
        assert!(job.children.len() <= self.info.caps.max_children, "too many children: {}", job.children.len());
        assert!(job.signatures.len() as u32 <= self.info.caps.max_new_sigs, "too many signatures: {}", job.signatures.len());
        let mut coverage = Coverage::empty();
        let mut leaves = u32::from(!job.signatures.is_empty());
        let mut child_claims = 0;
        for &c in &job.children {
            let e = self.store.entries.get(c.0 as usize).unwrap_or_else(|| panic!("unknown proof {c:?}"));
            assert!(e.ready, "proof {c:?} is not finished");
            coverage.union_with(&e.info.coverage);
            leaves += e.info.leaves;
            child_claims += e.info.coverage.len();
        }
        for s in &job.signatures {
            coverage.insert(s.0);
        }
        let made = ProofId(self.store.entries.len() as u32);
        self.store.entries.push(Entry {
            info: ProofInfo { coverage, rate: job.rate, verified: true, subnet: None, received_at: None, leaves },
            valid: true,
            parents: job.children.clone(),
            ready: false,
        });
        let started = self.backend.start_prove(lane, &job, &self.store, made);
        let id = JobId(self.jobs.len() as u32);
        let op = match (job.children.is_empty(), job.signatures.is_empty()) {
            (true, _) => "AGGREGATE",
            (false, true) => "RECURSE",
            _ => "EXTEND",
        };
        self.jobs.push(JobRec {
            job: id.0,
            lane,
            op,
            children: job.children.len() as u32,
            child_claims,
            new_sigs: job.signatures.len() as u32,
            rate: job.rate,
            start: self.backend.now(),
            end: 0,
            predicted: started.predicted,
            verify_us: started.verify_us,
            useful: false,
            result: String::new(),
            result_claims: 0,
        });
        self.target.push(Target::Prove(made));
        self.lane_job[lane] = Some(id);
        id
    }

    fn verify(&mut self, lane: usize, proof: ProofId) -> JobId {
        assert!(self.lane_idle(lane), "verify on busy lane {lane}");
        let started = self.backend.start_verify(lane, proof, &self.store);
        let id = JobId(self.jobs.len() as u32);
        let e = self.store.get(proof);
        self.jobs.push(JobRec {
            job: id.0,
            lane,
            op: "VERIFY",
            children: 1,
            child_claims: e.info.coverage.len(),
            new_sigs: 0,
            rate: e.info.rate,
            start: self.backend.now(),
            end: 0,
            predicted: started.predicted,
            verify_us: started.verify_us,
            useful: false,
            result: String::new(),
            result_claims: 0,
        });
        self.target.push(Target::Verify(proof));
        self.lane_job[lane] = Some(id);
        id
    }

    fn abort(&mut self, job: JobId) -> bool {
        match self.jobs.get(job.0 as usize) {
            Some(j) if self.lane_job[j.lane] == Some(job) => self.backend.abort(j.lane),
            _ => false,
        }
    }

    fn publish(&mut self, proof: ProofId) {
        assert!(self.published.is_none(), "second publish");
        assert!(self.store.get(proof).info.rate >= self.info.final_rate, "publish below the final rate");
        self.published = Some((self.backend.now(), proof));
    }

    fn set_timer(&mut self, at: Micros, token: u64) {
        let at = at.max(self.backend.now());
        self.timer_seq += 1;
        self.timers.push(Reverse((at, self.timer_seq, token)));
    }

    fn cost(&self, shape: &JobShape) -> Micros {
        match &self.est {
            Some(e) => e.estimate(shape),
            None => self.backend.model().prove_us(shape, THREADS),
        }
    }
}

impl DriverCtx<'_, '_> {
    /// If the strategy published nothing, publish the largest finished proof of this prover at the final rate, stamped
    /// `at`. A proof from a failed or running job is not finished, and neither is one whose job ended after `at`.
    /// Nothing happens when no such proof exists.
    fn fallback(&mut self, at: Micros) {
        if self.published.is_some() {
            return;
        }
        let best = self
            .target
            .iter()
            .zip(&self.jobs)
            .filter_map(|(t, j)| match t {
                Target::Prove(id) if j.end <= at => Some(*id),
                _ => None,
            })
            .filter(|id| {
                let e = self.store.get(*id);
                e.ready && e.info.rate >= self.info.final_rate
            })
            .max_by_key(|id| (self.store.get(*id).info.coverage.len(), id.0));
        if let Some(id) = best {
            self.published = Some((at, id));
        }
    }

    /// Write a completion into the record and the store, and turn it into an event.
    fn complete(&mut self, d: Done) -> Event {
        let job = self.lane_job[d.lane].take().expect("completion on an idle lane");
        let rec = &mut self.jobs[job.0 as usize];
        rec.end = d.at;
        match (self.target[job.0 as usize], d.outcome) {
            (Target::Prove(id), Outcome::Proved { coverage, rate, .. }) => {
                let e = &mut self.store.entries[id.0 as usize];
                e.info.coverage = *coverage;
                e.info.rate = rate;
                e.ready = true;
                rec.result = "ok".into();
                rec.result_claims = e.info.coverage.len();
                if let Some(est) = self.est.as_deref_mut() {
                    est.observe(&Observation { children: rec.children as usize, new_sigs: rec.new_sigs, rate: rec.rate, took_us: d.at.saturating_sub(rec.start) });
                }
                Event::Proved { job, lane: d.lane, result: Ok(id) }
            }
            (Target::Verify(proof), Outcome::Verified(ok)) => {
                rec.result = if ok { "ok" } else { "invalid" }.into();
                if ok {
                    self.store.entries[proof.0 as usize].info.verified = true;
                }
                Event::Verified { job, lane: d.lane, proof, ok }
            }
            (Target::Prove(made), Outcome::Failed(f)) => {
                rec.result = format!("{f:?}");
                if let Failure::InvalidChild(c) = &f {
                    // The backend verifies children in order: those before the bad one passed.
                    let parents = self.store.entries[made.0 as usize].parents.clone();
                    for p in parents.iter().take_while(|p| *p != c) {
                        self.store.entries[p.0 as usize].info.verified = true;
                    }
                }
                Event::Proved { job, lane: d.lane, result: Err(f) }
            }
            (_, Outcome::Failed(f)) => {
                rec.result = format!("{f:?}");
                Event::Proved { job, lane: d.lane, result: Err(f) }
            }
            _ => panic!("backend answered a job with the wrong outcome"),
        }
    }
}

/// `run_slot_with` and the backend's own table as the strategy's cost.
pub fn run_slot(backend: &mut dyn Backend, strategy: &mut dyn Strategy, spec: &RunSpec) -> SlotRecord {
    run_slot_with(backend, strategy, spec, None)
}

/// One slot. `est` prices jobs for the strategy and learns from every job that finishes; keep one across slots.
pub fn run_slot_with<'e>(backend: &mut dyn Backend, strategy: &mut dyn Strategy, spec: &RunSpec, est: Option<&'e mut dyn CostEstimator>) -> SlotRecord {
    let meta = &spec.set.meta;
    let shape = if meta.shape == "merge" { Shape::Merge } else { Shape::Leaf };
    let info = SlotInfo { shape, deadline: spec.deadline, final_rate: meta.final_rate, expected_inputs: meta.expected_inputs, caps: spec.caps };
    let run = if shape == Shape::Merge { spec.slot } else { (spec.slot * meta.subnets + spec.subnet as u32) * meta.views + spec.view as u32 };
    backend.begin_slot(spec.set, run);
    let sched = &spec.set.slots[spec.slot as usize];
    let arrivals: Vec<Arrival> = sched
        .arrivals
        .iter()
        .filter(|a| match shape {
            Shape::Leaf => a.kind == Kind::Vote && a.subnet == spec.subnet && a.view == spec.view,
            Shape::Merge => a.kind == Kind::Aggregate,
        })
        .map(|a| Arrival { t: a.t_us, kind: a.kind, id: a.id, valid: a.valid })
        .collect();
    let claim_of = |id: u32| SUBNET_SIZE * spec.subnet as u32 + id;
    let inputs: Vec<InputRec> = arrivals
        .iter()
        .map(|a| match a.kind {
            Kind::Vote => InputRec { t: a.t, claims: vec![claim_of(a.id)], valid: true },
            Kind::Aggregate => {
                let e = &spec.set.pool[a.id as usize];
                InputRec { t: a.t, claims: e.signer_ids().map(|s| SUBNET_SIZE * e.subnet as u32 + s).collect(), valid: a.valid }
            }
        })
        .collect();

    let mut ctx = DriverCtx {
        backend,
        est,
        info,
        store: ProofStore::default(),
        timers: BinaryHeap::new(),
        timer_seq: 0,
        lane_job: vec![None; info.caps.lanes],
        jobs: Vec::new(),
        target: Vec::new(),
        published: None,
    };
    strategy.on_event(Event::SlotStart, &mut ctx);
    let (mut ai, mut pending): (usize, Option<Done>) = (0, None);
    let mut held: Option<Done> = None;
    let deadline = spec.deadline;
    // Inputs stop at `deadline`; completions and timers go on until the end of the grace window.
    let end = deadline + spec.grace;
    loop {
        let t_arr = arrivals.get(ai).filter(|a| a.t <= deadline).map_or(Micros::MAX, |a| a.t);
        let t_tim = ctx.timers.peek().map_or(Micros::MAX, |r| r.0.0);
        let busy = ctx.lane_job.iter().any(Option::is_some);
        if t_arr == Micros::MAX && t_tim == Micros::MAX && !busy && pending.is_none() {
            break;
        }
        let wake = if ctx.backend.now() < deadline { deadline } else { end };
        let tn = t_arr.min(t_tim).min(wake).max(ctx.backend.now());
        let done = pending.take().or_else(|| ctx.backend.wait(tn));
        if let Some(d) = done {
            if d.at > end {
                // A wake-up that overshot: this job ended after the window. The strategy never sees it, the
                // drain below writes it down.
                held = Some(d);
            } else if t_arr <= d.at {
                // Arrivals first: they are delivered below, then this completion.
                pending = Some(d);
            } else {
                let event = ctx.complete(d);
                strategy.on_event(event, &mut ctx);
                continue;
            }
        }
        let now = ctx.backend.now();
        while arrivals.get(ai).is_some_and(|a| a.t <= now.min(deadline)) {
            let a = &arrivals[ai];
            ai += 1;
            let event = match a.kind {
                Kind::Vote => Event::Signature(SigId(claim_of(a.id))),
                Kind::Aggregate => {
                    let e = &spec.set.pool[a.id as usize];
                    let mut coverage = Coverage::empty();
                    for s in e.signer_ids() {
                        coverage.insert(SUBNET_SIZE * e.subnet as u32 + s);
                    }
                    let id = ProofId(ctx.store.entries.len() as u32);
                    ctx.store.entries.push(Entry {
                        info: ProofInfo { coverage, rate: e.rate, verified: false, subnet: Some(e.subnet), received_at: Some(a.t), leaves: 1 },
                        valid: a.valid,
                        parents: Vec::new(),
                        ready: true,
                    });
                    ctx.backend.receive(id, e, a.valid);
                    Event::Aggregate(id)
                }
            };
            strategy.on_event(event, &mut ctx);
        }
        if let Some(d) = pending.take() {
            let event = ctx.complete(d);
            strategy.on_event(event, &mut ctx);
        }
        while ctx.timers.peek().is_some_and(|r| r.0.0 <= now) {
            let Reverse((_, _, token)) = ctx.timers.pop().unwrap();
            strategy.on_event(Event::Timer(token), &mut ctx);
        }
        if now >= end {
            break;
        }
        // Past the deadline the slot is settled once a proof is out, or once no job can still bring a better one.
        if now >= deadline && (ctx.published.is_some() || !ctx.lane_job.iter().any(Option::is_some)) {
            break;
        }
    }
    // The driver's own timer, not the strategy's: at the end of the grace window, or when nothing more can happen,
    // a slot that published nothing publishes the best proof finished by then. The stamp is the real time, so a
    // wake-up that overshot the window shows as late. A job still running is not that proof.
    let at = ctx.backend.now();
    // A job still running at the deadline is written down, not delivered.
    if let Some(d) = held.take() {
        let _ = ctx.complete(d);
    }
    while ctx.lane_job.iter().any(Option::is_some) {
        let d = ctx.backend.wait(Micros::MAX).expect("a lane is busy");
        let _ = ctx.complete(d);
    }
    ctx.fallback(at);
    finish(ctx, inputs, spec)
}

fn finish(mut ctx: DriverCtx, inputs: Vec<InputRec>, spec: &RunSpec) -> SlotRecord {
    let mut useful = HashSet::new();
    let published = ctx.published.map(|(t, id)| {
        let mut stack = vec![id];
        while let Some(p) = stack.pop() {
            if useful.insert(p) {
                stack.extend(ctx.store.get(p).parents.iter().copied());
            }
        }
        (t, ctx.store.get(id).info.coverage.clone())
    });
    for (j, t) in ctx.jobs.iter_mut().zip(&ctx.target) {
        // A job is useful if it made an ancestor of the published proof, or verified one.
        j.useful = matches!(t, Target::Prove(p) | Target::Verify(p) if useful.contains(p));
    }
    SlotRecord { shape: ctx.info.shape, deadline: spec.deadline, inputs, published, jobs: ctx.jobs, sent_override: None, grace: spec.grace }
}
