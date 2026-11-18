//! Two leaf strategies with no cost estimate. Neither calls `Ctx::cost`, keeps a margin, or guards the deadline.
//! They are baselines for readers outside the project.
//!
//! - `greedy_simple`: batch all queued votes whenever the prover is idle.
//! - `once_at`: wait for a fixed time, then prove once.

use super::Params;
use crate::strategy::{Ctx, Strategy};
use crate::types::*;
use std::collections::VecDeque;

/// Batching as a BLS aggregator does it. When the prover is idle and votes are queued, one job takes all of them at the
/// final rate: an EXTEND of the kept proof, or an AGGREGATE when there is none. Jobs start until `D`.
///
/// The strategy never publishes. At `D` plus grace the driver's fallback publishes the largest proof finished by then.
/// A job still running at that time is wasted. A job that fails puts its votes and the kept proof back.
pub struct GreedySimple {
    waiting: VecDeque<SigId>,
    kept: Option<ProofId>,
    /// The kept proof and the votes of the job in flight, to restore if it fails.
    in_flight: Option<(Option<ProofId>, Vec<SigId>)>,
}

impl GreedySimple {
    pub fn new() -> Self {
        Self { waiting: VecDeque::new(), kept: None, in_flight: None }
    }

    fn start(&mut self, ctx: &mut dyn Ctx) {
        if self.in_flight.is_some() || self.waiting.is_empty() || ctx.now() >= ctx.slot().deadline {
            return;
        }
        let n = self.waiting.len().min(ctx.slot().caps.max_new_sigs as usize);
        let signatures: Vec<SigId> = self.waiting.drain(..n).collect();
        let kept = self.kept.take();
        self.in_flight = Some((kept, signatures.clone()));
        ctx.prove(0, Job { children: kept.into_iter().collect(), signatures, rate: ctx.slot().final_rate });
    }
}

impl Default for GreedySimple {
    fn default() -> Self {
        Self::new()
    }
}

impl Strategy for GreedySimple {
    fn name(&self) -> String {
        "greedy_simple".into()
    }

    fn on_event(&mut self, event: Event, ctx: &mut dyn Ctx) {
        match event {
            Event::Signature(s) => {
                self.waiting.push_back(s);
                self.start(ctx);
            }
            Event::Proved { result, .. } => {
                let launched = self.in_flight.take();
                match result {
                    Ok(id) => self.kept = Some(id),
                    Err(_) => {
                        if let Some((kept, signatures)) = launched {
                            self.kept = kept;
                            signatures.into_iter().rev().for_each(|s| self.waiting.push_front(s));
                        }
                    }
                }
                self.start(ctx);
            }
            Event::SlotStart | Event::Aggregate(_) | Event::Verified { .. } | Event::Timer(_) => {}
        }
    }
}

/// Wait, then prove once. At `D - lead_ms` it starts one AGGREGATE of all queued votes at the final rate, and publishes
/// it when it ends. Later votes are left out. If the job ends after `D` plus grace, nothing is published.
///
/// `lead_ms` defaults to 2200: the lead for main on 8 cores. The measured AGGREGATE of 1582 votes at rate 2 took
/// 2115 ms (live `rec3-typical`, design/AUDIT.md finding 1). The default adds 4 % to that. Set it for other builds.
pub struct OnceAt {
    lead: Micros,
    waiting: VecDeque<SigId>,
    started: bool,
}

pub const DEFAULT_LEAD_MS: f64 = 2200.0;

impl OnceAt {
    pub fn new(p: &Params) -> Self {
        Self { lead: (p.f("lead_ms", DEFAULT_LEAD_MS) * 1000.0).round() as Micros, waiting: VecDeque::new(), started: false }
    }
}

impl Strategy for OnceAt {
    fn name(&self) -> String {
        format!("once_at:lead_ms={}", self.lead as f64 / 1000.0)
    }

    fn on_event(&mut self, event: Event, ctx: &mut dyn Ctx) {
        match event {
            Event::SlotStart => ctx.set_timer(ctx.slot().deadline.saturating_sub(self.lead), 1),
            Event::Signature(s) => {
                if !self.started {
                    self.waiting.push_back(s);
                }
            }
            Event::Timer(_) => {
                if self.started || self.waiting.is_empty() {
                    return;
                }
                self.started = true;
                let n = self.waiting.len().min(ctx.slot().caps.max_new_sigs as usize);
                let signatures: Vec<SigId> = self.waiting.drain(..n).collect();
                ctx.prove(0, Job { children: Vec::new(), signatures, rate: ctx.slot().final_rate });
            }
            Event::Proved { result: Ok(id), .. } => ctx.publish(id),
            Event::Proved { result: Err(_), .. } | Event::Aggregate(_) | Event::Verified { .. } => {}
        }
    }
}
