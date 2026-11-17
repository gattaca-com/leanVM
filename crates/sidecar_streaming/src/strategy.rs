//! A strategy is a state machine: it receives events and answers with calls on [`Ctx`].
//! It never sees payload bytes, flags, the kept-proof slot or the rings.

use crate::types::*;

pub trait Ctx {
    fn now(&self) -> Micros;
    fn slot(&self) -> &SlotInfo;
    fn proof(&self, id: ProofId) -> &ProofInfo;
    fn lane_idle(&self, lane: usize) -> bool;
    /// Start `job` on an idle lane. Panics if the lane is busy or a child id is unknown.
    fn prove(&mut self, lane: usize, job: Job) -> JobId;
    /// Verify a received aggregate on an idle lane.
    fn verify(&mut self, lane: usize, proof: ProofId) -> JobId;
    /// Stop a running job. Returns false, and does nothing, unless `caps.abort`.
    fn abort(&mut self, job: JobId) -> bool;
    /// Publish `proof` as the slot's one output. Panics on a second call, or below the final rate.
    fn publish(&mut self, proof: ProofId);
    /// Deliver `Event::Timer(token)` at `at`, or now if `at` is past.
    fn set_timer(&mut self, at: Micros, token: u64);
    /// Predicted duration of a job of this shape on one lane.
    fn cost(&self, shape: &JobShape) -> Micros;
}

pub trait Strategy {
    /// Includes its parameters.
    fn name(&self) -> String;
    fn on_event(&mut self, event: Event, ctx: &mut dyn Ctx);
}
