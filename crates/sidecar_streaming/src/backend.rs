//! What the driver needs from something that proves: a virtual clock and a cost model, or a running sidecar.

use crate::cost::CostModel;
use crate::types::*;
use arrival_model::schedule::{PoolEntry, ScheduleSet};

/// Threads the cost models are fitted at: one lane on the target.
pub const THREADS: u32 = 8;

#[derive(Clone, Debug)]
pub enum Outcome {
    Proved { coverage: Box<Coverage>, rate: Rate, bytes: u32 },
    Verified(bool),
    Failed(Failure),
}

#[derive(Clone, Debug)]
pub struct Done {
    pub lane: usize,
    pub at: Micros,
    pub outcome: Outcome,
}

/// What a backend says about a job it just started.
#[derive(Clone, Copy, Debug, Default)]
pub struct Started {
    /// Noise-free model prediction.
    pub predicted: Micros,
    /// Lane time spent verifying children (inside a prove job) or the proof (a verify job).
    pub verify_us: Micros,
}

pub struct Entry {
    pub info: ProofInfo,
    /// False for a received aggregate that fails verification.
    pub valid: bool,
    pub parents: Vec<ProofId>,
    /// False while the job that makes it runs.
    pub ready: bool,
}

#[derive(Default)]
pub struct ProofStore {
    pub entries: Vec<Entry>,
}

impl ProofStore {
    pub fn get(&self, id: ProofId) -> &Entry {
        &self.entries[id.0 as usize]
    }
}

pub trait Backend {
    fn caps(&self) -> Caps;
    /// The truth of this backend: what the simulator draws durations from. A strategy reaches it only when the run
    /// asks for `--strategy-cost table`; see `calib`.
    fn model(&self) -> &dyn CostModel;
    fn now(&self) -> Micros;
    /// `run` identifies the run within the set: `(slot * subnets + subnet) * views + view` for a leaf set, `slot` for a merge set.
    /// The simulator draws independent noise per run.
    fn begin_slot(&mut self, set: &ScheduleSet, run: u32);
    /// A received aggregate enters the proof store.
    fn receive(&mut self, id: ProofId, pool_entry: &PoolEntry, valid: bool);
    fn start_prove(&mut self, lane: usize, job: &Job, store: &ProofStore, made: ProofId) -> Started;
    fn start_verify(&mut self, lane: usize, proof: ProofId, store: &ProofStore) -> Started;
    fn abort(&mut self, lane: usize) -> bool;
    /// Advance to `until` at most. Return the first completion at or before it, else None with `now() == until`.
    fn wait(&mut self, until: Micros) -> Option<Done>;
}
