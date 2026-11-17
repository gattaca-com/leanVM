//! Ids, claim sets, jobs and events: the vocabulary between a strategy and a backend.

pub type Micros = u64;
/// `log_inv_rate`, 1 to 4.
pub type Rate = u8;

/// Claims of one subnet are numbered `1500 * subnet + signer`.
pub const SUBNET_SIZE: u32 = 1500;
pub const MAX_CLAIMS: usize = 24000;
const WORDS: usize = MAX_CLAIMS / 64;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct SigId(pub u32);
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ProofId(pub u32);
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct JobId(pub u32);

/// A set of claim indices: a fixed bitmap of 24000 bits.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Coverage {
    words: [u64; WORDS],
}

impl Default for Coverage {
    fn default() -> Self {
        Self::empty()
    }
}

impl Coverage {
    pub fn empty() -> Self {
        Self { words: [0; WORDS] }
    }

    pub fn len(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    pub fn insert(&mut self, claim: u32) {
        self.words[claim as usize / 64] |= 1 << (claim % 64);
    }

    pub fn contains(&self, claim: u32) -> bool {
        self.words[claim as usize / 64] >> (claim % 64) & 1 == 1
    }

    pub fn union_with(&mut self, other: &Coverage) {
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a |= b;
        }
    }

    /// `|self ∪ other|`.
    pub fn union_len(&self, other: &Coverage) -> u32 {
        self.words.iter().zip(&other.words).map(|(a, b)| (a | b).count_ones()).sum()
    }

    /// Whether every claim of `other` is in `self`.
    pub fn contains_all(&self, other: &Coverage) -> bool {
        self.words.iter().zip(&other.words).all(|(a, b)| b & !a == 0)
    }

    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.words.iter().enumerate().flat_map(|(w, &bits)| (0..64u32).filter(move |b| bits >> b & 1 == 1).map(move |b| w as u32 * 64 + b))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Shape {
    Leaf,
    Merge,
}

#[derive(Clone, Copy, Debug)]
pub struct Caps {
    /// 1 for today's sidecar.
    pub lanes: usize,
    /// False for today's sidecar.
    pub abort: bool,
    /// False for today's sidecar: `verify` runs on the prover's lane.
    pub verify_lane: bool,
    pub max_children: usize,
    pub max_new_sigs: u32,
}

impl Default for Caps {
    fn default() -> Self {
        Caps { lanes: 1, abort: false, verify_lane: false, max_children: 16, max_new_sigs: 1500 }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SlotInfo {
    pub shape: Shape,
    pub deadline: Micros,
    pub final_rate: Rate,
    /// 1500 signatures, or 16 subnets.
    pub expected_inputs: u32,
    pub caps: Caps,
}

#[derive(Clone, Debug)]
pub struct ProofInfo {
    pub coverage: Coverage,
    pub rate: Rate,
    /// True for proofs this prover made, and after a successful verify.
    pub verified: bool,
    /// Some for a received aggregate.
    pub subnet: Option<u8>,
    pub received_at: Option<Micros>,
    /// Leaf proofs in the claim tree below this proof (a received aggregate counts 1).
    pub leaves: u32,
}

/// Prove the union of `children` and `signatures` at `rate`.
/// No children: AGGREGATE. No signatures: RECURSE. Both: EXTEND.
/// Unverified children are verified by the backend before proving.
#[derive(Clone, Debug)]
pub struct Job {
    pub children: Vec<ProofId>,
    pub signatures: Vec<SigId>,
    pub rate: Rate,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Failure {
    /// Verification found it; the job proved nothing.
    InvalidChild(ProofId),
    /// The backend refused the job (limits, bad request).
    Rejected(String),
    Aborted,
}

#[derive(Clone, Debug)]
pub enum Event {
    SlotStart,
    /// Already validated by the consensus layer.
    Signature(SigId),
    /// Received, not verified.
    Aggregate(ProofId),
    Proved { job: JobId, lane: usize, result: Result<ProofId, Failure> },
    Verified { job: JobId, lane: usize, proof: ProofId, ok: bool },
    Timer(u64),
}

/// The shape of a job, for cost queries about jobs that do not exist yet.
#[derive(Clone, Debug)]
pub struct JobShape {
    pub children: Vec<ChildShape>,
    pub new_sigs: u32,
    pub rate: Rate,
}

#[derive(Clone, Copy, Debug)]
pub struct ChildShape {
    pub claims: u32,
    pub rate: Rate,
    pub verify: bool,
    pub leaves: u32,
}

impl JobShape {
    pub fn aggregate(new_sigs: u32, rate: Rate) -> JobShape {
        JobShape { children: Vec::new(), new_sigs, rate }
    }

    /// An EXTEND of one kept proof of `kept` claims, already verified.
    pub fn extend(kept: u32, kept_rate: Rate, new_sigs: u32, rate: Rate) -> JobShape {
        if kept == 0 {
            return JobShape::aggregate(new_sigs, rate);
        }
        JobShape { children: vec![ChildShape { claims: kept, rate: kept_rate, verify: false, leaves: 1 }], new_sigs, rate }
    }
}
