//! `SimBackend`: a virtual clock and a cost model.

use crate::backend::*;
use crate::cost::CostModel;
use crate::rng::SplitMix64;
use crate::types::*;
use arrival_model::schedule::{PoolEntry, ScheduleSet};

struct Running {
    end: Micros,
    outcome: Outcome,
}

pub struct SimBackend {
    model: Box<dyn CostModel>,
    caps: Caps,
    now: Micros,
    lanes: Vec<Option<Running>>,
    rng: SplitMix64,
    seed: u64,
    noise: bool,
}

impl SimBackend {
    pub fn new(model: Box<dyn CostModel>, caps: Caps, seed: u64, noise: bool) -> Self {
        let lanes = (0..caps.lanes).map(|_| None).collect();
        Self { model, caps, now: 0, lanes, rng: SplitMix64::new(seed), seed, noise }
    }

    fn noisy(&mut self, us: Micros) -> Micros {
        if !self.noise || self.model.sigma() == 0.0 {
            return us;
        }
        let z = self.rng.normal();
        ((us as f64) * (self.model.sigma() * z).exp()).round().max(1.0) as Micros
    }
}

pub fn child_shape(info: &ProofInfo, verify: bool) -> ChildShape {
    ChildShape { claims: info.coverage.len(), rate: info.rate, verify, leaves: info.leaves }
}

impl Backend for SimBackend {
    fn caps(&self) -> Caps {
        self.caps
    }

    fn model(&self) -> &dyn CostModel {
        self.model.as_ref()
    }

    fn now(&self) -> Micros {
        self.now
    }

    fn begin_slot(&mut self, set: &ScheduleSet, slot: u32) {
        let _ = set;
        self.now = 0;
        self.lanes.iter_mut().for_each(|l| *l = None);
        self.rng = SplitMix64::derive(self.seed, slot as u64);
    }

    fn receive(&mut self, _id: ProofId, _pool_entry: &PoolEntry, _valid: bool) {}

    fn start_prove(&mut self, lane: usize, job: &Job, store: &ProofStore, made: ProofId) -> Started {
        assert!(self.lanes[lane].is_none(), "lane {lane} is busy");
        let mut shape = JobShape { children: Vec::new(), new_sigs: job.signatures.len() as u32, rate: job.rate };
        let mut coverage = Coverage::empty();
        let mut verify_total = 0;
        let mut fail: Option<(Micros, ProofId)> = None;
        for &c in &job.children {
            let e = store.get(c);
            let unverified = !e.info.verified;
            shape.children.push(child_shape(&e.info, unverified));
            if unverified {
                verify_total += self.model.verify_us(shape.children.last().unwrap());
                if !e.valid && fail.is_none() {
                    fail = Some((verify_total, c));
                }
            }
            coverage.union_with(&e.info.coverage);
        }
        for s in &job.signatures {
            coverage.insert(s.0);
        }
        let predicted = self.model.prove_us(&shape, THREADS);
        let (dur, outcome) = match fail {
            Some((t, c)) => (t, Outcome::Failed(Failure::InvalidChild(c))),
            None => {
                let bytes = self.model.proof_bytes(&shape);
                (self.noisy(predicted), Outcome::Proved { coverage: Box::new(coverage), rate: job.rate, bytes })
            }
        };
        let _ = made;
        self.lanes[lane] = Some(Running { end: self.now + dur, outcome });
        Started { predicted, verify_us: verify_total }
    }

    fn start_verify(&mut self, lane: usize, proof: ProofId, store: &ProofStore) -> Started {
        assert!(self.lanes[lane].is_none(), "lane {lane} is busy");
        let e = store.get(proof);
        let us = self.model.verify_us(&child_shape(&e.info, true));
        let dur = self.noisy(us);
        self.lanes[lane] = Some(Running { end: self.now + dur, outcome: Outcome::Verified(e.valid) });
        Started { predicted: us, verify_us: us }
    }

    fn abort(&mut self, lane: usize) -> bool {
        if !self.caps.abort || self.lanes[lane].is_none() {
            return false;
        }
        let end = self.now + self.model.abort_us();
        self.lanes[lane] = Some(Running { end, outcome: Outcome::Failed(Failure::Aborted) });
        true
    }

    fn wait(&mut self, until: Micros) -> Option<Done> {
        let next = self.lanes.iter().enumerate().filter_map(|(i, l)| l.as_ref().map(|r| (r.end, i))).min();
        match next {
            Some((end, lane)) if end <= until => {
                self.now = self.now.max(end);
                let r = self.lanes[lane].take().unwrap();
                Some(Done { lane, at: end, outcome: r.outcome })
            }
            _ => {
                self.now = self.now.max(until);
                None
            }
        }
    }
}
