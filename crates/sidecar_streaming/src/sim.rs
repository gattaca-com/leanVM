//! `SimBackend`: a virtual clock and a cost model.

use crate::backend::*;
use crate::cost::CostModel;
use crate::rng::SplitMix64;
use crate::types::*;
use arrival_model::schedule::{PoolEntry, ScheduleSet};
use std::collections::HashMap;

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
    replay: Option<Replay>,
}

/// Measured over predicted durations of live jobs, by op class and size. The simulator draws from them instead of
/// its own 0.1 % noise, which measures repeatability and not the error of a prediction (AUDIT.md finding 7).
#[derive(Clone, Debug, Default)]
pub struct Replay {
    /// By op (`AGGREGATE`, `EXTEND`, `RECURSE`) and whether the job has `LARGE` or more claims and signatures.
    pools: HashMap<(String, bool), Vec<f64>>,
}

/// Jobs with this many child claims plus new signatures or more are drawn from the large pool.
const LARGE: u32 = 700;
/// A pool with fewer ratios than this is not used on its own: the op's pool is.
const MIN_POOL: usize = 5;

impl Replay {
    /// From job files of live runs (`slot,job,lane,op,children,child_claims,new_sigs,rate,start_us,end_us,predicted_us,result`).
    /// Only jobs with result `ok` count. Fails if no job does.
    pub fn from_job_files(texts: &[String]) -> Result<Replay, String> {
        let mut r = Replay::default();
        for text in texts {
            for line in text.lines().skip(1).filter(|l| !l.trim().is_empty()) {
                let f: Vec<&str> = line.split(',').collect();
                let bad = || format!("bad live job line: {line}");
                if f.len() < 12 {
                    return Err(bad());
                }
                if f[11] != "ok" {
                    continue;
                }
                let n = |i: usize| f[i].parse::<u64>().map_err(|_| bad());
                let (claims, new, start, end, pred) = (n(5)?, n(6)?, n(8)?, n(9)?, n(10)?);
                if pred == 0 || end <= start {
                    continue;
                }
                let large = claims + new >= LARGE as u64;
                r.pools.entry((f[3].to_string(), large)).or_default().push((end - start) as f64 / pred as f64);
            }
        }
        if r.pools.is_empty() {
            return Err("no finished live job to replay".into());
        }
        Ok(r)
    }

    pub fn jobs(&self) -> usize {
        self.pools.values().map(Vec::len).sum()
    }

    /// A ratio for a job of `op` with `size` claims and signatures, or None if the op was never seen.
    fn draw(&self, op: &str, size: u32, rng: &mut SplitMix64) -> Option<f64> {
        let own = self.pools.get(&(op.to_string(), size >= LARGE)).filter(|v| v.len() >= MIN_POOL);
        let pool: Vec<&f64> = match own {
            Some(v) => v.iter().collect(),
            None => [false, true].iter().filter_map(|l| self.pools.get(&(op.to_string(), *l))).flatten().collect(),
        };
        (!pool.is_empty()).then(|| *pool[(rng.next_u64() % pool.len() as u64) as usize])
    }
}

impl SimBackend {
    /// Draw job durations from `replay` (prove jobs only) in place of the model's noise. `no_noise` does not turn it off.
    pub fn with_replay(mut self, replay: Replay) -> Self {
        self.replay = Some(replay);
        self
    }

    pub fn new(model: Box<dyn CostModel>, caps: Caps, seed: u64, noise: bool) -> Self {
        let lanes = (0..caps.lanes).map(|_| None).collect();
        Self { model, caps, now: 0, lanes, rng: SplitMix64::new(seed), seed, noise, replay: None }
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

    fn begin_slot(&mut self, set: &ScheduleSet, run: u32) {
        let _ = set;
        self.now = 0;
        self.lanes.iter_mut().for_each(|l| *l = None);
        self.rng = SplitMix64::derive(self.seed, run as u64);
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
                let op = match (shape.children.is_empty(), shape.new_sigs) {
                    (true, _) => "AGGREGATE",
                    (false, 0) => "RECURSE",
                    _ => "EXTEND",
                };
                let size = shape.new_sigs + shape.children.iter().map(|c| c.claims).sum::<u32>();
                let replayed = self.replay.as_ref().and_then(|r| r.draw(op, size, &mut self.rng));
                let dur = match replayed {
                    Some(ratio) => ((predicted as f64) * ratio).round().max(1.0) as Micros,
                    None => self.noisy(predicted),
                };
                (dur, Outcome::Proved { coverage: Box::new(coverage), rate: job.rate, bytes })
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
