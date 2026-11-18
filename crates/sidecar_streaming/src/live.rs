//! `LiveBackend`: a running `leanvm-sidecar` on a shared channel (SPEC 7.4).
//!
//! The clock is real: `Instant` at `begin_slot` is time 0. Arrivals are injected by the driver at their
//! scheduled times. Version 1 never uses `USE_KEPT`: children travel in the payload, each with its claims.
//!
//! Built and unit-tested here; a live run needs the sidecar and the AVX2 box (see `streamsim live`).

use crate::backend::*;
use crate::cli::*;
use crate::cost::CostModel;
use crate::metric::ROW_HEADER;
use crate::report::report;
use crate::runner::{JOBS_HEADER, Study, job_line, run_set};
use crate::strategies::Spec;
use crate::types::*;
use crate::wire::*;
use arrival_model::schedule::{PoolEntry, ScheduleSet};
use leanvm_sidecar_client::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const RESP_CAP: usize = 16 << 20;
const POLL: Duration = Duration::from_micros(50);
/// Idle time between slots.
const SETTLE: Duration = Duration::from_millis(200);
/// A response that does not come in this long is a hung sidecar.
const HUNG: Duration = Duration::from_secs(300);

enum What {
    Prove { made: ProofId, rate: Rate },
    Verify,
}

struct Inflight {
    id: u64,
    what: What,
    children: Vec<ProofId>,
    epoch: u64,
    sent: Instant,
}

pub struct LiveBackend {
    client: Client,
    model: Box<dyn CostModel>,
    table: SigTable,
    label: String,
    cache_root: PathBuf,
    t0: Instant,
    /// `(proof, claims)` of every proof of the slot, by `ProofId`.
    proofs: HashMap<u32, (Vec<u8>, Vec<u8>)>,
    made: Vec<ProofId>,
    /// Pool proofs by pool entry id, for the set in `pool_key`.
    pool: HashMap<u32, (Vec<u8>, Vec<u8>)>,
    pool_key: Option<u64>,
    inflight: Option<Inflight>,
    /// A request that could not be sent (no room in the channel), reported by the next `wait`.
    rejected: Option<String>,
    /// How late each wake-up came, in microseconds.
    pub lateness_us: Vec<u64>,
    /// Jump the clock over the idle start of a slot: the first wait of a slot, with no request out, ends 100 ms before its target.
    pub skip_idle: bool,
    first_wait: bool,
}

fn status_text(r: &Response) -> String {
    format!("status {} detail {}: {}", r.status, r.detail, String::from_utf8_lossy(&r.body))
}

fn fnv(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= *b as u64;
        *h = h.wrapping_mul(0x100_0000_01b3);
    }
}

impl LiveBackend {
    pub fn open(channel: &Path, signatures: &Path, model: Box<dyn CostModel>, label: &str, cache_root: &Path) -> Result<LiveBackend, String> {
        let client = Client::open(channel).map_err(|e| format!("{}: {e}", channel.display()))?;
        let data = std::fs::read(signatures).map_err(|e| format!("{}: {e}", signatures.display()))?;
        let table = SigTable::new(data, client.pub_key_size(), client.message_len(), client.sig_size(), 1500)?;
        Ok(LiveBackend {
            client,
            model,
            table,
            label: label.to_string(),
            cache_root: cache_root.to_path_buf(),
            t0: Instant::now(),
            proofs: HashMap::new(),
            made: Vec::new(),
            pool: HashMap::new(),
            pool_key: None,
            inflight: None,
            rejected: None,
            lateness_us: Vec::new(),
            skip_idle: false,
            first_wait: false,
        })
    }

    fn elapsed(&self) -> Micros {
        self.t0.elapsed().as_micros() as Micros
    }

    /// Send one request and wait for its response, outside any slot.
    fn call(&mut self, op: u32, flags: u32, rate: u32, payload: &[u8]) -> Result<Response, String> {
        let id = self.client.send(op, flags, rate, payload, RESP_CAP).map_err(|e| e.to_string())?;
        let started = Instant::now();
        loop {
            if let Some(r) = self.client.try_recv() {
                if r.id < id {
                    // A response left over from before a restart.
                    continue;
                }
                if r.id != id {
                    return Err(format!("response {} for request {id}", r.id));
                }
                return Ok(r);
            }
            if started.elapsed() > HUNG {
                return Err("no response from the sidecar".into());
            }
            std::thread::sleep(POLL);
        }
    }

    fn pool_dir(&self, key: u64) -> PathBuf {
        // A pool proof is an AGGREGATE output: a one-node claim tree. The perf backend decodes any child it did not
        // remember (it keeps the trees of the last 32 proofs it made) as one leaf node, so a pool of any size works,
        // and a cached proof stays valid across sidecar restarts.
        self.cache_root.join(&self.label).join(format!("{key:016x}"))
    }

    /// Build every aggregate of the pool once, through this sidecar, and cache it on disk.
    fn ensure_pool(&mut self, set: &ScheduleSet) -> Result<(), String> {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for e in &set.pool {
            fnv(&mut h, &[e.subnet, e.rate]);
            fnv(&mut h, &e.id.to_le_bytes());
            e.signers.iter().for_each(|w| fnv(&mut h, &w.to_le_bytes()));
        }
        if self.pool_key == Some(h) {
            return Ok(());
        }
        let dir = self.pool_dir(h);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        self.pool.clear();
        for e in &set.pool {
            let (pf, cf) = (dir.join(format!("{}.proof", e.id)), dir.join(format!("{}.claims", e.id)));
            if let (Ok(p), Ok(c)) = (std::fs::read(&pf), std::fs::read(&cf)) {
                self.pool.insert(e.id, (p, c));
                continue;
            }
            let ids: Vec<SigId> = e.signer_ids().map(|s| SigId(SUBNET_SIZE * e.subnet as u32 + s)).collect();
            let r = self.call(OP_AGGREGATE, 0, e.rate as u32, &signatures_payload(&self.table, &ids))?;
            if r.status != ST_OK {
                return Err(format!("building pool proof {}: {}", e.id, status_text(&r)));
            }
            let b = split_body(&r.body)?;
            std::fs::write(&pf, b.proof).map_err(|e| e.to_string())?;
            std::fs::write(&cf, b.claims).map_err(|e| e.to_string())?;
            self.pool.insert(e.id, (b.proof.to_vec(), b.claims.to_vec()));
        }
        self.pool_key = Some(h);
        Ok(())
    }

    /// `VERIFY` every proof this slot's jobs made. Returns how many verified; an error names the first that did not.
    pub fn verify_made(&mut self) -> Result<usize, String> {
        let ids = self.made.clone();
        for id in &ids {
            let (proof, claims) = self.proofs.get(&id.0).ok_or("a made proof is missing")?.clone();
            let r = self.call(OP_VERIFY, 0, 0, &verify_payload(&Child { claims: &claims, proof: &proof }))?;
            if r.status != ST_OK {
                return Err(format!("proof {} does not verify: {}", id.0, status_text(&r)));
            }
        }
        Ok(ids.len())
    }

    fn child(&self, id: ProofId) -> Child<'_> {
        let (proof, claims) = &self.proofs[&id.0];
        Child { claims, proof }
    }

    fn finish(&mut self, inf: Inflight, r: Response) -> Done {
        let at = self.elapsed();
        let outcome = match inf.what {
            What::Prove { made, rate } => match r.status {
                ST_OK => match split_body(&r.body).and_then(|b| claims_coverage(&self.table, b.claims).map(|c| (b, c))) {
                    Ok((b, coverage)) => {
                        let bytes = b.proof.len() as u32 - 8;
                        self.proofs.insert(made.0, (b.proof.to_vec(), b.claims.to_vec()));
                        self.made.push(made);
                        Outcome::Proved { coverage: Box::new(coverage), rate, bytes }
                    }
                    Err(e) => Outcome::Failed(Failure::Rejected(e)),
                },
                ST_INVALID_PROOF if (r.detail as usize) < inf.children.len() => Outcome::Failed(Failure::InvalidChild(inf.children[r.detail as usize])),
                _ => Outcome::Failed(Failure::Rejected(status_text(&r))),
            },
            What::Verify => Outcome::Verified(r.status == ST_OK),
        };
        Done { lane: 0, at, outcome }
    }
}

impl Backend for LiveBackend {
    fn caps(&self) -> Caps {
        Caps { lanes: 1, abort: false, verify_lane: false, max_children: 16, max_new_sigs: 2048 }
    }

    fn model(&self) -> &dyn CostModel {
        self.model.as_ref()
    }

    fn now(&self) -> Micros {
        self.elapsed()
    }

    fn begin_slot(&mut self, set: &ScheduleSet, _run: u32) {
        // The signature file lays subnets out by the set's subnet size; a set the file cannot cover is refused.
        self.table.set_subnet_size(set.meta.subnet_size, set.meta.subnets).unwrap_or_else(|e| panic!("live backend: {e}"));
        if set.meta.shape == "merge" {
            self.ensure_pool(set).unwrap_or_else(|e| panic!("live backend: {e}"));
        }
        std::thread::sleep(SETTLE);
        self.proofs.clear();
        self.made.clear();
        self.t0 = Instant::now();
        self.first_wait = true;
    }

    fn receive(&mut self, id: ProofId, pool_entry: &PoolEntry, valid: bool) {
        let (proof, mut claims) = self.pool.get(&pool_entry.id).expect("pool proof was built").clone();
        if !valid {
            // A valid proof sent with one claim changed.
            if let Some(last) = claims.last_mut() {
                *last ^= 1;
            }
        }
        self.proofs.insert(id.0, (proof, claims));
    }

    fn start_prove(&mut self, lane: usize, job: &Job, store: &ProofStore, made: ProofId) -> Started {
        assert!(lane == 0 && self.inflight.is_none(), "one request at a time");
        let children: Vec<Child> = job.children.iter().map(|&c| self.child(c)).collect();
        let (op, payload) = match (job.children.is_empty(), job.signatures.is_empty()) {
            (true, _) => (OP_AGGREGATE, signatures_payload(&self.table, &job.signatures)),
            (false, true) => (OP_RECURSE, recurse_payload(&children)),
            _ => (OP_EXTEND, extend_payload(&children, &self.table, &job.signatures)),
        };
        let unverified = job.children.iter().any(|&c| !store.get(c).info.verified);
        let flags = if unverified && op != OP_AGGREGATE { FLAG_VERIFY_CHILDREN } else { 0 };
        let shape = JobShape {
            children: job.children.iter().map(|&c| crate::sim::child_shape(&store.get(c).info, !store.get(c).info.verified)).collect(),
            new_sigs: job.signatures.len() as u32,
            rate: job.rate,
        };
        match self.client.send(op, flags, job.rate as u32, &payload, RESP_CAP) {
            Ok(id) => self.inflight = Some(Inflight { id, what: What::Prove { made, rate: job.rate }, children: job.children.clone(), epoch: self.client.epoch(), sent: Instant::now() }),
            Err(e) => self.rejected = Some(e.to_string()),
        }
        Started { predicted: self.model.prove_us(&shape, THREADS), verify_us: 0 }
    }

    fn start_verify(&mut self, lane: usize, proof: ProofId, store: &ProofStore) -> Started {
        assert!(lane == 0 && self.inflight.is_none(), "one request at a time");
        let payload = verify_payload(&self.child(proof));
        match self.client.send(OP_VERIFY, 0, 0, &payload, RESP_CAP) {
            Ok(id) => self.inflight = Some(Inflight { id, what: What::Verify, children: Vec::new(), epoch: self.client.epoch(), sent: Instant::now() }),
            Err(e) => self.rejected = Some(e.to_string()),
        }
        let us = self.model.verify_us(&crate::sim::child_shape(&store.get(proof).info, true));
        Started { predicted: us, verify_us: us }
    }

    fn abort(&mut self, _lane: usize) -> bool {
        false
    }

    fn wait(&mut self, until: Micros) -> Option<Done> {
        let first = std::mem::replace(&mut self.first_wait, false);
        if first && self.skip_idle && self.inflight.is_none() && self.rejected.is_none() && until != Micros::MAX {
            let now = self.elapsed();
            if until > now + 100_000 {
                self.t0 -= Duration::from_micros(until - 100_000 - now);
            }
        }
        if let Some(why) = self.rejected.take() {
            let at = self.elapsed();
            return Some(Done { lane: 0, at, outcome: Outcome::Failed(Failure::Rejected(why)) });
        }
        loop {
            if let Some(inf) = self.inflight.take() {
                if let Some(r) = self.client.try_recv() {
                    if r.id < inf.id {
                        // A response to a request sent before a restart: the server still served it. Drop it.
                        self.inflight = Some(inf);
                        continue;
                    }
                    if r.id > inf.id {
                        let at = self.elapsed();
                        return Some(Done { lane: 0, at, outcome: Outcome::Failed(Failure::Rejected(format!("response {} for request {}", r.id, inf.id))) });
                    }
                    return Some(self.finish(inf, r));
                }
                if self.client.epoch() != inf.epoch {
                    let at = self.elapsed();
                    return Some(Done { lane: 0, at, outcome: Outcome::Failed(Failure::Rejected("the sidecar restarted".into())) });
                }
                assert!(inf.sent.elapsed() < HUNG, "no response from the sidecar");
                self.inflight = Some(inf);
            } else if until == Micros::MAX {
                return None;
            }
            let now = self.elapsed();
            if now >= until {
                self.lateness_us.push(now - until);
                return None;
            }
            std::thread::sleep(POLL);
        }
    }
}

fn loadavg() -> f64 {
    std::fs::read_to_string("/proc/loadavg").ok().and_then(|s| s.split_whitespace().next()?.parse().ok()).unwrap_or(0.0)
}

/// `streamsim live --set DIR --model FILE --channel PATH --strategy S --slots a..b --out FILE`
pub fn run(a: &Args) -> Result<(), String> {
    if loadavg() > 2.0 && !a.flag("force") {
        return Err(format!("load average {:.1} is above 2: another job is running (use --force to override)", loadavg()));
    }
    let lock = Path::new("target/streaming-live.lock");
    if let Ok(pid) = std::fs::read_to_string(lock)
        && Path::new(&format!("/proc/{}", pid.trim())).exists()
    {
        return Err(format!("another live session is running (pid {})", pid.trim()));
    }
    let _ = std::fs::create_dir_all("target");
    std::fs::write(lock, std::process::id().to_string()).map_err(|e| e.to_string())?;
    let result = run_locked(a);
    let _ = std::fs::remove_file(lock);
    result
}

fn run_locked(a: &Args) -> Result<(), String> {
    let set_dir = a.need("set")?;
    let set = ScheduleSet::read(Path::new(set_dir))?;
    let (model, model_backend) = load_args(a, Path::new(a.need("model")?))?;
    let label = a.get("backend").map_or(model_backend, str::to_string);
    let signatures = a.get("signatures").unwrap_or("bins/leanvm-sidecar/testdata/signatures-24000.bin");
    let cache = a.get("cache").unwrap_or("target/streaming-cache");
    let shape = if set.meta.shape == "merge" { Shape::Merge } else { Shape::Leaf };
    let deadlines = match a.get("deadline-ms") {
        Some(t) => parse_deadlines(t)?,
        None => vec![set.meta.deadline_us],
    };
    let slots = parse_range(a.get("slots").unwrap_or("100000"), set.meta.slots)?;
    let mut spec = Spec::parse(a.need("strategy")?)?;
    if spec.shape() != shape {
        return Err(format!("strategy {} does not fit a {:?} set", spec.name, shape));
    }
    if shape == Shape::Merge {
        spec.default_param("verify", a.get("verify-policy").unwrap_or("all_idle"));
    }
    let mut backend = LiveBackend::open(Path::new(a.need("channel")?), Path::new(signatures), model, &label, Path::new(cache))?;
    backend.skip_idle = !a.flag("no-skip-idle");
    let set_name = Path::new(set_dir).file_name().map_or(set_dir.to_string(), |s| s.to_string_lossy().into());
    let job = Study { spec, forecast: load_forecast(a)?, cost: cost_kind(a)?, grace: grace_us(a)? };
    let results = run_set(&mut backend, &set, &set_name, &label, &job, &deadlines, slots, |r| eprintln!("slot {} D {} ms: coverage {:.3}", r.row.slot, r.row.deadline_us / 1000, r.row.m.coverage))?;
    let mut text = format!("{ROW_HEADER}\n");
    let mut jobs = vec![JOBS_HEADER.to_string()];
    let mut rows = Vec::new();
    for r in results {
        text += &r.row.to_csv();
        text.push('\n');
        jobs.extend(r.record.jobs.iter().map(|j| job_line(r.row.slot, j)));
        rows.push(r.row);
    }
    std::fs::write(a.need("out")?, text).map_err(|e| e.to_string())?;
    if let Some(p) = a.get("jobs") {
        std::fs::write(p, jobs.join("\n") + "\n").map_err(|e| e.to_string())?;
    }
    let late = &backend.lateness_us;
    if !late.is_empty() {
        let mut v = late.clone();
        v.sort_unstable();
        println!("wake-up lateness: median {} us, max {} us over {} waits", v[v.len() / 2], v[v.len() - 1], v.len());
    }
    print!("{}", report(&rows, a.get("baseline"), false));
    Ok(())
}
