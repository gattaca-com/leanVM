//! The arrival model: a pipeline chosen and tuned by a parameter file.
//!
//! ```text
//! votes:       population, presence, base time, lateness, subnet skew, per-view loss and jitter, transforms, slot jitter
//! aggregates:  candidates per subnet, base time or composed time, subnet skew, transforms, slot jitter
//! ```
//!
//! `shape leaf` generates the votes of `subnets` subnets, each seen by `agg.per_subnet` views. The
//! default is one subnet (`subnets 1`); a set for `streamsim e2e` asks for 16.

use crate::compose::{ComposeEntry, read_compose_file};
use crate::layers::{Compiled, Item, Transform};
use crate::rng::SplitMix64;
use crate::sampler::{Dir, Env, Sampler, parse_sampler};
use crate::schedule::{Arrival, Kind, Meta, PoolEntry, ScheduleSet, SlotSchedule, parse_key_values, sort_arrivals};
use crate::xmss;
use std::path::{Path, PathBuf};

pub struct Params {
    entries: Vec<(String, String)>,
    pub dir: PathBuf,
    /// The text as read, then any overrides appended as `key value` lines.
    pub text: String,
}

impl Params {
    pub fn parse(text: &str, dir: &Path) -> Params {
        Params { entries: parse_key_values(text), dir: dir.to_path_buf(), text: text.to_string() }
    }

    pub fn from_file(path: &Path) -> Result<Params, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Params::parse(&text, path.parent().unwrap_or(Path::new("."))))
    }

    /// The last value given for `key`.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    pub fn set(&mut self, key: &str, value: &str) {
        self.entries.push((key.to_string(), value.to_string()));
        if !self.text.ends_with('\n') && !self.text.is_empty() {
            self.text.push('\n');
        }
        self.text.push_str(&format!("{key} {value}\n"));
    }

    fn num(&self, key: &str, default: f64) -> Result<f64, String> {
        match self.get(key) {
            Some(v) => v.parse().map_err(|_| format!("{key}: not a number: {v}")),
            None => Ok(default),
        }
    }
}

pub trait ArrivalModel {
    /// The schedule of slot `slot`. Depends on `(seed, slot)` only: slots can be generated in any order.
    fn slot(&self, seed: u64, slot: u32) -> SlotSchedule;
    /// Aggregates refer to these. Empty for a votes-only model.
    fn pool(&self, seed: u64) -> Vec<PoolEntry>;
    /// Layout, default deadline and counts for `meta.txt`. `slots` and `seed` are filled in by [`generate`].
    fn meta(&self) -> Meta;
}

type Boxed = Box<dyn Sampler>;

struct Chain(Vec<Compiled>);

impl Chain {
    fn apply(&self, items: &mut [Item], rng: &mut SplitMix64) {
        self.0.iter().for_each(|t| t.apply(items, rng));
    }
}

struct Shared {
    subnet_skew: Option<Boxed>,
    slot_jitter: Option<Boxed>,
    transforms: Chain,
}

struct Pipeline {
    leaf: bool,
    subnets: u32,
    subnet_size: u32,
    views: u32,
    meta: Meta,
    att_due_ms: f64,
    votes_base: Option<Boxed>,
    votes_missing: f64,
    late_frac: f64,
    late_extra: Option<Boxed>,
    timer_frac: f64,
    timer_delay: Option<Boxed>,
    view_jitter: Option<Boxed>,
    view_missing: f64,
    votes: Shared,
    pool_sizes: Vec<u32>,
    agg_rate: u8,
    agg_base: Option<Boxed>,
    compose: Option<Vec<ComposeEntry>>,
    agg_delay: Option<Boxed>,
    agg_missing: f64,
    agg_invalid: f64,
    aggs: Shared,
}

fn layout_ms(slot_ms: f64, bps: f64) -> f64 {
    ((bps * slot_ms + 5000.0) / 10000.0).floor()
}

fn us(ms: f64) -> u64 {
    (ms.max(0.0) * 1000.0).round() as u64
}

fn opt_sampler(p: &Params, key: &str, env: &Env, dir: &Dir) -> Result<Option<Boxed>, String> {
    match p.get(key) {
        None | Some("none") | Some("") => Ok(None),
        Some(e) => parse_sampler(e, env, dir).map(Some).map_err(|e| format!("{key}: {e}")),
    }
}

fn chain(texts: Vec<Transform>, env: &Env, dir: &Dir) -> Result<Chain, String> {
    Ok(Chain(texts.iter().map(|t| t.compile(env, dir)).collect::<Result<_, _>>()?))
}

fn transforms(p: &Params, key: &str, extra: Vec<Transform>, env: &Env, dir: &Dir) -> Result<Chain, String> {
    let mut list = match p.get(key) {
        Some(t) => Transform::parse_list(t, env)?,
        None => Vec::new(),
    };
    list.extend(extra);
    chain(list, env, dir)
}

/// The only constructor.
pub fn build(params: &Params) -> Result<Box<dyn ArrivalModel>, String> {
    let leaf = match params.get("shape") {
        Some("leaf") => true,
        Some("merge") => false,
        _ => return Err("shape must be `leaf` or `merge`".into()),
    };
    let slot_ms = params.num("slot_ms", 12000.0)?;
    let att = layout_ms(slot_ms, params.num("att_due_bps", 2500.0)?);
    let agg = layout_ms(slot_ms, params.num("agg_due_bps", 5000.0)?);
    let ptc = layout_ms(slot_ms, params.num("ptc_due_bps", 7500.0)?);
    let mut env = Env::new();
    for (k, v) in parse_key_values(&params.text) {
        if let Ok(x) = v.parse::<f64>() {
            env.insert(k, x);
        }
    }
    for (k, v) in [("att_due", att), ("agg_due", agg), ("ptc_due", ptc), ("slot_end", slot_ms), ("slot_ms", slot_ms)] {
        env.insert(k.to_string(), v);
    }
    let dir = Dir(params.dir.clone());
    let subnets = params.num("subnets", if leaf { 1.0 } else { 16.0 })? as u32;
    let subnet_size = params.num("subnet_size", 1500.0)? as u32;
    let views = params.num("agg.per_subnet", 3.0)? as u32;
    let deadline_ms = match params.get("deadline") {
        None => if leaf { agg } else { slot_ms },
        Some("att_due") => att,
        Some("agg_due") => agg,
        Some("ptc_due") => ptc,
        Some("slot_end") => slot_ms,
        Some(v) => v.parse().map_err(|_| format!("deadline: {v}"))?,
    };
    let horizon_ms = params.num("horizon_ms", slot_ms + 4000.0)?;
    let meta = Meta {
        shape: if leaf { "leaf" } else { "merge" }.into(),
        slots: 0,
        slot_us: us(slot_ms),
        att_due_us: us(att),
        agg_due_us: us(agg),
        ptc_due_us: us(ptc),
        deadline_us: us(deadline_ms),
        horizon_us: us(horizon_ms),
        final_rate: params.num("final_rate", if leaf { 2.0 } else { 1.0 })? as u8,
        expected_inputs: if leaf { subnet_size } else { subnets },
        seed: 0,
        source: "generated".into(),
        subnet_size,
        subnets,
        views: if leaf { views } else { 1 },
    };
    let xm = xmss::expand(&|k| params.get(&format!("xmss.{k}")).map(str::to_string), &env)?;
    let votes_base = opt_sampler(params, "votes.base", &env, &dir)?;
    if leaf && votes_base.is_none() {
        return Err("shape leaf needs votes.base".into());
    }
    let compose = match params.get("agg.compose") {
        Some(f) if f != "none" => {
            let p = Path::new(f);
            let path = if p.is_absolute() || !params.dir.join(p).exists() { p.to_path_buf() } else { params.dir.join(p) };
            Some(read_compose_file(&path)?)
        }
        _ => None,
    };
    let agg_base = opt_sampler(params, "agg.base", &env, &dir)?;
    if !leaf && agg_base.is_none() && compose.is_none() {
        return Err("shape merge needs agg.base or agg.compose".into());
    }
    let pool_sizes = match params.get("agg.pool_sizes") {
        Some(v) => v.split(',').map(|x| x.trim().parse().map_err(|_| format!("agg.pool_sizes: {x}"))).collect::<Result<_, _>>()?,
        None => vec![1500, 1485, 1455, 1400],
    };
    Ok(Box::new(Pipeline {
        leaf,
        subnets,
        subnet_size,
        views,
        meta,
        att_due_ms: att,
        votes_base,
        votes_missing: params.num("votes.missing", 0.0)?,
        late_frac: params.num("votes.late_frac", 0.0)?,
        late_extra: opt_sampler(params, "votes.late_extra", &env, &dir)?,
        timer_frac: params.num("votes.timer_frac", 0.0)?,
        timer_delay: opt_sampler(params, "votes.timer_delay", &env, &dir)?,
        view_jitter: opt_sampler(params, "votes.view_jitter", &env, &dir)?,
        view_missing: params.num("votes.view_missing", 0.0)?,
        votes: Shared {
            subnet_skew: opt_sampler(params, "votes.subnet_skew", &env, &dir)?,
            slot_jitter: opt_sampler(params, "votes.slot_jitter", &env, &dir)?,
            transforms: transforms(params, "votes.transforms", xm.votes, &env, &dir)?,
        },
        pool_sizes,
        agg_rate: params.num("agg.rate", 2.0)? as u8,
        agg_base,
        compose,
        agg_delay: opt_sampler(params, "agg.delay", &env, &dir)?,
        agg_missing: params.num("agg.missing", 0.0)?,
        agg_invalid: params.num("agg.invalid", 0.0)?,
        aggs: Shared {
            subnet_skew: opt_sampler(params, "agg.subnet_skew", &env, &dir)?,
            slot_jitter: opt_sampler(params, "agg.slot_jitter", &env, &dir)?,
            transforms: transforms(params, "agg.transforms", xm.aggregates, &env, &dir)?,
        },
    }))
}

impl Pipeline {
    fn words(&self) -> usize {
        (self.subnet_size as usize).div_ceil(64)
    }

    fn finish(&self, kind: Kind, mut items: Vec<Item>, jitter: f64) -> Vec<Arrival> {
        let mut out: Vec<Arrival> = items
            .drain(..)
            .map(|i| Arrival { t_us: us(i.t + jitter), kind, subnet: i.subnet, view: i.view, id: i.id, valid: i.valid })
            .collect();
        sort_arrivals(&mut out);
        out
    }

    fn slot_jitter(shared: &Shared, seed: u64, slot: u32, tag: u64) -> f64 {
        shared.slot_jitter.as_ref().map_or(0.0, |s| s.sample(&mut SplitMix64::derive(seed, slot as u64, tag)))
    }

    fn skew(shared: &Shared, seed: u64, slot: u32, tag: u64) -> f64 {
        shared.subnet_skew.as_ref().map_or(0.0, |s| s.sample(&mut SplitMix64::derive(seed, slot as u64, tag)))
    }

    fn vote_slot(&self, seed: u64, slot: u32) -> Vec<Arrival> {
        let base = self.votes_base.as_ref().expect("leaf has votes.base");
        let mut out = Vec::new();
        for subnet in 0..self.subnets {
            let mut rng = SplitMix64::derive(seed, slot as u64, 1000 + subnet as u64);
            let skew = Self::skew(&self.votes, seed, slot, 2000 + subnet as u64);
            let mut sent = Vec::new();
            for id in 0..self.subnet_size {
                if self.votes_missing > 0.0 && rng.f64() < self.votes_missing {
                    continue;
                }
                let mut t = base.sample(&mut rng);
                if self.late_frac > 0.0 || self.timer_frac > 0.0 {
                    let u = rng.f64();
                    if u < self.timer_frac {
                        t = self.att_due_ms + self.timer_delay.as_ref().map_or(0.0, |s| s.sample(&mut rng));
                    } else if u < self.timer_frac + self.late_frac {
                        t += self.late_extra.as_ref().map_or(0.0, |s| s.sample(&mut rng));
                    }
                }
                sent.push((id, t + skew));
            }
            for view in 0..self.views {
                let mut rv = SplitMix64::derive(seed, slot as u64, 3000 + subnet as u64 * 64 + view as u64);
                let mut items = Vec::with_capacity(sent.len());
                for &(id, t) in &sent {
                    if self.view_missing > 0.0 && rv.f64() < self.view_missing {
                        continue;
                    }
                    let extra = self.view_jitter.as_ref().map_or(0.0, |s| s.sample(&mut rv));
                    items.push(Item { t: t + extra, subnet: subnet as u8, view: view as u8, id, valid: true });
                }
                self.votes.transforms.apply(&mut items, &mut rv);
                out.extend(items);
            }
        }
        let jitter = Self::slot_jitter(&self.votes, seed, slot, 4000);
        self.finish(Kind::Vote, out, jitter)
    }

    fn agg_slot(&self, seed: u64, slot: u32) -> Vec<Arrival> {
        let pool = self.pool(seed);
        let mut items = Vec::new();
        for e in pool.iter().filter(|e| e.slot.is_none_or(|s| s == slot)) {
            let mut rng = SplitMix64::derive(seed, slot as u64, 5000 + e.id as u64);
            if self.agg_missing > 0.0 && rng.f64() < self.agg_missing {
                continue;
            }
            let valid = !(self.agg_invalid > 0.0 && rng.f64() < self.agg_invalid);
            let t = match &self.compose {
                Some(c) => {
                    let delay = self.agg_delay.as_ref().map_or(0.0, |s| s.sample(&mut rng));
                    c[e.id as usize].published_us as f64 / 1000.0 + delay
                }
                None => self.agg_base.as_ref().expect("merge has agg.base").sample(&mut rng),
            };
            let t = t + Self::skew(&self.aggs, seed, slot, 6000 + e.subnet as u64);
            let view = match &self.compose {
                Some(c) => c[e.id as usize].view,
                None => (e.id % self.per_subnet()) as u8,
            };
            items.push(Item { t, subnet: e.subnet, view, id: e.id, valid });
        }
        let mut rng = SplitMix64::derive(seed, slot as u64, 7000);
        self.aggs.transforms.apply(&mut items, &mut rng);
        let jitter = Self::slot_jitter(&self.aggs, seed, slot, 8000);
        self.finish(Kind::Aggregate, items, jitter)
    }

    fn per_subnet(&self) -> u32 {
        self.views.max(1)
    }
}

impl ArrivalModel for Pipeline {
    fn slot(&self, seed: u64, slot: u32) -> SlotSchedule {
        SlotSchedule { arrivals: if self.leaf { self.vote_slot(seed, slot) } else { self.agg_slot(seed, slot) } }
    }

    fn pool(&self, seed: u64) -> Vec<PoolEntry> {
        if self.leaf {
            return Vec::new();
        }
        if let Some(c) = &self.compose {
            return c
                .iter()
                .enumerate()
                .map(|(i, e)| PoolEntry { subnet: e.subnet, id: i as u32, signers: e.signers.clone(), rate: e.rate, slot: Some(e.slot) })
                .collect();
        }
        let mut pool = Vec::new();
        for subnet in 0..self.subnets {
            for c in 0..self.per_subnet() {
                let id = subnet * self.per_subnet() + c;
                let size = self.pool_sizes[c as usize % self.pool_sizes.len()].min(self.subnet_size);
                let mut rng = SplitMix64::derive(seed, 9_000_000, id as u64);
                let mut order: Vec<u32> = (0..self.subnet_size).collect();
                let mut signers = vec![0u64; self.words()];
                for i in 0..size as usize {
                    let j = i + rng.below((order.len() - i) as u64) as usize;
                    order.swap(i, j);
                    let s = order[i] as usize;
                    signers[s / 64] |= 1 << (s % 64);
                }
                pool.push(PoolEntry { subnet: subnet as u8, id, signers, rate: self.agg_rate, slot: None });
            }
        }
        pool
    }

    fn meta(&self) -> Meta {
        self.meta.clone()
    }
}

/// The transform lists of votes and aggregates, as text, for `arrivalgen --explain`.
pub fn explain(params: &Params) -> Result<Vec<(String, Vec<String>)>, String> {
    let mut env = Env::new();
    for (k, v) in parse_key_values(&params.text) {
        if let Ok(x) = v.parse::<f64>() {
            env.insert(k, x);
        }
    }
    let xm = xmss::expand(&|k| params.get(&format!("xmss.{k}")).map(str::to_string), &env)?;
    let show = |key: &str, extra: &[Transform]| -> Result<Vec<String>, String> {
        let mut list = match params.get(key) {
            Some(t) => Transform::parse_list(t, &env)?,
            None => Vec::new(),
        };
        list.extend(extra.iter().cloned());
        Ok(list.iter().map(|t| t.to_string()).collect())
    };
    Ok(vec![("votes".into(), show("votes.transforms", &xm.votes)?), ("aggregates".into(), show("agg.transforms", &xm.aggregates)?)])
}

/// A whole schedule set: `slots` slots from `seed`.
pub fn generate(model: &dyn ArrivalModel, params: &Params, slots: u32, seed: u64) -> ScheduleSet {
    let mut meta = model.meta();
    meta.slots = slots;
    meta.seed = seed;
    ScheduleSet {
        meta,
        params_txt: Some(params.text.clone()),
        pool: model.pool(seed),
        slots: (0..slots).map(|s| model.slot(seed, s)).collect(),
    }
}
