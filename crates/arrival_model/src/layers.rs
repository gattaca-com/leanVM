//! Generic transforms of a slot's arrival list. Nothing here is specific to a signature scheme.

use crate::rng::SplitMix64;
use crate::sampler::{Dir, Env, Sampler, call, eval_num, parse_sampler, split_args};
use std::fmt;

/// An arrival while it is being shaped: time in milliseconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Item {
    pub t: f64,
    pub subnet: u8,
    pub view: u8,
    pub id: u32,
    pub valid: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Transform {
    Shift(f64),
    Stretch(f64, f64),
    /// Sampler expression.
    Delay(String),
    /// Hop-count sampler expression, milliseconds per hop.
    Hops(String, f64),
    Cap(f64),
    CapPerSubnet(f64),
}

impl fmt::Display for Transform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Transform::Shift(c) => write!(f, "shift({c})"),
            Transform::Stretch(t0, s) => write!(f, "stretch({t0}, {s})"),
            Transform::Delay(e) => write!(f, "delay({e})"),
            Transform::Hops(e, ms) => write!(f, "hops({e}, {ms})"),
            Transform::Cap(r) => write!(f, "cap({r})"),
            Transform::CapPerSubnet(r) => write!(f, "cap_per_subnet({r})"),
        }
    }
}

impl Transform {
    pub fn parse(text: &str, env: &Env) -> Result<Transform, String> {
        let (name, inner) = call(text).ok_or(format!("not a transform: `{text}`"))?;
        let args = split_args(inner);
        let n = |i: usize| -> Result<f64, String> { eval_num(args.get(i).ok_or(format!("`{text}`: missing argument"))?, env) };
        Ok(match name {
            "shift" => Transform::Shift(n(0)?),
            "stretch" => Transform::Stretch(n(0)?, n(1)?),
            "delay" => Transform::Delay(inner.trim().to_string()),
            "hops" => Transform::Hops(args.first().ok_or("`hops` needs a sampler")?.to_string(), n(1)?),
            "cap" => Transform::Cap(n(0)?),
            "cap_per_subnet" => Transform::CapPerSubnet(n(0)?),
            _ => return Err(format!("unknown transform `{name}`")),
        })
    }

    pub fn parse_list(text: &str, env: &Env) -> Result<Vec<Transform>, String> {
        text.split(';').map(str::trim).filter(|s| !s.is_empty() && *s != "none").map(|s| Transform::parse(s, env)).collect()
    }

    pub fn compile(&self, env: &Env, dir: &Dir) -> Result<Compiled, String> {
        Ok(match self {
            Transform::Shift(c) => Compiled::Shift(*c),
            Transform::Stretch(t0, s) => Compiled::Stretch(*t0, *s),
            Transform::Delay(e) => Compiled::Delay(parse_sampler(e, env, dir)?),
            Transform::Hops(e, ms) => Compiled::Hops(parse_sampler(e, env, dir)?, *ms),
            Transform::Cap(r) => Compiled::Cap(*r, false),
            Transform::CapPerSubnet(r) => Compiled::Cap(*r, true),
        })
    }
}

pub enum Compiled {
    Shift(f64),
    Stretch(f64, f64),
    Delay(Box<dyn Sampler>),
    Hops(Box<dyn Sampler>, f64),
    Cap(f64, bool),
}

/// A FIFO with the given service rate, in time order: `t'_i = max(t_i, t'_{i-1} + 1/rate)`.
fn cap(items: &mut [Item], rate_per_s: f64) {
    let gap = 1000.0 / rate_per_s;
    let mut prev = f64::NEG_INFINITY;
    for it in items.iter_mut() {
        it.t = it.t.max(prev + gap);
        prev = it.t;
    }
}

fn by_time(items: &mut [Item]) {
    items.sort_by(|a, b| a.t.total_cmp(&b.t));
}

impl Compiled {
    pub fn apply(&self, items: &mut [Item], rng: &mut SplitMix64) {
        match self {
            Compiled::Shift(c) => items.iter_mut().for_each(|i| i.t += c),
            Compiled::Stretch(t0, s) => items.iter_mut().for_each(|i| i.t = t0 + s * (i.t - t0)),
            Compiled::Delay(s) => items.iter_mut().for_each(|i| i.t += s.sample(rng)),
            Compiled::Hops(s, ms) => items.iter_mut().for_each(|i| i.t += s.sample(rng).round().max(0.0) * ms),
            Compiled::Cap(rate, per_subnet) => {
                by_time(items);
                if *per_subnet {
                    let mut subnets: Vec<u8> = items.iter().map(|i| i.subnet).collect();
                    subnets.sort_unstable();
                    subnets.dedup();
                    for s in subnets {
                        let mut group: Vec<Item> = items.iter().filter(|i| i.subnet == s).copied().collect();
                        cap(&mut group, *rate);
                        let mut g = group.into_iter();
                        for it in items.iter_mut().filter(|i| i.subnet == s) {
                            *it = g.next().unwrap();
                        }
                    }
                } else {
                    cap(items, *rate);
                }
                by_time(items);
            }
        }
    }
}
