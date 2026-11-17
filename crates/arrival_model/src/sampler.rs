//! Base samplers: expressions such as `lognormal(300, 0.4)` or `mix(0.6: normal(300, 60), 0.4: const(900))`.
//! Every sample is in milliseconds.

use crate::rng::SplitMix64;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub trait Sampler {
    fn sample(&self, rng: &mut SplitMix64) -> f64;
}

/// Names usable in a numeric argument, such as `att_due` or `t_open`.
pub type Env = HashMap<String, f64>;

/// Where file names in an expression are looked up: the given directory, then the working directory.
#[derive(Clone, Debug, Default)]
pub struct Dir(pub PathBuf);

impl Dir {
    fn resolve(&self, name: &str) -> PathBuf {
        let p = Path::new(name);
        if p.is_absolute() || !self.0.join(p).exists() {
            p.to_path_buf()
        } else {
            self.0.join(p)
        }
    }
}

enum S {
    Const(f64),
    Uniform(f64, f64),
    Normal(f64, f64),
    LogNormal(f64, f64),
    Exp(f64),
    Pareto(f64, f64),
    /// `(t, cdf)` points.
    Cdf(Vec<(f64, f64)>),
    /// `(u, cdf)` points, mapped to `t_open + u * span`.
    Scaled(Vec<(f64, f64)>, f64, f64),
    Mix(Vec<(f64, S)>),
    Sum(Box<S>, Box<S>),
    Clamp(Box<S>, f64, f64),
}

/// The value at cumulative probability `p` of a `(value, cdf)` table, linear between points.
pub fn inverse_cdf(points: &[(f64, f64)], p: f64) -> f64 {
    let first = points[0];
    if p <= first.1 {
        return first.0;
    }
    for w in points.windows(2) {
        let ((v0, c0), (v1, c1)) = (w[0], w[1]);
        if p <= c1 {
            return if c1 > c0 { v0 + (v1 - v0) * (p - c0) / (c1 - c0) } else { v1 };
        }
    }
    points[points.len() - 1].0
}

impl Sampler for S {
    fn sample(&self, rng: &mut SplitMix64) -> f64 {
        match self {
            S::Const(c) => *c,
            S::Uniform(lo, hi) => lo + (hi - lo) * rng.f64(),
            S::Normal(mu, sigma) => mu + sigma * rng.normal(),
            S::LogNormal(median, sigma) => median * (sigma * rng.normal()).exp(),
            S::Exp(mean) => -mean * (1.0 - rng.f64()).ln(),
            S::Pareto(scale, alpha) => scale * (1.0 - rng.f64()).powf(-1.0 / alpha),
            S::Cdf(points) => inverse_cdf(points, rng.f64()),
            S::Scaled(points, t_open, span) => t_open + inverse_cdf(points, rng.f64()) * span,
            S::Mix(parts) => {
                let mut u = rng.f64();
                for (p, s) in parts {
                    if u < *p {
                        return s.sample(rng);
                    }
                    u -= p;
                }
                parts[parts.len() - 1].1.sample(rng)
            }
            S::Sum(a, b) => a.sample(rng) + b.sample(rng),
            S::Clamp(s, lo, hi) => s.sample(rng).clamp(*lo, *hi),
        }
    }
}

/// Split at commas that are not inside parentheses.
pub fn split_args(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut depth, mut start) = (0i32, 0);
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    let last = s[start..].trim();
    if !last.is_empty() || !out.is_empty() {
        out.push(last);
    }
    out
}

/// `name(args)` as `(name, args)`.
pub fn call(s: &str) -> Option<(&str, &str)> {
    let s = s.trim();
    let open = s.find('(')?;
    let inner = s[open + 1..].strip_suffix(')')?;
    Some((s[..open].trim(), inner))
}

/// A number, a name from `env`, or sums, differences and products of them.
pub fn eval_num(s: &str, env: &Env) -> Result<f64, String> {
    let s = s.trim();
    let b = s.as_bytes();
    for i in (1..b.len()).rev() {
        let exponent = matches!(b[i - 1], b'e' | b'E') && i >= 2 && (b[i - 2].is_ascii_digit() || b[i - 2] == b'.');
        if (b[i] == b'+' || b[i] == b'-') && !exponent && !matches!(b[i - 1], b'+' | b'-' | b'*') {
            let (l, r) = (eval_num(&s[..i], env)?, eval_num(&s[i + 1..], env)?);
            return Ok(if b[i] == b'+' { l + r } else { l - r });
        }
    }
    if let Some(i) = s.rfind('*') {
        return Ok(eval_num(&s[..i], env)? * eval_num(&s[i + 1..], env)?);
    }
    if let Some(rest) = s.strip_prefix('-') {
        return Ok(-eval_num(rest, env)?);
    }
    s.parse::<f64>().or_else(|_| env.get(s).copied().ok_or(format!("unknown number or name `{s}`")))
}

fn read_points(path: &Path) -> Result<Vec<(f64, f64)>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut pts = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 2 {
            continue;
        }
        if let (Ok(a), Ok(b)) = (f[f.len() - 2].trim().parse::<f64>(), f[f.len() - 1].trim().parse::<f64>()) {
            pts.push((a, b));
        }
    }
    if pts.is_empty() {
        return Err(format!("{}: no points", path.display()));
    }
    Ok(pts)
}

fn build(expr: &str, env: &Env, dir: &Dir) -> Result<S, String> {
    let (name, inner) = call(expr).ok_or(format!("not a sampler expression: `{expr}`"))?;
    let args = split_args(inner);
    let nums = |n: usize| -> Result<Vec<f64>, String> {
        if args.len() != n {
            return Err(format!("`{name}` takes {n} arguments: `{expr}`"));
        }
        args.iter().map(|a| eval_num(a, env)).collect()
    };
    Ok(match name {
        "const" => S::Const(nums(1)?[0]),
        "uniform" => {
            let v = nums(2)?;
            S::Uniform(v[0], v[1])
        }
        "normal" => {
            let v = nums(2)?;
            S::Normal(v[0], v[1])
        }
        "lognormal" => {
            let v = nums(2)?;
            S::LogNormal(v[0], v[1])
        }
        "exp" => S::Exp(nums(1)?[0]),
        "pareto" => {
            let v = nums(2)?;
            S::Pareto(v[0], v[1])
        }
        "cdf" => {
            if args.len() != 1 {
                return Err(format!("`cdf` takes a file: `{expr}`"));
            }
            S::Cdf(read_points(&dir.resolve(args[0]))?)
        }
        "scaled" => {
            if args.len() != 3 {
                return Err(format!("`scaled` takes a file, t_open and T: `{expr}`"));
            }
            S::Scaled(read_points(&dir.resolve(args[0]))?, eval_num(args[1], env)?, eval_num(args[2], env)?)
        }
        "mix" => {
            let mut parts = Vec::new();
            for a in &args {
                let (p, s) = a.split_once(':').ok_or(format!("`mix` part needs `p: sampler`: `{a}`"))?;
                parts.push((eval_num(p, env)?, build(s, env, dir)?));
            }
            if parts.is_empty() {
                return Err("`mix` has no parts".into());
            }
            S::Mix(parts)
        }
        "sum" => {
            if args.len() != 2 {
                return Err(format!("`sum` takes 2 samplers: `{expr}`"));
            }
            S::Sum(Box::new(build(args[0], env, dir)?), Box::new(build(args[1], env, dir)?))
        }
        "clamp" => {
            if args.len() != 3 {
                return Err(format!("`clamp` takes a sampler, lo and hi: `{expr}`"));
            }
            S::Clamp(Box::new(build(args[0], env, dir)?), eval_num(args[1], env)?, eval_num(args[2], env)?)
        }
        _ => return Err(format!("unknown sampler `{name}`")),
    })
}

pub fn parse_sampler(expr: &str, env: &Env, dir: &Dir) -> Result<Box<dyn Sampler>, String> {
    Ok(Box::new(build(expr, env, dir)?))
}
