//! Command-line helpers shared by the `streamsim` commands.

use crate::cost::{CostModel, Kv, load_model};
use crate::forecast::Forecast;
use crate::types::Micros;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

/// `--key value` pairs and bare flags.
pub struct Args {
    map: HashMap<String, String>,
}

const FLAGS: &[&str] = &["abort", "verify-lane", "no-noise", "by-arrival", "force", "ignored", "no-skip-idle"];

impl Args {
    pub fn parse(args: &[String]) -> Result<Args, String> {
        let mut map = HashMap::new();
        let mut i = 0;
        while i < args.len() {
            let key = args[i].strip_prefix("--").ok_or(format!("unexpected argument `{}`", args[i]))?.to_string();
            if FLAGS.contains(&key.as_str()) {
                map.insert(key, "1".to_string());
                i += 1;
            } else {
                let v = args.get(i + 1).ok_or(format!("--{key} needs a value"))?;
                // `report` takes several values after one flag: they are joined with a space.
                let e = map.entry(key).or_insert_with(String::new);
                if !e.is_empty() {
                    e.push(' ');
                }
                e.push_str(v);
                i += 2;
                while i < args.len() && !args[i].starts_with("--") {
                    e.push(' ');
                    e.push_str(&args[i]);
                    i += 1;
                }
            }
        }
        Ok(Args { map })
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.map.get(key).map(String::as_str)
    }

    pub fn need(&self, key: &str) -> Result<&str, String> {
        self.get(key).ok_or(format!("missing --{key}"))
    }

    pub fn flag(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    pub fn num<T: std::str::FromStr>(&self, key: &str, default: T) -> Result<T, String> {
        match self.get(key) {
            Some(v) => v.parse().map_err(|_| format!("--{key}: bad value `{v}`")),
            None => Ok(default),
        }
    }
}

/// `a,b,c` or `lo:hi:step`, in milliseconds, as microseconds.
pub fn parse_deadlines(text: &str) -> Result<Vec<Micros>, String> {
    let bad = || format!("bad deadline list `{text}`");
    if text.contains(':') {
        let p: Vec<f64> = text.split(':').map(|x| x.parse::<f64>().map_err(|_| bad())).collect::<Result<_, _>>()?;
        if p.len() != 3 || p[2] <= 0.0 {
            return Err(bad());
        }
        let mut out = Vec::new();
        let mut x = p[0];
        while x <= p[1] + 1e-9 {
            out.push((x * 1000.0).round() as Micros);
            x += p[2];
        }
        return Ok(out);
    }
    text.split(',').map(|x| x.trim().parse::<f64>().map(|v| (v * 1000.0).round() as Micros).map_err(|_| bad())).collect()
}

/// `a..b`, or a single count `n` as `0..n`.
pub fn parse_range(text: &str, max: u32) -> Result<std::ops::Range<u32>, String> {
    let bad = || format!("bad slot range `{text}`");
    let (a, b) = match text.split_once("..") {
        Some((a, b)) => (a.parse::<u32>().map_err(|_| bad())?, b.parse::<u32>().map_err(|_| bad())?),
        None => (0, text.parse::<u32>().map_err(|_| bad())?),
    };
    Ok(a..b.min(max))
}

/// A model and its backend label (the `backend` line of the file).
pub fn load(path: &Path) -> Result<(Box<dyn CostModel>, String), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let backend = Kv::parse(&text).0.get("backend").cloned().unwrap_or_else(|| "model".into());
    Ok((load_model(path)?, backend))
}

/// `load`, then apply `--cross-rate-ms X` (the extra cost of a kept proof of another rate), to test the worst case.
pub fn load_args(a: &Args, path: &Path) -> Result<(Box<dyn CostModel>, String), String> {
    let (mut model, name) = load(path)?;
    if let Some(ms) = a.get("cross-rate-ms") {
        model.set_cross_rate_us(ms.parse::<f64>().map_err(|_| format!("--cross-rate-ms: bad value `{ms}`"))? * 1000.0);
    }
    Ok((model, name))
}

/// `--strategy-cost calibrated|table`, default calibrated.
pub fn cost_kind(a: &Args) -> Result<crate::calib::CostKind, String> {
    a.get("strategy-cost").map_or(Ok(crate::calib::CostKind::Calibrated), crate::calib::CostKind::parse)
}

pub fn load_forecast(args: &Args) -> Result<Option<Rc<Forecast>>, String> {
    args.get("train").map(|d| Forecast::from_set(Path::new(d)).map(Rc::new)).transpose()
}
