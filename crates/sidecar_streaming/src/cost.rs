//! Cost models: how long a job takes on one lane. Nothing else in the harness holds a cost number;
//! every duration comes from a model file, and a model file comes from one cost CSV (`fit.rs`).

use crate::types::*;
use std::collections::BTreeMap;

pub trait CostModel {
    /// Predicted lane time of a job, in microseconds, without noise. Includes the native
    /// verification of children that are not verified.
    fn prove_us(&self, shape: &JobShape, threads: u32) -> Micros;
    fn verify_us(&self, child: &ChildShape) -> Micros;
    /// Sigma of log(duration): the simulator's multiplicative noise.
    fn sigma(&self) -> f64;
    fn proof_bytes(&self, shape: &JobShape) -> u32;
    /// Override the extra time of an EXTEND whose kept proof has another rate (to test the worst case).
    fn set_cross_rate_us(&mut self, _us: f64) {}
    /// Time to stop a running job, if the backend could.
    fn abort_us(&self) -> Micros {
        5000
    }
}

/// The text of a model file: `key value` lines, `#` comments.
pub struct Kv(pub BTreeMap<String, String>, pub Vec<(String, String)>);

impl Kv {
    pub fn parse(text: &str) -> Kv {
        let list = arrival_model::schedule::parse_key_values(text);
        Kv(list.iter().cloned().collect(), list)
    }

    pub fn f(&self, key: &str, default: f64) -> f64 {
        self.0.get(key).and_then(|v| v.split_whitespace().next()?.parse().ok()).unwrap_or(default)
    }

    pub fn rates(&self, prefix: &str) -> [f64; 5] {
        let mut a = [0.0; 5];
        for (r, x) in a.iter_mut().enumerate().skip(1) {
            *x = self.f(&format!("{prefix}.{r}"), 0.0);
        }
        a
    }
}

fn rate_ix(r: Rate) -> usize {
    (r as usize).clamp(1, 4)
}

// ---------------------------------------------------------------------------------------------
// Rows form
// ---------------------------------------------------------------------------------------------

/// One parameter set of the rows form. Work is counted in rows, in units of one signature.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParamSet {
    pub step: f64,
    pub w0: f64,
    pub t0: [f64; 5],
    pub lin: [f64; 5],
    pub pad: [f64; 5],
    pub wc: [f64; 5],
    pub wk: f64,
    pub wl: f64,
}

impl ParamSet {
    pub fn padded(&self, w: f64) -> f64 {
        if self.step <= 0.0 || w <= 0.0 {
            return 0.0;
        }
        self.step * 2f64.powf((w / self.step).log2().ceil())
    }

    /// Rows to verify one child of this rate in the guest. A rate with no fitted value uses the first one that has.
    fn wc_of(&self, rate: Rate) -> f64 {
        let w = self.wc[rate_ix(rate)];
        if w != 0.0 { w } else { self.wc.iter().copied().find(|x| *x != 0.0).unwrap_or(0.0) }
    }

    pub fn work(&self, shape: &JobShape) -> f64 {
        self.w0
            + shape.new_sigs as f64
            + shape.children.iter().map(|c| self.wc_of(c.rate) + self.wk * c.claims as f64 + self.wl * c.leaves as f64).sum::<f64>()
    }

    pub fn time(&self, shape: &JobShape) -> f64 {
        let w = self.work(shape);
        let r = rate_ix(shape.rate);
        self.t0[r] + self.lin[r] * w + self.pad[r] * self.padded(w)
    }

    fn write(&self, prefix: &str, out: &mut String) {
        use std::fmt::Write;
        let _ = writeln!(out, "{prefix}step {}", self.step);
        let _ = writeln!(out, "{prefix}w0 {}", self.w0);
        for (name, a) in [("t0", &self.t0), ("lin", &self.lin), ("pad", &self.pad), ("wc", &self.wc)] {
            for r in 1..=4 {
                if a[r] != 0.0 {
                    let _ = writeln!(out, "{prefix}{name}.{r} {}", a[r]);
                }
            }
        }
        let _ = writeln!(out, "{prefix}wk {}", self.wk);
        let _ = writeln!(out, "{prefix}wl {}", self.wl);
    }

    fn read(kv: &Kv, prefix: &str) -> ParamSet {
        let rates = |n: &str| kv.rates(&format!("{prefix}{n}"));
        ParamSet {
            step: kv.f(&format!("{prefix}step"), 0.0),
            w0: kv.f(&format!("{prefix}w0"), 0.0),
            t0: rates("t0"),
            lin: rates("lin"),
            pad: rates("pad"),
            wc: rates("wc"),
            wk: kv.f(&format!("{prefix}wk"), 0.0),
            wl: kv.f(&format!("{prefix}wl"), 0.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Common {
    pub backend: String,
    pub threads: u32,
    pub isa: String,
    pub sigma: f64,
    pub v: [f64; 5],
    pub bytes: [f64; 5],
    pub abort_us: f64,
    /// Extra time of an EXTEND whose kept proof has another rate than the job (SPEC 5.5, N1).
    pub cross_rate_us: f64,
}

impl Common {
    fn read(kv: &Kv) -> Common {
        Common {
            backend: kv.0.get("backend").cloned().unwrap_or_default(),
            threads: kv.f("threads", 8.0) as u32,
            isa: kv.0.get("isa").cloned().unwrap_or_default(),
            sigma: kv.f("sigma", 0.0),
            v: kv.rates("v"),
            bytes: kv.rates("bytes"),
            abort_us: kv.f("abort_us", 5000.0),
            cross_rate_us: kv.f("cross_rate_us", 0.0),
        }
    }

    pub fn write(&self, form: &str, out: &mut String) {
        use std::fmt::Write;
        let _ = writeln!(out, "form {form}");
        let _ = writeln!(out, "backend {}", self.backend);
        let _ = writeln!(out, "threads {}", self.threads);
        let _ = writeln!(out, "isa {}", self.isa);
        let _ = writeln!(out, "sigma {}", self.sigma);
        let _ = writeln!(out, "cross_rate_us {}", self.cross_rate_us);
        for r in 1..=4 {
            if self.v[r] != 0.0 {
                let _ = writeln!(out, "v.{r} {}", self.v[r]);
            }
            if self.bytes[r] != 0.0 {
                let _ = writeln!(out, "bytes.{r} {}", self.bytes[r]);
            }
        }
    }

    fn verify(&self, rate: Rate) -> f64 {
        let r = rate_ix(rate);
        if self.v[r] != 0.0 { self.v[r] } else { self.v.iter().copied().find(|x| *x != 0.0).unwrap_or(0.0) }
    }

    /// Cross-rate cost of an EXTEND: one `cross_rate_us` for each child whose rate is not the job's.
    pub fn cross(&self, shape: &JobShape) -> f64 {
        self.cross_rate_us * shape.children.iter().filter(|c| c.rate != shape.rate).count() as f64
    }

    fn unverified(&self, shape: &JobShape) -> f64 {
        shape.children.iter().filter(|c| c.verify).map(|c| self.verify(c.rate)).sum()
    }

    fn bytes(&self, rate: Rate) -> u32 {
        let r = rate_ix(rate);
        if self.bytes[r] != 0.0 { self.bytes[r] as u32 } else { 250_000 }
    }
}

#[derive(Clone, Debug)]
pub struct RowsModel {
    pub common: Common,
    /// Jobs with new signatures: AGGREGATE and EXTEND.
    pub ext: ParamSet,
    /// Jobs without: RECURSE.
    pub merge: ParamSet,
}

impl RowsModel {
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        self.common.write("rows", &mut s);
        self.ext.write("", &mut s);
        self.merge.write("merge.", &mut s);
        s
    }
}

impl CostModel for RowsModel {
    fn prove_us(&self, shape: &JobShape, _threads: u32) -> Micros {
        let set = if shape.new_sigs == 0 && !shape.children.is_empty() { &self.merge } else { &self.ext };
        let cross = if shape.new_sigs > 0 { self.common.cross(shape) } else { 0.0 };
        (set.time(shape) + self.common.unverified(shape) + cross).max(1.0).round() as Micros
    }
    fn verify_us(&self, child: &ChildShape) -> Micros {
        self.common.verify(child.rate).round() as Micros
    }
    fn sigma(&self) -> f64 {
        self.common.sigma
    }
    fn proof_bytes(&self, shape: &JobShape) -> u32 {
        self.common.bytes(shape.rate)
    }
    fn set_cross_rate_us(&mut self, us: f64) {
        self.common.cross_rate_us = us;
    }
    fn abort_us(&self) -> Micros {
        self.common.abort_us as Micros
    }
}

// ---------------------------------------------------------------------------------------------
// Table form
// ---------------------------------------------------------------------------------------------

/// A child with more claims than this is a root of an earlier merge, not a leaf (a leaf has at most 2048 signatures).
pub const ROOT_CLAIMS: u32 = 2048;

/// Measured `(x, microseconds)` points. The cost at `x` is that of the smallest measured point at or above `x`:
/// a step function, so an unmeasured stretch is never read as smooth, and the answer is never below what was measured
/// just above. Past the last point the cost is flat, or grows by `slope` per unit if the curve says so.
/// `jumps` are the intervals `(a, b, rise)` where the data shows a jump; they document the steps.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Curve {
    pub pts: Vec<(f64, f64)>,
    pub jumps: Vec<(f64, f64, f64)>,
    pub slope: f64,
}

impl Curve {
    pub fn at(&self, x: f64) -> f64 {
        let Some(last) = self.pts.last() else { return 0.0 };
        // The points are sorted by x: the first one at or above `x`.
        match self.pts.get(self.pts.partition_point(|p| p.0 < x)) {
            Some(p) => p.1,
            None => last.1 + self.slope * (x - last.0),
        }
    }

    pub fn max_x(&self) -> f64 {
        self.pts.last().map_or(0.0, |p| p.0)
    }
}

/// Measured curves. EXTEND depends on its new signatures and on the kept size: one curve per measured kept size.
/// RECURSE has a curve over the child count for children of 100 and of 1500 claims (1700 costs the same as 1500),
/// and a table of what a kept root adds. A kept proof of another rate than the job's adds `common.cross_rate_us`.
#[derive(Clone, Debug)]
pub struct TableModel {
    pub common: Common,
    /// Signatures of the largest measured AGGREGATE, per rate: nothing is priced beyond it by extrapolation.
    pub agg: [Curve; 5],
    /// Per rate, a curve over new signatures for each measured kept size, ascending.
    pub ext: [Vec<(f64, Curve)>; 5],
    pub rec_small: [Curve; 5],
    pub rec: [Curve; 5],
    /// What a kept root child adds to a RECURSE, by the number of fresh children, from rate 1 rows.
    pub kept_excess: Curve,
    pub small_claims: f64,
    pub big_claims: f64,
}

fn write_curve(s: &mut String, name: &str, r: usize, c: &Curve) {
    use std::fmt::Write;
    let _ = writeln!(s, "{name}.{r}.slope {}", c.slope);
    for (x, y) in &c.pts {
        let _ = writeln!(s, "{name}.{r}.pt {x} {y}");
    }
    for (a, b, rise) in &c.jumps {
        let _ = writeln!(s, "{name}.{r}.jump {a} {b} {rise}");
    }
}

impl TableModel {
    pub fn to_text(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        self.common.write("table", &mut s);
        for r in 1..=4 {
            write_curve(&mut s, "agg", r, &self.agg[r]);
            for (kept, c) in &self.ext[r] {
                write_curve(&mut s, &format!("ext{kept}"), r, c);
            }
            write_curve(&mut s, "recs", r, &self.rec_small[r]);
            write_curve(&mut s, "rec", r, &self.rec[r]);
        }
        write_curve(&mut s, "kept", 1, &self.kept_excess);
        let _ = writeln!(s, "small_claims {}", self.small_claims);
        let _ = writeln!(s, "big_claims {}", self.big_claims);
        s
    }

    fn read(kv: &Kv) -> TableModel {
        let mut c: BTreeMap<String, [Curve; 5]> = BTreeMap::new();
        for (k, v) in &kv.1 {
            let parts: Vec<&str> = k.split('.').collect();
            let known = ["agg", "recs", "rec", "kept"].contains(&parts[0]) || parts[0].strip_prefix("ext").is_some_and(|k| k.parse::<f64>().is_ok());
            if parts.len() != 3 || !known {
                continue;
            }
            let Ok(r) = parts[1].parse::<usize>() else { continue };
            let curve = &mut c.entry(parts[0].to_string()).or_default()[r.min(4)];
            let nums: Vec<f64> = v.split_whitespace().filter_map(|x| x.parse().ok()).collect();
            match (parts[2], nums.as_slice()) {
                ("slope", [s]) => curve.slope = *s,
                ("pt", [x, y]) => curve.pts.push((*x, *y)),
                ("jump", [a, b, rise]) => curve.jumps.push((*a, *b, *rise)),
                _ => {}
            }
        }
        let mut ext: [Vec<(f64, Curve)>; 5] = Default::default();
        let ext_keys: Vec<String> = c.keys().filter(|k| k.starts_with("ext")).cloned().collect();
        for key in ext_keys {
            let kept: f64 = key[3..].parse().unwrap();
            for (r, curve) in c.remove(&key).unwrap().into_iter().enumerate() {
                if !curve.pts.is_empty() {
                    ext[r].push((kept, curve));
                }
            }
        }
        ext.iter_mut().for_each(|v| v.sort_by(|a, b| a.0.total_cmp(&b.0)));
        let mut take = |n: &str| c.remove(n).unwrap_or_default();
        let kept_excess = take("kept").into_iter().nth(1).unwrap_or_default();
        TableModel {
            common: Common::read(kv),
            agg: take("agg"),
            ext,
            rec_small: take("recs"),
            rec: take("rec"),
            kept_excess,
            small_claims: kv.f("small_claims", 100.0),
            big_claims: kv.f("big_claims", 1500.0),
        }
    }

    /// EXTEND time: the curve of the smallest measured kept size at or above `kept` (the largest, if none is).
    fn ext_at(&self, r: usize, kept: f64, new: f64) -> f64 {
        let v = &self.ext[r];
        let Some(last) = v.last() else { return self.agg[r].at(new) };
        let (_, curve) = v.iter().find(|(k, _)| *k >= kept).unwrap_or(last);
        curve.at(new)
    }

    fn rec_at(&self, r: usize, n: f64, avg_claims: f64) -> f64 {
        let (big, small) = (self.rec[r].at(n), self.rec_small[r].at(n));
        if small <= 0.0 {
            return big;
        }
        let w = ((avg_claims - self.small_claims) / (self.big_claims - self.small_claims)).clamp(0.0, 1.0);
        small + (big - small) * w
    }
}

impl CostModel for TableModel {
    fn prove_us(&self, shape: &JobShape, _threads: u32) -> Micros {
        let r = rate_ix(shape.rate);
        let n = shape.children.len();
        let t = if n == 0 {
            self.agg[r].at(shape.new_sigs as f64)
        } else if shape.new_sigs == 0 {
            let roots = shape.children.iter().filter(|c| c.claims > ROOT_CLAIMS).count();
            let fresh: Vec<&ChildShape> = shape.children.iter().filter(|c| c.claims <= ROOT_CLAIMS).collect();
            let avg = if fresh.is_empty() { self.big_claims } else { fresh.iter().map(|c| c.claims as f64).sum::<f64>() / fresh.len() as f64 };
            // A root costs what a leaf does in the table, plus the measured excess of a kept root.
            self.rec_at(r, n as f64, avg) + roots as f64 * self.kept_excess.at(fresh.len() as f64)
        } else {
            // The first child is the kept proof, inside the EXTEND curve; each further child costs what the RECURSE table adds.
            let extra = if n > 1 { self.rec[r].at(n as f64) - self.rec[r].at(1.0) } else { 0.0 };
            self.ext_at(r, shape.children[0].claims as f64, shape.new_sigs as f64) + extra + self.common.cross(shape)
        };
        (t + self.common.unverified(shape)).max(1.0).round() as Micros
    }
    fn verify_us(&self, child: &ChildShape) -> Micros {
        self.common.verify(child.rate).round() as Micros
    }
    fn sigma(&self) -> f64 {
        self.common.sigma
    }
    fn proof_bytes(&self, shape: &JobShape) -> u32 {
        self.common.bytes(shape.rate)
    }
    fn abort_us(&self) -> Micros {
        self.common.abort_us as Micros
    }
    fn set_cross_rate_us(&mut self, us: f64) {
        self.common.cross_rate_us = us;
    }
}

pub fn parse_model(text: &str) -> Result<Box<dyn CostModel>, String> {
    let kv = Kv::parse(text);
    match kv.0.get("form").map(String::as_str) {
        Some("rows") => Ok(Box::new(RowsModel { common: Common::read(&kv), ext: ParamSet::read(&kv, ""), merge: ParamSet::read(&kv, "merge.") })),
        Some("table") => Ok(Box::new(TableModel::read(&kv))),
        _ => Err("model file: `form` must be `rows` or `table`".into()),
    }
}

pub fn load_model(path: &std::path::Path) -> Result<Box<dyn CostModel>, String> {
    parse_model(&std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?)
}
