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
        }
    }

    pub fn write(&self, form: &str, out: &mut String) {
        use std::fmt::Write;
        let _ = writeln!(out, "form {form}");
        let _ = writeln!(out, "backend {}", self.backend);
        let _ = writeln!(out, "threads {}", self.threads);
        let _ = writeln!(out, "isa {}", self.isa);
        let _ = writeln!(out, "sigma {}", self.sigma);
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
        (set.time(shape) + self.common.unverified(shape)).max(1.0).round() as Micros
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
}

// ---------------------------------------------------------------------------------------------
// Table form
// ---------------------------------------------------------------------------------------------

/// Measured `(x, microseconds)` points, linear between them. A marked segment holds a jump at a position:
/// the left plateau runs on to that position at the table's slope, then the cost steps up to the right line.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Curve {
    pub pts: Vec<(f64, f64)>,
    /// `(left point index, jump position)`.
    pub jumps: Vec<(usize, f64)>,
    pub slope: f64,
}

impl Curve {
    pub fn at(&self, x: f64) -> f64 {
        let p = &self.pts;
        if p.is_empty() {
            return 0.0;
        }
        if p.len() == 1 || x <= p[0].0 {
            return (p[0].1 + self.slope * (x - p[0].0)).max(p[0].1 * 0.5);
        }
        let last = p[p.len() - 1];
        if x >= last.0 {
            return last.1 + self.slope * (x - last.0);
        }
        let i = p.iter().rposition(|q| q.0 <= x).unwrap();
        let ((x0, y0), (x1, y1)) = (p[i], p[i + 1]);
        if let Some((_, pos)) = self.jumps.iter().find(|(j, _)| *j == i) {
            return if x < *pos { y0 + self.slope * (x - x0) } else { y1 - self.slope * (x1 - x) };
        }
        y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    }
}


/// Measured curves, linear between points. EXTEND depends on its new signatures and, within noise, not on the kept proof.
/// RECURSE has two curves over the child count, for children of 100 and of 1500 claims; other sizes interpolate.
#[derive(Clone, Debug)]
pub struct TableModel {
    pub common: Common,
    pub agg: [Curve; 5],
    /// Per rate, a curve over new signatures for each measured kept size, ascending.
    pub ext: [Vec<(f64, Curve)>; 5],
    pub rec_small: [Curve; 5],
    pub rec: [Curve; 5],
    pub small_claims: f64,
    pub big_claims: f64,
}

impl TableModel {
    pub fn to_text(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        self.common.write("table", &mut s);
        let mut curve = |name: &str, r: usize, c: &Curve| {
            let _ = writeln!(s, "{name}.{r}.slope {}", c.slope);
            for (x, y) in &c.pts {
                let _ = writeln!(s, "{name}.{r}.pt {x} {y}");
            }
            for (i, pos) in &c.jumps {
                let _ = writeln!(s, "{name}.{r}.jump {i} {pos}");
            }
        };
        for r in 1..=4 {
            curve("agg", r, &self.agg[r]);
            for (kept, c) in &self.ext[r] {
                curve(&format!("ext{kept}"), r, c);
            }
            curve("recs", r, &self.rec_small[r]);
            curve("rec", r, &self.rec[r]);
        }
        let _ = writeln!(s, "small_claims {}", self.small_claims);
        let _ = writeln!(s, "big_claims {}", self.big_claims);
        s
    }

    fn read(kv: &Kv) -> TableModel {
        let mut c: BTreeMap<String, [Curve; 5]> = BTreeMap::new();
        for (k, v) in &kv.1 {
            let parts: Vec<&str> = k.split('.').collect();
            if parts.len() != 3 || !(["agg", "recs", "rec"].contains(&parts[0]) || parts[0].strip_prefix("ext").is_some_and(|k| k.parse::<f64>().is_ok())) {
                continue;
            }
            let Ok(r) = parts[1].parse::<usize>() else { continue };
            let curve = &mut c.entry(parts[0].to_string()).or_default()[r.min(4)];
            let nums: Vec<f64> = v.split_whitespace().filter_map(|x| x.parse().ok()).collect();
            match (parts[2], nums.as_slice()) {
                ("slope", [s]) => curve.slope = *s,
                ("pt", [x, y]) => curve.pts.push((*x, *y)),
                ("jump", [i, p]) => curve.jumps.push((*i as usize, *p)),
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
        TableModel {
            common: Common::read(kv),
            agg: take("agg"),
            ext,
            rec_small: take("recs"),
            rec: take("rec"),
            small_claims: kv.f("small_claims", 100.0),
            big_claims: kv.f("big_claims", 1500.0),
        }
    }
}

impl TableModel {
    /// Bilinear in (kept claims, new signatures); the kept size is clamped to the measured range.
    fn ext_at(&self, r: usize, kept: f64, new: f64) -> f64 {
        let v = &self.ext[r];
        let Some(first) = v.first() else { return self.agg[r].at(new) };
        let kept = kept.clamp(first.0, v[v.len() - 1].0);
        let hi = v.iter().position(|(k, _)| *k >= kept).unwrap();
        if hi == 0 || v[hi].0 == kept {
            return v[hi].1.at(new);
        }
        let ((k0, c0), (k1, c1)) = (&v[hi - 1], &v[hi]);
        let w = (kept - k0) / (k1 - k0);
        c0.at(new) * (1.0 - w) + c1.at(new) * w
    }
}

impl CostModel for TableModel {
    fn prove_us(&self, shape: &JobShape, _threads: u32) -> Micros {
        let r = rate_ix(shape.rate);
        let n = shape.children.len() as f64;
        let t = if shape.children.is_empty() {
            self.agg[r].at(shape.new_sigs as f64)
        } else if shape.new_sigs == 0 {
            let claims: f64 = shape.children.iter().map(|c| c.claims as f64).sum();
            let (big, small) = (self.rec[r].at(n), self.rec_small[r].at(n));
            let w = ((claims / n - self.small_claims) / (self.big_claims - self.small_claims)).clamp(0.0, 1.5);
            small + (big - small) * w
        } else {
            // One kept child is inside the EXTEND curve; each further child costs the slope of the RECURSE curve.
            self.ext_at(r, shape.children[0].claims as f64, shape.new_sigs as f64) + (n - 1.0) * self.rec[r].slope
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
