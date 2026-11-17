//! Coverage at the deadline: per slot, and over slots (SPEC section 4).

use crate::types::*;

/// One input of a slot: a signature (one claim) or a candidate aggregate.
#[derive(Clone, Debug)]
pub struct InputRec {
    pub t: Micros,
    pub claims: Vec<u32>,
    pub valid: bool,
}

#[derive(Clone, Debug)]
pub struct JobRec {
    pub job: u32,
    pub lane: usize,
    pub op: &'static str,
    pub children: u32,
    pub child_claims: u32,
    pub new_sigs: u32,
    pub rate: Rate,
    pub start: Micros,
    pub end: Micros,
    pub predicted: Micros,
    /// Lane time spent verifying: all of a `VERIFY` job, the child verification inside a prove job.
    pub verify_us: Micros,
    /// An ancestor of the published proof, or a verification of one.
    pub useful: bool,
    pub result: String,
    /// Claims in the proof a prove job made; 0 for any other outcome.
    pub result_claims: u32,
}

/// What the driver writes down about one slot.
#[derive(Clone, Debug)]
pub struct SlotRecord {
    pub shape: Shape,
    pub deadline: Micros,
    pub inputs: Vec<InputRec>,
    pub published: Option<(Micros, Coverage)>,
    pub jobs: Vec<JobRec>,
    /// `end_to_end` charges `sent` to the votes of all 16 subnets.
    pub sent_override: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Metrics {
    pub got: u32,
    pub arrived: u32,
    pub sent: u32,
    pub coverage: f64,
    pub coverage_sent: f64,
    pub published_us: Option<Micros>,
    pub cut_us: Option<Micros>,
    pub staleness_us: Option<Micros>,
    pub slack_us: Option<Micros>,
    pub jobs: u32,
    pub busy_us: Micros,
    pub wasted_us: Micros,
    pub verify_us: Micros,
}

/// N/A (NaN) when there is nothing to divide by: a slot with nothing arrived has no coverage to measure.
fn ratio(a: u32, b: u32) -> f64 {
    if b == 0 { f64::NAN } else { a as f64 / b as f64 }
}

pub fn slot_metrics(rec: &SlotRecord) -> Metrics {
    let on_time = rec.published.as_ref().filter(|(t, _)| *t <= rec.deadline);
    let got = on_time.map_or(0, |(_, c)| c.len());
    let mut by_d = Coverage::empty();
    let mut all = Coverage::empty();
    for i in rec.inputs.iter().filter(|i| i.valid) {
        for &c in &i.claims {
            all.insert(c);
            if i.t <= rec.deadline {
                by_d.insert(c);
            }
        }
    }
    let (arrived, sent) = (by_d.len(), rec.sent_override.unwrap_or_else(|| all.len()));
    // The earliest moment at which the valid inputs that had arrived hold every claim of the proof. A later input
    // that is a subset of the proof does not move it.
    let cut = on_time.and_then(|(_, p)| {
        let mut order: Vec<&InputRec> = rec.inputs.iter().filter(|i| i.valid).collect();
        order.sort_by_key(|i| i.t);
        let (mut seen, mut left) = (Coverage::empty(), p.len());
        if left == 0 {
            return None;
        }
        for i in order {
            for &c in &i.claims {
                if p.contains(c) && !seen.contains(c) {
                    seen.insert(c);
                    left -= 1;
                }
            }
            if left == 0 {
                return Some(i.t);
            }
        }
        None
    });
    Metrics {
        got,
        arrived,
        sent,
        coverage: ratio(got, arrived),
        coverage_sent: ratio(got, sent),
        published_us: rec.published.as_ref().map(|(t, _)| *t),
        cut_us: cut,
        staleness_us: on_time.zip(cut).map(|((t, _), c)| t.saturating_sub(c)),
        slack_us: on_time.map(|(t, _)| rec.deadline - t),
        jobs: rec.jobs.len() as u32,
        busy_us: rec.jobs.iter().map(|j| j.end - j.start).sum(),
        wasted_us: rec.jobs.iter().filter(|j| !j.useful).map(|j| j.end - j.start).sum(),
        verify_us: rec.jobs.iter().map(|j| j.verify_us).sum(),
    }
}

/// A line of a slot record file.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub set: String,
    pub backend: String,
    pub strategy: String,
    pub deadline_us: Micros,
    pub slot: u32,
    pub m: Metrics,
}

pub const ROW_HEADER: &str = "set,backend,strategy,deadline_us,slot,got,arrived,sent,coverage,coverage_sent,published_us,cut_us,staleness_us,slack_us,jobs,busy_us,wasted_us,verify_us";

fn opt(v: Option<u64>) -> String {
    v.map_or(String::new(), |x| x.to_string())
}

impl Row {
    pub fn to_csv(&self) -> String {
        let m = &self.m;
        format!(
            "{},{},{},{},{},{},{},{},{:.6},{:.6},{},{},{},{},{},{},{},{}",
            self.set,
            self.backend,
            self.strategy,
            self.deadline_us,
            self.slot,
            m.got,
            m.arrived,
            m.sent,
            m.coverage,
            m.coverage_sent,
            opt(m.published_us),
            opt(m.cut_us),
            opt(m.staleness_us),
            opt(m.slack_us),
            m.jobs,
            m.busy_us,
            m.wasted_us,
            m.verify_us
        )
    }

    pub fn from_csv(line: &str) -> Result<Row, String> {
        let f: Vec<&str> = line.split(',').collect();
        let bad = || format!("bad slot record line: {line}");
        if f.len() != 18 {
            return Err(bad());
        }
        let u = |i: usize| f[i].parse::<u64>().map_err(|_| bad());
        let o = |i: usize| if f[i].is_empty() { Ok(None) } else { u(i).map(Some) };
        let x = |i: usize| f[i].parse::<f64>().map_err(|_| bad());
        Ok(Row {
            set: f[0].into(),
            backend: f[1].into(),
            strategy: f[2].into(),
            deadline_us: u(3)?,
            slot: u(4)? as u32,
            m: Metrics {
                got: u(5)? as u32,
                arrived: u(6)? as u32,
                sent: u(7)? as u32,
                coverage: x(8)?,
                coverage_sent: x(9)?,
                published_us: o(10)?,
                cut_us: o(11)?,
                staleness_us: o(12)?,
                slack_us: o(13)?,
                jobs: u(14)? as u32,
                busy_us: u(15)?,
                wasted_us: u(16)?,
                verify_us: u(17)?,
            },
        })
    }
}

/// Nearest-rank quantile of the valid (not NaN) values: `x[ceil(k N / 100) - 1]`. NaN if there are none.
pub fn quantile(xs: &[f64], k: u32) -> f64 {
    let mut v: Vec<f64> = xs.iter().copied().filter(|x| !x.is_nan()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let rank = (k as usize * v.len()).div_ceil(100).max(1);
    v[rank - 1]
}

/// Mean of the valid (not NaN) values. NaN if there are none.
pub fn mean(xs: &[f64]) -> f64 {
    let v: Vec<f64> = xs.iter().copied().filter(|x| !x.is_nan()).collect();
    v.iter().sum::<f64>() / v.len() as f64
}

#[derive(Clone, Debug)]
pub struct Summary {
    /// Slots with a coverage.
    pub n: usize,
    /// Slots with nothing arrived by `D`: no coverage, left out of every aggregate of coverage.
    pub empty: usize,
    pub p50: f64,
    pub p10: f64,
    pub mean: f64,
    pub sent_p50: f64,
    pub sent_p10: f64,
    pub staleness_ms: f64,
    pub slack_ms: f64,
    pub jobs: f64,
    pub busy_ms: f64,
    pub wasted_ms: f64,
    pub verify_ms: f64,
}

pub fn summarize(rows: &[&Row]) -> Summary {
    let c: Vec<f64> = rows.iter().map(|r| r.m.coverage).collect();
    let s: Vec<f64> = rows.iter().map(|r| r.m.coverage_sent).collect();
    let avg = |f: &dyn Fn(&Metrics) -> Option<u64>| {
        let v: Vec<f64> = rows.iter().filter_map(|r| f(&r.m)).map(|x| x as f64 / 1000.0).collect();
        if v.is_empty() { f64::NAN } else { mean(&v) }
    };
    Summary {
        n: rows.iter().filter(|r| !r.m.coverage.is_nan()).count(),
        empty: rows.iter().filter(|r| r.m.coverage.is_nan()).count(),
        p50: quantile(&c, 50),
        p10: quantile(&c, 10),
        mean: mean(&c),
        sent_p50: quantile(&s, 50),
        sent_p10: quantile(&s, 10),
        staleness_ms: avg(&|m| m.staleness_us),
        slack_ms: avg(&|m| m.slack_us),
        jobs: mean(&rows.iter().map(|r| r.m.jobs as f64).collect::<Vec<_>>()),
        busy_ms: avg(&|m| Some(m.busy_us)),
        wasted_ms: avg(&|m| Some(m.wasted_us)),
        verify_ms: avg(&|m| Some(m.verify_us)),
    }
}

/// Mean of `a - b` in coverage over paired slots, and its standard error.
pub fn paired_diff(a: &[f64], b: &[f64]) -> (f64, f64) {
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).filter(|x| !x.is_nan()).collect();
    let m = mean(&d);
    if d.len() < 2 {
        return (m, 0.0);
    }
    let var = d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (d.len() - 1) as f64;
    (m, (var / d.len() as f64).sqrt())
}

/// `D_99`: the smallest swept deadline at which p10 coverage is at least 0.99. A deadline whose slots are all empty has
/// no p10 (NaN) and never counts.
pub fn d99(points: &[(Micros, f64)]) -> Option<Micros> {
    points.iter().filter(|(_, p10)| *p10 >= 0.99).map(|(d, _)| *d).min()
}
