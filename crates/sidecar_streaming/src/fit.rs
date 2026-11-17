//! `streamsim fit`: a model file from a cost CSV (SPEC 5.3).
//!
//! The CSV has the columns of `BRIEF-cost.md`: `op,rate,n_sigs,children,child_sigs,kept_sigs,new_sigs,verify_children,run,ms,proof_bytes`.
//! The rows form is tried first. If its median relative residual is above 3 % or its maximum above 8 %,
//! the table form is written instead. In the CSV, a kept proof and an EXTEND share one rate, and the
//! children of a RECURSE are rate-2 leaves.

use crate::cost::*;
use crate::types::*;
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct Point {
    pub op: String,
    pub rate: u8,
    pub n_sigs: u32,
    pub children: u32,
    pub child_sigs: u32,
    pub kept: u32,
    pub new_sigs: u32,
    pub verify_children: bool,
    /// Microseconds, after dropping a cold first run.
    pub us: f64,
    pub std_ln: f64,
    pub bytes: u32,
}

/// `(op, rate, n_sigs, children, child_sigs, kept_sigs, new_sigs, verify_children)`.
type PointKey = (String, u8, u32, u32, u32, u32, u32, bool);

pub fn read_csv(text: &str) -> Result<Vec<Point>, String> {
    let mut groups: BTreeMap<PointKey, (Vec<f64>, u32)> = BTreeMap::new();
    for line in text.lines().skip(1).filter(|l| !l.is_empty()) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 11 {
            return Err(format!("bad cost line: {line}"));
        }
        if f[9] == "FAIL" {
            continue;
        }
        let n = |i: usize| f[i].parse::<u32>().map_err(|_| format!("bad cost line: {line}"));
        let key = (f[0].to_string(), n(1)? as u8, n(2)?, n(3)?, n(4)?, n(5)?, n(6)?, n(7)? != 0);
        let e = groups.entry(key).or_default();
        e.0.push(f[9].parse::<f64>().map_err(|_| format!("bad cost line: {line}"))? * 1000.0);
        e.1 = n(10)?;
    }
    Ok(groups
        .into_iter()
        .map(|(k, (mut runs, bytes))| {
            // Drop run 1 if it is the slowest by more than 10 %.
            if runs.len() > 2 {
                let rest_max = runs[1..].iter().cloned().fold(0.0, f64::max);
                if runs[0] > 1.1 * rest_max {
                    runs.remove(0);
                }
            }
            let logs: Vec<f64> = runs.iter().map(|x| x.ln()).collect();
            let m = logs.iter().sum::<f64>() / logs.len() as f64;
            let std_ln = (logs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / logs.len() as f64).sqrt();
            Point { op: k.0, rate: k.1, n_sigs: k.2, children: k.3, child_sigs: k.4, kept: k.5, new_sigs: k.6, verify_children: k.7, us: median(&mut runs), std_ln, bytes }
        })
        .collect())
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    if v.is_empty() {
        return 0.0;
    }
    if v.len() % 2 == 1 { v[v.len() / 2] } else { (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0 }
}

/// Least squares of `y ~ sum_j x_j f_j` over the rows `(f, y)`, with every `x_j >= 0`: the best of all subsets of features.
pub fn nnls3(rows: &[([f64; 3], f64)]) -> ([f64; 3], f64) {
    let mut best = ([0.0; 3], f64::INFINITY);
    for mask in 1u32..8 {
        let idx: Vec<usize> = (0..3).filter(|j| mask >> j & 1 == 1).collect();
        let k = idx.len();
        let mut a = [[0.0f64; 4]; 3];
        for (i, &p) in idx.iter().enumerate() {
            for (j, &q) in idx.iter().enumerate() {
                a[i][j] = rows.iter().map(|(f, _)| f[p] * f[q]).sum();
            }
            a[i][3] = rows.iter().map(|(f, y)| f[p] * y).sum();
        }
        // Gaussian elimination with partial pivoting.
        let mut ok = true;
        for c in 0..k {
            let piv = (c..k).max_by(|&x, &y| a[x][c].abs().total_cmp(&a[y][c].abs())).unwrap();
            a.swap(c, piv);
            if a[c][c].abs() < 1e-18 {
                ok = false;
                break;
            }
            for r in 0..k {
                if r != c {
                    let m = a[r][c] / a[c][c];
                    for cc in c..4 {
                        a[r][cc] -= m * a[c][cc];
                    }
                }
            }
        }
        if !ok {
            continue;
        }
        let mut x = [0.0; 3];
        for (i, &p) in idx.iter().enumerate() {
            x[p] = a[i][3] / a[i][i];
        }
        if x.iter().any(|v| *v < 0.0) {
            continue;
        }
        let sse: f64 = rows.iter().map(|(f, y)| (f[0] * x[0] + f[1] * x[1] + f[2] * x[2] - y).powi(2)).sum();
        if sse < best.1 {
            best = (x, sse);
        }
    }
    best
}

fn rate_points<'a>(pts: &'a [Point], op: &str, rate: u8) -> Vec<&'a Point> {
    pts.iter().filter(|p| p.op == op && p.rate == rate && !p.verify_children).collect()
}

fn padded(step: f64, w: f64) -> f64 {
    ParamSet { step, ..Default::default() }.padded(w)
}

/// Relative-residual rows of `T = t0 + lin * W + pad * P(W)`: `(f, y)` scaled by `1 / T`.
fn rows_for(ws: &[(f64, f64)], step: f64) -> Vec<([f64; 3], f64)> {
    ws.iter().map(|&(w, t)| ([1.0 / t, w / t, padded(step, w) / t], 1.0)).collect()
}

fn fit_aggregate(pts: &[Point], rates: &[u8]) -> Result<ParamSet, String> {
    let fit_at = |step: f64| {
        let mut set = ParamSet { step, ..Default::default() };
        let mut total = 0.0;
        for &r in rates {
            let ws: Vec<(f64, f64)> = rate_points(pts, "AGGREGATE", r).iter().map(|p| (p.n_sigs as f64, p.us)).collect();
            let (x, sse) = nnls3(&rows_for(&ws, step));
            set.t0[r as usize] = x[0];
            set.lin[r as usize] = x[1];
            set.pad[r as usize] = x[2];
            total += sse;
        }
        (total, set)
    };
    // Sizes between two measured points do not tell where the jump is: take the middle of the plateau of steps that fit exactly as well as the best.
    let scores: Vec<(f64, f64)> = (100..=400).map(|s| (s as f64, fit_at(s as f64).0)).collect();
    let best = scores.iter().map(|x| x.1).fold(f64::INFINITY, f64::min);
    if !best.is_finite() {
        return Err("no AGGREGATE rows".into());
    }
    let mut near: Vec<f64> = scores.iter().filter(|x| x.1 <= best * (1.0 + 1e-6) + 1e-12).map(|x| x.0).collect();
    near.sort_by(f64::total_cmp);
    // A step and twice a step explain the same jumps; take the lowest run of steps.
    let end = near.windows(2).position(|w| w[1] - w[0] > 1.5).map_or(near.len(), |i| i + 1);
    Ok(fit_at(near[end / 2]).1)
}

fn fit_extend(pts: &[Point], rates: &[u8], set: &mut ParamSet) {
    let mut wk_best = (f64::INFINITY, 0.0);
    // `wk` is shared over rates, `wc` is per rate.
    for wk in (0..=40).map(|i| i as f64 * 0.005) {
        let mut total = 0.0;
        let mut wcs = Vec::new();
        for &r in rates {
            let ps = rate_points(pts, "EXTEND", r);
            let mut best = (f64::INFINITY, 0.0);
            for wc in (0..=200).map(|i| i as f64 * 5.0) {
                let sse: f64 = ps
                    .iter()
                    .map(|p| {
                        let w = p.new_sigs as f64 + wc + wk * p.kept as f64;
                        let pred = set.t0[r as usize] + set.lin[r as usize] * w + set.pad[r as usize] * set.padded(w);
                        ((pred - p.us) / p.us).powi(2)
                    })
                    .sum();
                if sse < best.0 {
                    best = (sse, wc);
                }
            }
            total += best.0;
            wcs.push(best.1);
        }
        if total < wk_best.0 {
            wk_best = (total, wk);
            for (&r, wc) in rates.iter().zip(&wcs) {
                set.wc[r as usize] = *wc;
            }
        }
    }
    set.wk = wk_best.1;
}

fn fit_recurse(pts: &[Point], rates: &[u8], child_rate: usize) -> ParamSet {
    let mut best: Option<(f64, ParamSet)> = None;
    for step in (50..=2000).step_by(10) {
        let step = step as f64;
        for wk in (0..=40).map(|i| i as f64 * 0.005) {
            for wc in (0..=150).map(|i| i as f64 * 10.0) {
                let mut set = ParamSet { step, wk, ..Default::default() };
                set.wc[child_rate] = wc;
                let mut total = 0.0;
                for &r in rates {
                    let ws: Vec<(f64, f64)> =
                        rate_points(pts, "RECURSE", r).iter().map(|p| (p.children as f64 * (wc + wk * p.child_sigs as f64), p.us)).collect();
                    let (x, sse) = nnls3(&rows_for(&ws, step));
                    set.t0[r as usize] = x[0];
                    set.lin[r as usize] = x[1];
                    set.pad[r as usize] = x[2];
                    total += sse;
                }
                if best.as_ref().is_none_or(|b| total < b.0) {
                    best = Some((total, set));
                }
            }
        }
    }
    best.map(|b| b.1).unwrap_or_default()
}

fn common(pts: &[Point], backend: &str, threads: u32, isa: &str) -> Common {
    let mut c = Common { backend: backend.into(), threads, isa: isa.into(), sigma: 0.0, v: [0.0; 5], bytes: [0.0; 5], abort_us: 5000.0, cross_rate_us: 0.0 };
    for r in 1..=4u8 {
        let mut v: Vec<f64> = pts.iter().filter(|p| p.op == "VERIFY" && p.rate == r && p.children == 0).map(|p| p.us).collect();
        if !v.is_empty() {
            c.v[r as usize] = median(&mut v);
        }
        let mut b: Vec<f64> = pts.iter().filter(|p| p.op == "AGGREGATE" && p.rate == r && p.n_sigs >= 1000).map(|p| p.bytes as f64).collect();
        if !b.is_empty() {
            c.bytes[r as usize] = median(&mut b);
        }
    }
    // N1: a kept proof made at rate 1 and extended at rate 2 (`EXTEND_X`) against the same cell with a rate-2 kept proof (`EXTEND_S`).
    let mut penalties: Vec<f64> = Vec::new();
    for x in pts.iter().filter(|p| p.op == "EXTEND_X") {
        if let Some(s) = pts.iter().find(|s| s.op == "EXTEND_S" && s.kept == x.kept && s.new_sigs == x.new_sigs) {
            penalties.push(x.us - s.us);
        }
    }
    if !penalties.is_empty() {
        c.cross_rate_us = (penalties.iter().sum::<f64>() / penalties.len() as f64).max(0.0).round();
    }
    let mut s: Vec<f64> = pts.iter().map(|p| p.std_ln).collect();
    c.sigma = (median(&mut s) * 1e4).round() / 1e4;
    c
}

fn shape_of(p: &Point) -> Option<JobShape> {
    let r = p.rate;
    Some(match p.op.as_str() {
        "AGGREGATE" => JobShape::aggregate(p.n_sigs, r),
        "EXTEND" => JobShape::extend(p.kept, r, p.new_sigs, r),
        "RECURSE" => JobShape {
            children: (0..p.children).map(|_| ChildShape { claims: p.child_sigs, rate: 2, verify: p.verify_children, leaves: 1 }).collect(),
            new_sigs: 0,
            rate: r,
        },
        _ => return None,
    })
}

pub struct Report {
    pub model_text: String,
    pub form: &'static str,
    pub median_res: f64,
    pub max_res: f64,
    pub lines: Vec<String>,
}

fn residuals(model: &dyn CostModel, pts: &[Point]) -> (Vec<String>, f64, f64) {
    let mut lines = Vec::new();
    let mut res = Vec::new();
    for p in pts {
        let Some(shape) = shape_of(p) else { continue };
        if p.verify_children {
            continue;
        }
        let pred = model.prove_us(&shape, 8) as f64;
        let rel = (pred - p.us) / p.us;
        res.push(rel.abs());
        lines.push(format!(
            "{:9} rate {} n {:4} children {:2}x{:4} kept {:4} new {:4}: measured {:9.1} ms, predicted {:9.1} ms, residual {:+6.1} %",
            p.op,
            p.rate,
            p.n_sigs,
            p.children,
            p.child_sigs,
            p.kept,
            p.new_sigs,
            p.us / 1000.0,
            pred / 1000.0,
            rel * 100.0
        ));
    }
    let max = res.iter().cloned().fold(0.0, f64::max);
    (lines, median(&mut res), max)
}

/// A rise of this many microseconds above what the series' median slope predicts is a jump.
const JUMP_US: f64 = 60_000.0;

/// Medians over repeated x, sorted by x.
fn median_by_x(points: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    let mut by: BTreeMap<u64, Vec<f64>> = BTreeMap::new();
    for (x, y) in points {
        by.entry(x as u64).or_default().push(y);
    }
    by.into_iter().map(|(x, mut v)| (x as f64, median(&mut v))).collect()
}

/// Intervals `(a, b, rise)` between adjacent measured points where the cost rises by more than `JUMP_US` over what the
/// median slope predicts, and the slope is over three times the median. `(a, b]` means the jump is somewhere in it:
/// where points are one apart, it is at `b`.
pub fn find_jumps(pts: &[(f64, f64)]) -> (Vec<(f64, f64, f64)>, f64) {
    let segs: Vec<(f64, f64, f64, f64)> = pts.windows(2).map(|w| ((w[1].1 - w[0].1) / (w[1].0 - w[0].0), w[0].0, w[1].0, w[1].1 - w[0].1)).collect();
    let mut slopes: Vec<f64> = segs.iter().map(|s| s.0).collect();
    let med = median(&mut slopes).max(10.0);
    let jumps = segs.iter().filter(|s| s.0 > 3.0 * med && s.3 - med * (s.2 - s.1) > JUMP_US).map(|s| (s.1, s.2, s.3 - med * (s.2 - s.1))).collect();
    (jumps, med)
}

/// A curve from points. `extend`: past the last point, grow by the median slope; else stay flat.
fn curve(points: Vec<(f64, f64)>, extend: bool) -> Curve {
    let pts = median_by_x(points);
    let (jumps, slope) = if pts.len() > 1 { find_jumps(&pts) } else { (Vec::new(), 0.0) };
    Curve { pts, jumps, slope: if extend { slope } else { 0.0 } }
}

/// The table form from every measured point. The cost at a size is that of the next measured point at or above it.
pub fn fit_table(pts: &[Point], common: Common, rates: &[u8]) -> TableModel {
    let mut m = TableModel {
        common,
        agg: Default::default(),
        ext: Default::default(),
        rec_small: Default::default(),
        rec: Default::default(),
        kept_excess: Curve::default(),
        small_claims: 100.0,
        big_claims: 1500.0,
    };
    for &r in rates {
        let r_ix = r as usize;
        let agg: Vec<(f64, f64)> = rate_points(pts, "AGGREGATE", r).iter().map(|p| (p.n_sigs as f64, p.us)).collect();
        m.agg[r_ix] = curve(agg, false);
        let mut kepts: Vec<u32> = rate_points(pts, "EXTEND", r).iter().map(|p| p.kept).collect();
        kepts.sort_unstable();
        kepts.dedup();
        for k in kepts {
            let ext: Vec<(f64, f64)> = rate_points(pts, "EXTEND", r).iter().filter(|p| p.kept == k).map(|p| (p.new_sigs as f64, p.us)).collect();
            m.ext[r_ix].push((k as f64, curve(ext, true)));
        }
        // Children of 1700 claims cost what 1500 do.
        let rec = |claims: &[u32]| -> Vec<(f64, f64)> {
            rate_points(pts, "RECURSE", r).iter().filter(|p| claims.contains(&p.child_sigs)).map(|p| (p.children as f64, p.us)).collect()
        };
        m.rec[r_ix] = curve(rec(&[1500, 1700]), false);
        m.rec_small[r_ix] = curve(rec(&[100]), false);
    }
    // A kept root adds, over the same number of fresh children, what its rows show (rate 1).
    let kept: Vec<(f64, f64)> = rate_points(pts, "RECURSE_KEPT", 1)
        .iter()
        .map(|p| (p.children as f64, (p.us - m.rec[1].at(p.children as f64 + 1.0)).max(0.0)))
        .collect();
    m.kept_excess = curve(kept, false);
    m
}

pub struct Validation {
    pub lines: Vec<String>,
    pub median: f64,
    pub max: f64,
    /// The worst under-estimate (most negative relative error), 0 if none.
    pub under: f64,
    pub n: usize,
}

/// Hold out every `every`-th point of each series of `jumps_csv` (not its ends), refit from what is left of `union_csv`,
/// and predict them. A point just under a jump, with its neighbour held out too, is read from the other side of the jump:
/// the errors say how far the step function is from the truth where the data is sparse.
pub fn validate_holdout(union_csv: &str, jumps_csv: &str, every: usize, backend: &str) -> Result<Validation, String> {
    let all = read_csv(union_csv)?;
    let jump_pts = read_csv(jumps_csv)?;
    type Series = (String, u8, u32, u32);
    let series_of = |p: &Point| -> Series { (p.op.clone(), p.rate, if p.op == "EXTEND" { p.kept } else { 0 }, if p.op == "RECURSE" { p.child_sigs } else { 0 }) };
    let x_of = |p: &Point| -> u32 {
        match p.op.as_str() {
            "AGGREGATE" => p.n_sigs,
            "EXTEND" => p.new_sigs,
            _ => p.children,
        }
    };
    let mut by: BTreeMap<Series, Vec<&Point>> = BTreeMap::new();
    for p in jump_pts.iter().filter(|p| ["AGGREGATE", "EXTEND", "RECURSE"].contains(&p.op.as_str()) && !p.verify_children) {
        by.entry(series_of(p)).or_default().push(p);
    }
    let mut held: Vec<Point> = Vec::new();
    for v in by.values_mut() {
        v.sort_by_key(|p| x_of(p));
        let n = v.len();
        for (i, p) in v.iter().enumerate() {
            if i > 0 && i + 1 < n && i % every.max(2) == every.max(2) / 2 {
                held.push((*p).clone());
            }
        }
    }
    let same = |a: &Point, b: &Point| series_of(a) == series_of(b) && x_of(a) == x_of(b);
    let train: Vec<Point> = all.iter().filter(|p| !held.iter().any(|h| same(h, p))).cloned().collect();
    let rates: Vec<u8> = [1u8, 2].into_iter().filter(|&r| train.iter().any(|p| p.op == "AGGREGATE" && p.rate == r)).collect();
    let model = fit_table(&train, common(&train, backend, 8, "avx2"), &rates);
    let mut errs = Vec::new();
    let mut under = 0.0f64;
    let mut lines = Vec::new();
    for h in &held {
        let Some(shape) = shape_of(h) else { continue };
        let pred = model.prove_us(&shape, 8) as f64;
        let rel = (pred - h.us) / h.us;
        under = under.min(rel);
        errs.push((rel.abs(), format!("{:9} rate {} kept {:4} new {:4} n {:4} children {:2}x{:4}: measured {:8.1} ms, predicted {:8.1} ms, error {:+6.1} %", h.op, h.rate, h.kept, h.new_sigs, h.n_sigs, h.children, h.child_sigs, h.us / 1000.0, pred / 1000.0, rel * 100.0)));
    }
    if errs.is_empty() {
        return Err("no held-out points".into());
    }
    let mut mags: Vec<f64> = errs.iter().map(|e| e.0).collect();
    let (median, max) = (median(&mut mags), errs.iter().map(|e| e.0).fold(0.0, f64::max));
    errs.sort_by(|a, b| b.0.total_cmp(&a.0));
    lines.extend(errs.iter().take(10).map(|e| e.1.clone()));
    Ok(Validation { lines, median, max, under: under.min(0.0), n: errs.len() })
}

/// The CSVs as one: the first file's header, then every row. A point measured in two files has all its runs.
pub fn union_csv(texts: &[&str]) -> String {
    let mut out = String::new();
    for (i, t) in texts.iter().enumerate() {
        for (j, line) in t.lines().enumerate() {
            if j == 0 && i > 0 {
                continue;
            }
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}


/// Fit a model file from a cost CSV. `form` forces `rows` or `table`; `None` tries rows, then table.
pub fn fit(csv: &str, backend: &str, threads: u32, isa: &str, form: Option<&str>) -> Result<Report, String> {
    let pts = read_csv(csv)?;
    let rates: Vec<u8> = [1u8, 2].into_iter().filter(|&r| pts.iter().any(|p| p.op == "AGGREGATE" && p.rate == r)).collect();
    if rates.is_empty() {
        return Err("no AGGREGATE rows".into());
    }
    let common = common(&pts, backend, threads, isa);
    if form == Some("table") {
        let table = fit_table(&pts, common, &rates);
        let (lines, med, max) = residuals(&table, &pts);
        return Ok(Report { model_text: table.to_text(), form: "table", median_res: med, max_res: max, lines });
    }
    let mut ext = fit_aggregate(&pts, &rates)?;
    fit_extend(&pts, &rates, &mut ext);
    let merge = fit_recurse(&pts, &rates, 2);
    let rows = RowsModel { common: common.clone(), ext, merge };
    let (lines, med, max) = residuals(&rows, &pts);
    let accepted = med <= 0.03 && max <= 0.08;
    if form == Some("rows") || (form.is_none() && accepted) {
        return Ok(Report { model_text: rows.to_text(), form: "rows", median_res: med, max_res: max, lines });
    }
    let table = fit_table(&pts, common, &rates);
    let (tlines, tmed, tmax) = residuals(&table, &pts);
    let mut lines_out = vec![format!("rows form: median residual {:.1} %, max {:.1} %: not accepted", med * 100.0, max * 100.0)];
    lines_out.extend(tlines);
    Ok(Report { model_text: table.to_text(), form: "table", median_res: tmed, max_res: tmax, lines: lines_out })
}

/// Verification cross-check: RECURSE with `VERIFY_CHILDREN` against without, per child.
pub fn verify_cross_check(csv: &str) -> Result<Vec<String>, String> {
    let pts = read_csv(csv)?;
    let mut out = Vec::new();
    for p in pts.iter().filter(|p| p.op == "RECURSE" && p.verify_children) {
        if let Some(q) = pts.iter().find(|q| q.op == "RECURSE" && !q.verify_children && q.rate == p.rate && q.children == p.children && q.child_sigs == p.child_sigs) {
            out.push(format!("rate {} children {}: (with - without) / children = {:.2} ms per child", p.rate, p.children, (p.us - q.us) / p.children as f64 / 1000.0));
        }
    }
    Ok(out)
}
