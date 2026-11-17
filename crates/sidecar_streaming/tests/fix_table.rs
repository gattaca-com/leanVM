//! Regression tests for findings 1, 2 and 9 of `design/REVIEW.md` (jumps, cross-rate EXTEND, RECURSE) and for
//! pricing a leaf of up to 2048 signatures from measured points.
mod common;
use common::*;
use sidecar_streaming::cost::{CostModel, parse_model};
use sidecar_streaming::fit::{self, Point, find_jumps, read_csv, union_csv};
use sidecar_streaming::types::*;
use std::path::PathBuf;

const HEADER: &str = "op,rate,n_sigs,children,child_sigs,kept_sigs,new_sigs,verify_children,run,ms,proof_bytes\n";

/// Rows of a point, three runs of the same value.
fn rows(op: &str, rate: u8, key: [u32; 6], ms: f64, out: &mut String) {
    for run in 1..=3 {
        out.push_str(&format!("{op},{rate},{},{},{},{},{},{},{run},{ms:.3},250000\n", key[0], key[1], key[2], key[3], key[4], key[5]));
    }
}

/// A synthetic AGGREGATE series with a doubling ladder of jumps at 176, 352, 703 and 1406, one point per 8 signatures
/// and one per signature next to each jump.
fn ladder_csv() -> (String, Vec<u32>) {
    let jumps = [176u32, 352, 703, 1406];
    let cost = |n: u32| 40.0 + 0.5 * n as f64 + jumps.iter().filter(|&&j| n >= j).count() as f64 * 150.0;
    let mut xs: Vec<u32> = (1..=2048).step_by(8).collect();
    for j in jumps {
        xs.extend([j - 2, j - 1, j, j + 1]);
    }
    xs.sort_unstable();
    xs.dedup();
    let mut csv = String::from(HEADER);
    for n in xs {
        rows("AGGREGATE", 2, [n, 0, 0, 0, 0, 0], cost(n), &mut csv);
    }
    (csv, jumps.to_vec())
}

/// Finding 1: every jump the data shows is in the model, at its measured size, and no unmeasured stretch is read as smooth.
#[test]
fn model_reproduces_each_measured_jump() {
    let (csv, jumps) = ladder_csv();
    let report = fit::fit(&csv, "t", 8, "avx2", Some("table")).unwrap();
    let m = parse_model(&report.model_text).unwrap();
    let at = |n: u32| m.prove_us(&JobShape::aggregate(n, 2), 8) as f64 / 1000.0;
    let meas = |n: u32| 40.0 + 0.5 * n as f64 + jumps.iter().filter(|&&j| n >= j).count() as f64 * 150.0;
    for j in jumps.clone() {
        assert!((at(j - 1) / meas(j - 1) - 1.0).abs() < 0.1, "below the jump at {j}");
        assert!((at(j) / meas(j) - 1.0).abs() < 0.1, "at the jump at {j}");
        assert!(at(j) > at(j - 1) + 100.0, "the jump at {j} is lost");
    }
    // Between two measured points the next one, at or above, is used: a size just under a jump that was not measured
    // costs what the jump point does, never the smooth interpolation below it.
    assert_eq!(at(1000), at(1001));
    assert!(at(1405) <= at(1406));
    // The model file lists the jumps.
    assert!(report.model_text.contains("agg.2.jump 1405 1406"), "{}", report.model_text.lines().filter(|l| l.contains("jump")).collect::<Vec<_>>().join("; "));
    // No extrapolation: past the largest measured size the cost is flat.
    assert_eq!(at(2048), at(5000));
}

fn real(name: &str) -> Option<String> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../cost").join(name);
    std::fs::read_to_string(p).ok()
}

fn real_model(backend: &str) -> Option<(Box<dyn CostModel>, Vec<Point>)> {
    let a = real(&format!("{backend}-avx2-8c.csv"))?;
    let b = real(&format!("{backend}-avx2-8c-jumps.csv"))?;
    let csv = union_csv(&[&a, &b]);
    let report = fit::fit(&csv, backend, 8, "avx2", Some("table")).ok()?;
    Some((parse_model(&report.model_text).ok()?, read_csv(&csv).ok()?))
}

fn find<'a>(pts: &'a [Point], op: &str, rate: u8, f: impl Fn(&Point) -> bool) -> Option<&'a Point> {
    pts.iter().find(|p| p.op == op && p.rate == rate && f(p))
}

/// Finding 1 on the box8 data: the ladder on main and each AGGREGATE and EXTEND jump are reproduced within 10 %.
#[test]
fn box8_jumps_are_reproduced() {
    for backend in ["main", "perf"] {
        let Some((m, pts)) = real_model(backend) else {
            eprintln!("no cost data for {backend}: skipped");
            continue;
        };
        let mut checked = 0;
        for rate in [1u8, 2] {
            let agg: Vec<(f64, f64)> = pts.iter().filter(|p| p.op == "AGGREGATE" && p.rate == rate).map(|p| (p.n_sigs as f64, p.us)).collect();
            let mut agg = agg;
            agg.sort_by(|a, b| a.0.total_cmp(&b.0));
            agg.dedup_by(|a, b| a.0 == b.0);
            for (a, b, _) in find_jumps(&agg).0 {
                for n in [a as u32, b as u32] {
                    let p = find(&pts, "AGGREGATE", rate, |p| p.n_sigs == n).unwrap();
                    let got = m.prove_us(&JobShape::aggregate(n, rate), 8) as f64;
                    assert!((got / p.us - 1.0).abs() <= 0.1, "{backend} rate {rate} AGGREGATE {n}: model {got}, measured {}", p.us);
                    checked += 1;
                }
            }
            for kept in [300u32, 1350, 1500] {
                let mut ext: Vec<(f64, f64)> = pts.iter().filter(|p| p.op == "EXTEND" && p.rate == rate && p.kept == kept).map(|p| (p.new_sigs as f64, p.us)).collect();
                ext.sort_by(|a, b| a.0.total_cmp(&b.0));
                ext.dedup_by(|a, b| a.0 == b.0);
                for (a, b, _) in find_jumps(&ext).0 {
                    for new in [a as u32, b as u32] {
                        let p = find(&pts, "EXTEND", rate, |p| p.kept == kept && p.new_sigs == new).unwrap();
                        let got = m.prove_us(&JobShape::extend(kept, rate, new, rate), 8) as f64;
                        assert!((got / p.us - 1.0).abs() <= 0.1, "{backend} rate {rate} EXTEND kept {kept} new {new}: model {got}, measured {}", p.us);
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 20, "{backend}: only {checked} jump points checked");
    }
    // The ladder on main is there.
    if let Some((m, _)) = real_model("main") {
        let at = |n| m.prove_us(&JobShape::aggregate(n, 2), 8) as f64;
        for (lo, hi) in [(175, 176), (351, 352), (702, 703), (1405, 1406)] {
            assert!(at(hi) > at(lo) + 80_000.0, "no jump at {hi} on main");
        }
    }
}

/// Finding 2: the cross-rate cost is the mean of the measured penalties, applies to an EXTEND whose kept proof has
/// another rate, and can be overridden.
#[test]
fn cross_rate_extend_is_priced() {
    let mut csv = String::from(HEADER);
    // Same-rate cells at 100 ms, cross cells 20 and 60 ms higher: mean 40 ms.
    for (kept, new, s, x) in [(300u32, 1u32, 100.0, 120.0), (300, 150, 200.0, 260.0)] {
        rows("EXTEND_S", 2, [0, 0, 0, kept, new, 0], s, &mut csv);
        rows("EXTEND_X", 2, [0, 0, 0, kept, new, 0], x, &mut csv);
        rows("EXTEND", 2, [0, 0, 0, kept, new, 0], s, &mut csv);
    }
    rows("AGGREGATE", 2, [1, 0, 0, 0, 0, 0], 40.0, &mut csv);
    let report = fit::fit(&csv, "t", 8, "avx2", Some("table")).unwrap();
    assert!(report.model_text.contains("cross_rate_us 40000"), "{}", report.model_text);
    let mut m = parse_model(&report.model_text).unwrap();
    let same = m.prove_us(&JobShape::extend(300, 2, 150, 2), 8);
    let cross = m.prove_us(&JobShape::extend(300, 1, 150, 2), 8);
    assert_eq!(cross - same, 40_000);
    // The worst case seen, from the command line.
    m.set_cross_rate_us(89_000.0);
    assert_eq!(m.prove_us(&JobShape::extend(300, 1, 150, 2), 8) - same, 89_000);
    // A job of one rate is not charged.
    assert_eq!(m.prove_us(&JobShape::extend(300, 2, 150, 2), 8), same);
}

/// Finding 2 on the box8 data: the fitted figures are the ones measured, +35 ms on main and +30 ms on perf.
#[test]
fn box8_cross_rate_figures() {
    for (backend, lo, hi) in [("main", 33_000.0, 36_500.0), ("perf", 28_000.0, 32_000.0)] {
        let Some((m, _)) = real_model(backend) else { continue };
        let same = m.prove_us(&JobShape::extend(300, 2, 150, 2), 8) as f64;
        let cross = m.prove_us(&JobShape::extend(300, 1, 150, 2), 8) as f64;
        assert!((lo..=hi).contains(&(cross - same)), "{backend}: cross-rate cost {}", cross - same);
    }
}

/// Finding 9: RECURSE is priced from the measured table, uneven as it is, and a kept root adds its measured excess.
#[test]
fn box8_recurse_is_the_measured_table() {
    for backend in ["main", "perf"] {
        let Some((m, pts)) = real_model(backend) else { continue };
        let fresh = |c: u32, claims: u32, rate: Rate| JobShape {
            children: vec![ChildShape { claims, rate: 2, verify: false, leaves: 1 }; c as usize],
            new_sigs: 0,
            rate,
        };
        let mut seen = 0;
        for p in pts.iter().filter(|p| p.op == "RECURSE" && !p.verify_children && p.child_sigs == 1500) {
            let got = m.prove_us(&fresh(p.children, 1500, p.rate), 8) as f64;
            assert!((got / p.us - 1.0).abs() <= 0.01, "{backend} RECURSE rate {} c {}: model {got}, measured {}", p.rate, p.children, p.us);
            seen += 1;
        }
        assert!(seen >= 30, "{backend}: {seen} RECURSE points");
        // Children of 1700 cost what 1500 do.
        assert_eq!(m.prove_us(&fresh(16, 1700, 1), 8), m.prove_us(&fresh(16, 1500, 1), 8));
        // A kept root plus fresh children, at rate 1: the measured rows.
        for p in pts.iter().filter(|p| p.op == "RECURSE_KEPT") {
            let mut shape = fresh(p.children, 1500, 1);
            shape.children.push(ChildShape { claims: 24000, rate: 1, verify: false, leaves: 16 });
            let got = m.prove_us(&shape, 8) as f64;
            assert!((got / p.us - 1.0).abs() <= 0.02, "{backend} kept root + {}: model {got}, measured {}", p.children, p.us);
        }
    }
    if let Some((m, _)) = real_model("main") {
        // Main at rate 1: 16 children cost less than 15, as measured.
        let c = |n: usize| m.prove_us(&JobShape { children: vec![ChildShape { claims: 1500, rate: 2, verify: false, leaves: 1 }; n], new_sigs: 0, rate: 1 }, 8);
        assert!(c(16) < c(15));
    }
}

/// A leaf of 1700 or 2048 signatures is priced from the points measured there, with no extrapolation.
#[test]
fn leaves_up_to_2048_are_priced_from_measurements() {
    for backend in ["main", "perf"] {
        let Some((m, pts)) = real_model(backend) else { continue };
        // Measured at 1692, 1724 and 2048 (step 32 above 1500, finer near jumps); 1700 is priced as the next point up, 1724.
        for n in [1692u32, 1724, 2048] {
            for rate in [1u8, 2] {
                let p = find(&pts, "AGGREGATE", rate, |p| p.n_sigs == n).unwrap_or_else(|| panic!("{backend}: no AGGREGATE point at {n}"));
                let got = m.prove_us(&JobShape::aggregate(n, rate), 8) as f64;
                assert!((got / p.us - 1.0).abs() <= 0.01, "{backend} rate {rate} n {n}: model {got}, measured {}", p.us);
            }
        }
        for rate in [1u8, 2] {
            let (at, next) = (m.prove_us(&JobShape::aggregate(1700, rate), 8), m.prove_us(&JobShape::aggregate(1724, rate), 8));
            assert_eq!(at, next, "{backend} rate {rate}");
        }
    }
    assert_eq!(Caps::default().max_new_sigs, 2048);
}

/// Every strategy runs a subnet of 1700 votes: nothing assumes 1500.
#[test]
fn strategies_run_a_subnet_of_1700() {
    let params = arrival_model::model::Params::parse(
        "shape leaf\nslot_ms 4000\nsubnets 1\nsubnet_size 1700\nagg.per_subnet 1\nvotes.base uniform(567, 1034)\n",
        std::path::Path::new("."),
    );
    let set = make_set(&params, 5, 3);
    assert_eq!(set.meta.expected_inputs, 1700);
    let mut backend = sim(SMOOTH);
    for s in ["one_shot_late", "greedy_fit", "min_batch_fit", "cheap_chain", "step_cut_fit"] {
        for slot in 0..5 {
            let (_, m) = run(&mut backend, &set, slot, 3_000_000, s, None);
            assert_eq!((m.arrived, m.got), (1700, 1700), "{s}");
        }
    }
}
