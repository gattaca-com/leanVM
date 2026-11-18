mod common;
use arrival_model::schedule::{Arrival, Kind, Meta, PoolEntry, ScheduleSet, SlotSchedule};
use common::*;
use sidecar_streaming::backend::Backend;
use sidecar_streaming::bound::{clairvoyant_leaf, universal_leaf};
use sidecar_streaming::cost::CostModel;
use sidecar_streaming::driver::{RunSpec, run_slot};
use sidecar_streaming::fit::fit;
use sidecar_streaming::metric::*;
use sidecar_streaming::rng::SplitMix64;
use sidecar_streaming::select::select_exact;
use sidecar_streaming::strategies::with_margin;
use sidecar_streaming::strategy::{Ctx, Strategy};
use sidecar_streaming::types::*;
use std::collections::BTreeSet;

fn cov(claims: &[u32]) -> Coverage {
    let mut c = Coverage::empty();
    claims.iter().for_each(|&x| c.insert(x));
    c
}

#[test]
fn coverage_ops() {
    let mut rng = SplitMix64::new(3);
    for _ in 0..200 {
        let mut a = (Coverage::empty(), BTreeSet::new());
        let mut b = (Coverage::empty(), BTreeSet::new());
        for _ in 0..(rng.next_u64() % 300) {
            let x = (rng.next_u64() % MAX_CLAIMS as u64) as u32;
            a.0.insert(x);
            a.1.insert(x);
        }
        for _ in 0..(rng.next_u64() % 300) {
            let x = (rng.next_u64() % MAX_CLAIMS as u64) as u32;
            b.0.insert(x);
            b.1.insert(x);
        }
        assert_eq!(a.0.len() as usize, a.1.len());
        assert_eq!(a.0.union_len(&b.0) as usize, a.1.union(&b.1).count());
        assert_eq!(a.0.contains_all(&b.0), b.1.is_subset(&a.1));
        let mut u = a.0.clone();
        u.union_with(&b.0);
        assert_eq!(u.iter().collect::<Vec<_>>(), a.1.union(&b.1).copied().collect::<Vec<_>>());
        assert!(u.contains_all(&a.0) && u.contains_all(&b.0));
    }
}

fn rust_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

#[test]
fn isolation() {
    let mut files = Vec::new();
    rust_files(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut files);
    assert!(files.len() > 10);
    let key = ["arrival", "_model::"].concat();
    for f in files {
        for (n, line) in std::fs::read_to_string(&f).unwrap().lines().enumerate() {
            let mut rest = line;
            while let Some(i) = rest.find(&key) {
                let after = &rest[i + key.len()..];
                let ident: String = after.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
                assert_eq!(ident, "schedule", "{}:{}: only the schedule module may be named: {line}", f.display(), n + 1);
                rest = after;
            }
        }
    }
}

fn leaf_record(inputs: &[(u64, u32)], published: Option<(u64, &[u32])>, d: u64) -> SlotRecord {
    SlotRecord {
        shape: Shape::Leaf,
        deadline: d,
        inputs: inputs.iter().map(|&(t, c)| InputRec { t, claims: vec![c], valid: true }).collect(),
        published: published.map(|(t, c)| (t, cov(c))),
        jobs: Vec::new(),
        sent_override: None,
        grace: 0,
    }
}

#[test]
fn metric_hand_cases() {
    let inputs = [(100, 0), (200, 1), (300, 2), (1500, 3)];
    // Publish before D: all three that arrived; the late vote counts in `sent` only.
    let m = slot_metrics(&leaf_record(&inputs, Some((900, &[0, 1, 2])), 1000));
    assert_eq!((m.got, m.arrived, m.sent), (3, 3, 4));
    assert_eq!(m.coverage, 1.0);
    assert_eq!(m.coverage_sent, 0.75);
    assert_eq!((m.cut_us, m.staleness_us, m.slack_us), (Some(300), Some(600), Some(100)));
    // Publish exactly at D counts; 1 us after does not.
    assert_eq!(slot_metrics(&leaf_record(&inputs, Some((1000, &[0, 1, 2])), 1000)).got, 3);
    let late = slot_metrics(&leaf_record(&inputs, Some((1001, &[0, 1, 2])), 1000));
    assert_eq!((late.got, late.coverage, late.slack_us), (0, 0.0, None));
    // No publish.
    let none = slot_metrics(&leaf_record(&inputs, None, 1000));
    assert_eq!((none.got, none.coverage, none.published_us), (0, 0.0, None));
    // A partial proof.
    assert_eq!(slot_metrics(&leaf_record(&inputs, Some((900, &[0, 1])), 1000)).coverage, 2.0 / 3.0);
    // Merge: overlapping candidates count once; an invalid one counts nowhere.
    let c1: Vec<u32> = (0..10).collect();
    let c2: Vec<u32> = (5..15).collect();
    let rec = SlotRecord {
        shape: Shape::Merge,
        deadline: 1000,
        inputs: vec![
            InputRec { t: 100, claims: c1, valid: true },
            InputRec { t: 200, claims: c2, valid: true },
            InputRec { t: 300, claims: (100..200).collect(), valid: false },
            InputRec { t: 2000, claims: (1500..1600).collect(), valid: true },
        ],
        published: Some((900, cov(&(0..15).collect::<Vec<_>>()))),
        jobs: Vec::new(),
        sent_override: None,
        grace: 0,
    };
    let m = slot_metrics(&rec);
    assert_eq!((m.got, m.arrived, m.sent), (15, 15, 115));
    // `end_to_end` charges the votes of all subnets.
    let rec = SlotRecord { sent_override: Some(30), ..rec };
    assert_eq!(slot_metrics(&rec).coverage_sent, 0.5);
}

#[test]
fn quantiles_are_nearest_rank() {
    let xs: Vec<f64> = (1..=10).map(|x| x as f64).collect();
    assert_eq!(quantile(&xs, 50), 5.0);
    assert_eq!(quantile(&xs, 10), 1.0);
    assert_eq!(quantile(&xs, 90), 9.0);
    assert_eq!(d99(&[(1, 0.5), (2, 0.995), (3, 1.0)]), Some(2));
}

/// A strategy that does what a closure says, and logs every event.
type OnStart = Box<dyn FnMut(&mut dyn Ctx)>;

struct Script {
    log: Vec<String>,
    on_start: OnStart,
}

impl Strategy for Script {
    fn name(&self) -> String {
        "script".into()
    }
    fn on_event(&mut self, event: Event, ctx: &mut dyn Ctx) {
        self.log.push(match &event {
            Event::SlotStart => "start".into(),
            Event::Signature(s) => format!("sig{}", s.0),
            Event::Aggregate(p) => format!("agg{}", p.0),
            Event::Proved { result, .. } => format!("proved:{}", result.is_ok()),
            Event::Verified { ok, .. } => format!("verified:{ok}"),
            Event::Timer(t) => format!("timer{t}"),
        });
        if matches!(event, Event::SlotStart) {
            (self.on_start)(ctx);
        }
    }
}

fn hand_set(shape: &str, arrivals: Vec<Arrival>, pool: Vec<PoolEntry>) -> ScheduleSet {
    ScheduleSet {
        meta: Meta {
            shape: shape.into(),
            slots: 1,
            slot_us: 4_000_000,
            att_due_us: 1_000_000,
            agg_due_us: 2_000_000,
            ptc_due_us: 3_000_000,
            deadline_us: 3_000_000,
            horizon_us: 8_000_000,
            final_rate: if shape == "merge" { 1 } else { 2 },
            expected_inputs: 1500,
            seed: 0,
            source: "recorded".into(),
            subnet_size: 1500,
            subnets: 1,
            views: 1,
        },
        params_txt: None,
        pool,
        slots: vec![SlotSchedule { arrivals }],
    }
}

fn vote(t_us: u64, id: u32) -> Arrival {
    Arrival { t_us, kind: Kind::Vote, subnet: 0, view: 0, id, valid: true }
}

fn run_script(backend: &mut dyn Backend, set: &ScheduleSet, s: &mut Script, d: u64) -> SlotRecord {
    let caps = backend.caps();
    run_slot(backend, s, &RunSpec { set, slot: 0, subnet: 0, view: 0, deadline: d, caps, grace: 0 })
}

#[test]
fn driver_event_order() {
    let mut backend = sim(SMOOTH);
    let t = backend.model().prove_us(&JobShape::aggregate(1, 2), 8);
    // At time `t` three things are due: a vote, the end of the job, and a timer.
    let set = hand_set("leaf", vec![vote(t, 0)], Vec::new());
    let mut s = Script {
        log: Vec::new(),
        on_start: Box::new(move |ctx| {
            ctx.prove(0, Job { children: vec![], signatures: vec![SigId(0)], rate: 2 });
            ctx.set_timer(t, 7);
        }),
    };
    run_script(&mut backend, &set, &mut s, 3_000_000);
    assert_eq!(s.log, ["start", "sig0", "proved:true", "timer7"]);
}

#[test]
#[should_panic(expected = "busy lane")]
fn busy_lane_prove_panics() {
    let mut backend = sim(SMOOTH);
    let set = hand_set("leaf", vec![], Vec::new());
    let mut s = Script {
        log: Vec::new(),
        on_start: Box::new(|ctx| {
            let job = Job { children: vec![], signatures: vec![SigId(0)], rate: 2 };
            ctx.prove(0, job.clone());
            ctx.prove(0, job);
        }),
    };
    run_script(&mut backend, &set, &mut s, 3_000_000);
}

#[test]
#[should_panic(expected = "second publish")]
fn second_publish_panics() {
    let mut backend = sim(SMOOTH);
    let set = hand_set("leaf", vec![], Vec::new());
    let mut s = Script {
        log: Vec::new(),
        on_start: Box::new(|ctx| {
            ctx.prove(0, Job { children: vec![], signatures: vec![SigId(0)], rate: 2 });
        }),
    };
    struct Twice(Script);
    impl Strategy for Twice {
        fn name(&self) -> String {
            "twice".into()
        }
        fn on_event(&mut self, e: Event, ctx: &mut dyn Ctx) {
            if let Event::Proved { result: Ok(p), .. } = &e {
                ctx.publish(*p);
                ctx.publish(*p);
            }
            self.0.on_event(e, ctx);
        }
    }
    let mut t = Twice(Script { log: Vec::new(), on_start: std::mem::replace(&mut s.on_start, Box::new(|_| {})) });
    run_slot(&mut backend, &mut t, &RunSpec { set: &set, slot: 0, subnet: 0, view: 0, deadline: 3_000_000, caps: Caps::default(), grace: 0 });
}

#[test]
fn sim_one_shot_exact() {
    const TICK: u64 = 5_000;
    for text in [SMOOTH, STEPPED] {
        let set = make_set(&leaf_params(), 20, 5);
        let mut backend = sim(text);
        let m = model(text);
        for slot in 0..20 {
            // D falls inside the vote window: the cut is early.
            let d = 1_500_000;
            let (rec, met) = run(&mut backend, &set, slot, d, "one_shot_late", None);
            let t_p = met.published_us.expect("published");
            assert!(t_p <= d, "published {t_p} for D {d}");
            // Smooth cost: the last job starts in the last tick and ends within one tick, plus the strategy's margin, of D.
            // A step in the cost can leave more room.
            assert!(text != SMOOTH || t_p + TICK + 12_000 > d, "published {t_p} for D {d}");
            let n = met.got;
            let cut = t_p - m.prove_us(&JobShape::aggregate(n, 2), 8);
            // The proof holds every vote that arrived by its start, up to what fits.
            let arrived = rec.inputs.iter().filter(|i| i.t <= cut).count() as u32;
            assert!(n <= arrived);
            let fits = |n: u32| cut + with_margin(m.prove_us(&JobShape::aggregate(n, 2), 8)) <= d;
            assert!(fits(n), "slot {slot}: n {n} does not fit");
            assert!(n == arrived || !fits(n + 1), "slot {slot}: n {n} arrived {arrived}");
        }
    }
}

#[test]
fn sim_invalid_child() {
    let entry = |id: u32, claims: std::ops::Range<u32>| {
        let mut signers = vec![0u64; 24];
        for s in claims {
            signers[s as usize / 64] |= 1 << (s % 64);
        }
        PoolEntry { subnet: 0, id, signers, rate: 2, slot: None }
    };
    let pool = vec![entry(0, 0..100), entry(1, 100..200), entry(2, 200..300)];
    let agg = |t: u64, id: u32, valid: bool| Arrival { t_us: t, kind: Kind::Aggregate, subnet: 0, view: 0, id, valid };
    let set = hand_set("merge", vec![agg(1000, 0, true), agg(2000, 1, false), agg(3000, 2, true)], pool);
    let mut backend = sim(STEPPED);
    let v = backend.model().verify_us(&ChildShape { claims: 100, rate: 2, verify: true, leaves: 1 });
    struct Try(Vec<ProofId>, Vec<(u64, Result<ProofId, Failure>)>);
    impl Strategy for Try {
        fn name(&self) -> String {
            "try".into()
        }
        fn on_event(&mut self, e: Event, ctx: &mut dyn Ctx) {
            match e {
                Event::Aggregate(p) => {
                    self.0.push(p);
                    if self.0.len() == 3 {
                        ctx.prove(0, Job { children: self.0.clone(), signatures: vec![], rate: 1 });
                    }
                }
                Event::Proved { result, .. } => self.1.push((ctx.now(), result)),
                _ => {}
            }
        }
    }
    let mut t = Try(Vec::new(), Vec::new());
    run_slot(&mut backend, &mut t, &RunSpec { set: &set, slot: 0, subnet: 0, view: 0, deadline: 3_000_000, caps: Caps::default(), grace: 0 });
    // The second child is bad: the job fails after verifying the first two, and names the second.
    assert_eq!(t.1.len(), 1);
    assert_eq!(t.1[0].0, 3000 + 2 * v);
    assert_eq!(t.1[0].1, Err(Failure::InvalidChild(ProofId(1))));
}

#[test]
fn rows_model_steps() {
    // One kept child of fixed rows 150 (wc), so W = 150 + new; the padded size jumps where W crosses 170 * 2^k.
    let m = model(STEPPED);
    let cost = |n: u32| m.prove_us(&JobShape::extend(1000, 2, n, 2), 8);
    let mut prev = cost(1);
    let mut jumps = Vec::new();
    for n in 2..=2500 {
        let c = cost(n);
        assert!(c >= prev, "not monotone at {n}");
        if c - prev > 1000 {
            jumps.push(n);
        }
        prev = c;
    }
    // W = 150 + n crosses 170, 340, 680 and 1360 at n = 21, 191, 531 and 1211: the jump is at the first n past each.
    assert_eq!(jumps, vec![21, 191, 531, 1211]);
    // The smooth model has none.
    let m = model(SMOOTH);
    let jump = (2..=2500).any(|n| m.prove_us(&JobShape::aggregate(n, 2), 8) - m.prove_us(&JobShape::aggregate(n - 1, 2), 8) > 1000);
    assert!(!jump);
}

#[test]
fn fit_recovers_parameters() {
    let truth = model(STEPPED);
    let mut rng = SplitMix64::new(11);
    let mut csv = String::from("op,rate,n_sigs,children,child_sigs,kept_sigs,new_sigs,verify_children,run,ms,proof_bytes\n");
    let mut row = |op: &str, rate: u8, key: [u32; 6], shape: JobShape, csv: &mut String| {
        let us = truth.prove_us(&shape, 8) as f64;
        for run in 1..=5 {
            let ms = us / 1000.0 * (1.0 + 0.02 * rng.normal()).max(0.5);
            csv.push_str(&format!("{op},{rate},{},{},{},{},{},{},{run},{ms:.3},250000\n", key[0], key[1], key[2], key[3], key[4], key[5]));
        }
    };
    for rate in [1u8, 2] {
        for n in [1u32, 16, 32, 64, 100, 128, 150, 200, 256, 300, 384, 500, 512, 600, 750, 1000, 1024, 1200, 1500] {
            row("AGGREGATE", rate, [n, 0, 0, 0, 0, 0], JobShape::aggregate(n, rate), &mut csv);
        }
        for kept in [150u32, 300, 750, 1350] {
            for new in [1u32, 50, 150, 300, 750] {
                row("EXTEND", rate, [0, 0, 0, kept, new, 0], JobShape::extend(kept, rate, new, rate), &mut csv);
            }
        }
        for (c, s) in [(2u32, 1500u32), (3, 1500), (4, 1500), (6, 1500), (8, 1500), (12, 1500), (16, 1500), (2, 100), (4, 100), (16, 100)] {
            let shape = JobShape {
                children: vec![ChildShape { claims: s, rate: 2, verify: false, leaves: 1 }; c as usize],
                new_sigs: 0,
                rate,
            };
            row("RECURSE", rate, [0, c, s, 0, 0, 0], shape, &mut csv);
        }
        for (c, n) in [(0u32, 1500u32), (16, 0)] {
            for run in 1..=5 {
                csv.push_str(&format!("VERIFY,{rate},{n},{c},{},0,0,0,{run},6.5,250000\n", if c > 0 { 1500 } else { 0 }));
            }
        }
    }
    let report = fit(&csv, "test", 8, "avx2", Some("rows")).unwrap();
    assert_eq!(report.form, "rows");
    eprintln!("{}", report.model_text);
    let fitted = sidecar_streaming::cost::parse_model(&report.model_text).unwrap();
    let mut worst: f64 = 0.0;
    let mut shapes = vec![];
    for rate in [1u8, 2] {
        for n in [10u32, 90, 150, 260, 320, 750, 1100, 1500] {
            // Away from the jumps, where a step off by one row would be seen.
            shapes.push(JobShape::aggregate(n, rate));
            shapes.push(JobShape::extend(900, rate, n, rate));
        }
        for c in [2usize, 5, 9, 16] {
            shapes.push(JobShape { children: vec![ChildShape { claims: 1500, rate: 2, verify: false, leaves: 1 }; c], new_sigs: 0, rate });
        }
    }
    for s in &shapes {
        let (a, b) = (truth.prove_us(s, 8) as f64, fitted.prove_us(s, 8) as f64);
        let rel = (a - b).abs() / a;
        if rel > 0.05 {
            eprintln!("{s:?}: truth {a}, fitted {b}");
        }
        worst = worst.max(rel);
    }
    assert!(worst <= 0.05, "worst relative error {worst}");
}

fn brute(arr: &[u64], m: &dyn CostModel, d: u64, f: Rate, cheap: Rate) -> u32 {
    // Every op sequence from the empty start: step to a longer prefix at some rate, or convert to the final rate.
    #[allow(clippy::too_many_arguments)]
    fn go(arr: &[u64], m: &dyn CostModel, d: f64, f: Rate, cheap: Rate, i: usize, rk: Rate, t: f64, best: &mut usize) {
        if rk == f && t <= d && i > *best {
            *best = i;
        }
        for j in i + 1..=arr.len() {
            for r in [cheap, f] {
                let shape = if i == 0 { JobShape::aggregate(j as u32, r) } else { JobShape::extend(i as u32, rk, (j - i) as u32, r) };
                let nt = t.max(arr[j - 1] as f64) + m.prove_us(&shape, 8) as f64;
                if nt <= d {
                    go(arr, m, d, f, cheap, j, r, nt, best);
                }
            }
        }
        if i > 0 && rk != f {
            let nt = t + m.prove_us(&JobShape::extend(i as u32, rk, 0, f), 8) as f64;
            if nt <= d {
                go(arr, m, d, f, cheap, i, f, nt, best);
            }
        }
    }
    let mut best = 0;
    go(arr, m, d as f64, f, cheap, 0, 0, 0.0, &mut best);
    best as u32
}

#[test]
fn dp_matches_brute_force() {
    let m = model(STEPPED);
    let mut rng = SplitMix64::new(9);
    let mut agree = 0;
    for _ in 0..60 {
        let mut arr: Vec<u64> = (0..8).map(|_| 50_000 + rng.next_u64() % 400_000).collect();
        arr.sort_unstable();
        let d = 150_000 + rng.next_u64() % 700_000;
        let (dp, bf) = (clairvoyant_leaf(&arr, m.as_ref(), d, 2, 1, 1), brute(&arr, m.as_ref(), d, 2, 1));
        // The DP only counts arrivals by `d`; so does the brute force through the deadline on every proof.
        assert_eq!(dp, bf, "arr {arr:?} d {d}");
        agree += 1;
    }
    assert_eq!(agree, 60);
}

#[test]
fn bound_above_strategies() {
    let set = make_set(&leaf_params(), 25, 21);
    let train = make_set(&leaf_params(), 30, 22);
    for text in [STEPPED, SMOOTH] {
        let mut backend = sim(text);
        let m = model(text);
        for d in [1_200_000u64, 1_500_000, 2_000_000] {
            for slot in 0..25 {
                let arr: Vec<u64> = set.slots[slot as usize].arrivals.iter().map(|a| a.t_us).collect();
                // Grid 1 is the only grid for which this is a bound.
                let clair = clairvoyant_leaf(&arr, m.as_ref(), d, 2, 1, 1);
                let univ = universal_leaf(&arr, m.as_ref(), d, 2, 1);
                assert!(clair <= univ, "clairvoyant {clair} above universal {univ} (slot {slot}, D {d})");
                let strategies = ["one_shot_late", "greedy_fit", "min_batch_fit:m=300", "fixed_fit:b=600", "step_cut_fit", "cheap_chain", "two_stage", "plan_cover"];
                for s in strategies {
                    // The planner replans often: a few slots.
                    if s == "plan_cover" && slot >= 4 {
                        continue;
                    }
                    let (_, met) = run(&mut backend, &set, slot, d, s, Some(&train));
                    assert!(met.got <= clair, "{s}: got {} above clairvoyant {clair} (slot {slot}, D {d}, {text:.20})", met.got);
                    assert!(met.got <= univ, "{s}: got {} above universal {univ}", met.got);
                }
            }
        }
    }
}

#[test]
fn strategies_publish_once() {
    for (params, names, ds, slots) in [
        (leaf_params(), sidecar_streaming::strategies::LEAF, [1_500_000u64, 2_000_000, 3_000_000], 200),
        (merge_params(), sidecar_streaming::strategies::MERGE, [3_000_000u64, 3_500_000, 4_000_000], 200),
    ] {
        let set = make_set(&params, slots, 31);
        let train = make_set(&params, 30, 77);
        let mut backend = sim(STEPPED);
        for name in names {
            // The two baselines without a cost estimate guarantee nothing before `D`: see `fix_simple.rs`.
            if matches!(*name, "greedy_simple" | "once_at") {
                continue;
            }
            // The planners replan often: fewer slots.
            let n = if name.starts_with("plan_") { 8 } else { slots };
            for d in ds {
                for slot in 0..n {
                    let (rec, m) = run(&mut backend, &set, slot, d, name, Some(&train));
                    let (t, p) = rec.published.as_ref().unwrap_or_else(|| panic!("{name}: no publish (slot {slot}, D {d})"));
                    assert!(*t <= d && !p.is_empty(), "{name}: published {t} for D {d}");
                    assert!(m.got > 0);
                }
            }
        }
    }
}

#[test]
fn coverage_monotone_in_d() {
    let set = make_set(&leaf_params(), 30, 41);
    let train = make_set(&leaf_params(), 30, 43);
    for text in [STEPPED, SMOOTH] {
        let mut backend = sim(text);
        for (name, n) in [("one_shot_late", 30u32), ("plan_cover", 8)] {
            let mut prev = 0.0;
            for d in [1_000_000u64, 1_200_000, 1_400_000, 1_700_000, 2_000_000] {
                let mean: f64 = (0..n).map(|s| run(&mut backend, &set, s, d, name, Some(&train)).1.coverage_sent).sum::<f64>() / n as f64;
                assert!(mean + 1e-9 >= prev, "{name}: mean coverage_sent fell from {prev} to {mean} at D {d}");
                prev = mean;
            }
        }
    }
}

#[test]
fn select_exact_matches_brute_force() {
    let mut rng = SplitMix64::new(5);
    for _ in 0..200 {
        // 4 subnets x 3 candidates over a small claim range, 6 child slots, a random base.
        let mut cands: Vec<(ProofId, u8, Coverage)> = Vec::new();
        for s in 0..4u8 {
            for c in 0..3 {
                let claims: Vec<u32> = (0..12).filter(|_| rng.next_u64().is_multiple_of(3)).map(|x| s as u32 * SUBNET_SIZE + x).collect();
                cands.push((ProofId(s as u32 * 3 + c), s, cov(&claims)));
            }
        }
        let mut base_claims: Vec<u32> = Vec::new();
        for s in 0..4u32 {
            for x in 0..12 {
                if rng.next_u64().is_multiple_of(5) {
                    base_claims.push(s * SUBNET_SIZE + x);
                }
            }
        }
        let base = cov(&base_claims);
        let refs: Vec<(ProofId, u8, &Coverage)> = cands.iter().map(|c| (c.0, c.1, &c.2)).collect();
        let (chosen, gain) = select_exact(&refs, &base, 6);
        assert!(chosen.len() <= 6);
        let mut u = base.clone();
        for id in &chosen {
            u.union_with(&cands[id.0 as usize].2);
        }
        assert_eq!(u.len() - base.len(), gain);
        let mut best = 0;
        for mask in 0u32..(1 << 12) {
            if mask.count_ones() > 6 {
                continue;
            }
            let mut u = base.clone();
            (0..12).filter(|i| mask >> i & 1 == 1).for_each(|i| u.union_with(&cands[i].2));
            best = best.max(u.len() - base.len());
        }
        assert_eq!(gain, best);
    }
}
