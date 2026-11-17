use arrival_model::layers::{Item, Transform};
use arrival_model::model::{Params, build, generate};
use arrival_model::rng::SplitMix64;
use arrival_model::sampler::{Dir, Env, inverse_cdf, parse_sampler};
use arrival_model::schedule::ScheduleSet;
use arrival_model::xmss;
use std::path::PathBuf;

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("arrival_model-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn leaf_params(extra: &str) -> Params {
    let text = format!("shape leaf\nsubnets 2\nsubnet_size 100\nagg.per_subnet 2\nvotes.base lognormal(300, 0.5)\nvotes.missing 0.1\n{extra}");
    Params::parse(&text, &std::env::temp_dir())
}

fn merge_params() -> Params {
    let text = "shape merge\nsubnets 3\nsubnet_size 100\nagg.per_subnet 3\nagg.pool_sizes 100,90\nagg.base normal(7000, 100)\nagg.invalid 0.2\nagg.missing 0.1\n";
    Params::parse(text, &std::env::temp_dir())
}

fn set_bytes(dir: &std::path::Path) -> Vec<Vec<u8>> {
    ["meta.txt", "params.txt", "pool.csv", "slots.csv"].iter().map(|f| std::fs::read(dir.join(f)).unwrap_or_default()).collect()
}

#[test]
fn schedule_roundtrip() {
    for p in [leaf_params(""), merge_params()] {
        let model = build(&p).unwrap();
        let (a, b, c) = (tmp("a"), tmp("b"), tmp("c"));
        let set = generate(model.as_ref(), &p, 6, 42);
        set.write(&a).unwrap();
        // Read and write again: byte-identical.
        ScheduleSet::read(&a).unwrap().write(&b).unwrap();
        assert_eq!(set_bytes(&a), set_bytes(&b));
        // Same params and seed: the same set.
        generate(build(&p).unwrap().as_ref(), &p, 6, 42).write(&c).unwrap();
        assert_eq!(set_bytes(&a), set_bytes(&c));
        // Slots in any order are identical.
        for slot in [5u32, 0, 3, 3] {
            assert_eq!(model.slot(42, slot), set.slots[slot as usize]);
        }
        assert!(set.slots.iter().all(|s| !s.arrivals.is_empty()));
    }
}

fn write_cdf(dir: &std::path::Path, name: &str, header: &str, pts: &[(f64, f64)]) -> String {
    let mut s = format!("{header}\n");
    for (a, b) in pts {
        s += &format!("{a},{b}\n");
    }
    std::fs::write(dir.join(name), s).unwrap();
    name.to_string()
}

fn quantile(xs: &mut [f64], p: f64) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[((p * xs.len() as f64).ceil() as usize - 1).min(xs.len() - 1)]
}

#[test]
fn samplers_match() {
    let dir = tmp("samplers");
    let pts = [(100.0, 0.0), (200.0, 0.3), (250.0, 0.8), (400.0, 1.0)];
    let cdf_file = write_cdf(&dir, "t.csv", "t_ms,cdf", &pts);
    let u_pts = [(0.0, 0.0), (0.4, 0.5), (1.0, 1.0)];
    let u_file = write_cdf(&dir, "u.csv", "u,cdf", &u_pts);
    let z = |p: f64| if p < 0.5 { -1.2815516 } else if p > 0.5 { 1.2815516 } else { 0.0 };
    type Q = Box<dyn Fn(f64) -> f64>;
    let cases: Vec<(String, Q)> = vec![
        ("const(7)".into(), Box::new(|_| 7.0)),
        ("uniform(100, 500)".into(), Box::new(|p| 100.0 + 400.0 * p)),
        ("normal(300, 60)".into(), Box::new(move |p| 300.0 + 60.0 * z(p))),
        ("lognormal(300, 0.4)".into(), Box::new(move |p| 300.0 * (0.4 * z(p)).exp())),
        ("exp(100)".into(), Box::new(|p| -100.0 * (1.0 - p).ln())),
        ("pareto(100, 3)".into(), Box::new(|p| 100.0 * (1.0 - p).powf(-1.0 / 3.0))),
        (format!("cdf({cdf_file})"), Box::new(move |p| inverse_cdf(&pts, p))),
        (format!("scaled({u_file}, 1700, 500)"), Box::new(move |p| 1700.0 + 500.0 * inverse_cdf(&u_pts, p))),
        ("mix(0.5: const(100), 0.5: const(900))".into(), Box::new(|p| if p < 0.5 { 100.0 } else { 900.0 })),
        ("sum(const(100), uniform(0, 100))".into(), Box::new(|p| 100.0 + 100.0 * p)),
        ("clamp(uniform(0, 1000), 100, 800)".into(), Box::new(|p| (1000.0 * p).clamp(100.0, 800.0))),
    ];
    let mut env = Env::new();
    env.insert("x".into(), 1.0);
    for (expr, q) in cases {
        let s = parse_sampler(&expr, &env, &Dir(dir.clone())).unwrap();
        let mut rng = SplitMix64::new(7);
        let mut xs: Vec<f64> = (0..200_000).map(|_| s.sample(&mut rng)).collect();
        for p in [0.1, 0.9] {
            let (got, want) = (quantile(&mut xs, p), q(p));
            assert!((got - want).abs() <= 0.02 * want.abs() + 1e-9, "{expr} p{p}: got {got}, want {want}");
        }
        // The median of a mix sits on a step; skip it there.
        if !expr.starts_with("mix") && !expr.starts_with("clamp") {
            let (got, want) = (quantile(&mut xs, 0.5), q(0.5));
            assert!((got - want).abs() <= 0.02 * want.abs() + 1e-9, "{expr} p50: got {got}, want {want}");
        }
    }
}

fn items(ts: &[f64]) -> Vec<Item> {
    ts.iter().enumerate().map(|(i, &t)| Item { t, subnet: (i % 2) as u8, view: 0, id: i as u32, valid: true }).collect()
}

#[test]
fn transforms() {
    let env = Env::new();
    let dir = Dir::default();
    let mut rng = SplitMix64::new(1);
    let run = |t: Transform, ts: &[f64], rng: &mut SplitMix64| {
        let mut v = items(ts);
        t.compile(&env, &dir).unwrap().apply(&mut v, rng);
        v
    };
    let t = |v: &[Item]| v.iter().map(|i| i.t).collect::<Vec<_>>();
    assert_eq!(t(&run(Transform::Shift(5.0), &[1.0, 2.0], &mut rng)), vec![6.0, 7.0]);
    assert_eq!(t(&run(Transform::Stretch(10.0, 2.0), &[10.0, 20.0], &mut rng)), vec![10.0, 30.0]);
    assert_eq!(t(&run(Transform::Hops("const(3)".into(), 5.0), &[0.0, 1.0], &mut rng)), vec![15.0, 16.0]);
    assert_eq!(t(&run(Transform::Delay("const(4)".into()), &[0.0], &mut rng)), vec![4.0]);
    // cap: never reorders, never exceeds its rate (100 per second = 10 ms apart).
    let ts: Vec<f64> = (0..200).map(|i| (i / 10) as f64 * 3.0).collect();
    let out = run(Transform::Cap(100.0), &ts, &mut rng);
    let ids: Vec<u32> = out.iter().map(|i| i.id).collect();
    assert_eq!(ids, (0..200).collect::<Vec<u32>>());
    assert!(out.windows(2).all(|w| w[1].t - w[0].t >= 10.0 - 1e-9));
    assert!(out.iter().zip(&ts).all(|(o, t)| o.t >= *t));
    // A burst that is already slower than the cap is untouched.
    assert_eq!(t(&run(Transform::Cap(100.0), &[0.0, 50.0, 100.0], &mut rng)), vec![0.0, 50.0, 100.0]);
    // Per subnet: two queues, so two items in the same millisecond on different subnets both pass.
    let out = run(Transform::CapPerSubnet(100.0), &[0.0, 0.0, 1.0, 1.0], &mut rng);
    assert_eq!(t(&out), vec![0.0, 0.0, 10.0, 10.0]);
}

fn gen_bytes(p: &Params) -> Vec<Vec<u8>> {
    let dir = tmp("neutral");
    generate(build(p).unwrap().as_ref(), p, 4, 9).write(&dir).unwrap();
    ["pool.csv", "slots.csv"].iter().map(|f| std::fs::read(dir.join(f)).unwrap_or_default()).collect()
}

#[test]
fn xmss_neutral() {
    let plain = leaf_params("");
    let neutral = leaf_params(
        "xmss.sig_bytes 96\nxmss.base_bytes 96\nxmss.link_mbps 100\nxmss.hops uniform(1, 3)\nxmss.rx_mbps 0\nxmss.dup_factor 1\n\
         xmss.relay_verify_us 0\nxmss.relay_cores 1\nxmss.agg_verify_us 0\nxmss.agg_cores 1\nxmss.sign_delay const(0)\n\
         xmss.proof_bytes 1000\nxmss.agg_base_bytes 1000\nxmss.agg_link_mbps 100\nxmss.agg_hops const(2)\n",
    );
    assert_eq!(gen_bytes(&plain), gen_bytes(&neutral));
    let active = leaf_params("xmss.sig_bytes 3096\nxmss.base_bytes 96\nxmss.link_mbps 100\nxmss.hops uniform(1, 3)\n");
    assert_ne!(gen_bytes(&plain), gen_bytes(&active));
}

#[test]
fn xmss_expansion() {
    let env = Env::new();
    let knobs: Vec<(&str, &str)> = vec![
        ("sig_bytes", "3096"),
        ("base_bytes", "96"),
        ("link_mbps", "100"),
        ("hops", "uniform(1, 3)"),
        ("rx_mbps", "50"),
        ("dup_factor", "2"),
        ("relay_verify_us", "800"),
        ("relay_cores", "2"),
        ("agg_verify_us", "100"),
        ("agg_cores", "4"),
        ("sign_delay", "exp(5)"),
        ("proof_bytes", "300000"),
        ("agg_base_bytes", "1000"),
        ("agg_link_mbps", "1000"),
        ("agg_hops", "const(2)"),
        ("agg_relay_verify_us", "7000"),
        ("agg_rx_mbps", "500"),
        ("agg_dup_factor", "3"),
    ];
    let get = |k: &str| knobs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string());
    let x = xmss::expand(&get, &env).unwrap();
    let hop = (3096.0 - 96.0) * 8.0 / (100.0 * 1000.0);
    assert_eq!(
        x.votes,
        vec![
            Transform::Delay("exp(5)".into()),
            Transform::Hops("uniform(1, 3)".into(), hop),
            Transform::Hops("uniform(1, 3)".into(), 0.8),
            Transform::Cap(2.0 * 1e6 / 800.0),
            Transform::Cap(50.0 * 125_000.0 / (3096.0 * 2.0)),
            Transform::Cap(4.0 * 1e6 / 100.0),
            Transform::Shift(0.1),
        ]
    );
    let hop = (300000.0 - 1000.0) * 8.0 / (1000.0 * 1000.0);
    assert_eq!(
        x.aggregates,
        vec![
            Transform::Hops("const(2)".into(), hop),
            Transform::Hops("const(2)".into(), 7.0),
            Transform::Cap(500.0 * 125_000.0 / (300000.0 * 3.0)),
        ]
    );
}
