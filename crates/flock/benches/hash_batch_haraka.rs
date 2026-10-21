//! Standalone batch Haraka-512 proving over the GF(2^8)-coefficient circuit, isolated from the VM.
//!
//! ```text
//! BENCH_REPEAT=3 BENCH_COOLDOWN=2 FLOCK_N_LOG=18 cargo bench -p flock --bench hash_batch_haraka
//! ```
//!
//! `FLOCK_N_LOG` is the log of the number of calls; the slot count rounds up to a power of two.
//! `BENCH_TRACING=1` prints the final pass's span tree, and `-- --json` the proving time as Bencher Metric Format.

use std::time::Instant;

use bench::{Metric, Plan, Timing, bencher_json};
use fiat_shamir::digest_words;
use fiat_shamir::transcript::{Proof, ProverState};
use flock::haraka::{CALLS_PER_SLOT, HarakaSetup, Input, ROWS_LOG, ring_switch_open};
use pcs::pack::LOG_PACKING;
use pcs::stack_open::open_batch_mixed_whir_stacked;
use pcs::whir::{INITIAL_FOLDING_FACTOR, LOG_INV_RATE_0, commit, config_for_rate};
use primitives::{field::F64, pretty_integer, test_rng::Rng};

#[global_allocator]
static ALLOCATOR: bench::Jemalloc = bench::Jemalloc;

fn main() {
    bench::init_tracing_from_env();
    let requested_n_log: usize = std::env::var("FLOCK_N_LOG")
        .ok()
        .map(|s| s.parse().expect("FLOCK_N_LOG must be an integer"))
        .unwrap_or(13);
    let n = 1usize << requested_n_log;
    let setup = HarakaSetup::new(n);
    let n_log = setup.n_log();
    let n = n.min((1 << n_log) * CALLS_PER_SLOT);
    let mu = setup.m_bits() - LOG_PACKING;
    assert!(
        mu >= 15,
        "FLOCK_N_LOG too small: need a committed witness with mu >= 15"
    );
    let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15 ^ n as u64);
    let inputs: Vec<Input> = (0..n)
        .map(|_| std::array::from_fn(|_| (rng.next_u64() & 0xff) as u8))
        .collect();
    let config = config_for_rate(mu, LOG_INV_RATE_0).expect("WHIR configuration");
    let prove_pass = || {
        let _span = tracing::info_span!("Flock Haraka prove", n_log).entered();
        let t_pass = Instant::now();
        let t = Instant::now();
        let w = setup.generate_witness(&inputs);
        let witness_s = t.elapsed().as_secs_f64();
        // SAFETY: `F64` is `repr(transparent)` over `u64`.
        let q_flock: &[F64] = unsafe { std::slice::from_raw_parts(w.z.as_ptr().cast(), w.z.len()) };
        assert_eq!(q_flock.len(), 1 << mu);
        let mut ps = ProverState::new(digest_words(&SEED[0]), digest_words(&SEED[1]));
        let t_prove = Instant::now();
        let t = Instant::now();
        let (commitment, prover_data) = commit(q_flock, mu, INITIAL_FOLDING_FACTOR, LOG_INV_RATE_0);
        ps.add_root(&commitment.root);
        let commit_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let stage = setup.prove_zerocheck(&w, &mut ps);
        let zerocheck_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let reduced = setup.prove_lincheck(stage, &w.z_lincheck, &mut ps);
        let lincheck_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let ring = ring_switch_open(n, 0, &reduced);
        open_batch_mixed_whir_stacked(&mut ps, mu, q_flock, &prover_data, &config, &[], &ring);
        let open_s = t.elapsed().as_secs_f64();
        let prove_s = t_prove.elapsed().as_secs_f64();
        let proof = ps.into_proof();
        let pass_s = t_pass.elapsed().as_secs_f64();
        (
            proof,
            [witness_s, commit_s, zerocheck_s, lincheck_s, open_s, prove_s, pass_s],
        )
    };
    let plan = Plan::from_env();
    let mut stages: [Timing; 7] = std::array::from_fn(|_| Timing::default());
    let (proof, _) = plan.warm_then_measure(|final_pass| {
        let _quiet = (!final_pass).then(bench::suppress_tracing);
        let (out, secs) = prove_pass();
        for (timing, s) in stages.iter_mut().zip(secs) {
            timing.push(s);
        }
        out
    });
    let [witness, commit_stage, zerocheck, lincheck, open, prove, pass] = stages.map(|t| {
        let mut kept = Timing::default();
        for &s in &t.samples()[1..] {
            kept.push(s);
        }
        kept
    });
    let proof = scalar_proof(&proof);
    let (_, verify_time) = Plan::new(plan.repeat, 0).measure_quiet(|_final_pass| {
        verifier::verify_flock_batch_haraka(&SEED[0], &SEED[1], ROWS_LOG + n_log, LOG_INV_RATE_0, &proof)
            .expect("Flock Haraka batch proof verifies");
    });
    if std::env::args().any(|arg| arg == "--json") {
        let report = [(
            format!("flock-haraka-batch-{n}"),
            vec![("latency", Metric::nanoseconds(&prove))],
        )];
        println!("{}", bencher_json(&report));
        return;
    }
    let pass_s = pass.mean();
    let share = |s: f64| format!("{:>5.1}%", 100.0 * s / pass_s);
    let ms = |t: &Timing| format!("{:>8.1} ms{:<9}{}", t.mean() * 1e3, t.spread(), share(t.mean()));
    let named = witness.mean() + commit_stage.mean() + zerocheck.mean() + lincheck.mean() + open.mean();
    println!(
        "\nFlock Haraka-512 batch proving, {} calls (2^{n_log} blocks)",
        pretty_integer(n)
    );
    println!("  witness-gen                     : {}", ms(&witness));
    println!("  commit                          : {}", ms(&commit_stage));
    println!("  zerocheck                       : {}", ms(&zerocheck));
    println!("  lincheck                        : {}", ms(&lincheck));
    println!("  pcs opening                     : {}", ms(&open));
    println!(
        "  other                           : {:>8.1} ms{:<9}{}",
        (pass_s - named) * 1e3,
        "",
        share(pass_s - named)
    );
    println!("  ------------------------------------------");
    println!("  prove TOTAL (witness excluded)  : {}", ms(&prove));
    println!(
        "  verify (scalar, one thread)     : {:>8.1} ms",
        verify_time.mean() * 1e3
    );
    let prove_s = prove.mean();
    println!(
        "  throughput                      : {:>14} calls/s{}",
        pretty_integer((n as f64 / prove_s).round() as u64),
        prove.spread()
    );
    println!(
        "  (~{:.1} leanXMSS/s equivalent at 201 calls/signature)",
        n as f64 / prove_s / 201.0
    );
}

const SEED: [[u8; 32]; 2] = [[1; 32], [2; 32]];

fn scalar_proof(proof: &Proof) -> verifier::Proof {
    let batches = proof.merkle.iter().map(|paths| verifier::PrunedMerklePaths {
        leaf_data: paths
            .leaf_data
            .iter()
            .map(|row| row.iter().map(|word| verifier::K(word.0)).collect())
            .collect(),
        sibling_hashes: paths.sibling_hashes.clone(),
    });
    verifier::Proof {
        stream: proof
            .stream
            .iter()
            .map(|x| verifier::E::new(x.c0, x.c1, x.c2))
            .collect(),
        merkle_openings: verifier::MerkleOpenings::Pruned(batches.collect()),
    }
}
