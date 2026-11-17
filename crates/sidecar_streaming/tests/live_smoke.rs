//! Needs a running sidecar: `LEANVM_CHANNEL=/dev/shm/... LEANVM_MODEL=cost/model-<backend>-avx2-8c.txt cargo test --release -p sidecar_streaming --test live_smoke -- --ignored`.
mod common;
use common::*;
use sidecar_streaming::live::LiveBackend;
use sidecar_streaming::report::compare_jobs;
use sidecar_streaming::runner::job_line;
use std::path::Path;

#[test]
#[ignore]
fn live_smoke() {
    let channel = std::env::var("LEANVM_CHANNEL").expect("LEANVM_CHANNEL");
    let model_path = std::env::var("LEANVM_MODEL").expect("LEANVM_MODEL");
    let model = sidecar_streaming::cost::load_model(Path::new(&model_path)).unwrap();
    let sigs = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bins/leanvm-sidecar/testdata/signatures-24000.bin");
    let mut backend = LiveBackend::open(Path::new(&channel), &sigs, model, "live", Path::new("target/streaming-cache")).unwrap();
    let set = make_set(&leaf_params(), 2, 5);
    let mut lines = Vec::new();
    for slot in 0..2 {
        let (rec, m) = run(&mut backend, &set, slot, 2_000_000, "greedy_fit", None);
        assert!(m.got > 0, "slot {slot}: nothing published");
        // Every proof the jobs made verifies.
        assert!(backend.verify_made().unwrap() >= 1);
        lines.extend(rec.jobs.iter().map(|j| job_line(slot, j)));
    }
    // Job durations within 30 % of the model, by op and size.
    let table = compare_jobs(&lines);
    println!("{table}");
    for row in table.lines().skip(2) {
        let ratio: f64 = row.trim_matches('|').split('|').next_back().unwrap().trim().parse().unwrap();
        assert!((0.7..=1.3).contains(&ratio), "measured over predicted {ratio}: {row}");
    }
}
