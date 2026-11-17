//! Regression tests for the merge findings of `design/REVIEW.md`.
mod common;
use arrival_model::model::Params;
use common::*;
use sidecar_streaming::metric::mean;
use std::path::Path;

fn invalid_params(invalid: &str) -> Params {
    let text = format!("shape merge\nslot_ms 4000\nsubnets 16\nagg.per_subnet 3\nagg.base uniform(1500, 2400)\nagg.invalid {invalid}\n");
    Params::parse(&text, Path::new("."))
}

/// Finding 8: under inline verification an invalid candidate must not cost its whole subnet.
#[test]
fn inline_verification_survives_invalid_candidates() {
    let set = make_set(&invalid_params("0.2"), 40, 12);
    let mut backend = sim(SMOOTH);
    for s in ["two_round:verify=inline", "best_per_subnet_late:verify=inline", "fill_16:verify=inline"] {
        let cov: Vec<f64> = (0..40).map(|slot| run(&mut backend, &set, slot, 4_000_000, s, None).1.coverage).collect();
        // With 3 candidates per subnet and 20 % invalid, 99 % of subnets keep a valid one.
        assert!(mean(&cov) > 0.93, "{s}: mean coverage {}", mean(&cov));
    }
}
