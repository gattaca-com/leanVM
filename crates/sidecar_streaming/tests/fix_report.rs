//! The report shows empties and the mean beside the median, and labels a small-sample p10 (AUDIT.md finding 8).
use sidecar_streaming::metric::*;
use sidecar_streaming::report::{compare, p10_text, report};

fn row(slot: u32, got: u32, arrived: u32) -> Row {
    let cov = if arrived == 0 { f64::NAN } else { got as f64 / arrived as f64 };
    Row {
        set: "s".into(),
        backend: "b".into(),
        strategy: "x".into(),
        deadline_us: 4_000_000,
        slot,
        m: Metrics {
            got,
            arrived,
            sent: arrived,
            coverage: cov,
            coverage_sent: cov,
            published_us: (got > 0).then_some(3_990_000),
            cut_us: None,
            staleness_us: None,
            slack_us: None,
            jobs: 3,
            busy_us: 0,
            wasted_us: 0,
            verify_us: 0,
        },
    }
}

/// Nine full slots and one with nothing published: the median stays 1, the mean and the empty share do not.
#[test]
fn one_empty_slot_in_ten_moves_the_mean_and_the_share() {
    let mut rows: Vec<Row> = (0..9).map(|s| row(s, 1500, 1500)).collect();
    rows.push(row(9, 0, 1500));
    // A slot nobody voted in is not empty, only quiet.
    rows.push(row(10, 0, 0));
    let sm = summarize(&rows.iter().collect::<Vec<_>>());
    assert_eq!((sm.n, sm.empty, sm.unpublished), (10, 1, 1));
    assert_eq!(sm.p50, 1.0);
    assert!((sm.mean - 0.9).abs() < 1e-12);
    assert!((sm.empty_share - 0.1).abs() < 1e-12);
    assert_eq!(sm.mean_published, 1.0);
}

#[test]
fn the_table_has_the_new_columns_and_labels_a_small_p10() {
    let mut rows: Vec<Row> = (0..9).map(|s| row(s, 1500, 1500)).collect();
    rows.push(row(9, 0, 1500));
    let text = report(&rows, None, false);
    assert!(text.contains("| no votes | empty | p50 | p10 | mean | mean non-empty |"), "{text}");
    assert!(text.contains("0.100"), "the empty share: {text}");
    assert!(text.contains("0.000 (min of 10)"), "p10 of ten slots is the minimum: {text}");
    // Thirty slots or more: a plain number.
    let many: Vec<Row> = (0..40).map(|s| row(s, 1500, 1500)).collect();
    assert!(!report(&many, None, false).contains("min of"));
}

#[test]
fn p10_labels() {
    assert_eq!(p10_text(0.5, 10), "0.500 (min of 10)");
    assert_eq!(p10_text(0.5, 3), "0.500 (min of 3)");
    assert_eq!(p10_text(0.5, 24), "0.500 (n=24)");
    assert_eq!(p10_text(0.5, 30), "0.500");
    assert_eq!(p10_text(f64::NAN, 0), "NaN");
}

/// Sim against live: the median passes while three slots of ten are empty. The comparison now says so.
#[test]
fn compare_names_the_empty_slots() {
    let sim: Vec<Row> = (0..10).map(|s| row(s, 1500, 1500)).collect();
    let live: Vec<Row> = (0..10).map(|s| if s < 3 { row(s, 0, 1500) } else { row(s, 1500, 1500) }).collect();
    let text = compare(&sim, &live);
    assert!(text.contains("empty slots live 3, sim 0"), "{text}");
    assert!(text.contains("mean coverage live 0.7000, sim 1.0000"), "{text}");
}
