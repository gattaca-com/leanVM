use arrival_model::model::{Params, build, generate};
use arrival_model::schedule::{Kind, resample_votes};

/// A recorded leaf set resampled to more votes keeps its spread: the same quantiles, N votes, each id once.
#[test]
fn resample_keeps_the_arrival_shape() {
    let p = Params::parse("shape leaf\nsubnets 2\nsubnet_size 400\nagg.per_subnet 1\nvotes.base lognormal(300, 0.5)\n", std::path::Path::new("."));
    let set = generate(build(&p).unwrap().as_ref(), &p, 4, 7);
    let big = resample_votes(&set, 460, 1).unwrap();
    assert_eq!((big.meta.subnet_size, big.meta.expected_inputs), (460, 460));
    let again = resample_votes(&set, 460, 1).unwrap();
    assert_eq!(big.slots, again.slots);
    for (a, b) in set.slots.iter().zip(&big.slots) {
        for subnet in 0..2u8 {
            let times = |s: &[arrival_model::schedule::Arrival]| -> Vec<u64> {
                let mut t: Vec<u64> = s.iter().filter(|x| x.kind == Kind::Vote && x.subnet == subnet).map(|x| x.t_us).collect();
                t.sort_unstable();
                t
            };
            let (ta, tb) = (times(&a.arrivals), times(&b.arrivals));
            assert_eq!(tb.len(), 460);
            for k in [10usize, 50, 90] {
                let (qa, qb) = (ta[ta.len() * k / 100] as f64, tb[tb.len() * k / 100] as f64);
                assert!((qa - qb).abs() <= 0.02 * qa, "p{k}: {qa} against {qb}");
            }
            let mut ids: Vec<u32> = b.arrivals.iter().filter(|x| x.subnet == subnet).map(|x| x.id).collect();
            ids.sort_unstable();
            assert_eq!(ids, (0..460).collect::<Vec<u32>>());
        }
    }
    // Only votes can be resampled.
    let mut with_pool = set;
    with_pool.pool.push(arrival_model::schedule::PoolEntry { subnet: 0, id: 0, signers: vec![0], rate: 2, slot: None });
    assert!(resample_votes(&with_pool, 10, 1).is_err());
}
