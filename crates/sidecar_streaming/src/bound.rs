//! Upper bounds on coverage (STRATEGIES.md 4): what no strategy can beat, for one slot, one `D`, one cost model.

use crate::backend::THREADS;
use crate::cost::CostModel;
use crate::select::select_exact;
use crate::types::*;

fn cost(m: &dyn CostModel, shape: &JobShape) -> f64 {
    m.prove_us(shape, THREADS) as f64
}

/// Clairvoyant chain: knowing the whole schedule, the most votes any one-lane chain can have in a
/// final-rate proof by `d`. `arr` is the sorted arrival times of one run. Prefixes in arrival order lose
/// nothing, since cost depends on counts only. Only every `grid`-th prefix is a node, plus the last one that arrived.
/// **Only `grid` 1 is an upper bound.** A larger grid restricts the chain, so its value is a lower bound on the optimum.
///
/// `f(j, r)` is the earliest time a rate-`r` proof of the first `j` arrivals exists:
/// `min(a_j + agg(j, r), min_i,rk max(f(i, rk), a_j) + ext(i, j - i, rk, r), f(j, rk) + conv(j, rk, r))`.
pub fn clairvoyant_leaf(arr: &[Micros], model: &dyn CostModel, d: Micros, final_rate: Rate, cheap: Rate, grid: usize) -> u32 {
    let m = arr.partition_point(|&t| t <= d);
    if m == 0 {
        return 0;
    }
    let rates: Vec<Rate> = if cheap < final_rate { vec![cheap, final_rate] } else { vec![final_rate] };
    let fi = rates.len() - 1;
    let mut idx: Vec<usize> = (grid.max(1)..m).step_by(grid.max(1)).collect();
    idx.push(m);
    idx.dedup();
    let pos: Vec<usize> = std::iter::once(0).chain(idx).collect();
    let a = |j: usize| arr[j - 1] as f64;
    // f[node][rate]; node 0 is the empty prefix.
    let mut f = vec![vec![f64::INFINITY; rates.len()]; pos.len()];
    let mut best = 0;
    for j in 1..pos.len() {
        for (ri, &r) in rates.iter().enumerate() {
            let mut t = a(pos[j]) + cost(model, &JobShape::aggregate(pos[j] as u32, r));
            for i in 1..j {
                for (rk_i, &rk) in rates.iter().enumerate() {
                    if f[i][rk_i].is_finite() {
                        let shape = JobShape::extend(pos[i] as u32, rk, (pos[j] - pos[i]) as u32, r);
                        t = t.min(f[i][rk_i].max(a(pos[j])) + cost(model, &shape));
                    }
                }
            }
            f[j][ri] = t;
        }
        for ri in 0..fi {
            let t = f[j][ri] + cost(model, &JobShape::extend(pos[j] as u32, rates[ri], 0, final_rate));
            f[j][fi] = f[j][fi].min(t);
        }
        if f[j][fi] <= d as f64 {
            best = pos[j];
        }
    }
    best as u32
}

/// Universal bound, any strategy. A vote needs a final-rate job that starts after it and ends by `d`
/// (last-job bound); and the votes arriving after `t` need their own lane time within `d - t` (work bound).
pub fn universal_leaf(arr: &[Micros], model: &dyn CostModel, d: Micros, final_rate: Rate, cheap: Rate) -> u32 {
    let m = arr.partition_point(|&t| t <= d);
    let rates: Vec<Rate> = if cheap < final_rate { vec![cheap, final_rate] } else { vec![final_rate] };
    // The cheapest final-rate job with one new input: a one-signature AGGREGATE or an EXTEND of a one-claim proof.
    let c_min = rates
        .iter()
        .map(|&rk| cost(model, &JobShape::extend(1, rk, 1, final_rate)))
        .chain([cost(model, &JobShape::aggregate(1, final_rate))])
        .fold(f64::INFINITY, f64::min);
    let mut best = arr.partition_point(|&t| (t as f64) <= d as f64 - c_min);
    // Work per signature: no job proves signatures cheaper than the best ratio, so `x` of lane time proves at most `x / e` of them.
    let e = (1..=2048u32)
        .flat_map(|m| rates.iter().map(move |&r| (m, r)))
        .map(|(m, r)| cost(model, &JobShape::aggregate(m, r)) / m as f64)
        .fold(f64::INFINITY, f64::min);
    let n_of = |x: f64| (x.max(0.0) / e) as usize;
    for ti in (0..m).step_by(25) {
        let budget = d as f64 - arr[ti] as f64;
        best = best.min(ti + n_of(budget) + 25);
    }
    best.min(m) as u32
}

/// Upper bound for `Merge`: the claims in the union of all valid candidates that arrived by `d`.
/// No root can hold a claim that no valid candidate carried in time. `cands` are `(arrival, claims, valid)`.
pub fn merge_upper_bound(cands: &[(Micros, &Coverage, bool)], d: Micros) -> u32 {
    let mut u = Coverage::empty();
    cands.iter().filter(|c| c.2 && c.0 <= d).for_each(|c| u.union_with(c.1));
    u.len()
}

/// A reference value for `Merge`, **not a bound**: the best single round of up to `max_children` children,
/// by the exact selection of 3.2, on the candidates that arrived by `d - cost(n)` for the best `n`.
/// Strategies with several rounds or a kept root can exceed it.
/// `cands` are `(arrival, subnet, claims, valid)`.
pub fn merge_one_round_reference(cands: &[(Micros, u8, &Coverage, bool)], model: &dyn CostModel, d: Micros, final_rate: Rate, max_children: usize) -> u32 {
    let job = |n: usize, claims: u32| {
        cost(model, &JobShape { children: vec![ChildShape { claims, rate: 2, verify: true, leaves: 1 }; n], new_sigs: 0, rate: final_rate })
    };
    let valid: Vec<&(Micros, u8, &Coverage, bool)> = cands.iter().filter(|c| c.3).collect();
    let empty = Coverage::empty();
    let pick = |cut: f64, budget: usize| {
        let items: Vec<(ProofId, u8, &Coverage)> =
            valid.iter().enumerate().filter(|(_, c)| c.0 as f64 <= cut).map(|(i, c)| (ProofId(i as u32), c.1, c.2)).collect();
        select_exact(&items, &empty, budget).1
    };
    (1..=max_children).map(|n| pick(d as f64 - job(n, 1500), n)).max().unwrap_or(0)
}
