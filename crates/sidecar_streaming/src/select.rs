//! Child selection for `Merge` (STRATEGIES.md 3.2): the most claims, beyond a base, from at most `budget`
//! candidates. Subnets have disjoint claims, so the problem splits by subnet and the budget is a knapsack.

use crate::types::*;

/// Largest number of candidates of one subnet that is enumerated exhaustively. Above it, only the
/// ones with the largest own gain are enumerated.
const MAX_PER_SUBNET: usize = 12;

/// `(candidate, subnet, coverage)` in, the chosen candidates and the claims they add to `base` out.
pub fn select_exact(cands: &[(ProofId, u8, &Coverage)], base: &Coverage, budget: usize) -> (Vec<ProofId>, u32) {
    let base_len = base.len();
    let mut subnets: Vec<u8> = cands.iter().map(|c| c.1).collect();
    subnets.sort_unstable();
    subnets.dedup();
    // Per subnet: for each k, the best gain and the subset that makes it.
    let mut options: Vec<Vec<(u32, Vec<ProofId>)>> = Vec::new();
    for s in subnets {
        let mut cs: Vec<&(ProofId, u8, &Coverage)> = cands.iter().filter(|c| c.1 == s).collect();
        if cs.len() > MAX_PER_SUBNET {
            cs.sort_by_key(|c| std::cmp::Reverse(c.2.union_len(base)));
            cs.truncate(MAX_PER_SUBNET);
        }
        let kmax = cs.len().min(budget);
        let mut best: Vec<(u32, Vec<ProofId>)> = vec![(0, Vec::new()); kmax + 1];
        for mask in 1u32..(1 << cs.len()) {
            let k = mask.count_ones() as usize;
            if k > kmax {
                continue;
            }
            let mut u = base.clone();
            for (i, c) in cs.iter().enumerate() {
                if mask >> i & 1 == 1 {
                    u.union_with(c.2);
                }
            }
            let gain = u.len() - base_len;
            if gain > best[k].0 || best[k].1.is_empty() {
                best[k] = (gain, cs.iter().enumerate().filter(|(i, _)| mask >> i & 1 == 1).map(|(_, c)| c.0).collect());
            }
        }
        options.push(best);
    }
    // Knapsack over subnets. State: the best (gain, fewest children) with at most `b` children.
    let n = options.len();
    let mut dp = vec![vec![(0u32, 0usize); budget + 1]; n + 1];
    let mut take = vec![vec![0usize; budget + 1]; n];
    for (i, opt) in options.iter().enumerate() {
        for b in 0..=budget {
            let mut best = (dp[i][b].0, std::cmp::Reverse(dp[i][b].1));
            let mut pick = 0;
            for (k, (gain, _)) in opt.iter().enumerate().skip(1) {
                if k > b {
                    break;
                }
                let cand = (dp[i][b - k].0 + gain, std::cmp::Reverse(dp[i][b - k].1 + k));
                if cand > best {
                    best = cand;
                    pick = k;
                }
            }
            dp[i + 1][b] = (best.0, best.1.0);
            take[i][b] = pick;
        }
    }
    let mut chosen = Vec::new();
    let mut b = budget;
    for i in (0..n).rev() {
        let k = take[i][b];
        chosen.extend(options[i][k].1.iter().copied());
        b -= k;
    }
    chosen.reverse();
    (chosen, dp[n][budget].0)
}
