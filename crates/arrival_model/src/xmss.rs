//! The leanXMSS adjustment layer: physical knobs that expand into generic transforms.
//! Every knob defaults to "no effect", so a model without `xmss.*` keys is unchanged.

use crate::layers::Transform;
use crate::sampler::{Env, eval_num};

pub struct XmssLayer {
    pub votes: Vec<Transform>,
    pub aggregates: Vec<Transform>,
}

fn is_neutral_sampler(s: &str) -> bool {
    let s: String = s.split_whitespace().collect();
    s.is_empty() || s == "none" || s == "const(0)"
}

/// Expand the `xmss.*` keys, read through `get` (key without the `xmss.` prefix), into transforms.
pub fn expand(get: &dyn Fn(&str) -> Option<String>, env: &Env) -> Result<XmssLayer, String> {
    let num = |k: &str, default: f64| -> Result<f64, String> {
        match get(k) {
            Some(v) => eval_num(&v, env).map_err(|e| format!("xmss.{k}: {e}")),
            None => Ok(default),
        }
    };
    let sampler = |k: &str| get(k).filter(|s| !is_neutral_sampler(s));
    let mut votes = Vec::new();
    let mut aggs = Vec::new();

    // Votes: sign delay, hop delays, relay cap, receiver bandwidth cap, aggregator verify cap.
    let sig_bytes = num("sig_bytes", 0.0)?;
    let hops = sampler("hops");
    if let Some(s) = sampler("sign_delay") {
        votes.push(Transform::Delay(s));
    }
    let per_hop = (sig_bytes - num("base_bytes", 0.0)?) * 8.0 / (num("link_mbps", 0.0)? * 1000.0);
    if let (Some(h), true) = (&hops, per_hop.is_finite() && per_hop > 0.0) {
        votes.push(Transform::Hops(h.clone(), per_hop));
    }
    let relay_us = num("relay_verify_us", 0.0)?;
    if relay_us > 0.0 {
        if let Some(h) = &hops {
            votes.push(Transform::Hops(h.clone(), relay_us / 1000.0));
        }
        votes.push(Transform::Cap(num("relay_cores", 1.0)? * 1e6 / relay_us));
    }
    let rx = num("rx_mbps", 0.0)?;
    if rx > 0.0 && sig_bytes > 0.0 {
        votes.push(Transform::Cap(rx * 125_000.0 / (sig_bytes * num("dup_factor", 1.0)?)));
    }
    let agg_us = num("agg_verify_us", 0.0)?;
    if agg_us > 0.0 {
        votes.push(Transform::Cap(num("agg_cores", 1.0)? * 1e6 / agg_us));
        votes.push(Transform::Shift(agg_us / 1000.0));
    }

    // Aggregates.
    let proof_bytes = num("proof_bytes", 0.0)?;
    let agg_hops = sampler("agg_hops");
    let per_hop = (proof_bytes - num("agg_base_bytes", 0.0)?) * 8.0 / (num("agg_link_mbps", 0.0)? * 1000.0);
    if let (Some(h), true) = (&agg_hops, per_hop.is_finite() && per_hop > 0.0) {
        aggs.push(Transform::Hops(h.clone(), per_hop));
    }
    let relay_us = num("agg_relay_verify_us", 0.0)?;
    if relay_us > 0.0
        && let Some(h) = &agg_hops
    {
        aggs.push(Transform::Hops(h.clone(), relay_us / 1000.0));
    }
    let rx = num("agg_rx_mbps", 0.0)?;
    if rx > 0.0 && proof_bytes > 0.0 {
        aggs.push(Transform::Cap(rx * 125_000.0 / (proof_bytes * num("agg_dup_factor", 1.0)?)));
    }
    Ok(XmssLayer { votes, aggregates: aggs })
}
