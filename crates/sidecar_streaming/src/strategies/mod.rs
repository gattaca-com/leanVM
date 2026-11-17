//! The strategy registry. A strategy is named `name` or `name:key=value;key=value`.

pub mod leaf;
pub mod merge;

use crate::forecast::Forecast;
use crate::strategy::Strategy;
use crate::types::Shape;
use merge::Policy;
use std::collections::BTreeMap;
use std::rc::Rc;

/// A job's predicted duration plus the room a strategy keeps for the variation of a real prover: 1 % and 1 ms.
pub fn with_margin(us: u64) -> u64 {
    us + us / 100 + 1_000
}

#[derive(Clone, Debug, Default)]
pub struct Params(pub BTreeMap<String, String>);

impl Params {
    pub fn f(&self, key: &str, default: f64) -> f64 {
        self.0.get(key).and_then(|v| v.parse().ok()).unwrap_or(default)
    }
}

#[derive(Clone, Debug)]
pub struct Spec {
    pub name: String,
    pub params: Params,
}

pub const LEAF: &[&str] =
    &["one_shot_late", "two_stage", "min_batch_fit", "fixed_fit", "greedy_fit", "cheap_chain", "step_cut_fit", "plan_cover"];
pub const MERGE: &[&str] = &["best_per_subnet_late", "fill_16", "two_round", "union_chain", "plan_merge"];

/// The strategies registered for a shape.
pub fn registry(shape: Shape) -> &'static [&'static str] {
    match shape {
        Shape::Leaf => LEAF,
        Shape::Merge => MERGE,
    }
}

pub fn shape_of(name: &str) -> Option<Shape> {
    if LEAF.contains(&name) {
        Some(Shape::Leaf)
    } else if MERGE.contains(&name) {
        Some(Shape::Merge)
    } else {
        None
    }
}

impl Spec {
    pub fn parse(text: &str) -> Result<Spec, String> {
        let (name, rest) = text.split_once(':').unwrap_or((text, ""));
        shape_of(name).ok_or(format!("unknown strategy `{name}`"))?;
        let mut params = BTreeMap::new();
        for kv in rest.split([';', ',']).filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').ok_or(format!("bad strategy parameter `{kv}`"))?;
            params.insert(k.trim().to_string(), v.trim().to_string());
        }
        Ok(Spec { name: name.to_string(), params: Params(params) })
    }

    pub fn shape(&self) -> Shape {
        shape_of(&self.name).unwrap()
    }

    /// Set `key` unless the spec already has it.
    pub fn default_param(&mut self, key: &str, value: &str) {
        self.params.0.entry(key.to_string()).or_insert_with(|| value.to_string());
    }

    /// A fresh strategy for one slot.
    pub fn make(&self, forecast: Option<&Rc<Forecast>>) -> Result<Box<dyn Strategy>, String> {
        let p = &self.params;
        let need = || forecast.cloned().ok_or(format!("{} needs a forecast: pass --train DIR", self.name));
        let policy = Policy::parse(p.0.get("verify").map_or("all_idle", String::as_str))?;
        Ok(match self.name.as_str() {
            "one_shot_late" => Box::new(leaf::LeafStrategy::new(leaf::OneShotLate)),
            "two_stage" => Box::new(leaf::LeafStrategy::new(leaf::TwoStage::new(p))),
            "min_batch_fit" => Box::new(leaf::LeafStrategy::new(leaf::MinBatch::new(p))),
            "fixed_fit" => Box::new(leaf::LeafStrategy::new(leaf::FixedFit::new(p))),
            "greedy_fit" => Box::new(leaf::LeafStrategy::new(leaf::Greedy)),
            "cheap_chain" => Box::new(leaf::LeafStrategy::new(leaf::CheapChain::new(p))),
            "step_cut_fit" => Box::new(leaf::LeafStrategy::new(leaf::StepCutFit::new(p))),
            "plan_cover" => Box::new(leaf::LeafStrategy::new(leaf::PlanCover::new(p, need()?))),
            "best_per_subnet_late" => Box::new(merge::MergeStrategy::new(merge::BestPerSubnetLate, policy)),
            "fill_16" => Box::new(merge::MergeStrategy::new(merge::Fill16, policy)),
            "two_round" => Box::new(merge::MergeStrategy::new(merge::TwoRound::new(p), policy)),
            "union_chain" => Box::new(merge::MergeStrategy::new(merge::UnionChain::new(p), policy)),
            "plan_merge" => Box::new(merge::MergeStrategy::new(merge::PlanMerge::new(need()?), policy)),
            _ => unreachable!(),
        })
    }
}
