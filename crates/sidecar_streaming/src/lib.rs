//! A harness for strategies that aggregate inputs arriving over time, with `leanvm-sidecar` as a black box.
//!
//! A strategy ([`strategy`]) answers events with calls on a `Ctx`. A driver ([`driver`]) feeds it a
//! schedule and a backend: [`sim`] (a virtual clock and a cost model) or [`live`] (a running sidecar).
//! The metric ([`metric`]) is coverage at the deadline.

pub mod backend;
pub mod bound;
pub mod cli;
pub mod cost;
pub mod driver;
pub mod e2e;
pub mod fit;
pub mod forecast;
pub mod live;
pub mod report;
pub mod runner;
pub mod metric;
pub mod rng;
pub mod select;
pub mod sim;
pub mod strategies;
pub mod strategy;
pub mod types;
pub mod wire;
