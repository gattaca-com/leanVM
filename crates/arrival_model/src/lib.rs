//! Arrival models: how signatures and aggregates reach one node over a slot.
//!
//! The model's only input is a seed and a parameter file. Its only output is a schedule set
//! ([`schedule`]). Nothing here depends on the prover or on the streaming harness.

pub mod compose;
pub mod layers;
pub mod model;
pub mod rng;
pub mod sampler;
pub mod schedule;
pub mod xmss;
