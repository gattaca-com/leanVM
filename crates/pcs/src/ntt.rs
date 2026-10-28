// CREDIT: https://github.com/succinctlabs/flock (flock-core), MIT OR Apache-2.0.
//! The additive NTT (Lin-Chung-Han novel polynomial basis) WHIR commits with.

mod additive_ntt_f64;
mod cosets;
mod deep;
mod drain;
mod encode;
mod facade;
mod gathered;
mod kernels;
mod layers;
mod plan;
mod rows;
mod scratch;
mod source;
mod stage;
mod twiddles;
mod window;
pub use encode::{Encoded, Keep, RowSink};
pub use facade::AdditiveNttF64;
pub(crate) use kernels::transpose_butterfly_lanes;
pub use plan::Plan;
