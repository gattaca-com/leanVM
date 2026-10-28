// CREDIT: https://github.com/succinctlabs/flock (flock-core), MIT OR Apache-2.0.
//! The additive NTT (Lin-Chung-Han novel polynomial basis) WHIR commits with.

pub mod additive_ntt_f64;
mod deep;
mod gathered;
mod kernels;
mod rows;
mod stage;
mod twiddles;
pub use additive_ntt_f64::{AdditiveNttF64, Encoded, Keep, RowSink};
pub(crate) use kernels::transpose_butterfly_lanes;
pub use rows::DeepRows;
