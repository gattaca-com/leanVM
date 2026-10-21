// CREDIT: https://github.com/succinctlabs/flock, MIT OR Apache-2.0.
//! flock: a batched R1CS proving system for hash circuits over GF(2), reduced
//! to evaluation claims on the committed packed witness.
//!
//! Protocol flow (all challenges from the shared [`fiat_shamir`] transcript):
//!   1. The caller commits the packed Boolean witness `q_flock` (inside the VM's
//!      one stacked [`pcs`] commitment).
//!   2. [`zerocheck`] reduces `a·b ⊕ c = 0` over the cube to evaluation claims
//!      on `(â, b̂, ĉ)`.
//!   3. [`lincheck`] reduces those to the `2^k_skip` bit-slice values of `z` at
//!      one point, against the per-block matrices.
//!   4. The PCS binds that family of slices ([`haraka::SliceClaim`]) to the
//!      commitment.
//!
//! [`haraka`] is the one circuit: Haraka-512 as a per-call R1CS with GF(2^8)
//! coefficients, plus its witness generation and the leanVM-facing reduction
//! entry points (`HarakaSetup::{prove_zerocheck, prove_lincheck}`). Steps 2 to 4
//! above are circuit-agnostic: they take the block shape as plain numbers.

pub mod haraka;
pub mod lincheck;
pub mod zerocheck;

pub use haraka::SliceClaim;
