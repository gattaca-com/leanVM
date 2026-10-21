//! VM-provable hashing.
//!
//! The VM instruction exposes one Haraka-512 call, 64 bytes to 32. A guest hashes any length as the
//! chain [`primitives::hash::hash`] runs: 32-byte blocks behind the chaining value, one call each.

/// Haraka-512 of exactly 64 bytes (two 256-bit halves laid out little-endian), which is also the PCS
/// Merkle parent. Lives in [`fiat_shamir`] (the shared [`fiat_shamir::FiatShamirState`] state is built on it).
pub use fiat_shamir::compress;
