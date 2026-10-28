//! Small AArch64 SIMD helpers shared by the NEON kernels.
//!
//! The ARMv8.2 SHA3 extension, present on every Apple M core and enabled by
//! the workspace's `-C target-cpu=native`, provides `EOR3`, a three-way XOR
//! in one instruction. The binary-field kernels are XOR-dominated, so folding
//! pairs of dependent `EOR`s into single `EOR3`s cuts both instruction count
//! and dependency-chain length in the hot loops. Builds without the extension
//! transparently keep the two-`EOR` form.
//!
//! (`sha3` is the name of the ARMv8.2 feature *block* that carries `EOR3`; the
//! batched Keccak-256 in `hash` uses the same block for its own instructions.)

use core::arch::aarch64::*;

/// Three-way XOR of 64-bit-lane vectors.
///
/// `EOR3` where the build enables `sha3`, two `EOR`s otherwise.
/// The kernel is a `#[target_feature]` fn, so this plain wrapper is the one place that calls it.
#[inline(always)]
pub fn xor3_u64(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
    // SAFETY: the kernel needs `neon`, which aarch64 has, and `sha3`, which the cfg gate on its arm enables.
    unsafe { xor3_kernel(a, b, c) }
}

#[cfg(target_feature = "sha3")]
#[target_feature(enable = "neon,sha3")]
#[inline]
fn xor3_kernel(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
    veor3q_u64(a, b, c)
}

#[cfg(not(target_feature = "sha3"))]
#[target_feature(enable = "neon")]
#[inline]
fn xor3_kernel(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
    veorq_u64(a, veorq_u64(b, c))
}
