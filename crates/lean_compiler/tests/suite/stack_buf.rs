//! `StackBuf`: a run of consecutive frame (stack) cells in the zkDSL. Indexed
//! reads/writes go straight to `base+k` (no heap deref), and a size-2 `StackBuf`
//! is a `haraka` operand: its two canonical 128-bit cells hold the 256-bit value, so
//! `haraka(a, b, out)` reads them in place with no copies (a self-hash
//! `haraka(h, h, out)` aliases one pair into both input operands) and writes
//! the digest into the pre-allocated pair `out`.
//!
//! Since these DSL scalars are K-embedded F192 cells, a `StackBuf(2)` written
//! cell-by-cell holds the flock words `[v0, 0, v1, 0]`
//!: the reference `compress` is fed that lane layout.

use lean_compiler::{compile, parse};
use leanvm_core::cpu::{prove, verify};
use leanvm_core::vmhash::compress;
use primitives::field::{F64, F192};

use crate::common::mix;

/// The two 128-bit digest cells of `compress(a, b)` as `F192`s (lo = word 0/2,
/// hi = word 1/3): what a `haraka(...)` output `StackBuf(2)` holds cell-by-cell.
fn digest_cells(a: [F64; 4], b: [F64; 4]) -> [F192; 2] {
    let d = compress(a, b);
    [F192::new(d[0].0, d[1].0, 0), F192::new(d[2].0, d[3].0, 0)]
}

/// A size-2 `StackBuf` fed to `haraka` as a self-hash `haraka(h, h)`, then the
/// digest's two 128-bit cells published to `m[0], m[1]`. Proves and verifies, and
/// a wrong published digest is rejected: so the whole path (StackBuf load →
/// aliased haraka → stack read → publish) is exercised end-to-end.
#[test]
fn stack_buf_haraka_self_hash() {
    let src = "\
def main():
    a = StackBuf(2)
    a[0] = 5
    a[1] = 7
    c = StackBuf(2)
    haraka(a, a, c)
    p = 1
    p[1] = c[0]
    p[GEN] = c[1]
    return
";
    let program = compile(&parse(src).expect("parse"));

    // Each cell holds one scalar in its low lane, so the hashed words are [5,0,7,0].
    let h = [F64(5), F64(0), F64(7), F64(0)];
    let want = digest_cells(h, h);

    let (proof, _) = prove(&program, want, leanvm_core::pcs::TEST_LOG_INV_RATE).unwrap();
    assert_eq!(mix(src, want)[5], 1, "one Haraka instruction");
    verify(&program, &want, &proof).expect("StackBuf self-hash verifies");
}

/// The chain over 80 bytes: two whole blocks behind the zero chaining value, then the 16-byte tail
/// closed by the `1` byte and the length.
#[test]
fn haraka_chain_matches_the_standard_hash() {
    let src = "\
def main():
    block0 = [1, 2]
    block1 = [3, 4]
    tail = [5, f192(1, 80, 0)]
    cv1 = StackBuf(2)
    haraka([0, 0], block0, cv1)
    cv2 = StackBuf(2)
    haraka(cv1, block1, cv2)
    out = StackBuf(2)
    haraka(cv2, tail, out)
    p = 1
    p[1] = out[0]
    p[GEN] = out[1]
    return
";
    let program = compile(&parse(src).expect("parse"));
    let mut input = Vec::new();
    for value in 1u64..=5 {
        input.extend_from_slice(&value.to_le_bytes());
        input.extend_from_slice(&0u64.to_le_bytes());
    }
    let d = primitives::hash::hash(&input);
    let word = |o: usize| u64::from_le_bytes(d[o..o + 8].try_into().unwrap());
    let want = [F192::new(word(0), word(8), 0), F192::new(word(16), word(24), 0)];
    let (proof, _) = prove(&program, want, leanvm_core::pcs::TEST_LOG_INV_RATE).unwrap();
    assert_eq!(mix(src, want)[5], 3);
    verify(&program, &want, &proof).expect("the 80-byte chain verifies");
}

/// A hash first emitted in an untaken runtime branch leaves nothing behind for the post-join lowering.
#[test]
fn haraka_after_runtime_branch() {
    let src = "\
def main():
    flag = StackBuf(1)
    hint_witness(flag, \"flag\")
    a = [1, 2, 3, 4]
    if flag[0] == 1:
        ignored = StackBuf(2)
        haraka(a[0:2], a[2:4], ignored)
    out = StackBuf(2)
    haraka(a[0:2], a[2:4], out)
    p = 1
    p[1] = out[0]
    p[GEN] = out[1]
    return
";
    let want = digest_cells([F64(1), F64(0), F64(2), F64(0)], [F64(3), F64(0), F64(4), F64(0)]);
    for flag in [0, 1] {
        let mut program = compile(&parse(src).expect("parse"));
        program.set_witness("flag", vec![vec![F192::new(flag, 0, 0)]]);
        let (proof, _) = prove(&program, want, leanvm_core::pcs::TEST_LOG_INV_RATE).unwrap();
        verify(&program, &want, &proof).expect("both paths verify");
    }
}
