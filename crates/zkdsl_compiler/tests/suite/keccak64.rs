//! `keccak(.., a_shift=n, a_add=c, y_add=d)`: the `KECCAK64` instruction. The keywords lower to its immediates, and only
//! the 64-byte message of four cells with no state takes them. A `keccak` without them is the `KECCAK` instruction.

use crate::common::compile;
use isa::{Op, ZeroScale};
use primitives::field::F64;
use verifier::K;
use zkdsl_compiler::parse;

/// The `KECCAK64` instructions a program compiles to, as `(out_two, s_a, k_a, k_y)` in the words of their immediates.
fn keccak64s(src: &str) -> Vec<(bool, u64, u64, u64)> {
    compile(&parse(src).expect("parse"))
        .prog
        .iter()
        .filter_map(|op| match *op {
            Op::Keccak64 {
                out_two,
                s_a,
                s_inv,
                k_a,
                k_y,
                ..
            } => {
                assert_eq!(s_a * s_inv, K(1), "the stored inverse is the scale's");
                Some((out_two, s_a.0, k_a.0, k_y.0))
            }
            _ => None,
        })
        .collect()
}

/// The instructions of one opcode a program compiles to, by the number of `KECCAK`, `KECCAK64`, `MUL` and `XOR`.
fn opcodes(src: &str) -> [usize; 4] {
    let program = compile(&parse(src).expect("parse"));
    let count = |is: fn(&Op) -> bool| program.prog.iter().filter(|op| is(op)).count();
    [
        count(|op| matches!(op, Op::Keccak { .. })),
        count(|op| matches!(op, Op::Keccak64 { .. })),
        count(|op| matches!(op, Op::Mul { .. })),
        count(|op| matches!(op, Op::Xor { .. })),
    ]
}

/// The four message cells, indented as a function body (a string continuation would strip the indent).
const CELLS: &str = "    p = StackBuf(4)\n    p[0] = 5\n    p[1] = 7\n    p[2] = 9\n    p[3] = 11\n";

/// A program hashing the four cells of `p` into `d`, with the keyword arguments `kwargs`.
fn hashing(kwargs: &str, destination: usize) -> String {
    format!(
        "def main():\n{CELLS}    d = StackBuf({destination})\n    keccak([p[0], p[1]], [p[2], p[3]], d{kwargs})\n    return\n"
    )
}

#[test]
fn the_keywords_lower_to_the_immediates() {
    let g = F64::G;
    // Shift 3, constant 1000 on A, 5 on Y, a two-cell destination.
    let s_a = g.pow(3).inv();
    let got = keccak64s(&hashing(", a_shift=3, a_add=1000, y_add=5", 2));
    assert_eq!(got, [(true, s_a.0, (F64(1000) * s_a).0, 5)]);
    // One cell written; a constant alone, and an offset alone.
    assert_eq!(keccak64s(&hashing(", a_add=77", 1)), [(false, 1, 77, 0)]);
    assert_eq!(keccak64s(&hashing(", y_add=9", 2)), [(true, 1, 0, 9)]);
    // The shift is read as a compile-time integer expression.
    assert_eq!(keccak64s(&hashing(", a_shift=2 + 1", 2)), [(true, s_a.0, 0, 0)]);
}

/// A `keccak` without the keywords is the `KECCAK` instruction, whatever its form.
#[test]
fn a_plain_keccak_stays_a_keccak() {
    assert!(keccak64s(&hashing("", 2)).is_empty());
    assert_eq!(opcodes(&hashing("", 2))[..2], [1, 0]);
    assert_eq!(opcodes(&hashing(", a_add=1", 2))[..2], [0, 1]);
}

#[test]
#[should_panic(expected = "are for the 64-byte message of four cells")]
fn a_48_byte_message_refuses_the_keywords() {
    keccak64s(&hashing(", len=48, a_add=1", 2));
}

#[test]
#[should_panic(expected = "no tail=")]
fn an_80_byte_message_refuses_the_keywords() {
    keccak64s(&format!(
        "def main():\n{CELLS}    d = StackBuf(2)\n    keccak([p[0], p[1]], [p[2], p[3]], d, tail=[1, 2, 3, 4], len=80, y_add=1)\n    return\n"
    ));
}

#[test]
#[should_panic(expected = "no cap=")]
fn a_stateful_hash_refuses_the_keywords() {
    keccak64s(&format!(
        "def main():\n{CELLS}    s = StackBuf(13)\n    keccak([p[0], p[1]], [p[2], p[3]], s)\n    d = StackBuf(2)\n    keccak([p[0], p[1]], [p[2], p[3]], d, cap=s, a_add=1)\n    return\n"
    ));
}

#[test]
#[should_panic(expected = "unknown keccak keyword")]
fn a_continued_hash_refuses_the_keywords() {
    keccak64s(&format!(
        "def main():\n{CELLS}    s = StackBuf(13)\n    keccak_cells([p[0], p[1], p[2], p[3]], s, final=0)\n    d = StackBuf(2)\n    keccak_cells([p[0]], d, state=s, a_add=1)\n    return\n"
    ));
}

#[test]
#[should_panic(expected = "write one or two cells")]
fn a_state_destination_refuses_the_keywords() {
    keccak64s(&hashing(", a_shift=1", 13));
}

#[test]
#[should_panic(expected = "does not fit one 64-bit limb")]
fn a_constant_wider_than_a_limb_is_refused() {
    keccak64s(&hashing(", a_add=18446744073709551616", 2));
}

/// A scale of zero cannot be built, and `a_shift` never makes one: `g^(-n)` is a unit for every `n`.
#[test]
fn a_zero_scale_cannot_be_built() {
    assert_eq!(
        Op::keccak64([2, 3, 4, 5], 6, true, K(0), K(1), K(1)).err(),
        Some(ZeroScale)
    );
    for n in [
        0u128,
        1,
        63,
        64,
        u128::from(u64::MAX) - 1,
        u128::from(u64::MAX),
        u128::from(u64::MAX) + 1,
    ] {
        let src = hashing(&format!(", a_shift={n}"), 2);
        let [(_, s_a, ..)] = keccak64s(&src)[..] else {
            panic!("one KECCAK64")
        };
        assert_ne!(s_a, 0, "shift {n}");
    }
}

/// The program that spells `A g^n + c` and `Y + d` with `MUL` and `XOR` hashes with a `KECCAK` and those two
/// instructions; the one that says it in the keywords is one `KECCAK64`. That both hash alike is a test of the
/// experimental prover's `lean_vm_core`, which runs `KECCAK64`.
#[test]
fn the_keywords_replace_the_arithmetic_they_spell() {
    let spelled = [
        "    t = StackBuf(2)",
        "    keccak([p[0], p[1] * GEN ** 3 + 1000], [p[2], p[3] + 5], t)",
        "    return",
    ]
    .join("\n");
    let said = [
        "    d = StackBuf(2)",
        "    keccak([p[0], p[1]], [p[2], p[3]], d, a_shift=3, a_add=1000, y_add=5)",
        "    return",
    ]
    .join("\n");
    let [keccak, keccak64, mul, xor] = opcodes(&format!("def main():\n{CELLS}{spelled}\n"));
    assert_eq!((keccak, keccak64), (1, 0));
    assert!(mul > 0 && xor > 0, "the arithmetic is instructions");
    let [keccak, keccak64, ..] = opcodes(&format!("def main():\n{CELLS}{said}\n"));
    assert_eq!((keccak, keccak64), (0, 1));
}
