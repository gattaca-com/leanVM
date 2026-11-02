//! Back-to-back `match` statements lower as one chain (`FnLower::lower_match_chain`):
//! an arm of each dispatches straight into the next one's table, and only the last
//! one's arms meet at a join. These pin that a chain computes what separate matches
//! do, that it is cheaper by exactly the joins it drops, and that a match whose
//! scrutinee reads the previous match's result is not chained.

use lean_compiler::{compile, parse};
use leanvm_core::cpu::{prove, verify};
use primitives::field::{F64, F192, g_pow};

/// `acc` folds each scrutinee's arm into its own nibble. `sel` is a fourth match,
/// chained to the third, and `w` reads `sel`, so it must wait for the join. `sep`
/// sits between the second and third `acc` matches.
fn source(sep: &str) -> String {
    format!(
        "\
@inline
def step(acc, i: Const, k: Const):
    return acc + const((k + 1) * 16 ** i)

@inline
def pick(k: Const):
    return GEN ** k

def main():
    xs = StackBuf(3)
    hint_witness(xs, \"xs\")
    for i in unroll(0, 3):
        assert log(xs[i]) < 4
    acc = 0
    for i in unroll(0, 2):
        acc = match(log(xs[i]), range(0, 4), lambda k: step(acc, i, k))
{sep}    acc = match(log(xs[2]), range(0, 4), lambda k: step(acc, 2, k))
    sel = match(log(xs[0]), range(0, 4), lambda k: pick(k))
    w = match(log(sel), range(0, 4), lambda k: const(k + 7))
    p = GEN ** 0
    p[1] = acc
    p[GEN] = w
    return
"
    )
}

/// Proves and verifies one variant at scrutinees `g^ks`, returning its cycles.
fn run(sep: &str, ks: [usize; 3]) -> usize {
    let mut program = compile(&parse(&source(sep)).expect("parse"));
    program.set_witness("xs", vec![ks.iter().map(|&k| F192::from(g_pow(k))).collect()]);
    program.set_witness("z", vec![vec![F192::ZERO]]);
    let acc: u64 = ks.iter().enumerate().map(|(i, &k)| (k as u64 + 1) << (4 * i)).sum();
    let want = [F192::from(F64(acc)), F192::from(F64(ks[0] as u64 + 7))];
    let (proof, _) = prove(&program, want, leanvm_core::pcs::TEST_LOG_INV_RATE).unwrap();
    verify(&program, &want, &proof).expect("verifies");
    program.execute(want).unwrap().cycles
}

#[test]
fn consecutive_matches_chain() {
    for ks in [[0, 1, 2], [3, 2, 1], [1, 3, 0]] {
        let chained = run("", ks);
        // A hint is free but not a match, so it splits the chain in two. The
        // second arm of the first half then exits to a join: its `SET` and,
        // unless it is the table's last arm, its arm's `JUMP`.
        let split = run("    z = hint_witness(\"z\")\n", ks);
        let jump = usize::from(ks[1] != 3);
        assert_eq!(split, chained + 1 + jump, "scrutinees {ks:?}");
    }
}

/// An `@inline` body may hold a `match`: each call site expands its own dispatch,
/// with its own table, into the caller's frame.
#[test]
fn an_inline_body_may_dispatch() {
    let src = "\
@inline
def bump(x, acc):
    acc = match(log(x), range(0, 4), lambda k: acc + const(k + 1))
    return acc * GEN

def main():
    xs = StackBuf(2)
    hint_witness(xs, \"xs\")
    assert log(xs[0]) < 4
    assert log(xs[1]) < 4
    a = bump(xs[0], 0)
    b = bump(xs[1], a)
    p = GEN ** 0
    p[1] = a
    p[GEN] = b
    return
";
    for (k0, k1) in [(0, 3), (2, 1)] {
        let mut program = compile(&parse(src).expect("parse"));
        program.set_witness("xs", vec![vec![F192::from(g_pow(k0)), F192::from(g_pow(k1))]]);
        // `acc * GEN` doubles a polynomial: one bit up, no reduction at this size.
        let a = ((k0 as u64) + 1) << 1;
        let b = (a ^ (k1 as u64 + 1)) << 1;
        let want = [F192::from(F64(a)), F192::from(F64(b))];
        let (proof, _) = prove(&program, want, leanvm_core::pcs::TEST_LOG_INV_RATE).unwrap();
        verify(&program, &want, &proof).expect("verifies");
    }
}
