//! A frame holds no cell that no instruction names.
//!
//! Such a cell costs memory and three committed words a frame, and checks nothing. An `@inline`
//! call used to reserve its return cell before its body, and its tail `return` did not write that
//! cell when it aliased a cell or a g-address.

use crate::common::compile;
use zkdsl_compiler::{disassemble, parse};

/// The frame offsets named in a disassembly: every `fp[n]` and every `fp[n..+k]` run.
fn named_cells(text: &str) -> Vec<u32> {
    let mut cells = Vec::new();
    for line in text.lines() {
        let mut rest = line;
        while let Some(at) = rest.find("fp[") {
            rest = &rest[at + 3..];
            let digits = |s: &str| s.bytes().take_while(u8::is_ascii_digit).count();
            let n = digits(rest);
            let start: u32 = rest[..n].parse().expect("an offset");
            rest = &rest[n..];
            let len: u32 = match rest.strip_prefix("..+") {
                Some(r) => {
                    let k = digits(r);
                    r[..k].parse().expect("a run length")
                }
                None => 1,
            };
            cells.extend(start..start + len);
        }
    }
    cells
}

/// The cells below the largest named offset of `main`'s frame that no instruction of `main` or of the
/// jump tables, which share its frame, names. The frame size itself is not exposed by a compiled program.
/// Cells 0 and 1 are the ABI's return slots, which `main` leaves alone: they are not counted.
fn never_named_in_main(src: &str) -> Vec<u32> {
    let program = compile(&parse(src).expect("parse"));
    let mut named = Vec::new();
    for (name, entry, len) in &program.fn_ranges {
        if name == "main" || name == "__jump_tables" {
            let (lo, hi) = (*entry as usize, (*entry + *len) as usize);
            named.extend(named_cells(&disassemble(&program.prog[lo..hi])));
        }
    }
    let top = *named.iter().max().expect("main names a cell");
    (2..top).filter(|c| !named.contains(c)).collect()
}

/// A chained `match` whose arms are `@inline` calls, and a `keccak_cells` over a list of `@inline`
/// calls, one of which returns an alias of its argument.
const CHAINED_MATCH_AND_LIST: &str = "\
@inline
def step(x, k: Const):
    return x + const(k)

@inline
def pick(a, b, j: Const):
    w = a
    if const(j != 0):
        w = b
    return w

def main():
    p = 1
    groups = StackBuf(2)
    groups[0] = p[1]
    groups[1] = p[GEN]
    acc = p[GEN ** 2]
    for q in unroll(0, 2):
        acc = match(log(groups[q]), range(0, 4), lambda k: step(acc, k))
    leaf = StackBuf(1)
    keccak_cells([pick(acc, p[1], j) for j in unroll(0, 6)], leaf)
    p[GEN ** 3] = leaf[0]
    return
";

#[test]
fn a_chained_match_and_a_keccak_list_leave_no_cell_unnamed() {
    assert_eq!(never_named_in_main(CHAINED_MATCH_AND_LIST), Vec::<u32>::new());
}

/// The same call in a `let`, in an expression and as a tuple, each returning an alias.
#[test]
fn an_inline_return_that_aliases_reserves_no_cell() {
    let src = "\
@inline
def same(a):
    w = a
    return w

@inline
def both(a, b):
    return a, b

def main():
    p = 1
    x = same(p[1])
    y = same(p[GEN]) + same(x)
    u, v = both(x, y)
    p[GEN ** 2] = u + v + same(y)
    return
";
    assert_eq!(never_named_in_main(src), Vec::<u32>::new());
}
