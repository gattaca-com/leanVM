//! One source compiles to one program, always, and the same program it compiled
//! to yesterday.
//!
//! The bytecode digest leads the Fiat--Shamir transcript, so two builds of one
//! source that disagree are two incompatible proof systems, and the symptom is a
//! proof that stops verifying rather than a crash.
//!
//! Two different properties, and only the first is about determinism:
//!
//! * *Within a process*, compiling twice is a real perturbation rather than a
//!   repeat, since `RandomState` bumps its seed once per map, so the second
//!   compilation hashes with different keys than the first.
//! * *Across commits*, `GOLDEN` is a SNAPSHOT of the compiler's output. It does
//!   not prove determinism (nothing iterating a hash container reaches the
//!   bytecode today, and deliberately reversing the branch-output order at a join
//!   moves no digest). It earns its place a different way: a codegen change that
//!   was not intended shows up here and nowhere else, and every entry that moved
//!   this far was a change someone then had to justify.
//!
//! So a moved digest is a question, not a chore: update `GOLDEN` in the same
//! commit and say in the message which change moved it.

use std::collections::BTreeMap;
use std::fs;

use lean_compiler::{compile, parse};
use leanvm_core::cpu::Program;

/// `tests/programs/<name>.py` against the digest of the bytecode it compiles to.
/// The list is closed: a new program must be added here, so one cannot be added
/// without a digest.
#[rustfmt::skip]
const GOLDEN: &[(&str, &str)] = &[
    ("conditionals", "405bad3b46a07a1fe9a83165c33422669fc803c07650f6a3627d90a30ef9f1b1"),
    ("const_params", "a61e61fc18318aaa385750a9d51bca677a8933315a8febcdaea7fe4b9cb2aad9"),
    ("fibonacci", "de2ebd6a923077816d4526184553a5c45fa4fe588b206436eabb72f80f513fe1"),
    ("hash_heap_chain", "bf5b619a0b1839b93d7e327077c92538f663c55bf137a494596644f93d5a617c"),
    ("hash_slices", "709fd6a2de4ec60109dd5f6db3e79bab4a212fd3f075bd3c56b4d6bdeb2709fb"),
    ("heapbuf_dyn", "34c5fa8343579df3cf5570e16c8affeb6853f2dbf1d2208ce6659806607ad1eb"),
    ("hint", "75ba110cdf8d77b01f340896d2eb98ad71d5ae107febbb1f68fb6e9d5d61854c"),
    ("identities", "5e5d7c7c6e08dd5b394f47718612ea49b2fd1a513fe982b2cd37337a7bc861ae"),
    ("match", "43f7adfe3f8baf2b7eaff8d8d199d27c7c08afbfdcc0acb2ad9e9b167237ca9c"),
    ("match_arms", "9412246969c695867330d26815e5cec3a5c1988c3d7b904034cda6983589c7b2"),
    ("nested", "656c030740ba33ce74dae5aa8acc270a7267f17e956ddf2f76fb64aea13c2cd8"),
    ("runtime_loop", "7b8cbb8261ba9e2c0f8c835f20bca21fd86d8a2d25bef4376b0704373bae04d8"),
    ("scoping", "0dd1095b0322ebc97e5715f30becc871e3c7d12ff307a291fa2bef52915d52d6"),
    ("unroll", "607baf712352ca94d33bfd97d255cec95a85164a2f2db61e449ff8586b4065dc"),
    ("wots_walk", "cb2e51ae6be8731f254753baa893dd4487b7bfce3809c0d5f4f747af1a413e91"),
];

fn digest(p: &Program) -> String {
    primitives::hash::hash(format!("{:?}", p.prog()).as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Every program in `tests/programs/`, compiled twice.
#[test]
fn bytecode_is_reproducible() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/programs");
    let mut paths: Vec<_> = fs::read_dir(dir)
        .expect("tests/programs")
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "py"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no .py programs found");

    let mut actual: Vec<(String, String)> = Vec::new();
    for path in &paths {
        let name = path.file_stem().expect("file stem").to_string_lossy().into_owned();
        let src = fs::read_to_string(path).unwrap_or_else(|e| panic!("{name}: read: {e}"));
        let one = compile(&parse(&src).unwrap_or_else(|e| panic!("{name}: parse: {e}")));
        let two = compile(&parse(&src).unwrap_or_else(|e| panic!("{name}: parse: {e}")));
        assert_eq!(
            digest(&one),
            digest(&two),
            "{name}: two compilations of one source produced different bytecode, \
             so the compiler is reading a hash seed"
        );
        actual.push((name, digest(&one)));
    }

    let want: BTreeMap<&str, &str> = GOLDEN.iter().copied().collect();
    let moved: Vec<&str> = actual
        .iter()
        .filter(|(n, d)| want.get(n.as_str()) != Some(&d.as_str()))
        .map(|(n, _)| n.as_str())
        .collect();
    let dropped: Vec<&str> = want
        .keys()
        .copied()
        .filter(|n| !actual.iter().any(|(a, _)| a == n))
        .collect();
    assert!(
        dropped.is_empty(),
        "GOLDEN names a program that no longer exists: {dropped:?}"
    );
    if !moved.is_empty() {
        let table: String = actual
            .iter()
            .map(|(n, d)| format!("    (\"{n}\", \"{d}\"),\n"))
            .collect();
        panic!("bytecode changed for {moved:?}\n\nif that was intended, GOLDEN is now:\n{table}");
    }
}
