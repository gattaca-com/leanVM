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
    ("conditionals", "a688bb7316036e793f736e48fb8785d038a951ea91cedc4e51d6957d9b244806"),
    ("const_params", "6c9768a12abf9eb5cccfdf9a06e314991aef8486b8b6313db6cc9a02b2ed2eef"),
    ("fibonacci", "3a85dcfdca1c9451952254ce11f4cee7abe398eb0094409b0d87c747d2624836"),
    ("hash_heap_chain", "810899876923d7a2a5f878755b828b149f671d21742ab78d2d277c6a1ae2ba09"),
    ("hash_slices", "f58737b2a3f05c069af857a4ad69dd8346feb2993ad73a9a892f4b0058fc898e"),
    ("heapbuf_dyn", "09b33afca0f72fd4e731f94f430a63f362ff3fd0a5a3e21d3f38c1d0c1d46713"),
    ("hint", "c5aad11f1f317e7d18d963f822a38f396adcfc5ad67f2680e7a0b20a65b9a483"),
    ("identities", "55febdef5d82b30a3a53d604ff7de3dcd5a50d2fffd5f196f75336fdfee4a344"),
    ("match", "f7e4b98c1007d4811caa1b55c33d7e52ef9757fe8602872fcad7ee8c960a1fcd"),
    ("match_arms", "bb0b6d078556dcf37cd38762e754e5da74ea2d3603bda1075195e60f97b39d06"),
    ("nested", "55a7d7adac24ffccc52b4c34933072da157f56805f9c55c597222bb824ded51c"),
    ("runtime_loop", "5d1a0d6b8d4b3e3e01e0caa3af63246ea96f9160eb8085a1be7e2751c977e48e"),
    ("scoping", "2334159d7d80ab2aa3210a7eb9a1ed583e6159ea2b76b80e657bd2fb9ef4391f"),
    ("unroll", "3cd461dfc7102990d25d8ca9060812eed2a8ea70b05c588606885ad0da76964b"),
    ("wots_walk", "7f96b9b1f2644f7762f7c09f0ce66aadaf5e49a89ecd9f8c00a436913705b06e"),
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
