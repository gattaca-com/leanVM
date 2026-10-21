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
    ("conditionals", "d3d6c3479fd9ef3f1044d02513e80ec9b684718e63e6ff6493fdca514529d9b6"),
    ("const_params", "7fbe8824f76aaf90a537ff2e843a033e2da6a49ea57be394008429fef590c5d9"),
    ("fibonacci", "af4d75bfaff013ece7ce26e481b0c99a4be0d251376602248809a4a344fb3337"),
    ("hash_heap_chain", "910472d620d05d3408a948dcf6a596431e91ebdaf5235b4abc204b2a795a0b50"),
    ("hash_slices", "97136ebe4eb6656be8bef80afa4e9048e83b04245f6467e88e88d2a5eced2ae6"),
    ("heapbuf_dyn", "9a5a4a3b5a27627e8bf0dd6bdd65ff9818b3564e4ef22727ffd191149843c9fc"),
    ("hint", "f719a24da96d78cac97f8c6b64b4c7810a8b93dfa67738eaff8720f01ce123f8"),
    ("identities", "1b93fd44d3673a4bf86f100bd49f85900f9ce796c09180a17abbe592ccf3b040"),
    ("match", "d14d04d092de75c3adef1f99d28892c41025450f04865b758c20d1f701d8d8fb"),
    ("match_arms", "7942ffd90c712bcfee8d2f1d7f25871332bef1ebb1c408a7310b34eb2b93f3a2"),
    ("nested", "3008e79cdf0a619d57ca110c57f19cc750f89689398b10cae2c1ad487d8ed5d6"),
    ("runtime_loop", "667e7a7df84a9b17c51454ecb0f5fd2153e0a936c343980f92b084fcd7f204c6"),
    ("scoping", "e24ae7932597bf03607705ff26c840a24f8aaba1fac27494231ea13ca64e3098"),
    ("unroll", "9aa2b122df4f3ccdd7218d80457ac30fe98eec52cd832a89f0dfe53e06879885"),
    ("wots_walk", "eb8072c3b0815bc275e5b3b74e9bc1d149c27acac1a9b92821b4ee79ce994f9f"),
];

fn digest(p: &Program) -> String {
    primitives::hash::hash(format!("{:?}", p.prog).as_bytes())
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
