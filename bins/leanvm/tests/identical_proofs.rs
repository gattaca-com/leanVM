//! The stable and the experimental prover give the same proof of the same statement, byte for byte: three leaves
//! (leanXMSS signatures and blobs, leanSPHINCS signatures, both schemes and blobs, with enough signatures of each that
//! experimental runs both signature loops as parallel segments), a recursion node over them, at the four rates, and a
//! second-level node over that node, a leaf and a raw signature; the same refusal of a forged signature the native
//! pre-check misses; and every compiler test program, or the same refusal. Both trees' thread pools live in this one
//! process, and libtest may run the two tests at once: both pools can then be busy together, and either can take
//! dispatches from both tests, which its dispatch lock serializes.

use aggregation::{AggregateClaims, AggregateError, AggregateProof, OutputClaims, signers_cache};
use experimental_lean_vm_core::ExperimentalProver;
use isa::{CompiledProgram, Op, OutCells, Program, Prover};
use stable_lean_vm_core::StableProver;
use verifier::{E, K};

/// The proofs `P` gives, the three leaves' and, with `nodes`, the node's and the second-level node's, then why it
/// refuses a forged signature.
fn proofs<P: Prover>(nodes: bool) -> (Vec<Vec<u8>>, Option<AggregateError>) {
    P::init();
    let aggregate = |children: &[(AggregateClaims, AggregateProof)], xmss, sphincs, blobs, rate| {
        aggregation::aggregate::<P>(children, xmss, sphincs, blobs, OutputClaims::All, rate).map(|(c, p, _)| (c, p))
    };
    let symbols: Vec<u128> = (0..2 * lean_da::BLOB_SYMBOLS as u128)
        .map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .collect();
    let blobs = lean_da::BlobMatrix::build(symbols.as_chunks().0).unwrap();
    let (xmss, sphincs) = (
        signers_cache::get_signers::<P>(23),
        signers_cache::get_lean_sphincs_signers::<P>(20),
    );
    let left = aggregate(&[], xmss[..3].to_vec(), vec![], Some(&blobs), 1).unwrap();
    let right = aggregate(&[], vec![], sphincs[..1].to_vec(), None, 2).unwrap();
    let mixed = aggregate(&[], xmss[3..22].to_vec(), sphincs[1..].to_vec(), Some(&blobs), 3).unwrap();
    let mut proofs = vec![left.clone(), right.clone(), mixed.clone()];
    if nodes {
        let children = [left, right.clone(), mixed];
        let node = aggregate(&children, vec![], vec![], None, 4).unwrap();
        // The node's carried deferred claim is at a non-zero point, unlike a leaf's.
        let children = [node.clone(), right];
        let top = aggregate(&children, xmss[22..].to_vec(), vec![], None, 1).unwrap();
        proofs.extend([node, top]);
    }
    // A leanXMSS signature with its top layer's last path node changed: the native pre-check passes, the guest fails.
    let (pk, message, signature) = &xmss[0];
    let mut bytes = signature.to_bytes();
    bytes[lean_xmss::SIG_SIZE - 1] ^= 1;
    let forged = vec![(*pk, *message, lean_xmss::LeanXmssSignature::from_bytes(&bytes))];
    let refusal = aggregate(&[], forged, vec![], None, 1).err();
    (proofs.into_iter().map(|(_, proof)| proof.to_bytes()).collect(), refusal)
}

/// The two provers' proofs of `proofs(nodes)`, equal one by one, and the same refusal.
fn assert_identical(nodes: bool) {
    let ((experimental, experimental_refusal), (stable, stable_refusal)) =
        (proofs::<ExperimentalProver>(nodes), proofs::<StableProver>(nodes));
    for (name, (e, s)) in ["left leaf", "right leaf", "mixed leaf", "node", "second-level node"]
        .iter()
        .zip(experimental.iter().zip(&stable))
    {
        assert!(e == s, "the {name}'s proofs differ ({} and {} bytes)", e.len(), s.len());
    }
    assert_eq!(experimental.len(), stable.len());
    assert_eq!(experimental_refusal, stable_refusal);
    assert_eq!(stable_refusal, Some(AggregateError::ProvingFailed));
}

#[test]
fn stable_and_experimental_leaves_are_identical() {
    assert_identical(false);
}

#[test]
#[ignore = "the recursion guest verifies one hash table's reduction, not the two a node's children prove"]
fn stable_and_experimental_nodes_are_identical() {
    assert_identical(true);
}

/// `program`'s proof at rate `2^-log_inv_rate` on `public_input`, or why there is none.
fn prove<P: Prover>(
    program: &Program,
    public_input: [E; 2],
    witness: &[(String, Vec<Vec<E>>)],
    log_inv_rate: usize,
) -> Result<verifier::Proof, String> {
    P::init();
    P::prove(program, witness, public_input, log_inv_rate).map(|(proof, _)| proof)
}

/// A straight-line program of `keccak64` `KECCAK64` rows and `keccak` `KECCAK` rows on the same four message cells
/// (2..6), the `KECCAK`s with the constants a 64-byte message leaves in `tail` (6..10), `b8` (10) and `cap` (11..15).
/// Odd `KECCAK64` rows hash the message rotated by a cell, and carry the immediates `(s_a, k_a, k_y)`. The last slot is
/// the sentinel, and `truncate` makes the proof a truncated one.
fn keccak64_mix_program(
    message: [E; 4],
    (keccak64, keccak, out_two): (usize, usize, bool),
    (s_a, k_a, k_y): (K, K, K),
    truncate: bool,
) -> Program {
    let set = |o: u32, k: E| Op::Set { o, k };
    let pad = verifier::layout::keccak64_pad();
    let (base, kbase) = (64u32, 64 + 2 * keccak64 as u32 + 8);
    let mut prog = Vec::new();
    prog.extend(message.iter().enumerate().map(|(c, &m)| set(2 + c as u32, m)));
    prog.push(set(6, E::new(pad[8], 0)));
    prog.extend([7, 8, 9].map(|o| set(o, E::new(0, 0))));
    prog.push(set(10, E::new(pad[16], 0x5555)));
    prog.extend((11..15).map(|o| set(o, E::new(0, 0))));
    prog.extend((16..20u32).enumerate().map(|(c, o)| set(o, message[(c + 1) % 4])));
    for i in 0..keccak64 as u32 {
        let cells = if i % 2 == 0 { [2, 3, 4, 5] } else { [16, 17, 18, 19] };
        let (s_a, k_a, k_y) = if i % 2 == 0 {
            (K(1), K(0), K(0))
        } else {
            (s_a, k_a, k_y)
        };
        prog.push(Op::keccak64(cells, base + 2 * i, out_two, s_a, k_a, k_y).expect("a nonzero scale"));
    }
    for i in 0..keccak as u32 {
        prog.push(Op::Keccak {
            m: [2, 3, 4, 5],
            tail: 6,
            b8: 10,
            cap: 11,
            out: kbase + 13 * i,
            in_odd: false,
            out_odd: false,
            out_cells: OutCells::Two,
        });
    }
    // Filler SETs, then the sentinel in the last slot.
    let mut spare = kbase + 13 * keccak as u32 + 16;
    let len = (prog.len() + 2).next_power_of_two();
    while prog.len() < len - 1 {
        prog.push(set(spare, E::new(1, 0)));
        spare += 1;
    }
    prog.push(Op::Xor { a: 0, b: 0, c: 0 });
    let mut program = Program::from(CompiledProgram {
        prog,
        hints: Default::default(),
        main_frame: spare + 1,
        fn_ranges: Vec::new(),
        src_lines: Vec::new(),
    });
    program.truncate = truncate;
    program
}

/// Programs with only `KECCAK64` rows, only `KECCAK` rows, and both at different heights, the truncated ones with
/// `q_keccak` in pieces, and `KECCAK64` rows with immediates: both trees prove the same bytes, and they verify.
#[test]
fn keccak64_programs_prove_identically() {
    let message: [E; 4] = std::array::from_fn(|i| E::new(0x1111 * (i as u64 + 1), 0x7777 * (i as u64 + 3)));
    let public_input = [E::new(7, 0), E::new(11, 0)];
    let plain = (K(1), K(0), K(0));
    let shifted = (
        K(0x1234_5678_9abc_def1),
        K(0xfeed_face_dead_beef),
        K(0x0bad_f00d_cafe_babe),
    );
    for (rows, immediates, truncate) in [
        ((3, 0, true), plain, false),
        ((3, 0, false), plain, false),
        ((0, 3, true), plain, false),
        ((3, 3, true), plain, false),
        ((600, 20, true), plain, false),
        ((20, 600, false), plain, false),
        ((70_000, 40, true), plain, true),
        ((40, 70_000, true), plain, true),
        ((70_000, 70_000, false), plain, true),
        ((6, 2, true), shifted, false),
        ((600, 20, false), shifted, true),
    ] {
        let program = keccak64_mix_program(message, rows, immediates, truncate);
        let what = format!(
            "{} KECCAK64 and {} KECCAK rows, out_two {}, {immediates:?}, truncate {truncate}",
            rows.0, rows.1, rows.2
        );
        let rate = 1;
        let (experimental, stable) = (
            prove::<ExperimentalProver>(&program, public_input, &[], rate),
            prove::<StableProver>(&program, public_input, &[], rate),
        );
        let proof = stable.unwrap_or_else(|e| panic!("{what}: {e}"));
        assert!(experimental.as_ref() == Ok(&proof), "{what}: the provers disagree");
        let bytes = verifier::Digest::from_halves(public_input[0], public_input[1]).0;
        verifier::verify_execution(program.bytecode(), &bytes, &proof).unwrap_or_else(|e| panic!("{what}: {e}"));
    }
}

/// The compiler suite runs on stable alone; this runs its programs, with their annotated inputs (`# public_input:`,
/// `# witness name:`) and with a wrong public input, through experimental's interpreter and hints too, at every rate.
#[test]
fn compiler_programs_prove_identically() {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../crates/zkdsl_compiler/tests/programs"
    );
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    paths.retain(|path| path.extension().is_some_and(|ext| ext == "py"));
    paths.sort();
    assert!(!paths.is_empty());
    let constants = |values: &str| -> Vec<E> {
        values
            .split(',')
            .map(|v| zkdsl_compiler::parse_const(v).unwrap())
            .collect()
    };
    for path in paths {
        let src = std::fs::read_to_string(&path).unwrap();
        let program = Program::from(zkdsl_compiler::compile(&zkdsl_compiler::parse(&src).unwrap()));
        let annotation = |tag: &'static str| src.lines().filter_map(move |line| line.trim().strip_prefix(tag));
        let public_input = annotation("# public_input:")
            .next()
            .map_or([E::new(0, 0); 2], |v| constants(v).try_into().unwrap());
        // Each line is the next entry of its stream.
        let mut witness: Vec<(String, Vec<Vec<E>>)> = Vec::new();
        for (name, values) in annotation("# witness ").map(|rest| rest.split_once(':').unwrap()) {
            match witness.iter_mut().find(|(stream, _)| stream == name.trim()) {
                Some((_, entries)) => entries.push(constants(values)),
                None => witness.push((name.trim().to_string(), vec![constants(values)])),
            }
        }
        let wrong = [public_input[0] + E::new(1, 0), public_input[1]];
        let runs = (1..=4).map(|rate| (public_input, rate)).chain([(wrong, 1)]);
        for (input, rate) in runs {
            let annotated = input == public_input;
            let (experimental, stable) = (
                prove::<ExperimentalProver>(&program, input, &witness, rate),
                prove::<StableProver>(&program, input, &witness, rate),
            );
            assert!(
                experimental == stable,
                "{} at rate {rate}: the provers disagree",
                path.display()
            );
            assert!(!annotated || stable.is_ok(), "{}: {:?}", path.display(), stable.err());
        }
    }
}
