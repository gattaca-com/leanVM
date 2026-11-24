use lean_xmss::internals::*;
use lean_xmss::*;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn test_message() -> Message {
    std::array::from_fn(|i| (i * 3 + 7) as u8)
}

/// A full key: the whole top tree, and signatures in two bottom trees, prepared at
/// once on two threads, each with its own top-layer signature. The leaf index rides
/// the signature, so the verifier hashes at it: moved to a neighbour, or the message
/// changed, the signature no longer verifies.
#[test]
fn full_key_signs_and_verifies() {
    let (sk, pk) = key_gen(&[1; SECRET_KEY_LEN]);
    let message = test_message();
    let other = (7 << LAYER_HEIGHT) + 3;
    std::thread::scope(|s| {
        s.spawn(|| sk.prepare(other));
        sk.prepare(5);
    });
    let first = sign(&sk, &message, 5);
    assert!(verify(&pk, &message, &first));
    let other_tree = sign(&sk, &message, other);
    assert!(verify(&pk, &message, &other_tree));
    assert_ne!(
        lean_xmss::internals::sig_top(&first),
        lean_xmss::internals::sig_top(&other_tree)
    );
    assert_eq!(sign(&sk, &message, 5), first, "signing is deterministic");
    let mut bytes = first.to_bytes();
    bytes[..4].copy_from_slice(&4u32.to_le_bytes());
    assert!(!verify(&pk, &message, &LeanXmssSignature::from_bytes(&bytes)));
    let mut other_message = message;
    other_message[0] ^= 1;
    assert!(!verify(&pk, &other_message, &first));
}

/// Signatures in one bottom tree that leave a run of 256 leaves and come back to it
/// verify: a path is right whether it reuses the levels the previous signature built,
/// or builds them again for another run.
#[test]
fn signatures_across_leaf_runs_verify() {
    let (sk, pk) = key_gen(&[2; SECRET_KEY_LEN]);
    let message = test_message();
    for leaf_index in [5, 6, (1 << 8) + 1, 7] {
        assert!(
            verify(&pk, &message, &sign(&sk, &message, leaf_index)),
            "leaf {leaf_index}"
        );
    }
}

/// Threads signing with one key in different runs of 256 leaves of one bottom tree, at
/// once, make signatures that verify and that a copy of the key, signing one leaf at a
/// time, makes too.
#[test]
fn concurrent_signatures_across_leaf_runs_match_serial_ones() {
    let seed = [6; SECRET_KEY_LEN];
    let (sk, pk) = key_gen(&seed);
    let message = test_message();
    sk.prepare(0);
    let copy = LeanXmssExpandedSecretKey::from_cache_bytes(&seed, &pk, &sk.cache_to_bytes()).unwrap();
    let leaves: Vec<LeafIndex> = (0..4).flat_map(|run| [run << 8 | 1, run << 8 | 2]).collect();
    let concurrent: Vec<_> = std::thread::scope(|s| {
        let handles: Vec<_> = leaves
            .chunks(2)
            .map(|pair| s.spawn(|| pair.iter().map(|&leaf| sign(&sk, &message, leaf)).collect::<Vec<_>>()))
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    for (&leaf, signature) in leaves.iter().zip(&concurrent) {
        assert!(verify(&pk, &message, signature), "leaf {leaf}");
        assert_eq!(*signature, sign(&copy, &message, leaf), "leaf {leaf}");
    }
}

/// Tries every index, the last first, and returns the least hit: a pool may try them in
/// any order.
struct LastFirst;

impl Parallel for LastFirst {
    fn find_first(n: usize, pred: impl Fn(usize) -> bool + Sync) -> Option<usize> {
        (0..n).rev().filter(|&i| pred(i)).last()
    }
}

/// A search that tries trials out of order signs as the calling thread's search, on
/// both layers: the top layer's signature is made when the bottom tree is built.
#[test]
fn sign_with_any_search_order_matches_sign() {
    let seed = [4; SECRET_KEY_LEN];
    let (sk, pk) = key_gen(&seed);
    let other = LeanXmssExpandedSecretKey::from_cache_bytes(&seed, &pk, &sk.cache_to_bytes()).unwrap();
    let message = test_message();
    let leaf_index = (2 << LAYER_HEIGHT) + 9;
    let signature = sign_with::<LastFirst>(&sk, &message, leaf_index);
    assert_eq!(signature, sign(&other, &message, leaf_index));
    assert!(verify(&pk, &message, &signature));
}

/// A key loaded from its cache signs as the original.
#[test]
fn cache_round_trip() {
    let seed = [2; SECRET_KEY_LEN];
    let (sk, pk) = key_gen(&seed);
    let (first, other) = ((3 << LAYER_HEIGHT) + 77, (9 << LAYER_HEIGHT) + 65535);
    sk.prepare(first);
    sk.prepare(other);
    let bytes = sk.cache_to_bytes();
    let loaded = LeanXmssExpandedSecretKey::from_cache_bytes(&seed, &pk, &bytes).unwrap();
    assert_eq!(loaded.public_key(), pk);
    assert_eq!(loaded.cache_to_bytes(), bytes);
    for leaf_index in [first, other] {
        assert_eq!(
            sign(&loaded, &test_message(), leaf_index),
            sign(&sk, &test_message(), leaf_index)
        );
    }
    assert!(LeanXmssExpandedSecretKey::from_cache_bytes(&seed, &pk, &bytes[1..]).is_none());

    // Tampered bytes are refused: a top node, a bottom tree's index, a bottom node, its signature.
    for at in [
        0,
        CACHED_TREE_SIZE,
        CACHED_TREE_SIZE + CACHED_BOTTOM_SIZE - 1,
        CACHED_TREE_SIZE + 4,
    ] {
        let mut tampered = bytes.clone();
        tampered[at] ^= 1;
        assert!(
            LeanXmssExpandedSecretKey::from_cache_bytes(&seed, &pk, &tampered).is_none(),
            "byte {at}"
        );
    }
    // So is another key's cache.
    let (_, other_pk) = key_gen(&[9; 32]);
    assert!(LeanXmssExpandedSecretKey::from_cache_bytes(&seed, &other_pk, &bytes).is_none());
}

/// The cache keeps as many bottom trees as set, the most recently used.
#[test]
fn cache_keeps_the_set_number_of_bottom_trees() {
    let seed = [3; SECRET_KEY_LEN];
    let (sk, pk) = key_gen(&seed);
    let cached = |sk: &LeanXmssExpandedSecretKey| {
        sk.cache_to_bytes()[CACHED_TREE_SIZE..]
            .chunks(CACHED_BOTTOM_SIZE)
            .map(|b| b[0])
            .collect::<Vec<_>>()
    };
    let size = DEFAULT_BOTTOM_TREE_CACHE_SIZE as u8 + 1;
    sk.set_bottom_tree_cache_size(size.into());
    for tree in 0..=size {
        sk.prepare(u32::from(tree) << LAYER_HEIGHT);
    }
    assert_eq!(cached(&sk), Vec::from_iter(1..=size));

    // A loaded key keeps every saved tree, even beyond the default size.
    let loaded = LeanXmssExpandedSecretKey::from_cache_bytes(&seed, &pk, &sk.cache_to_bytes()).unwrap();
    assert_eq!(loaded.cache_to_bytes(), sk.cache_to_bytes());

    // Using a tree makes it the most recently used; shrinking keeps the most recently used.
    sk.prepare(u32::from(size - 1) << LAYER_HEIGHT);
    sk.set_bottom_tree_cache_size(2);
    assert_eq!(cached(&sk), [size, size - 1]);
}

/// Pins a full key's public key and one signature's bytes, so that a change to the
/// seed derivation, the addresses, the hash inputs or the serialization cannot pass
/// unnoticed. An implementation of `doc/lean_xmss` written from the text alone gives
/// the same bytes.
#[test]
fn known_answer() {
    let hex = |bytes: &[u8]| bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let seed: SecretKey = std::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(11));
    let (sk, pk) = key_gen(&seed);
    let message: Message = std::array::from_fn(|i| (i * 7 + 3) as u8);
    let signature = sign(&sk, &message, 0xA5C3_7E19);
    assert_eq!(
        hex(&pk.to_bytes()),
        "4066673001949b6ff8b9299fd4ea17d28787ad501aafe4b64e2488107b5e3ec3"
    );
    assert_eq!(
        hex(&keccak256::hash(&signature.to_bytes())),
        "f10d62beec95e64e1477028252c2c8683695eab9d8787ad8fa722f47b5fd5234"
    );
}

/// The address layout, and `Th` as the leanVM hash of `P | A | M`.
#[test]
fn address_layout() {
    let tw = tweak(TWEAK_NODE, layer_hi(1, 0x12), 0x89abcdef).step(5);
    assert_eq!(address(&tw)[..8], [0xef, 0xcd, 0xab, 0x89, 0x12, 0x00, 0x80, 0xa3]);
    assert_eq!(address(&tw)[8..], [0; 8]);
    let pp: PublicParam = std::array::from_fn(|i| i as u8 + 1);
    let payload = [9u8; N];
    let input = [&pp[..], &address(&tw), &payload].concat();
    assert_eq!(th(&pp, &tw, &payload)[..], keccak256::hash(&input)[..N]);
}

/// Detect changes to the encoding predicate through its grinding cost.
#[test]
#[ignore]
fn encoding_grinding_bits() {
    let n = 1 << 26;
    let mut rng = StdRng::seed_from_u64(0);
    let hits = (0..n).filter(|_| codeword(&rng.random()).is_some()).count();
    let bits = (n as f64 / hits as f64).log2();
    println!("Average grinding bits: {bits:.1}");
    assert!((19.0..21.0).contains(&bits), "grinding cost moved: {bits:.2} bits");
}
