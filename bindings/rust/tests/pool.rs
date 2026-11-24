//! The thread pool signing and aggregation share: once per process, so this test binary holds one
//! test, which sets it up.

use lean_pq::{setup, sphincs, xmss, Error};

/// Key generation, preparation and signing of both schemes never start the pool, and make what
/// the calling thread makes, before `setup` and on the pool after it; `setup` then accepts only
/// the size it set.
#[test]
fn key_generation_and_signing_leave_the_pool_to_setup() {
    let message = [7; 32];
    let check = |seed: [u8; 32], leaf_index: u32| {
        let key = xmss::SecretKey::from_seed(seed);
        let (expanded, public_key) = lean_xmss::key_gen(&seed);
        assert_eq!(key.public_key().to_bytes(), public_key.to_bytes());
        key.prepare(leaf_index);
        expanded.prepare(leaf_index);
        assert_eq!(key.cache_to_bytes(), expanded.cache_to_bytes());
        for leaf_index in [leaf_index, leaf_index + 1, (leaf_index ^ (1 << 16)) + 2] {
            let signature = key.sign(&message, leaf_index).to_bytes();
            assert_eq!(
                signature,
                lean_xmss::sign(&expanded, &message, leaf_index).to_bytes()
            );
        }
        let key = sphincs::SecretKey::from_seed(seed);
        assert_eq!(
            key.public_key().to_bytes(),
            lean_sphincs::key_gen(&seed).to_bytes()
        );
        let signature = key.sign(&message).to_bytes();
        assert_eq!(signature, lean_sphincs::sign(&seed, &message).to_bytes());
    };
    check([5; 32], (1 << 16) + 3);
    // None of that started the pool, so any count is still available.
    setup(2).unwrap();
    check([6; 32], (3 << 16) + 300);
    setup(2).unwrap();
    assert!(matches!(
        setup(1),
        Err(Error::SetupThreads {
            running: 2,
            threads: 1
        })
    ));
}
