//! The thread pool signing and aggregation share: once per process, so this test binary holds one
//! test, which sets it up.

use lean_pq::{setup, xmss, Error};

/// Signing never starts the pool, and signs as the calling thread does with or without it;
/// `setup` then sizes the pool and accepts only that size again.
#[test]
fn signing_leaves_the_pool_to_setup() {
    let seed = [5; 32];
    let message = [7; 32];
    let key = xmss::SecretKey::from_seed(seed);
    let (expanded, _) = lean_xmss::key_gen(&seed);
    let calling_thread = |leaf_index| lean_xmss::sign(&expanded, &message, leaf_index).to_bytes();
    let leaf_index = (1 << 16) + 3;
    assert_eq!(
        key.sign(&message, leaf_index).to_bytes(),
        calling_thread(leaf_index)
    );
    // Signing started no pool, so any count is still available.
    setup(2).unwrap();
    assert_eq!(
        key.sign(&message, leaf_index + 1).to_bytes(),
        calling_thread(leaf_index + 1)
    );
    setup(2).unwrap();
    assert!(matches!(
        setup(1),
        Err(Error::SetupThreads {
            running: 2,
            threads: 1
        })
    ));
}
