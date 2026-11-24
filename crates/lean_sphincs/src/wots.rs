//! The one-time signature: `v` hash chains of `2^w - 1` steps, and the
//! target-sum code that replaces the Winternitz checksum (WOTS+C), hashed as
//! FIPS 205's WOTS+ is (§5).
//!
//! A codeword is `v` chunks summing to `T`. Two distinct words of equal sum
//! cannot be ordered componentwise, so revealing chain position `x_i` on every
//! chain gives a forger nothing: any other codeword needs a position below one
//! of the revealed ones, a preimage of a chain step. The price is that most
//! messages do not encode into the code at all, hence the counter the signer
//! searches for and the signature carries.

use crate::*;

/// A one-time key of the hypertree: its layer, counted from the bottom, its tree
/// on that layer, and its leaf in that tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OtsKey {
    pub layer: usize,
    pub tree: u64,
    pub leaf: u32,
}

impl OtsKey {
    /// The key of `layer` on the way from bottom leaf `idx` up: the bits below
    /// the layer dropped, the rest split into a tree and a leaf.
    pub fn of_index(layer: usize, idx: u64) -> Self {
        let height = LAYER_HEIGHTS[layer];
        let e = idx >> LAYER_HEIGHTS[..layer].iter().sum::<usize>();
        Self {
            layer,
            tree: e >> height,
            leaf: (e & ((1 << height) - 1)) as u32,
        }
    }

    /// The address of the key's tree, type `WOTS_HASH`.
    pub fn tree_adrs(&self) -> Adrs {
        Adrs::new(self.layer as u32, self.tree)
    }

    /// The key's address of type `t`: its tree's, with its leaf as the key pair.
    pub fn adrs(&self, t: u32) -> Adrs {
        self.tree_adrs().with_type(t).with_key_pair(self.leaf)
    }
}

/// The starts of the `v` chains of a one-time key, one `PRF` each.
fn wots_secrets(sk: &ExpandedKey, key: &OtsKey) -> [Digest; V] {
    core::array::from_fn(|i| prf(&sk.pk_seed, &sk.sk_seed, &prf_adrs(key, i)))
}

/// The address of the `PRF` that starts chain `i` of `key`.
fn prf_adrs(key: &OtsKey, i: usize) -> Adrs {
    key.adrs(WOTS_PRF).with_chain(i as u32)
}

/// `chain`: `steps` steps of chain `i` from position `start`; a step's hash
/// address is the position it leaves.
fn chain(pk_seed: &PkSeed, key: &OtsKey, i: usize, start: usize, steps: usize, value: Digest) -> Digest {
    debug_assert!(start + steps < CHAIN_LEN);
    (start..start + steps).fold(value, |current, from| th(pk_seed, &chain_adrs(key, i, from), &current))
}

/// The address of the step of chain `i` of `key` that leaves position `from`.
fn chain_adrs(key: &OtsKey, i: usize, from: usize) -> Adrs {
    key.adrs(WOTS_HASH).with_chain(i as u32).with_hash(from as u32)
}

/// The Merkle leaf of a one-time key, its public key: `T_v` over its chain ends.
fn wots_leaf_hash(pk_seed: &PkSeed, key: &OtsKey, ends: &[Digest; V]) -> Digest {
    th_digests(pk_seed, &key.adrs(WOTS_PK), ends)
}

/// The hash of `Enc(K, M, c)` before its counter: `PK.seed | ADRS_enc(K) | M`.
fn encoder(pk_seed: &PkSeed, key: &OtsKey, message: &Digest) -> keccak256::Hasher {
    let mut hasher = hasher(pk_seed, &key.adrs(WOTS_ENC));
    hasher.update(message);
    hasher
}

/// `Enc(K, M, c)`: `Trunc_n(H(PK.seed | ADRS_enc(K) | M | BE_32(c)))` as a codeword,
/// or `None` if this counter's digest is not one.
fn encode(encoder: &keccak256::Hasher, c: u32) -> Option<[u8; V]> {
    codeword(&keccak256::truncate(
        encoder.clone().update(&c.to_be_bytes()).finalize(),
    ))
}

/// `base_2b(D, w, v)` (FIPS 205 Algorithm 4): the digest as `v` chunks of `w` bits,
/// most significant first; a codeword if they sum to `T`.
fn codeword(digest: &Digest) -> Option<[u8; V]> {
    let d = u128::from_be_bytes(*digest);
    let x: [u8; V] = core::array::from_fn(|i| ((d >> (W * (V - 1 - i))) & (CHAIN_LEN as u128 - 1)) as u8);
    (x.iter().map(|&chunk| chunk as usize).sum::<usize>() == TARGET_SUM).then_some(x)
}

/// Whether the digest's `w`-bit chunks sum to `T`, the digest's bytes read in either
/// order: a chunk's two bits stay together under a byte swap.
fn sums_to_target(d: u128) -> bool {
    const _: () = assert!(W == 2);
    const LOW: u128 = u128::MAX / 3;
    (d & LOW).count_ones() + 2 * (d & !LOW).count_ones() == TARGET_SUM as u32
}

/// Counters tried at once.
const SEARCH_BATCH: usize = 64;

/// The least counter below `MAX_ENCODING_ATTEMPTS` for which `Enc(K, M, c)` is a
/// codeword. The counter's four bytes start the final block's lane 8 of a one-block
/// input, so the input's state is built once and a counter costs one permutation.
fn least_counter(pk_seed: &PkSeed, key: &OtsKey, message: &Digest) -> Option<u32> {
    const AT: usize = PREFIX_LEN + N;
    const _: () = assert!(AT.is_multiple_of(8) && AT + COUNTER_LEN < keccak256::RATE);
    let mut input = [0u8; AT + COUNTER_LEN];
    input[..N].copy_from_slice(pk_seed);
    input[N..PREFIX_LEN].copy_from_slice(key.adrs(WOTS_ENC).as_bytes());
    input[PREFIX_LEN..AT].copy_from_slice(message);
    let template = keccak256::final_state(&input);
    let (mut values, mut digests) = ([0; SEARCH_BATCH], [0; SEARCH_BATCH]);
    (0..MAX_ENCODING_ATTEMPTS).step_by(SEARCH_BATCH).find_map(|first| {
        let n = (MAX_ENCODING_ATTEMPTS - first).min(SEARCH_BATCH as u64) as usize;
        for (c, value) in (first..).zip(&mut values[..n]) {
            // `BE_32(c)` as the low bytes of a little-endian lane.
            *value = u128::from((c as u32).swap_bytes());
        }
        keccak256::digest_halves_from(&template, AT / 8, &values[..n], &mut digests[..n]);
        let hit = digests[..n].iter().position(|&d| sums_to_target(d))?;
        Some((first + hit as u64) as u32)
    })
}

/// `Wots.sign`: the LEAST admissible counter, and the chain value each chunk
/// opens. Deterministic in its inputs, which is what keeps one key to one
/// codeword per message: a resumed or randomized search would leak two
/// incomparable codewords.
pub(crate) fn wots_sign(sk: &ExpandedKey, key: &OtsKey, message: &Digest) -> Option<(u32, [Digest; V])> {
    let pk_seed = &sk.pk_seed;
    let c = least_counter(pk_seed, key, message)?;
    let x = encode(&encoder(pk_seed, key, message), c)?;
    let secrets = wots_secrets(sk, key);
    let signature = core::array::from_fn(|i| chain(pk_seed, key, i, 0, x[i] as usize, secrets[i]));
    Some((c, signature))
}

/// `Wots.leaf`: the leaf a claimed signature recovers, and the codeword it opens
/// at, or `None` if its counter is not admissible for the message.
pub fn wots_leaf(
    pk_seed: &PkSeed,
    key: &OtsKey,
    message: &Digest,
    c: u32,
    signature: &[Digest; V],
) -> Option<([u8; V], Digest)> {
    let x = encode(&encoder(pk_seed, key, message), c)?;
    let ends = core::array::from_fn(|i| {
        let start = x[i] as usize;
        chain(pk_seed, key, i, start, CHAIN_LEN - 1 - start, signature[i])
    });
    Some((x, wots_leaf_hash(pk_seed, key, &ends)))
}

/// `wots_pkGen`: the leaf of a one-time key, from the secret key, what
/// [`wots_public_leaves`] computes in batches.
#[cfg(all(test, feature = "std"))]
pub(crate) fn wots_public_leaf(sk: &ExpandedKey, key: &OtsKey) -> Digest {
    let secrets = wots_secrets(sk, key);
    let ends = core::array::from_fn(|i| chain(&sk.pk_seed, key, i, 0, CHAIN_LEN - 1, secrets[i]));
    wots_leaf_hash(&sk.pk_seed, key, &ends)
}

/// The leaves of the `n` keys of `key`'s tree from `key.leaf` on, from the secret key: what
/// building a tree spends its hashes on. As [`wots_secrets`], [`chain`] and [`wots_leaf_hash`]
/// compute them, each stage one batch over every key: the secrets, each chain step, then the
/// leaves.
#[cfg(feature = "std")]
pub(crate) fn wots_public_leaves(sk: &ExpandedKey, key: &OtsKey, n: usize) -> Vec<Digest> {
    let key_of = |k: usize| OtsKey {
        leaf: key.leaf + k as u32,
        ..*key
    };
    let seed: [Digest; 2] = *sk.sk_seed.as_chunks().0.as_array().unwrap();
    // Chain `i` of the `k`th key at `k * V + i`.
    let prf = (0..n * V).map(|k| (prf_adrs(&key_of(k / V), k % V), seed));
    let mut values = th_digests_many(&sk.pk_seed, prf);
    for from in 0..CHAIN_LEN - 1 {
        let steps = values.iter().enumerate();
        let steps = steps.map(|(k, &value)| (chain_adrs(&key_of(k / V), k % V, from), [value]));
        values = th_digests_many(&sk.pk_seed, steps);
    }
    let ends = values.as_chunks::<V>().0.iter().enumerate();
    th_digests_many(&sk.pk_seed, ends.map(|(k, &ends)| (key_of(k).adrs(WOTS_PK), ends)))
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// Batched leaves equal the one-key computation: one key, an unaligned first key with a
    /// count of no whole number of lanes, and the last keys of a tree, on every layer.
    #[test]
    fn batched_leaves_match_one_key() {
        let sk = ExpandedKey::new(&[7; 32]);
        for layer in 0..LAYERS {
            let last = 1 << LAYER_HEIGHTS[layer];
            for (leaf, n) in [(3, 1), (5, 18), (last - 5, 5)] {
                let key = OtsKey { layer, tree: 5, leaf };
                let leaves = wots_public_leaves(&sk, &key, n);
                assert_eq!(leaves.len(), n);
                for (leaf, got) in (key.leaf..).zip(&leaves) {
                    let want = wots_public_leaf(&sk, &OtsKey { leaf, ..key });
                    assert_eq!(*got, want, "layer {layer}, leaf {leaf}");
                }
            }
        }
    }

    /// The batched search finds the counter the one-at-a-time search over `encode` does.
    #[test]
    fn least_counter_matches_encode() {
        let sk = ExpandedKey::new(&[9; 32]);
        for leaf in 0..4 {
            let key = OtsKey {
                layer: 0,
                tree: 1,
                leaf,
            };
            let message = [leaf as u8; N];
            let encoder = encoder(&sk.pk_seed, &key, &message);
            let want = (0..u32::MAX).find(|&c| encode(&encoder, c).is_some());
            assert_eq!(least_counter(&sk.pk_seed, &key, &message), want, "leaf {leaf}");
        }
    }
}
