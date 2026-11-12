//! Merkle openings: one full path per query.

use crate::field::K;
use crate::hash::{Digest, keccak_hash};

pub fn hash_pair(left: &[u8], right: &[u8]) -> Digest {
    let mut pair = [0u8; 64];
    pair[..32].copy_from_slice(left);
    pair[32..].copy_from_slice(right);
    keccak_hash(&pair)
}

fn hash_leaf(image: &[K]) -> Digest {
    keccak_hash(&image.iter().flat_map(|word| word.to_bytes()).collect::<Vec<u8>>())
}

/// One query's own opening, the form the Rust and Lean verifiers read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MerklePath {
    pub leaf_index: u64,
    /// The whole leaf image, its zero prefix included.
    pub leaf: Vec<K>,
    /// Bottom level first.
    pub siblings: Vec<[u8; 32]>,
}

impl MerklePath {
    pub(crate) fn root(&self) -> Digest {
        let mut node = hash_leaf(&self.leaf);
        for (level, sibling) in self.siblings.iter().enumerate() {
            node = if self.leaf_index >> level & 1 == 0 {
                hash_pair(&node.0, sibling)
            } else {
                hash_pair(sibling, &node.0)
            };
        }
        node
    }
}

/// Compare roots of paths whose heights have already been checked, sharing identical inputs between adjacent sorted paths.
pub(crate) fn roots_match(paths: &[MerklePath], root: Digest) -> bool {
    match paths {
        [] => return true,
        [path] => return path.root() == root,
        _ => {}
    }

    let mut nodes: Vec<_> = paths.iter().map(|path| (path, Digest([0; 32]))).collect();
    nodes.sort_unstable_by(|(a, _), (b, _)| {
        a.leaf_index
            .cmp(&b.leaf_index)
            .then_with(|| a.leaf.iter().map(|word| word.0).cmp(b.leaf.iter().map(|word| word.0)))
    });
    for i in 0..nodes.len() {
        let path = nodes[i].0;
        nodes[i].1 = if i > 0 && path.leaf_index == nodes[i - 1].0.leaf_index && path.leaf == nodes[i - 1].0.leaf {
            nodes[i - 1].1
        } else {
            hash_leaf(&path.leaf)
        };
    }

    for level in 0..paths[0].siblings.len() {
        let mut previous = None;
        for (path, node) in &mut nodes {
            let index = path.leaf_index >> level;
            let sibling = path.siblings[level];
            let (left, right) = if index & 1 == 0 {
                (node.0, sibling)
            } else {
                (sibling, node.0)
            };
            let input = (index >> 1, left, right);
            *node = match previous {
                Some((previous_input, digest)) if input == previous_input => digest,
                _ => hash_pair(&left, &right),
            };
            previous = Some((input, *node));
        }
    }
    nodes.iter().all(|(_, node)| *node == root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VerifierError;
    use crate::transcript::{Proof, Transcript};

    fn random(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    fn tree(seed: &mut u64, height: usize, leaf_words: usize) -> (Digest, Vec<MerklePath>) {
        let leaves: Vec<Vec<K>> = (0..1 << height)
            .map(|_| (0..leaf_words).map(|_| K(random(seed))).collect())
            .collect();
        let mut levels = vec![leaves.iter().map(|leaf| hash_leaf(leaf)).collect::<Vec<_>>()];
        for level in 0..height {
            levels.push(
                levels[level]
                    .chunks_exact(2)
                    .map(|pair| hash_pair(&pair[0].0, &pair[1].0))
                    .collect(),
            );
        }
        let root = levels[height][0];
        let paths: Vec<_> = leaves
            .into_iter()
            .enumerate()
            .map(|(index, leaf)| MerklePath {
                leaf_index: index as u64,
                leaf,
                siblings: (0..height).map(|level| levels[level][(index >> level) ^ 1].0).collect(),
            })
            .collect();
        assert!(paths.iter().all(|path| path.root() == root));
        (root, paths)
    }

    fn check_opening(paths: &[MerklePath], root: Digest, height: usize, leaf_words: usize) {
        let proof = Proof {
            stream: vec![],
            merkle_openings: vec![paths.to_vec()],
        };
        let queries: Vec<_> = paths.iter().map(|path| path.leaf_index).collect();
        let valid = paths.iter().all(|path| path.root() == root);
        for checked in [true, false] {
            let mut transcript = Transcript::new(&proof, Digest([0; 32]), Digest([0; 32]), checked);
            let expected = if !checked || valid {
                Ok(paths.iter().map(|path| path.leaf.as_slice()).collect::<Vec<_>>())
            } else {
                Err(VerifierError::MerkleRootMismatch)
            };
            assert_eq!(transcript.merkle(root, 1 << height, &queries, leaf_words), expected);
            assert_eq!(transcript.finish(), Ok(()));
        }
    }

    #[test]
    fn shared_roots_match_paths() {
        let mut seed = 0x9E3779B97F4A7C15;
        for height in 0..=6 {
            for _ in 0..12 {
                let leaf_words = 1 + random(&mut seed) as usize % 24;
                let (root, tree_paths) = tree(&mut seed, height, leaf_words);
                let mut paths: Vec<_> = (0..48)
                    .map(|_| tree_paths[random(&mut seed) as usize % tree_paths.len()].clone())
                    .collect();
                paths[1] = paths[0].clone();
                check_opening(&[], root, height, leaf_words);
                check_opening(&paths[..1], root, height, leaf_words);
                check_opening(&paths, root, height, leaf_words);
                paths.reverse();
                check_opening(&paths, root, height, leaf_words);

                let changed = random(&mut seed) as usize % paths.len();
                paths[changed].leaf[0].0 ^= 1;
                check_opening(&paths, root, height, leaf_words);
                paths[changed].leaf[0].0 ^= 1;
                paths[changed].leaf_index ^= 1;
                check_opening(&paths, root, height, leaf_words);
                paths[changed].leaf_index ^= 1;
                paths[changed].leaf_index ^= 1 << height;
                check_opening(&paths, root, height, leaf_words);
                paths[changed].leaf_index ^= 1 << height;
                for level in 0..height {
                    paths[changed].siblings[level][0] ^= 1;
                    check_opening(&paths, root, height, leaf_words);
                    paths[changed].siblings[level][0] ^= 1;
                }
                let mut wrong_root = root;
                wrong_root.0[0] ^= 1;
                check_opening(&paths, wrong_root, height, leaf_words);
            }
        }
    }

    #[test]
    fn conflicting_shared_sibling() {
        let (root, mut paths) = tree(&mut 0xD1B54A32D192ED03, 3, 2);
        paths.truncate(2);
        paths[1].siblings[0][0] ^= 1;
        assert_eq!(paths[0].root(), root);
        assert_ne!(paths[1].root(), root);
        check_opening(&paths, root, 3, 2);
    }

    #[test]
    fn shapes_and_positions_are_checked_before_roots() {
        let (root, paths) = tree(&mut 0x94D049BB133111EB, 1, 2);
        let mut proof = Proof {
            stream: vec![],
            merkle_openings: vec![paths],
        };
        proof.merkle_openings[0][0].leaf_index ^= 1;
        proof.merkle_openings[0][1].siblings[0][0] ^= 1;
        for checked in [true, false] {
            proof.merkle_openings[0][1].leaf.push(K(0));
            let mut transcript = Transcript::new(&proof, Digest([0; 32]), Digest([0; 32]), checked);
            assert_eq!(
                transcript.merkle(root, 2, &[0, 1], 2),
                Err(VerifierError::MerkleOpeningShape)
            );
            proof.merkle_openings[0][1].leaf.pop();
            let mut transcript = Transcript::new(&proof, Digest([0; 32]), Digest([0; 32]), checked);
            assert_eq!(
                transcript.merkle(root, 2, &[0, 1], 2),
                Err(VerifierError::MerkleOpeningPosition)
            );
        }
    }
}
