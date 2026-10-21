//! Haraka-512 batch proving over a GF(2^8)-coefficient R1CS.
//!
//! The AES S-box is a field inverse, so with GF(2^8) coefficients in the
//! constraint matrices each of the 640 S-boxes of a call costs two degree-2
//! rows and eight committed bits, against 32 AND gates for the same S-box over
//! GF(2). A call commits 5,888 bits and 1,536 rows ([`circuit`]); the
//! univariate-skip zerocheck, the lincheck's sumcheck, the ring switch and the
//! PCS are shared with [`crate::hash`], so the transcript shape is BLAKE2s's
//! with a rectangular block.
//!
//! What this proves: every call's S-boxes are inverted correctly and its
//! output bits are the truncated feed-forward of the final state. Inputs and
//! outputs are free witness bits at fixed slots; binding them is the embedder's
//! job, as for BLAKE2s.

pub mod circuit;
pub mod kernels;
pub mod lincheck;
pub mod primitive;
pub mod witness;
pub mod zerocheck;

pub use circuit::{CALL_BITS, CALLS_PER_SLOT, CONST_COL, K_LOG, ROWS_LOG, USEFUL_BITS};
pub use primitive::{IN_LEN, OUT_LEN, haraka512};
pub use witness::{Input, PADDING_INPUT, Witness, generate};

use crate::hash::SliceClaim;
use crate::lincheck::QuirkyPoint;
use crate::zerocheck::K_SKIP;
use fiat_shamir::transcript::ProverState;
use pcs::pack::{LOG_PACKING, PACKING_WIDTH};
use pcs::stack_open::{RingSwitchClaim, RingSwitchOpen};

/// Slots for `n_calls` calls: at least 8 (one lincheck stripe).
pub fn min_n_log(n_calls: usize) -> usize {
    let slots = n_calls.max(1).div_ceil(CALLS_PER_SLOT).max(8);
    slots.next_power_of_two().trailing_zeros() as usize
}

/// The variable count of the committed column for `n_calls` calls.
pub fn qflock_kappa(n_calls: usize) -> usize {
    K_LOG + min_n_log(n_calls) - LOG_PACKING
}

/// The zerocheck's output: the quirky point lincheck runs at.
#[derive(Clone, Debug)]
pub struct ZerocheckStage {
    x_ab: QuirkyPoint,
    claim: crate::zerocheck::ZerocheckClaim,
}

#[derive(Clone, Debug)]
pub struct HarakaSetup {
    n_log: usize,
    mats: circuit::SlotMatrices,
}

impl std::fmt::Debug for circuit::SlotMatrices {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SlotMatrices({} rows)", self.a.len())
    }
}

impl Clone for circuit::SlotMatrices {
    fn clone(&self) -> Self {
        circuit::SlotMatrices {
            a: self.a.clone(),
            b: self.b.clone(),
            c: self.c.clone(),
        }
    }
}

impl HarakaSetup {
    pub fn new(n_calls: usize) -> Self {
        Self {
            n_log: min_n_log(n_calls),
            mats: circuit::SlotMatrices::build(),
        }
    }
    pub fn n_log(&self) -> usize {
        self.n_log
    }
    /// Zerocheck dimension: `log2` of the rows.
    pub fn m_rows(&self) -> usize {
        ROWS_LOG + self.n_log
    }
    /// `log2` of the committed bits.
    pub fn m_bits(&self) -> usize {
        K_LOG + self.n_log
    }
    pub fn generate_witness(&self, inputs: &[Input]) -> Witness {
        witness::generate(inputs, self.n_log)
    }
    pub fn prove_zerocheck(&self, w: &Witness, ps: &mut ProverState) -> ZerocheckStage {
        let zc = zerocheck::prove(&w.c, self.n_log, ps);
        let inner = ROWS_LOG - K_SKIP;
        ZerocheckStage {
            x_ab: QuirkyPoint {
                z_skip: zc.z,
                x_inner_rest: zc.mlv_challenges[..inner].to_vec(),
                x_outer: zc.mlv_challenges[inner..].to_vec(),
            },
            claim: zc,
        }
    }
    pub fn prove_lincheck(&self, stage: ZerocheckStage, z_lincheck: &[u8], ps: &mut ProverState) -> SliceClaim {
        let lc = lincheck::prove(&self.mats, z_lincheck, self.n_log, &stage.x_ab, &stage.claim, ps);
        let mut suffix_point = lc.r_inner_rest;
        suffix_point.extend_from_slice(&stage.x_ab.x_outer);
        SliceClaim {
            suffix_point,
            s_hat_v: lc.s_hat_v,
        }
    }
}

/// The reduction claim as a ring-switch opening at `offset` in the committed stack.
pub fn ring_switch_open(n_calls: usize, offset: usize, reduced: &SliceClaim) -> RingSwitchOpen {
    let qflock_vars = qflock_kappa(n_calls);
    assert_eq!(reduced.suffix_point.len(), qflock_vars);
    assert_eq!(reduced.s_hat_v.len(), PACKING_WIDTH);
    RingSwitchOpen {
        offset,
        qflock_vars,
        claims: vec![RingSwitchClaim {
            suffix_point: reduced.suffix_point.clone(),
            s_hat_v: Some(reduced.s_hat_v.clone()),
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The wire proof in the `verifier` crate's types.
    fn scalar_proof(proof: &fiat_shamir::transcript::Proof) -> verifier::Proof {
        let batches = proof.merkle.iter().map(|paths| verifier::PrunedMerklePaths {
            leaf_data: paths
                .leaf_data
                .iter()
                .map(|row| row.iter().map(|word| verifier::K(word.0)).collect())
                .collect(),
            sibling_hashes: paths.sibling_hashes.clone(),
        });
        verifier::Proof {
            stream: proof
                .stream
                .iter()
                .map(|x| verifier::E::new(x.c0, x.c1, x.c2))
                .collect(),
            merkle_openings: verifier::MerkleOpenings::Pruned(batches.collect()),
        }
    }
    use circuit::*;
    use primitives::field::F8;
    use primitives::test_rng::Rng;

    fn inputs(n: usize, seed: u64) -> Vec<Input> {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|_| std::array::from_fn(|_| (rng.next_u64() & 0xff) as u8))
            .collect()
    }

    #[test]
    fn witness_satisfies_the_circuit() {
        let setup = HarakaSetup::new(3 * CALLS_PER_SLOT + 1);
        let w = setup.generate_witness(&inputs(3 * CALLS_PER_SLOT + 1, 5));
        let sq = |v: u8| (F8(v) * F8(v)).0;
        for slot in [0, 3, 7] {
            let zf = |col: usize| witness::z_bit(&w, slot, col);
            for (r, (ra, (rb, rc))) in setup
                .mats
                .a
                .iter()
                .zip(setup.mats.b.iter().zip(&setup.mats.c))
                .enumerate()
            {
                let ev = |m: &Vec<(u32, u8)>| {
                    m.iter()
                        .fold(0u8, |acc, &(c, g)| if zf(c as usize) { acc ^ g } else { acc })
                };
                let (a, b, c) = (ev(ra), ev(rb), ev(rc));
                let row = slot * ROWS_PER_SLOT + r;
                let cv = w.c[row];
                let col = r / COL_ROWS;
                let want_b = if is_lin_col(col) { 1 } else { w.c[row ^ 1] };
                assert_eq!((a, b, c), (sq(cv), want_b, cv), "slot {slot} row {r}");
                assert_eq!((F8(a) * F8(b)).0, c, "slot {slot} row {r}");
            }
            assert!(zf(CONST_COL));
        }
    }

    #[test]
    fn verifier_walk_matches_the_matrices() {
        let mats = circuit::SlotMatrices::build();
        let mut rng = Rng::new(3);
        let w: Vec<primitives::field::F192> = (0..K)
            .map(|_| primitives::field::F192::new(rng.next_u64(), rng.next_u64(), rng.next_u64()))
            .collect();
        let we: Vec<verifier::E> = w.iter().map(|x| verifier::E::new(x.c0, x.c1, x.c2)).collect();
        let [va, vb, vc] = verifier::haraka_row_values(&we);
        let eval = |m: &Vec<(u32, u8)>| {
            m.iter().fold(primitives::field::F192::ZERO, |acc, &(c, g)| {
                acc + w[c as usize] * primitives::field::phi8_192(F8(g))
            })
        };
        for (r, ((ra, rb), rc)) in mats.a.iter().zip(&mats.b).zip(&mats.c).enumerate() {
            let want = |m| {
                let v = eval(m);
                verifier::E::new(v.c0, v.c1, v.c2)
            };
            assert_eq!(va[r], want(ra), "A row {r}");
            assert_eq!(vb[r], want(rb), "B row {r}");
            assert_eq!(vc[r], want(rc), "C row {r}");
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni"
    ))]
    #[test]
    fn gfni_round1_and_fold_match_scalar() {
        use kernels::gfni::{Acc24, Consts, convert8, fold_tile, round1_tile};
        use primitives::field::F192;
        let setup = HarakaSetup::new(16);
        let w = setup.generate_witness(&inputs(16, 4));
        let mut rng = Rng::new(21);
        let eq: Vec<F192> = (0..COLS_PER_SLOT)
            .map(|_| F192::new(rng.next_u64(), rng.next_u64(), rng.next_u64()))
            .collect();
        let m = kernels::extension_matrix();
        let (sa, sd) = kernels::round1_scalar(&w.c, &eq, 0..1, USED_COLS, COLS_PER_SLOT, &m);
        let k = Consts::new();
        let mut acc_ab = Acc24::zero();
        let mut acc_c = Acc24::zero();
        let mut eq8 = vec![0u8; kernels::COORDS * USED_COLS];
        unsafe {
            for t in (0..USED_COLS).step_by(8) {
                let inp: &[u8; 8 * kernels::COORDS] =
                    std::slice::from_raw_parts(eq[t..].as_ptr() as *const u8, 8 * kernels::COORDS)
                        .try_into()
                        .unwrap();
                let out: &mut [u8; 8 * kernels::COORDS] = (&mut eq8[kernels::COORDS * t..kernels::COORDS * (t + 8)])
                    .try_into()
                    .unwrap();
                convert8(&k.basis.to_coords, inp, out);
            }
            round1_tile(
                &k,
                w.c.as_ptr(),
                eq8.as_ptr(),
                USED_COLS,
                is_lin_col,
                &mut acc_ab,
                &mut acc_c,
            );
        }
        assert_eq!(acc_c.finish(&k.basis), sd, "round-1 D[s]");
        assert_eq!(acc_ab.finish(&k.basis), sa, "round-1 P^AB");
        let wts: Vec<F192> = (0..64)
            .map(|_| F192::new(rng.next_u64(), rng.next_u64(), rng.next_u64()))
            .collect();
        let mut ra = vec![F192::ZERO; COLS_PER_SLOT];
        let mut rb = vec![F192::ZERO; COLS_PER_SLOT];
        kernels::fold_scalar(&w.c, &wts, 0, USED_COLS, COLS_PER_SLOT, &mut ra, &mut rb);
        let wa: Box<[[u8; kernels::COORDS]; 64]> = Box::new(std::array::from_fn(|s| k.basis.to_coords(wts[s])));
        let wb: Box<[[u8; kernels::COORDS]; 64]> = Box::new(std::array::from_fn(|s| wa[s ^ 1]));
        let wsum = k.basis.to_coords(wts.iter().fold(F192::ZERO, |acc, &x| acc + x));
        let mut ga = vec![F192::ZERO; 64];
        let mut gb = vec![F192::ZERO; 64];
        let mut gc = vec![F192::ZERO; 64];
        for tile in 0..COLS_PER_SLOT / 64 {
            let lin_mask = (0..64).fold(0u64, |mm, j| if is_lin_col(64 * tile + j) { mm | 1 << j } else { mm });
            unsafe {
                fold_tile(
                    &k,
                    w.c.as_ptr().add(64 * tile * COL_ROWS),
                    lin_mask,
                    &wa,
                    &wb,
                    &wsum,
                    ga.as_mut_ptr(),
                    gb.as_mut_ptr(),
                    gc.as_mut_ptr(),
                )
            };
            assert_eq!(ga, ra[64 * tile..64 * tile + 64], "fold a, tile {tile}");
            assert_eq!(gb, rb[64 * tile..64 * tile + 64], "fold b, tile {tile}");
            for x in 0..64 {
                let col = 64 * tile + x;
                let want = (0..64).fold(F192::ZERO, |acc, s| {
                    acc + wts[s] * primitives::field::phi8_192(F8(w.c[col * COL_ROWS + s]))
                });
                assert_eq!(gc[x], want, "fold c, column {col}");
            }
        }
    }

    #[test]
    fn prove_and_verify() {
        use fiat_shamir::digest_words;
        use pcs::stack_open::open_batch_mixed_whir_stacked;
        use pcs::whir::{INITIAL_FOLDING_FACTOR, LOG_INV_RATE_0, commit, config_for_rate};
        use primitives::field::F64;
        let n = 25 * CALLS_PER_SLOT + 3;
        let setup = HarakaSetup::new(n);
        let n_log = setup.n_log();
        let mu = setup.m_bits() - LOG_PACKING;
        let w = setup.generate_witness(&inputs(n, 9));
        // SAFETY: `F64` is `repr(transparent)` over `u64`.
        let q_flock: &[F64] = unsafe { std::slice::from_raw_parts(w.z.as_ptr().cast(), w.z.len()) };
        let seed = [[1u8; 32], [2u8; 32]];
        let mut ps = ProverState::new(digest_words(&seed[0]), digest_words(&seed[1]));
        let (commitment, prover_data) = commit(q_flock, mu, INITIAL_FOLDING_FACTOR, LOG_INV_RATE_0);
        ps.add_root(&commitment.root);
        let stage = setup.prove_zerocheck(&w, &mut ps);
        let reduced = setup.prove_lincheck(stage, &w.z_lincheck, &mut ps);
        let ring = ring_switch_open(n, 0, &reduced);
        let config = config_for_rate(mu, LOG_INV_RATE_0).unwrap();
        open_batch_mixed_whir_stacked(&mut ps, mu, q_flock, &prover_data, &config, &[], &ring);
        let proof = ps.into_proof();
        let proof = scalar_proof(&proof);
        verifier::verify_flock_batch_haraka(&seed[0], &seed[1], ROWS_LOG + n_log, LOG_INV_RATE_0, &proof)
            .expect("verifies");
    }
}
