//! The zerocheck on byte-valued rows: `a(y) · b(y) + c(y) = 0` where the three
//! row values are GF(2^8) elements rather than bits. Round 1 extends each
//! 64-row column from `S` to `Λ` over GF(2^8) exactly as the bit zerocheck
//! does, so the transcript and the verifier are those of [`crate::zerocheck`];
//! only the prover's kernels differ. After round 1 the three tables are folded
//! at `z` into `F192` and the remaining rounds are the shared table rounds.

use super::circuit::*;
use super::kernels::{self, extension_matrix};
use crate::zerocheck::multilinear::{
    fold_and_compute_round_pair_into, fold_and_compute_round_single_into, fold_in_place_pair, fold_in_place_single,
    interpolate_at_z_combined, round_pair_naive, round_single_naive,
};
use crate::zerocheck::{K_SKIP, ZerocheckClaim, equality_tail, send_round};
use fiat_shamir::transcript::ProverState;
use primitives::field::{F8, F192, phi8_192};
use primitives::multilinear::{eq_table, lagrange_weights_naive};

fn add_vec(x: &mut [F192], y: &[F192]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += *b;
    }
}

/// The round-1 message in the naive convention: `P(λ) = Σ_x eq(x) (â b̂ + ĉ)(λ, x)` on `Λ`.
fn round1(c: &[u8], eq: &[F192], n_slots: usize) -> Vec<F192> {
    let m = extension_matrix();
    let n_ranges = n_slots.min(128);
    let per = n_slots.div_ceil(n_ranges);
    let (p_ab, d) = parallel::map_reduce(
        n_ranges,
        || (vec![F192::ZERO; 64], vec![F192::ZERO; 64]),
        |i| {
            let slots = i * per..((i + 1) * per).min(n_slots);
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "avx512bw",
                target_feature = "avx512vbmi",
                target_feature = "gfni"
            ))]
            {
                round1_gfni(c, eq, slots)
            }
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "avx512bw",
                target_feature = "avx512vbmi",
                target_feature = "gfni"
            )))]
            {
                kernels::round1_scalar(c, eq, slots, USED_COLS, COLS_PER_SLOT, &m)
            }
        },
        |mut x, y| {
            add_vec(&mut x.0, &y.0);
            add_vec(&mut x.1, &y.1);
            x
        },
    );
    // `ĉ` is linear in the rows, so its message is the extension of `D[s] = Σ_x eq(x) φ8(c(s, x))`.
    (0..64)
        .map(|lam| p_ab[lam] + (0..64).fold(F192::ZERO, |acc, s| acc + phi8_192(F8(m[s][lam])) * d[s]))
        .collect()
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
fn round1_gfni(c: &[u8], eq: &[F192], slots: std::ops::Range<usize>) -> (Vec<F192>, Vec<F192>) {
    use kernels::gfni::{Acc24, Consts, convert8, round1_tile};
    use std::sync::OnceLock;
    static CONSTS: OnceLock<Consts> = OnceLock::new();
    let k = CONSTS.get_or_init(Consts::new);
    let mut acc_ab = Acc24::zero();
    let mut acc_c = Acc24::zero();
    let mut eq8 = vec![0u8; kernels::COORDS * USED_COLS];
    // SAFETY: the target features are enabled at compile time; every pointer stays inside its slice.
    unsafe {
        for slot in slots {
            let x0 = slot * COLS_PER_SLOT;
            for t in (0..USED_COLS).step_by(8) {
                let inp: &[u8; 8 * kernels::COORDS] =
                    std::slice::from_raw_parts(eq[x0 + t..].as_ptr() as *const u8, 8 * kernels::COORDS)
                        .try_into()
                        .unwrap();
                let out: &mut [u8; 8 * kernels::COORDS] = (&mut eq8[kernels::COORDS * t..kernels::COORDS * (t + 8)])
                    .try_into()
                    .unwrap();
                convert8(&k.basis.to_coords, inp, out);
            }
            round1_tile(
                k,
                c.as_ptr().add(x0 * COL_ROWS),
                eq8.as_ptr(),
                USED_COLS,
                is_lin_col,
                &mut acc_ab,
                &mut acc_c,
            );
        }
    }
    (acc_ab.finish(&k.basis), acc_c.finish(&k.basis))
}

/// Fold the byte rows at `z`: the `F192` tables of `â`, `b̂` and `ĉ` over the rest variables, and the
/// first multilinear round's message `(G(1), G(∞))` with `r_eq` the eq challenges it does not bind.
fn fold_at_z(c: &[u8], w: &[F192], r_eq: &[F192], n_slots: usize) -> ([Vec<F192>; 3], (F192, F192)) {
    const COL_BITS: usize = (COLS_PER_SLOT / 2).trailing_zeros() as usize;
    assert_eq!(r_eq.len(), COL_BITS + n_slots.trailing_zeros() as usize);
    let eq_lo = eq_table(&r_eq[..COL_BITS]);
    let eq_hi = eq_table(&r_eq[COL_BITS..]);
    let n = n_slots * COLS_PER_SLOT;
    // SAFETY (x3): every slot of every table is written below, one slot per task.
    let mut am = unsafe { primitives::uninit_vec::<F192>(n) };
    let mut bm = unsafe { primitives::uninit_vec::<F192>(n) };
    let mut cm = unsafe { primitives::uninit_vec::<F192>(n) };
    let ac = parallel::Chunks::new(&mut am, COLS_PER_SLOT);
    let bc = parallel::Chunks::new(&mut bm, COLS_PER_SLOT);
    let cc = parallel::Chunks::new(&mut cm, COLS_PER_SLOT);
    let message = |oa: &[F192], ob: &[F192], oc: &[F192]| {
        let (mut g1, mut gi) = (F192::ZERO, F192::ZERO);
        for (p, &e) in eq_lo.iter().enumerate() {
            let (a0, a1, b0, b1) = (oa[2 * p], oa[2 * p + 1], ob[2 * p], ob[2 * p + 1]);
            g1 += e * (a1 * b1 + oc[2 * p + 1]);
            gi += e * ((a0 + a1) * (b0 + b1));
        }
        (g1, gi)
    };
    let msg = parallel::map_reduce(
        n_slots,
        || (F192::ZERO, F192::ZERO),
        |slot| {
            // SAFETY: one task per slot, each chunk taken once.
            let (oa, ob, oc) = unsafe { (ac.get(slot), bc.get(slot), cc.get(slot)) };
            fold_slot(c, w, slot, oa, ob, oc);
            let (g1, gi) = message(oa, ob, oc);
            (eq_hi[slot] * g1, eq_hi[slot] * gi)
        },
        |x, y| (x.0 + y.0, x.1 + y.1),
    );
    ([am, bm, cm], msg)
}

/// One slot's three tables from its byte rows.
fn fold_slot(c: &[u8], w: &[F192], slot: usize, oa: &mut [F192], ob: &mut [F192], oc: &mut [F192]) {
    {
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vbmi",
            target_feature = "gfni"
        ))]
        {
            use std::sync::OnceLock;
            static CONSTS: OnceLock<kernels::gfni::Consts> = OnceLock::new();
            let k = CONSTS.get_or_init(kernels::gfni::Consts::new);
            let wa: Box<[[u8; kernels::COORDS]; 64]> = Box::new(std::array::from_fn(|s| k.basis.to_coords(w[s])));
            let wb: Box<[[u8; kernels::COORDS]; 64]> = Box::new(std::array::from_fn(|s| wa[s ^ 1]));
            let wsum = k.basis.to_coords(w.iter().fold(F192::ZERO, |acc, &x| acc + x));
            let (wa, wb, wsum) = (&wa, &wb, &wsum);
            for tile in 0..COLS_PER_SLOT / 64 {
                let col0 = tile * 64;
                let lin_mask = (0..64).fold(0u64, |m, j| if is_lin_col(col0 + j) { m | 1 << j } else { m });
                // SAFETY: the target features are enabled at compile time; a tile is 64 whole columns of the slot.
                unsafe {
                    kernels::gfni::fold_tile(
                        k,
                        c.as_ptr().add((slot * COLS_PER_SLOT + col0) * COL_ROWS),
                        lin_mask,
                        wa,
                        wb,
                        wsum,
                        oa.as_mut_ptr().add(col0),
                        ob.as_mut_ptr().add(col0),
                        oc.as_mut_ptr().add(col0),
                    );
                }
            }
        }
        #[cfg(not(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vbmi",
            target_feature = "gfni"
        )))]
        {
            kernels::fold_scalar(c, w, slot, USED_COLS, COLS_PER_SLOT, oa, ob);
            for col in 0..COLS_PER_SLOT {
                oc[col] = if col < USED_COLS {
                    let x = slot * COLS_PER_SLOT + col;
                    (0..64).fold(F192::ZERO, |acc, s| {
                        let v = c[x * COL_ROWS + s];
                        if v == 0 { acc } else { acc + w[s] * phi8_192(F8(v)) }
                    })
                } else {
                    F192::ZERO
                };
            }
        }
    }
}

/// Prove `a·b + c = 0` over the `2^m` byte rows, `m = ROWS_LOG + n_log`.
pub fn prove(c: &[u8], n_log: usize, ps: &mut ProverState) -> ZerocheckClaim {
    let m = ROWS_LOG + n_log;
    let n_slots = 1usize << n_log;
    let n_mlv = m - K_SKIP;
    assert_eq!(c.len(), 1 << m);
    let r_rest = equality_tail(m, |n| ps.sample_vec(n));
    let span = tracing::info_span!("Round 1").entered();
    let eq = eq_table(&r_rest);
    let round1 = round1(c, &eq, n_slots);
    drop(eq);
    drop(span);
    for &x in &round1 {
        ps.add_scalar(x);
    }
    let z = ps.sample();
    let span = tracing::info_span!("Fold at z").entered();
    let lagrange = lagrange_weights_naive(K_SKIP, z);
    let ([mut a_mlv, mut b_mlv, mut c_mlv], (g1, g_inf)) = fold_at_z(c, &lagrange, &r_rest[1..], n_slots);
    drop(span);
    let span = tracing::info_span!("Table rounds").entered();
    let mut c_running = interpolate_at_z_combined(&round1, K_SKIP, z);
    let mut mlv_chis: Vec<F192> = Vec::with_capacity(n_mlv);
    c_running = send_round(ps, c_running, r_rest[0], None, g1, g_inf, &mut mlv_chis);
    let n_in = a_mlv.len();
    let (mut a_nxt, mut b_nxt, mut c_nxt) = if n_in >= 1024 {
        // SAFETY (x3): the fused rounds write every slot they read; a buffer is only read over the prefix a round just wrote.
        unsafe {
            (
                primitives::uninit_vec::<F192>(n_in / 2),
                primitives::uninit_vec::<F192>(n_in / 2),
                primitives::uninit_vec::<F192>(n_in / 2),
            )
        }
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    };
    for i in 0..(n_mlv - 1) {
        let chi_prev = mlv_chis[i];
        let log_n_before = a_mlv.len().trailing_zeros() as usize;
        let r_eq = &r_rest[i + 2..];
        let (m1, mi) = if log_n_before >= 10 {
            let half = a_mlv.len() / 2;
            let (m1, mi) = fold_and_compute_round_pair_into(
                &a_mlv,
                &b_mlv,
                &mut a_nxt[..half],
                &mut b_nxt[..half],
                chi_prev,
                r_eq,
            );
            let m1c = fold_and_compute_round_single_into(&c_mlv, &mut c_nxt[..half], chi_prev, r_eq);
            std::mem::swap(&mut a_mlv, &mut a_nxt);
            std::mem::swap(&mut b_mlv, &mut b_nxt);
            std::mem::swap(&mut c_mlv, &mut c_nxt);
            a_mlv.truncate(half);
            b_mlv.truncate(half);
            c_mlv.truncate(half);
            (m1 + m1c, mi)
        } else {
            fold_in_place_pair(&mut a_mlv, &mut b_mlv, chi_prev);
            fold_in_place_single(&mut c_mlv, chi_prev);
            let (m1, mi) = round_pair_naive(&a_mlv, &b_mlv, r_eq);
            (m1 + round_single_naive(&c_mlv, r_eq), mi)
        };
        let r = r_rest[i + 1];
        let g0 = (r == F192::ONE).then(|| {
            let eq = eq_table(r_eq);
            (0..eq.len()).fold(F192::ZERO, |acc, x| {
                acc + eq[x] * (a_mlv[2 * x] * b_mlv[2 * x] + c_mlv[2 * x])
            })
        });
        c_running = send_round(ps, c_running, r, g0, m1, mi, &mut mlv_chis);
    }
    let chi_last = *mlv_chis.last().expect("at least one challenge");
    fold_in_place_pair(&mut a_mlv, &mut b_mlv, chi_last);
    let final_a_eval = a_mlv[0];
    let final_b_eval = b_mlv[0];
    let final_c_eval = c_running + final_a_eval * final_b_eval;
    ps.add_scalar(final_a_eval);
    ps.add_scalar(final_b_eval);
    drop(span);
    ZerocheckClaim {
        z,
        mlv_challenges: mlv_chis,
        a_eval: final_a_eval,
        b_eval: final_b_eval,
        c_eval: final_c_eval,
    }
}
