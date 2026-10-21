//! Lincheck for a rectangular block: `2^ROWS_LOG` rows against `2^K_LOG` columns
//! per slot, with `C` a general GF(2^8) matrix rather than the identity. The
//! batched column marginal is `ξ_A + α ξ_B + α² ξ_C` with the constant-wire pin
//! at `α³`, folded against the quirky row weights; the product sumcheck over the
//! `K_LOG − 6` high column bits and the 64 transmitted slices are those of
//! [`crate::lincheck`].

use super::circuit::*;
use crate::lincheck::{
    LincheckClaim, QuirkyPoint, build_quirky_eq_table, partial_fold_packed_z_best, sumcheck_bind_both_and_eval_next,
    sumcheck_bind_top_in_place_par, sumcheck_round_eval_par,
};
use crate::zerocheck::K_SKIP;
use fiat_shamir::transcript::ProverState;
use pcs::ring_switch::inner_product_ext;
use primitives::field::{F64, F192, phi8_192};
use primitives::multilinear::eq_table;

/// `comb[col] = Σ_rows eq_rows[row] · φ8(A + α B + α² C)[row, col]` over one slot's matrices.
fn fold_alpha_batched(mats: &SlotMatrices, alpha: F192, eq_rows: &[F192]) -> Vec<F192> {
    let alpha_sq = alpha.square();
    let n_ranges = 64;
    let per = USED_ROWS.div_ceil(n_ranges);
    parallel::map_reduce(
        n_ranges,
        || vec![F192::ZERO; K],
        |i| {
            let mut acc = vec![F192::ZERO; K];
            let mut scatter = |rows: &[Vec<(u32, u8)>], scale: F192, lo: usize, hi: usize| {
                for row in lo..hi {
                    let e = eq_rows[row] * scale;
                    for &(col, g) in &rows[row] {
                        acc[col as usize] += e.mul_base(F64(phi8_192(primitives::field::F8(g)).c0));
                    }
                }
            };
            let (lo, hi) = (i * per, ((i + 1) * per).min(USED_ROWS));
            scatter(&mats.a, F192::ONE, lo, hi);
            scatter(&mats.b, alpha, lo, hi);
            scatter(&mats.c, alpha_sq, lo, hi);
            acc
        },
        |mut x, y| {
            for (a, b) in x.iter_mut().zip(&y) {
                *a += *b;
            }
            x
        },
    )
}

pub fn prove(
    mats: &SlotMatrices,
    z_lincheck: &[u8],
    n_log: usize,
    x_ab: &QuirkyPoint,
    zc: &crate::zerocheck::ZerocheckClaim,
    ps: &mut ProverState,
) -> LincheckClaim {
    assert_eq!(x_ab.x_inner_rest.len(), ROWS_LOG - K_SKIP);
    assert_eq!(x_ab.x_outer.len(), n_log);
    let alpha = ps.sample();
    let eq_rows =
        tracing::info_span!("Eq table").in_scope(|| build_quirky_eq_table(x_ab.z_skip, &x_ab.x_inner_rest, K_SKIP));
    let mut comb_vec = tracing::info_span!("Fold circuit").in_scope(|| fold_alpha_batched(mats, alpha, &eq_rows));
    comb_vec[CONST_COL] += alpha.square() * alpha;
    let mut z_vec = tracing::info_span!("Partial fold").in_scope(|| {
        let mut eq_x_outer = eq_table(&x_ab.x_outer);
        eq_x_outer.resize(eq_x_outer.len().max(8), F192::ZERO);
        partial_fold_packed_z_best(z_lincheck, K_LOG + n_log.max(3), K_LOG, USEFUL_BITS, &eq_x_outer)
    });
    let span = tracing::info_span!("Sumcheck").entered();
    let inner_rest_len = K_LOG - K_SKIP;
    let mut r_rounds = Vec::with_capacity(inner_rest_len);
    let (mut e1, mut einf) = sumcheck_round_eval_par(&comb_vec, &z_vec);
    let mut running = inner_product_ext(&comb_vec, &z_vec);
    debug_assert_eq!(
        running,
        zc.a_eval + alpha * zc.b_eval + alpha.square() * zc.c_eval + alpha.square() * alpha,
        "the batched column marginal disagrees with the zerocheck's claims"
    );
    for t in 0..inner_rest_len {
        let e0 = running + e1;
        ps.add_round_poly(&[e0, e0 + e1 + einf, einf], false);
        let r = ps.sample();
        running = (einf * r + (e0 + e1 + einf)) * r + e0;
        r_rounds.push(r);
        if t + 1 < inner_rest_len {
            let (ne1, neinf) = sumcheck_bind_both_and_eval_next(&mut comb_vec, &mut z_vec, r);
            e1 = ne1;
            einf = neinf;
        } else {
            sumcheck_bind_top_in_place_par(&mut z_vec, r);
        }
    }
    drop(span);
    for &x in &z_vec {
        ps.add_scalar(x);
    }
    let mut r_inner_rest = r_rounds;
    r_inner_rest.reverse();
    LincheckClaim {
        r_inner_rest,
        s_hat_v: z_vec,
    }
}
