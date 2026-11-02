//! Ring switching: the 64 slice claims a reduction leaves become one claim on the committed column.

use crate::field::{E, GEN, ONE, ZERO};
use crate::layout::K_BITS;
use crate::multilinear::poly_eval;
use crate::transcript::Transcript;

// The Frobenius shifts of the six stages composing Phi, one challenge each.
const RING_MAP_SHIFTS: [usize; 6] = [32, 16, 8, 4, 2, 1];

/// The drawn map, stage by stage: `a_p+1 = a_p + f_p a_p^(2^shift)`.
fn ring_map(mut value: E, challenges: &[E]) -> E {
    for (&challenge, shift) in challenges.iter().zip(RING_MAP_SHIFTS) {
        value += challenge * value.pow(1 << shift);
    }
    value
}

/// `sum_k c_k W(r^(2^k))`, the Frobenius powers taken coordinate by coordinate: the extension of `Phi(W(r, u))` from
/// the cube's points `u`, for a weight `W` with coefficients in F2, as an eq or a jagged weight has.
pub fn frobenius_weight(r: &[E], coefficients: &[E], weight: impl Fn(&[E]) -> E) -> E {
    let mut total = ZERO;
    let mut frobenius = r.to_vec();
    for &c in coefficients {
        total += c * weight(&frobenius);
        for value in frobenius.iter_mut() {
            *value = value.square();
        }
    }
    total
}

/// The 64 claims s\[i\] = z(i, point) become the one dense claim `sum_u W(u) qkeccak(u) = target`.
///
/// Draw Phi once they are fixed, then take the target `T = sum_i x^i Phi(s_i)` against the
/// MLE-friendly weight `W(u) = Phi(eq(point, u))`, whose extension is [`frobenius_weight`] over
/// the coefficients returned.
pub fn ring_switch(s: &[E], transcript: &mut Transcript) -> (E, Vec<E>) {
    let challenges = transcript.samples(RING_MAP_SHIFTS.len());
    // The same map as a Frobenius sum, `Phi(a) = sum_k c_k a^(2^k)` for k < 64.
    let coefficient = |k: usize| {
        let stages = challenges
            .iter()
            .zip(RING_MAP_SHIFTS)
            .filter(|&(_, shift)| k & shift != 0);
        stages.fold(ONE, |acc, (f, shift)| acc * f.pow(1 << (k % shift)))
    };
    let coefficients: Vec<E> = (0..K_BITS).map(coefficient).collect();
    let images: Vec<E> = s.iter().map(|&value| ring_map(value, &challenges)).collect();
    (poly_eval(&images, GEN), coefficients)
}
