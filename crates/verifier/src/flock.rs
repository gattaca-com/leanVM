//! Flock's zerocheck and ring switching, shared by the Haraka lincheck (`crate::haraka`).

use crate::VerifierError;
use crate::field::{E, GEN, ONE, ZERO};
use crate::layout::{FLOCK_K_SKIP, K_BITS};
use crate::multilinear::{dot, poly_eval};
use crate::transcript::{Transcript, sumcheck};

const PHI_BASIS: [u64; 8] = [
    0x0000000000000001,
    0x033CE8BEDDC8A656,
    0x512620375ED2A108,
    0x0C9E636090AAFC01,
    0xBA4F3CD82801769C,
    0xBA26E7904ADB4A47,
    0x467698598926DC01,
    0x4418AE808B28BDD0,
];

/// `PHI[value]`: the F2-linear image of the byte `value` in the PHI basis.
pub(crate) fn phi(value: usize) -> E {
    let bits = (0..8).filter(|bit| value >> bit & 1 == 1);
    E::new(bits.fold(0, |acc, bit| acc ^ PHI_BASIS[bit]), 0, 0)
}

const MEDIUM_GENERATOR: E = E::new(0x243F6A8885A308D3, 0x13198A2E03707344, 0xA4093822299F31D0);
const N_FIXED_CHALLENGES: usize = 7;

fn fixed_challenges() -> [E; N_FIXED_CHALLENGES] {
    let medium = |power: u32| {
        let value = MEDIUM_GENERATOR.pow(1 << power);
        value * (ONE + value).inv()
    };
    [
        phi(0xF7),
        phi(0x53),
        phi(0xB5),
        medium(0),
        medium(1),
        medium(2),
        medium(3),
    ]
}

/// The one barycentric denominator `PHI[:count]` has: `prod_(k != 0) PHI[k]`, inverted.
///
/// PHI is F2-linear in its index, so `PHI[i] + PHI[j] = PHI[i ^ j]`, and over a power-of-two prefix
/// `j -> i ^ j` only permutes the block. Every node is left the same product.
fn window_denominator(count: usize) -> E {
    (1..count).fold(ONE, |acc, k| acc * phi(k)).inv()
}

/// The barycentric weights of `PHI[:count]` at `point`, by prefix and suffix numerator products.
pub(crate) fn lagrange_weights(count: usize, point: E) -> Vec<E> {
    let differences: Vec<E> = (0..count).map(|node| point + phi(node)).collect();
    let mut prefix = vec![ONE; count + 1];
    let mut suffix = vec![ONE; count + 1];
    for i in 0..count {
        prefix[i + 1] = prefix[i] * differences[i];
        suffix[count - 1 - i] = suffix[count - i] * differences[count - 1 - i];
    }
    let denominator = window_denominator(count);
    (0..count).map(|i| prefix[i] * suffix[i + 1] * denominator).collect()
}

fn lagrange_interpolate(count: usize, values: &[E], point: E) -> E {
    dot(&lagrange_weights(count, point), values)
}

pub(crate) struct ZerocheckResult {
    pub(crate) z_skip: E,
    pub(crate) chi: Vec<E>,
    pub(crate) v_a: E,
    pub(crate) v_b: E,
    pub(crate) v_c: E,
}

/// The zerocheck: one univariate skip round, then nflock quadratic ones.
/// C rides those rounds with AB, so all three claims come out at one point.
pub(crate) fn verify_flock_zerocheck(
    log_n: usize,
    transcript: &mut Transcript,
) -> Result<ZerocheckResult, VerifierError> {
    // The point r: seven fixed coordinates, the rest sampled.
    let mut r: Vec<Option<E>> = fixed_challenges().map(Some).into();
    let sampled = transcript.samples(log_n - FLOCK_K_SKIP - N_FIXED_CHALLENGES);
    r.extend(sampled.into_iter().map(Some));

    // P = P^AB + P^C on the coset, then z_skip; the 64 zeros on Lambda are assumed.
    let p_coset = transcript.next_scalars(K_BITS)?;
    let z_skip = transcript.sample();
    let v_p = lagrange_interpolate(2 * K_BITS, &[vec![ZERO; K_BITS], p_coset].concat(), z_skip);

    // nflock quadratic rounds on P, closed by v_a, v_b.
    let (chi, running) = sumcheck(transcript, v_p, 3, &r)?;
    let (v_a, v_b) = (transcript.next_scalar()?, transcript.next_scalar()?);
    let v_c = running + v_a * v_b;
    Ok(ZerocheckResult {
        z_skip,
        chi,
        v_a,
        v_b,
        v_c,
    })
}

// The Frobenius shifts of the six stages composing Phi, one challenge each.
const RING_MAP_SHIFTS: [usize; 6] = [32, 16, 8, 4, 2, 1];

/// The drawn map, stage by stage: `a_p+1 = a_p + f_p a_p^(2^shift)`.
fn ring_map(mut value: E, challenges: &[E]) -> E {
    for (&challenge, shift) in challenges.iter().zip(RING_MAP_SHIFTS) {
        value += challenge * value.pow(1 << shift);
    }
    value
}

/// The weight `W(u) = Phi(eq(r, u))`, extended and evaluated by the opening at
/// `r_prime`: `sum_k c_k prod_n (1 + r_n^(2^k) + r'_n)`.
fn ring_weight(r: &[E], r_prime: &[E], coefficients: &[E]) -> E {
    assert_eq!(r.len(), r_prime.len());
    let mut total = ZERO;
    let mut frobenius = r.to_vec();
    for &c in coefficients {
        let mut product = c;
        for (value, &challenge) in frobenius.iter_mut().zip(r_prime) {
            product *= ONE + *value + challenge;
            *value = value.square();
        }
        total += product;
    }
    total
}

/// The 64 claims s\[i\] = z(i, point) become the one dense claim `sum_u W(u) qflock(u) = target`.
///
/// Draw Phi once they are fixed, then take the target `T = sum_i x^i Phi(s_i)` against the
/// MLE-friendly weight `W(u) = Phi(eq(point, u))`. Returns the target and W as a closure.
pub fn ring_switch(point: Vec<E>, s: &[E], transcript: &mut Transcript) -> (E, impl Fn(&[E]) -> E + use<>) {
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
    let target = poly_eval(&images, GEN);
    (target, move |r_prime: &[E]| ring_weight(&point, r_prime, &coefficients))
}
