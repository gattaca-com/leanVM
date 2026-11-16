//! The grind's search on the lane-wise backends (AVX-512, NEON, or one at a time). AVX2 without AVX-512 has the
//! plane-sliced kernel's grind instead, which takes this one's place under the same names.

use super::{Backend, LANES, Lane, permute, round};
use crate::{RC, RHO, ROUNDS, STATE_LANES};

/// The output lane [`GrindTemplate`] cuts the last round to.
const CUT_LANE: usize = 2;

/// Widest batch any backend takes.
const MAX_WIDTH: usize = 8;
const _: () = assert!(LANES <= MAX_WIDTH);

// The grind runs a first round, `RC[1..ROUNDS - 1]`, then a cut last round.
const _: () = assert!(ROUNDS >= 2, "the grind needs a first and a last round");

/// Nonces [`GrindTemplate::first_hit`] tests per call: [`LANES`] here, 32 on AVX2 without AVX-512.
pub const GRIND_LANES: usize = LANES;

/// A grind's fixed part: a state, the lane `lane` that takes each nonce, and the test on output lane
/// `out_lane` having no bit of `mask` set.
///
/// - The nonce is written into lane `lane` of [`LANES`] copies of `state`, one per candidate.
/// - Only the first round sees constant lanes: the 24 lanes but `lane`. Their share of it is computed
///   once here, as scalars, and the round runs on the lanes the nonce reaches.
/// - When `out_lane` is `CUT_LANE` the last round computes that one lane. Other lanes take a whole permutation.
///
/// Where the plane-sliced kernel runs (AVX2 without AVX-512), that kernel's grind stands in for it.
pub struct GrindTemplate {
    state: [u64; STATE_LANES],
    lane: usize,
    out_lane: usize,
    mask: u64,
    /// By θ-output lane: its value ρ-rotated, with a zero nonce.
    rotated: [u64; STATE_LANES],
    /// By θ-output lane: whether the nonce reaches it.
    varies: [bool; STATE_LANES],
    /// By θ-output lane: whether the nonce reaches it as `rotl(nonce, 1)`, else as the nonce.
    shifted: [bool; STATE_LANES],
    /// By lane after the first round: its value if the nonce does not reach it.
    constant: [u64; STATE_LANES],
    /// By lane after the first round: whether the nonce reaches it.
    reached: [bool; STATE_LANES],
}

impl GrindTemplate {
    pub fn new(state: &[u64; STATE_LANES], lane: usize, out_lane: usize, mask: u64) -> Self {
        assert!(lane < STATE_LANES && out_lane < STATE_LANES);
        let mut zeroed = *state;
        zeroed[lane] = 0;
        // θ with a zero nonce. The nonce adds to its own column's parity, which gives the column
        // after it `nonce` in θ's `d` and the column before it `rotl(nonce, 1)`.
        let c: [u64; 5] = std::array::from_fn(|x| {
            u64::xor5(zeroed[x], zeroed[x + 5], zeroed[x + 10], zeroed[x + 15], zeroed[x + 20])
        });
        let d: [u64; 5] = std::array::from_fn(|x| u64::theta_d(c[(x + 4) % 5], c[(x + 1) % 5]));
        let (after, before) = ((lane + 1) % 5, (lane + 4) % 5);
        let theta: [u64; STATE_LANES] = std::array::from_fn(|i| zeroed[i] ^ d[i % 5]);
        let rotated = std::array::from_fn(|i| theta[i].rotate_left(RHO[i]));
        let varies = std::array::from_fn(|i| i == lane || i % 5 == after || i % 5 == before);
        let shifted = std::array::from_fn(|i| i % 5 == before);
        macro_rules! rot {
            ($i:expr, $r:literal) => {
                rotated[$i]
            };
        }
        macro_rules! var {
            ($i:expr, $r:literal) => {
                varies[$i]
            };
        }
        let (rows, row_varies): ([[u64; 5]; 5], [[bool; 5]; 5]) = (rho_pi_rows!(rot), rho_pi_rows!(var));
        let mut constant = [0u64; STATE_LANES];
        let mut reached = [false; STATE_LANES];
        for y in 0..5 {
            for x in 0..5 {
                let (b0, b1, b2) = (rows[y][x], rows[y][(x + 1) % 5], rows[y][(x + 2) % 5]);
                constant[5 * y + x] = u64::chi(b0, b1, b2) ^ if y == 0 && x == 0 { RC[0] } else { 0 };
                reached[5 * y + x] = row_varies[y][x] || row_varies[y][(x + 1) % 5] || row_varies[y][(x + 2) % 5];
            }
        }
        Self {
            state: *state,
            lane,
            out_lane,
            mask,
            rotated,
            varies,
            shifted,
            constant,
            reached,
        }
    }

    /// The smallest `i < GRIND_LANES` for which nonce `first + i` passes the test.
    pub fn first_hit(&self, first: u64) -> Option<usize> {
        self.first_hit_with::<Backend>(first)
    }

    /// The smallest `i < S::WIDTH` for which nonce `first + i` passes the test, on the lane-wise kernel `S`.
    #[inline(always)]
    fn first_hit_with<S: Lane>(&self, first: u64) -> Option<usize> {
        let nonces: [u64; MAX_WIDTH] = std::array::from_fn(|i| first.wrapping_add(i as u64));
        let nonce = S::load(&nonces[..S::WIDTH]);
        let out = if self.out_lane == CUT_LANE {
            let (mut a, mut c) = self.first_round(nonce);
            for &rc in &RC[1..ROUNDS - 1] {
                round(&mut a, &mut c, rc);
            }
            // Lane 2 of the last round: χ of θ's lanes 12, 18 and 24 after ρ, and no ι.
            let d = [S::theta_d(c[1], c[3]), S::theta_d(c[2], c[4]), S::theta_d(c[3], c[0])];
            let b2 = a[12].xor(d[0]).rotl::<43>();
            let b3 = a[18].xor(d[1]).rotl::<21>();
            let b4 = a[24].xor(d[2]).rotl::<14>();
            S::chi(b2, b3, b4)
        } else {
            let mut st: [S; STATE_LANES] = std::array::from_fn(|i| S::splat(self.state[i]));
            st[self.lane] = nonce;
            permute(&mut st);
            st[self.out_lane]
        };
        let mut words = [0u64; MAX_WIDTH];
        out.store(&mut words[..S::WIDTH]);
        words[..S::WIDTH].iter().position(|w| w & self.mask == 0)
    }

    /// The first round on the nonces, and the column parities it leaves.
    #[inline(always)]
    fn first_round<S: Lane>(&self, nonce: S) -> ([S; STATE_LANES], [S; 5]) {
        let shifted = nonce.rotl::<1>();
        macro_rules! rotated {
            ($i:expr, $r:literal) => {{
                let base = S::splat(self.rotated[$i]);
                if self.varies[$i] {
                    let delta = if self.shifted[$i] { shifted } else { nonce };
                    base.xor(delta.rotl::<$r>())
                } else {
                    base
                }
            }};
        }
        let rows: [[S; 5]; 5] = rho_pi_rows!(rotated);
        let mut a = [S::splat(0); STATE_LANES];
        for (y, row) in rows.iter().enumerate() {
            for x in 0..5 {
                let i = 5 * y + x;
                a[i] = if self.reached[i] {
                    let out = S::chi(row[x], row[(x + 1) % 5], row[(x + 2) % 5]);
                    if i == 0 { out.xor(S::splat(RC[0])) } else { out }
                } else {
                    S::splat(self.constant[i])
                };
            }
        }
        let c = std::array::from_fn(|x| S::xor5(a[x], a[x + 5], a[x + 10], a[x + 15], a[x + 20]));
        (a, c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keccak_f;

    /// A grind's first hit is the scalar permutation's, for every nonce lane, output lane and mask,
    /// on the folded first round and the cut last round and on the plain path.
    #[test]
    fn grind_matches_scalar() {
        fn check<S: Lane>(name: &str) {
            let mut seed = 0x2545_f491_4f6c_dd1du64;
            let mut next = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };
            for lane in 0..STATE_LANES {
                let state: [u64; STATE_LANES] = std::array::from_fn(|_| next());
                for out_lane in [CUT_LANE, 0, 7] {
                    for mask in [0, 0x1, 0x3, 0xff, u64::MAX] {
                        let template = GrindTemplate::new(&state, lane, out_lane, mask);
                        for first in [0, 1000, u64::MAX - 2, next()] {
                            let want = (0..S::WIDTH as u64)
                                .map(|i| {
                                    let mut copy = state;
                                    copy[lane] = first.wrapping_add(i);
                                    keccak_f(&mut copy);
                                    copy[out_lane]
                                })
                                .position(|w| w & mask == 0);
                            assert_eq!(
                                template.first_hit_with::<S>(first),
                                want,
                                "{name}: lane {lane}, out lane {out_lane}, mask {mask:#x}, first {first}"
                            );
                        }
                    }
                }
            }
        }
        check::<u64>("scalar");
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        check::<super::super::x86::Avx512>("avx512");
        #[cfg(target_arch = "aarch64")]
        check::<super::super::arm::Neon>("neon");
    }
}
