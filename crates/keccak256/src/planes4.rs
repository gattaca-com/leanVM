//! Keccak-f\[1600\] over 16 states with every lane cut into four bit planes, for AVX2 without AVX-512.
//!
//! # Layout
//!
//! - A lane is four 256-bit vectors, one per plane `k < 4`.
//! - Bit `z = 4q + k` of state `s = 8g + t` is bit `t` of byte `16g + q` of plane `k`.
//! - So a plane holds the bits `z ≡ k (mod 4)`: sixteen bytes per group of eight states, a bit per state.
//!
//! # Why
//!
//! - AVX2 has no 64-bit rotate. The word-sliced round spends 87 of its 184 vector operations on rotations.
//! - Here a rotation by `4a + c` renames planes by `c`.
//! - It also rotates the bytes of each group by `a` or `a + 1`.
//! - The byte rotation is one `vpalignr` with an immediate, or nothing when it is a multiple of sixteen.
//! - The batch is 3.2 KB, so it stays in L1.
//! - Every vector is loaded and stored once a round.
//! - Zen 5 moves about two 256-bit loads and two stores a cycle.
//! - That bounds the round, so nothing else may touch memory.
//!
//! # Stack
//!
//! [`hash_sets_of_32`] holds about 19 KiB of locals, which fits a 2 MiB worker stack.

use core::arch::x86_64::*;

use crate::{END_BIT, OUT_LEN, PAD_FIRST, RATE, RATE_LANES, RC, ROUNDS, STATE_LANES};

/// States one batch holds.
const BATCH: usize = 16;
/// Planes of a lane.
const P: usize = 4;

/// One plane: two groups of eight states, a byte per `q` in each.
type Plane = [u64; 4];

/// A batch of states: `0[i][k]` is plane `k` of lane `i`.
#[derive(Clone)]
#[repr(C, align(64))]
struct Planes([[Plane; P]; STATE_LANES]);

impl Planes {
    const ZERO: Self = Self([[[0; 4]; P]; STATE_LANES]);
}

/// θ's column effects: `0[x][k]` is plane `k` of column `x`.
#[repr(C, align(64))]
struct Columns([[Plane; P]; 5]);

macro_rules! pair {
    ($i:expr, $r:literal) => {
        ($i, $r)
    };
}

/// Row `y` of the state after ρ and π: per lane, the θ output lane it takes and its left rotation.
const ROWS: [[(usize, usize); 5]; 5] = rho_pi_rows!(pair);

/// Where plane `k` of a lane rotated left by `r` comes from: the source plane, and how far its bytes rotate.
const fn source(r: usize, k: usize) -> (usize, usize) {
    let (a, c) = (r / P, r % P);
    if k >= c { (k - c, a) } else { (k + P - c, (a + 1) % 16) }
}

/// ι's constant as planes: every state takes the same bits.
const fn iota_planes(rc: u64) -> [Plane; P] {
    let mut planes = [[0u64; 4]; P];
    let mut z = 0;
    while z < 64 {
        if rc >> z & 1 == 1 {
            let (q, k) = (z / P, z % P);
            let mut g = 0;
            while g < 2 {
                planes[k][2 * g + q / 8] |= 0xff << (8 * (q % 8));
                g += 1;
            }
        }
        z += 1;
    }
    planes
}

/// ι's constant of every round.
#[repr(C, align(32))]
struct Iota([[Plane; P]; ROUNDS]);

static IOTA: Iota = Iota({
    let mut all = [[[0u64; 4]; P]; ROUNDS];
    let mut r = 0;
    while r < ROUNDS {
        all[r] = iota_planes(RC[r]);
        r += 1;
    }
    all
});

#[inline(always)]
unsafe fn ld(p: &Plane) -> __m256i {
    // SAFETY: a plane is 32 readable bytes at a 32-byte boundary.
    unsafe { _mm256_load_si256((p as *const Plane).cast()) }
}

#[inline(always)]
unsafe fn st(p: &mut Plane, v: __m256i) {
    // SAFETY: a plane is 32 writable bytes at a 32-byte boundary.
    unsafe { _mm256_store_si256((p as *mut Plane).cast(), v) }
}

/// The sixteen bytes of each group of `v` rotated up by `AMOUNT`, which is below sixteen.
#[inline(always)]
unsafe fn rot_bytes<const AMOUNT: usize>(v: __m256i) -> __m256i {
    const { assert!(AMOUNT < 16) };
    // SAFETY: the function runs under AVX2. Byte `j` of a group takes byte `j - AMOUNT` of it.
    unsafe {
        match AMOUNT {
            0 => v,
            1 => _mm256_alignr_epi8::<15>(v, v),
            2 => _mm256_alignr_epi8::<14>(v, v),
            3 => _mm256_alignr_epi8::<13>(v, v),
            4 => _mm256_alignr_epi8::<12>(v, v),
            5 => _mm256_alignr_epi8::<11>(v, v),
            6 => _mm256_alignr_epi8::<10>(v, v),
            7 => _mm256_alignr_epi8::<9>(v, v),
            8 => _mm256_alignr_epi8::<8>(v, v),
            9 => _mm256_alignr_epi8::<7>(v, v),
            10 => _mm256_alignr_epi8::<6>(v, v),
            11 => _mm256_alignr_epi8::<5>(v, v),
            12 => _mm256_alignr_epi8::<4>(v, v),
            13 => _mm256_alignr_epi8::<3>(v, v),
            14 => _mm256_alignr_epi8::<2>(v, v),
            15 => _mm256_alignr_epi8::<1>(v, v),
            _ => unreachable!(),
        }
    }
}

/// [`rot_bytes`] by an amount known only after inlining, which is below sixteen.
#[inline(always)]
unsafe fn rot(v: __m256i, amount: usize) -> __m256i {
    // SAFETY: forwarded.
    unsafe {
        match amount {
            0 => rot_bytes::<0>(v),
            1 => rot_bytes::<1>(v),
            2 => rot_bytes::<2>(v),
            3 => rot_bytes::<3>(v),
            4 => rot_bytes::<4>(v),
            5 => rot_bytes::<5>(v),
            6 => rot_bytes::<6>(v),
            7 => rot_bytes::<7>(v),
            8 => rot_bytes::<8>(v),
            9 => rot_bytes::<9>(v),
            10 => rot_bytes::<10>(v),
            11 => rot_bytes::<11>(v),
            12 => rot_bytes::<12>(v),
            13 => rot_bytes::<13>(v),
            14 => rot_bytes::<14>(v),
            15 => rot_bytes::<15>(v),
            _ => unreachable!(),
        }
    }
}

/// Plane `K` of row `Y` of the next state, and its share of the next column parities in `acc`.
///
/// The pointers are raw on purpose, for speed only. Every block is distinct, so references would be valid.
/// With references, the compiler knows a store to `dst` leaves `d` alone.
/// It then loads each effect once, finds no register for it, and spills it.
/// That costs a load and a store more per vector.
/// A raw `d` may alias `dst` as far as the compiler knows, so each use is a memory operand of its XOR.
/// A compiler update may change this. Check the cycles per permutation of the hash benchmark after one.
#[inline(always)]
unsafe fn unit<const Y: usize, const K: usize>(
    src: *const Planes,
    d: *const Columns,
    dst: *mut Planes,
    acc: &mut [__m256i; 5],
    iota: &[Plane; P],
) {
    // SAFETY: the function runs under AVX2, and the caller passes three live, disjoint blocks.
    unsafe {
        let b: [__m256i; 5] = std::array::from_fn(|x| {
            let (i, r) = ROWS[Y][x];
            let (from, amount) = source(r, K);
            rot(
                _mm256_xor_si256(ld(&(*src).0[i][from]), ld(&(*d).0[i % 5][from])),
                amount,
            )
        });
        for x in 0..5 {
            let mut out = _mm256_xor_si256(b[x], _mm256_andnot_si256(b[(x + 1) % 5], b[(x + 2) % 5]));
            if Y == 0 && x == 0 {
                out = _mm256_xor_si256(out, ld(&iota[K]));
            }
            st(&mut (*dst).0[5 * Y + x][K], out);
            acc[x] = if Y == 0 { out } else { _mm256_xor_si256(acc[x], out) };
        }
    }
}

/// Plane `K` of the next state, all five rows, and what its column parities give of the next θ.
///
/// - The parities of this plane stay in registers across the rows, and go out in `prev` for the next plane.
/// - `d_next[x][k] = c[x - 1][k] ^ c[x + 1][k - 1]`, so plane `K >= 1` of it is ready here.
/// - Plane 0 of it needs the last plane's parities with their bytes rotated by one, so plane 0's parities wait in `c0`.
#[inline(always)]
unsafe fn plane<const K: usize>(
    src: *const Planes,
    d: *const Columns,
    dst: *mut Planes,
    d_next: *mut Columns,
    c0: &mut [__m256i; 5],
    prev: &mut [__m256i; 5],
    iota: &[Plane; P],
) {
    // SAFETY: forwarded.
    unsafe {
        let mut acc = [_mm256_setzero_si256(); 5];
        unit::<0, K>(src, d, dst, &mut acc, iota);
        unit::<1, K>(src, d, dst, &mut acc, iota);
        unit::<2, K>(src, d, dst, &mut acc, iota);
        unit::<3, K>(src, d, dst, &mut acc, iota);
        unit::<4, K>(src, d, dst, &mut acc, iota);
        if K == 0 {
            *c0 = acc;
        } else {
            for x in 0..5 {
                st(
                    &mut (*d_next).0[x][K],
                    _mm256_xor_si256(acc[(x + 4) % 5], prev[(x + 1) % 5]),
                );
            }
        }
        if K == P - 1 {
            for x in 0..5 {
                st(
                    &mut (*d_next).0[x][0],
                    _mm256_xor_si256(c0[(x + 4) % 5], rot_bytes::<1>(acc[(x + 1) % 5])),
                );
            }
        }
        *prev = acc;
    }
}

/// What a batch needs beside its state: the other side of the round, and θ's effects of this round and the next.
#[repr(C, align(64))]
struct Scratch {
    other: Planes,
    d: [Columns; 2],
}

impl Scratch {
    const ZERO: Self = Self {
        other: Planes::ZERO,
        d: [const { Columns([[[0; 4]; P]; 5]) }; 2],
    };
}

/// One round of one batch, from `src` into `dst`, with θ's effects in `d` and the next round's into `d_next`.
///
/// It is not inlined, so one copy of the code serves every round and batch.
#[inline(never)]
#[target_feature(enable = "avx2")]
unsafe fn round(src: *const Planes, d: *const Columns, dst: *mut Planes, d_next: *mut Columns, iota: &[Plane; P]) {
    // SAFETY: forwarded.
    unsafe {
        let mut c0 = [_mm256_setzero_si256(); 5];
        let mut prev = [_mm256_setzero_si256(); 5];
        plane::<0>(src, d, dst, d_next, &mut c0, &mut prev, iota);
        plane::<1>(src, d, dst, d_next, &mut c0, &mut prev, iota);
        plane::<2>(src, d, dst, d_next, &mut c0, &mut prev, iota);
        plane::<3>(src, d, dst, d_next, &mut c0, &mut prev, iota);
    }
}

/// The rounds ping-pong between the state and the scratch, so an even count ends in the state.
const _: () = assert!(ROUNDS.is_multiple_of(2));

/// Keccak-f of every state of every batch, in place.
///
/// - The batches take each round in turn, so one copy of the round's code serves them all.
/// - A round ends in a barrier: the next θ needs every parity. Another batch's round covers that wait.
#[target_feature(enable = "avx2")]
fn permute(batches: &mut [Planes], scratch: &mut [Scratch]) {
    assert_eq!(batches.len(), scratch.len());
    // SAFETY: the function carries AVX2, and every pointer is to a live block of its own.
    unsafe {
        for (a, s) in batches.iter().zip(scratch.iter_mut()) {
            // The first round's θ, from the state itself.
            let c: [[__m256i; P]; 5] = std::array::from_fn(|x| {
                std::array::from_fn(|k| {
                    let lane = |y: usize| ld(&a.0[x + 5 * y][k]);
                    let ab = _mm256_xor_si256(lane(0), lane(1));
                    let cd = _mm256_xor_si256(lane(2), lane(3));
                    _mm256_xor_si256(_mm256_xor_si256(ab, cd), lane(4))
                })
            });
            for x in 0..5 {
                for k in 0..P {
                    let (from, amount) = source(1, k);
                    st(
                        &mut s.d[0].0[x][k],
                        _mm256_xor_si256(c[(x + 4) % 5][k], rot(c[(x + 1) % 5][from], amount)),
                    );
                }
            }
        }
        let (a, s) = (batches.as_mut_ptr(), scratch.as_mut_ptr());
        for r in 0..ROUNDS {
            let iota = &IOTA.0[r];
            for g in 0..batches.len() {
                let (state, other) = (a.add(g), &raw mut (*s.add(g)).other);
                let d = (&raw mut (*s.add(g)).d).cast::<Columns>();
                // Even rounds go from the state to the scratch, odd rounds back.
                let (src, dst) = if r % 2 == 0 { (state, other) } else { (other, state) };
                round(src, d.add(r % 2), dst, d.add(1 - r % 2), iota);
            }
        }
    }
}

/// Inputs the sliced hash takes at a time: two batches of 16, whose rounds alternate.
const SET: usize = 2 * BATCH;

/// The 8x8 bit transpose down eight vectors: bit `j` of every byte of `u[t]` trades places with bit `t` of `u[j]`.
///
/// Three rounds of swaps, on blocks of four, two and one bits. It is its own inverse.
#[inline(always)]
unsafe fn transpose_bits(u: &mut [__m256i; 8]) {
    // SAFETY: the function runs under AVX2.
    unsafe {
        macro_rules! stage {
            ($shift:literal, $mask:literal, $pairs:expr) => {
                let mask = _mm256_set1_epi8($mask);
                for (lo, hi) in $pairs {
                    let x = _mm256_and_si256(_mm256_xor_si256(_mm256_srli_epi64::<$shift>(u[lo]), u[hi]), mask);
                    u[hi] = _mm256_xor_si256(u[hi], x);
                    u[lo] = _mm256_xor_si256(u[lo], _mm256_slli_epi64::<$shift>(x));
                }
            };
        }
        stage!(4, 0x0f, [(0, 4), (1, 5), (2, 6), (3, 7)]);
        stage!(2, 0x33, [(0, 2), (1, 3), (4, 6), (5, 7)]);
        stage!(1, 0x55, [(0, 1), (2, 3), (4, 5), (6, 7)]);
    }
}

/// XOR words `0..words` of each input's block at byte `offset` into lanes `0..words` of the two batches.
///
/// - Input `8m + t` is state `t` of group `m / 2` of batch `m % 2`.
/// - Vector `t` holds one word of inputs `t`, `8 + t`, `16 + t` and `24 + t`.
/// - The bit transpose turns the eight vectors into eight planes of bytes, one per bit of a byte.
/// - Planes `k` and `k + 4` interleave into plane `k` of this layout: the low halves are one batch, the high the other.
#[inline(always)]
unsafe fn absorb(states: &mut [Planes; 2], data: &[u8], len: usize, offset: usize, words: usize) {
    assert!(data.len() == SET * len && offset + 8 * words <= len && words <= STATE_LANES);
    // SAFETY: the assert keeps every word read inside its input, and the function runs under AVX2.
    unsafe {
        let word = |j: usize, l: usize| {
            data.as_ptr()
                .add(j * len + offset + 8 * l)
                .cast::<i64>()
                .read_unaligned()
        };
        for l in 0..words {
            let mut u: [__m256i; 8] = std::array::from_fn(|t| {
                _mm256_set_epi64x(word(24 + t, l), word(16 + t, l), word(8 + t, l), word(t, l))
            });
            transpose_bits(&mut u);
            for k in 0..P {
                let (a, b) = (
                    _mm256_unpacklo_epi8(u[k], u[k + 4]),
                    _mm256_unpackhi_epi8(u[k], u[k + 4]),
                );
                for (state, v) in states.iter_mut().zip([a, b]) {
                    let plane = &mut state.0[l][k];
                    st(plane, _mm256_xor_si256(ld(plane), v));
                }
            }
        }
    }
}

/// The even bytes of each group, then its odd bytes.
#[repr(C, align(32))]
struct EvenOdd([u8; 32]);

static EVEN_ODD: EvenOdd = EvenOdd({
    let mut ctl = [0u8; 32];
    let mut b = 0;
    while b < 32 {
        let j = b % 16;
        ctl[b] = if j < 8 { 2 * j as u8 } else { 2 * (j as u8 - 8) + 1 };
        b += 1;
    }
    ctl
});

/// The 4x4 transpose of qwords: qword `j` of `out[p]` is qword `p` of `r[j]`.
#[inline(always)]
unsafe fn transpose4(r: [__m256i; 4]) -> [__m256i; 4] {
    // SAFETY: the function runs under AVX2, and these intrinsics touch only registers.
    unsafe {
        let t0 = _mm256_unpacklo_epi64(r[0], r[1]);
        let t1 = _mm256_unpackhi_epi64(r[0], r[1]);
        let t2 = _mm256_unpacklo_epi64(r[2], r[3]);
        let t3 = _mm256_unpackhi_epi64(r[2], r[3]);
        [
            _mm256_permute2x128_si256::<0x20>(t0, t2),
            _mm256_permute2x128_si256::<0x20>(t1, t3),
            _mm256_permute2x128_si256::<0x31>(t0, t2),
            _mm256_permute2x128_si256::<0x31>(t1, t3),
        ]
    }
}

/// Write the first [`OUT_LEN`] bytes of every state to `out`, in the order [`absorb`] reads the inputs.
#[inline(always)]
unsafe fn digests(states: &[Planes; 2], out: &mut [u8]) {
    assert_eq!(out.len(), SET * OUT_LEN);
    // SAFETY: the assert makes `out` 32 digests of 32 bytes, and the function runs under AVX2.
    unsafe {
        let even_odd = _mm256_load_si256(EVEN_ODD.0.as_ptr().cast());
        // `w[l][t]` is word `l` of inputs `t`, `8 + t`, `16 + t` and `24 + t`: [`absorb`] undone.
        let mut w = [[_mm256_setzero_si256(); 8]; 4];
        for (l, w) in w.iter_mut().enumerate() {
            for k in 0..P {
                let a = _mm256_shuffle_epi8(ld(&states[0].0[l][k]), even_odd);
                let b = _mm256_shuffle_epi8(ld(&states[1].0[l][k]), even_odd);
                w[k] = _mm256_unpacklo_epi64(a, b);
                w[k + 4] = _mm256_unpackhi_epi64(a, b);
            }
            transpose_bits(w);
        }
        for t in 0..8 {
            let rows = transpose4([w[0][t], w[1][t], w[2][t], w[3][t]]);
            for (m, row) in rows.into_iter().enumerate() {
                _mm256_storeu_si256(out.as_mut_ptr().add((8 * m + t) * OUT_LEN).cast(), row);
            }
        }
    }
}

/// Keccak-256 of the leading whole sets of 32 of the `out.len() / OUT_LEN` inputs of `len` bytes, from the
/// sponge state `init`. Returns how many inputs it hashed: none unless `len` is whole words.
#[target_feature(enable = "avx2")]
fn hash_sets(data: &[u8], len: usize, init: &[u64; STATE_LANES], out: &mut [u8]) -> usize {
    let sets = out.len() / OUT_LEN / SET;
    if sets == 0 || !len.is_multiple_of(8) {
        return 0;
    }
    /// The padding's two bits as planes.
    const PAD: [Plane; P] = iota_planes(PAD_FIRST as u64);
    const END: [Plane; P] = iota_planes(END_BIT);
    let (pad, end) = (PAD, END);
    // A Merkle tree starts almost every sponge from zero, and a call may be one set of nodes: no setup then.
    let mut start = Planes::ZERO;
    if init.iter().any(|&word| word != 0) {
        for (lane, &word) in start.0.iter_mut().zip(init) {
            *lane = iota_planes(word);
        }
    }
    let (blocks, words) = (len / RATE, len % RATE / 8);
    let mut states = [start.clone(), start.clone()];
    let mut scratch = [Scratch::ZERO, Scratch::ZERO];
    for s in 0..sets {
        let data = &data[s * SET * len..][..SET * len];
        states[0].0 = start.0;
        states[1].0 = start.0;
        // SAFETY: the function carries AVX2.
        unsafe {
            for block in 0..blocks {
                absorb(&mut states, data, len, block * RATE, RATE_LANES);
                permute(&mut states, &mut scratch);
            }
            // The final block: its whole words, then the padding, which every input has in the same lanes.
            absorb(&mut states, data, len, blocks * RATE, words);
            for state in states.iter_mut() {
                for k in 0..P {
                    for g in 0..4 {
                        state.0[words][k][g] ^= pad[k][g];
                        state.0[RATE_LANES - 1][k][g] ^= end[k][g];
                    }
                }
            }
            permute(&mut states, &mut scratch);
            digests(&states, &mut out[s * SET * OUT_LEN..][..SET * OUT_LEN]);
        }
    }
    sets * SET
}

/// Nonces a [`Grind`] tests per call.
pub const GRIND_NONCES: usize = SET;

/// The tested bits of a grind as planes, aligned for [`ld`].
#[repr(C, align(32))]
struct Mask([Plane; P]);

/// A grind's fixed part in this layout: the state with a zero nonce lane, and the tested bits.
///
/// A call tests [`GRIND_NONCES`] consecutive nonces. All `ROUNDS` rounds run in full, so any output lane serves.
pub struct Grind {
    start: Planes,
    lane: usize,
    out_lane: usize,
    mask: Mask,
}

impl Grind {
    /// The grind of `state` with the nonce in lane `lane`, testing that `mask` has no bit set in lane `out_lane`.
    pub fn new(state: &[u64; STATE_LANES], lane: usize, out_lane: usize, mask: u64) -> Self {
        let mut start = Planes::ZERO;
        for (i, (planes, &word)) in start.0.iter_mut().zip(state).enumerate() {
            if i != lane {
                *planes = iota_planes(word);
            }
        }
        Self {
            start,
            lane,
            out_lane,
            mask: Mask(iota_planes(mask)),
        }
    }

    /// The smallest `i < GRIND_NONCES` for which nonce `first + i` leaves no bit of the mask set in the output lane.
    pub fn first_hit(&self, first: u64) -> Option<usize> {
        let mut states = [self.start.clone(), self.start.clone()];
        let mut scratch = [Scratch::ZERO, Scratch::ZERO];
        // SAFETY: AVX2 is enabled at compile time, and every plane is a live block of its own.
        unsafe {
            // The nonces as planes, as `absorb` lays out one word of 32 inputs: input `8m + t` is state `t`
            // of group `m / 2` of batch `m % 2`.
            let word = |j: usize| first.wrapping_add(j as u64) as i64;
            let mut u: [__m256i; 8] =
                std::array::from_fn(|t| _mm256_set_epi64x(word(24 + t), word(16 + t), word(8 + t), word(t)));
            transpose_bits(&mut u);
            for k in 0..P {
                st(&mut states[0].0[self.lane][k], _mm256_unpacklo_epi8(u[k], u[k + 4]));
                st(&mut states[1].0[self.lane][k], _mm256_unpackhi_epi8(u[k], u[k + 4]));
            }
            permute(&mut states, &mut scratch);
            // Byte `m` of `fail` holds, in bit `t`, whether input `8m + t` has a tested bit set.
            let mut fail = [0u8; 4];
            for (batch, state) in states.iter().enumerate() {
                let mut any = _mm256_setzero_si256();
                for k in 0..P {
                    any = _mm256_or_si256(
                        any,
                        _mm256_and_si256(ld(&state.0[self.out_lane][k]), ld(&self.mask.0[k])),
                    );
                }
                // Each group of eight states holds sixteen bytes, one per `q`: OR them into one.
                let words: [u64; 4] = std::mem::transmute(any);
                for g in 0..2 {
                    let w = words[2 * g] | words[2 * g + 1];
                    let w = w | w >> 32;
                    let w = w | w >> 16;
                    fail[2 * g + batch] = (w | w >> 8) as u8;
                }
            }
            let hits = !u32::from_le_bytes(fail);
            (hits != 0).then(|| hits.trailing_zeros() as usize)
        }
    }
}

/// Hash the leading whole sets of 32 inputs of `data` with [`hash_sets`].
///
/// Returns what is left of `data` and `out` for another kernel: all of it unless `len` is whole words.
pub(crate) fn hash_sets_of_32<'a, 'b>(
    data: &'a [u8],
    len: usize,
    init: &[u64; STATE_LANES],
    out: &'b mut [u8],
) -> (&'a [u8], &'b mut [u8]) {
    assert_eq!(data.len(), out.len() / OUT_LEN * len);
    // SAFETY: AVX2 is enabled at compile time.
    #[allow(unused_unsafe)]
    let done = unsafe { hash_sets(data, len, init, out) };
    (&data[done * len..], &mut out[done * OUT_LEN..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{hash_from_state, hash_many_from_state, keccak_f, zero_prefix_state};

    /// A xorshift generator: every call is a new word.
    struct Rng(u64);

    impl Rng {
        fn word(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn bytes(&mut self, n: usize) -> Vec<u8> {
            (0..n).map(|_| (self.word() >> 24) as u8).collect()
        }

        fn state(&mut self) -> [u64; STATE_LANES] {
            std::array::from_fn(|_| self.word())
        }
    }

    /// Slice `states` bit by bit. Slow: for tests.
    fn slice_naive(states: &[[u64; STATE_LANES]], out: &mut Planes) {
        assert_eq!(states.len(), BATCH);
        *out = Planes::ZERO;
        for (s, state) in states.iter().enumerate() {
            let (g, t) = (s / 8, s % 8);
            for i in 0..STATE_LANES {
                for z in 0..64 {
                    let (q, k) = (z / P, z % P);
                    out.0[i][k][2 * g + q / 8] |= (state[i] >> z & 1) << (8 * (q % 8) + t);
                }
            }
        }
    }

    /// The inverse of [`slice_naive`].
    fn unslice_naive(planes: &Planes, states: &mut [[u64; STATE_LANES]]) {
        assert_eq!(states.len(), BATCH);
        for (s, state) in states.iter_mut().enumerate() {
            let (g, t) = (s / 8, s % 8);
            for i in 0..STATE_LANES {
                state[i] = (0..64).fold(0, |acc, z| {
                    let (q, k) = (z / P, z % P);
                    acc | (planes.0[i][k][2 * g + q / 8] >> (8 * (q % 8) + t) & 1) << z
                });
            }
        }
    }

    /// The permutation of 16 states per batch is the scalar one of each.
    #[test]
    fn permutation_matches_scalar() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for groups in [1usize, 2, 3] {
            let mut states: Vec<[u64; STATE_LANES]> = (0..groups * BATCH).map(|_| rng.state()).collect();
            let mut batches = vec![Planes::ZERO; groups];
            let mut scratch: Vec<Scratch> = (0..groups).map(|_| Scratch::ZERO).collect();
            for (planes, states) in batches.iter_mut().zip(states.chunks(BATCH)) {
                slice_naive(states, planes);
            }
            // SAFETY: the test is compiled only with AVX2 enabled.
            #[allow(unused_unsafe)]
            unsafe {
                permute(&mut batches, &mut scratch)
            };
            let mut got = states.clone();
            for (planes, got) in batches.iter().zip(got.chunks_mut(BATCH)) {
                unslice_naive(planes, got);
            }
            for state in &mut states {
                keccak_f(state);
            }
            assert_eq!(got, states, "{groups} batches");
        }
    }

    const COUNTS: [usize; 9] = [0, 1, 31, 32, 33, 63, 64, 95, 256];
    const LENGTHS: [usize; 11] = [0, 8, 16, 120, 128, 136, 256, 264, 272, 400, 512];
    /// A guard byte around `out`, which a kernel must leave alone.
    const GUARD: u8 = 0xa5;

    /// Starting sponge states: zero, after one zero block, and random words.
    fn inits(rng: &mut Rng) -> [[u64; STATE_LANES]; 3] {
        [[0; STATE_LANES], zero_prefix_state(1), rng.state()]
    }

    /// Random inputs, one per word of a stream, from byte `offset` of their buffer.
    fn random_inputs(rng: &mut Rng, n: usize, len: usize, offset: usize) -> Vec<u8> {
        let mut buf = vec![0; offset];
        buf.extend(rng.bytes(n * len));
        buf
    }

    /// Check every digest of `out` that `done` inputs cover against the scalar hash, and the guards.
    fn check(data: &[u8], len: usize, init: &[u64; STATE_LANES], out: &[u8], done: usize, what: &str) {
        for i in 0..out.len() / OUT_LEN {
            let digest = &out[i * OUT_LEN..(i + 1) * OUT_LEN];
            if i < done {
                let want = hash_from_state(&data[i * len..(i + 1) * len], init);
                assert_eq!(digest, &want[..], "{what}: input {i}");
            } else {
                assert!(digest.iter().all(|&b| b == GUARD), "{what}: output {i} was written");
            }
        }
    }

    /// The kernel's digests are the scalar ones, for random inputs and starts, unaligned data and output.
    #[test]
    fn sets_match_scalar() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for init in inits(&mut rng) {
            for len in LENGTHS {
                for n in COUNTS {
                    for (data_at, out_at) in [(0, 0), (1, 1), (3, 5)] {
                        let what = format!("len {len}, count {n}, offsets {data_at} and {out_at}");
                        let buf = random_inputs(&mut rng, n, len, data_at);
                        let data = &buf[data_at..];
                        let mut out_buf = vec![GUARD; out_at + n * OUT_LEN];
                        let out = &mut out_buf[out_at..];
                        // SAFETY: the test is compiled only with AVX2 enabled.
                        #[allow(unused_unsafe)]
                        let done = unsafe { hash_sets(data, len, &init, out) };
                        assert_eq!(done, n / SET * SET, "{what}: inputs hashed");
                        check(data, len, &init, out, done, &what);
                        assert!(
                            out_buf[..out_at].iter().all(|&b| b == GUARD),
                            "{what}: wrote before out"
                        );
                    }
                }
            }
        }
    }

    /// Lengths that are not whole words go to another kernel: nothing is hashed or written.
    #[test]
    fn partial_words_are_left_alone() {
        let mut rng = Rng(0x1234_5678_9abc_def1);
        for len in [1usize, 5, 7, 9, 135, 137, 513] {
            let data = rng.bytes(64 * len);
            let mut out = vec![GUARD; 64 * OUT_LEN];
            // SAFETY: the test is compiled only with AVX2 enabled.
            #[allow(unused_unsafe)]
            let done = unsafe { hash_sets(&data, len, &[0; STATE_LANES], &mut out) };
            assert_eq!(done, 0, "len {len}");
            assert!(out.iter().all(|&b| b == GUARD), "len {len}");
        }
    }

    /// The entry point agrees with the scalar hash on every count, whatever kernel takes each part.
    #[test]
    fn entry_point_matches_scalar() {
        let mut rng = Rng(0x0dd_ba11_cafe_f00d);
        for init in inits(&mut rng) {
            for len in LENGTHS.into_iter().chain([5, 137, 513]) {
                for n in COUNTS {
                    let data = random_inputs(&mut rng, n, len, 0);
                    let mut out = vec![0; n * OUT_LEN];
                    hash_many_from_state(&data, len, &init, &mut out);
                    check(&data, len, &init, &out, n, &format!("len {len}, count {n}"));
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn output_must_fit_the_data() {
        hash_sets_of_32(&[0; 10], 8, &[0; STATE_LANES], &mut [0; 2 * OUT_LEN]);
    }
}
