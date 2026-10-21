//! Haraka-512 on AES-NI: one `aesenc` per AES round per state, the dword mix as unpacks.

use super::*;
use std::arch::x86_64::*;

/// Calls one batched step keeps in flight: sixteen AES states cover the instruction's latency.
pub const WIDTH: usize = 4;

#[inline]
unsafe fn ld(p: &[u8]) -> __m128i {
    unsafe { _mm_loadu_si128(p.as_ptr() as *const __m128i) }
}
#[inline]
unsafe fn st(p: &mut [u8], v: __m128i) {
    unsafe { _mm_storeu_si128(p.as_mut_ptr() as *mut __m128i, v) }
}

#[inline]
fn round_constants() -> &'static [__m128i; 40] {
    use std::sync::OnceLock;
    static RC: OnceLock<[__m128i; 40]> = OnceLock::new();
    // SAFETY: a plain unaligned load of each 16-byte constant.
    RC.get_or_init(|| std::array::from_fn(|i| unsafe { ld(&rc_bytes(i)) }))
}

#[inline]
unsafe fn mix(s: &mut [__m128i; STATES]) {
    unsafe {
        let tmp = _mm_unpacklo_epi32(s[0], s[1]);
        s[0] = _mm_unpackhi_epi32(s[0], s[1]);
        s[1] = _mm_unpacklo_epi32(s[2], s[3]);
        s[2] = _mm_unpackhi_epi32(s[2], s[3]);
        s[3] = _mm_unpacklo_epi32(s[0], s[2]);
        s[0] = _mm_unpackhi_epi32(s[0], s[2]);
        s[2] = _mm_unpackhi_epi32(s[1], tmp);
        s[1] = _mm_unpacklo_epi32(s[1], tmp);
    }
}

/// The truncated feed-forward: `(state0.hi, state1.hi, state2.lo, state3.lo)` of `s ⊕ msg`.
#[inline]
unsafe fn truncate(s: &[__m128i; STATES], m: &[__m128i; STATES], out: &mut [u8]) {
    unsafe {
        let f: [__m128i; STATES] = std::array::from_fn(|b| _mm_xor_si128(s[b], m[b]));
        st(&mut out[..16], _mm_unpackhi_epi64(f[0], f[1]));
        st(&mut out[16..32], _mm_unpacklo_epi64(f[2], f[3]));
    }
}

pub fn haraka512(msg: &[u8; IN_LEN]) -> [u8; OUT_LEN] {
    let rc = round_constants();
    let mut out = [0u8; OUT_LEN];
    // SAFETY: the target features are enabled at compile time.
    unsafe {
        let m: [__m128i; STATES] = std::array::from_fn(|b| ld(&msg[16 * b..16 * b + 16]));
        let mut s = m;
        for r in 0..ROUNDS {
            for j in 0..AES_PER_ROUND {
                for blk in 0..STATES {
                    s[blk] = _mm_aesenc_si128(s[blk], rc[rc_index(r, j, blk)]);
                }
            }
            mix(&mut s);
        }
        truncate(&s, &m, &mut out);
    }
    out
}

/// [`WIDTH`] calls interleaved: input `i` at `msgs[64i..]`, digest `i` at `outs[32i..]`.
pub fn haraka512_x4(msgs: &[u8], outs: &mut [u8]) {
    let rc = round_constants();
    // SAFETY: the target features are enabled at compile time; the slices are sized by the caller.
    unsafe {
        let m: [[__m128i; STATES]; WIDTH] =
            std::array::from_fn(|i| std::array::from_fn(|b| ld(&msgs[64 * i + 16 * b..64 * i + 16 * b + 16])));
        let mut s = m;
        for r in 0..ROUNDS {
            for j in 0..AES_PER_ROUND {
                for blk in 0..STATES {
                    let k = rc[rc_index(r, j, blk)];
                    for lane in &mut s {
                        lane[blk] = _mm_aesenc_si128(lane[blk], k);
                    }
                }
            }
            for lane in &mut s {
                mix(lane);
            }
        }
        for i in 0..WIDTH {
            truncate(&s[i], &m[i], &mut outs[32 * i..32 * i + 32]);
        }
    }
}

#[cfg(target_feature = "gfni")]
const GFNI_ID: i64 = 0x0102040810204080;

/// Same contract as [`haraka512_witness_scalar`], the inverses read off the state with GFNI.
#[cfg(target_feature = "gfni")]
pub fn haraka512_witness(
    msg: &[u8; IN_LEN],
    xs: &mut [u8; SBOX_PER_CALL],
    us: &mut [u8; SBOX_PER_CALL],
) -> [u8; OUT_LEN] {
    let rc = round_constants();
    let mut out = [0u8; OUT_LEN];
    // SAFETY: the target features are enabled at compile time.
    unsafe {
        let id = _mm_set1_epi64x(GFNI_ID);
        let m: [__m128i; STATES] = std::array::from_fn(|b| ld(&msg[16 * b..16 * b + 16]));
        let mut s = m;
        for r in 0..ROUNDS {
            for j in 0..AES_PER_ROUND {
                for blk in 0..STATES {
                    let k = sbox_index(r, j, blk, 0);
                    st(&mut xs[k..k + 16], s[blk]);
                    st(&mut us[k..k + 16], _mm_gf2p8affineinv_epi64_epi8::<0>(s[blk], id));
                    s[blk] = _mm_aesenc_si128(s[blk], rc[rc_index(r, j, blk)]);
                }
            }
            mix(&mut s);
        }
        truncate(&s, &m, &mut out);
    }
    out
}
