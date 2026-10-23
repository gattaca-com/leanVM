//! The x86-64 backends: the AVX2 and AVX-512 batches.

#[cfg(target_feature = "avx512f")]
mod zmm;

#[cfg(target_feature = "avx2")]
use super::batch::Lanes32;
// Only the AVX-512 digest store needs it.
#[cfg(target_feature = "avx512f")]
use super::OUT_LEN;
#[cfg(target_feature = "avx2")]
use core::arch::x86_64::*;

/// AVX2: eight lanes.
///
/// There is no 32-bit vector rotate before AVX-512.
///
/// The byte-aligned rotations (16, 8) become one `vpshufb`, the others (12, 7) a shift pair.
///
/// On an AVX-512 target only the tests use it.
#[cfg(target_feature = "avx2")]
#[cfg_attr(target_feature = "avx512f", allow(dead_code))]
#[derive(Clone, Copy)]
pub(super) struct Avx2(__m256i);

#[cfg(target_feature = "avx2")]
impl Lanes32 for Avx2 {
    const WIDTH: usize = 8;

    #[inline(always)]
    unsafe fn load(p: *const u32) -> Self {
        Self(unsafe { _mm256_loadu_si256(p.cast()) })
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut u32) {
        unsafe { _mm256_storeu_si256(p.cast(), self.0) }
    }
    #[inline(always)]
    fn splat(x: u32) -> Self {
        Self(unsafe { _mm256_set1_epi32(x as i32) })
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Self(unsafe { _mm256_add_epi32(self.0, o.0) })
    }
    #[inline(always)]
    fn xor(self, o: Self) -> Self {
        Self(unsafe { _mm256_xor_si256(self.0, o.0) })
    }
    #[inline(always)]
    fn rotr<const N: u32>(self) -> Self {
        // Byte shuffles rotating each 32-bit lane right by 2 and by 1 bytes, per 16-byte half.
        const ROT16: [i8; 16] = [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13];
        const ROT8: [i8; 16] = [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12];
        unsafe {
            let shuf = |m: [i8; 16]| {
                let half = _mm_loadu_si128(m.as_ptr().cast());
                Self(_mm256_shuffle_epi8(self.0, _mm256_set_m128i(half, half)))
            };
            match N {
                16 => shuf(ROT16),
                8 => shuf(ROT8),
                12 => Self(_mm256_or_si256(
                    _mm256_srli_epi32(self.0, 12),
                    _mm256_slli_epi32(self.0, 20),
                )),
                7 => Self(_mm256_or_si256(
                    _mm256_srli_epi32(self.0, 7),
                    _mm256_slli_epi32(self.0, 25),
                )),
                _ => unreachable!("BLAKE2s rotates by 16, 12, 8 or 7"),
            }
        }
    }

    /// Two 8x8 transposes: each block is two `ymm`, words 0..8 and 8..16.
    #[inline(always)]
    unsafe fn transpose(src: *const u8, stride: usize, block: &mut [Self; 16]) {
        unsafe {
            for half in 0..2 {
                let r: [__m256i; 8] =
                    std::array::from_fn(|l| _mm256_loadu_si256(src.add(l * stride + 32 * half).cast()));
                let mut t = [_mm256_setzero_si256(); 8];
                for k in 0..4 {
                    t[2 * k] = _mm256_unpacklo_epi32(r[2 * k], r[2 * k + 1]);
                    t[2 * k + 1] = _mm256_unpackhi_epi32(r[2 * k], r[2 * k + 1]);
                }
                let s: [__m256i; 8] = [
                    _mm256_unpacklo_epi64(t[0], t[2]),
                    _mm256_unpackhi_epi64(t[0], t[2]),
                    _mm256_unpacklo_epi64(t[1], t[3]),
                    _mm256_unpackhi_epi64(t[1], t[3]),
                    _mm256_unpacklo_epi64(t[4], t[6]),
                    _mm256_unpackhi_epi64(t[4], t[6]),
                    _mm256_unpacklo_epi64(t[5], t[7]),
                    _mm256_unpackhi_epi64(t[5], t[7]),
                ];
                for k in 0..4 {
                    let w = 8 * half + k;
                    block[w] = Self(_mm256_permute2x128_si256(s[k], s[k + 4], 0x20));
                    block[w + 4] = Self(_mm256_permute2x128_si256(s[k], s[k + 4], 0x31));
                }
            }
        }
    }
}

/// AVX-512: sixteen lanes, and `vprord` makes every rotation one instruction.
///
/// Two groups run together, so neither waits on its own dependency chain.
#[cfg(target_feature = "avx512f")]
#[derive(Clone, Copy)]
pub(super) struct Avx512(__m512i);

#[cfg(target_feature = "avx512f")]
impl Lanes32 for Avx512 {
    const WIDTH: usize = 16;

    #[inline(always)]
    unsafe fn load(p: *const u32) -> Self {
        Self(unsafe { _mm512_loadu_si512(p.cast()) })
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut u32) {
        unsafe { _mm512_storeu_si512(p.cast(), self.0) }
    }
    #[inline(always)]
    fn splat(x: u32) -> Self {
        Self(unsafe { _mm512_set1_epi32(x as i32) })
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Self(unsafe { _mm512_add_epi32(self.0, o.0) })
    }
    #[inline(always)]
    fn xor(self, o: Self) -> Self {
        Self(unsafe { _mm512_xor_si512(self.0, o.0) })
    }
    #[inline(always)]
    fn rotr<const N: u32>(self) -> Self {
        unsafe {
            match N {
                16 => Self(_mm512_ror_epi32(self.0, 16)),
                12 => Self(_mm512_ror_epi32(self.0, 12)),
                8 => Self(_mm512_ror_epi32(self.0, 8)),
                7 => Self(_mm512_ror_epi32(self.0, 7)),
                _ => unreachable!("BLAKE2s rotates by 16, 12, 8 or 7"),
            }
        }
    }

    /// Two groups: 32 state vectors, the whole register file.
    const GROUPS: usize = 2;

    #[inline(always)]
    unsafe fn compress_groups<const G: usize>(h: &mut [[Self; 8]; G], m: [&[Self; 16]; G], t: u64, last: bool) {
        // A lone tail group takes the generic rounds.
        if G != 2 {
            // SAFETY: forwarded from the caller.
            return unsafe { super::batch::compress_groups::<Self, G>(h, m, t, last) };
        }
        // SAFETY: the kernel's layouts, two groups of 8 vectors and two blocks of 16.
        unsafe {
            zmm::compress_x2(
                h.as_mut_ptr().cast(),
                m[0].as_ptr().cast(),
                m[1].as_ptr().cast(),
                t,
                last,
            )
        }
    }

    /// An 8x16 to 16x8 transpose for the digests.
    ///
    /// Two unpack phases leave each digest in two 128-bit lanes:
    ///
    /// ```text
    ///     digest 4L + c    lane L of s[c], then lane L of s[4 + c]
    /// ```
    ///
    /// One `vpermt2q` joins the halves of two digests, 4 apart, and two 256-bit stores write them.
    #[inline(always)]
    unsafe fn store_digests(h: &[Self; 8], out: *mut u8) {
        unsafe {
            let mut s = [_mm512_setzero_si512(); 8];
            for a in 0..2 {
                let (r0, r1, r2, r3) = (h[4 * a].0, h[4 * a + 1].0, h[4 * a + 2].0, h[4 * a + 3].0);
                let lo01 = _mm512_unpacklo_epi32(r0, r1);
                let hi01 = _mm512_unpackhi_epi32(r0, r1);
                let lo23 = _mm512_unpacklo_epi32(r2, r3);
                let hi23 = _mm512_unpackhi_epi32(r2, r3);
                s[4 * a] = _mm512_unpacklo_epi64(lo01, lo23);
                s[4 * a + 1] = _mm512_unpackhi_epi64(lo01, lo23);
                s[4 * a + 2] = _mm512_unpacklo_epi64(hi01, hi23);
                s[4 * a + 3] = _mm512_unpackhi_epi64(hi01, hi23);
            }
            // Quadwords 0..8 are `s[c]`, 8..16 are `s[4 + c]`: pick lanes 2p and 2p + 1 of both, interleaved.
            let idx = [
                _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0),
                _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4),
            ];
            for c in 0..4 {
                for (p, &i) in idx.iter().enumerate() {
                    let v = _mm512_permutex2var_epi64(s[c], i, s[4 + c]);
                    let d = out.add((8 * p + c) * OUT_LEN);
                    _mm256_storeu_si256(d.cast(), _mm512_castsi512_si256(v));
                    _mm256_storeu_si256(d.add(4 * OUT_LEN).cast(), _mm512_extracti64x4_epi64::<1>(v));
                }
            }
        }
    }

    /// A 16x16 transpose: each block is exactly one `zmm`.
    ///
    /// ```text
    ///     2 unpack phases           four rows of one column per 128-bit lane
    ///     2 shuffle_i32x4 phases    collect the four row groups
    /// ```
    #[inline(always)]
    unsafe fn transpose(src: *const u8, stride: usize, block: &mut [Self; 16]) {
        unsafe {
            let r: [__m512i; 16] = std::array::from_fn(|l| _mm512_loadu_si512(src.add(l * stride).cast()));
            // Phases 1 and 2: lane L of s[4a + c] holds column 4L + c of rows 4a..4a + 4.
            let mut s = [_mm512_setzero_si512(); 16];
            for a in 0..4 {
                let (r0, r1, r2, r3) = (r[4 * a], r[4 * a + 1], r[4 * a + 2], r[4 * a + 3]);
                let lo01 = _mm512_unpacklo_epi32(r0, r1);
                let hi01 = _mm512_unpackhi_epi32(r0, r1);
                let lo23 = _mm512_unpacklo_epi32(r2, r3);
                let hi23 = _mm512_unpackhi_epi32(r2, r3);
                s[4 * a] = _mm512_unpacklo_epi64(lo01, lo23);
                s[4 * a + 1] = _mm512_unpackhi_epi64(lo01, lo23);
                s[4 * a + 2] = _mm512_unpacklo_epi64(hi01, hi23);
                s[4 * a + 3] = _mm512_unpackhi_epi64(hi01, hi23);
            }
            // Phases 3 and 4: per c, a 4x4 transpose of 128-bit lanes over s[c], s[4 + c], s[8 + c], s[12 + c].
            //
            // Stage one joins the low lane pair or the high lane pair of two inputs, stage two takes their even or odd lanes.
            for c in 0..4 {
                let lo = |x: usize, y: usize| _mm512_shuffle_i32x4::<0x44>(s[x], s[y]);
                let hi = |x: usize, y: usize| _mm512_shuffle_i32x4::<0xEE>(s[x], s[y]);
                let (t0, t1) = (lo(c, 4 + c), hi(c, 4 + c));
                let (t2, t3) = (lo(8 + c, 12 + c), hi(8 + c, 12 + c));
                block[c] = Self(_mm512_shuffle_i32x4::<0x88>(t0, t2));
                block[4 + c] = Self(_mm512_shuffle_i32x4::<0xDD>(t0, t2));
                block[8 + c] = Self(_mm512_shuffle_i32x4::<0x88>(t1, t3));
                block[12 + c] = Self(_mm512_shuffle_i32x4::<0xDD>(t1, t3));
            }
        }
    }
}

/// One compression with the state held as four rows of four words.
///
/// The four column G steps run as one vector step, then the diagonal ones after a lane rotation.
/// A lone compression has no second input to fill lanes with, so this is its only parallelism.
#[cfg(target_feature = "avx512vl")]
#[inline]
pub(super) fn compress_rows(h: &[u32; 8], m: &[u32; 16], t: u64, f0: u32, f1: u32) -> [u32; 8] {
    use super::{IV, SIGMA};

    #[inline(always)]
    fn g(a: &mut __m128i, b: &mut __m128i, c: &mut __m128i, d: &mut __m128i, mx: __m128i, my: __m128i) {
        // SAFETY: the module is compiled only with AVX-512VL enabled.
        unsafe {
            *a = _mm_add_epi32(_mm_add_epi32(*a, *b), mx);
            *d = _mm_ror_epi32::<16>(_mm_xor_si128(*d, *a));
            *c = _mm_add_epi32(*c, *d);
            *b = _mm_ror_epi32::<12>(_mm_xor_si128(*b, *c));
            *a = _mm_add_epi32(_mm_add_epi32(*a, *b), my);
            *d = _mm_ror_epi32::<8>(_mm_xor_si128(*d, *a));
            *c = _mm_add_epi32(*c, *d);
            *b = _mm_ror_epi32::<7>(_mm_xor_si128(*b, *c));
        }
    }

    let words = |s: &[usize; 16], k: usize| {
        // SAFETY: plain register construction, no memory access.
        unsafe { _mm_setr_epi32(m[s[k]] as i32, m[s[k + 2]] as i32, m[s[k + 4]] as i32, m[s[k + 6]] as i32) }
    };
    // SAFETY: the loads read 16 bytes inside `h` and `IV`, unaligned loads allowed.
    unsafe {
        let mut a = _mm_loadu_si128(h.as_ptr().cast());
        let mut b = _mm_loadu_si128(h.as_ptr().add(4).cast());
        let mut c = _mm_loadu_si128(IV.as_ptr().cast());
        let flags = _mm_setr_epi32(t as i32, (t >> 32) as i32, f0 as i32, f1 as i32);
        let mut d = _mm_xor_si128(_mm_loadu_si128(IV.as_ptr().add(4).cast()), flags);
        for s in &SIGMA {
            g(&mut a, &mut b, &mut c, &mut d, words(s, 0), words(s, 1));
            // Diagonals into columns: row r turns left by r lanes.
            b = _mm_shuffle_epi32::<0b00_11_10_01>(b);
            c = _mm_shuffle_epi32::<0b01_00_11_10>(c);
            d = _mm_shuffle_epi32::<0b10_01_00_11>(d);
            g(&mut a, &mut b, &mut c, &mut d, words(s, 8), words(s, 9));
            b = _mm_shuffle_epi32::<0b10_01_00_11>(b);
            c = _mm_shuffle_epi32::<0b01_00_11_10>(c);
            d = _mm_shuffle_epi32::<0b00_11_10_01>(d);
        }
        let lo = _mm_xor_si128(_mm_loadu_si128(h.as_ptr().cast()), _mm_xor_si128(a, c));
        let hi = _mm_xor_si128(_mm_loadu_si128(h.as_ptr().add(4).cast()), _mm_xor_si128(b, d));
        let mut out = [0u32; 8];
        _mm_storeu_si128(out.as_mut_ptr().cast(), lo);
        _mm_storeu_si128(out.as_mut_ptr().add(4).cast(), hi);
        out
    }
}
