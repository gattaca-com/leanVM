//! The round-1 shift-reduce over 64 windows at once, one window per byte lane.
//!
//! The table kernel reads sixteen 64-byte table rows per row pair, which is the load
//! ports' limit. Here byte `p` of 64 consecutive windows is one register, a plane. The
//! extension of lane `j` is eight `vgf2p8affineqb`s, one per byte `b` of a row, by the
//! 8x8 bit matrix that byte `b` contributes to lane `j`. The matrices are broadcast
//! qwords shared by `a` and `b`, so the loop is bound by the affine pipes, not by loads.
//!
//! The planes come from a 64x64 byte transpose of the windows, and the shift-reduced
//! lanes go back through one.

use core::arch::x86_64::*;

use pcs::ntt::InvNttTableByteSingleGf8;

use super::{ELL, N_CHUNKS};

/// Windows per group.
pub(super) const GROUP: usize = 64;
/// Bytes of `a` (or `b`) one window reads.
pub(super) const WINDOW_BYTES: usize = N_CHUNKS * 8;

/// `mats[j][b]`: the `vgf2p8affineqb` qword for byte `b` of a window row entering lane `j`.
pub(super) struct Matrices([[u64; 8]; ELL]);

impl Matrices {
    pub(super) fn new(inv_table: &InvNttTableByteSingleGf8) -> Self {
        let table = inv_table.data_ptr();
        let mut mats = [[0u64; 8]; ELL];
        for (j, row) in mats.iter_mut().enumerate() {
            for (b, mat) in row.iter_mut().enumerate() {
                // Unit row `1 << s` of the table is the column of input bit `s`, and
                // byte `b` enters lane `j` at `j ^ 8b`.
                for s in 0..8 {
                    // SAFETY: the table is `256 * 64` bytes and `j ^ 8b < 64`.
                    let column = unsafe { *table.add((64 << s) + (j ^ (8 * b))) };
                    for i in 0..8 {
                        // Result bit `i` is the parity of matrix byte `7 - i` with the input.
                        *mat |= u64::from(column >> i & 1) << (8 * (7 - i) + s);
                    }
                }
            }
        }
        Self(mats)
    }
}

/// What the group kernel works in: both operands' planes, the Horner chain per lane, and a transpose stage.
pub(super) struct Scratch {
    planes_a: [__m512i; GROUP],
    planes_b: [__m512i; GROUP],
    acc: [__m512i; ELL],
    transposed: [__m512i; GROUP],
}

impl Scratch {
    pub(super) fn new() -> Box<Self> {
        // SAFETY: an all-zero bit pattern is a valid register value.
        unsafe { Box::new_zeroed().assume_init() }
    }
}

/// `dst[c * 64 + r] = src[r * 64 + c]` for the 64x64 bytes at `src`.
///
/// Two rounds of 8x8 transposes: bytes within each qword across 8 rows, then qwords across
/// 8 groups of rows.
///
/// # Safety
/// `src` and `dst` are 4096 bytes, `src` readable and `dst` writable, and they do not overlap.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn transpose64(src: *const u8, dst: *mut u8, scratch: &mut [__m512i; GROUP]) {
    unsafe {
        for g in 0..8 {
            let r: [__m512i; 8] = core::array::from_fn(|i| _mm512_loadu_si512(src.add((8 * g + i) * 64).cast()));
            let t: [__m512i; 8] = [
                _mm512_unpacklo_epi8(r[0], r[1]),
                _mm512_unpackhi_epi8(r[0], r[1]),
                _mm512_unpacklo_epi8(r[2], r[3]),
                _mm512_unpackhi_epi8(r[2], r[3]),
                _mm512_unpacklo_epi8(r[4], r[5]),
                _mm512_unpackhi_epi8(r[4], r[5]),
                _mm512_unpacklo_epi8(r[6], r[7]),
                _mm512_unpackhi_epi8(r[6], r[7]),
            ];
            let s: [__m512i; 8] = [
                _mm512_unpacklo_epi16(t[0], t[2]),
                _mm512_unpackhi_epi16(t[0], t[2]),
                _mm512_unpacklo_epi16(t[1], t[3]),
                _mm512_unpackhi_epi16(t[1], t[3]),
                _mm512_unpacklo_epi16(t[4], t[6]),
                _mm512_unpackhi_epi16(t[4], t[6]),
                _mm512_unpacklo_epi16(t[5], t[7]),
                _mm512_unpackhi_epi16(t[5], t[7]),
            ];
            scratch[8 * g] = _mm512_unpacklo_epi32(s[0], s[4]);
            scratch[8 * g + 1] = _mm512_unpackhi_epi32(s[0], s[4]);
            scratch[8 * g + 2] = _mm512_unpacklo_epi32(s[1], s[5]);
            scratch[8 * g + 3] = _mm512_unpackhi_epi32(s[1], s[5]);
            scratch[8 * g + 4] = _mm512_unpacklo_epi32(s[2], s[6]);
            scratch[8 * g + 5] = _mm512_unpackhi_epi32(s[2], s[6]);
            scratch[8 * g + 6] = _mm512_unpacklo_epi32(s[3], s[7]);
            scratch[8 * g + 7] = _mm512_unpackhi_epi32(s[3], s[7]);
        }
        // Register `x` of group `g` now holds, in qword `q`, the 8 bytes of rows `8g..8g+8` at
        // column `8u + b`, for `u = 2 (q / 2) + x / 4` and `b = 2 (x % 4) + q % 2`.
        for x in 0..8 {
            let a: [__m512i; 8] = core::array::from_fn(|g| scratch[8 * g + x]);
            let lo = [
                _mm512_unpacklo_epi64(a[0], a[1]),
                _mm512_unpacklo_epi64(a[2], a[3]),
                _mm512_unpacklo_epi64(a[4], a[5]),
                _mm512_unpacklo_epi64(a[6], a[7]),
            ];
            let hi = [
                _mm512_unpackhi_epi64(a[0], a[1]),
                _mm512_unpackhi_epi64(a[2], a[3]),
                _mm512_unpackhi_epi64(a[4], a[5]),
                _mm512_unpackhi_epi64(a[6], a[7]),
            ];
            // Qwords 0 and 4, then 2 and 6, come from `lo`; 1 and 5, then 3 and 7, from `hi`.
            let columns = |v: [__m512i; 4], first: usize, second: usize, out: &mut [__m512i; 8]| {
                let b0 = _mm512_shuffle_i64x2::<0x88>(v[0], v[1]);
                let b1 = _mm512_shuffle_i64x2::<0xDD>(v[0], v[1]);
                let c0 = _mm512_shuffle_i64x2::<0x88>(v[2], v[3]);
                let c1 = _mm512_shuffle_i64x2::<0xDD>(v[2], v[3]);
                out[first] = _mm512_shuffle_i64x2::<0x88>(b0, c0);
                out[first + 4] = _mm512_shuffle_i64x2::<0xDD>(b0, c0);
                out[second] = _mm512_shuffle_i64x2::<0x88>(b1, c1);
                out[second + 4] = _mm512_shuffle_i64x2::<0xDD>(b1, c1);
            };
            let mut z = [_mm512_setzero_si512(); 8];
            columns(lo, 0, 2, &mut z);
            columns(hi, 1, 3, &mut z);
            for (q, z) in z.iter().enumerate() {
                let column = 8 * (2 * (q / 2) + x / 4) + 2 * (x % 4) + q % 2;
                _mm512_storeu_si512(dst.add(column * 64).cast(), *z);
            }
        }
    }
}

/// One byte-row `k` of 64 windows: both extensions of every lane, their product, and the Horner step.
///
/// `FIRST` starts the Horner chain, and a `prefetch` line is touched per lane, so that the next
/// group's rows arrive while this one computes.
#[inline(always)]
unsafe fn shift_reduce_row<const FIRST: bool, const PREFETCH: bool>(
    pa: &[__m512i; 8],
    pb: &[__m512i; 8],
    mats: &Matrices,
    acc: &mut [__m512i; ELL],
    prefetch: *const u8,
) {
    unsafe {
        let x = _mm512_set1_epi8(2);
        for (j, acc) in acc.iter_mut().enumerate() {
            if PREFETCH {
                _mm_prefetch::<_MM_HINT_T0>(prefetch.wrapping_add(64 * j).cast());
            }
            let m: [__m512i; 8] = core::array::from_fn(|i| _mm512_set1_epi64(mats.0[j][i] as i64));
            let mut ea = _mm512_gf2p8affine_epi64_epi8::<0>(pa[0], m[0]);
            let mut eb = _mm512_gf2p8affine_epi64_epi8::<0>(pb[0], m[0]);
            for i in 1..8 {
                ea = _mm512_xor_si512(ea, _mm512_gf2p8affine_epi64_epi8::<0>(pa[i], m[i]));
                eb = _mm512_xor_si512(eb, _mm512_gf2p8affine_epi64_epi8::<0>(pb[i], m[i]));
            }
            let y = _mm512_gf2p8mul_epi8(ea, eb);
            *acc = if FIRST {
                y
            } else {
                _mm512_xor_si512(_mm512_gf2p8mul_epi8(*acc, x), y)
            };
        }
    }
}

/// The 64 shift-reduced rows of 64 consecutive windows: `out[w][j]` is lane `j` of window `w`.
///
/// `prefetch` are three streams, 64 lines each, to touch meanwhile.
///
/// # Safety
/// `a` and `b` are readable for `GROUP * WINDOW_BYTES` bytes.
#[target_feature(enable = "avx512f,avx512bw,gfni")]
pub(super) unsafe fn shift_reduce_group(
    a: *const u8,
    b: *const u8,
    mats: &Matrices,
    scratch: &mut Scratch,
    out: &mut [[u8; ELL]; GROUP],
    prefetch: [*const u8; 3],
) {
    unsafe {
        let Scratch {
            planes_a,
            planes_b,
            acc,
            transposed,
        } = scratch;
        transpose64(a, planes_a.as_mut_ptr().cast(), transposed);
        transpose64(b, planes_b.as_mut_ptr().cast(), transposed);
        for k in (0..N_CHUNKS).rev() {
            let pa: [__m512i; 8] = core::array::from_fn(|i| planes_a[8 * k + i]);
            let pb: [__m512i; 8] = core::array::from_fn(|i| planes_b[8 * k + i]);
            match N_CHUNKS - 1 - k {
                0 => shift_reduce_row::<true, true>(&pa, &pb, mats, acc, prefetch[0]),
                1 => shift_reduce_row::<false, true>(&pa, &pb, mats, acc, prefetch[1]),
                2 => shift_reduce_row::<false, true>(&pa, &pb, mats, acc, prefetch[2]),
                _ => shift_reduce_row::<false, false>(&pa, &pb, mats, acc, core::ptr::null()),
            }
        }
        transpose64(acc.as_ptr().cast(), out.as_mut_ptr().cast(), transposed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pcs::ntt::AdditiveNttGf8;
    use primitives::field::F8;
    use primitives::test_rng::Rng;

    #[test]
    fn transpose_is_transpose() {
        let mut rng = Rng::new(5);
        let src: Vec<u8> = (0..4096).map(|_| rng.next_u8()).collect();
        let mut dst = vec![0u8; 4096];
        let mut scratch = Scratch::new();
        // SAFETY: the target features are enabled at compile time.
        unsafe { transpose64(src.as_ptr(), dst.as_mut_ptr(), &mut scratch.transposed) };
        for r in 0..64 {
            for c in 0..64 {
                assert_eq!(dst[c * 64 + r], src[r * 64 + c], "r={r} c={c}");
            }
        }
    }

    #[test]
    fn group_matches_table_kernel() {
        let inv = InvNttTableByteSingleGf8::new(&AdditiveNttGf8::new(6, F8::ZERO), &AdditiveNttGf8::new(6, F8(1 << 6)));
        let mut rng = Rng::new(9);
        let a: Vec<u8> = (0..GROUP * WINDOW_BYTES).map(|_| rng.next_u8()).collect();
        let b: Vec<u8> = (0..GROUP * WINDOW_BYTES).map(|_| rng.next_u8()).collect();
        let mut want = vec![[0u8; ELL]; GROUP];
        // SAFETY: the target features are enabled at compile time.
        unsafe { super::super::shift_reduce_ab_gfni_512(&a, &b, &inv, 0, &mut want) };
        let mut got = vec![[0u8; ELL]; GROUP];
        let mats = Matrices::new(&inv);
        let mut scratch = Scratch::new();
        // SAFETY: as above, and `a` and `b` are `GROUP * WINDOW_BYTES` bytes.
        unsafe {
            shift_reduce_group(
                a.as_ptr(),
                b.as_ptr(),
                &mats,
                &mut scratch,
                got.as_mut_slice().try_into().unwrap(),
                [core::ptr::null(); 3],
            )
        };
        assert_eq!(got, want);
    }
}
