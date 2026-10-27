// CREDIT: https://github.com/succinctlabs/flock (flock-prover), MIT OR Apache-2.0.
//! Bit-packing and R1CS-row helpers for the monolithic hash R1CS modules
//! (only `hash` in this vendored subset).

use primitives::stream::Stream;
use std::ops::{BitAnd, BitOr, BitOrAssign, BitXor};

/// Blocks built side by side: lane `l` of every vector belongs to block `l` of a group.
pub(crate) const LANES: usize = 8;

/// One 32-bit value per lane.
#[derive(Clone, Copy)]
#[repr(align(32))]
#[cfg_attr(
    all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vl"),
    allow(dead_code)
)]
pub(crate) struct V32(pub [u32; LANES]);

/// One 64-bit value per lane: a row of the group's word-major tables.
#[derive(Clone, Copy)]
#[repr(align(64))]
pub(crate) struct V64(pub [u64; LANES]);

/// The lane-wise operations as AVX-512 instructions, which the optimizer would otherwise
/// rebuild from scalar code into slower shuffles; elsewhere, plain loops over the lanes.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vl"))]
macro_rules! lanes {
    ($x86:expr, $any:expr) => {
        // SAFETY: the features are enabled at compile time.
        unsafe { $x86 }
    };
}
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vl")))]
macro_rules! lanes {
    ($x86:expr, $any:expr) => {
        $any
    };
}

/// The vector register that holds a lane array, and back.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vl"))]
mod reg {
    use super::{V32, V64};
    use core::arch::x86_64::{__m256i, __m512i};
    use core::mem::transmute;

    impl V32 {
        #[inline(always)]
        pub(super) fn reg(self) -> __m256i {
            // SAFETY: the same 32 bytes.
            unsafe { transmute::<Self, __m256i>(self) }
        }

        #[inline(always)]
        pub(super) fn of(x: __m256i) -> Self {
            // SAFETY: the same 32 bytes.
            unsafe { transmute::<__m256i, Self>(x) }
        }
    }

    impl V64 {
        #[inline(always)]
        pub(super) fn reg(self) -> __m512i {
            // SAFETY: the same 64 bytes.
            unsafe { transmute::<Self, __m512i>(self) }
        }

        #[inline(always)]
        pub(super) fn of(x: __m512i) -> Self {
            // SAFETY: the same 64 bytes.
            unsafe { transmute::<__m512i, Self>(x) }
        }
    }
}
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vl"))]
use core::arch::x86_64::*;

impl V32 {
    #[inline(always)]
    pub(crate) fn splat(x: u32) -> Self {
        Self([x; LANES])
    }

    #[inline(always)]
    pub(crate) fn wrapping_add(self, o: Self) -> Self {
        lanes!(
            Self::of(_mm256_add_epi32(self.reg(), o.reg())),
            Self(std::array::from_fn(|l| self.0[l].wrapping_add(o.0[l])))
        )
    }

    #[inline(always)]
    pub(crate) fn rotate_right(self, n: u32) -> Self {
        lanes!(
            Self::of(_mm256_rorv_epi32(self.reg(), _mm256_set1_epi32(n as i32))),
            Self(self.0.map(|x| x.rotate_right(n)))
        )
    }

    #[inline(always)]
    pub(crate) fn shl(self, n: u32) -> Self {
        lanes!(
            Self::of(_mm256_sll_epi32(self.reg(), _mm_cvtsi32_si128(n as i32))),
            Self(self.0.map(|x| x << n))
        )
    }

    #[inline(always)]
    pub(crate) fn shr(self, n: u32) -> Self {
        lanes!(
            Self::of(_mm256_srl_epi32(self.reg(), _mm_cvtsi32_si128(n as i32))),
            Self(self.0.map(|x| x >> n))
        )
    }
}

impl BitXor for V32 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, o: Self) -> Self {
        lanes!(
            Self::of(_mm256_xor_si256(self.reg(), o.reg())),
            Self(std::array::from_fn(|l| self.0[l] ^ o.0[l]))
        )
    }
}

impl BitAnd for V32 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, o: Self) -> Self {
        lanes!(
            Self::of(_mm256_and_si256(self.reg(), o.reg())),
            Self(std::array::from_fn(|l| self.0[l] & o.0[l]))
        )
    }
}

impl V64 {
    pub(crate) const ZERO: Self = Self([0; LANES]);

    #[inline(always)]
    pub(crate) fn splat(x: u64) -> Self {
        Self([x; LANES])
    }

    /// Each lane zero-extended.
    #[inline(always)]
    pub(crate) fn widen(v: V32) -> Self {
        lanes!(Self::of(_mm512_cvtepu32_epi64(v.reg())), Self(v.0.map(u64::from)))
    }

    /// `lo` in the low half of each lane and `hi` in the high half.
    #[inline(always)]
    pub(crate) fn pair(lo: V32, hi: V32) -> Self {
        Self::widen(lo) | Self::widen(hi).shl(32)
    }

    #[inline(always)]
    fn shl(self, n: usize) -> Self {
        lanes!(
            Self::of(_mm512_sll_epi64(self.reg(), _mm_cvtsi64_si128(n as i64))),
            Self(self.0.map(|x| x << n))
        )
    }

    #[inline(always)]
    fn shr(self, n: usize) -> Self {
        lanes!(
            Self::of(_mm512_srl_epi64(self.reg(), _mm_cvtsi64_si128(n as i64))),
            Self(self.0.map(|x| x >> n))
        )
    }
}

impl BitOr for V64 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, o: Self) -> Self {
        lanes!(
            Self::of(_mm512_or_si512(self.reg(), o.reg())),
            Self(std::array::from_fn(|l| self.0[l] | o.0[l]))
        )
    }
}

impl BitOrAssign for V64 {
    #[inline(always)]
    fn bitor_assign(&mut self, o: Self) {
        *self = *self | o;
    }
}

/// Transpose eight rows of eight lanes: word `j` of the result's block `l` is lane `l` of row `j`.
///
/// On AVX-512, three rounds of shuffles: by word, by 128-bit pair, then by 256-bit half.
#[inline(always)]
fn transpose_lanes(rows: &[V64]) -> [[u64; LANES]; LANES] {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vl"))]
    // SAFETY: the features are enabled at compile time, and each load and store is one 64-byte row.
    unsafe {
        let ld = |j: usize| _mm512_load_si512(rows[j].0.as_ptr().cast());
        let (t0, t1) = (_mm512_unpacklo_epi64(ld(0), ld(1)), _mm512_unpackhi_epi64(ld(0), ld(1)));
        let (t2, t3) = (_mm512_unpacklo_epi64(ld(2), ld(3)), _mm512_unpackhi_epi64(ld(2), ld(3)));
        let (t4, t5) = (_mm512_unpacklo_epi64(ld(4), ld(5)), _mm512_unpackhi_epi64(ld(4), ld(5)));
        let (t6, t7) = (_mm512_unpacklo_epi64(ld(6), ld(7)), _mm512_unpackhi_epi64(ld(6), ld(7)));
        let u0 = _mm512_shuffle_i64x2::<0x88>(t0, t2);
        let u1 = _mm512_shuffle_i64x2::<0x88>(t1, t3);
        let u2 = _mm512_shuffle_i64x2::<0xDD>(t0, t2);
        let u3 = _mm512_shuffle_i64x2::<0xDD>(t1, t3);
        let u4 = _mm512_shuffle_i64x2::<0x88>(t4, t6);
        let u5 = _mm512_shuffle_i64x2::<0x88>(t5, t7);
        let u6 = _mm512_shuffle_i64x2::<0xDD>(t4, t6);
        let u7 = _mm512_shuffle_i64x2::<0xDD>(t5, t7);
        let v = [
            _mm512_shuffle_i64x2::<0x88>(u0, u4),
            _mm512_shuffle_i64x2::<0x88>(u1, u5),
            _mm512_shuffle_i64x2::<0x88>(u2, u6),
            _mm512_shuffle_i64x2::<0x88>(u3, u7),
            _mm512_shuffle_i64x2::<0xDD>(u0, u4),
            _mm512_shuffle_i64x2::<0xDD>(u1, u5),
            _mm512_shuffle_i64x2::<0xDD>(u2, u6),
            _mm512_shuffle_i64x2::<0xDD>(u3, u7),
        ];
        let mut out = [[0u64; LANES]; LANES];
        for (o, v) in out.iter_mut().zip(v) {
            _mm512_storeu_si512(o.as_mut_ptr().cast(), v);
        }
        out
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vl")))]
    std::array::from_fn(|l| std::array::from_fn(|j| rows[j].0[l]))
}

/// A `64·NW`-bit record per lane, composed in registers.
pub(crate) struct BitRecord<const NW: usize> {
    w: [V64; NW],
}

impl<const NW: usize> BitRecord<NW> {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self { w: [V64::ZERO; NW] }
    }

    /// OR a (pre-masked) value into record bits `[POS, POS + width)`.
    /// `POS` is const so the straddle branch and shifts fold at compile time.
    #[inline(always)]
    pub(crate) fn push<const POS: usize>(&mut self, val: V32) {
        let v = V64::widen(val);
        let (idx, s) = (POS >> 6, POS & 63);
        self.w[idx] |= v.shl(s);
        if s > 32 {
            self.w[idx + 1] |= v.shr(64 - s);
        }
    }

    /// Append `bits` of the record to a bit stream whose first word is `out[0]`, `base` bits in.
    ///
    /// `carry` is the stream's partial word at `base`: it holds the earlier records' bits and is
    /// replaced by the partial word at `base + bits`. Every complete word is written, none is read.
    #[inline(always)]
    pub(crate) fn append(&self, out: &mut [V64], base: usize, bits: usize, carry: &mut V64) {
        let (first, s) = (base >> 6, base & 63);
        let complete = ((base + bits) >> 6) - first;
        let mut word = *carry;
        for j in 0..=complete {
            if j < NW {
                word |= self.w[j].shl(s);
            }
            if j > 0 && s > 0 {
                word |= self.w[j - 1].shr(64 - s);
            }
            if j < complete {
                out[first + j] = word;
            } else {
                *carry = word;
            }
            word = V64::ZERO;
        }
    }
}

/// One 32-bit ADD's witness parts: `(sum, left, right, carry_aux)` with
/// `left/right/carry_aux` masked to the low 31 bits (bit 31 is the discarded
/// mod-2³² carry-out; the carry slot is 31 bits wide).
#[inline(always)]
pub(crate) fn add_carry_parts(x: V32, y: V32) -> (V32, V32, V32, V32) {
    let sum = x.wrapping_add(y);
    let cin = sum ^ x ^ y;
    let mask = V32::splat(0x7FFF_FFFF);
    let left = (x ^ cin) & mask;
    let right = (y ^ cin) & mask;
    let carry_aux = left & right;
    (sum, left, right, carry_aux)
}

/// One fused three-operand ADD's witness parts (see
/// `gf2::walk_add3_fused` for the row algebra): the sum, then each
/// layer's `(left, right, product)` triple.
///
/// The majority triple is masked to bits 0..=30. The ripple triple is masked
/// to bits 1..=30 **and shifted down by one**, so its slot `j` holds bit
/// `j + 1`, matching the 30-slot ripple run.
#[inline(always)]
pub(crate) fn add3_fused_parts(x: V32, y: V32, z: V32) -> (V32, (V32, V32, V32), (V32, V32, V32)) {
    let mask_lo31 = V32::splat(0x7FFF_FFFF);
    let mask_lo30 = V32::splat(0x3FFF_FFFF);
    let maj_left = (x ^ z) & mask_lo31;
    let maj_right = (y ^ z) & mask_lo31;
    let maj_aux = maj_left & maj_right;
    // p + 2·maj, where maj[i] = maj_aux[i] ⊕ z[i] is the bitwise majority.
    let p = x ^ y ^ z;
    let q = (maj_aux ^ (z & mask_lo31)).shl(1);
    let sum = p.wrapping_add(q);
    let cin = sum ^ p ^ q;
    let rip_left = ((p ^ cin).shr(1)) & mask_lo30;
    let rip_right = ((q ^ cin).shr(1)) & mask_lo30;
    let rip_aux = rip_left & rip_right;
    (sum, (maj_left, maj_right, maj_aux), (rip_left, rip_right, rip_aux))
}

/// The word-major rows of one group, `z | a | b`, which a group builder writes.
pub struct GroupRows(pub(crate) Vec<V64>);

impl GroupRows {
    /// Rows for `words` words of each of the three tables.
    pub(crate) fn new(words: usize) -> Self {
        Self(vec![V64::ZERO; 3 * words])
    }
}

/// Row `i` of `rows` is word `i` of each lane's block: transpose into the block-major `table`.
///
/// Block `l` takes `words` words from `l * words`, for the first `group_blocks` lanes.
pub(crate) fn blocks_from_rows(rows: &[V64], words: usize, group_blocks: usize, table: &mut [u64]) {
    // Each block takes one tile, `LANES` words, from every transposed tile of rows.
    let mut tiles = table[..group_blocks * words]
        .chunks_exact_mut(words)
        .map(|block| block.as_chunks_mut::<LANES>().0.iter_mut());
    let mut cols: [Option<_>; LANES] = std::array::from_fn(|_| tiles.next());
    for tile in rows.chunks_exact(LANES) {
        for (col, out) in cols.iter_mut().zip(transpose_lanes(tile)) {
            if let Some(col) = col {
                *col.next().expect("a tile per tile of rows") = out;
            }
        }
    }
}

/// The `u64` words of the rows, which keep their 64-byte alignment.
pub(crate) fn words_of_mut(rows: &mut [V64]) -> &mut [u64] {
    // SAFETY: a `V64` is `LANES` `u64`s with no padding, and `u64`'s alignment divides `V64`'s.
    unsafe { core::slice::from_raw_parts_mut(rows.as_mut_ptr().cast(), rows.len() * LANES) }
}

/// The bytes of the `u64` words on a little-endian target.
pub(crate) fn packed_bytes(words: &[u64]) -> &[u8] {
    const _: () = assert!(
        cfg!(target_endian = "little"),
        "packed witness bytes assume little-endian"
    );
    // SAFETY: `u64` has no padding or invalid bit patterns, and `u8`'s
    // alignment divides `u64`'s, so the words are a valid `8 · len` byte slice.
    unsafe { core::slice::from_raw_parts(words.as_ptr().cast::<u8>(), words.len() * 8) }
}

// ---------------------------------------------------------------------------
// Generic witness packing driver.
// ---------------------------------------------------------------------------

/// Groups a task builds one after the other, each streaming while the next builds.
const TASK_GROUPS: usize = 16;

/// Drive the parallel chunked witness build for `n_blocks` instances padded
/// to `2^n_blocks_log` slots. Returns `(z, a, b)`: the three bit-packed `u64` tables
/// (`WORDS` per instance). `a` and `b` may be shorter than `z`, see [`drive_witness_packed_into`].
///
/// `build_group(states, rows, progress)` builds eight blocks at once, one per lane, from `states`.
/// `rows` is `3 · WORDS` rows, the `z`, `a` and `b` tables word-major: row `i` holds word `i` of
/// each lane's block, and every row is written. It calls `progress(done, steps)` as its steps finish.
/// `initial_states.len()` may be less than `2^n_blocks_log`: `padding` builds a real block in
/// every trailing slot, so that the constant column is all-ones across *every* batched instance
/// (see `lincheck's `LincheckCircuit::const_pin_col``).
pub(crate) fn drive_witness_packed<const WORDS: usize, S: Sync, F>(
    initial_states: &[S],
    padding: &S,
    n_blocks_log: usize,
    build_group: F,
) -> (Vec<u64>, Vec<u64>, Vec<u64>)
where
    F: Fn(&[&S; LANES], &mut [V64], &mut dyn FnMut(usize, usize)) + Sync,
{
    let total_words = (1usize << n_blocks_log) * WORDS;
    // SAFETY: `drive_witness_packed_into` writes every word of `z`.
    let mut z = unsafe { primitives::uninit_vec::<u64>(total_words) };
    let (a, b) =
        drive_witness_packed_into::<WORDS, S, F>(initial_states, padding, n_blocks_log, &mut z, true, build_group);
    (z, a, b)
}

/// [`drive_witness_packed`] with `z` written into the caller's buffer, which may
/// be uninitialized: every word is written.
///
/// The returned `a` and `b` stop after the group holding block `n_blocks`, the first all-padding block.
/// The zerocheck reads that one padding block and none after it.
///
/// With `write_ab` false, `a` and `b` keep their size but stay untouched: they are neither transposed nor streamed.
pub(crate) fn drive_witness_packed_into<const WORDS: usize, S: Sync, F>(
    initial_states: &[S],
    padding: &S,
    n_blocks_log: usize,
    z: &mut [u64],
    write_ab: bool,
    build_group: F,
) -> (Vec<u64>, Vec<u64>)
where
    F: Fn(&[&S; LANES], &mut [V64], &mut dyn FnMut(usize, usize)) + Sync,
{
    let n_total = 1usize << n_blocks_log;
    let n_blocks = initial_states.len();
    assert!(
        n_blocks <= n_total,
        "{n_blocks} blocks > 2^{n_blocks_log} = {n_total} slots"
    );
    // A group is eight blocks; below eight, one group holds them all.
    let group_blocks = n_total.min(LANES);
    let n_groups = n_total / group_blocks;

    let total_words = n_total * WORDS;
    assert_eq!(z.len(), total_words, "z holds every instance's packed words");
    // SAFETY (x3): group `g` publishes chunk `g` of every table in full below, `a` and `b` only for `g < ab_groups`.
    // The chunks are a group's blocks of packed words.
    let ab_groups = (n_blocks / group_blocks + 1).min(n_groups);
    let mut a = unsafe { primitives::uninit_vec::<u64>(ab_groups * group_blocks * WORDS) };
    let mut b = unsafe { primitives::uninit_vec::<u64>(ab_groups * group_blocks * WORDS) };

    // Three output tables, indexed by the same group, each taking the group's blocks' packed words.
    let group_words = group_blocks * WORDS;
    let scratch_words = LANES * WORDS;
    let z_chunks = parallel::Chunks::new(z, group_words);
    let a_chunks = parallel::Chunks::new(&mut a, group_words);
    let b_chunks = parallel::Chunks::new(&mut b, group_words);
    debug_assert_eq!(a_chunks.count(), ab_groups);

    // Each group builds in its worker's scratch, which stays in cache, then streams out.
    //
    //     rows     z | a | b   word-major: row i is word i of the 8 blocks, one per lane
    //     buf      z | a | b   block-major: the rows transposed, the layout of the output
    //
    // Building in place instead would fetch every output line before writing it.
    //
    // Streaming stores go to memory at its pace, and a burst of them stalls the core.
    // So a group streams a slice at a time while the next builds, in the other of the two `bufs`.
    // A task's last group streams when the task ends.
    //
    // Groups past the last real block are identical, so a worker builds one and streams it again
    // while `template` says the buf it last built still holds it.
    parallel::map_reduce_with_state(
        n_groups.div_ceil(TASK_GROUPS),
        || {
            let buf = || vec![0u64; 3 * scratch_words];
            (vec![V64::ZERO; 3 * WORDS], [buf(), buf()], 0usize, false)
        },
        || (),
        |(rows, bufs, last, template), (), task| {
            let stream = Stream::new();
            // Stream the part `from..to` of `of` of group `g`'s tables out of `buf`, whole lines at a time.
            // SAFETY: each group `g` takes chunk `g` of each table, one slice at a time, and
            // all three tables stay borrowed for the whole dispatch.
            let emit = |g: usize, buf: &[u64], from: usize, to: usize, of: usize| unsafe {
                let words = (group_words / 8 * from / of * 8)..(group_words / 8 * to / of * 8);
                let (z, ab) = buf.split_at(scratch_words);
                let (a, b) = ab.split_at(scratch_words);
                stream.copy(&mut z_chunks.get(g)[words.clone()], &z[words.clone()]);
                if write_ab && g < ab_groups {
                    stream.copy(&mut a_chunks.get(g)[words.clone()], &a[words.clone()]);
                    stream.copy(&mut b_chunks.get(g)[words.clone()], &b[words]);
                }
            };
            let mut pending = None;
            for g in task * TASK_GROUPS..((task + 1) * TASK_GROUPS).min(n_groups) {
                let is_padding = LANES * g >= n_blocks;
                if is_padding && *template {
                    if let Some(p) = pending.take() {
                        emit(p, &bufs[*last], 0, 1, 1);
                    }
                    emit(g, &bufs[*last], 0, 1, 1);
                    continue;
                }
                let [b0, b1] = &mut *bufs;
                let (prev_buf, cur_buf) = if *last == 0 { (&*b0, b1) } else { (&*b1, b0) };
                let prev = pending.take();
                let (mut sent, mut of) = (0, 1);
                let states = std::array::from_fn(|l| initial_states.get(LANES * g + l).unwrap_or(padding));
                build_group(&states, rows, &mut |done, steps| {
                    if let Some(p) = prev {
                        emit(p, prev_buf, sent, done, steps);
                    }
                    (sent, of) = (done, steps);
                });
                if let Some(p) = prev {
                    emit(p, prev_buf, sent, of, of);
                }
                let n_tables = if write_ab { 3 } else { 1 };
                for (t_rows, table) in rows.chunks(WORDS).zip(cur_buf.chunks_mut(scratch_words)).take(n_tables) {
                    blocks_from_rows(t_rows, WORDS, group_blocks, table);
                }
                (*last, *template) = (1 - *last, is_padding);
                pending = Some(g);
            }
            if let Some(p) = pending {
                emit(p, &bufs[*last], 0, 1, 1);
            }
        },
        |(), ()| (),
    );

    (a, b)
}
