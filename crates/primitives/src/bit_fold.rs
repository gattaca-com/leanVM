//! Weighted sums of packed bit rows, the fold that turns witness bits into F192 values.
//!
//! A row is `8 * CHUNKS` bits and every bit carries a fixed F192 weight:
//!
//! ```text
//!     fold(row) = sum_{s : bit s of row is set} w_s
//! ```
//!
//! The map is GF(2)-linear in the row's bits, so it splits into one 8-bit piece per byte.
//!
//! - Portable: a 256-entry subset-sum table per byte, one lookup per byte.
//! - GFNI: an 8x8 bit matrix per (input byte, output byte), applied to 64 rows by one instruction.

use crate::field::{F192, F192x8};

use crate::multilinear::eq_table as build_eq;

/// Rows folded per call.
pub const BLOCK: usize = 64;

/// Whether the fold runs on GFNI rather than the byte tables.
pub const GFNI: bool = cfg!(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
));

/// The weights of every bit of a row, prepared for folding.
#[derive(Clone, Debug)]
pub struct BitFold {
    /// Bytes per row.
    n_chunks: usize,
    /// The weight of every bit of a row.
    weights: Vec<F192>,
    /// The target's fold kernel data.
    imp: Imp,
}

impl BitFold {
    /// Prepare `weights`, one per bit of a row.
    ///
    /// # Panics
    ///
    /// Panics unless a row is 8, 16, 32, 64 or 128 bytes.
    pub fn new(weights: &[F192]) -> Self {
        let n_chunks = weights.len() / 8;
        assert!(
            weights.len() == 8 * n_chunks && n_chunks.is_power_of_two() && (8..=128).contains(&n_chunks),
            "a row is 8 to 128 bytes, a power of two"
        );
        Self {
            n_chunks,
            weights: weights.to_vec(),
            imp: Imp::new(weights),
        }
    }

    /// The fold of a position at multilinear level `t`, once `rho_1..rho_t` are bound.
    ///
    /// Position `q` covers the `2^t` consecutive rows `q * 2^t + u`, so its weights are a tensor:
    ///
    /// ```text
    ///     w[64 u + s] = eq(rho, u) * L_s(z)        u in 0..2^t, s in 0..64
    /// ```
    pub fn at_level(lagrange: &[F192], rho: &[F192]) -> Self {
        let weights: Vec<F192> = build_eq(rho)
            .iter()
            .flat_map(|&e| lagrange.iter().map(move |&l| e * l))
            .collect();
        Self::new(&weights)
    }

    /// Bytes per row.
    pub fn n_chunks(&self) -> usize {
        self.n_chunks
    }

    /// The weight of every bit of a row.
    pub fn weights(&self) -> &[F192] {
        &self.weights
    }

    /// Fold up to 64 consecutive rows into `out[..rows.len()]`.
    #[inline]
    pub fn fold_block<const CHUNKS: usize>(&self, rows: &[[u8; CHUNKS]], out: &mut [F192; BLOCK]) {
        debug_assert_eq!(CHUNKS, self.n_chunks);
        assert!(rows.len() <= BLOCK);
        self.imp.fold_block(rows, out);
    }

    /// Fold up to 64 consecutive rows into eight vectors, eight values each, in the order `GROUP_LOG` picks.
    ///
    /// Value `p` is the fold of row `p`, and a short block is zero-padded. Lane `l` of vector `r` holds the value
    ///
    /// ```text
    ///     p = (r mod G) + G * l + 8 G * (r div G)        G = 2^GROUP_LOG
    /// ```
    ///
    /// So values group in runs of `G`, with run member `r mod G` in register `r mod G`, and eight consecutive runs
    /// fill the lanes of one vector. The operands of a lane-wise product over such a group are then whole vectors.
    #[inline]
    pub fn fold_block_x8<const CHUNKS: usize, const GROUP_LOG: usize>(&self, rows: &[[u8; CHUNKS]]) -> [F192x8; 8] {
        const { assert!(GROUP_LOG <= 2) };
        debug_assert_eq!(CHUNKS, self.n_chunks);
        assert!(rows.len() <= BLOCK);
        self.imp.fold_block_x8::<CHUNKS, GROUP_LOG>(rows)
    }

    /// Fold 64 rows given as byte columns: `cols[j][p]` is byte `j` of row `p`.
    ///
    /// This is the form [`crate::bits::bit_transpose_64bytes`] leaves a block of words in, and the GFNI fold's own
    /// intermediate, so it skips the fold's input transpose.
    #[inline]
    pub fn fold_columns<const CHUNKS: usize>(&self, cols: &[[u8; BLOCK]; CHUNKS], out: &mut [F192; BLOCK]) {
        debug_assert_eq!(CHUNKS, self.n_chunks);
        self.imp.fold_columns(cols, out);
    }
}

#[cfg(any(
    test,
    not(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi"
    ))
))]
/// One 256-entry subset-sum table per byte of a row: entry `[j][v]` sums the weights of the set bits of `v` at byte `j`.
fn lookup_tables(weights: &[F192]) -> Vec<[F192; 256]> {
    weights
        .as_chunks::<8>()
        .0
        .iter()
        .map(|w| {
            let mut sums = [F192::ZERO; 256];
            // Each entry adds its lowest set bit's weight to an entry already built.
            for v in 1..256usize {
                let low = v & v.wrapping_neg();
                sums[v] = sums[v ^ low] + w[low.trailing_zeros() as usize];
            }
            sums
        })
        .collect()
}

#[cfg(any(
    test,
    not(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi"
    ))
))]
/// The fold of one row through the byte tables: one lookup and one XOR per byte.
#[inline(always)]
fn fold_row_lookup<const CHUNKS: usize>(tables: &[[F192; 256]], row: &[u8; CHUNKS]) -> F192 {
    let tables: &[[F192; 256]; CHUNKS] = tables.try_into().expect("one table per byte");
    row.iter()
        .zip(tables)
        .fold(F192::ZERO, |acc, (&v, sums)| acc + sums[usize::from(v)])
}

/// The value in lane `l` of vector `r` of [`BitFold::fold_block_x8`], for groups of `2^g`.
const fn lane_value(g: usize, r: usize, l: usize) -> usize {
    (r & ((1 << g) - 1)) | (l << g) | ((r >> g) << (g + 3))
}

#[cfg(any(
    test,
    not(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq"
    ))
))]
/// The vectors of [`BitFold::fold_block_x8`], from the 64 values in natural order.
fn x8_from_values<const G: usize>(values: &[F192; BLOCK]) -> [F192x8; 8] {
    std::array::from_fn(|r| F192x8::from_array(&std::array::from_fn(|l| values[lane_value(G, r, l)])))
}

#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
)))]
use portable::Imp;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
))]
use gfni::Imp;

#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
)))]
mod portable {
    use super::{BLOCK, F192, F192x8, fold_row_lookup, lookup_tables, x8_from_values};

    /// The byte tables.
    #[derive(Clone, Debug)]
    pub(super) struct Imp {
        /// One subset-sum table per byte of a row.
        tables: Vec<[F192; 256]>,
    }

    impl Imp {
        pub(super) fn new(weights: &[F192]) -> Self {
            Self {
                tables: lookup_tables(weights),
            }
        }

        #[inline]
        pub(super) fn fold_block<const CHUNKS: usize>(&self, rows: &[[u8; CHUNKS]], out: &mut [F192; BLOCK]) {
            for (o, row) in out.iter_mut().zip(rows) {
                *o = fold_row_lookup(&self.tables, row);
            }
        }

        #[inline]
        pub(super) fn fold_block_x8<const CHUNKS: usize, const G: usize>(&self, rows: &[[u8; CHUNKS]]) -> [F192x8; 8] {
            let mut values = [F192::ZERO; BLOCK];
            self.fold_block(rows, &mut values);
            x8_from_values::<G>(&values)
        }

        #[inline]
        pub(super) fn fold_columns<const CHUNKS: usize>(&self, cols: &[[u8; BLOCK]; CHUNKS], out: &mut [F192; BLOCK]) {
            for (p, o) in out.iter_mut().enumerate() {
                *o = cols
                    .iter()
                    .zip(&self.tables)
                    .fold(F192::ZERO, |acc, (col, sums)| acc + sums[usize::from(col[p])]);
            }
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
))]
pub mod gfni {
    //! The GFNI fold of 64 rows at a time.
    //!
    //! ```text
    //!     1. transpose   64 rows x CHUNKS bytes  ->  CHUNKS registers, register j = byte j of the 64 rows
    //!     2. multiply    out byte o += A[j][o] * (register j), one affine instruction for all 64 rows
    //!     3. transpose   24 registers of output bytes  ->  64 F192 values
    //! ```
    //!
    //! Both transposes are butterflies of two-register byte permutes.
    //!
    //! A stage swaps one bit of the byte offset inside a register with one bit of the register index.

    use core::arch::x86_64::*;
    use std::mem::MaybeUninit;
    use std::sync::LazyLock;

    #[cfg(not(target_feature = "vpclmulqdq"))]
    use super::x8_from_values;
    use super::{BLOCK, F192, F192x8, lane_value};

    /// Bytes of an F192.
    pub const OUT_BYTES: usize = 24;

    /// One butterfly stage: registers `z` and `z | zbit` exchange bytes through two permutes.
    ///
    /// The register bit `zbit` is a call argument, so a caller's constant loop unrolls into registers.
    #[derive(Clone, Copy, Debug)]
    struct Stage {
        /// Byte sources of the low register of each pair; bit 6 picks the high one.
        lo: [u8; 64],
        /// Byte sources of the high register of each pair.
        hi: [u8; 64],
    }

    impl Stage {
        /// Swap byte-offset bit `fbit` with the register bit, then permute each output by `sigma`.
        ///
        /// Output offset `f` takes the byte the plain swap leaves at offset `sigma(f)`.
        fn swap(fbit: usize, sigma: impl Fn(usize) -> usize) -> Self {
            // The plain swap: a byte keeps its offset except bit `fbit`, which trades places with the register bit.
            let lo = |f: usize| if f & fbit == 0 { f } else { 64 | (f ^ fbit) };
            let hi = |f: usize| if f & fbit == 0 { f | fbit } else { 64 | f };
            Self {
                lo: std::array::from_fn(|f| lo(sigma(f)) as u8),
                hi: std::array::from_fn(|f| hi(sigma(f)) as u8),
            }
        }

        #[inline(always)]
        fn apply<const N: usize>(&self, zbit: usize, regs: &mut [__m512i; N]) {
            // SAFETY: both index arrays are 64 bytes.
            let (lo, hi) = unsafe {
                (
                    _mm512_loadu_si512(self.lo.as_ptr().cast()),
                    _mm512_loadu_si512(self.hi.as_ptr().cast()),
                )
            };
            for z in (0..N).filter(|z| z & zbit == 0) {
                let (u, w) = (regs[z], regs[z | zbit]);
                // SAFETY: the module is compiled only with these target features enabled.
                unsafe {
                    regs[z] = _mm512_permutex2var_epi8(u, lo, w);
                    regs[z | zbit] = _mm512_permutex2var_epi8(u, hi, w);
                }
            }
        }
    }

    /// Stages taking eight registers of one output coefficient, register = byte `o` and offset = value `p`, to qwords.
    ///
    /// Stage `s` swaps register bit `s` with the value bit that [`lane_value`] gives register bit `s`.
    /// The last stage then orders each register's offsets as qword = lane, byte = `o`.
    ///
    /// So register `r`, qword `l` is the coefficient of value `lane_value(g, r, l)`.
    static OUTPUT_STAGES: LazyLock<[[Stage; 3]; 3]> = LazyLock::new(|| {
        [0, 1, 2].map(|g| {
            [0, 1, 2].map(|s| {
                Stage::swap(lane_value(g, 1 << s, 0), |f| {
                    if s == 2 { lane_value(g, f & 7, f >> 3) } else { f }
                })
            })
        })
    });

    /// The coefficient registers of 24 registers of output bytes, register `o` holding byte `o` of 64 values.
    ///
    /// Per coefficient, register `r` and qword `l` hold the coefficient of value `lane_value(G, r, l)`.
    #[inline(always)]
    fn coefficients<const G: usize>(acc: &[__m512i; OUT_BYTES]) -> [[__m512i; 8]; 3] {
        let mut c: [[__m512i; 8]; 3] = std::array::from_fn(|i| std::array::from_fn(|o| acc[8 * i + o]));
        for (s, stage) in OUTPUT_STAGES[G].iter().enumerate() {
            c.iter_mut().for_each(|c| stage.apply(1 << s, c));
        }
        c
    }

    /// Store 24 registers of output bytes, register `o` holding byte `o` of 64 values, as those 64 values.
    ///
    /// # Safety
    ///
    /// The CPU must have the target features this function enables.
    #[inline]
    #[target_feature(enable = "avx512f", enable = "avx512vbmi")]
    pub fn store_f192(acc: &[__m512i; OUT_BYTES], out: &mut [F192; BLOCK]) {
        let [c0, c1, c2] = coefficients::<0>(acc);

        // Interleave the three coefficients of values 8k..8k+8 into 24 consecutive qwords.
        //
        //     zmm 0   c0 c1 c2 | c0 c1 c2 | c0 c1        values 0, 1, 2
        //     zmm 1   c2 | c0 c1 c2 | c0 c1 c2 | c0      values 2, 3, 4, 5
        //     zmm 2   c1 c2 | c0 c1 c2 | c0 c1 c2        values 5, 6, 7
        //
        // Indices 0..8 pick c0, 8..16 pick c1; the masked permute then drops c2 into its slots.
        let idx01 = [
            _mm512_setr_epi64(0, 8, 0, 1, 9, 0, 2, 10),
            _mm512_setr_epi64(0, 3, 11, 0, 4, 12, 0, 5),
            _mm512_setr_epi64(13, 0, 6, 14, 0, 7, 15, 0),
        ];
        let idx2 = [
            _mm512_setr_epi64(0, 0, 0, 0, 0, 1, 0, 0),
            _mm512_setr_epi64(2, 0, 0, 3, 0, 0, 4, 0),
            _mm512_setr_epi64(0, 5, 0, 0, 6, 0, 0, 7),
        ];
        let mask2: [__mmask8; 3] = [0b0010_0100, 0b0100_1001, 0b1001_0010];
        let dst = out.as_mut_ptr().cast::<u8>();
        for k in 0..8 {
            for i in 0..3 {
                let v = _mm512_permutex2var_epi64(c0[k], idx01[i], c1[k]);
                let v = _mm512_mask_permutexvar_epi64(v, mask2[i], idx2[i], c2[k]);
                // SAFETY: the three stores of value group k cover out[8k..8k+8], 192 bytes.
                unsafe { _mm512_storeu_si512(dst.add(192 * k + 64 * i).cast(), v) };
            }
        }
    }

    /// Byte sources gathering byte `o` of eight weights into qword `o`, in reverse weight order.
    ///
    /// Output register `g`, qword `l`, byte `7 - s` takes byte `8g + l` of weight `s`, input byte `24 s + 8g + l`.
    ///
    /// Returns the two-register sources, the third-register sources, and the mask of bytes from the third.
    const fn gather_index(g: usize) -> ([u8; 64], [u8; 64], u64) {
        let (mut lo, mut hi, mut mask) = ([0u8; 64], [0u8; 64], 0u64);
        let mut p = 0;
        while p < 64 {
            let (l, s) = (p / 8, 7 - p % 8);
            let src = 24 * s + 8 * g + l;
            if src < 128 {
                lo[p] = src as u8;
            } else {
                hi[p] = (src - 128) as u8;
                mask |= 1 << p;
            }
            p += 1;
        }
        (lo, hi, mask)
    }

    /// The gathers of the three output registers.
    const GATHER: [([u8; 64], [u8; 64], u64); 3] = [gather_index(0), gather_index(1), gather_index(2)];

    /// The GFNI matrices of eight weights, one per output byte.
    ///
    /// Matrix `o` maps an input byte `x` to byte `o` of `sum_{s : bit s of x} w_s`.
    ///
    /// The affine instruction computes result bit `k` as the parity of `x` against matrix byte `7 - k`.
    /// So matrix byte `7 - k` has bit `s` set when bit `k` of byte `o` of `w_s` is set.
    ///
    /// ```text
    ///     1. gather     qword o  =  byte o of w_7, ..., w_0        (a byte transpose)
    ///     2. transpose  each qword as an 8x8 bit matrix           (an affine against the reversed identity)
    /// ```
    ///
    /// # Safety
    ///
    /// The CPU must have the target features this function enables.
    #[inline]
    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vbmi", enable = "gfni")]
    pub fn weight_matrices(w: &[F192; 8]) -> [u64; OUT_BYTES] {
        let src = w.as_ptr().cast::<u8>();
        // SAFETY: eight weights are 192 bytes, exactly three registers.
        let v: [__m512i; 3] = std::array::from_fn(|i| unsafe { _mm512_loadu_si512(src.add(64 * i).cast()) });
        // Byte t of each qword is the unit vector 1 << (7 - t).
        let reversed_identity = _mm512_set1_epi64(0x0102_0408_1020_4080);
        let mut out = [0u64; OUT_BYTES];
        let dst = out.as_mut_ptr().cast::<u8>();
        for (g, (lo, hi, mask)) in GATHER.iter().enumerate() {
            // SAFETY: the index arrays are 64 bytes, and store g fills out[8g..8g+8].
            unsafe {
                let q = _mm512_permutex2var_epi8(v[0], _mm512_loadu_si512(lo.as_ptr().cast()), v[1]);
                let q = _mm512_mask_permutexvar_epi8(q, *mask, _mm512_loadu_si512(hi.as_ptr().cast()), v[2]);
                let a = _mm512_gf2p8affine_epi64_epi8::<0>(reversed_identity, q);
                _mm512_storeu_si512(dst.add(64 * g).cast(), a);
            }
        }
        out
    }

    /// The stages `first..first + log2(TILE)` of the input transpose of `CHUNKS`-byte rows, one tile of `TILE` registers
    /// at a time: register `r` is read through `load` and written to `dst`.
    ///
    /// A stage swaps offset bit `s` with register bit `s + skew`, so a group of stages exchanges within the tiles
    /// of registers that differ only in that group's register bits.
    #[inline(always)]
    fn stages<const CHUNKS: usize, const TILE: usize>(
        input: &[Stage],
        first: usize,
        load: impl Fn(usize) -> __m512i,
        dst: *mut __m512i,
    ) {
        let skew = CHUNKS.trailing_zeros().saturating_sub(6) as usize;
        let (low, bits) = (first + skew, TILE.trailing_zeros() as usize);
        for t in 0..CHUNKS / TILE {
            // The tile's register `k` is register `base | k << low`.
            let base = ((t >> low) << (low + bits)) | (t & ((1 << low) - 1));
            let mut tile: [__m512i; TILE] = std::array::from_fn(|k| load(base | k << low));
            for i in 0..bits {
                input[first + i].apply(1 << i, &mut tile);
            }
            for (k, r) in tile.into_iter().enumerate() {
                // SAFETY: every register index is below CHUNKS, and `dst` holds CHUNKS registers.
                unsafe { _mm512_store_si512(dst.add(base | k << low).cast(), r) };
            }
        }
    }

    /// Phase 2, from one register per input byte: every output byte accumulates one affine product per input register.
    ///
    /// With `PREFETCH`, `next` is the block that follows, and its cache lines load two per input register pair.
    /// The pass is affine-bound, so the next block's fetch hides behind it, where a burst before the pass stalled.
    #[inline(always)]
    fn accumulate_into<const N: usize, const PREFETCH: bool>(
        acc: &mut [__m512i; OUT_BYTES],
        regs: &[__m512i; N],
        matrices: &[[u64; OUT_BYTES]; N],
        next: *const u8,
    ) {
        // SAFETY: the module is compiled only with these target features enabled.
        unsafe {
            for (i, (pair, m)) in regs.as_chunks::<2>().0.iter().zip(matrices.as_chunks::<2>().0).enumerate() {
                if PREFETCH {
                    // A prefetch past the end of the data is harmless.
                    _mm_prefetch::<_MM_HINT_T0>(next.wrapping_add(128 * i).cast());
                    _mm_prefetch::<_MM_HINT_T0>(next.wrapping_add(128 * i + 64).cast());
                }
                for o in 0..OUT_BYTES {
                    let g0 = _mm512_gf2p8affine_epi64_epi8::<0>(pair[0], _mm512_set1_epi64(m[0][o] as i64));
                    let g1 = _mm512_gf2p8affine_epi64_epi8::<0>(pair[1], _mm512_set1_epi64(m[1][o] as i64));
                    acc[o] = _mm512_ternarylogic_epi64::<0x96>(acc[o], g0, g1);
                }
            }
        }
    }

    /// Stages taking row-major rows of `2^c` bytes, 64 of them, to one register per input byte.
    fn input_stages(c: usize) -> Vec<Stage> {
        // Input address of a byte: bits 0..c are its byte j in the row, bits c..c+6 its row p.
        //
        //     in a register        offset = address bits 0..6     register = address bits 6..c+6
        //     after the stages     offset = p                     register = j, rotated when c > 6
        //
        // Stage s swaps offset bit s with register bit s + (c - 6 when c > 6).
        let n_stages = c.min(6);
        // Without a rotation, a short row leaves offset = (p high bits, then p low bits).
        let plain = |p: usize| {
            if c >= 6 {
                p
            } else {
                (p >> (6 - c)) | ((p & ((1 << (6 - c)) - 1)) << c)
            }
        };
        (0..n_stages)
            .map(|s| {
                let last = s + 1 == n_stages;
                Stage::swap(1 << s, |f| if last { plain(f) } else { f })
            })
            .collect()
    }

    /// The eight vectors of 24 registers of output bytes, in the order of [`BitFold::fold_block_x8`] for group size `2^G`.
    #[cfg(target_feature = "vpclmulqdq")]
    #[inline(always)]
    fn x8_from_acc<const G: usize>(acc: &[__m512i; OUT_BYTES]) -> [F192x8; 8] {
        let [c0, c1, c2] = coefficients::<G>(acc);
        std::array::from_fn(|r| F192x8::from_coeffs(c0[r], c1[r], c2[r]))
    }

    /// The matrices of every byte of a row's weights, as [`weight_matrices`] gives them.
    pub fn byte_matrices(weights: &[F192]) -> Vec<[u64; OUT_BYTES]> {
        // SAFETY: the module is compiled only with these target features enabled.
        weights
            .as_chunks()
            .0
            .iter()
            .map(|w| unsafe { weight_matrices(w) })
            .collect()
    }

    /// 64 lines of 64 bytes, transposed: register `j` holds byte `j` of every line, one line per byte of the register.
    ///
    /// The lines of a block of instances sit one block apart, so folding with lines as lanes puts an instance in
    /// each lane.
    pub struct LineRegs([__m512i; 64]);

    impl LineRegs {
        /// Registers to be [`load`](Self::load)ed.
        pub fn new() -> Self {
            // SAFETY: the module is compiled only with these target features enabled.
            Self(unsafe { [_mm512_setzero_si512(); 64] })
        }

        #[cfg(test)]
        pub(super) fn registers(&self) -> &[__m512i; 64] {
            &self.0
        }

        /// Replace the registers with the transpose of `lines`.
        ///
        /// Four stages of unpacks inside the 128-bit lanes, then two of lane shuffles. On a Zen 5 the unpacks issue
        /// on more ports than a byte permute does. Both groups work in place: the first writes each register where
        /// the second leaves it, so the transpose needs no scratch to stay in the L1.
        #[inline(always)]
        pub fn load(&mut self, lines: [&[u8; 64]; 64]) {
            let dst = self.0.as_mut_ptr();
            // SAFETY: a line is 64 bytes, and `dst` holds 64 registers.
            unsafe {
                // Register bit `s` against the offset bits inside a 128-bit lane, 16 registers at a time.
                for t in 0..4 {
                    let mut x: [__m512i; 16] =
                        std::array::from_fn(|k| _mm512_loadu_si512(lines[16 * t + k].as_ptr().cast()));
                    unpack_stage::<1, 16>(&mut x, |u, w| (_mm512_unpacklo_epi8(u, w), _mm512_unpackhi_epi8(u, w)));
                    unpack_stage::<2, 16>(&mut x, |u, w| {
                        (_mm512_unpacklo_epi16(u, w), _mm512_unpackhi_epi16(u, w))
                    });
                    unpack_stage::<4, 16>(&mut x, |u, w| {
                        (_mm512_unpacklo_epi32(u, w), _mm512_unpackhi_epi32(u, w))
                    });
                    unpack_stage::<8, 16>(&mut x, |u, w| {
                        (_mm512_unpacklo_epi64(u, w), _mm512_unpackhi_epi64(u, w))
                    });
                    for (k, r) in x.into_iter().enumerate() {
                        _mm512_store_si512(dst.add(BYTE_OF_REG[16 * t + k] as usize).cast(), r);
                    }
                }
                // Register bits 4 and 5 against the lane index, four registers at a time.
                for k in 0..16 {
                    let at = [0, 16, 32, 48].map(|o| dst.add(BYTE_OF_REG[o + k] as usize));
                    let [a, b, c, d] = at.map(|p| _mm512_load_si512(p.cast()));
                    let (a, b) = (_mm512_shuffle_i64x2::<0x44>(a, b), _mm512_shuffle_i64x2::<0xEE>(a, b));
                    let (c, d) = (_mm512_shuffle_i64x2::<0x44>(c, d), _mm512_shuffle_i64x2::<0xEE>(c, d));
                    let (a, c) = (_mm512_shuffle_i64x2::<0x88>(a, c), _mm512_shuffle_i64x2::<0xDD>(a, c));
                    let (b, d) = (_mm512_shuffle_i64x2::<0x88>(b, d), _mm512_shuffle_i64x2::<0xDD>(b, d));
                    for (p, r) in at.into_iter().zip([a, b, c, d]) {
                        _mm512_store_si512(p.cast(), r);
                    }
                }
            }
        }
    }

    /// The byte of the lines that register `z` holds when the stages are done: the four low bits of `z` reversed, and
    /// the two high bits swapped.
    const BYTE_OF_REG: [u8; 64] = {
        let mut f = [0u8; 64];
        let mut z = 0;
        while z < 64 {
            f[z] = ((z as u8 & 15).reverse_bits() >> 4) | (z as u8 >> 4 & 1) << 5 | (z as u8 >> 5) << 4;
            z += 1;
        }
        f
    };

    /// One stage of the lane unpacks: registers `z` and `z | BIT` become the low and the high halves of their interleave.
    #[inline(always)]
    fn unpack_stage<const BIT: usize, const N: usize>(
        x: &mut [__m512i; N],
        interleave: impl Fn(__m512i, __m512i) -> (__m512i, __m512i),
    ) {
        for i in 0..N / 2 {
            let z = i / BIT * 2 * BIT + i % BIT;
            (x[z], x[z | BIT]) = interleave(x[z], x[z | BIT]);
        }
    }

    /// 24 registers of output bytes, register `o` holding byte `o` of 64 sums of weights, one per lane.
    #[derive(Clone, Copy)]
    pub struct ByteAcc([__m512i; OUT_BYTES]);

    impl ByteAcc {
        pub fn zero() -> Self {
            // SAFETY: the module is compiled only with these target features enabled.
            Self(unsafe { [_mm512_setzero_si512(); OUT_BYTES] })
        }

        /// Add, per lane, the sum of the weights of the set bits of registers `first..first + N`.
        ///
        /// `matrices[j]` weighs the bits of register `first + j`, as [`byte_matrices`] builds them.
        #[inline(always)]
        pub fn add_fold<const N: usize>(&mut self, regs: &LineRegs, first: usize, matrices: &[[u64; OUT_BYTES]; N]) {
            accumulate_into::<N, false>(
                &mut self.0,
                regs.0[first..first + N].try_into().expect("N registers"),
                matrices,
                std::ptr::null(),
            );
        }

        /// The 64 sums as eight vectors of eight, lane `l` of vector `v` being sum [`x8_lane`](Self::x8_lane)`(v, l)`.
        #[cfg(target_feature = "vpclmulqdq")]
        #[inline(always)]
        pub fn to_x8(&self) -> [F192x8; 8] {
            let c: [[__m512i; 8]; 3] = std::array::from_fn(|i| {
                let mut x: [__m512i; 8] = std::array::from_fn(|o| self.0[8 * i + o]);
                // SAFETY: the module is compiled only with these target features enabled.
                unsafe {
                    unpack_stage::<1, 8>(&mut x, |u, w| (_mm512_unpacklo_epi8(u, w), _mm512_unpackhi_epi8(u, w)));
                    unpack_stage::<2, 8>(&mut x, |u, w| {
                        (_mm512_unpacklo_epi16(u, w), _mm512_unpackhi_epi16(u, w))
                    });
                    unpack_stage::<4, 8>(&mut x, |u, w| {
                        (_mm512_unpacklo_epi32(u, w), _mm512_unpackhi_epi32(u, w))
                    });
                }
                x
            });
            std::array::from_fn(|r| F192x8::from_coeffs(c[0][r], c[1][r], c[2][r]))
        }

        /// The lane of the 64 sums that lane `l` of vector `v` of [`to_x8`](Self::to_x8) holds.
        ///
        /// The byte unpacks inside the 128-bit lanes leave the sums in this order, which costs no cross-lane shuffle.
        pub const fn x8_lane(v: usize, l: usize) -> usize {
            2 * ((v as u8).reverse_bits() >> 5) as usize + (l & 1) + 16 * (l >> 1)
        }
    }

    /// The GFNI matrices and the input transpose.
    #[derive(Clone, Debug)]
    pub(super) struct Imp {
        /// `matrices[24 r + o]` maps register `r` of the input transpose to output byte `o`.
        matrices: Vec<u64>,
        /// Stages taking row-major bytes to one register per input byte.
        input: Vec<Stage>,
    }

    impl Imp {
        pub(super) fn new(weights: &[F192]) -> Self {
            let n_chunks = weights.len() / 8;
            let c = n_chunks.trailing_zeros() as usize;

            let skew = c.saturating_sub(6);

            // Register `r` then holds input byte `j(r)`: its bits are register bits skew.. then 0..skew.
            let byte_of_reg = |r: usize| (r >> skew) | ((r & ((1 << skew) - 1)) << (c - skew));
            let bytes: &[[F192; 8]] = weights.as_chunks().0;
            // SAFETY: the module is compiled only with these target features enabled.
            let matrices = (0..n_chunks)
                .flat_map(|r| unsafe { weight_matrices(&bytes[byte_of_reg(r)]) })
                .collect();
            Self {
                matrices,
                input: input_stages(c),
            }
        }

        #[inline]
        pub(super) fn fold_block<const CHUNKS: usize>(&self, rows: &[[u8; CHUNKS]], out: &mut [F192; BLOCK]) {
            let acc = self.accumulate_rows(rows);
            // SAFETY: the module is compiled only with these target features enabled.
            unsafe { store_f192(&acc, out) };
        }

        /// The block as eight vectors, straight from the accumulators.
        #[cfg(target_feature = "vpclmulqdq")]
        #[inline]
        pub(super) fn fold_block_x8<const CHUNKS: usize, const G: usize>(&self, rows: &[[u8; CHUNKS]]) -> [F192x8; 8] {
            x8_from_acc::<G>(&self.accumulate_rows(rows))
        }

        #[cfg(not(target_feature = "vpclmulqdq"))]
        #[inline]
        pub(super) fn fold_block_x8<const CHUNKS: usize, const G: usize>(&self, rows: &[[u8; CHUNKS]]) -> [F192x8; 8] {
            let mut values = [F192::ZERO; BLOCK];
            self.fold_block(rows, &mut values);
            x8_from_values::<G>(&values)
        }

        /// Phases 1 and 2 of up to 64 rows: 24 registers of output bytes.
        #[inline(always)]
        fn accumulate_rows<const CHUNKS: usize>(&self, rows: &[[u8; CHUNKS]]) -> [__m512i; OUT_BYTES] {
            // The block that follows, for the prefetch.
            let next = rows.as_ptr().wrapping_add(BLOCK).cast();
            // A short block folds from a zero-padded copy.
            if rows.len() < BLOCK {
                let mut padded = [[0u8; CHUNKS]; BLOCK];
                padded[..rows.len()].copy_from_slice(rows);
                return self.fold_full::<CHUNKS>(&padded, next);
            }
            self.fold_full::<CHUNKS>(rows.try_into().expect("a full block"), next)
        }

        #[inline(always)]
        fn fold_full<const CHUNKS: usize>(
            &self,
            rows: &[[u8; CHUNKS]; BLOCK],
            next: *const u8,
        ) -> [__m512i; OUT_BYTES] {
            // Phase 1: CHUNKS registers of 64 bytes, row-major, then one register per input byte.
            let mut regs = MaybeUninit::<[__m512i; CHUNKS]>::uninit();
            let dst = regs.as_mut_ptr().cast::<__m512i>();
            let c = CHUNKS.trailing_zeros() as usize;
            // Stages run in groups of up to three, each group on tiles of registers that stay in registers.
            // The first group reads the rows, the next ones the registers the previous group wrote.
            let mut from = rows.as_ptr().cast::<__m512i>();
            for first in (0..c.min(6)).step_by(3) {
                // SAFETY: `from` holds CHUNKS registers, the rows or what the previous group wrote.
                let load = |r: usize| unsafe { _mm512_loadu_si512(from.add(r).cast()) };
                match c.min(6) - first {
                    1 => stages::<CHUNKS, 2>(&self.input, first, load, dst),
                    2 => stages::<CHUNKS, 4>(&self.input, first, load, dst),
                    _ => stages::<CHUNKS, 8>(&self.input, first, load, dst),
                }
                from = dst;
            }
            // SAFETY: the first group wrote every register.
            self.accumulate::<CHUNKS, true>(unsafe { regs.assume_init_ref() }, next)
        }

        #[inline]
        pub(super) fn fold_columns<const CHUNKS: usize>(&self, cols: &[[u8; BLOCK]; CHUNKS], out: &mut [F192; BLOCK]) {
            // SAFETY: the module is compiled only with these target features enabled.
            unsafe { store_f192(&self.fold_columns_full::<CHUNKS>(cols), out) }
        }

        #[inline]
        #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vbmi", enable = "gfni")]
        fn fold_columns_full<const CHUNKS: usize>(&self, cols: &[[u8; BLOCK]; CHUNKS]) -> [__m512i; OUT_BYTES] {
            // The register order the input stages leave: register `r` holds input byte `j(r)`.
            let c = CHUNKS.trailing_zeros() as usize;
            let skew = c.saturating_sub(6);
            let byte_of_reg = |r: usize| (r >> skew) | ((r & ((1 << skew) - 1)) << (c - skew));
            // SAFETY: each column is one 64-byte register.
            let regs: [__m512i; CHUNKS] =
                std::array::from_fn(|r| unsafe { _mm512_loadu_si512(cols[byte_of_reg(r)].as_ptr().cast()) });
            self.accumulate::<CHUNKS, false>(&regs, std::ptr::null())
        }

        /// Phase 2 of the block, from one register per input byte: 24 registers of output bytes.
        ///
        /// With `PREFETCH`, `next` is the block that follows (see [`accumulate_into`]).
        #[inline(always)]
        fn accumulate<const CHUNKS: usize, const PREFETCH: bool>(
            &self,
            regs: &[__m512i; CHUNKS],
            next: *const u8,
        ) -> [__m512i; OUT_BYTES] {
            // SAFETY: the module is compiled only with these target features enabled.
            let mut acc = unsafe { [_mm512_setzero_si512(); OUT_BYTES] };
            let matrices = self.matrices.as_chunks::<OUT_BYTES>().0;
            accumulate_into::<CHUNKS, PREFETCH>(&mut acc, regs, matrices.try_into().expect("a matrix per register"), next);
            acc
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_rng::Rng;

    /// Register `j` holds byte `j` of the 64 lines, line `p` in byte `p`.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi"
    ))]
    #[test]
    fn line_regs_transpose_the_lines() {
        let mut rng = Rng::new(0x71_4E);
        let lines: Vec<[u8; 64]> = (0..64).map(|_| std::array::from_fn(|_| rng.next_u64() as u8)).collect();
        let mut regs = gfni::LineRegs::new();
        regs.load(std::array::from_fn(|p| &lines[p]));
        for j in 0..64 {
            let mut got = [0u8; 64];
            // SAFETY: the cfg above enables the target features, and a register is 64 bytes.
            unsafe { core::arch::x86_64::_mm512_storeu_si512(got.as_mut_ptr().cast(), regs.registers()[j]) };
            assert_eq!(got, std::array::from_fn(|p| lines[p][j]), "register {j}");
        }
    }

    /// Every row width, full and short blocks, against the byte-table fold.
    #[test]
    fn fold_block_matches_lookup() {
        fn check<const CHUNKS: usize>(rng: &mut Rng) {
            // Random weights, one per bit of a CHUNKS-byte row.
            let weights: Vec<F192> = (0..8 * CHUNKS).map(|_| rng.ext()).collect();
            let fold = BitFold::new(&weights);
            let tables = lookup_tables(&weights);

            // Full blocks of random rows, then a short block that exercises the zero padding.
            for len in [BLOCK, BLOCK, 5] {
                let rows: Vec<[u8; CHUNKS]> = (0..len)
                    .map(|_| std::array::from_fn(|_| rng.next_u64() as u8))
                    .collect();
                let mut out = [F192::ZERO; BLOCK];
                fold.fold_block(&rows, &mut out);
                for (p, row) in rows.iter().enumerate() {
                    assert_eq!(
                        out[p],
                        fold_row_lookup(&tables, row),
                        "CHUNKS={CHUNKS}, len={len}, row {p}"
                    );
                }
                // The vector orders, against the same values in natural order.
                let x8 = |vectors: [F192x8; 8]| vectors.map(F192x8::to_array);
                assert_eq!(
                    x8(fold.fold_block_x8::<CHUNKS, 0>(&rows)),
                    x8(x8_from_values::<0>(&out)),
                    "CHUNKS={CHUNKS}, len={len}, g=0"
                );
                assert_eq!(
                    x8(fold.fold_block_x8::<CHUNKS, 1>(&rows)),
                    x8(x8_from_values::<1>(&out)),
                    "CHUNKS={CHUNKS}, len={len}, g=1"
                );
                assert_eq!(
                    x8(fold.fold_block_x8::<CHUNKS, 2>(&rows)),
                    x8(x8_from_values::<2>(&out)),
                    "CHUNKS={CHUNKS}, len={len}, g=2"
                );
                if len == BLOCK {
                    let cols: [[u8; BLOCK]; CHUNKS] = std::array::from_fn(|j| std::array::from_fn(|p| rows[p][j]));
                    let mut by_columns = [F192::ZERO; BLOCK];
                    fold.fold_columns(&cols, &mut by_columns);
                    assert_eq!(by_columns, out, "CHUNKS={CHUNKS}, columns");
                }
            }
        }
        let mut rng = Rng::new(0xB17_F01D);
        check::<8>(&mut rng);
        check::<16>(&mut rng);
        check::<32>(&mut rng);
        check::<64>(&mut rng);
        check::<128>(&mut rng);
    }

    /// The byte tables against the definition: the sum of the weights of the set bits.
    #[test]
    fn lookup_matches_definition() {
        let mut rng = Rng::new(0x5E7_B175);
        let weights: Vec<F192> = (0..64).map(|_| rng.ext()).collect();
        let tables = lookup_tables(&weights);
        for _ in 0..64 {
            let row: [u8; 8] = std::array::from_fn(|_| rng.next_u64() as u8);
            // Bit s of the row is bit s % 8 of byte s / 8.
            let direct = (0..64)
                .filter(|s| (row[s / 8] >> (s % 8)) & 1 == 1)
                .fold(F192::ZERO, |acc, s| acc + weights[s]);
            assert_eq!(fold_row_lookup(&tables, &row), direct, "row={row:02x?}");
        }
    }
}
