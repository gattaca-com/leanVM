//! The lane-major message: its transpose, and the stage a gathered first pass fills from it.

use primitives::field::F64;
use primitives::stream::Stream;

use super::scratch::with_scratch;
use super::source::Lanes;
use super::window::STREAM_MIN_WORDS;
use crate::stack_open::EQUAL_BLOCK_LOG;

/// Transpose a lane-major message into the row-major order the encodes read.
///
/// # Layout
///
/// ```text
///     input, lane-major:   [ block 0 | block 1 | ... | block n-1 ]   each 2^log_rows words
///     output, row-major:   row x = [ 0, ..., 0, block b[0] word x, ..., block b[n-1] word x ]
/// ```
///
/// - `blocks` lists the `n` message blocks to encode, and a row is `row_words >= n` words:
///   the first `row_words - n` lanes are zero padding.
/// - Codeword lane `pad + i` takes message block `blocks[i]`. A commitment lists its blocks descending.
/// - Any lane count works: a commitment encodes only the lanes that carry data, and a caller pads
///   the row to the vector width.
///
/// # Why descending
///
/// - The absent all-zero lanes lead every Merkle leaf image, so the proof can leave them out.
/// - `whir::commit` has the full reason.
///
/// # Performance
///
/// - Rows are handled one tile at a time.
/// - Each lane adds a contiguous burst of words to an L1-resident tile.
/// - The finished tile goes out as one contiguous run.
/// - That keeps an n-way gather at a `2^log_rows` stride near memory bandwidth.
pub(super) fn transpose_lane_major(out: &mut [F64], msg: &[F64], blocks: &[usize], row_words: usize, log_rows: usize) {
    let rows = 1usize << log_rows;
    assert!(!blocks.is_empty(), "a commitment needs at least one lane");
    assert!(blocks.len() <= row_words, "a row holds its lanes");
    assert!(
        blocks.iter().all(|&b| (b + 1) * rows <= msg.len()),
        "a block lies in the message"
    );
    assert_eq!(out.len(), row_words * rows, "the output is whole padded rows");

    /// Words per cache-resident row tile.
    const TILE_WORDS: usize = 4096;
    assert!(row_words <= TILE_WORDS, "a codeword row must fit the transpose tile");
    // Largest power-of-two row count whose tile fits: both it and `rows` are then
    // powers of two, so the tiles cover every row. They have to: this is the sole
    // initializer of an uninitialized codeword's message region, and a truncating
    // tile count would leave the tail reading the previous phase's plausible bytes,
    // whose symptom is a proof that stops verifying rather than a crash.
    let tile_rows = (1usize << (TILE_WORDS / row_words).ilog2()).min(rows);
    assert_eq!(rows % tile_rows, 0, "row tiles must cover every row");

    // Nothing reads the output back before a buffer this large is evicted.
    let stream = out.len() >= STREAM_MIN_WORDS;
    let pad = row_words - blocks.len();
    parallel::chunks_mut(out, tile_rows * row_words, |t, out_tile| {
        with_scratch(out_tile.len(), |tile| {
            // First message row this tile covers.
            let r0 = t * tile_rows;
            // Fill the tile column by column: lane `lane` takes every `row_words`-th word.
            for lane in 0..row_words {
                let slots = tile[lane..].iter_mut().step_by(row_words);
                if lane < pad {
                    slots.for_each(|slot| *slot = F64::ZERO);
                } else {
                    let src = &msg[blocks[lane - pad] * rows + r0..][..tile_rows];
                    for (slot, &word) in slots.zip(src) {
                        *slot = word;
                    }
                }
            }
            // Write the finished tile out as one contiguous run.
            if stream {
                Stream::new().copy(out_tile, tile);
            } else {
                out_tile.copy_from_slice(tile);
            }
        });
    });
}

/// Residues a lane-major first pass takes at once, as a log: one cache line of a lane block.
pub(super) const LOG_BATCH: usize = 3;
pub(super) const BATCH: usize = 1 << LOG_BATCH;
/// Most batches a lane-major task fills side by side, so a lane's adjacent lines are read together.
const STAGE_SUBS: usize = 2;

/// Batches a lane-major task fills side by side, as one stage of that many batches.
///
/// # Why this value
///
/// - A batch reads one 64 byte line per lane block and row, 16 KB from the next row's.
/// - Each such line opens its own DRAM page: 4 threads read it at 20 GB/s, where 128 byte runs reach 31 GB/s.
/// - A second batch's line is the neighbour of the first's, so the fill reads both in one pass.
/// - The stage doubles to 2^(g + 10) words, which spills from L2 into L3.
/// - Measured on the base encode at 8:1 (rate 2 / rate 1, ms): 1 thread 944 -> 919 / 540 -> 504,
///   2 threads 496 -> 482 / 287 -> 270, 4 threads 294 -> 286 / 169 -> 162.
/// - At 8 threads it changes nothing, and at 16 threads it costs 3 %: the stage then crowds the shared L2 and L3.
/// - So a pool of more than 4 threads keeps one batch.
/// - A step under two batches has no neighbour line to read.
pub(super) fn stage_subs(lane_major: bool, log_step: usize) -> usize {
    let fits = log_step >= LOG_BATCH + STAGE_SUBS.ilog2() as usize;
    if lane_major && fits && parallel::num_threads() <= 4 {
        STAGE_SUBS
    } else {
        1
    }
}

/// The transpose of a lane-major message through a stage, for one gathered pass.
///
/// - It owns the message and the geometry of the fill: the step, the message rows, the row width and the batches.
/// - [`Stage::fill`] moves one octet of lanes of `subs` batches of residues into a stage buffer.
/// - A task fills the stage, then each block of the pass transforms a copy of each batch.
///
/// # Repeated blocks
///
/// - A line in a run of repeated blocks reads the equal line of the run's head instead, which is in cache.
/// - The head line has the same residue: it lies `k * D` words back, `D = max(2^log_step, 2^EQUAL_BLOCK_LOG)`.
/// - `D` is `m` blocks, so a block `r` repeats into its run reads `r / m * D` words back.
/// - The fill read that line `k` stage rows before, so the run's other lines never leave DRAM.
pub(super) struct Stage<'a> {
    lanes: Lanes<'a>,
    log_step: usize,
    msg_rows: usize,
    row_words: usize,
    subs: usize,
    /// Per encoded lane and block of `2^EQUAL_BLOCK_LOG` words, how far back its lines are read, or empty.
    back: Vec<usize>,
}

impl<'a> Stage<'a> {
    /// The stage of a pass whose groups are `2^log_step` rows apart, over `lanes` of `msg_rows` rows each.
    pub(super) fn new(lanes: Lanes<'a>, log_step: usize, msg_rows: usize, row_words: usize, subs: usize) -> Self {
        let per_lane = msg_rows >> EQUAL_BLOCK_LOG;
        let mut back = Vec::new();
        if !lanes.repeats.is_empty() && per_lane > 0 {
            let d = (1usize << log_step).max(1 << EQUAL_BLOCK_LOG);
            back = vec![0; lanes.blocks.len() * per_lane];
            for (back, &b) in back.chunks_exact_mut(per_lane).zip(lanes.blocks) {
                let mut run = 0;
                for (h, back) in back.iter_mut().enumerate() {
                    run = if h > 0 && lanes.repeats[b * per_lane + h] {
                        run + 1
                    } else {
                        0
                    };
                    *back = run / (d >> EQUAL_BLOCK_LOG) * d;
                }
            }
        }
        Self {
            lanes,
            log_step,
            msg_rows,
            row_words,
            subs,
            back,
        }
    }

    /// Transpose one octet of lanes of `subs` batches of [`BATCH`] consecutive residues from `t0` into `stage`.
    ///
    /// - `stage` is `subs` stages of one batch each, and batch `u` starts at residue `t0 + u * BATCH`.
    /// - A row's tiles of all batches go together, so the adjacent lines of one lane block are read together.
    /// - Message row `j` is `msg_rows` words apart per lane block, and word `w` of a row of `row_words` words
    ///   is message block `blocks[w - pad]`, or zero while `w` is below the padding.
    /// - Residue `t0 + s` takes rows `t0 + s + i * step` for `i < rows`, `step = 2^log_step`.
    /// - The octet is lanes `8 * octet..8 * octet + 8`. Together with the residues, they are 64 lanes of one transform.
    /// - `stage` has `rows` rows of 64 words.
    ///   Lane `8 * octet + k` of row `t0 + s + i * step` is word `(i * BATCH + s) * 8 + k`.
    /// - One tile is 8 words of 8 lane blocks, so each 64 byte line of a lane block is read once, whole.
    #[inline]
    pub(super) fn fill(&self, stage: &mut [F64], t0: usize, octet: usize) {
        // SAFETY: AVX-512F is enabled at compile time.
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            fill_stage_tiles(self, stage, t0, octet)
        }
        // SAFETY: AVX2 is enabled at compile time.
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
        unsafe {
            fill_stage_tiles(self, stage, t0, octet)
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", target_feature = "avx512f"),
            all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f"))
        )))]
        fill_stage_tiles(self, stage, t0, octet);
    }
}

/// [`Stage::fill`], a target-feature function where AVX-512 or AVX2 is enabled, so its tiles move through registers.
#[cfg_attr(
    all(target_arch = "x86_64", target_feature = "avx512f"),
    target_feature(enable = "avx512f"),
    inline
)]
#[cfg_attr(
    all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")),
    target_feature(enable = "avx2"),
    inline
)]
fn fill_stage_tiles(st: &Stage, stage: &mut [F64], t0: usize, octet: usize) {
    let Stage {
        lanes: Lanes { words: msg, blocks, .. },
        log_step,
        msg_rows,
        row_words,
        subs,
        ref back,
    } = *st;
    let sub_words = stage.len() / subs;
    let rows = sub_words / (BATCH * 8);
    let pad = row_words - blocks.len();
    let per_lane = msg_rows >> EQUAL_BLOCK_LOG;
    // The first word of line `k` of the tile at message row `j`, which is lane `8 * octet + k`.
    // A padding lane has none, and is never read.
    let line = |k: usize, j: usize| {
        (8 * octet + k).checked_sub(pad).map_or(0, |l| {
            let back = back.get(l * per_lane + (j >> EQUAL_BLOCK_LOG)).copied().unwrap_or(0);
            blocks[l] * msg_rows + j - back
        })
    };
    // The lines of a tile that are padding lanes.
    let pad_lines = (0..8).filter(|k| 8 * octet + k < pad).fold(0u8, |m, k| m | 1 << k);
    for i in 0..rows {
        for (u, sub) in stage.chunks_exact_mut(sub_words).enumerate() {
            // Line `k` of the tile: 8 residues of one lane block, or zeros for a padding lane.
            let j = t0 + u * BATCH + (i << log_step);
            fill_tile(sub, i * BATCH, msg, std::array::from_fn(|k| line(k, j)), pad_lines);
        }
    }
}

/// Write `stage[tile + s][k] = msg[src[k] + s]` for the 8 x 8 words of eight lines, a line in `zeros` being zeros.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline]
#[target_feature(enable = "avx512f")]
fn fill_tile(stage: &mut [F64], tile: usize, msg: &[F64], src: [usize; 8], zeros: u8) {
    use core::arch::x86_64::*;
    use primitives::simd::{load, store, transpose8};
    let r: [__m512i; 8] = std::array::from_fn(|k| {
        if zeros >> k & 1 == 1 {
            _mm512_setzero_si512()
        } else {
            load(&msg[src[k]..], 0)
        }
    });
    let cols = transpose8(r);
    for (s, col) in cols.into_iter().enumerate() {
        store(stage, tile + s, col);
    }
}

/// [`fill_tile`] as two 4x4 transposes of YMM halves per half of the lines.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
#[inline]
#[target_feature(enable = "avx2")]
fn fill_tile(stage: &mut [F64], tile: usize, msg: &[F64], src: [usize; 8], zeros: u8) {
    use core::arch::x86_64::*;
    use primitives::{load_at, store_at};
    // Line `k`, half `h`: words `4h..4h + 4` of the line's 8 residues.
    let r: [[__m256i; 2]; 8] = std::array::from_fn(|k| {
        if zeros >> k & 1 == 1 {
            [_mm256_setzero_si256(); 2]
        } else {
            [load_at(msg, src[k]), load_at(msg, src[k] + 4)]
        }
    });
    // Residue `s` is word `s` of every line: lines 0..4 in its first half, 4..8 in its second.
    for h in 0..2 {
        for q in 0..2 {
            let t = |k: usize| r[4 * q + k][h];
            let (a0, a1) = (_mm256_unpacklo_epi64(t(0), t(1)), _mm256_unpackhi_epi64(t(0), t(1)));
            let (a2, a3) = (_mm256_unpacklo_epi64(t(2), t(3)), _mm256_unpackhi_epi64(t(2), t(3)));
            let cols = [
                _mm256_permute2x128_si256::<0x20>(a0, a2),
                _mm256_permute2x128_si256::<0x20>(a1, a3),
                _mm256_permute2x128_si256::<0x31>(a0, a2),
                _mm256_permute2x128_si256::<0x31>(a1, a3),
            ];
            for (i, col) in cols.into_iter().enumerate() {
                store_at(stage, (tile + 4 * h + i) * 8 + 4 * q, col);
            }
        }
    }
}

/// [`fill_tile`] word by word.
#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx512f"),
    all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f"))
)))]
#[inline(always)]
fn fill_tile(stage: &mut [F64], tile: usize, msg: &[F64], src: [usize; 8], zeros: u8) {
    for s in 0..8 {
        for k in 0..8 {
            stage[(tile + s) * 8 + k] = if zeros >> k & 1 == 1 {
                F64::ZERO
            } else {
                msg[src[k] + s]
            };
        }
    }
}
