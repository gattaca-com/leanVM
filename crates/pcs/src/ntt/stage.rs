//! The lane-major message: its transpose, and the stage a gathered first pass fills from it.

use flux_profiler::timed;
use primitives::field::F64;
use primitives::stream::Stream;

use super::additive_ntt_f64::{STREAM_MIN_WORDS, with_scratch};
use crate::stack_open::EQUAL_BLOCK_LOG;

/// Transpose a lane-major message into the row-major order the encoder reads.
///
/// # Layout
///
/// ```text
///     input, lane-major:   [ block 0 | block 1 | ... | block n-1 ]   each 2^rows_log words
///     output, row-major:   row x = [ 0, ..., 0, block n-1 word x, ..., block 0 word x ]
/// ```
///
/// - A row is `row_words >= n` words: the first `row_words - n` lanes are zero padding.
/// - Codeword lane `t` past the padding takes message block `row_words - 1 - t`.
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
/// - That keeps an n-way gather at a `2^rows_log` stride near memory bandwidth.
#[timed]
pub(super) fn transpose_lane_major(out: &mut [F64], msg: &[F64], n_lanes: usize, row_words: usize, log_rows: usize) {
    let rows = 1usize << log_rows;
    assert!(n_lanes > 0, "a commitment needs at least one lane");
    assert!(n_lanes <= row_words, "a row holds its lanes");
    assert_eq!(msg.len(), n_lanes * rows, "message is n_lanes contiguous lane blocks");
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
    let pad = row_words - n_lanes;
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
                    let src = &msg[(row_words - 1 - lane) * rows + r0..][..tile_rows];
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
    if lane_major && fits && parallel::num_threads() <= 4 { STAGE_SUBS } else { 1 }
}

/// The tiles of a stage that repeat an earlier tile, as a bit per lane of each octet.
pub(super) struct StageCopies {
    /// A repeat tile is a copy of the tile this many rows before it.
    back: usize,
    rows: usize,
    n_octets: usize,
    /// Per `t0 >> EQUAL_BLOCK_LOG`, octet of lanes and row of the stage: bit `k` is set if lane `8 * octet + k`
    /// of the tile repeats the tile `back` rows before.
    masks: Vec<u8>,
}

impl StageCopies {
    pub(super) fn none() -> Self {
        Self { back: 0, rows: 0, n_octets: 0, masks: Vec::new() }
    }

    /// The masks of one octet of the batch of residues at `t0`, or none when no tile of the stage repeats.
    fn octet(&self, octet: usize, t0: usize) -> &[u8] {
        let at = ((t0 >> EQUAL_BLOCK_LOG) * self.n_octets + octet) * self.rows;
        self.masks.get(at..at + self.rows).unwrap_or(&[])
    }
}

/// Which tiles of a lane-major stage are copies of the tile `back` rows before them.
///
/// - Stage row `i` of residue `r` is message row `r + i * 2^log_step` of every lane. Going back `back` rows
///   is going back `D = max(2^log_step, 2^EQUAL_BLOCK_LOG)` words: a whole number `m` of blocks.
/// - The line of row `i` equals the line of row `i - back` if the lane's `m` blocks up to the one holding
///   the line each repeat their predecessor: a run of repeats at least `m` long ends there.
/// - The block holding the line is `(i << log_step >> EQUAL_BLOCK_LOG) + u`, with `u = r >> EQUAL_BLOCK_LOG`
///   below `m`, so one mask table serves every residue of a `u`.
/// - A padding lane is zero in both rows, so its bit is set too.
/// - Rows before `back` read the message, as do all rows of a message with no map.
///
/// `repeats` is the message's map of blocks that equal their predecessor. A lane's first block never repeats.
pub(super) fn stage_copies(
    repeats: &[bool],
    log_step: usize,
    rows: usize,
    msg_rows: usize,
    n_lanes: usize,
    row_words: usize,
) -> StageCopies {
    if repeats.is_empty() || !msg_rows.is_multiple_of(1 << EQUAL_BLOCK_LOG) {
        return StageCopies::none();
    }
    let distance = (1usize << log_step).max(1 << EQUAL_BLOCK_LOG);
    let (back, m) = (distance >> log_step, distance >> EQUAL_BLOCK_LOG);
    if back >= rows {
        return StageCopies::none();
    }
    let (pad, n_octets, per_lane) = (row_words - n_lanes, row_words / 8, msg_rows >> EQUAL_BLOCK_LOG);
    let mut masks = vec![0u8; m * n_octets * rows];
    // The length of the run of repeats ending at each block of the lane.
    let mut run = vec![0usize; per_lane];
    for lane in pad..row_words {
        let blocks = &repeats[(row_words - 1 - lane) * per_lane..][..per_lane];
        let mut length = 0;
        for (h, (&repeat, run)) in blocks.iter().zip(&mut run).enumerate() {
            length = if repeat && h > 0 { length + 1 } else { 0 };
            *run = length;
        }
        for u in 0..m {
            let column = &mut masks[(u * n_octets + lane / 8) * rows..][..rows];
            for (i, mask) in column.iter_mut().enumerate().skip(back) {
                if run[(i << log_step >> EQUAL_BLOCK_LOG) + u] >= m {
                    *mask |= 1 << (lane % 8);
                }
            }
        }
    }
    for lane in 0..pad {
        for u in 0..m {
            let column = &mut masks[(u * n_octets + lane / 8) * rows..][..rows];
            column.iter_mut().skip(back).for_each(|mask| *mask |= 1 << (lane % 8));
        }
    }
    StageCopies { back, rows, n_octets, masks }
}

/// Transpose one octet of lanes of `subs` batches of [`BATCH`] consecutive residues of a lane-major message into `stage`.
///
/// - `stage` is `subs` stages of one batch each, and batch `u` starts at residue `t0 + u * BATCH`.
/// - A row's tiles of all batches go together: the lines of one lane block that sit next to each other are read next to each other.
/// - Message row `j` is `msg_rows` words apart per lane block, and word `w` of a row of `row_words` words
///   is lane block `row_words - 1 - w` of the message, or zero while `w` is below the padding.
/// - Residue `t0 + s` takes rows `t0 + s + i * step` for `i < rows`, `step = 2^log_step`.
/// - The octet is lanes `8 * octet..8 * octet + 8`. Together with the residues, they are 64 lanes of one transform.
/// - `stage` has `rows` rows of 64 words: lane `8 * octet + k` of row `t0 + s + i * step` is word `(i * BATCH + s) * 8 + k`.
/// - One tile is 8 words of 8 lane blocks, so each 64 byte line of a lane block is read once, whole.
/// - A lane of tile `i` with its bit set in the octet's [`StageCopies`] is copied from the tile `back` rows
///   before, which is already in the stage, and its line is not read.
pub(super) fn fill_stage(
    stage: &mut [F64],
    msg: &[F64],
    t0: usize,
    log_step: usize,
    msg_rows: usize,
    n_lanes: usize,
    row_words: usize,
    octet: usize,
    copies: &StageCopies,
    subs: usize,
) {
    // SAFETY: AVX-512F is enabled at compile time.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe {
        fill_stage_tiles(stage, msg, t0, log_step, msg_rows, n_lanes, row_words, octet, copies, subs)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    fill_stage_tiles(stage, msg, t0, log_step, msg_rows, n_lanes, row_words, octet, copies, subs);
}

/// [`fill_stage`], a target-feature function where AVX-512 is enabled, so its tiles move through registers.
#[cfg_attr(all(target_arch = "x86_64", target_feature = "avx512f"), target_feature(enable = "avx512f"), inline)]
fn fill_stage_tiles(
    stage: &mut [F64],
    msg: &[F64],
    t0: usize,
    log_step: usize,
    msg_rows: usize,
    n_lanes: usize,
    row_words: usize,
    octet: usize,
    copies: &StageCopies,
    subs: usize,
) {
    let back = copies.back;
    let sub_words = stage.len() / subs;
    let rows = sub_words / (BATCH * 8);
    let pad = row_words - n_lanes;
    // Lane block of line `k` of the tile, which is lane `8 * octet + k`.
    let block = |k: usize| row_words - 1 - (8 * octet + k);
    // The lines of a tile that are padding lanes.
    let pad_lines = (0..8).filter(|k| 8 * octet + k < pad).fold(0u8, |m, k| m | 1 << k);
    let masks: [&[u8]; STAGE_SUBS] = std::array::from_fn(|u| if u < subs { copies.octet(octet, t0 + u * BATCH) } else { &[] });
    for i in 0..rows {
        // The sub-batches' lines of one lane are adjacent: a row's tiles go together.
        for (u, masks) in masks.iter().enumerate().take(subs) {
            let j = t0 + u * BATCH + (i << log_step);
            let sub = &mut stage[u * sub_words..][..sub_words];
            let copy = masks.get(i).copied().unwrap_or(0);
            if copy != 0xFF {
                // Line `k` of the tile: 8 residues of one lane block, or zeros for a padding lane or a copy.
                let src: [usize; 8] = std::array::from_fn(|k| block(k) * msg_rows + j);
                fill_tile(sub, i * BATCH, msg, src, pad_lines | copy);
            }
            if copy != 0 {
                // Tile `i - back` is a tile of the stage, and `copies` has no bit before row `back`.
                copy_tile(sub, i, i - back, copy);
            }
        }
    }
}

/// Write `sub[tile + s][k] = msg[src[k] + s]` for the 8 x 8 words of eight lines, a line in `zeros` being zeros.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline]
#[target_feature(enable = "avx512f")]
fn fill_tile(sub: &mut [F64], tile: usize, msg: &[F64], src: [usize; 8], zeros: u8) {
    use core::arch::x86_64::*;
    use primitives::simd::{load, store};
    let r: [__m512i; 8] =
        std::array::from_fn(|k| if zeros >> k & 1 == 1 { _mm512_setzero_si512() } else { load(&msg[src[k]..], 0) });
    // Pairs of columns: lanes `[(r0, r1), (r2, r3)]`, then 128-bit lane picks to finish.
    let (a0, a1) = (_mm512_unpacklo_epi64(r[0], r[1]), _mm512_unpackhi_epi64(r[0], r[1]));
    let (a2, a3) = (_mm512_unpacklo_epi64(r[2], r[3]), _mm512_unpackhi_epi64(r[2], r[3]));
    let (a4, a5) = (_mm512_unpacklo_epi64(r[4], r[5]), _mm512_unpackhi_epi64(r[4], r[5]));
    let (a6, a7) = (_mm512_unpacklo_epi64(r[6], r[7]), _mm512_unpackhi_epi64(r[6], r[7]));
    let b0 = _mm512_shuffle_i64x2::<0x88>(a0, a2);
    let b1 = _mm512_shuffle_i64x2::<0x88>(a1, a3);
    let b2 = _mm512_shuffle_i64x2::<0xDD>(a0, a2);
    let b3 = _mm512_shuffle_i64x2::<0xDD>(a1, a3);
    let b4 = _mm512_shuffle_i64x2::<0x88>(a4, a6);
    let b5 = _mm512_shuffle_i64x2::<0x88>(a5, a7);
    let b6 = _mm512_shuffle_i64x2::<0xDD>(a4, a6);
    let b7 = _mm512_shuffle_i64x2::<0xDD>(a5, a7);
    let cols = [
        _mm512_shuffle_i64x2::<0x88>(b0, b4),
        _mm512_shuffle_i64x2::<0x88>(b1, b5),
        _mm512_shuffle_i64x2::<0x88>(b2, b6),
        _mm512_shuffle_i64x2::<0x88>(b3, b7),
        _mm512_shuffle_i64x2::<0xDD>(b0, b4),
        _mm512_shuffle_i64x2::<0xDD>(b1, b5),
        _mm512_shuffle_i64x2::<0xDD>(b2, b6),
        _mm512_shuffle_i64x2::<0xDD>(b3, b7),
    ];
    for (s, col) in cols.into_iter().enumerate() {
        store(sub, tile + s, col);
    }
}

/// [`fill_tile`] word by word.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn fill_tile(sub: &mut [F64], tile: usize, msg: &[F64], src: [usize; 8], zeros: u8) {
    for s in 0..8 {
        for k in 0..8 {
            sub[(tile + s) * 8 + k] = if zeros >> k & 1 == 1 { F64::ZERO } else { msg[src[k] + s] };
        }
    }
}

/// Copy the lanes of tile `from` of a stage into tile `i`, for the bits set in `mask`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline]
#[target_feature(enable = "avx512f")]
fn copy_tile(sub: &mut [F64], i: usize, from: usize, mask: u8) {
    use core::arch::x86_64::*;
    use primitives::simd::{load, store};
    for s in 0..BATCH {
        let (kept, copied) = (load(sub, i * BATCH + s), load(sub, from * BATCH + s));
        store(sub, i * BATCH + s, _mm512_mask_blend_epi64(mask, kept, copied));
    }
}

/// [`copy_tile`] word by word.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn copy_tile(sub: &mut [F64], i: usize, from: usize, mask: u8) {
    for s in 0..BATCH {
        for k in (0..8).filter(|k| mask >> k & 1 == 1) {
            sub[(i * BATCH + s) * 8 + k] = sub[(from * BATCH + s) * 8 + k];
        }
    }
}
