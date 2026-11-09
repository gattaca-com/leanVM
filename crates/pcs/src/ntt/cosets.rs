//! The coset-major encode: a small codeword, built one coset at a time in cache and never written.

use primitives::field::F64;
use primitives::log2_strict_usize;

use super::encode::{Codeword, RowSink};
use super::layers::no_hook;
use super::openings::built_rows;
use super::plan::L2_WORDS;
use super::scratch::with_scratch;
use super::twiddles::AdditiveNttF64;

/// Words of one coset a coset-major encode may span.
///
/// # Why this value
///
/// - 2^19 words is 4 MiB: the largest coset it admits is 2^13 rows of 48 words, 3 MiB.
/// - A coset runs in a scratch that lives in L2 and L3, and never reaches DRAM.
/// - Four nodes of two threads each share one 32 MiB L3, so a worker's share is 4 MiB.
/// - A larger coset spills to DRAM, and then the codeword's write pass is the cheaper plan.
const COSET_WORDS: usize = 1 << 19;

impl AdditiveNttF64 {
    /// Whether a message of `msg_words` words encodes coset by coset: one coset fits [`COSET_WORDS`].
    pub(crate) fn encodes_by_coset(msg_words: usize) -> bool {
        msg_words <= COSET_WORDS
    }

    /// RS-encode a row-major message of `num_ntts` lanes coset by coset, handing `on_rows` every finished block.
    ///
    /// - The codeword is `2^r` cosets, each a transform of one copy of the message from layer `r` on.
    /// - A task builds each of its cosets in scratch and hashes it there, so no codeword is ever written.
    /// - Small cosets share a scratch, so the blocks are all one size and reach the hasher in batches.
    /// - The [`Codeword::Cosets`] keeps the message, and [`AdditiveNttF64::rows`] rebuilds a row from it.
    pub(crate) fn encode_cosets(
        &self,
        msg: Vec<F64>,
        num_ntts: usize,
        log_inv_rate: usize,
        on_rows: &RowSink<'_>,
    ) -> Codeword {
        let (len, rows) = (msg.len(), msg.len() / num_ntts);
        let group = 1usize << (L2_WORDS / len).max(1).ilog2().min(log_inv_rate as u32);
        parallel::for_each_chunk((1 << log_inv_rate) / group, |lo, hi| {
            with_scratch(group * len, |scratch| {
                for g in lo..hi {
                    for (i, coset) in scratch.chunks_exact_mut(len).enumerate() {
                        self.build_coset(coset, &msg, num_ntts, log_inv_rate, g * group + i, None);
                    }
                    on_rows(g * group * rows, scratch);
                }
            });
        });
        Codeword::Cosets {
            msg,
            num_ntts,
            log_inv_rate,
        }
    }

    /// The rows at the sorted `positions` of a coset-major encode of `msg`: one transform per coset touched.
    pub(super) fn coset_rows(
        &self,
        msg: &[F64],
        num_ntts: usize,
        log_inv_rate: usize,
        positions: &[usize],
    ) -> Vec<F64> {
        let log_rows = log2_strict_usize(msg.len() / num_ntts);
        built_rows(num_ntts, log_rows, positions, |_, coset, live, buf| {
            self.build_coset(buf, msg, num_ntts, log_inv_rate, coset, Some(live))
        })
    }

    /// Coset `coset` of the codeword, built into `buf` (one message long) from the message.
    ///
    /// With `live`, only those rows of the coset come out right.
    fn build_coset(
        &self,
        buf: &mut [F64],
        msg: &[F64],
        num_ntts: usize,
        log_inv_rate: usize,
        coset: usize,
        live: Option<&[usize]>,
    ) {
        let log_d = log2_strict_usize(msg.len() / num_ntts) + log_inv_rate;
        self.run_layers(
            buf,
            Some(msg),
            log_d,
            num_ntts,
            log_inv_rate,
            log_inv_rate,
            coset,
            live,
            no_hook,
        );
    }
}
