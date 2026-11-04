//! Coset-major encoding: small codewords, built one coset at a time in cache.

use primitives::field::F64;
use primitives::log2_strict_usize;

use super::encode::RowSink;
use super::layers::{Layers, no_hook};
use super::plan::L2_WORDS;
use super::openings::built_rows;
use super::scratch::with_scratch;

/// The coset-major encode of a codeword that is `2^r` cosets, each a transform of one copy of the message.
///
/// - It owns the shape: the log of the codeword's rows, the lanes and the log inverse rate `r`.
/// - Coset `c` is the message transformed from layer `r` on, at block index `c` of layer `r`.
/// - A task builds each of its cosets in scratch and hashes it there, so no codeword is ever written.
/// - Use it when [`CosetEncoder::fits`] holds: the scratch is then cache-resident.
pub(super) struct CosetEncoder<'t> {
    layers: Layers<'t>,
    log_d: usize,
    num_ntts: usize,
    log_inv_rate: usize,
}

impl<'t> CosetEncoder<'t> {
    /// The encode of a message of `msg_words` words, `num_ntts` a row.
    pub(super) fn new(layers: Layers<'t>, msg_words: usize, num_ntts: usize, log_inv_rate: usize) -> Self {
        assert!(num_ntts > 0 && msg_words.is_multiple_of(num_ntts));
        let log_d = log2_strict_usize(msg_words / num_ntts) + log_inv_rate;
        assert!(log_d <= layers.log_domain_size());
        Self { layers, log_d, num_ntts, log_inv_rate }
    }

    /// Whether a message of `msg_words` words encodes coset by coset: one coset fits the cache budget [`COSET_WORDS`].
    pub(super) fn fits(msg_words: usize) -> bool {
        msg_words <= COSET_WORDS
    }

    /// Hand `sink` every finished block of rows, and write no codeword.
    ///
    /// - Several cosets that are small share a scratch, so the blocks are all one size and reach the hasher in batches.
    /// - [`CosetEncoder::rows`] recomputes any row from the message.
    pub(super) fn encode(&self, msg: &[F64], sink: &RowSink<'_>) {
        let rows = msg.len() / self.num_ntts;
        let group = 1usize << (L2_WORDS / msg.len()).max(1).ilog2().min(self.log_inv_rate as u32);
        parallel::for_each_chunk((1 << self.log_inv_rate) / group, |lo, hi| {
            with_scratch(group * msg.len(), |scratch| {
                for g in lo..hi {
                    for (i, coset) in scratch.chunks_exact_mut(msg.len()).enumerate() {
                        self.build(coset, msg, g * group + i, None);
                    }
                    sink(g * group * rows, scratch);
                }
            });
        });
    }

    /// Codeword rows `positions` (sorted) of the encode [`CosetEncoder::encode`] ran, back to back.
    ///
    /// It runs one transform per coset the positions touch.
    pub(super) fn rows(&self, msg: &[F64], positions: &[usize]) -> Vec<F64> {
        built_rows(self.num_ntts, self.log_d - self.log_inv_rate, positions, |_, coset, live, buf| {
            self.build(buf, msg, coset, Some(live))
        })
    }

    /// Coset `coset` of the codeword, built into `buf` (one message long) from the message.
    ///
    /// With `live`, only those rows of the coset come out right.
    fn build(&self, buf: &mut [F64], msg: &[F64], coset: usize, live: Option<&[usize]>) {
        let (log_d, first, end) = (self.log_d, self.log_inv_rate, self.log_d);
        self.layers.run(buf, Some(msg), log_d, self.num_ntts, first, end, self.log_inv_rate, coset, live, no_hook);
    }
}

/// Words of one coset a coset-major encode may span.
///
/// # Why this value
///
/// - 2^19 words is 4 MiB: the largest coset it admits is 2^13 rows of 48 words, 3 MiB.
/// - A coset runs in a scratch that lives in L2 and L3, and never reaches DRAM.
/// - Four nodes of two threads each share one 32 MiB L3, so a worker's share is 4 MiB.
/// - A larger coset spills to DRAM, and then the codeword's write pass is the cheaper plan.
const COSET_WORDS: usize = 1 << 19;
