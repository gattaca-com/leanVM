//! Where the zerocheck passes read the packed bits `a`, `b`, `c` from.
//!
//! A pass reads each bit table once, in order, a few KB at a time. The bits come from one of two sources:
//!
//! - **Given**: three tables in memory.
//! - **Rebuilt**: a [`Rebuild`] regenerates a group of instances into a per-worker [`Scratch`] that stays in cache.
//!   The pass then reads the scratch, and the tables never cross memory.
//!
//! A pass asks for a [`Window`] over a byte range of the tables. A rebuilt window is whole groups.

use super::multilinear::PackedWitness;
use crate::witness::{GroupRows, packed_bytes};
use std::ops::Range;

/// Rebuilds the three bit tables one group of instances at a time.
pub trait Rebuild: Sync {
    /// Bytes of each table that a group covers.
    fn group_bytes(&self) -> usize;
    /// Bytes of each table in all.
    fn len_bytes(&self) -> usize;
    /// Word-major scratch for [`Self::rebuild`], one per worker.
    fn new_rows(&self) -> GroupRows;
    /// Write the `a`, `b`, `c` bytes of `group` into `out`, each [`Self::group_bytes`] long.
    fn rebuild(&self, group: usize, rows: &mut GroupRows, out: [&mut [u64]; 3]);
}

/// The bit tables a pass reads.
#[derive(Clone, Copy)]
pub enum Bits<'a> {
    /// The tables as they are in memory.
    Given(PackedWitness<'a>),
    /// The tables, regenerated group by group.
    Rebuilt(&'a dyn Rebuild),
}

impl<'a> From<PackedWitness<'a>> for Bits<'a> {
    fn from(bits: PackedWitness<'a>) -> Self {
        Bits::Given(bits)
    }
}

/// A worker's cache-resident groups of rebuilt tables.
pub struct Scratch {
    rows: GroupRows,
    /// `a`, `b`, `c`, each `slots` groups of words.
    tables: Vec<u64>,
    /// The group each slot holds.
    held: Vec<usize>,
    group_words: usize,
}

/// A byte range of the three tables, as slices.
#[derive(Clone, Copy)]
pub struct Window<'s> {
    /// `a`, `b`, `c`, from byte `first` of the tables on.
    pub tables: [&'s [u8]; 3],
    /// The byte of the tables that `tables` starts at.
    pub first: usize,
}

impl<'s> Window<'s> {
    /// The rows `first_row..first_row + len` of every table, each row `CHUNKS` bytes of the whole witness.
    #[inline(always)]
    pub fn rows<const CHUNKS: usize>(&self, first_row: usize, len: usize) -> [&'s [[u8; CHUNKS]]; 3] {
        let from = first_row * CHUNKS - self.first;
        self.tables.map(|t| t[from..from + len * CHUNKS].as_chunks().0)
    }
}

impl<'a> Bits<'a> {
    /// Bytes of each table in all: `2^m / 8`.
    pub fn len_bytes(&self) -> usize {
        match self {
            Bits::Given(p) => p.c.len(),
            Bits::Rebuilt(r) => r.len_bytes(),
        }
    }

    /// Check that the tables hold every instance a pass over `padding` reads.
    ///
    /// That is all of them, or with a padding tail those up to and including the template block `live_blocks`.
    /// A given `a`, `b` may stop there; a rebuilt table has them all.
    pub fn assert_covers(&self, m: usize, padding: &super::PaddingSpec) {
        let total = (1usize << m) / 8;
        assert_eq!(self.len_bytes(), total);
        let Bits::Given(p) = self else { return };
        let tail = padding.live_blocks < 1usize << (m - padding.k_log);
        let needed = if tail {
            (padding.live_blocks + 1) << (padding.k_log - 3)
        } else {
            total
        };
        assert!(
            !tail || padding.k_log >= super::K_SKIP + super::N_INNER,
            "blocks hold whole round-1 units"
        );
        assert!(
            p.a.len() == p.b.len() && (needed..=total).contains(&p.a.len()),
            "a, b cover the blocks read"
        );
    }

    /// The number of `CHUNKS`-byte rows, a power of two.
    pub fn positions<const CHUNKS: usize>(&self) -> usize {
        let n = self.len_bytes() / CHUNKS;
        assert!(
            n * CHUNKS == self.len_bytes() && n.is_power_of_two(),
            "a power-of-two number of whole rows"
        );
        n
    }

    /// A worker's scratch for windows of up to `window_bytes`; none when the tables are given.
    pub fn scratch(&self, window_bytes: usize) -> Option<Scratch> {
        let Bits::Rebuilt(r) = self else { return None };
        let slots = window_bytes.div_ceil(r.group_bytes()).next_power_of_two();
        let group_words = r.group_bytes() / 8;
        Some(Scratch {
            rows: r.new_rows(),
            tables: vec![0; 3 * slots * group_words],
            held: vec![usize::MAX; slots],
            group_words,
        })
    }

    /// Whether `range` lies in one group, which a window of a one-slot scratch needs.
    pub fn in_one_group(&self, range: Range<usize>) -> bool {
        match self {
            Bits::Given(_) => true,
            Bits::Rebuilt(r) => range.start / r.group_bytes() == (range.end - 1) / r.group_bytes(),
        }
    }

    /// The bytes `range` of the tables. A rebuilt window rebuilds the groups it covers that `scratch` does not hold.
    ///
    /// The window may reach past `range` to whole groups. Its groups must fit the scratch without wrapping.
    #[inline]
    pub fn window<'s>(&'s self, range: Range<usize>, scratch: &'s mut Option<Scratch>) -> Window<'s> {
        match (self, scratch) {
            (Bits::Given(p), _) => Window {
                tables: [p.a, p.b, p.c],
                first: 0,
            },
            (Bits::Rebuilt(r), Some(s)) => s.window(*r, range),
            (Bits::Rebuilt(_), None) => panic!("rebuilt bits need a scratch"),
        }
    }
}

impl Scratch {
    fn window<'s>(&'s mut self, r: &dyn Rebuild, range: Range<usize>) -> Window<'s> {
        let (group_bytes, slots) = (r.group_bytes(), self.held.len());
        debug_assert!(range.start < range.end && range.end <= r.len_bytes());
        let (g0, g1) = (range.start / group_bytes, (range.end - 1) / group_bytes);
        let s0 = g0 % slots;
        assert!(s0 + (g1 - g0) < slots, "the window's groups fit the scratch");
        let table_words = slots * self.group_words;
        for g in g0..=g1 {
            let slot = g % slots;
            if self.held[slot] == g {
                continue;
            }
            let mut out = self
                .tables
                .chunks_exact_mut(table_words)
                .map(|t| &mut t[slot * self.group_words..][..self.group_words]);
            let out = std::array::from_fn(|_| out.next().expect("three tables"));
            r.rebuild(g, &mut self.rows, out);
            self.held[slot] = g;
        }
        let from = s0 * group_bytes;
        let to = from + (g1 - g0 + 1) * group_bytes;
        let mut t = self
            .tables
            .chunks_exact(table_words)
            .map(|t| &packed_bytes(t)[from..to]);
        Window {
            tables: std::array::from_fn(|_| t.next().expect("three tables")),
            first: g0 * group_bytes,
        }
    }
}
