//! Runtime hint machinery shared by the interpreter and the compiler: the
//! resolved hint ops a [`super::Program`] carries ([`RHint`]), and the g-power
//! table + reverse index the hint interpreter grows on demand.

use primitives::field::{F64, mul_by_g};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Frame-relative offset operand (matches the compiler's `ir::Off`).
pub type Off = u32;

/// Cells a program can address: [`GPow`] covers no more.
pub(crate) const MAX_CELLS: u32 = 1 << 28;

/// The `g^k` table paired with a reverse index `g^k ↦ k`, both grown on demand
/// (recursion depth, and so the address range, is unbounded).
///
/// The interpreter needs both directions: a cell index becomes the address `g^k`,
/// and a pointer word read back out of memory must be inverted to the index it
/// addresses. Three things keep this cheap.
///
/// The reverse index holds nothing but the exponent, confirming a candidate slot
/// against the forward table, so a slot is four bytes where a `HashMap` bucket is
/// sixteen plus a control byte. Keys are field elements, hence effectively
/// uniform, which is all the placement asks for: one multiplicative mix, then
/// linear probing. The table stays at most half full to keep probing short.
///
/// It indexes only the exponents [`Self::note`] announces, or a lookup resolved:
/// a run addresses millions of cells but stores only the few thousand bases of
/// its frames and buffers, and one random write per addressable cell into a table
/// far too large for cache is what building the whole inverse costs.
///
/// A lookup that misses falls back on a coarse index of every [`STRIDE`]th
/// exponent: it divides the address by `g` up to `STRIDE - 1` times until a
/// multiple of `STRIDE` appears, then announces the result. An address that
/// arrives by a route the interpreter cannot label, a pointer into the middle of
/// a buffer, costs a few probes once and is a plain hit afterwards.
pub struct GPow {
    pow: Vec<F64>,
    /// The announced exponents, so widening rebuilds from them rather than
    /// scanning the old table.
    noted: Vec<u32>,
    /// `k + 1` at the slot `g^k` mixes to; 0 marks an empty slot. Power-of-two
    /// length.
    index: Vec<u32>,
    /// `k / STRIDE + 1` at the slot `g^k` mixes to, for the `coarsed` multiples
    /// `k` of `STRIDE`. Power-of-two length, at most half full.
    coarse: Vec<u32>,
    coarsed: usize,
    /// The last run's cell mapping, all zero again, for the next run to take.
    pub(crate) cells: Option<super::execute::Cells>,
}

/// Spacing of the coarse index, and so the most probes a missed lookup costs.
const STRIDE: usize = 16;

impl Default for GPow {
    fn default() -> Self {
        Self::new()
    }
}

impl GPow {
    /// Seeded with `g^0 = 1`, nothing announced.
    pub fn new() -> Self {
        Self {
            pow: vec![F64::ONE],
            noted: Vec::new(),
            index: vec![0; 1 << 13],
            coarse: Vec::new(),
            coarsed: 0,
            cells: None,
        }
    }

    /// Prepare powers and zeroed VM memory for `capacity` cells before proving.
    /// Larger executions can grow the tables and touch more pages on demand.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero, exceeds the VM address space, or its memory mapping fails.
    pub fn setup(capacity: usize) -> Self {
        assert!((1..=MAX_CELLS as usize).contains(&capacity), "invalid setup capacity");
        crate::init_prover();
        let mut g = Self::new();
        g.grow_to(capacity - 1);
        g.cells = Some(super::execute::Cells::prepare(capacity).expect("reserve VM memory for setup"));
        g
    }

    /// The number of exponents covered by the precomputed powers.
    pub fn capacity(&self) -> usize {
        self.pow.len()
    }

    /// Covering index `upto`, every exponent up to there announced, so that
    /// [`Self::lookup`] alone finds any of them.
    pub fn dense(upto: usize) -> Self {
        let mut g = Self::new();
        g.grow_to(upto);
        g.noted.reserve(upto + 1);
        for k in 0..=upto {
            g.note(k);
        }
        g
    }

    /// `g^k`. Panics past the grown range, which is the honest outcome: every
    /// index the interpreter reads has been covered by a [`Self::grow_to`].
    #[inline(always)]
    pub fn pow(&self, k: usize) -> F64 {
        self.pow[k]
    }

    /// The forward table, `g^0` to `g^{covered - 1}`.
    pub(crate) fn powers(&self) -> &[F64] {
        &self.pow
    }

    /// Extend the forward table to cover index `upto`.
    #[inline]
    pub fn grow_to(&mut self, upto: usize) {
        if upto >= self.pow.len() {
            self.extend(upto);
        }
    }

    #[cold]
    fn extend(&mut self, upto: usize) {
        assert!(upto < MAX_CELLS as usize, "address space overflow (program too large)");
        // ×g is ×x = `mul_by_g` (shift+fold), not a PMULL.
        let mut p = *self.pow.last().unwrap();
        let n = self.pow.len();
        self.pow.extend((n..=upto).map(|_| {
            p = mul_by_g(p);
            p
        }));
        let want = self.pow.len().div_ceil(STRIDE);
        if want * 2 > self.coarse.len() {
            self.coarse = vec![0; (want * 2).next_power_of_two().max(1 << 12)];
            self.coarsed = 0;
        }
        let mask = self.coarse.len() - 1;
        for j in self.coarsed..want {
            let mut i = slot_of(self.pow[j * STRIDE].0, mask);
            while self.coarse[i] != 0 {
                i = (i + 1) & mask;
            }
            self.coarse[i] = j as u32 + 1;
        }
        self.coarsed = want;
    }

    /// Announce that `g^k` may come back for inversion, because the interpreter
    /// is about to store it in memory as an address. Covers `k` first. Idempotent.
    pub fn note(&mut self, k: usize) {
        self.grow_to(k);
        if self.noted.len() * 2 >= self.index.len() {
            self.widen();
        }
        let mask = self.index.len() - 1;
        let mut i = slot_of(self.pow[k].0, mask);
        loop {
            let e = self.index[i];
            if e == 0 {
                break;
            }
            // Exponents are distinct (`g` has order `2^64 − 1`), so finding `k`
            // along its own probe chain means it is already announced.
            if e - 1 == k as u32 {
                return;
            }
            i = (i + 1) & mask;
        }
        self.index[i] = k as u32 + 1;
        self.noted.push(k as u32);
    }

    /// The discrete log of `x`, if it is a covered g-power.
    #[inline]
    pub fn log(&mut self, x: F64) -> Option<u32> {
        self.lookup(x).or_else(|| self.resolve(x))
    }

    /// `log` for an address that is not announced: divide by `g` down to a
    /// multiple of `STRIDE`, then announce the exponent.
    #[cold]
    fn resolve(&mut self, x: F64) -> Option<u32> {
        let mask = self.coarse.len() - 1;
        let mut y = x;
        for j in 0..STRIDE {
            let mut i = slot_of(y.0, mask);
            while self.coarse[i] != 0 {
                let k = (self.coarse[i] as usize - 1) * STRIDE;
                if self.pow[k] == y {
                    let k = k + j;
                    return (k < self.pow.len()).then(|| {
                        self.note(k);
                        k as u32
                    });
                }
                i = (i + 1) & mask;
            }
            y = div_by_g(y);
        }
        None
    }

    /// The exponent of `x` if announced.
    #[inline]
    pub fn lookup(&self, x: F64) -> Option<u32> {
        let mask = self.index.len() - 1;
        let mut i = slot_of(x.0, mask);
        loop {
            let e = self.index[i];
            if e == 0 {
                return None;
            }
            if self.pow[(e - 1) as usize] == x {
                return Some(e - 1);
            }
            i = (i + 1) & mask;
        }
    }

    /// Rebuild the index at a size that holds `noted` at most half full.
    fn widen(&mut self) {
        let cap = (self.noted.len() * 2 + 1).next_power_of_two().max(1 << 13);
        let mut index = vec![0u32; cap];
        let mask = cap - 1;
        for &k in &self.noted {
            let mut i = slot_of(self.pow[k as usize].0, mask);
            while index[i] != 0 {
                i = (i + 1) & mask;
            }
            index[i] = k + 1;
        }
        self.index = index;
    }
}

const LOG_BABY: u32 = 17;

fn dlog_table() -> &'static (GPow, F64) {
    static TABLE: OnceLock<(GPow, F64)> = OnceLock::new();
    TABLE.get_or_init(|| {
        let baby = GPow::dense((1usize << LOG_BABY) - 1);
        let giant = mul_by_g(baby.pow((1usize << LOG_BABY) - 1)).inv();
        (baby, giant)
    })
}

pub(crate) fn setup() {
    dlog_table();
}

/// Bounded discrete log for exponent hints; the guest checks the hinted bits.
pub(super) fn bounded_dlog(x: F64, nbits: u32) -> Option<u128> {
    let (baby, giant) = dlog_table();
    let mut y = x;
    let max_giant = if nbits > LOG_BABY {
        1u64 << (nbits - LOG_BABY)
    } else {
        1
    };
    for a in 0..max_giant {
        if let Some(j) = baby.lookup(y) {
            return Some((a as u128) << LOG_BABY | j as u128);
        }
        y *= *giant;
    }
    None
}

/// The inverse of [`mul_by_g`]: bit 0 of its output says whether it folded.
#[inline(always)]
fn div_by_g(a: F64) -> F64 {
    F64(if a.0 & 1 == 1 { (a.0 ^ 0x1B) >> 1 | 1 << 63 } else { a.0 >> 1 })
}

/// Bits 32..64 of the mix are the well-distributed ones; the mask keeps as many
/// of them as the table is wide.
#[inline(always)]
fn slot_of(key: u64, mask: usize) -> usize {
    (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize & mask
}

/// Where a computed-advice bit buffer lives. Frame cells are ordinary memory, so
/// a run of them serves as well as a heap region and costs no `DEREF` to index at
/// a compile-time offset; the heap form stays for a buffer indexed at run time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitsDest {
    /// Frame cells `fp+base+k`.
    Stack(Off),
    /// Heap cells `m[fp+ptr]·g^k`, the pointer read at run time.
    Heap(Off),
}

/// The hints of a program flattened into a dense per-pc index, built once at assembly.
///
/// A map keyed by pc would be probed each step for what is almost always a miss, so
/// `at` holds, per pc, 0 for no hints or `1 + i` for list `i`. List `i` is
/// `hints[starts[i]..starts[i + 1]]`, and `streams` holds per hint the index into
/// `named` of its witness stream (`u32::MAX` for a hint that has none).
#[derive(Clone)]
pub(crate) struct HintPlan {
    pub(crate) at: Vec<u32>,
    starts: Vec<u32>,
    hints: Vec<RHint>,
    streams: Vec<u32>,
    /// Per distinct witness stream, a hint that names it, so a run resolves each name once.
    named: Vec<u32>,
}

impl HintPlan {
    pub(crate) fn new(hints: HashMap<u32, Vec<RHint>>, code_len: usize) -> Self {
        let mut plan = Self {
            at: vec![0; code_len],
            starts: vec![0],
            hints: Vec::new(),
            streams: Vec::new(),
            named: Vec::new(),
        };
        for (pc, hs) in hints {
            for h in hs {
                let id = stream_name(&h).map_or(u32::MAX, |name| {
                    let same = |&i: &u32| stream_name(&plan.hints[i as usize]) == Some(name);
                    plan.named.iter().position(same).unwrap_or_else(|| {
                        plan.named.push(plan.hints.len() as u32);
                        plan.named.len() - 1
                    }) as u32
                });
                plan.hints.push(h);
                plan.streams.push(id);
            }
            plan.starts.push(plan.hints.len() as u32);
            plan.at[pc as usize] = plan.starts.len() as u32 - 1;
        }
        plan
    }

    /// The hints at list `at - 1` (`at` from [`Self::at`]), each with its stream index.
    pub(crate) fn list(&self, at: u32) -> impl Iterator<Item = (&RHint, u32)> {
        let r = self.starts[at as usize - 1] as usize..self.starts[at as usize] as usize;
        self.hints[r.clone()].iter().zip(self.streams[r].iter().copied())
    }

    /// The names of the distinct witness streams, in stream index order.
    pub(crate) fn names(&self) -> impl Iterator<Item = &str> {
        self.named.iter().filter_map(|&i| stream_name(&self.hints[i as usize]))
    }
}

/// The witness stream a hint pops, if any.
fn stream_name(h: &RHint) -> Option<&str> {
    match h {
        RHint::WitnessStack { name, .. } | RHint::WitnessHeap { name, .. } => Some(name),
        _ => None,
    }
}

/// A hint resolved to concrete offsets/sizes, keyed by global program counter.
#[derive(Clone, Debug)]
pub enum RHint {
    /// Announce a pointer into a reserved run of frames.
    FrameAddress { offset: Off },
    /// Reserve `(log_g(m[end] * start_inverse) + 1)` consecutive frames.
    AllocFrames {
        ptr: Off,
        size: u32,
        end: Off,
        start_inverse: F64,
    },
    /// Allocate a fresh region of `size` cells and write `g^{base}` to the cell.
    Alloc { ptr: Off, size: u32 },
    /// `Alloc` with the cell count read at runtime as the g-power exponent of
    /// `m[fp+size]`.
    AllocDyn { ptr: Off, size: Off },
    /// Pop stream `name`'s next entry (`len` values) into frame cells `fp+base+k`.
    WitnessStack { name: String, base: Off, len: u32 },
    /// Pop stream `name`'s next entry (`len` values) into heap cells `m[fp+ptr]·g^{lo+k}`.
    WitnessHeap { name: String, ptr: Off, lo: u32, len: u32 },
    /// Write `g^max(log2_ceil(value), floor)` into `fp+dst`, where `value` is the
    /// integer reconstructed from the `nbits` bits at `bits`.
    Log2Ceil {
        bits: BitsDest,
        dst: Off,
        nbits: u32,
        floor: u32,
    },
    /// Write the `nbits` bits of `m[fp+value]` from bit `start` on into `bits`.
    BitDecompose {
        value: Off,
        bits: BitsDest,
        nbits: u32,
        start: u32,
    },
    /// Write the `nbits` bits of `n`, where `m[fp+value] = g^n` (a bounded
    /// discrete log at witness generation), into `bits`.
    BitDecomposeExp { value: Off, bits: BitsDest, nbits: u32 },
    /// Write the first `len` K-coordinate limbs of `m[fp+value]` to
    /// `m[fp+base..]`. Computed advice; callers constrain the result.
    FieldLimbs { value: Off, base: Off, len: u32 },
    /// Write `m[fp+value]⁻¹` to `m[fp+dst]`, or `0` when the value is zero.
    /// Untrusted: `assert a != b` multiplies the two back together and asserts
    /// `1`, which a zero value cannot satisfy (`FnLower::lower_assert_ne`).
    Inverse { value: Off, dst: Off },
    /// Prover-side debug print (`print(...)` in the zkDSL): display the value
    /// of `m[fp+cell]` at this program point. Witness generation only.
    Print { label: String, cell: Off },
}
