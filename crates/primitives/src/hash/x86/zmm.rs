//! Two 16-lane compressions in one hand-scheduled AVX-512 kernel.
//!
//! On Zen 5, a 512-bit add, xor or rotate has a 2-cycle latency.
//!
//! One 16-lane group then waits on its own G chains, and most vector pipes idle.
//!
//! Two groups double the independent work, and fill the whole register file:
//!
//! ```text
//!     zmm0  .. zmm15    group A's state
//!     zmm16 .. zmm31    group B's state
//! ```
//!
//! Nothing else fits, so every update is in place and message words stay in memory.
//!
//! The compiler spills at this pressure, hence assembly.
//!
//! # Schedule
//!
//! One G is a chain of 12 dependent ops, and a group runs four G's per half round.
//!
//! Group B runs five slots behind group A, so the two groups are never on the same step.
//!
//! A slot is the steps that issue together, and `period!` lists the 14 slots of one half round.
//!
//! The second message add sits late, off the dependency chain.
//!
//! Lockstep, with both groups on the same step, measured slower.
//!
//! Rotating by 16 or 8 with `vpshufb` instead of `vprord` also measured slower.

use crate::hash::IV;

/// `op zmmD, zmmD, zmmS`.
macro_rules! rr {
    ($op:literal, $d:literal, $s:literal) => {
        concat!($op, " zmm", $d, ", zmm", $d, ", zmm", $s, "\n")
    };
}

/// `zmmD += m[w]`, with the message word as a memory operand.
macro_rules! addm {
    ($d:literal, $m:literal, $w:literal) => {
        concat!("vpaddd zmm", $d, ", zmm", $d, ", zmmword ptr [", $m, " + 64*", $w, "]\n")
    };
}

/// `zmmD >>>= n`.
macro_rules! ror {
    ($d:literal, $n:literal) => {
        concat!("vprord zmm", $d, ", zmm", $d, ", ", $n, "\n")
    };
}

/// One step of the G function, for each `(a, b, c, d, m, x, y)` of one half round and group.
///
/// The tuple holds the four state rows, the message base and the two message words.
#[rustfmt::skip]
macro_rules! step {
    (mx  $(($a:literal, $b:literal, $c:literal, $d:literal, $m:literal, $x:literal, $y:literal))*) => { concat!($(addm!($a, $m, $x),)*) };
    (my  $(($a:literal, $b:literal, $c:literal, $d:literal, $m:literal, $x:literal, $y:literal))*) => { concat!($(addm!($a, $m, $y),)*) };
    (ab  $(($a:literal, $b:literal, $c:literal, $d:literal, $m:literal, $x:literal, $y:literal))*) => { concat!($(rr!("vpaddd", $a, $b),)*) };
    (cd  $(($a:literal, $b:literal, $c:literal, $d:literal, $m:literal, $x:literal, $y:literal))*) => { concat!($(rr!("vpaddd", $c, $d),)*) };
    (xda $(($a:literal, $b:literal, $c:literal, $d:literal, $m:literal, $x:literal, $y:literal))*) => { concat!($(rr!("vpxord", $d, $a),)*) };
    (xbc $(($a:literal, $b:literal, $c:literal, $d:literal, $m:literal, $x:literal, $y:literal))*) => { concat!($(rr!("vpxord", $b, $c),)*) };
    (d16 $(($a:literal, $b:literal, $c:literal, $d:literal, $m:literal, $x:literal, $y:literal))*) => { concat!($(ror!($d, 16),)*) };
    (d8  $(($a:literal, $b:literal, $c:literal, $d:literal, $m:literal, $x:literal, $y:literal))*) => { concat!($(ror!($d, 8),)*) };
    (b12 $(($a:literal, $b:literal, $c:literal, $d:literal, $m:literal, $x:literal, $y:literal))*) => { concat!($(ror!($b, 12),)*) };
    (b7  $(($a:literal, $b:literal, $c:literal, $d:literal, $m:literal, $x:literal, $y:literal))*) => { concat!($(ror!($b, 7),)*) };
}

/// One step of group `$grp` (`A` or `B`) for the half round `$spec`.
///
/// A spec is `[cols s0 .. s7]` or `[diag s8 .. s15]`: the half round and its message schedule.
///
/// `[none]` is no half round, so the step is empty.
#[rustfmt::skip]
macro_rules! at {
    ($step:ident, $grp:ident, [none]) => { "" };
    ($step:ident, A, [cols $s0:literal $s1:literal $s2:literal $s3:literal $s4:literal $s5:literal $s6:literal $s7:literal]) => {
        step!($step (0, 4, 8, 12, "{ma}", $s0, $s1) (1, 5, 9, 13, "{ma}", $s2, $s3) (2, 6, 10, 14, "{ma}", $s4, $s5) (3, 7, 11, 15, "{ma}", $s6, $s7))
    };
    ($step:ident, B, [cols $s0:literal $s1:literal $s2:literal $s3:literal $s4:literal $s5:literal $s6:literal $s7:literal]) => {
        step!($step (16, 20, 24, 28, "{mb}", $s0, $s1) (17, 21, 25, 29, "{mb}", $s2, $s3) (18, 22, 26, 30, "{mb}", $s4, $s5) (19, 23, 27, 31, "{mb}", $s6, $s7))
    };
    ($step:ident, A, [diag $s0:literal $s1:literal $s2:literal $s3:literal $s4:literal $s5:literal $s6:literal $s7:literal]) => {
        step!($step (0, 5, 10, 15, "{ma}", $s0, $s1) (1, 6, 11, 12, "{ma}", $s2, $s3) (2, 7, 8, 13, "{ma}", $s4, $s5) (3, 4, 9, 14, "{ma}", $s6, $s7))
    };
    ($step:ident, B, [diag $s0:literal $s1:literal $s2:literal $s3:literal $s4:literal $s5:literal $s6:literal $s7:literal]) => {
        step!($step (16, 21, 26, 31, "{mb}", $s0, $s1) (17, 22, 27, 28, "{mb}", $s2, $s3) (18, 23, 24, 29, "{mb}", $s4, $s5) (19, 20, 25, 30, "{mb}", $s6, $s7))
    };
}

/// One period of 14 slots: group A on half round `$cur`, group B five slots behind.
///
/// Group B's steps from the first slots of the period still belong to the half round `$prev`.
///
/// Within a slot, group A goes first.
#[rustfmt::skip]
macro_rules! period {
    ($cur:tt, $prev:tt) => {
        concat!(
            // slot 0
            at!(mx, A, $cur),
            // slot 1
            at!(ab, A, $cur), at!(xda, B, $prev), at!(d8, B, $prev),
            // slots 2 and 3
            at!(cd, B, $prev), at!(xbc, B, $prev),
            // slot 4
            at!(xda, A, $cur), at!(d16, A, $cur), at!(b7, B, $prev),
            // slot 5
            at!(cd, A, $cur), at!(mx, B, $cur),
            // slot 6
            at!(ab, B, $cur),
            // slot 7
            at!(my, A, $cur),
            // slot 8
            at!(xbc, A, $cur), at!(b12, A, $cur), at!(ab, A, $cur),
            // slot 9
            at!(xda, B, $cur), at!(d16, B, $cur),
            // slot 10
            at!(xda, A, $cur), at!(d8, A, $cur), at!(cd, B, $cur),
            // slot 11
            at!(cd, A, $cur),
            // slot 12
            at!(xbc, A, $cur), at!(my, B, $cur),
            // slot 13
            at!(b7, A, $cur), at!(xbc, B, $cur), at!(b12, B, $cur), at!(ab, B, $cur),
        )
    };
}

/// The periods of half rounds `$cur, ..`, then group B's last slots.
macro_rules! chain {
    ($prev:tt;) => { period!([none], $prev) };
    ($prev:tt; $cur:tt $($rest:tt)*) => { concat!(period!($cur, $prev), chain!($cur; $($rest)*)) };
}

/// Load `v[i] = h[i]` and broadcast `v[8 + i]` from `src`, for both groups.
macro_rules! init {
    ($($i:literal $j:literal $vi:literal $vj:literal $src:literal),*) => {
        concat!($(
            "vmovdqu64 zmm", $i, ", zmmword ptr [{h} + 64*", $i, "]\n",
            "vmovdqu64 zmm", $j, ", zmmword ptr [{h} + 512 + 64*", $i, "]\n",
            "vpbroadcastd zmm", $vi, ", ", $src, "\n",
            "vmovdqa64 zmm", $vj, ", zmm", $vi, "\n",
        )*)
    };
}

/// `h[i] ^= v[i] ^ v[i + 8]` for both groups.
///
/// One `vpternlogd` each: `0x96` is the three-way xor.
macro_rules! feed_forward {
    ($($i:literal $j:literal $vi:literal $vj:literal),*) => {
        concat!($(
            "vpternlogd zmm", $i, ", zmm", $vi, ", zmmword ptr [{h} + 64*", $i, "], 0x96\n",
            "vmovdqu64 zmmword ptr [{h} + 64*", $i, "], zmm", $i, "\n",
            "vpternlogd zmm", $j, ", zmm", $vj, ", zmmword ptr [{h} + 512 + 64*", $i, "], 0x96\n",
            "vmovdqu64 zmmword ptr [{h} + 512 + 64*", $i, "], zmm", $j, "\n",
        )*)
    };
}

/// Compress two transposed 16-lane blocks at byte counter `t`.
///
/// Both chaining values are updated in place:
///
/// ```text
///     byte 64 * i          group A's word i
///     byte 512 + 64 * i    group B's word i
/// ```
///
/// The counter and final flag arrive in registers, so no per-block row round-trips through memory.
///
/// # Safety
///
/// - `h` must be valid for reads and writes of `2 * 8 * 16` `u32`.
/// - `ma` and `mb` must each be valid for reads of `16 * 16` `u32`.
#[inline(always)]
pub(super) unsafe fn compress_x2(h: *mut u32, ma: *const u32, mb: *const u32, t: u64, last: bool) {
    let iv: &'static [u32; 8] = &IV;
    // SAFETY: the caller guarantees the three buffers.
    // The kernel touches no stack and no memory other than theirs and `IV`.
    unsafe {
        core::arch::asm!(
            // v[0..8] = h, and v[8..16] = the IV row with the counter and flag folded in.
            init!(
                0 16 8 24 "dword ptr [{iv}]",
                1 17 9 25 "dword ptr [{iv} + 4]",
                2 18 10 26 "dword ptr [{iv} + 8]",
                3 19 11 27 "dword ptr [{iv} + 12]",
                4 20 12 28 "{t0:e}",
                5 21 13 29 "{t1:e}",
                6 22 14 30 "{f:e}",
                7 23 15 31 "dword ptr [{iv} + 28]"
            ),
            // The ten rounds (RFC 7693 message schedule), as 20 half rounds.
            chain!([none];
                [cols 0 1 2 3 4 5 6 7] [diag 8 9 10 11 12 13 14 15]
                [cols 14 10 4 8 9 15 13 6] [diag 1 12 0 2 11 7 5 3]
                [cols 11 8 12 0 5 2 15 13] [diag 10 14 3 6 7 1 9 4]
                [cols 7 9 3 1 13 12 11 14] [diag 2 6 5 10 4 0 15 8]
                [cols 9 0 5 7 2 4 10 15] [diag 14 1 11 12 6 8 3 13]
                [cols 2 12 6 10 0 11 8 3] [diag 4 13 7 5 15 14 1 9]
                [cols 12 5 1 15 14 13 4 10] [diag 0 7 6 3 9 2 8 11]
                [cols 13 11 7 14 12 1 3 9] [diag 5 0 15 4 8 6 2 10]
                [cols 6 15 14 9 11 3 0 8] [diag 12 2 13 7 1 4 10 5]
                [cols 10 2 8 4 7 6 1 5] [diag 15 11 9 14 3 12 13 0]
            ),
            // h[i] ^= v[i] ^ v[i + 8], written back in place.
            feed_forward!(
                0 16 8 24, 1 17 9 25, 2 18 10 26, 3 19 11 27, 4 20 12 28, 5 21 13 29, 6 22 14 30, 7 23 15 31
            ),
            h = in(reg) h,
            ma = in(reg) ma,
            mb = in(reg) mb,
            iv = in(reg) iv.as_ptr(),
            t0 = in(reg) iv[4] ^ t as u32,
            t1 = in(reg) iv[5] ^ (t >> 32) as u32,
            f = in(reg) if last { !iv[6] } else { iv[6] },
            out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _,
            out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
            out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
            out("zmm12") _, out("zmm13") _, out("zmm14") _, out("zmm15") _,
            out("zmm16") _, out("zmm17") _, out("zmm18") _, out("zmm19") _,
            out("zmm20") _, out("zmm21") _, out("zmm22") _, out("zmm23") _,
            out("zmm24") _, out("zmm25") _, out("zmm26") _, out("zmm27") _,
            out("zmm28") _, out("zmm29") _, out("zmm30") _, out("zmm31") _,
            options(nostack),
        );
    }
}
