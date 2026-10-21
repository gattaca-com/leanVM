//! The precompile and the hints: the two places a value arrives without an
//! instruction computing it.
//!
//! `haraka` is a STATEMENT, not an expression: it writes its digest into a
//! two-cell run the caller names, so a pre-written destination checks the digest
//! instead of computing it, by the same write-once rule as any store.
//!
//! A hint writes values the prover chose and the circuit did not, so **the
//! program must constrain them**. A hint names its destination's PHYSICAL cells,
//! and since every store emits there is nothing for the compiler to prepare: a
//! later `s[k] = <checked value>` is a second write of that cell, which is the
//! assertion that pins the hinted value.

use super::*;

impl FnLower<'_> {
    /// `haraka(a, b, out)`: the digest of the two 256-bit operands lands in the
    /// existing 2-cell run `out` (write-once: if `out` was already written, this
    /// asserts the digest equals it). A heap `out` slice takes the digest via a
    /// fresh stack pair and two `DEREF`s after the hash, the store direction
    /// being the same instruction as the load (write-once fills the unset side).
    fn lower_haraka(&mut self, args: &[Expr]) {
        if args.len() != 3 {
            self.fail("haraka takes three arguments: (a, b, out)")
        };
        let a = self.haraka_input(&args[0]);
        let b = self.haraka_input(&args[1]);
        let (c, heap_out) = match self.haraka_operand(&args[2]) {
            CellRun::Stack { base, .. } => (base, None),
            CellRun::Heap { ptr, lo, .. } => (self.alloc_stack(2), Some((ptr, lo))),
        };
        // Each operand is two 128-bit chunk cells; the opcode addresses the four
        // input cells independently (`haraka_input` forwards the real chunk sources
        // where it can). The digest occupies the two consecutive output cells `c, g·c`.
        self.emit(LOp::Haraka {
            ins: [a[0], a[1], b[0], b[1]],
            c,
        });
        if let Some((ptr, lo)) = heap_out {
            for k in 0..2 {
                self.deref(ptr, lo + k, c + k, DerefMode::Cell);
            }
        }
    }

    /// The statement-position builtins, `true` if `f` was one of them (else the
    /// caller emits an ordinary call). The `hint_*` ones queue prover-side
    /// advice, re-checked in-circuit by their caller: `hint_decompose_bits`
    /// writes a value's bits into a buffer, `hint_decompose_bits_exponent` the
    /// bits of `n` where the value is `g^n` (a bounded dlog at witness
    /// generation), `hint_f192_limbs` a value's coordinate limbs.
    pub(super) fn lower_builtin(&mut self, f: &str, args: &[Expr]) -> bool {
        match f {
            "hint_decompose_bits" | "hint_decompose_bits_exponent" => {
                let plain = f == "hint_decompose_bits";
                if args.len() != 3 && !(plain && args.len() == 4) {
                    self.fail(format!(
                        "{f} takes `(bits, value, nbits{})`, got {} arguments",
                        if plain { "[, start]" } else { "" },
                        args.len()
                    ))
                };
                let nbits = self.const_index(&args[2]);
                let bits = self.bits_dest(&args[0], nbits, f);
                let value = self.expr(&args[1]);
                self.pending.push(Hint::Resolved(if plain {
                    let start = args.get(3).map_or(0, |start| self.const_index(start));
                    let end = u64::from(start) + u64::from(nbits);
                    if end > 192 {
                        self.fail(format!("{f} reads bits {start}..{end} of a 192-bit word"))
                    };
                    RHint::BitDecompose {
                        value,
                        bits,
                        nbits,
                        start,
                    }
                } else {
                    RHint::BitDecomposeExp { value, bits, nbits }
                }));
            }
            "haraka" => self.lower_haraka(args),
            "assert_in_k" => {
                if args.len() != 2 {
                    self.fail("assert_in_k(a, b) takes two scalar cells")
                };
                let a = self.expr(&args[0]);
                let b = self.expr(&args[1]);
                let zero = self.zero();
                self.emit(LOp::Jump { oc: zero, od: a, of: b });
            }
            "hint_f192_limbs" => {
                if args.len() != 2 {
                    self.fail(format!(
                        "hint_f192_limbs takes two arguments, `(dest, value)`, got {}",
                        args.len()
                    ))
                };
                let (base, len) = self.stack_of(&args[0]).unwrap_or_else(|| {
                    self.fail(format!(
                        "hint_f192_limbs writes 1..=3 frame cells, so its destination must be a \
                             StackBuf, got `{:?}`",
                        args[0]
                    ))
                });
                if !((1..=3).contains(&len)) {
                    self.fail("hint_f192_limbs destination must have 1..=3 cells")
                };
                let value = self.expr(&args[1]);
                // Names the physical cells, as the two consumers above do: whatever
                // the program stores into them afterwards is a second write, and so
                // the assertion that pins these limbs.
                self.pending
                    .push(Hint::Resolved(RHint::FieldLimbs { value, base, len }));
            }
            _ => return false,
        }
        true
    }

    /// Resolve a `haraka` operand: a [`Self::cell_run`] pinned to exactly 2
    /// cells, a 256-bit value being two 128-bit cells. Stack operands are used
    /// in place; heap operands must be bridged through the stack, since
    /// `Haraka` addresses only frame cells (see [`Self::haraka_input`]).
    fn haraka_operand(&mut self, e: &Expr) -> CellRun {
        let run = self.cell_run(e);
        if run.cells() != 2 {
            self.fail("a haraka operand must span exactly 2 cells (two 128-bit words); slice a larger buffer: `buf[lo:lo + 2]`")
        };
        run
    }

    /// A `haraka` *input* operand as its two independently-addressed 128-bit
    /// chunk bases (each chunk is ONE 128-bit cell): stack runs in place; a heap
    /// slice is pulled into a fresh stack pair first, one `DEREF` per cell
    /// (`m[ptr·g^{lo+k}] == m[fp+t+k]`, the `β` immediate doing the pointer
    /// offset). The heap cells must already be written.
    ///
    /// A LIST LITERAL names its two words directly and allocates nothing. The
    /// opcode addresses its four input chunks independently, so an operand
    /// assembled out of values living elsewhere never has to be gathered into a
    /// consecutive run: `haraka([a, b], …)` is the spelling that says so.
    pub(super) fn haraka_input(&mut self, e: &Expr) -> [Off; 2] {
        if let Expr::ListLit(words) = e {
            if words.len() != 2 {
                self.fail(format!(
                    "a haraka operand written as a list needs exactly 2 words, got {}",
                    words.len()
                ))
            };
            return [self.expr(&words[0]), self.expr(&words[1])];
        }
        match self.haraka_operand(e) {
            CellRun::Stack { base, .. } => [base, base + 1],
            CellRun::Heap { ptr, lo, .. } => {
                let t = self.alloc_stack(2);
                for k in 0..2 {
                    self.deref(ptr, lo + k, t + k, DerefMode::Cell);
                }
                [t, t + 1]
            }
        }
    }

    /// A computed-advice bit buffer's destination ([`BitsDest`]). Not
    /// [`Self::cell_run`]: these builtins take a bare `HeapBuf` and carry the
    /// length in `nbits`, where a cell run would demand a slice.
    pub(super) fn bits_dest(&mut self, e: &Expr, nbits: u32, what: &str) -> BitsDest {
        match self.stack_of(e) {
            Some((base, len)) => {
                if len < nbits {
                    self.fail(format!(
                        "{what} needs {nbits} cells, its StackBuf destination has {len}"
                    ))
                };
                // The hint names the physical cells, as `hint_f192_limbs` does.
                BitsDest::Stack(base)
            }
            None => {
                // Bounds-checked like the `StackBuf` arm above, and like every
                // other heap consumer. Without this a `HeapBuf` destination wrote
                // `nbits` cells with nothing checking the buffer held them, so the
                // bits ran on into the next buffer while the same call with a
                // `StackBuf` destination was rejected.
                self.check_heap_bound(e, 0, u128::from(nbits));
                BitsDest::Heap(self.expr(e))
            }
        }
    }

    /// `hint_witness(dest, "name")`: resolve `dest` to a run of cells and
    /// queue the witness-fill hint (no instructions: the values are written
    /// by the runner before the next instruction executes, unconstrained).
    pub(super) fn lower_hint_witness(&mut self, dest: &Expr, name: &str) {
        let name = name.to_string();
        let hint = match self.cell_run(dest) {
            CellRun::Stack { base, len } => RHint::WitnessStack { name, base, len },
            CellRun::Heap { ptr, lo, len } => RHint::WitnessHeap { name, ptr, lo, len },
        };
        self.pending.push(Hint::Resolved(hint));
    }
}
