import LeanVM.Protocol.Verifier
import LeanVM.Protocol.WithHashBridge

/-!
# The compiled verifier: the Fiat-Shamir sponge against the pinned one's, and garbage rejected

`verify` is the compiled reading of `leanVmVerifierProgram` with Keccak-256 and Keccak-f[1600]. The
honest proofs are the fixtures `tests/fixtures/verifier/*` (accepted by the compiled verifier
`leanvm-verify`, `scripts/fuzz-verifier.py`); this file checks the compiled reading where the
pinned verifier can be run without a proof, on the scalar stream `stream rate n`: the eight announced
sizes (memory `2^16`, the row counts `4, 2, 2, 4, 4, 8`, the rate exponent), two canonical root
halves, then `n` scalars `fe(10 + k)`.

* The seed and the sponge: the first two steps of the program (`readAnnouncedSizes`, `commitStack`)
  and five challenges, in the compiled reading, give the challenges and the sponge state the
  verifier's `Transcript` gives on the same stream after `next_scalars(8)`, `Digest::from_halves`
  of `next_scalars(2)` and `samples(5)` (the verifier at leanVM
  this branch's twelve-round Rust verifier). This binds the IV's
  preimage (the version tag, then the bytecode), the seed permutation of the IV and the public
  input, the duplex absorbs with their domain tags, the squeezes, the order of the reads, and the
  stack root's shape.
* A root half with a nonzero top limb is rejected, as `Digest::from_halves` rejects it.
* Garbage proofs run to a rejection: the empty proof and the stream cut after the root (the
  verifier: `proof stream exhausted`), a hundred scalars after it (the verifier: `GKR layer 17:
  children do not match the sumcheck`), and a rate exponent of `5` (`invalid PCS inverse rate`).
-/

namespace LeanVMTests.Protocol.Verifier.Compiled

open LeanVM.Parameters LeanVM.Semantics LeanVM.Protocol LeanVM.Protocol.Verifier
open LeanVM.Protocol.Compile LeanVM.Protocol.KeccakHash

/-- The driver's 64-bit word generator. -/
def fw (k lane : ℕ) : ℕ :=
  ((k + 1) * 0x9E3779B97F4A7C15 + lane * 0xD1B54A32D192ED03 + 0x632BE59BD9B4E019) % 2 ^ 64

/-- An `E` element from its three limbs. -/
def e (c0 c1 c2 : ℕ) : E := E.ofLimbs (K.ofBits c0) (K.ofBits c1) (K.ofBits c2)

/-- The driver's `fe(k)`. -/
def fe (i : ℕ) : E := e (fw i 0) (fw i 1) (fw i 2)

/-- The bytecode: 32 words. -/
def sampleBytecode : List K := (List.range 32).map fun i ↦ K.ofBits (fw i 9)

/-- The public input: the 32 bytes `fw(i, 10) mod 256`. -/
def samplePublicInput : Digest :=
  digestOfBytes ((List.range 32).map fun i ↦ (fw i 10 % 256).toUInt8)

/-- The sample row counts: `xor, mul, set, deref, jump, keccak`. -/
def sampleRows : List ℕ := [4, 2, 2, 4, 4, 8]

/-- The sizes, the root halves (`top` the root's second half's top limb), then `n` scalars. -/
def stream (rate n : ℕ) (top : ℕ := 0) : List E :=
  [eOfNat 16] ++ sampleRows.map eOfNat ++ [eOfNat rate] ++
    [e (fw 8 0) (fw 8 1) 0, e (fw 9 0) (fw 9 1) top] ++ (List.range n).map fun k ↦ fe (10 + k)

/-- The first two steps of the program, then five challenges. -/
def prefixProgram : VerifierProgram (List E) := do
  let (layout, rate) ← readAnnouncedSizes sampleBytecode
  commitStack layout.stackLog rate
  VerifierProgram.sampleChallenges 5

/-- The compiled reading of `prefixProgram` on a stream, from the deployed seed. -/
def prefixRun (s : List E) : Option (List E × VerifierProgramState) :=
  Id.run (Compile.run (m := Id) keccakF1600 prefixProgram
    ⟨⟨seed sampleBytecode samplePublicInput, []⟩, s, [], []⟩)

/-! ## The sponge agrees with the pinned verifier's -/

#guard ((prefixRun (stream 4 0)).map (·.1)) =
    some [(e 0xb2d5680ab5cca42a 0x6cba1af4bd3536bd 0xebd0f21a934c8aaf), (e 0x74fed2383b1a8469
     0x8aa5fabac8933325 0x10db3a5c7e2f1376), (e 0xf70bcdc6dd99b1d6 0x63d0dede4d0a81dc 0x4524bf8af25e58d3), (e
     0x47f98b9d54fbcc13 0x42cf3ee6bb7e90c2 0x242953ae207d5a8d), (e 0xea09c3cc94d1d1db 0xce86fd975a0ef70c
     0xdaaa5d7876b9942d)]

#guard ((prefixRun (stream 4 0)).map fun r ↦ (r.2.sponge.state.toList, r.2.sponge.pending.length)) =
    some ([0xea09c3cc94d1d1db, 0xce86fd975a0ef70c, 0xdaaa5d7876b9942d, 0x9a5ad54814cb776, 0x81fb709c6988d63b,
     0xa44772348e9fe4a8, 0x74d6eeeedc8743c4, 0x95c02e6c42ce0939, 0xe8c1ca0b05c2898d, 0xc37fbb1a45b63aea,
     0xceffab1a8fed6e59, 0x11bb007a46450d76, 0x9e3ba86e9841189d, 0xc9138e77a4cab1f4, 0x7d945594647dbd28,
     0xa59f057531d6e005, 0xab4f87fbd8af2f67, 0x4ca220ff1aa77f99, 0x8578d2ef7266f1b9, 0x3442da09c96f6155,
     0x7983d8785702dba7, 0xc52cee168f2566a3, 0x1b0c609c21463c03, 0xbcaaa20542bea95b, 0xf09a3a59c20ff8b7], 0)

/- The stack root is commitment `0`, of `2^(19 - 6 + 4)` leaves of `2^6` words. -/
#guard ((prefixRun (stream 4 0)).map fun r ↦ r.2.roots.map fun c ↦ (c.2.1, c.2.2)) =
    some [(17, 64)]

/- A root half with a nonzero top limb. -/
#guard (prefixRun (stream 4 0 1)).isNone

/-! ## Garbage is rejected -/

#guard verify sampleBytecode samplePublicInput ⟨[], []⟩ = false
#guard verify sampleBytecode samplePublicInput ⟨stream 4 0, []⟩ = false
#guard verify sampleBytecode samplePublicInput ⟨stream 4 100, []⟩ = false
#guard verify sampleBytecode samplePublicInput ⟨stream 5 100, []⟩ = false

/-- The deployed verifier is the one over a permutation, at `Id` and Keccak-f[1600]. -/
example (π : Proof) :
    Id.run (verifyWith (m := Id) keccakF1600 sampleBytecode samplePublicInput π) =
      verify sampleBytecode samplePublicInput π := by
  rw [verifyWith_eq_withHash, keccak256With_keccakF1600]
  rfl

end LeanVMTests.Protocol.Verifier.Compiled
