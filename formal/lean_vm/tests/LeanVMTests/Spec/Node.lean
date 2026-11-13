import Statement.LeanVmGuest
import LeanXmss.Scheme
import Statement.LeanSphincs
import Statement.Fast.Schemes
import LeanDa.Scheme
import Statement.Fast.Keccak
import Statement.Fast.QuadExt
import Statement.Fast.FieldWords
import LeanVM.Facts.Verifier
import LeanVM.Facts.NodeWith
import LeanVM.Facts.DaWith
import LeanVMTests.Spec.NodeVectors

/-!
# The deferring verifier and the node's statement: inhabitants, known answers and mutations

* `claimedBytecodeValue` answers with the claimed value at the claimed point and rejects
  elsewhere, and the deferring verifier's seed is the deployed verifier's.
* The deferring verifier rejects garbage, and so does the deployed one on the same proof.
* `DeferredClaim.HasBytecodeShape` rejects a bytecode point one coordinate short (whose cells, hashed
  unpadded, also change the digest).
* Known answers for the claim lists' digests (`claimListDigest` of each scheme's claims) and the
  batching transcript's seed (`batchSeed`), computed by an independent transcription of
  `claim_list_digest` and `FiatShamirState` (`crates/rec_aggregation/src/aggregation.rs`,
  `crates/fiat_shamir/src/lib.rs` at leanVM `4af2cdddd7cd41b9df2cc7e123e314a91b0b4462`), which the
  field change left as they were.
* `batch` on no children is inhabited; the outer verifier's membership challenge
  (`membershipChallengeOf`) of a root is not zero.
* The Rust vectors (`NodeVectors.lean`, written by `scripts/guest-export` from the deployed
  `crates/rec_aggregation/src/aggregation/` at the guest's shape, `bytecode_vars = 23`): the
  statement's encodings and the batching of two children's true claims on the guest's stacked
  bytecode are the Spec's, with the permutation as a parameter too (`batchWith`); a batching hint off
  by one, the children swapped, and another seed or no DA claim in the statement change the
  outcome. What is slow in the
  interpreter, `lake exe node-vectors` (`cli/NodeVectors.lean`) checks compiled: that their seed
  is `deployedSeed guest` (the Keccak-256 IV of the guest's `2^23` bytecode words), and the DA
  root's vector digest.
-/

namespace LeanVMTests.Spec.Node

open LeanVM.Parameters LeanVM.Semantics LeanVM.Protocol LeanVM.Protocol.Verifier
open LeanVM.Protocol.Compile LeanVM.Protocol.KeccakHash LeanVM.Node

/-- The UTF-8 bytes of a string. -/
def utf8 (s : String) : List UInt8 := s.toUTF8.toList

/-- A seed. -/
def seed0 : Digest := keccak256 (utf8 "seed")

/-! ## The deferring verifier -/

/-- A claim on the fixed polynomial. -/
def c : BytecodeClaim := ⟨[1, 0], 1⟩

example (words : List K) :
    claimedBytecodeValue c words c.bytecodePoint c.bytecodeValue = .ret () := by
  simp [claimedBytecodeValue, VerifierProgram.require]

-- Mutation: at another point the claimed evaluation rejects.
example (words : List K) : claimedBytecodeValue c words [0, 0] c.bytecodeValue = .fail := by
  simp [claimedBytecodeValue, VerifierProgram.require, c]

/-- The deferring verifier's seed is the deployed verifier's, from the deployed IV. -/
example (bytecode : List K) (publicInput : Digest) :
    seed bytecode publicInput = keccakF1600 (seedInput (fiatShamirIv bytecode) publicInput) := rfl

/-- A bytecode of 32 words. -/
def bytecode : List K := (List.range 32).map K.ofBits

#guard verifyDeferred bytecode (fiatShamirIv bytecode) default c ⟨[], []⟩ = false
#guard verify bytecode default ⟨[], []⟩ = false

/-! ## The deferred claim's shape -/

/-- A deferred claim of the bytecode's shape, at zero. -/
def zeroClaim : DeferredClaim := ⟨List.replicate 5 0, 0⟩

/-- A deferred claim one coordinate short. -/
def shortClaim : DeferredClaim := ⟨List.replicate 4 0, 0⟩

example : zeroClaim.HasBytecodeShape bytecode ∧ zeroClaim.HasZeroPoint bytecode := by
  simp [DeferredClaim.HasBytecodeShape, DeferredClaim.HasZeroPoint, zeroClaim, bytecode]

/-- No claims. -/
def noClaims : NodeClaims := ⟨[], [], zeroDigest⟩

-- Mutation: a bytecode point one coordinate short; its cells, hashed unpadded, change the digest,
-- and `HasBytecodeShape` rejects it.
#guard statementDigest default ⟨noClaims, zeroClaim⟩ ≠ statementDigest default ⟨noClaims, shortClaim⟩

example : ¬ shortClaim.HasBytecodeShape bytecode := by
  simp [DeferredClaim.HasBytecodeShape, shortClaim, bytecode]

/-- The digest is Keccak-256 of the preimage the collision candidates name. -/
example (seed : Digest) (s : NodeStatement) :
    statementDigest seed s = keccak256 (statementPreimage seed s) := rfl

/-! ## The statement's encodings: known answers -/

/-- The leanXMSS claims. -/
def xmss : List LeanXmssClaim :=
  [(⟨BitVec.ofNat 128 0x0123456789abcdef0011223344556677,
      BitVec.ofNat 128 0x8899aabbccddeeff0f1e2d3c4b5a6978⟩,
    BitVec.ofNat 256 0xdeadbeef00112233445566778899aabbccddeeff102132435465768798a9bacb),
   (⟨BitVec.ofNat 128 0x77665544332211000123456789abcdef,
      BitVec.ofNat 128 0x0f0e0d0c0b0a09080706050403020100⟩, BitVec.ofNat 256 1)]

/-- The leanSPHINCS claims. -/
def sphincs : List LeanSphincsClaim :=
  [(⟨BitVec.ofNat 128 0xfedcba98765432100123456789abcdef, BitVec.ofNat 128 1⟩,
    BitVec.ofNat 256 (2 ^ 256 - 1))]

#guard claimListDigest xmssPublicKeyBytes xmss =
  #v[0xa4ec226585891c3d, 0xa85b86133b887198, 0xa41bc1142d412ba, 0x3c1b43782ff2091d]

#guard claimListDigest Sphincs.PublicKey.toBytes sphincs =
  #v[0xe7e247b70a882189, 0xdb03bad3b98d1589, 0x8309c511a3c41d31, 0x943ad6332bd6dc5d]

/- An empty list hashes the empty string. -/
#guard claimListDigest xmssPublicKeyBytes [] =
  #v[0x6dde3b94f02ddde3, 0x5cf35960c39ee382, 0x6bcc38dfe22067d7, 0x4a3aaafcdd9fb610]

/-! ## The batching's seed -/

/- `FiatShamirState::from_label(RECURSION_AGG_LABEL)`'s state, whose lanes `16..25` the guest
bakes as `AGG_SEED_*`. -/
#guard batchSeed = #v[0x7b01102c79c5b7be, 0x847cf1f24d9ec34e, 0x2d90d578c3add43,
  0x638d5fc17f034837, 0x8908fffd6a3db3c9, 0x70e2e123d94ff318, 0x39b5efa3670d3575,
  0x7ac4eddee7fa6aae, 0xc962960061cd2f36, 0x385b463db8a8e892, 0xab7b8c8359e943ea,
  0x14c4b5acebf4dac3, 0x835fad8dcef3a2d, 0xa5faeaf1fdf71df3, 0xb411f0b74a25102e,
  0x32bdc960d924a99, 0x50a09cd97911d301, 0x694116f29aa38728, 0x7ab5e534f0f15094,
  0x23b9ced806e153a6, 0x6f277a43e4e11b16, 0x2aae2785bf1acea2, 0xe92ef88e65b14f4f,
  0xc134733b28492059, 0xf5645b06f0464d29]

/- No children: the empty sumcheck closes on the zero claim, the nonce `3`, the least one passing the proof of
work. -/
#guard (batch [] ⟨E.ofLimbs (K.ofBits 3) 0, [], 0⟩).isSome

/-! ## The Rust vectors -/

#guard claimListDigest xmssPublicKeyBytes NodeVectors.xmss = NodeVectors.xmssListDigest
#guard claimListDigest Sphincs.PublicKey.toBytes NodeVectors.sphincs = NodeVectors.sphincsListDigest
#guard daClaimDigest NodeVectors.daClaim = NodeVectors.daDigest
#guard zeroDigest = NodeVectors.daNone

/-- The statement of the vectors' claims, deferring their claim. -/
def vectorStatement : NodeStatement :=
  ⟨⟨NodeVectors.xmss, NodeVectors.sphincs, NodeVectors.daDigest⟩, NodeVectors.defer⟩

#guard statementDigest NodeVectors.seed vectorStatement = NodeVectors.statementDigest

-- Mutation: another seed, another digest.
#guard statementDigest seed0 vectorStatement ≠ NodeVectors.statementDigest

/- The batching of the two children, with the Rust's hints, defers the Rust's claim. -/
#guard (batch NodeVectors.children NodeVectors.hints).map (·.bytecodePoint) =
  some NodeVectors.batched.bytecodePoint
#guard (batch NodeVectors.children NodeVectors.hints).map (·.bytecodeValue) =
  some NodeVectors.batched.bytecodeValue

/- The batching with Keccak-f as a parameter, from `batchSeed`, is the same. -/
#guard (Id.run (batchWith (m := Id) keccakF1600 batchSeed NodeVectors.children
  NodeVectors.hints)).map (·.bytecodePoint) = some NodeVectors.batched.bytecodePoint

-- Mutation: a final value one off.
#guard (batch NodeVectors.children { NodeVectors.hints with
  finalBytecodeValue := NodeVectors.hints.finalBytecodeValue + 1 }).isNone

-- Mutation: the children swapped, the same hints.
#guard (batch NodeVectors.children.reverse NodeVectors.hints).isNone

-- Mutation: no DA claim, the zero digest in its place.
#guard statementDigest NodeVectors.seed { vectorStatement with
  claims := { vectorStatement.claims with daDigest := zeroDigest } } ≠ NodeVectors.statementDigest

/-! ## The DA claim -/

/-- A root. -/
def root : Digest := keccak256 []

#guard LeanVM.Da.membershipChallengeOf root != 0

end LeanVMTests.Spec.Node
