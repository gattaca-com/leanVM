import Statement.LeanVmGuest
import LeanXmss.Scheme
import Statement.LeanSphincs
import Statement.Fast.Schemes
import LeanDa.Scheme
import Statement.Fast.Keccak
import Statement.Fast.CubicExt
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
* `DeferredClaim.HasBytecodeShape` rejects a bytecode point one coordinate short, which the digest alone
  cannot tell apart (`statementDigest` carries no lengths: the short claim's odd cell count is
  paired with a zero cell).
* Known answers for the statement's encodings (`claimListDigest` of each scheme's claims,
  `daClaimDigest`, `statementDigest`), the batching transcript's seed (`batchSeed`) and the
  batching of two children (`batch`): computed by an independent transcription of
  `claim_list_digest`, `root_vector_digest`, `statement_digest`, `FiatShamirState` and
  `aggregate_deferred_claims` (`crates/rec_aggregation/src/aggregation.rs`,
  `crates/fiat_shamir/src/lib.rs` at leanVM `4af2cdddd7cd41b9df2cc7e123e314a91b0b4462`, the
  statement's digest at `990b935263dc7f793f280914f262ca7955ddc3e0`, its one DA digest at
  `93ebecc5f87ed7b1248a9772c392f9a41b6ac8d0`, the batching's one challenge and its powers at
  `c445e3d51a5a85031df47240595698f3b24897bf`) over the pinned verifier's Keccak and field
  arithmetic. The fixed-input pins and batching hints now use this branch's twelve-round Rust
  permutation and field arithmetic. The batching's hints are the transcription's honest sumcheck messages; Lean's
  transcript must squeeze the same challenges for the claim to close at the transcription's point,
  which is a true claim on the table.
  Mutations: a wrong final value, or a child's false claim, and the batching rejects.
* `batch` on no children is inhabited; the outer verifier's membership challenge
  (`membershipChallengeOf`) of a root is not zero.
* The Rust vectors (`NodeVectors.lean`, written by `scripts/guest-export` from the deployed
  `crates/rec_aggregation/src/aggregation/` at the guest's shape, `bytecode_vars = 23`): the
  statement's encodings and the batching of two children's true claims on the guest's stacked
  bytecode are the Spec's, and a batching hint off by one rejects. What is slow in the
  interpreter, `lake exe node-vectors` (`cli/NodeVectors.lean`) checks compiled: that their seed
  is `deployedSeed guest` (the Keccak-256 IV of the guest's `2^23` bytecode words), and the DA
  root's vector digest.
-/

namespace LeanVMTests.Spec.Node

open LeanVM.Parameters LeanVM.Semantics LeanVM.Protocol LeanVM.Protocol.Verifier
open LeanVM.Protocol.Compile LeanVM.Protocol.KeccakHash LeanVM.Node

/-- A field element from its limbs. -/
def e (c0 c1 c2 : ℕ) : E := E.ofLimbs (K.ofBits c0) (K.ofBits c1) (K.ofBits c2)

/-- The UTF-8 bytes of a string. -/
def utf8 (s : String) : List UInt8 := s.toUTF8.toList

/-! ## The deferring verifier -/

/-- A claim on the fixed polynomial. -/
def c : BytecodeClaim := ⟨[1, 0], 1⟩

example (words : List K) :
    claimedBytecodeValue c words c.bytecodePoint = .ret c.bytecodeValue := by
  simp [claimedBytecodeValue, VerifierProgram.require]; rfl

-- Mutation: at another point the claimed evaluation rejects.
example (words : List K) : claimedBytecodeValue c words [0, 0] = .fail := by
  simp [claimedBytecodeValue, VerifierProgram.require, c]; rfl

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

-- Mutation: a bytecode point one coordinate short has the same digest (its five cells are paired
-- with a zero cell, the six zero cells of `zeroClaim`), and `Shaped` is what rejects it.
#guard statementDigest default ⟨noClaims, zeroClaim⟩ = statementDigest default ⟨noClaims, shortClaim⟩

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

/-- A DA claim: a root and a membership challenge. -/
def daClaim : DaClaim := ⟨keccak256 (utf8 "root0"), e 1 2 3⟩

/-- A statement's DA digest (the digest of the earlier, root and vector digest, claim). -/
def da : Digest := #v[0x12da73668473f030, 0x3b3c425c5e7d748d, 0x039ebf81841b3952, 0xde4718e69830fa95]

/-- A seed. -/
def seed0 : Digest := keccak256 (utf8 "seed")

/-- A deferred claim: five coordinates, one with a top limb, and a value. -/
def defer0 : DeferredClaim :=
  ⟨[e 1 2 3, e 4 5 6, e 7 8 9, e 10 11 12, e 0xffffffffffffffff 0 1], e 13 14 15⟩

#guard claimListDigest xmssPublicKeyBytes xmss =
  #v[0xa4ec226585891c3d, 0xa85b86133b887198, 0xa41bc1142d412ba, 0x3c1b43782ff2091d]

#guard claimListDigest Sphincs.PublicKey.toBytes sphincs =
  #v[0xe7e247b70a882189, 0xdb03bad3b98d1589, 0x8309c511a3c41d31, 0x943ad6332bd6dc5d]

/- An empty list hashes the empty string. -/
#guard claimListDigest xmssPublicKeyBytes [] =
  #v[0x6dde3b94f02ddde3, 0x5cf35960c39ee382, 0x6bcc38dfe22067d7, 0x4a3aaafcdd9fb610]

/- The DA claim's digest: Keccak-256 of the root, the challenge's three limbs and a zero word
(computed independently by another Keccak-256). -/
#guard daClaimDigest daClaim =
  #v[0x8d761eabbf3e05a0, 0x2a30029d7471b868, 0x9b6bb3665dcbafff, 0x763f971bd0dce805]

#guard statementDigest seed0 ⟨⟨xmss, sphincs, da⟩, defer0⟩ =
  #v[0xfb415dd67e08a5e6, 0x206539780ead98d8, 0xf556a1f87cee1558, 0xed15719bf5e13b43]

/- The same claim one coordinate short: five cells, paired with a zero cell. -/
#guard statementDigest seed0 ⟨⟨xmss, sphincs, da⟩, { defer0 with
    bytecodePoint := defer0.bytecodePoint.take 4 }⟩ =
  #v[0x7c8f9a9b823d113, 0xf2864140a9fec5d5, 0xef892254c963d114, 0x3a7a501c67e1ac31]

-- Mutation: other signature lists give another digest.
#guard statementDigest seed0 ⟨⟨[], [], da⟩, defer0⟩ =
  #v[0x5e9429b319e7017c, 0x8761878f0f88d1ce, 0xa7954139fbc8d1e2, 0xcd4ba2e12249c8eb]

-- Mutation: no DA claim, the zero digest in its place.
#guard statementDigest seed0 ⟨⟨xmss, sphincs, zeroDigest⟩, defer0⟩ =
  #v[0x4d1aac84f2339ab9, 0x1c559acd93976958, 0xa23f56c298ba543c, 0x175dc8156271c216]

/-! ## The batching: known answers and mutations -/

/- `FiatShamirState::from_label(RECURSION_AGG_LABEL)`'s state, whose lanes `16..25` the guest
bakes as `AGG_SEED_*`. -/
#guard batchSeed =   #v[0x7b01102c79c5b7be, 0x847cf1f24d9ec34e, 0x2d90d578c3add43, 0x638d5fc17f034837, 0x8908fffd6a3db3c9,
    0x70e2e123d94ff318, 0x39b5efa3670d3575, 0x7ac4eddee7fa6aae, 0xc962960061cd2f36, 0x385b463db8a8e892,
    0xab7b8c8359e943ea, 0x14c4b5acebf4dac3, 0x835fad8dcef3a2d, 0xa5faeaf1fdf71df3, 0xb411f0b74a25102e,
    0x32bdc960d924a99, 0x50a09cd97911d301, 0x694116f29aa38728, 0x7ab5e534f0f15094, 0x23b9ced806e153a6,
    0x6f277a43e4e11b16, 0x2aae2785bf1acea2, 0xe92ef88e65b14f4f, 0xc134733b28492059, 0xf5645b06f0464d29]

/-- A table of eight words, the claims' polynomial. -/
def table : List E := (List.range 8).map fun i ↦ ofK (K.ofBits (3 * i + 1))

/-- The table's value at a point. -/
def mle (point : List E) : E := multilinearEval table point

/-- Child `i`: its public input, a true fresh claim and a true carried claim. -/
def child (i : ℕ) : ChildClaims :=
  let p := [e (i + 1) 2 0, e 5 i 7, e 0 0 (i + 9)]
  let q := [e i 1 1, e 2 2 i, e (i + 3) 0 0]
  ⟨keccak256 [i.toUInt8], ⟨p, mle p⟩, ⟨q, mle q⟩⟩

/-- Two children. -/
def children : List ChildClaims := [child 0, child 1]

/- The transcription's value of child 0's fresh claim. -/
#guard (child 0).freshClaim.bytecodeValue = e 0x3b2 0x222 0x7ee

/-- The transcription's honest hints. -/
def hints : BatchingHints :=
  ⟨[((e 0xfed3621cbcaf95ff 0x70b27439e7ad8532 0xfb292ff9cd116741), (e 0x6c316d445b29b15e 0x7e3adf842867a818
    0xf583bf9075b31d73)), ((e 0xe4d382b230957227 0x89793daa9806c14c 0xbabcb5d4d0be26d5), (e 0xffcfa853569eb034
    0x6506e48b2b2936ad 0x1275a042a4d6a8a8)), ((e 0x2092062d1c67dbbc 0x67db76905c941594 0xa36545bd756feb26), (e
    0x67af58d71ceb117f 0x1e907b0d4a41a2b8 0x7b2dd57452916205))], (e 0xb1e34676160bf5be 0xb22502621a139ca9
    0xed51d0c93c99ee2)⟩

/-- The transcription's batched point. -/
def batchedPoint : List E :=
  [(e 0x348b8da705dab3c 0x4a0f07253f23f679 0x8c1921b2319f3ee6), (e 0x192422a82a953dc1 0x9ddf6bd1eef1325b
    0xd301f1095374cf03), (e 0x2fe66cd913b0274 0x16b48030478bb366 0x5728203ea9402b74)]

#guard (batch children hints).map (·.bytecodePoint) = some batchedPoint
#guard (batch children hints).map (·.bytecodeValue) = some hints.finalBytecodeValue

/- The batched claim is true of the table. -/
#guard mle batchedPoint = hints.finalBytecodeValue

/- The batching with Keccak-f as a parameter, from `batchSeed`, is the same. -/
#guard (Id.run (batchWith (m := Id) keccakF1600 batchSeed children hints)).map (·.bytecodePoint) =
  some batchedPoint

-- Mutation: a wrong final value.
#guard (batch children { hints with finalBytecodeValue := hints.finalBytecodeValue + 1 }).isNone

-- Mutation: child 1's carried claim false, the same hints.
#guard (batch [child 0, { child 1 with carriedClaim := { (child 1).carriedClaim with
  bytecodeValue := (child 1).carriedClaim.bytecodeValue + 1 } }] hints).isNone

-- Mutation: the children swapped, the same hints.
#guard (batch [child 1, child 0] hints).isNone

/- No children: the empty sumcheck closes on the zero claim. -/
#guard (batch [] ⟨[], 0⟩).isSome

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

-- Mutation: a final value one off.
#guard (batch NodeVectors.children { NodeVectors.hints with
  finalBytecodeValue := NodeVectors.hints.finalBytecodeValue + 1 }).isNone

/-! ## The DA claim -/

/-- A root. -/
def root : Digest := keccak256 []

#guard LeanVM.Da.membershipChallengeOf root != 0

end LeanVMTests.Spec.Node
