import LeanVM.Protocol.KeccakHash
import Statement.Fast.Keccak
import Statement.LeanVmGuest
import LeanVM.Guest.AcceptableChild

/-!
# Keccak tests

`keccak256` against this branch's twelve-round Rust verifier: the empty message and `"abc"`,
and byte ramps of lengths 135, 136, 137, 272 and 300 across the block boundary. The permutation
is also checked on the zero state and lanes `0 … 24`; Merkle hashing on two digests and on
words `1, 2`; and eight hash-chain steps from the zero digest. Digest encoding is checked by a
round trip and rejection of a nonzero top limb.

`Statement.Fast.Keccak` compiles the hash here (`@[csimp]`); the `"abc"` vector and the zero
state's permutation are also decided in the kernel, on the definitions themselves.
-/

namespace LeanVMTests.Protocol.KeccakHash

open LeanVM.Parameters LeanVM.Semantics LeanVM.Protocol LeanVM.Protocol.KeccakHash

/-- The bytes `0, 1, …, n - 1`, modulo `256`. -/
def ramp (n : ℕ) : List UInt8 := (List.range n).map Nat.toUInt8

/-- Keccak-256 of `"abc"`. -/
def abcDigest : Digest :=
  #v[0x226b855d9d36fb88, 0xb605640a2d6e4cc, 0x493726b3cd4427a7, 0x14c93b85d9cc9107]

/-! ## Keccak-256 -/

#guard keccak256 [] =
  #v[0x6dde3b94f02ddde3, 0x5cf35960c39ee382, 0x6bcc38dfe22067d7, 0x4a3aaafcdd9fb610]

#guard keccak256 [0x61, 0x62, 0x63] = abcDigest

example : keccak256 [0x61, 0x62, 0x63] = abcDigest := by decide +kernel

/- The digest in byte order. -/
#guard digestBytes abcDigest =   [0x88, 0xfb, 0x36, 0x9d, 0x5d, 0x85, 0x6b, 0x22, 0xcc, 0xe4, 0xd6, 0xa2, 0x40, 0x56, 0x60, 0xb, 0xa7, 0x27,
    0x44, 0xcd, 0xb3, 0x26, 0x37, 0x49, 0x7, 0x91, 0xcc, 0xd9, 0x85, 0x3b, 0xc9, 0x14]

#guard keccak256 (ramp 135) =
  #v[0xd2d01f40783c291d, 0x8f9baa01e0e148d7, 0x5752618344b5e663, 0x15304c93d16fce27]

#guard keccak256 (ramp 136) =
  #v[0x8fe195b83dc11273, 0xe3f8ada0264d1765, 0xae5197068349cffd, 0x9a5455d3d4556aa2]

#guard keccak256 (ramp 137) =
  #v[0xc85c19a2ebdab03b, 0x1350eca03eb3e79d, 0x4f7a112baad75a6c, 0xd10f0453a8c16a2b]

#guard keccak256 (ramp 272) =
  #v[0x743b33e97e66ac19, 0xa0b29b421f23dd6b, 0x7114b56d923fc829, 0x3deff155476e1129]

#guard keccak256 (ramp 300) =
  #v[0x7a905311d8192255, 0xb8b984503866676b, 0x619bd24ce55df2a7, 0x8d68cd401cb7de64]

-- Mutation: one byte changed.
#guard keccak256 [0x61, 0x62, 0x64] ≠ abcDigest

/-! ## Keccak-f[1600] -/

/-- The zero state's image. -/
def keccakFZero : KeccakState :=
  #v[0x8e5e5438b9a78617, 0xd9cd6a50f259d01e, 0x87b8e7c652a91f35, 0x1093e067cde4e0c5, 0xb033ab90f2d95a45,
    0xe0a72f72a8dd1a45, 0xc53780aa14672f9c, 0x3edd47f50051071d, 0xb3a31d310c178acc, 0x79b586a59257aaa0,
    0xbc4a7c3db3b1f99b, 0x68874063e68a6793, 0x5c6c03332e0e2566, 0x9caa1202b9f030da, 0x5f3b9a782bcf7a9f,
    0xe536c1e061ae7923, 0x6de9b618b73c87ec, 0x2abed1f170918ac2, 0x6aabbd53daed24b7, 0xbfc1416a2c2ee15a,
    0xc6cfe036b90952af, 0x45503617dc7060d7, 0x625611b2c29f7ae4, 0xd43671db2c30647a, 0xcffd0d76222ca01c]

#guard keccakF1600 (Vector.replicate 25 0) = keccakFZero

example : keccakF1600 (Vector.replicate 25 0) = keccakFZero := by decide +kernel

#guard keccakF1600 (Vector.ofFn fun i ↦ i.val.toUInt64) =
  #v[0x31ccb6fee8eeccfe, 0x57bf3dcca8d742e7, 0x33c23c8e00d5fd2d, 0x27408b85c213997d, 0x442b508505b591ae,
    0xe7f3957f8698d9d0, 0x24e9ce4cb83dbdf3, 0xc6ed14e10f4998ba, 0xa445718c4dd30e41, 0xa618c4ddc4f4c14b,
    0x862dab386c0b9ed0, 0xfdade9dec4f977c, 0x38aa031a06ff1231, 0xf4b748a9ffecfc5c, 0xd0af5893c33a5f19,
    0x4dc1ff1ef5fa9c46, 0xb15d80df5456c26b, 0x3a66709440a0c35b, 0xebbdd410f2e7a223, 0x7020a73b189a733d,
    0xa3ea1df2b9a8f601, 0xd15d52bc81a76225, 0xeaac3058e82f6ac1, 0x1de0c38ae5544e5e, 0x72fa1a9d2dc565dd]

/-! ## The Merkle hashes -/

/- `hash_pair` of the `"abc"` digest with itself. -/
#guard hashPair abcDigest abcDigest =
  #v[0x87aec39cbd8b0ff3, 0x2ed4bf5275390cf, 0xe39d46bfa7b8099, 0xdb9cfa39a6452870]

/- A node is one permutation of its children's 64 bytes, padded, from the zero state. -/
#guard hashPair abcDigest abcDigest =
  let s := absorbBlock (Vector.replicate 25 0) (padBlock (digestBytes abcDigest ++ digestBytes abcDigest))
  #v[Keccak.lane s 0, Keccak.lane s 1, Keccak.lane s 2, Keccak.lane s 3]

/- `hash_words` of the words `1, 2`: the sixteen bytes `01 00 … 00 02 00 … 00`. -/
#guard merkleLeafDigest [K.ofBits 1, K.ofBits 2] =
  #v[0x1bc38afb8f68831e, 0xa26c3f85a3cfbb8d, 0x1ebcc2954f58f9b5, 0xe4ee498d2f9746df]

/- The hash chain of `hash_chain.rs`: `compress(d, d)` eight times from the zero digest. -/
#guard (List.range 8).foldl (fun d _ ↦ hashPair d d) (Vector.replicate 4 0) =
  #v[0x821071627431279d, 0x43987ccf951ca772, 0x1c3ed9303d01582c, 0x3957a876e86a0c33]

/-! ## The digest encoding -/

/- The first half of the `"abc"` digest: its words `0` and `1` as the two low limbs. -/
#guard (digestHalves abcDigest).1 =
  E.ofLimbs (K.ofBits 0x226b855d9d36fb88) (K.ofBits 0xb605640a2d6e4cc) 0

#guard (digestHalves abcDigest).2 =
  E.ofLimbs (K.ofBits 0x493726b3cd4427a7) (K.ofBits 0x14c93b85d9cc9107) 0

#guard digestFromHalves (digestHalves abcDigest) = some abcDigest

/- A nonzero top limb in either half is rejected (`Digest::from_halves`). -/
#guard digestFromHalves
  (E.ofLimbs (K.ofBits 0x226b855d9d36fb88) (K.ofBits 0xb605640a2d6e4cc) (K.ofBits 1),
    (digestHalves abcDigest).2) = none

#guard digestFromHalves
  ((digestHalves abcDigest).1,
    E.ofLimbs (K.ofBits 0x493726b3cd4427a7) (K.ofBits 0x14c93b85d9cc9107) (K.ofBits 1)) = none

end LeanVMTests.Protocol.KeccakHash
