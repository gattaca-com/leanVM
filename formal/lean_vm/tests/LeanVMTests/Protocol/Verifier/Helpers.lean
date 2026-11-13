import LeanVM.Protocol.Verifier

/-!
# Verifier helper tests: differential against the verifier at its pin

Every expected value below is the Rust verifier's (`crates/verifier`) at `leanvm-rev`, on inputs drawn from the word generator `fw`, which the driver mirrors exactly; the
driver is the verifier crate's sources with their items made crate-visible and a module printing
each probe, kept in the port's scratch, not in the repository. Values that no change of the
protocol touched (the RMFE, the carry-less arithmetic, Keccak-f, Keccak-256, the IV and the seed)
are those of the earlier pins (`4af2cdddd`, `990b93526`), the others the driver's at `afe891a73`
but `fold_row` (at `71205cca4`) and `derive_config`'s query counts (at the pin): nothing else they
read changed since. The line citations are to
`crates/verifier/src/` at the pin. Compiled checks (`#guard`): `E` multiplication does not reduce
in the kernel from module files.

Covered: `powers`, `eq_kernel`, `multilinear_eval` (on `E` and on lifted `K` tables), `eq_eval`,
`dot`, `index_mle`, `poly_eval`, `gpow`, `padding_weight`, `log2_ceil`, `log2_strict`,
`selector_point`, `grinding_bits`; `subspace_roots`, `Subspace::at`, `novel_eval`, `fold_row`,
`induced_weight`, `ext_row`; the ring map's `ring_coefficients`, `ring_psi`, `ring_target`,
`ring_weight` and `frobenius_8`; `stack_offsets`, `Placement::stack_point` and `eq_above`; the
tables' columns, widths, global bases, log-height floors, constraint counts, count columns and
padding rows, `jump_constraints`, `keccak_cell_lanes`, `layout_limb`, the `q_keccak` slots
(`qkeccak_slot`) and output columns; `build_layout` on a sample bytecode and sizes (log-heights,
stack log-size, every placement, every bus block's shape and flushed tuple at a sample row),
`bus_layout` of the three sides; `derive_config` on all 56 admissible pairs; `along`,
`linear_transpose`, `instance_weights`, the slot weights of `open`, the RMFE's factors,
`RMFE_EMBED`, `RMFE_PROJECT`, the carry-less product and division and the Keccak reduction's
constants; `KECCAK_RC`, `KECCAK_RHO`, `KECCAK_PI`, `keccak_f`, `keccak_hash` across the block
boundary, the sponge's `duplex`, `sample` and `absorb_nonce`; `TAG` (`versionTag`), the IV
(`fiatShamirIv`), the public input's halves, and the Fiat-Shamir seed (`Sponge::new`: Keccak-f of
`iv ‖ pi ‖ 0`). `build_layout`'s rejections are checked against what the verifier rejects on the
same inputs.
-/

namespace LeanVMTests.Protocol.Verifier.Helpers

open LeanVM.Parameters LeanVM.Semantics LeanVM.Protocol LeanVM.Protocol.Verifier
open LeanVM.Protocol.KeccakHash LeanVM.Protocol.Compile

/-! ## Inputs, as the driver draws them -/

/-- The driver's 64-bit word generator. -/
def fw (k lane : ℕ) : ℕ :=
  ((k + 1) * 0x9E3779B97F4A7C15 + lane * 0xD1B54A32D192ED03 + 0x632BE59BD9B4E019) % 2 ^ 64

/-- An `E` element from its two limbs. -/
def e (c0 c1 : ℕ) : E := E.ofLimbs (K.ofBits c0) (K.ofBits c1)

/-- The driver's `fe(k)`. -/
def fe (i : ℕ) : E := e (fw i 0) (fw i 1)

/-- `fe(a), …, fe(a + n - 1)`. -/
def fes (a n : ℕ) : List E := (List.range n).map fun i ↦ fe (a + i)

/-- A lifted `K` table of sixteen entries. -/
def kTable : List E := (List.range 16).map fun i ↦ ofK (K.ofBits (fw i 5))

/-- Three fold rounds, `α, β` the elements `fe(910 + 2r), fe(911 + 2r)`. -/
def sampleRounds : List FoldRound :=
  (List.range 3).map fun r ↦ ⟨fe (910 + 2 * r), fe (911 + 2 * r)⟩

/-- A domain point outside the small subspaces: the word `fw(3, 7)`. -/
def samplePoint : K := K.ofBits (fw 3 7)

/-- A sample placement (its two checks printed at the pin, `../leanvm-rev`). -/
def samplePlacement : Placement := { variableCount := 3, index := 0b1011001 }

/-- The sample bytecode: 32 words, so `log_bytecode = 1`. -/
def sampleBytecode : List K := (List.range 32).map fun i ↦ K.ofBits (fw i 9)

/-- The sample row counts: `xor, mul, set, deref, jump, keccak`. Their log-heights are
`2, 1, 1, 2, 2, 9`, `KECCAK`'s raised to its floor `KECCAK_MIN_STEPS_LOG`. -/
def sampleRows : List ℕ := [3, 2, 2, 4, 3, 5]

/-- The sample layout, at memory log-size 16. -/
def sampleLayout : Layout :=
  (buildLayout sampleBytecode 16 sampleRows).getD ⟨0, 0, [], [], [], [], [], 0, []⟩

/-- Each block's log-height, owner, arity, and its flushed tuple at the row `fe(200 + i)`,
batched by powers of `fe 300`. -/
def blockProbe (blocks : List BusBlock) : List (ℕ × ℕ × ℕ × E) :=
  blocks.map fun b ↦ (b.logRows, b.owner, b.coordinates.length,
    dot ((powers (fe 300) 16).take b.coordinates.length)
      (b.coordinates.map (·.evaluate fun i ↦ fe (200 + i))))

/-- A side's bus layout: its depth and every placement's size and offset. -/
def busProbe (frameworkLogRows : List ℕ) (blocks : List BusBlock) : ℕ × List (ℕ × ℕ) :=
  let bl := busLayout frameworkLogRows blocks
  (bl.depth, (bl.framework ++ bl.tables).map fun q ↦ (q.variableCount, q.index))

/-- The sample public input: the 32 bytes `fw(i, 10) mod 256`. -/
def samplePublicInput : Digest :=
  digestOfBytes ((List.range 32).map fun i ↦ (fw i 10 % 256).toUInt8)

/-- The `n` bytes `fw(i, 15) mod 256`, hashed across Keccak-256's block boundary. -/
def sampleBytes (n : ℕ) : List UInt8 := (List.range n).map fun i ↦ (fw i 15 % 256).toUInt8

/-- A sponge state: the lanes `fw(i, 12)`. -/
def sampleState : KeccakState := Vector.ofFn fun i : Fin 25 ↦ (fw i 12).toUInt64

/-- `Sponge::duplex` from `sampleState` with these scalars waiting, and the challenge
`sample` reads off the result. -/
def duplexProbe (pending : List (E × ℕ)) : List UInt64 × E :=
  let s := keccakF1600 (duplexInput ⟨sampleState, pending, none⟩)
  (s.toList, challengeOf s)

/-- `Sponge::absorb_nonce` from `sampleState` with these scalars waiting: the proof of work, the
state, and the challenge it leaves for the next `sample`. -/
def nonceProbe (pending : List (E × ℕ)) (nonce : E) : UInt64 × List UInt64 × Option E :=
  let (pow, sponge) := Id.run (absorbNonce keccakF1600 ⟨sampleState, pending, none⟩ nonce)
  (pow, sponge.state.toList, sponge.ground)

/-- The `(j, k)` pairs `along` is probed at. -/
def alongPairs : List (ℕ × ℕ) :=
  [(0, 1), (63, 4), (64, 2), (255, 3), (1000, 1), (1599, 4), (1280, 0)]

/-- `linear_transpose` on the weights `fes 1000 1600`: its length, its first two entries,
and all of it batched by powers of `fe 501`. -/
def transposeProbe : ℕ × List E × E :=
  let t := keccakLinearTranspose (fes 1000 1600)
  (t.length, t.take 2, dot (powers (fe 501) 1600) t)

/-- A Keccak claim at `log_steps = 9`: a five-coordinate pack index and a segment index. -/
def sampleClaim : KeccakClaim := ⟨fe 609, fes 640 16, fes 610 6, fes 1000 1600⟩

/-- The slot weights at `log_steps = 9` of `sampleClaim` on `count` segments from `first`: the
number of rows, whether each is 64 wide, and all of them batched by powers of `fe 620`. At the pin
they are no function of their own: `open` (`keccak.rs:212-225`) sums them against the slot
challenges' eq weights; the driver computes them by its loop with the slot weight left out. -/
def slotProbe (first count : ℕ) : ℕ × Bool × E :=
  let w := keccakSlotWeights 9 sampleClaim first count
  (w.length, w.all (·.length == 64), dot (powers (fe 620) 4096) w.flatten)

/-- The pairs the carry-less product is probed at: a 20-bit and a 12-bit polynomial. -/
def carrylessInputs : List (ℕ × ℕ) :=
  (List.range 4).map fun i ↦ (fw i 13 % 2 ^ 20, fw i 14 % 2 ^ 12)

/-- The pairs the carry-less division is probed at: a 40-bit dividend by a degree-11 divisor, then
two small ones. -/
def divmodInputs : List (ℕ × ℕ) :=
  (List.range 4).map (fun i ↦ (fw i 13 % 2 ^ 40, fw i 14 % 2 ^ 12 ||| 2 ^ 11)) ++
    [(0b1011, 0b10), (5, 0b100101)]

/-- The Keccak reduction's and the layout's constants, in the driver's order. -/
def keccakConstants : List ℕ :=
  [keccakDepth, keccakSegments, keccakSegmentBits, qkeccakSlotBits, keccakMinStepsLog, keccakBits,
    keccakLocalBits, keccakLaneBits, kBits, pack, embedBits, productBits, globalColumnCount,
    busBits, Whir.minLogStack, Whir.maxLogStack]

/-! ## Constants and conventions -/

/- `y`, the generator of `E` over `K`, is a root of its modulus `y^2 + x·y + 1`. -/
#guard yE * yE = gen * yE + 1
#guard gen = ofK (K.ofBits 2)
#guard (buildLayout sampleBytecode 16 sampleRows).isSome
#guard sampleLayout.push.all fun b ↦ b.rows == sampleRows.getD b.owner 0
#guard sampleLayout.placements.length = globalColumnCount + tableWidths.sum

/- Every column a table's flushes name exists (`cols`'s `expect`, `layout.rs:104-111`): a missing
name would be `columnIndexByName`'s out-of-range index, the table's width. -/
#guard tables.all fun t ↦ (t.flushes.push ++ t.flushes.pull).all fun tuple ↦
  tuple.all fun f ↦ f.terms.all fun (m, _) ↦ m.all (· < t.width)

/-! ## `build_layout` rejects what the verifier rejects

On each input the driver ran `build_layout` and recorded whether it rejected. -/

/- A bytecode length that is not a power of two; one shorter than `2^BUS_BITS`; exactly
`2^BUS_BITS`. -/
#guard (buildLayout (sampleBytecode.take 24) 16 sampleRows).isNone
#guard (buildLayout (sampleBytecode.take 8) 16 sampleRows).isNone
#guard (buildLayout (sampleBytecode.take 16) 16 sampleRows).isSome
/- The memory window `[1, 30]`. -/
#guard (buildLayout sampleBytecode 0 sampleRows).isNone
#guard (buildLayout sampleBytecode 1 sampleRows).isSome
#guard (buildLayout sampleBytecode 31 sampleRows).isNone
#guard (buildLayout sampleBytecode 30 sampleRows).isSome
/- The row cap `2^30`, on any table. -/
#guard (buildLayout sampleBytecode 16 [2 ^ 30 + 1, 2, 2, 4, 3, 5]).isNone
#guard (buildLayout sampleBytecode 16 [2 ^ 30, 2, 2, 4, 3, 5]).isSome
#guard (buildLayout sampleBytecode 16 [3, 2, 2, 4, 3, 2 ^ 30]).isSome
/- No `KECCAK` row is no rejection: the table's height is its floor. Past the floor it grows. -/
#guard ((buildLayout sampleBytecode 16 [3, 2, 2, 4, 3, 0]).map (·.tableLogHeights)) =
  some [2, 1, 1, 2, 2, 9]
#guard ((buildLayout sampleBytecode 16 [3, 2, 2, 4, 3, 600]).map (·.tableLogHeights)) =
  some [2, 1, 1, 2, 2, 10]
/- Not one row count per table (the Rust verifier reads exactly `N_TABLES`, `lib.rs:248`). -/
#guard (buildLayout sampleBytecode 16 [3, 2, 2, 4, 3, 5, 0]).isNone
#guard (buildLayout sampleBytecode 16 [3, 2, 2, 4, 3]).isNone

/-! ## `derive_config` outside the window -/

#guard (Whir.ladder 15 1).isNone ∧ (Whir.ladder 33 1).isNone
#guard (Whir.ladder 20 0).isNone ∧ (Whir.ladder 20 5).isNone

/-! ## Field and multilinear arithmetic (`field.rs:255-264`, `multilinear.rs`, `layout.rs:113`;
`transcript.rs:95-104`) -/

#guard (powers (fe 1) 6) =
    [(e 0x1 0x0), (e 0x9f9ad90ed849d843 0x71502341a9dcc546),
     (e 0xe1a37ba00fcb757 0x6b7f4a078cf4033), (e 0xa8122ee46f0a5aa9 0x207e9faa54d4477e),
     (e 0x1386ae69256209bc 0x2935cd2a2d6d520a), (e 0x75aabec44b6c29e4 0x149ffcea20068577)]

#guard (eqKernel [fe 2, fe 3, fe 4]) =
    [(e 0xbead2bd434e75d5f 0xda6b629b43525dbe), (e 0x3d20be3cf61a211 0x4fcc71c241dada95),
     (e 0x59418220818cc34 0x21a6e468f3306334), (e 0xc2aa7e2ea5b77ff9 0xfff7675fd604dd9a),
     (e 0x425c891d2a16dba4 0xa5ca2492f4f2a42b), (e 0x232a65ab074ef486 0x9dd2217f5e0b9e70),
     (e 0xc4b7e823417d1e96 0x51803e9a6db7dbfa), (e 0xdf8042ae3a0c7d36 0x226eab19e0f2d824)]

#guard (multilinearEval ((List.range 8).map fe) [fe 10, fe 11, fe 12]) =
    (e 0x3e614120d15a4536 0xcd07a5dac74e097a)

#guard (multilinearEval kTable (fes 13 4)) =
    (e 0x1dc3a1c24ded43f0 0xf8b7ffe93761c398)

#guard (eqEval [fe 20, fe 21, fe 22] [fe 23, fe 24, fe 25]) =
    (e 0x82bd01e07f82614f 0x7851b73699908985)

#guard (dot [fe 26, fe 27, fe 28] [fe 29, fe 30, fe 31]) =
    (e 0x51628321ce552fa1 0x8fa5c83b4a6033d3)

#guard (indexMle [fe 32, fe 33, fe 34, fe 35]) =
    (e 0x76b748ec823a93fb 0xf9c17010b7cf0a2c)

#guard (polyEval [fe 40, fe 41, fe 42, fe 43, fe 44] (fe 45)) =
    (e 0x9ad61792f8d1c662 0xe7ba24ba7baefa70)

#guard (Verifier.gpow 5) =
    (e 0x20 0x0)

#guard (Verifier.gpow (2 ^ 32 - 1)) =
    (e 0xae3111b3bb81428f 0x0)

#guard ([0, 5, 11, 16].map (paddingWeight (fes 120 4))) =
    [(e 0x1 0x0), (e 0xfe088f14baad8355 0x396e02003469a810),
     (e 0x36fb865ece8ef208 0x83f8ca436afff1ea), (e 0x0 0x0)]

#guard ([0, 1, 2, 3, 5, 64, 65, 2 ^ 20].map log2Ceil) =
    [0, 0, 1, 2, 3, 6, 7, 20]

#guard ([16, 15, 17, 1, 0, 2 ^ 20].map log2Strict) =
    [some 4, none, none, some 0, none, some 20]

#guard (selectorPoint 13 6) =
    [(e 0x1 0x0), (e 0x0 0x0), (e 0x1 0x0), (e 0x1 0x0), (e 0x0 0x0), (e 0x0 0x0)]

/- `grinding_bits`, at the inputs of `fiat_shamir`'s own test -/
#guard ([0, 1, 2, 8, 9, 31, 46, 64, 65, 2 ^ 20].map grindingBits) =
    [0, 0, 0, 0, 1, 2, 3, 3, 4, 17]

/-! ## WHIR helpers (`whir.rs:72-202`) -/

#guard ((subspaceRoots 12).map kToNat) =
    [0x1, 0x6, 0x168, 0x1166880, 0x11616686880801b, 0x726b6e3c6e39234e, 0x796adf34d64462c2,
     0xf1060866be6fa9c, 0x6a06f1b0a76acb05, 0xd2d14e471a7186d9, 0x6d55ab94bb5f9605,
     0xd08c5570c6423a2a, 0xa0ca298b954731]

#guard ((subspaceAt 5 samplePoint).map kToNat) =
    [0x97fed3e591e34b82, 0x820abeab5326f91b, 0xa114ae0175dc5efb, 0xbe64d74b712dcf9b,
     0x78e3eabfdbd2fc1f, 0xbbb44a699f093317]

#guard (novelEval (fes 50 7) (subspaceAt 5 samplePoint)) =
    (e 0xb6f010b2bcb2ddf3 0x4ea90edd296a3d9)

/- `2^5 + 1` coefficients: the last weighs `What_5`. -/
#guard (novelEval (fes 50 33) (subspaceAt 5 samplePoint)) =
    (e 0x87e252e36d1b76a7 0xe08d5e4b4a5b7a2e)

#guard (foldRow (fes 900 8) (K.ofBits 23) sampleRounds) =
    (e 0xb8d373c1d29ea65a 0xba103fe1bfbd4c47)

#guard (inducedWeight 5 [3, 17, 30, 0] (powers (fe 60) 4) [fe 61, fe 62, fe 63, fe 64, fe 65]) =
    (e 0x1f414c84d37628a1 0xfc45dc207f904f40)

#guard (extRow ((List.range 6).map fun i ↦ K.ofBits (fw i 6))) =
    [(e 0xeba31c864270ea40 0x89da963fc1bb6655), (e 0x28120ff94105e26a 0xc64989b2c0505e7f),
     (e 0x6481036c3f9ada94 0x2b87d25bee556a9)]

/- An odd word is dropped. -/
#guard (extRow ((List.range 7).map fun i ↦ K.ofBits (fw i 6))) =
    [(e 0xeba31c864270ea40 0x89da963fc1bb6655), (e 0x28120ff94105e26a 0xc64989b2c0505e7f),
     (e 0x6481036c3f9ada94 0x2b87d25bee556a9)]

/-! ## Ring switching (`ring_switch.rs:10-68`) -/

#guard (ringCoefficients (fes 80 8) (fes 88 8)) =
    [(e 0x50a3732d98b25847 0x485783167b605da4), (e 0x6abb07138c18f0f9 0xeac0f214fe325859),
     (e 0xd611ecc1cc4d7387 0x4c9e9d11365c1009), (e 0x12259a8cb8ab8986 0xcc029a2a82c1df76),
     (e 0xe567e0c2e347c200 0xd1433bb6c65b3eef), (e 0x69ffd590d5ce4d57 0x7c4f1ab7870c6460),
     (e 0x8c4f9c70125ac3f5 0xb8f6fe4a0ec0f6b8), (e 0x37cbf2331969ec0d 0xb3130f3790a148b4),
     (e 0x6b37f06cee3af49e 0xa2731ed82dc1ca60), (e 0xf7efb79f9c3f4b85 0x8f015b52604972b1),
     (e 0xde572696c46b9d4f 0xa4545fa66490a55d), (e 0x1452537f52385cca 0x342dea07aacf85b5),
     (e 0xe134b152c370c9ca 0x27cf634c9ea7e054), (e 0xe99dfd5eefc375ad 0xb97246efb1831500),
     (e 0xc9e0fa00f3e10266 0x7e8374871f70dcc7), (e 0x4830007e983476f1 0x83a07c3a3ce4c8fc),
     (e 0x31998518d78293e 0xc70177ed1621782), (e 0x665719ec07847cf5 0x161e89c73e72a776),
     (e 0xf677b54994c01f5f 0x3ef09127b630250f), (e 0x6b18eedae39b86b8 0xa781a1b9af8e17ce),
     (e 0x140e5c59be8ffc1c 0xc212ca142beea2b1), (e 0xccc99e8f38368d8d 0xa22ff0ef6cf6971e),
     (e 0xf981c0f6300698ec 0xc38bd33b406d1e8b), (e 0x601f24f47b8b4b98 0x2d1e568a5efc6a89),
     (e 0xe4cd38e05ed82c32 0x8743589f25c0c4d2), (e 0xa21982c467a0e41 0x6de2080435fad976),
     (e 0x27245558a61e31a3 0x37aed79f9f48980b), (e 0xa6737ae9057e6d48 0x21e6aee3abdfcb4f),
     (e 0xad83ad6089574a19 0x907b2785d8cb8ba), (e 0x2a96e6a8a92f269c 0x2a3d3963c4c6338e),
     (e 0x5618cf20676d722f 0x9c9ff4efdfb18922), (e 0x5260511768780187 0xb65099e25fa961a4),
     (e 0x57717a286aae26df 0x54f6dd51432bfc56), (e 0x9bbd0bb06816822c 0xf87b83080eaf117d),
     (e 0xd1cba5a6ca17829a 0xd1029176ba5f290), (e 0x94478543effd4a5f 0x982e46b42467c97e),
     (e 0x2c9a2089ca3ef389 0x27025004df19c53a), (e 0xeb3b1bc451ebdb52 0x678ffd8eeaa46d9b),
     (e 0x2633586dd089f9f0 0xea6c9a4ffeed6340), (e 0x3ce88fd4eafdadb5 0x2a2461e73fe5f42c),
     (e 0xe610862aa209e590 0x6ee7470aa8a54db4), (e 0xfb05cf13c32e8dd6 0xc382dceeb934ef62),
     (e 0xe274fac9a3d02b03 0x8ea64c0fe553be15), (e 0x704d785f0f74af52 0x3543596cc84b8279),
     (e 0x18954d11d787dbe1 0x690318d6b15d7c1f), (e 0x66ac6c02d71561cf 0x6fc5b31d9837a89d),
     (e 0xc56eb4882a99e0f3 0x28cea2759a37504d), (e 0x22cbbfd05d23ef9f 0xa3dc0e8c508898b4),
     (e 0xf4abe83726c2a35e 0x20609568eba0870d), (e 0x5f8b3250b8cb068e 0x9074e1ad5bf4d69b),
     (e 0xc3c0e4e98922d4dc 0xb8c6e9ad428ada02), (e 0x1c3c14cb7e00faf8 0x1feaddd455ec9a8d),
     (e 0x2cfc85a970655be1 0xf3ecab42599a8a7f), (e 0x3a4ea0cd9267e037 0xd282f0ec3fab5310),
     (e 0xd1fe62fbcd70f92b 0x86848a0a6de29968), (e 0xc66219c21c0d31ab 0x1fddc983bd14871a),
     (e 0xee238a1ddcb4cc62 0x1076e966ccf1012c), (e 0xaa3447cda6a54d 0x7c102d47a198ac87),
     (e 0x697b03fdc18b2c1f 0xb63a1b13bb08eb27), (e 0x60be2559a1170a21 0x6df0832ec7cee6dd),
     (e 0x802a0b932c334ecc 0x29b24e4ec9fc781c), (e 0xee98666e19e50c19 0x85fc104d06ba77ff),
     (e 0x3638fa1b4cf435c2 0x883f3ea488e72c88), (e 0xfb74e1c597a32a29 0x649b0ee661d7fa42)]

#guard (ringPsi (fe 86) (fes 88 8)) =
    (e 0xb254215a0dad7123 0xa6def745624dd0b9)

#guard (ringTarget (fes 130 64) (fes 80 8) (fes 88 8)) =
    (e 0x9a2f7a8e451a7d5e 0x92fdd92ed9c8d7c3)

#guard (ringWeight (fes 71 4) (fes 75 4) (fes 100 64)) =
    (e 0x3f0a017409b44f2b 0xfd9d5985b95b676d)

/- `ring_switch`'s weight `W` at `r' = fes 75 4` -/
#guard (ringWeight (fes 71 4) (fes 75 4) (ringCoefficients (fes 80 8) (fes 88 8))) =
    (e 0xd929f3139f6cb600 0xa9c0309d15183135)

#guard (frobenius8 (fe 87) 3) =
    (e 0x6d621675ade71890 0xc961818aa4701716)

/-! ## Placements (`multilinear.rs:114-148`) -/

#guard (stackOffsets [3, 5, 5, 0, 2, 5, 1]) =
    ([96, 0, 32, 110, 104, 64, 108], 7)

#guard (samplePlacement.stackPoint [fe 90, fe 91, fe 92] 9) =
    [(e 0xa0e42a8c192efb90 0x729974beeac1e893), (e 0x3f1ba445987977a5 0x10d0ee786a0c64a8),
     (e 0xdd531dff17c3f3ba 0xaf086831e956e0bd), (e 0x1 0x0), (e 0x1 0x0), (e 0x0 0x0), (e 0x1 0x0),
     (e 0x0 0x0), (e 0x0 0x0)]

#guard (samplePlacement.eqAbove (fes 93 9)) =
    (e 0x8e1ca664b6991f1a 0xcae2304f3a4d6e6e)

/-! ## The tables (`layout.rs:129-481`) -/

#guard (tables.map Table.columns) =
    [["pc", "fp", "o_a", "o_b", "o_c", "va_0", "va_1", "vb_0", "vb_1", "cnt_a", "cnt_b", "cnt_c",
      "cnt_bc"],
     ["pc", "fp", "o_a", "o_b", "o_c", "va_0", "va_1", "vb_0", "vb_1", "cnt_a", "cnt_b", "cnt_c",
      "cnt_bc"], ["pc", "fp", "o", "k_0", "k_1", "cnt", "cnt_bc"],
     ["pc", "fp", "o1", "o2", "o3", "f_pc", "f_fp", "ptr", "v3_0", "v3_1", "cnt_ptr", "cnt_target",
      "cnt_local", "cnt_bc"],
     ["pc", "fp", "o_c", "o_d", "o_f", "v_cond", "v_pc", "v_fp", "cnt_c", "cnt_d", "cnt_f",
      "cnt_bc", "w", "b"],
     ["pc", "fp", "o_m0", "o_m1", "o_m2", "o_m3", "o_tail", "o_b8", "o_cap", "o_out", "odd_in",
      "odd_out", "out_two", "out_all", "free", "in_0", "in_1", "in_2", "in_3", "in_4", "in_5",
      "in_6", "in_7", "in_8", "in_9", "in_10", "in_11", "in_12", "in_13", "in_14", "in_15", "in_16",
      "in_17", "in_18", "in_19", "in_20", "in_21", "in_22", "in_23", "in_24", "out_0", "out_1",
      "out_2", "out_3", "out_4", "out_5", "out_6", "out_7", "out_8", "out_9", "out_10", "out_11",
      "out_12", "out_13", "out_14", "out_15", "out_16", "out_17", "out_18", "out_19", "out_20",
      "out_21", "out_22", "out_23", "out_24", "cnt_0", "cnt_1", "cnt_2", "cnt_3", "cnt_4", "cnt_5",
      "cnt_6", "cnt_7", "cnt_8", "cnt_9", "cnt_10", "cnt_11", "cnt_12", "cnt_13", "cnt_14",
      "cnt_15", "cnt_16", "cnt_17", "cnt_18", "cnt_19", "cnt_20", "cnt_21", "cnt_22", "cnt_23",
      "cnt_24", "cnt_25", "cnt_bc"]]

#guard (tableWidths) =
    [13, 13, 7, 14, 14, 92]

#guard (globalColumnBases) =
    [5, 18, 31, 38, 52, 66]

#guard (tableLogFloors) =
    [0, 0, 0, 0, 0, 9]

#guard (tables.map Table.nConstraints) =
    [0, 0, 0, 0, 2, 0]

#guard (tables.map Table.countColumns) =
    [[9, 10, 11, 12], [9, 10, 11, 12], [5, 6], [10, 11, 12, 13], [8, 9, 10, 11],
     [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
      88, 89, 90, 91]]

#guard (tables.map fun t ↦ t.padding.map eToNat) =
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0x8e5e5438b9a78617, 0xd9cd6a50f259d01e, 0x87b8e7c652a91f35,
      0x1093e067cde4e0c5, 0xb033ab90f2d95a45, 0xe0a72f72a8dd1a45, 0xc53780aa14672f9c,
      0x3edd47f50051071d, 0xb3a31d310c178acc, 0x79b586a59257aaa0, 0xbc4a7c3db3b1f99b,
      0x68874063e68a6793, 0x5c6c03332e0e2566, 0x9caa1202b9f030da, 0x5f3b9a782bcf7a9f,
      0xe536c1e061ae7923, 0x6de9b618b73c87ec, 0x2abed1f170918ac2, 0x6aabbd53daed24b7,
      0xbfc1416a2c2ee15a, 0xc6cfe036b90952af, 0x45503617dc7060d7, 0x625611b2c29f7ae4,
      0xd43671db2c30647a, 0xcffd0d76222ca01c, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0]]

#guard (jumpConstraints (fes 400 14)) =
    [(e 0x81ab9cd5332cf16a 0xc9caef27e887674d), (e 0x7344c6060c9320ca 0x987a8c1955c70dee)]

#guard ([false, true].map fun odd ↦ (List.range 13).map (LeanVM.Semantics.cellLanes odd)) =
    [[(some 0, some 1), (some 2, some 3), (some 4, some 5), (some 6, some 7), (some 8, some 9),
      (some 10, some 11), (some 12, some 13), (some 14, some 15), (some 16, none),
      (some 17, some 18), (some 19, some 20), (some 21, some 22), (some 23, some 24)],
     [(none, some 0), (some 1, some 2), (some 3, some 4), (some 5, some 6), (some 7, some 8),
      (some 9, some 10), (some 11, some 12), (some 13, some 14), (some 15, some 16),
      (some 17, some 18), (some 19, some 20), (some 21, some 22), (some 23, some 24)]]

#guard ((layoutLimb (some 3) (some 5) 7 (some 9)).evaluate fun i ↦ fe (150 + i)) =
    (e 0xd51b5df9c3a2114e 0xaed8f199a2d42f42)

#guard ((layoutLimb none (some 5) 7 (some 9)).evaluate fun i ↦ fe (150 + i)) =
    (e 0x82a6265599796dc4 0xb572cac242f27e3b)

#guard ((layoutLimb (some 3) none 7 none).evaluate fun i ↦ fe (150 + i)) =
    (e 0xd0b726c353e61095 0xbe9f7b339d0c5441)

#guard ((layoutLimb none none 7 none).evaluate fun i ↦ fe (150 + i)) =
    (e 0x0 0x0)

#guard ((layoutLimb none none 7 (some 9)).evaluate fun i ↦ fe (150 + i)) =
    (e 0x45d7f98b68426d39 0x178d43be39d55a3c)

#guard ((layoutLimb (some 4) (some 4) 7 none).evaluate fun i ↦ fe (150 + i)) =
    (e 0x2ec298ebebce00d0 0x77e31ebd60edd3)

#guard ((layoutLimb none (some 9) 7 (some 9)).evaluate fun i ↦ fe (150 + i)) =
    (e 0x45d7f98b68426d39 0x178d43be39d55a3c)

/- `build_layout`'s `q_keccak` slots (`qkeccak_slot`), and the output lanes' columns (`outputs`) -/
#guard ((List.range 200).filterMap fun c ↦ (keccakSlot c).map (c, ·)) =
    [(66, 50), (67, 51), (68, 52), (69, 53), (70, 54), (71, 55), (72, 56), (73, 57), (74, 58),
     (75, 59), (76, 60), (77, 61), (78, 62), (79, 63), (81, 0), (82, 1), (83, 2), (84, 3), (85, 4),
     (86, 5), (87, 6), (88, 7), (89, 8), (90, 9), (91, 10), (92, 11), (93, 12), (94, 13), (95, 14),
     (96, 15), (97, 16), (98, 17), (99, 18), (100, 19), (101, 20), (102, 21), (103, 22), (104, 23),
     (105, 24)]

#guard (keccakOutputColumns) =
    [106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
     125, 126, 127, 128, 129, 130]

/-! ## `build_layout` on the sample (`bus.rs:77-103`, `layout.rs:483-567`) -/

#guard (sampleLayout.tableLogHeights) =
    [2, 1, 1, 2, 2, 9]

#guard (sampleLayout.stackLog) =
    18

#guard (sampleLayout.logBytecode) =
    1

#guard (sampleLayout.placements.map (·.map fun p ↦ (p.variableCount, p.index))) =
    [some (16, 0), some (16, 65536), some (16, 131072), some (1, 243876), some (15, 196608),
     some (2, 243712), some (2, 243716), some (2, 243720), some (2, 243724), some (2, 243728),
     some (2, 243732), some (2, 243736), some (2, 243740), some (2, 243744), some (2, 243748),
     some (2, 243752), some (2, 243756), some (2, 243760), some (1, 243878), some (1, 243880),
     some (1, 243882), some (1, 243884), some (1, 243886), some (1, 243888), some (1, 243890),
     some (1, 243892), some (1, 243894), some (1, 243896), some (1, 243898), some (1, 243900),
     some (1, 243902), some (1, 243904), some (1, 243906), some (1, 243908), some (1, 243910),
     some (1, 243912), some (1, 243914), some (1, 243916), some (2, 243764), some (2, 243768),
     some (2, 243772), some (2, 243776), some (2, 243780), some (2, 243784), some (2, 243788),
     some (2, 243792), some (2, 243796), some (2, 243800), some (2, 243804), some (2, 243808),
     some (2, 243812), some (2, 243816), some (2, 243820), some (2, 243824), some (2, 243828),
     some (2, 243832), some (2, 243836), some (2, 243840), some (2, 243844), some (2, 243848),
     some (2, 243852), some (2, 243856), some (2, 243860), some (2, 243864), some (2, 243868),
     some (2, 243872), some (9, 222208), some (9, 222720), some (9, 223232), some (9, 223744),
     some (9, 224256), some (9, 224768), some (9, 225280), some (9, 225792), some (9, 226304),
     some (9, 226816), some (9, 227328), some (9, 227840), some (9, 228352), some (9, 228864),
     some (9, 229376), some (9, 196608), some (9, 197120), some (9, 197632), some (9, 198144),
     some (9, 198656), some (9, 199168), some (9, 199680), some (9, 200192), some (9, 200704),
     some (9, 201216), some (9, 201728), some (9, 202240), some (9, 202752), some (9, 203264),
     some (9, 203776), some (9, 204288), some (9, 204800), some (9, 205312), some (9, 205824),
     some (9, 206336), some (9, 206848), some (9, 207360), some (9, 207872), some (9, 208384),
     some (9, 208896), none, none, none, none, none, none, none, none, none, none, none, none, none,
     none, none, none, none, none, none, none, none, none, none, none, none, some (9, 229888),
     some (9, 230400), some (9, 230912), some (9, 231424), some (9, 231936), some (9, 232448),
     some (9, 232960), some (9, 233472), some (9, 233984), some (9, 234496), some (9, 235008),
     some (9, 235520), some (9, 236032), some (9, 236544), some (9, 237056), some (9, 237568),
     some (9, 238080), some (9, 238592), some (9, 239104), some (9, 239616), some (9, 240128),
     some (9, 240640), some (9, 241152), some (9, 241664), some (9, 242176), some (9, 242688),
     some (9, 243200)]

#guard (blockProbe sampleLayout.push) =
    [(2, 0, 3, (e 0xc77f4d63a694c8bc 0x49db39674d2a61ac)),
     (2, 0, 9, (e 0x5477aa853dfdf47f 0xda20611594aca9aa)),
     (2, 0, 5, (e 0x14787529efbe2405 0x8a32441475ba3804)),
     (2, 0, 5, (e 0x30bf17c5c69d8f19 0x7203d24986d6705)),
     (2, 0, 5, (e 0x82f66e6c811d1319 0xe6685e0342accb44)),
     (1, 1, 3, (e 0xc77f4d63a694c8bc 0x49db39674d2a61ac)),
     (1, 1, 9, (e 0x48b8b39d507e266e 0x5946a5b3d81f5847)),
     (1, 1, 5, (e 0x14787529efbe2405 0x8a32441475ba3804)),
     (1, 1, 5, (e 0x30bf17c5c69d8f19 0x7203d24986d6705)),
     (1, 1, 5, (e 0xdee72ec8999b27f2 0xe6083333e99f7fe4)),
     (1, 2, 3, (e 0xc77f4d63a694c8bc 0x49db39674d2a61ac)),
     (1, 2, 9, (e 0xfcffafbd8028aa77 0x2445999093296cc2)),
     (1, 2, 5, (e 0x3d6d8e988a2b26ae 0x4429d17ccb80f1ed)),
     (2, 3, 3, (e 0xc77f4d63a694c8bc 0x49db39674d2a61ac)),
     (2, 3, 9, (e 0xcf687a2e5165cb37 0xcafce0693b18d671)),
     (2, 3, 5, (e 0xf373c23aeb590ca9 0x226ac29cac305ae3)),
     (2, 3, 5, (e 0x999a50d4e8ff6a30 0xc9bdb27b2921d093)),
     (2, 3, 5, (e 0x30f51b189cf98456 0xfec3a2fa2f62e170)),
     (2, 4, 3, (e 0xf0b42a7910c706ef 0x145db5d09a7db01c)),
     (2, 4, 9, (e 0x20dbcd4330881f33 0x8ff272ed83641a2c)),
     (2, 4, 5, (e 0xfde1a663790b2da6 0x9d459073bc457344)),
     (2, 4, 5, (e 0x466fea2f5d4b4377 0xf4b61427e33572e6)),
     (2, 4, 5, (e 0xcee82f496bdc6693 0xd47cfb53f5060b)),
     (9, 5, 3, (e 0xc77f4d63a694c8bc 0x49db39674d2a61ac)),
     (9, 5, 16, (e 0x8b7ecbb5b7e0df54 0x58e67b7dcb7b363e)),
     (9, 5, 5, (e 0x4b8f3afa5d0f5f5a 0xe3376dd52b7435fe)),
     (9, 5, 5, (e 0xec9ded693c405084 0x94c15383ba388d9e)),
     (9, 5, 5, (e 0x47e5b36a6a952ad9 0x11125929fe7924fd)),
     (9, 5, 5, (e 0xeae33bb7ea10b28b 0x4414dfd140e07108)),
     (9, 5, 5, (e 0xba640eae40267b8 0xd759cb9d63f099be)),
     (9, 5, 5, (e 0xcdc4d475e4161a01 0xbe6d046238316bcd)),
     (9, 5, 5, (e 0x2269c9dc24ca537e 0x8aa212aad3f42f)),
     (9, 5, 5, (e 0x5d5717bc77ba0d20 0xe99851f94416ad62)),
     (9, 5, 5, (e 0x1bb05d0af5184fca 0xe465371aa362e3dc)),
     (9, 5, 5, (e 0xc6ca2d7de6807043 0x4a7e831a59676f8e)),
     (9, 5, 5, (e 0xf58a0beb0989ea1d 0x816de8916ebfab49)),
     (9, 5, 5, (e 0x631e8896fd395b06 0x9e12f4f01e82fc05)),
     (9, 5, 5, (e 0x8359db553322439d 0xc6383a97501bc813)),
     (9, 5, 5, (e 0xdda35c52de7b266b 0x68f4af1669046a7d)),
     (9, 5, 5, (e 0x16f3eeccc0e7a23b 0x55c79af15490c3c8)),
     (9, 5, 5, (e 0xb978a31d5a087c22 0x5ae46655ed3d44d0)),
     (9, 5, 5, (e 0xb8d6616935975033 0x12fb4022241c3816)),
     (9, 5, 5, (e 0x505dd217454a67b 0x9439915de0dc9bed)),
     (9, 5, 5, (e 0xef76ba727ac304e4 0x6b2fc445e40dfef6)),
     (9, 5, 5, (e 0x5ee69819afad3059 0x637fa006e9a32bf5)),
     (9, 5, 5, (e 0x8a2320cc84d693a 0x2ad4ad5856efe34a)),
     (9, 5, 5, (e 0x38b9d4197f9f2439 0x15c17fdf69573439)),
     (9, 5, 5, (e 0x8ca3641aeacc282f 0xbcca9f3ef2d2a62e)),
     (9, 5, 5, (e 0xec6a916d0ecd35bf 0x77ad755680cc0b0d)),
     (9, 5, 5, (e 0xf425a1f22f635b30 0x21ee7e413179c02a)),
     (9, 5, 5, (e 0xcf1d50c367c4e54c 0x3d1b72babc6ce5e3))]

#guard (blockProbe sampleLayout.pull) =
    [(2, 0, 3, (e 0x480baa0adf35c21e 0x2b89c30f85e10827)),
     (2, 0, 9, (e 0x8b4db1992bd45ffb 0x93b6b34d9809e2b)),
     (2, 0, 5, (e 0x7446233a1ce2fd65 0xe4b76e41aa5f1ae8)),
     (2, 0, 5, (e 0x34e116e043806f3d 0x55c576ca9492b3e6)),
     (2, 0, 5, (e 0x7b29651ac624dffa 0x90cd0947506a617f)),
     (1, 1, 3, (e 0x480baa0adf35c21e 0x2b89c30f85e10827)),
     (1, 1, 9, (e 0x9782a88146578dea 0x8a5daf9295336fc6)),
     (1, 1, 5, (e 0x7446233a1ce2fd65 0xe4b76e41aa5f1ae8)),
     (1, 1, 5, (e 0x34e116e043806f3d 0x55c576ca9492b3e6)),
     (1, 1, 5, (e 0x273825bedea2eb11 0x90ad6477fb59d5df)),
     (1, 2, 3, (e 0x480baa0adf35c21e 0x2b89c30f85e10827)),
     (1, 2, 9, (e 0x68f00db998f8bdd8 0xb1777c69e57c67a5)),
     (1, 2, 5, (e 0xb2fcad3840a4ebc1 0x686f8833c6ea25d6)),
     (2, 3, 3, (e 0x480baa0adf35c21e 0x2b89c30f85e10827)),
     (2, 3, 9, (e 0x164e7aa31a4f6484 0x80e63fd47251d312)),
     (2, 3, 5, (e 0xf72dc31f6e44ec8d 0x708f8972a0cf8e00)),
     (2, 3, 5, (e 0x60455ba2afc6a6d3 0xbf18e53f3be77aa8)),
     (2, 3, 5, (e 0xefcf00048ad02fd2 0x2dd8a8db624ed6f1)),
     (2, 4, 3, (e 0x480baa0adf35c21e 0x2b89c30f85e10827)),
     (2, 4, 9, (e 0xd904c63577b1d3d0 0xf95725a991a2b017)),
     (2, 4, 5, (e 0x8bf0a9d5ec04167 0xa1776b229380be6e)),
     (2, 4, 5, (e 0x2651bc3cae179a17 0x9a333e723cd0500a)),
     (2, 4, 5, (e 0xcab62e6ceec186b7 0x523137155f0ad2e8)),
     (9, 5, 3, (e 0x480baa0adf35c21e 0x2b89c30f85e10827)),
     (9, 5, 16, (e 0x95603ba76a57962b 0xaefb703326110973)),
     (9, 5, 5, (e 0x84f42ad3e3701b77 0xe34e785742d3b20b)),
     (9, 5, 5, (e 0x70432dba32f88bac 0x1c0775bc7117bbd6)),
     (9, 5, 5, (e 0x9028c56cae10fe75 0x1625760f76a71e91)),
     (9, 5, 5, (e 0xc0ca243858b41815 0x965e06d1558b28d8)),
     (9, 5, 5, (e 0x5a58d56ee3bbe07c 0x752b89239df60784)),
     (9, 5, 5, (e 0xfded56235bea7461 0xc925b264ec79cb26)),
     (9, 5, 5, (e 0x5eb343e0b7d0e683 0x37aeb3c520818161)),
     (9, 5, 5, (e 0x52a604101484a04c 0x7214332bbc537eb8)),
     (9, 5, 5, (e 0x1c5d3ce7a2521666 0x8fba7127482ab9fd)),
     (9, 5, 5, (e 0x6220362bd2ea9e6c 0x29011d1690b5162f)),
     (9, 5, 5, (e 0x60a34c5b7d3ebb73 0x2982048bbed83fca)),
     (9, 5, 5, (e 0x8f0ebe28795380ca 0x8249a6e42d296019)),
     (9, 5, 5, (e 0x9af7cd7681baf4e7 0x2945c5797ebf5932)),
     (9, 5, 5, (e 0x5a984f5b0026acac 0xf67365ad5d0c506)),
     (9, 5, 5, (e 0x871d1c6959402272 0x6cf245307f514ed8)),
     (9, 5, 5, (e 0xcad157ac9b62617 0x2d184d30e65a4f77)),
     (9, 5, 5, (e 0xe3c9ed6a88f8616a 0x79db456300aa0dc8)),
     (9, 5, 5, (e 0x7b7451ceac3780d6 0x9bef6f164bd9fec4)),
     (9, 5, 5, (e 0x54b25295922f2516 0x1a09fa51da772613)),
     (9, 5, 5, (e 0x1897fc5a7cadbada 0x13575886e90d2743)),
     (9, 5, 5, (e 0x86379b9e3b2da2e5 0xa98381c7bbddd474)),
     (9, 5, 5, (e 0x4eb467f13ad61547 0x71470ffa9fa259d3)),
     (9, 5, 5, (e 0xeaf1687882da352c 0x6a48a6289df21f82)),
     (9, 5, 5, (e 0x3d3f094a710445da 0x96154d6190f8a62f)),
     (9, 5, 5, (e 0x2d53279831b846af 0xbca652d799d3df79)),
     (9, 5, 5, (e 0x41c4bb77a7df8d4b 0x7b1b671fefe4f6ed))]

#guard (blockProbe sampleLayout.count) =
    [(2, 0, 1, (e 0x2cadbfc644cea953 0xfe6309f916619656)),
     (2, 0, 1, (e 0xcae5397fc4192568 0x9c9a83b295ac126b)),
     (2, 0, 1, (e 0x691cb3394363a17d 0x3ad1fd6c14f68e80)),
     (2, 0, 1, (e 0x7542cf2c2ae1d92 0xd909772594410a95)),
     (1, 1, 1, (e 0x2cadbfc644cea953 0xfe6309f916619656)),
     (1, 1, 1, (e 0xcae5397fc4192568 0x9c9a83b295ac126b)),
     (1, 1, 1, (e 0x691cb3394363a17d 0x3ad1fd6c14f68e80)),
     (1, 1, 1, (e 0x7542cf2c2ae1d92 0xd909772594410a95)),
     (1, 2, 1, (e 0xb3cfd8e047a4b8ff 0x858523131937a602)),
     (1, 2, 1, (e 0x52075299c6ef3514 0x23bc9ccc98822217)),
     (2, 3, 1, (e 0xcae5397fc4192568 0x9c9a83b295ac126b)),
     (2, 3, 1, (e 0x691cb3394363a17d 0x3ad1fd6c14f68e80)),
     (2, 3, 1, (e 0x7542cf2c2ae1d92 0xd909772594410a95)),
     (2, 3, 1, (e 0xa58ba6ac41f899a7 0x7740f0df138b86aa)),
     (2, 4, 1, (e 0x8e76460cc5842d3e 0x602b903f97171a41)),
     (2, 4, 1, (e 0x2cadbfc644cea953 0xfe6309f916619656)),
     (2, 4, 1, (e 0xcae5397fc4192568 0x9c9a83b295ac126b)),
     (2, 4, 1, (e 0x691cb3394363a17d 0x3ad1fd6c14f68e80)),
     (9, 5, 1, (e 0xc8d0605a1d19cdeb 0x9a85aa8ceeacbaee)),
     (9, 5, 1, (e 0x6707da139c644a00 0x38bd24466df73703)),
     (9, 5, 1, (e 0x53f53cd1baec615 0xd6f49dffed41b318)),
     (9, 5, 1, (e 0xa376cd869af9422a 0x752c17b96c8c2f2d)),
     (9, 5, 1, (e 0x41ae47401a43be3f 0x13639172ebd6ab42)),
     (9, 5, 1, (e 0xdfe5c0f9998e3a54 0xb19b0b2c6b212757)),
     (9, 5, 1, (e 0x7e1d3ab318d8b669 0x4fd284e5ea6ba36c)),
     (9, 5, 1, (e 0x1c54b46c9823327e 0xee09fe9f69b61f81)),
     (9, 5, 1, (e 0xba8c2e26176dae93 0x8c417858e9009b96)),
     (9, 5, 1, (e 0x58c3a7df96b82aa8 0x2a78f212684b17ab)),
     (9, 5, 1, (e 0xf6fb21991602a6bd 0xc8b06bcbe79593c0)),
     (9, 5, 1, (e 0x95329b52954d22d2 0x66e7e58566e00fd5)),
     (9, 5, 1, (e 0x336a150c14979ee7 0x51f5f3ee62a8bea)),
     (9, 5, 1, (e 0xd1a18ec593e21afc 0xa356d8f8657507ff)),
     (9, 5, 1, (e 0x6fd9087f132c9711 0x418e52b1e4bf8414)),
     (9, 5, 1, (e 0xe10823892771326 0xdfc5cc6b640a0029)),
     (9, 5, 1, (e 0xac47fbf211c18f3b 0x7dfd4624e3547c3e)),
     (9, 5, 1, (e 0x4a7f75ab910c0b50 0x1c34bfde629ef853)),
     (9, 5, 1, (e 0xe8b6ef6510568765 0xba6c3997e1e97468)),
     (9, 5, 1, (e 0x86ee691e8fa1037a 0x58a3b3516133f07d)),
     (9, 5, 1, (e 0x2525e2d80eeb7f8f 0xf6db2d0ae07e6c92)),
     (9, 5, 1, (e 0xc35d5c918e35fba4 0x9512a6c45fc8e8a7)),
     (9, 5, 1, (e 0x6194d64b0d8077b9 0x334a207ddf1364bc)),
     (9, 5, 1, (e 0xffcc50048ccaf3ce 0xd1819a375e5de0d1)),
     (9, 5, 1, (e 0x9e03c9be0c156fe3 0x6fb913f0dda85ce6)),
     (9, 5, 1, (e 0x3c3b43778b5febf8 0xdf08daa5cf2d8fb)),
     (9, 5, 1, (e 0xda72bd310aaa680d 0xac280763dc3d5510))]

#guard (busProbe [0, 16, 1] sampleLayout.push) =
    (17,
     [(0, 79950), (16, 0), (1, 79932), (2, 79872), (2, 79876), (2, 79880), (2, 79884), (2, 79888),
      (1, 79934), (1, 79936), (1, 79938), (1, 79940), (1, 79942), (1, 79944), (1, 79946),
      (1, 79948), (2, 79892), (2, 79896), (2, 79900), (2, 79904), (2, 79908), (2, 79912),
      (2, 79916), (2, 79920), (2, 79924), (2, 79928), (9, 65536), (9, 66048), (9, 66560),
      (9, 67072), (9, 67584), (9, 68096), (9, 68608), (9, 69120), (9, 69632), (9, 70144),
      (9, 70656), (9, 71168), (9, 71680), (9, 72192), (9, 72704), (9, 73216), (9, 73728),
      (9, 74240), (9, 74752), (9, 75264), (9, 75776), (9, 76288), (9, 76800), (9, 77312),
      (9, 77824), (9, 78336), (9, 78848), (9, 79360)])

#guard (busProbe [0, 16, 1] sampleLayout.pull) =
    (17,
     [(0, 79950), (16, 0), (1, 79932), (2, 79872), (2, 79876), (2, 79880), (2, 79884), (2, 79888),
      (1, 79934), (1, 79936), (1, 79938), (1, 79940), (1, 79942), (1, 79944), (1, 79946),
      (1, 79948), (2, 79892), (2, 79896), (2, 79900), (2, 79904), (2, 79908), (2, 79912),
      (2, 79916), (2, 79920), (2, 79924), (2, 79928), (9, 65536), (9, 66048), (9, 66560),
      (9, 67072), (9, 67584), (9, 68096), (9, 68608), (9, 69120), (9, 69632), (9, 70144),
      (9, 70656), (9, 71168), (9, 71680), (9, 72192), (9, 72704), (9, 73216), (9, 73728),
      (9, 74240), (9, 74752), (9, 75264), (9, 75776), (9, 76288), (9, 76800), (9, 77312),
      (9, 77824), (9, 78336), (9, 78848), (9, 79360)])

#guard (busProbe [] sampleLayout.count) =
    (14,
     [(2, 13824), (2, 13828), (2, 13832), (2, 13836), (1, 13872), (1, 13874), (1, 13876),
      (1, 13878), (1, 13880), (1, 13882), (2, 13840), (2, 13844), (2, 13848), (2, 13852),
      (2, 13856), (2, 13860), (2, 13864), (2, 13868), (9, 0), (9, 512), (9, 1024), (9, 1536),
      (9, 2048), (9, 2560), (9, 3072), (9, 3584), (9, 4096), (9, 4608), (9, 5120), (9, 5632),
      (9, 6144), (9, 6656), (9, 7168), (9, 7680), (9, 8192), (9, 8704), (9, 9216), (9, 9728),
      (9, 10240), (9, 10752), (9, 11264), (9, 11776), (9, 12288), (9, 12800), (9, 13312)])

/-! ## `derive_config` (`whir.rs:24-70`) -/

/-- `derive_config(log_n, rate)` for every admissible pair: folds, rates, queries. -/
def refConfigs : List (ℕ × ℕ × List ℕ × List ℕ × List ℕ) :=
  [(16, 1, [6, 4, 4], [1, 4, 7], [230, 60, 42]),
   (17, 1, [6, 4, 4], [1, 4, 7], [225, 58, 38]),
   (18, 1, [6, 4, 4], [1, 4, 7], [222, 57, 35]),
   (19, 1, [6, 4, 4], [1, 4, 7], [220, 56, 34]),
   (20, 1, [6, 4, 4, 4], [1, 4, 7, 10], [219, 55, 33, 27]),
   (21, 1, [6, 4, 4, 4], [1, 4, 7, 10], [218, 55, 32, 25]),
   (22, 1, [6, 4, 4, 4], [1, 4, 7, 10], [218, 55, 32, 24]),
   (23, 1, [6, 4, 4, 4], [1, 4, 7, 10], [218, 55, 32, 23]),
   (24, 1, [6, 4, 4, 4, 4], [1, 4, 7, 10, 13], [218, 55, 32, 23, 20]),
   (25, 1, [6, 4, 4, 4, 4], [1, 4, 7, 10, 13], [218, 55, 32, 23, 19]),
   (26, 1, [6, 4, 4, 4, 4], [1, 4, 7, 10, 13], [218, 55, 32, 23, 18]),
   (27, 1, [6, 4, 4, 4, 4], [1, 4, 7, 10, 13], [218, 55, 32, 23, 18]),
   (28, 1, [6, 4, 4, 4, 4, 4], [1, 4, 7, 10, 13, 16], [218, 55, 32, 23, 18, 16]),
   (29, 1, [6, 4, 4, 4, 4, 4], [1, 4, 7, 10, 13, 16], [218, 55, 32, 23, 18, 15]),
   (30, 1, [6, 4, 4, 4, 4, 4], [1, 4, 7, 10, 13, 16], [218, 55, 32, 23, 18, 15]),
   (16, 2, [6, 4, 4], [2, 5, 8], [112, 47, 35]),
   (17, 2, [6, 4, 4], [2, 5, 8], [111, 46, 32]),
   (18, 2, [6, 4, 4], [2, 5, 8], [110, 45, 31]),
   (19, 2, [6, 4, 4], [2, 5, 8], [110, 45, 29]),
   (20, 2, [6, 4, 4, 4], [2, 5, 8, 11], [109, 44, 29, 24]),
   (21, 2, [6, 4, 4, 4], [2, 5, 8, 11], [109, 44, 28, 23]),
   (22, 2, [6, 4, 4, 4], [2, 5, 8, 11], [109, 44, 28, 22]),
   (23, 2, [6, 4, 4, 4], [2, 5, 8, 11], [109, 44, 28, 21]),
   (24, 2, [6, 4, 4, 4, 4], [2, 5, 8, 11, 14], [109, 44, 28, 21, 18]),
   (25, 2, [6, 4, 4, 4, 4], [2, 5, 8, 11, 14], [109, 44, 28, 21, 17]),
   (26, 2, [6, 4, 4, 4, 4], [2, 5, 8, 11, 14], [109, 44, 28, 21, 17]),
   (27, 2, [6, 4, 4, 4, 4], [2, 5, 8, 11, 14], [109, 44, 28, 21, 17]),
   (28, 2, [6, 4, 4, 4, 4, 4], [2, 5, 8, 11, 14, 17], [109, 44, 28, 21, 17, 15]),
   (29, 2, [6, 4, 4, 4, 4, 4], [2, 5, 8, 11, 14, 17], [109, 44, 28, 21, 17, 15]),
   (30, 2, [6, 4, 4, 4, 4, 4], [2, 5, 8, 11, 14, 17], [109, 44, 28, 21, 17, 15]),
   (16, 3, [6, 4, 4], [3, 6, 9], [74, 39, 30]),
   (17, 3, [6, 4, 4], [3, 6, 9], [74, 38, 28]),
   (18, 3, [6, 4, 4], [3, 6, 9], [73, 37, 27]),
   (19, 3, [6, 4, 4], [3, 6, 9], [73, 37, 26]),
   (20, 3, [6, 4, 4, 4], [3, 6, 9, 12], [73, 37, 25, 22]),
   (21, 3, [6, 4, 4, 4], [3, 6, 9, 12], [73, 37, 25, 20]),
   (22, 3, [6, 4, 4, 4], [3, 6, 9, 12], [73, 37, 25, 20]),
   (23, 3, [6, 4, 4, 4], [3, 6, 9, 12], [73, 37, 25, 19]),
   (24, 3, [6, 4, 4, 4, 4], [3, 6, 9, 12, 15], [73, 37, 25, 19, 17]),
   (25, 3, [6, 4, 4, 4, 4], [3, 6, 9, 12, 15], [73, 37, 25, 19, 16]),
   (26, 3, [6, 4, 4, 4, 4], [3, 6, 9, 12, 15], [73, 37, 25, 19, 16]),
   (27, 3, [6, 4, 4, 4, 4], [3, 6, 9, 12, 15], [73, 37, 25, 19, 16]),
   (28, 3, [6, 4, 4, 4, 4, 4], [3, 6, 9, 12, 15, 18], [73, 37, 25, 19, 16, 15]),
   (29, 3, [6, 4, 4, 4, 4, 4], [3, 6, 9, 12, 15, 18], [73, 37, 25, 19, 16, 14]),
   (30, 3, [6, 4, 4, 4, 4, 4], [3, 6, 9, 12, 15, 18], [73, 37, 25, 19, 16, 15]),
   (16, 4, [6, 4, 4], [4, 7, 10], [56, 33, 27]),
   (17, 4, [6, 4, 4], [4, 7, 10], [55, 32, 25]),
   (18, 4, [6, 4, 4], [4, 7, 10], [55, 32, 24]),
   (19, 4, [6, 4, 4], [4, 7, 10], [55, 32, 23]),
   (20, 4, [6, 4, 4, 4], [4, 7, 10, 13], [55, 32, 23, 20]),
   (21, 4, [6, 4, 4, 4], [4, 7, 10, 13], [55, 32, 23, 19]),
   (22, 4, [6, 4, 4, 4], [4, 7, 10, 13], [55, 32, 23, 18]),
   (23, 4, [6, 4, 4, 4], [4, 7, 10, 13], [55, 32, 23, 18]),
   (24, 4, [6, 4, 4, 4, 4], [4, 7, 10, 13, 16], [55, 32, 23, 18, 16]),
   (25, 4, [6, 4, 4, 4, 4], [4, 7, 10, 13, 16], [55, 32, 23, 18, 15]),
   (26, 4, [6, 4, 4, 4, 4], [4, 7, 10, 13, 16], [55, 32, 23, 18, 15]),
   (27, 4, [6, 4, 4, 4, 4], [4, 7, 10, 13, 16], [55, 32, 23, 18, 15]),
   (28, 4, [6, 4, 4, 4, 4, 4], [4, 7, 10, 13, 16, 19], [55, 32, 23, 18, 15, 14]),
   (29, 4, [6, 4, 4, 4, 4, 4], [4, 7, 10, 13, 16, 19], [55, 32, 23, 18, 16, 14]),
   (30, 4, [6, 4, 4, 4, 4, 4], [4, 7, 10, 13, 16, 19], [55, 32, 23, 18, 16, 14])]

#guard refConfigs.length = 60
#guard refConfigs.all fun (n, r, folds, rates, queries) ↦
  decide ((Whir.ladder n r).map (fun ls ↦
    (ls.map (·.fold), ls.map (·.logInvRate), ls.map (·.queries))) = some (folds, rates, queries))

/-! ## Keccak GKR and the RMFE (`keccak.rs:14-113`, `:212-225`) -/

#guard (alongPairs.map fun (j, k) ↦ keccakAlong j k) =
    [64, 319, 192, 127, 1064, 1535, 1280]

#guard (transposeProbe) =
    (1600, [(e 0xca8abc6d967083b1 0x23e90670f9332354), (e 0xe18c283116a57e56 0x3d63a0587881af69)],
     (e 0x18512301d3bb1def 0x58f71514239c1028))

#guard (keccakInstanceWeights (fe 600)) =
    [(e 0xa52fefc1ac0d7a0 0xaafd3b9a726acd34), (e 0xacfe43ba7bb6e9bd 0x5a7f7f3b1a274ba),
     (e 0x2332778e532fc3ec 0xf6d55c6b09decc68), (e 0x171aadeba89d3bfe 0xc46ffcbc452f4842),
     (e 0x1884961b9484a0b3 0xe15bd89b05d5a7aa), (e 0x7a84ab7916508023 0x940a4d7330a93a0b),
     (e 0x3a8e63c129aa464a 0x3bc6c1f50e7c6af2), (e 0x159f64052fdb06ca 0x6f9caf57bf95618c),
     (e 0xc289527a4cd91db7 0xe3ffa88c291323a4), (e 0x2452c1cde88ed6f0 0x632e5b58ef41269a),
     (e 0xa45b3f2a2c038545 0xed0ae5a668a1567c), (e 0x1795531c103995e4 0x1ff11c0eb8ee98ca),
     (e 0xfc501772b3dd0922 0x4086d18d1e79f0f7), (e 0xa3ccb4106931fc89 0x7249f88792352b6a),
     (e 0x3356022b7c67f956 0xaf170f5564911e72), (e 0xb6c9bb9d329b71bd 0x65a3d0bc80700cb7)]

#guard (slotProbe 0 2) =
    (64, true, (e 0xee64d4d570e2c99e 0x44df39a864249bb5))

#guard (slotProbe 1 1) =
    (64, true, (e 0xcc1c5bf753aae3d9 0x9b117bda43b57d11))

#guard (rmfeFactors) =
    [(0b10, 1), (0b11, 1), (0b1011, 2), (0b1101, 2), (0b100101, 3), (0b101001, 3), (0b101111, 3)]

#guard (rmfeEmbed) =
    [0x40c50b, 0x7f79f2, 0x4b4e0a, 0x72740c, 0x5357da, 0x1d1c9e, 0x4be6e, 0x785246, 0x97cdc,
     0x109e7e, 0x1f1454, 0x213cfc, 0x4cfcc8, 0x775770, 0x187386, 0x818a16]

#guard (rmfeProject) =
    [0x1, 0x7fffffffffff, 0x274e9d3a74e9, 0x1d3a74e9d3a7, 0x65cb972e5cb9, 0x5cb972e5cb97,
     0x6690aec7cd21, 0x5d8f9a42bb1f, 0x73485763e690, 0x3750967c6ea1, 0x12cf8dd4259f, 0x1ba84b3e3750,
     0xad0c9f715a1, 0x64fb8ad0c9f, 0x456864fb8ad0, 0x400000000000]

#guard (carrylessInputs.map fun (a, b) ↦ carryless a b) =
    [0xbee80b8, 0x5d073452, 0x436b037e, 0x18ea27ac]

#guard (divmodInputs.map fun (a, f) ↦ carrylessDivmod a f) =
    [(0x1b08c182, 0x5e5), (0x1d3534da, 0x1c8), (0x95fdc62, 0xbb), (0x3f3d41f, 0x1b9), (0x5, 0x1),
     (0x0, 0x5)]

#guard (keccakConstants) =
    [6, 2, 1, 6, 9, 1600, 11, 5, 64, 16, 24, 47, 5, 4, 16, 30]

#guard (ringMapWidth) = 8

/-! ## Keccak-f, Keccak-256 and the sponge (`hash.rs:5-98`, `transcript.rs:17-90`) -/

#guard (keccakRC) =
    [0x8000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003, 0x8000000000008002,
     0x8000000000000080, 0x800a, 0x800000008000000a, 0x8000000080008081, 0x8000000000008080, 0x80000001,
     0x8000000080008008]

#guard (keccakRho) =
    [0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14]

#guard ((List.range 25).map keccakPi) =
    [0, 10, 20, 5, 15, 16, 1, 11, 21, 6, 7, 17, 2, 12, 22, 23, 8, 18, 3, 13, 14, 24, 9, 19, 4]

#guard ((keccakF1600 (Vector.replicate 25 0)).toList) =
    [0x8e5e5438b9a78617, 0xd9cd6a50f259d01e, 0x87b8e7c652a91f35, 0x1093e067cde4e0c5,
     0xb033ab90f2d95a45, 0xe0a72f72a8dd1a45, 0xc53780aa14672f9c, 0x3edd47f50051071d,
     0xb3a31d310c178acc, 0x79b586a59257aaa0, 0xbc4a7c3db3b1f99b, 0x68874063e68a6793,
     0x5c6c03332e0e2566, 0x9caa1202b9f030da, 0x5f3b9a782bcf7a9f, 0xe536c1e061ae7923,
     0x6de9b618b73c87ec, 0x2abed1f170918ac2, 0x6aabbd53daed24b7, 0xbfc1416a2c2ee15a,
     0xc6cfe036b90952af, 0x45503617dc7060d7, 0x625611b2c29f7ae4, 0xd43671db2c30647a,
     0xcffd0d76222ca01c]

#guard ((keccakF1600 sampleState).toList) =
    [0x7a6bcb3543c63cf2, 0x5c865f20e227846d, 0x5e0fe9067c99e021, 0x6d1c2bf1231d7c1,
     0xcb27ee646dba7bd3, 0xdb3b4650ec191b3d, 0xd913d92e7b9e5ee4, 0xbeb222617893db1f,
     0x23a4074872ebf1c6, 0xad8941c1403bf478, 0x76f1ed9d9ba2d63, 0x42d623b1725fb8f9,
     0xb40b644e69f2993, 0xabf41e1a7f095999, 0xc75216c3a9a02510, 0x6cebdfbc49b67934,
     0xe69c54a2e6adc7ac, 0x95f9a8f569d3805c, 0xe38abf8c72db5c97, 0x9104ee2dc00f0a41,
     0xcb954c30ac3c095, 0x7f650b1e24f9bbda, 0x99ae534516150f86, 0xa54642a1d434097a,
     0xa0cec1ca5ab3b503]

#guard ((keccak256 (sampleBytes 0)).toList) =
    [0x6dde3b94f02ddde3, 0x5cf35960c39ee382, 0x6bcc38dfe22067d7, 0x4a3aaafcdd9fb610]

#guard ((keccak256 (sampleBytes 1)).toList) =
    [0x42dac62a0fcea04f, 0xd02c3116f9e011a4, 0xe9ce399598d27186, 0x7bfb3e3670b844c3]

#guard ((keccak256 (sampleBytes 135)).toList) =
    [0xb2d3f728b0bf7d9c, 0xe2909e3a4e4f7dac, 0x7eb388b814e7198, 0xc6ebecac8245475f]

#guard ((keccak256 (sampleBytes 136)).toList) =
    [0x4610a67dad6e8422, 0x2bed2fdf0bdbe065, 0x757bb473420342e2, 0xe6605238e022132]

#guard ((keccak256 (sampleBytes 137)).toList) =
    [0x20c4148028cdbcce, 0xdd4b7db9bc821988, 0xd3409eecbcde92f1, 0x9a8b705bc4b02235]

#guard ((keccak256 (sampleBytes 300)).toList) =
    [0x4da46d1c2e0238d1, 0xa61d0dd2292f0487, 0xb204225b95037ead, 0x3a4ef5710ffbe4c8]

/- `Sponge::duplex` from `sampleState`, and `sample`'s challenge -/
#guard (duplexProbe []) =
    ([0x108d09169002cf8c, 0x30f3e4dfaaca8f55, 0x11398df5f52e665, 0x1a7ff87b930ae94b,
      0xc5ce99e7a8b66ce, 0xf8935e938d5ef37, 0x97f7af8058c6d4f4, 0x62fb1e6e00c40caa,
      0x126e35f1c92b6b57, 0x4da52b4a8828fb96, 0x68abdadc16faeb0e, 0x91ae4fccfbbf5404,
      0xff12b3888409d192, 0x668d42e3a9ba449b, 0x723aac9d2782f350, 0xfd8fd92aa9346287,
      0x23ca46bbc00fc5df, 0x80fe5ecc642b80f3, 0x487b8fe5d533ba2a, 0x9f7722cc20c71d88,
      0x28f73f535d8c5bf4, 0x1839fcd6dafb756a, 0xcae43f8f5b5f0202, 0xf87120987ec258f3,
      0x3d9381fdc918324a], (e 0x108d09169002cf8c 0x30f3e4dfaaca8f55))

#guard (duplexProbe [(fe 800, 1), (fe 801, 4), (fe 802, 1)]) =
    ([0x59f168e3dcd1c1be, 0x4f6aeb3957ec1b36, 0x9d81bda8b6d8c9e, 0x39505b9cc72abaee,
      0x27e8ecbf2b0c76c1, 0x108f4ae1763665b7, 0x2ac6ad2d5bf18375, 0xf042aa0d77853fc9,
      0x9458987230f14712, 0x3911962867c50eed, 0xa0c4bb186d44228f, 0x24879ded96671097,
      0x723c9be0f5bfe949, 0x6bf2cdb6ae8dbc1, 0xd0b0cb9a9edec651, 0xdba900294f2ab56d,
      0xa2f3d2d80738a6c9, 0xcf5ada804ea3863b, 0x3532e627dc065355, 0x71ddf7a864906ab6,
      0xb2214c7556593b20, 0x24dd91989f2dd5a7, 0x4104e378d122b414, 0x55320e2da0c0a9c0,
      0x256ac97d8d0e532d], (e 0x59f168e3dcd1c1be 0x4f6aeb3957ec1b36))

#guard (duplexProbe [(fe 803, 1), (fe 804, 1), (fe 805, 4), (fe 806, 1)]) =
    ([0xf8154e6bc0b72f79, 0xfdea40120ca6f053, 0x74daca53b1ce0708, 0x73d2a4a8afd3581d,
      0x35da44fc1508cb47, 0xc430509828a04939, 0x14c2da3e2b06f1f1, 0x7bb36e3efcc6f7ea,
      0xed2f7300d6817e69, 0xd9662ef6b90f619f, 0x2d1a43635e29794c, 0x2e3dce13bd5a5fe7,
      0xce85a5c5a062f4aa, 0xefdde988a6bcdc32, 0xd4b3f51568650001, 0x76af7b56e23aa985,
      0xb915f0e8b176deb0, 0x4ec610579bf8bf82, 0xe7169ea4e8aed302, 0xf7887b04fa823fcb,
      0xfbebf61305d622c8, 0x41205aba39e6c5af, 0xd04ba76648b36a73, 0xcdfa720a094745b2,
      0xc6b3b81f32c2fbf2], (e 0xf8154e6bc0b72f79 0xfdea40120ca6f053))

/- `Sponge::absorb_nonce` after one waiting scalar, and after four (the nonce's absorb permutes
them first) -/
#guard (nonceProbe [(fe 808, 1)] (fe 809)) =
    (0xc269953c6e29c391,
     [0x27b22ffac5191fe6, 0xd4e062c09b725bda, 0xc269953c6e29c391, 0x205f792ac21fd8ad,
      0x99089432e9a5ebab, 0xfcfb2bbf1b53c02a, 0xc49133616e9c7bfc, 0x6618120aeb129956,
      0xd0c7e8e9d4fc0765, 0x76bfc196c8471a6b, 0x1553bf60e9ec067d, 0xba0a17d0a169d98b,
      0x8a19fa62d0ba6045, 0x15f432ced82f863e, 0xa13d761b3fcba43a, 0xd3781d58e67eff8,
      0x775ef218aebbcd1, 0xd3ba95f7fbc9b579, 0xa9708bb5f55ef289, 0x74e3af63954f9269,
      0x51754249806c8cdd, 0x3f859c1a4ae0c5c9, 0x94ba58db90c8b0a4, 0x877596e5d6c50ee2,
      0x2582e94b4c26675b], some (e 0x27b22ffac5191fe6 0xd4e062c09b725bda))

#guard (nonceProbe [(fe 810, 1), (fe 811, 1), (fe 812, 1), (fe 813, 1)] (fe 814)) =
    (0xc7306517eae3b827,
     [0x3c551bd42841b6e1, 0x4776b06e7878fdd4, 0xc7306517eae3b827, 0xc3a5a9ac95bc436f,
      0x7792bae946e8a064, 0x416c86105de64dc1, 0x825e33c66177d436, 0x7ba27dbe68389865,
      0xb7c2c727f5a1479e, 0xbdb9fecab69fc6cf, 0xda7634345d570955, 0x56f5b1dfd46b1d17,
      0xb1cf07dca15c9b5e, 0x532dc7b98f062031, 0x74fab35f46291278, 0xa0729c8b7a832ee8,
      0xd7c2c4e5a8256e0a, 0x290c8d8598dd9150, 0x792ae0d1063d922, 0x9ae76b5c7d1bd2c0,
      0xac534caef3bc62e1, 0x68e07292381d5011, 0x37642d61f483cf10, 0xe37d65cc4eca4c0d,
      0x1c7a0d86240ebbfe], some (e 0x3c551bd42841b6e1 0x4776b06e7878fdd4))

/-! ## The tag, the IV and the seed (`layout.rs:68`, `bytecode.rs:45-53`, `transcript.rs:36-49`) -/

#guard (versionTag) =
    [108, 101, 97, 110, 118, 109, 45, 118, 49]

#guard ((keccak256 (sampleBytecode.flatMap K.toBytes)).toList) =
    [0xeaf4a0c4747d0a15, 0xf911ff2de945bf19, 0x341f0c9e51a7eccf, 0x303f1bd4476aa97f]

#guard ((fiatShamirIv sampleBytecode).toList) =
    [0x6d721778f66e6403, 0x26fc756874958aa6, 0x94e32737534a3a2f, 0x29bbd84d1727a49e]

#guard (samplePublicInput.toList) =
    [0xdfcab5a08b76614c, 0x87725d48331e09f4, 0x2f1a05f0dbc6b19c, 0xd7c2ad98836e5944]

#guard ([(digestHalves samplePublicInput).1, (digestHalves samplePublicInput).2]) =
    [(e 0xdfcab5a08b76614c 0x87725d48331e09f4), (e 0x2f1a05f0dbc6b19c 0xd7c2ad98836e5944)]

#guard ((seed sampleBytecode samplePublicInput).toList) =
    [0xdc9398448325f766, 0xa677895b1d3355be, 0x3c392b0312556407, 0x2875e8ea0d6be3f0,
     0x42ec0e7d78cc1dea, 0xfc1a95a40ccda188, 0x47a4499af6881fc8, 0xee7ac5478d060ae5,
     0x64b20d83fe03ca4d, 0xb5b880e961d30b96, 0xff63ec7bee269d43, 0xc1b02f6754d889bc,
     0x6420ffdb081d4fef, 0xa633d7e6d7da4b76, 0xa16aa96a077c0b35, 0x82b42f1cf673a1b2,
     0xf859129fdc83f749, 0x68cfc1e34b95260f, 0x1b1849d3c57aaed7, 0xe2424be842d25825,
     0x596a833e328a2809, 0xf4d80415d128c2f0, 0x1b79713c2b2464f6, 0x24811e78dce92f5d,
     0x93fb99b7e539ac0f]

end LeanVMTests.Protocol.Verifier.Helpers
