/-
  LeanVM.Guest.Value.KeccakConst

  The guest's baked constants that are Keccak-f[1600] or Keccak-256 of fixed inputs, checked by
  kernel evaluation.
-/

module

public import LeanVM.Guest.Value.KeccakSponge
public import Statement.LeanVmGuest
public import LeanXmss.Scheme
public import Statement.LeanSphincs
public import Statement.Fast.Schemes
public import LeanDa.Scheme
public import Statement.Fast.Keccak
public import Statement.Fast.CubicExt
public import Statement.Fast.FieldWords
public import LeanVM.Facts.Verifier
public import LeanVM.Guest.Batch.Def
public import LeanDa.WithHash
public import LeanDa.Commitment
public import LeanVM.Guest.WithHash

/-!
# Baked Keccak constants

The host bakes into the guest (`placeholder_map`,
`crates/rec_aggregation/src/aggregation/placeholders.rs` at leanVM main `40f877dc`)
constants that are permutations or hashes of fixed inputs. Each is an equation between closed
terms, checked here by `decide +kernel` through the evaluator of `KeccakEval` (`keccakF_eq_iff`,
`keccak256_eq_iff`): no compiled code is trusted.

* `keccakF_zero`: Keccak-f[1600] of the zero state, the `KECCAK` table's padding row
  (`Table.padding`, the guest's `COORD_PADDING`);
* `batchSeed_eq`, `aggSeed_cells`: the batching transcript's initial state, whose five capacity
  cells are `AGG_SEED_0 .. AGG_SEED_4` (`FiatShamirState::from_label(RECURSION_AGG_LABEL)
  .capacity_cells()`, `aggregation/placeholders.rs:823-826`);
* `paddingCell_eq`, `paddingBlobRoot_eq`, `daPadBlob_cells`: LeanDA's padding blob root, the root
  of the tree over `128` copies of the zero cell's digest (`padding_root`, `padding_tree`,
  `crates/lean_da/src/commit.rs:70-78`), whose two cells are `DA_PAD_BLOB_0/1` (`hash_to_scalars`,
  `aggregation/placeholders.rs:841-847`).

-/

namespace LeanVM.Guest.Value.Keccak

open LeanVM.Parameters LeanVM.Semantics LeanVM.Protocol LeanVM.Protocol.KeccakHash
open LeanVM.Guest.Value.KeccakEval

@[expose] public section

/-! ## Keccak-f of the zero state -/

/-- Keccak-f[1600] of the zero state. -/
def keccakFZero : KeccakState :=
  #v[10258729603806234135, 15694317172351946782, 9779821429811388213, 1194545062623895749,
    12696680413525531205, 16187959555112376901, 14210968617165991836, 4529855917802456861,
    12944221850486409932, 8770063895056853664, 13567793431856675227, 7532059699614214035,
    6659701467325015398, 11288855218876330202, 6862248298150853279, 16516601852883532067,
    7920061636977395692, 3080130030073907906, 7686445356890793143, 13817397056013001050,
    14325915465326088879, 4994551462864380119, 7085870523181136612, 15291534770936243322,
    14987149935989530652]

/-- **Keccak-f[1600] of the zero state**, the `KECCAK` table's padding row's output. -/
theorem keccakF_zero : keccakF1600 zeroState = keccakFZero := by
  rw [keccakF_eq_iff]; decide +kernel

/-! ## The batching seed (`AGG_SEED_*`) -/

/-- Keccak-256 of the batching label. -/
def batchLabelDigest : Digest :=
  #v[5958355089351161392, 3895877486435481892, 13189585015277973799, 8155364092246439433]

/-- The batching label's bytes. -/
theorem batchLabel_eq : Node.batchingLabel = [108, 101, 97, 110, 118, 109, 47, 114, 101, 99, 117, 114,
    115, 105, 111, 110, 45, 97, 103, 103, 114, 101, 103, 97, 116, 105, 111, 110, 47, 118, 49] := by
  rw [Node.batchingLabel, show "leanvm/recursion-aggregation/v1" = String.ofList ['l', 'e', 'a', 'n',
      'v', 'm', '/', 'r', 'e', 'c', 'u', 'r', 's', 'i', 'o', 'n', '-', 'a', 'g', 'g', 'r', 'e', 'g',
      'a', 't', 'i', 'o', 'n', '/', 'v', '1'] from rfl,
    String.toUTF8_eq_toByteArray, String.toByteArray_ofList]
  decide +kernel

theorem keccak256_batchLabel : keccak256 Node.batchingLabel = batchLabelDigest := by
  rw [batchLabel_eq, keccak256_eq_iff]; decide +kernel

/-- The seed input of two digests, lane by lane. -/
theorem seedInput_eq (iv pi : Digest) :
    Compile.seedInput iv pi = #v[iv[0], iv[1], iv[2], iv[3], pi[0], pi[1], pi[2], pi[3],
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] := by
  apply Vector.ext
  intro i hi
  interval_cases i <;> simp [Compile.seedInput]

/-- The batching transcript's initial state. -/
def batchSeedValue : KeccakState :=
  #v[8863383324849452990, 9546771333105238862, 205209927688052035, 7173495066135316535,
    9874423646884049865, 8134311421070537496, 4158493316034147701, 8846457109692967598,
    14511325877726162742, 4060916719909660818, 12356624498398151658, 1496520730491542211,
    591654735395633709, 11960130085504622067, 12975416671444733998, 228518743143238297,
    5809815977160987393, 7584368478700930856, 8842225458955243668, 2574316089242637222,
    8009504894323071766, 3075439050930966178, 16802640550241062735, 13921879046025191513,
    17682358122322480425]

/-- **The batching seed's value.** -/
theorem batchSeed_eq : Node.batchSeed = batchSeedValue := by
  rw [Node.batchSeed, keccak256_batchLabel, seedInput_eq, keccakF_eq_iff]; decide +kernel

/-- `AGG_SEED_0 .. AGG_SEED_4` as the host bakes them (`dsl_u128` of the capacity cells). -/
def aggSeed : List ℕ :=
  [5809815977160987393, 163110270083386361275972183059004098344,
    147749286942701772590315303804891911078, 309954009992831123486648533959107595938,
    326181934902162170568833791729712308313]

/-- **`AGG_SEED_c` is cell `8 + c` of the batching seed** (even layout): the guest's
`seed = [AGG_SEED_0, …, AGG_SEED_4]` (`aggregate_claims`, `lean_ethereum.py:4686`) is the
capacity a `cap=seed` duplex step reads. -/
theorem aggSeed_cells (c : Fin 5) :
    LeanVM.Guest.litField (aggSeed.getD c 0) =
      cellsOfState Node.batchSeed false ⟨8 + c, by omega⟩ := by
  rw [batchSeed_eq]
  apply E.ext
  intro i
  fin_cases c <;> fin_cases i <;> rfl

/-! ## LeanDA's padding blob root (`DA_PAD_BLOB_*`) -/

/-- The digest of the zero cell. -/
def daPadCellDigest : Digest :=
  #v[17231807523556001412, 11334583167549852587, 4230190959671849756, 6800010031774260075]

/-- **LeanDA's padding cell digest.** -/
theorem paddingCell_eq : LeanDa.paddingCellWithHash Da.keccak256Bytes = LeanVM.Protocol.Digest.toBytes daPadCellDigest := by
  rw [LeanDa.paddingCellWithHash, Da.keccak256Bytes]
  refine congrArg LeanVM.Protocol.Digest.toBytes ?_
  rw [keccak256_eq_iff]; decide +kernel

/-- The zero codeword's blob root, level `7` of its tree. -/
def daPadBlobDigest : Digest :=
  #v[17635339603416681651, 5630583628039031207, 11942528933372280140, 15641231512148347094]

/-- Level `k` of the zero codeword's blob tree, `k < 7`: every node is the digest of two copies
of the level below; level `0` is the zero cell's digest. -/
def daPadLevel : ℕ → Digest
  | 0 => #v[17231807523556001412, 11334583167549852587, 4230190959671849756, 6800010031774260075]
  | 1 => #v[16671839831485496442, 13498090413648725542, 409056307263560780, 8037267510228062226]
  | 2 => #v[1238029063201015468, 5250604002934528125, 11454616167926117566, 16027911285484357013]
  | 3 => #v[17566753067139240467, 3504373812059925739, 8952767560965887101, 12328213688158902233]
  | 4 => #v[4304066973709982938, 10704535185091906026, 258983712954398263, 15783002668873081167]
  | 5 => #v[11139264634143931524, 2394727625377817916, 16414582794622146606, 14896498438981527146]
  | 6 => #v[9181592992930920266, 12614060863729478393, 7245103791816593521, 8499269302496010472]
  | _ => daPadBlobDigest

/-- One level up the zero codeword's blob tree. -/
theorem daPadLevel_succ : ∀ k < 7, keccak256 (digestBytes (daPadLevel k) ++ digestBytes (daPadLevel k)) =
    daPadLevel (k + 1) := by
  intro k hk
  interval_cases k <;> (rw [keccak256_eq_iff]; decide +kernel)

/-- The root of a tree over `2^k` copies of one leaf. -/
theorem merkleRoot_const (H : List UInt8 → Common.DigestBytes) (d : Common.DigestBytes) :
    ∀ k, LeanDa.merkleRootWithHash H (k + 1) (fun _ ↦ d) =
      LeanDa.merkleRootWithHash H k (fun _ ↦ H (LeanDa.pairBytes d d)) := fun _ ↦ rfl

theorem daPadLevel_toBytes_succ {k : ℕ} (hk : k < 7) :
    Da.keccak256Bytes (LeanDa.pairBytes (daPadLevel k).toBytes (daPadLevel k).toBytes) =
      (daPadLevel (k + 1)).toBytes := by
  rw [Da.keccak256Bytes, ← daPadLevel_succ k hk]
  rfl

/-- **LeanDA's padding blob root** (`padding_root`, `commit.rs:70-78`): the zero codeword's blob
root is `daPadBlobDigest`. -/
theorem paddingBlobRoot_eq :
    LeanDa.paddingBlobRootWithHash Da.keccak256Bytes = LeanVM.Protocol.Digest.toBytes daPadBlobDigest := by
  have hrep : List.replicate LeanDa.cellsPerRow (LeanDa.paddingCellWithHash Da.keccak256Bytes) =
      (List.range (2 ^ 7)).map fun _ ↦ (daPadLevel 0).toBytes := by
    rw [paddingCell_eq, List.map_const', List.length_range]; rfl
  rw [LeanDa.paddingBlobRootWithHash, hrep, LeanDa.treeRoot_map_range]
  rw [merkleRoot_const, daPadLevel_toBytes_succ (by norm_num), merkleRoot_const,
    daPadLevel_toBytes_succ (by norm_num), merkleRoot_const, daPadLevel_toBytes_succ (by norm_num),
    merkleRoot_const, daPadLevel_toBytes_succ (by norm_num), merkleRoot_const,
    daPadLevel_toBytes_succ (by norm_num), merkleRoot_const, daPadLevel_toBytes_succ (by norm_num),
    merkleRoot_const, daPadLevel_toBytes_succ (by norm_num)]
  rfl

/-- **`DA_PAD_BLOB_0`, `DA_PAD_BLOB_1`** (`aggregation/placeholders.rs:845-847`, `lean_ethereum.py:4360-4361`)
are the padding blob root's two cells. -/
theorem daPadBlob_cells :
    (LeanVM.Guest.word 17635339603416681651 5630583628039031207 0,
      LeanVM.Guest.word 11942528933372280140 15641231512148347094 0) =
      digestHalves daPadBlobDigest := by
  simp only [digestHalves, daPadBlobDigest, LeanVM.Guest.word]; rfl

end
end LeanVM.Guest.Value.Keccak
