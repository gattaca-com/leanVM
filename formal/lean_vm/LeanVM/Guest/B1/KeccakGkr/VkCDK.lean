/-
  LeanVM.Guest.B1.KeccakGkr.VkCDK

  What the kernel checks of `verify_keccak`'s openings.
-/

module

public import LeanVM.Guest.B1.KeccakGkr.VkCCK

/-!
# The openings' blocks

`vkOutKer`: of the committed outputs' opening, the loads that are not stores and the residual
loads' pointers of its lane weights (`vkOutSL`) and their dot product (`vkOutDL`).
-/

namespace LeanVM.Guest.B1.VerifySub

open LeanVM.Parameters LeanVM.Semantics LeanVM.Guest.Core LeanVM.Guest.Value LeanVM.Guest.B1
open LeanVM.Guest.B1.Fwd

@[expose] public section

/-- The committed outputs' lane weights. -/
abbrev vkOutSL : List Stmt := slotLanesT 731 751 752 1 1

/-- The committed outputs' dot product. -/
abbrev vkOutDL : List Stmt := dotLoadsT ((List.range 25).map (slotLaneV 752 1)) (.n 276) 30 935

theorem vkOut_ker :
    noneIn [.load (.n 741) (.n 739) 0] vkSt = true ∧
    vkOutSL.all isAtom = true ∧ canonChk [] (atomsL ([] ++ vkOutSL)) = true ∧
    (resid (atomsL vkOutSL)).all (ldR [(.n 731, 0, 6), (.n 751, 0, 1)]) = true ∧
    noneIn (resid (atomsL vkOutSL)) vkSt = true ∧
    vkOutDL.all isAtom = true ∧ canonChk [] (atomsL ([] ++ vkOutDL)) = true ∧
    (resid (atomsL vkOutDL)).all (ldR [(.n 276, 30, 55)]) = true ∧
    noneIn (resid (atomsL vkOutDL)) vkSt = true := by
  decide +kernel

/-- The inputs' pairs: the low point's weights and the last claim's `μ`. -/
abbrev vkInPD : List Stmt := pairsDotT (.n 1027) (.n 203) (consecOffs 14 138 16) 1082

theorem vkIn_ker :
    noneIn [.load (.n 1017) (.n 203) 132] vkSt = true ∧
    noneIn (eqtreeReads (.n 9) (.n 1027) 39 1028 4) vkSt = true ∧
    vkInPD.all isAtom = true ∧ canonChk [] (atomsL ([] ++ vkInPD)) = true ∧
    (resid (atomsL vkInPD)).all (ldR [(.n 1027, 14, 30), (.n 203, 138, 154)]) = true ∧
    noneIn (resid (atomsL vkInPD)) vkSt = true := by
  decide +kernel

/-- The slices but the ring map's first challenge. -/
def slicesPre (cur : V) (capb b : ℕ) : List Stmt :=
  ((List.range 16).map fun g ↦
    (if g = 0 then [] else
      [keccakA (slCells b (g - 1) ++ cap5 (slCap capb b (g - 1)))
        ((List.range 13).map fun k ↦ .n (slP b g + k))]) ++
    ((List.range 4).map fun j ↦
      fsNextT cur (4 * g + j) (slB b (4 * g + j)) ++
        [.atom (.load (.n (slB b (4 * g + j))) (.n 10) (4 * g + j))]).flatten).flatten ++
  [keccakA (slCells b 15 ++ cap5 (slCap capb b 15)) ((List.range 13).map fun k ↦ .n (b + 835 + k))]

theorem slicesT_pre (cur : V) (capb b : ℕ) :
    slicesT cur capb b = slicesPre cur capb b ++ challengeT (.n (b + 835)) (.n (b + 836)) (b + 848) :=
  rfl

/-- The canonical values of the slices' permutations' inputs: the capacity, the reads' cells. -/
def slA : List ℕ :=
  [1157, 1158, 1159, 1160, 1161] ++ (List.range 64).flatMap fun z ↦ [slB 1162 z + 7, slB 1162 z + 9]

/-- A residual atom of the slices: a read at the cursor, or the store of a read. -/
def slRes (a : Atom) : Bool := match a with
  | .load d p off => (p == .n 1155 && decide (off < 64) && !vkSt.contains a) ||
      (p == .n 10 && decide (off < 64) && d == .n (slB 1162 off) && vkSt.contains a)
  | _ => false

theorem vkSlices_ker :
    (slicesPre (.n 1155) 1157 1162).all isAtom = true ∧
    canonChk slA (atomsL ([] ++ slicesPre (.n 1155) 1157 1162)) = true ∧
    (resid (atomsL (slicesPre (.n 1155) 1157 1162))).all slRes = true := by
  decide +kernel

/-- The inputs' lane weights. -/
abbrev vkFinSL : List Stmt := slotLanesT 1147 2018 2019 0 2

/-- The shifts of the slices. -/
abbrev vkFinFT : List Stmt := foldT (fun j ↦ 2253 + j) 2317 (.n 2252)

/-- `keccak_form`. -/
abbrev vkFinFA : List Stmt := (formAll (.n 3415) vkSh (slotLaneV 2019 2) 3416).1

theorem vkFin_ker :
    noneIn ([.load (.n 2017) (.n 203) 134, .load (.n 2252) (.n 203) 136,
      .load (.n 3415) (.n 203) 135] ++ (List.range 64).map fun z ↦ .load (.n (2253 + z)) (.n 10) z)
      vkSt = true ∧
    vkFinSL.all isAtom = true ∧ canonChk [] (atomsL ([] ++ vkFinSL)) = true ∧
    (resid (atomsL vkFinSL)).all (ldR [(.n 1147, 0, 6), (.n 2018, 0, 1)]) = true ∧
    noneIn (resid (atomsL vkFinSL)) vkSt = true ∧
    vkFinFT.all isAtom = true ∧ canonChk [] (atomsL ([] ++ vkFinFT)) = true ∧
    (resid (atomsL vkFinFT)).all (ldR [(.n 2252, 0, 6)]) = true ∧
    noneIn (resid (atomsL vkFinFT)) vkSt = true ∧
    vkFinFA.all isAtom = true ∧ canonChk [] (atomsL ([] ++ vkFinFA)) = true ∧
    (resid (atomsL vkFinFA)).all (ldR [(.n 3415, 0, 25)]) = true ∧
    noneIn (resid (atomsL vkFinFA)) vkSt = true ∧
    (formAll (.n 3415) vkSh (slotLaneV 2019 2) 3416).2.2 = 3629 := by
  decide +kernel

end
end LeanVM.Guest.B1.VerifySub
