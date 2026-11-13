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
abbrev vkOutSL : List Stmt := slotLanesT 689 702 703 1 1

/-- The committed outputs' dot product. -/
abbrev vkOutDL : List Stmt := dotLoadsT ((List.range 25).map (slotLaneV 703 1)) (.n 234) 30 886

theorem vkOut_ker :
    noneIn [.load (.n 699) (.n 697) 0] vkSt = true ∧
    vkOutSL.all isAtom = true ∧
    (resid (atomsL vkOutSL)).all (ldR [(.n 689, 0, 6), (.n 702, 0, 1)]) = true ∧
    noneIn (resid (atomsL vkOutSL)) vkSt = true ∧
    vkOutDL.all isAtom = true ∧
    (resid (atomsL vkOutDL)).all (ldR [(.n 234, 30, 55)]) = true ∧
    noneIn (resid (atomsL vkOutDL)) vkSt = true := by
  decide +kernel

/-- The inputs' pairs: the low point's weights and the last claim's `μ`. -/
abbrev vkInPD : List Stmt := pairsDotT (.n 978) (.n 161) (consecOffs 14 138 16) 1033

theorem vkIn_ker :
    noneIn [.load (.n 968) (.n 161) 132] vkSt = true ∧
    noneIn (eqtreeReads (.n 9) (.n 978) 39 979 4) vkSt = true ∧
    vkInPD.all isAtom = true ∧
    (resid (atomsL vkInPD)).all (ldR [(.n 978, 14, 30), (.n 161, 138, 154)]) = true ∧
    noneIn (resid (atomsL vkInPD)) vkSt = true := by
  decide +kernel

/-- A residual atom of the slices: a read at the cursor, or the store of a read. -/
def slRes (a : Atom) : Bool := match a with
  | .load d p off => (p == .n 1106 && decide (off < 64) && !vkSt.contains a) ||
      (p == .n 10 && decide (off < 64) && d == .n (slB 1113 off) && vkSt.contains a)
  | _ => false

theorem vkSlices_ker :
    (slicesT (.n 1106) 1108 1113).all isAtom = true ∧
    (resid (atomsL (slicesT (.n 1106) 1108 1113))).all slRes = true := by
  decide +kernel

/-- The inputs' lane weights. -/
abbrev vkFinSL : List Stmt := slotLanesT 1098 1514 1515 0 2

/-- The shifts of the slices. -/
abbrev vkFinFT : List Stmt := foldT (fun j ↦ 1749 + j) 1813 (.n 1748)

/-- `keccak_form`. -/
abbrev vkFinFA : List Stmt := (formAll (.n 2911) vkSh (slotLaneV 1515 2) 2912).1

theorem vkFin_ker :
    noneIn ([.load (.n 1513) (.n 161) 134, .load (.n 1748) (.n 161) 136,
      .load (.n 2911) (.n 161) 135] ++ (List.range 64).map fun z ↦ .load (.n (1749 + z)) (.n 10) z)
      vkSt = true ∧
    vkFinSL.all isAtom = true ∧
    (resid (atomsL vkFinSL)).all (ldR [(.n 1098, 0, 6), (.n 1514, 0, 1)]) = true ∧
    noneIn (resid (atomsL vkFinSL)) vkSt = true ∧
    vkFinFT.all isAtom = true ∧
    (resid (atomsL vkFinFT)).all (ldR [(.n 1748, 0, 6)]) = true ∧
    noneIn (resid (atomsL vkFinFT)) vkSt = true ∧
    vkFinFA.all isAtom = true ∧
    (resid (atomsL vkFinFA)).all (ldR [(.n 2911, 0, 25)]) = true ∧
    noneIn (resid (atomsL vkFinFA)) vkSt = true ∧
    (formAll (.n 2911) vkSh (slotLaneV 1515 2) 2912).2.2 = 3125 := by
  decide +kernel

end
end LeanVM.Guest.B1.VerifySub
