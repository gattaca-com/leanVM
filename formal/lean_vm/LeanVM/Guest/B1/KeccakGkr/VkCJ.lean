/-
  LeanVM.Guest.B1.KeccakGkr.VkCJ

  What the kernel checks of `verify_keccak`'s body as a flat function: its statements, buffers,
  calls.
-/

module

public import LeanVM.Guest.B1.KeccakGkr.VkCI
public import LeanVM.Guest.B1.KeccakGkr.VkHolds
public import LeanVM.Guest.B1.Tables.FlatAt

/-!
# The body, flat

`vkT = vkPre ++ [.ret vkVs]` (`vkT_pre`). `vkPre_kernel`: its statements are flat, it takes no
frame addresses, its seven buffers (`vkAllocs`); it dispatches nothing, so its calls are `vkCalls`
(`vkPre_callSF`).
-/

namespace LeanVM.Guest.B1.VerifySub

open LeanVM.Parameters LeanVM.Semantics LeanVM.Guest.Core LeanVM.Guest.Value LeanVM.Guest.B1
open LeanVM.Guest.Memory
open LeanVM.Protocol LeanVM.Protocol.KeccakHash LeanVM.Protocol.Compile LeanVM.Protocol.Verifier

@[expose] public section

/-- The inputs' check and the return but the return. -/
def vkFinPre : List Stmt := vkFinT.dropLast

/-- The returned values. -/
def vkVs : List V := [.n 3636, .n 3637, .n 3638, .n 3639, .n 3640, .n 3643, .n 3644, .n 3645]

/-- The body but its return. -/
def vkPre : List Stmt := vkHeadT ++ vkFirstT ++ vkStepsT ++ vkOutT ++ vkInT ++ vkFinPre

theorem vkFinT_pre : vkFinT = vkFinPre ++ [.ret vkVs] := stmtsEqb_sound (by decide +kernel)

theorem vkT_pre : vkT = vkPre ++ [.ret vkVs] := by
  rw [vkT, vkPre, vkFinT_pre]; simp only [List.append_assoc]

/-- The buffers. -/
def vkAllocs : List (V × (ℕ ⊕ V)) :=
  [(.n 60, .inr (.n 59)), (.n 92, .inl 5), (.n 203, .inl 154), (.n 205, .inl 30), (.n 276, .inl 62),
    (.n 384, .inl 2), (.n 1027, .inl 30)]

/-- Not a dispatch. -/
def vkNoDispB : Stmt → Bool
  | .dispatch .. => false
  | _ => true

/-- Step `s`'s call's arguments and returns. -/
def stepCall (s : ℕ) : Option V × String × List V × List V :=
  let T := stepT0 s (stepB s)
  if s = 0 then
    (none, layerName 1, (stepCap s).map .n ++ [.n 200, .n 202, .n (T + 1), .n (T + 3), .n 33,
      .n 47, .n 203, .n (T + 5), .n (T - 1)], (List.range 6).map fun j ↦ .n (T + 6 + j))
  else
    (none, layerName 0, (stepCap s).map .n ++ [.n T, .n (T + 1), .n (stepCur s), .n (T + 3),
      .n 33, .n 47, .n (T + 5), .n (T + 7), .n (T - 1)], (List.range 6).map fun j ↦ .n (T + 8 + j))

/-- **The calls.** -/
def vkCalls : List (Option V × String × List V × List V) :=
  [(some (.n 62), "__loop10", [.n 63, .n 17, .n 60, .n 65], [])] ++
    (List.range 6).map stepCall ++
    [(none, "keccak_plain_rounds__L6_L0", [.n 725, .n 726, .n 727, .n 728, .n 729, .n 732,
        .n 733, .n 730, .n 193, .n 731], (List.range 7).map fun j ↦ .n (734 + j)),
      (some (.n 1011), "__loop11", [.n 1012, .n 17, .n 1014, .n 60], []),
      (none, "keccak_plain_rounds__L4_L1", [.n 734, .n 735, .n 736, .n 737, .n 738, .n 748,
        .n 750, .n 1016, .n 1017, .n 1019], (List.range 7).map fun j ↦ .n (1020 + j)),
      (none, "keccak_plain_rounds__L6_L0", [.n 1020, .n 1021, .n 1022, .n 1023, .n 1024, .n 1148,
        .n 1149, .n 1025, .n 1026, .n 1147], (List.range 7).map fun j ↦ .n (1150 + j)),
      (some (.n 3632), "__loop12", [.n 3633, .n 17, .n 3635, .n 2017], [])]

theorem vkPre_kernel :
    vkPre.all Stmt.flat = true ∧ vkPre.filterMap Main.Stmt.addrS = [] ∧
    vkPre.filterMap Main.Stmt.allocSz = vkAllocs ∧ vkPre.all vkNoDispB = true ∧
    vkPre.filterMap Main.Stmt.callS = vkCalls := by
  decide +kernel

theorem vk_callSF_of_noDisp (σ : Val) {s : Stmt} (h : vkNoDispB s = true) :
    s.callSF σ = Main.Stmt.callS s := by
  cases s with
  | dispatch x fs args rets => simp [vkNoDispB] at h
  | _ => rfl

theorem vkPre_callSF (σ : Val) : vkPre.filterMap (Stmt.callSF σ) = vkCalls := by
  rw [← vkPre_kernel.2.2.2.2]
  exact List.filterMap_congr fun _ hs ↦
    vk_callSF_of_noDisp σ (List.all_eq_true.mp vkPre_kernel.2.2.2.1 _ hs)

theorem vkPre_disp (σ : Val) : ∀ s ∈ vkPre, DispOk σ s := by
  intro s hs
  have h := List.all_eq_true.mp vkPre_kernel.2.2.2.1 s hs
  cases s with
  | dispatch x fs args rets => simp [vkNoDispB] at h
  | _ => trivial

end
end LeanVM.Guest.B1.VerifySub
