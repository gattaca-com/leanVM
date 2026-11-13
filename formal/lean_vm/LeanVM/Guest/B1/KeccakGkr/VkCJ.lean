/-
  LeanVM.Guest.B1.KeccakGkr.VkCJ

  What the kernel checks of `verify_keccak`'s body as a flat function: its statements, buffers,
  calls.
-/

module

public import LeanVM.Guest.B1.KeccakGkr.VkCCK.Steps
public import LeanVM.Guest.B1.KeccakGkr.VkHolds
public import LeanVM.Guest.B1.Tables.FlatAt
public import LeanVM.Guest.Data.NameStrings

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
def vkVs : List V := [.n 3132, .n 3133, .n 3134, .n 3135, .n 3136, .n 3139, .n 3140]

/-- The body but its return. -/
def vkPre : List Stmt := vkHeadT ++ vkFirstT ++ vkStepsT ++ vkOutT ++ vkInT ++ vkFinPre

theorem vkFinT_pre : vkFinT = vkFinPre ++ [.ret vkVs] := stmtsEqb_sound (by decide +kernel)

theorem vkT_pre : vkT = vkPre ++ [.ret vkVs] := by
  rw [vkT, vkPre, vkFinT_pre]; simp only [List.append_assoc]

/-- The buffers. -/
def vkAllocs : List (V × (ℕ ⊕ V)) :=
  [(.n 61, .inr (.n 60)), (.n 87, .inl 5), (.n 161, .inl 154), (.n 163, .inl 30), (.n 234, .inl 62),
    (.n 342, .inl 2), (.n 978, .inl 30)]

/-- Not a dispatch. -/
def vkNoDispB : Stmt → Bool
  | .dispatch .. => false
  | _ => true

/-- Step `s`'s call's arguments and returns. -/
def stepCall (s : ℕ) : Option V × String × List V × List V :=
  let T := stepT0 s (stepB s)
  if s = 0 then
    (none, layerName 1, (stepCap s).map .n ++ [.n 159, .n 160, .n (T + 1), .n (T + 3), .n 34,
      .n 48, .n 161, .n (T + 5), .n (T - 1), .n 11], (List.range 6).map fun j ↦ .n (T + 6 + j))
  else
    (none, layerName 0, (stepCap s).map .n ++ [.n T, .n (T + 1), .n (stepCur s), .n (T + 3),
      .n 34, .n 48, .n (T + 5), .n (T + 7), .n (T - 1), .n 11], (List.range 6).map fun j ↦ .n (T + 8 + j))

/-- **The calls.** -/
def vkCalls : List (Option V × String × List V × List V) :=
  [(some (.n 63), name% «verify_keccak_loop0», [.n 64, .n 18, .n 61, .n 66], [])] ++
    (List.range 6).map stepCall ++
    [(none, "keccak_plain_rounds__L6_L0", [.n 683, .n 684, .n 685, .n 686, .n 687, .n 690,
        .n 691, .n 688, .n 158, .n 689, .n 11], (List.range 7).map fun j ↦ .n (692 + j)),
      (some (.n 962), name% «verify_keccak_loop1», [.n 963, .n 18, .n 965, .n 61], []),
      (none, "keccak_plain_rounds__L4_L1", [.n 692, .n 693, .n 694, .n 695, .n 696, .n 700,
        .n 701, .n 967, .n 968, .n 970, .n 11], (List.range 7).map fun j ↦ .n (971 + j)),
      (none, "keccak_plain_rounds__L6_L0", [.n 971, .n 972, .n 973, .n 974, .n 975, .n 1099,
        .n 1100, .n 976, .n 977, .n 1098, .n 11], (List.range 7).map fun j ↦ .n (1101 + j)),
      (some (.n 3128), name% «verify_keccak_loop2», [.n 3129, .n 18, .n 3131, .n 1513], [])]

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
