/-
  LeanVM.Guest.B1.KeccakGkr.VkCCK

  `verify_keccak`'s steps: the kernel's checks of their cores, gathered.
-/

module

public import LeanVM.Guest.B1.KeccakGkr.VkCCK.K0
public import LeanVM.Guest.B1.KeccakGkr.VkCCK.K1
public import LeanVM.Guest.B1.KeccakGkr.VkCCK.K2

/-!
# The steps' cores, checked

`step_ker`: what the kernel checks of each of the six steps' cores (`stepKer`,
`VkCCK/Steps.lean`), checked two steps per module (`VkCCK/K0.lean` … `VkCCK/K2.lean`).
-/

namespace LeanVM.Guest.B1.VerifySub

@[expose] public section

theorem step_ker : ∀ s < 6, stepKer s (stepB s) := by
  intro s hs
  interval_cases s
  · exact step_ker_0
  · exact step_ker_1
  · exact step_ker_2
  · exact step_ker_3
  · exact step_ker_4
  · exact step_ker_5

end
end LeanVM.Guest.B1.VerifySub
