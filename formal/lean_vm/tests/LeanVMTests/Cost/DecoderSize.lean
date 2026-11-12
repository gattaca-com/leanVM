import LeanVM.Cost.DecoderSpec

/-!
# Tests: the guest decoder program's size

The compiled guest decoder's program (`LeanVM.Cost.compiledGuestWitnessDecoder`,
`Cost/DecoderSpec.lean`): its
exact size, `462862`, of which the guest's frame cells are `415313`, and its top's index;
`Cost/DecoderSize.lean` proves the size at most
`461876 + 2·(decTopC.costIdx - decTopC.costDeps.length)`, a conservative upper bound within
`guestWitnessDecoderSizeBound`, as `LeanVMTests/Cost/Efficient.lean` records the extractor's.
-/

namespace LeanVMTests.Cost.DecoderSize

open LeanVM.Cost LeanVM.Cost.Decoder

#guard compiledGuestWitnessDecoder.size == 462862
#guard decTopC.costIdx == 1058
#guard 461876 + 2 * (decTopC.costIdx - decTopC.costDeps.length) == 463130
#guard compiledGuestWitnessDecoder.size ≤ LeanVM.Node.guestWitnessDecoderSizeBound

end LeanVMTests.Cost.DecoderSize
