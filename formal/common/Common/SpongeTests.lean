import Common.Sponge

/-! The twelve-round sponge of `Common/Sponge.lean` on vectors from leanVM's `hash.rs`, evaluated: a file
of its own, outside the module system, since a module cannot evaluate its own definitions. -/

namespace Common

/-! Vectors from `primitives::hash::hash` over `Keccak-p[1600, 12]`, on each side of a block
boundary. Each input is the byte ramp `i mod 256`; each digest was independently cross-checked
by absorbing its padded blocks with the `keccak` crate's `with_p1600::<12>`. -/

#guard hexOfBytes (keccak256 (countingBytes 0)).toList ==
  "e3dd2df0943bde6d82e39ec36059f35cd76720e2df38cc6b10b69fddfcaa3a4a"
#guard hexOfBytes (keccak256 (countingBytes 1)).toList ==
  "73ebf1d543d855a3c5e4be6322f75604c254f70394b396884b6010fcca694722"
#guard hexOfBytes (keccak256 (countingBytes 135)).toList ==
  "1d293c78401fd0d2d748e1e001aa9b8f63e6b5448361525727ce6fd1934c3015"
#guard hexOfBytes (keccak256 (countingBytes 136)).toList ==
  "7312c13db895e18f65174d26a0adf8e3fdcf4983069751aea26a55d4d355549a"
#guard hexOfBytes (keccak256 (countingBytes 137)).toList ==
  "3bb0daeba2195cc89de7b33ea0ec50136c5ad7aa2b117a4f2b6ac1a853040fd1"
#guard hexOfBytes (keccak256 (countingBytes 272)).toList ==
  "19ac667ee9333b746bdd231f429bb2a029c83f926db5147129116e4755f1ef3d"

end Common
