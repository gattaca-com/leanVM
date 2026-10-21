<h1 align="center">leanVM</h1>

<p align="center">
  <img src="./doc/images/banner.svg" alt="leanVM">
</p>

<h3 align="center">minimal hash-based zkVM, for post-quantum Ethereum</h3>

<p align="center">
  <a href="https://github.com/leanEthereum/leanVM/releases/download/doc-latest/leanVM.pdf"><img src="https://img.shields.io/badge/Documentation-PDF-blue?style=for-the-badge&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0xNCAySDZjLTEuMSAwLTIgLjktMiAydjE2YzAgMS4xLjg5IDIgMS45OSAySDE4YzEuMSAwIDItLjkgMi0yVjhsLTYtNnpNOC41IDE0LjVoMS4yNWMuOTcgMCAxLjc1LS43OCAxLjc1LTEuNzVTMTAuNzIgMTEgOS43NSAxMUg3LjV2Nmgxdi0yLjV6bTAtMVYxMmgxLjI1Yy40MSAwIC43NS4zNC43NS43NXMtLjM0Ljc1LS43NS43NUg4LjV6bTUuNSAzLjVoMnYtMWgtMnYtMWgydi0xaC0ydi0xLjVjMC0uMjguMjItLjUuNS0uNUgxN3YtMWgtMmMtLjgzIDAtMS41LjY3LTEuNSAxLjVWMTd6TTEzIDlWMy41TDE4LjUgOUgxM3oiLz48L3N2Zz4=" alt="Documentation"></a>
  <a href="./python-verifier/verifier.py"><img src="https://img.shields.io/badge/verifier-python-yellow?style=for-the-badge&logo=python&logoColor=white" alt="Python verifier"></a>
</p>

<table align="center">
  <tr>
    <td><a href="#leanxmss-aggregation">leanXMSS aggregation</a></td>
    <td align="right"><b>1.49K/s</b></td>
  </tr>
  <tr>
    <td><a href="#leansphincs-aggregation">leanSPHINCS aggregation</a></td>
    <td align="right"><b>633/s</b></td>
  </tr>
  <tr>
    <td><a href="#data-availability">leanDA commitment</a></td>
    <td align="right"><b>1.4 MiB/s</b></td>
  </tr>
</table>
<table align="center">
  <tr>
    <td><a href="#recursion">2-to-1 recursion</a></td>
    <td align="right"><b>0.40s</b></td>
  </tr>
  <tr>
    <td><a href="#hashing">hash calls</a></td>
    <td align="right"><b>960K/s</b></td>
  </tr>
  <tr>
    <td><a href="#fibonacci">cheap cycles</a></td>
    <td align="right"><b>6,2M/s</b></td>
  </tr>
</table>

**This branch** replaces BLAKE2s by Haraka-512 v2 throughout: the VM opcode, the Flock circuit, the
transcript, the Merkle trees, both signature schemes, LeanDA and the recursion guest. The numbers above
and below are measured against `main` on the same machine; see [hashing](#hashing) for why.

## security

leanVM is designed for security:

 * 128-bit ROM (64-bit QROM) soundness
 * no proximity gap conjecture
 * end-to-end formal verification
 * a traditional hash function

**warning**: Formal verification is [in progress](https://github.com/Verified-zkEVM/leanerVM). leanVM is not (yet) production ready.

## work in progress

Expect leanVM to change significantly:

* **hash**: Haraka-512 v2 on this branch (BLAKE2s on `main`). SHA2, SHA3, BLAKE3 are actively considered.
* **ISA**: A migration from leanISA to RISC-V (rv64im) is planned.
* **zk**: Support for zero-knowledge is planned.

**note**: Prior to binary fields leanVM used [KoalaBear](https://crates.io/crates/p3-koala-bear) and [Poseidon](https://eprint.iacr.org/2019/458). The historical design is in [this branch](https://github.com/leanEthereum/leanVM/tree/koalabear).

## benchmarks

**machine**: AMD Ryzen 9 9950X (16 cores, 32 threads, 60 GB RAM), `main` at `3a37fb6e` against this branch,
back to back.

| benchmark | `main` (BLAKE2s) | this branch (Haraka-512) | |
| --- | --- | --- | --- |
| `aggregate --xmss 1100` | 1.149 s, 957 sig/s, witness 2^26.52 | 0.739 s, 1,488 sig/s, witness 2^25.88 | 1.55× |
| `aggregate --xmss 1290` | 1.270 s, 1,016 sig/s, witness 2^26.67 | 1.356 s, 952 sig/s, witness 2^26.76 | 0.94× |
| `aggregate --sphincs 650` | 1.246 s, 522 sig/s, witness 2^26.61 | 1.026 s, 633 sig/s, witness 2^26.18 | 1.21× |
| `aggregate --sphincs 850` | 1.379 s, 616 sig/s, witness 2^26.78 | 1.424 s, 597 sig/s, witness 2^26.80 | 0.97× |
| `aggregate --blobs 16` | 1.425 s, 1.40 MiB/s | 1.469 s, 1.36 MiB/s | 0.97× |
| `recursion --n 2 --xmss-per-leaf 900` | 0.388 s | 0.400 s | 0.97× |
| verify (one thread, scalar) | 4.3 ms | 14 ms | |
| hash batch, 262,144 calls | 530K compressions/s | 960K calls/s | 1.81× |

On a c7i.4xlarge (Sapphire Rapids, 16 vCPU), the same binaries:

| benchmark | `main` (BLAKE2s) | this branch (Haraka-512) | |
| --- | --- | --- | --- |
| `aggregate --xmss 1100` | 1.735 s, 634 sig/s | 1.034 s, 1,064 sig/s | 1.68× |
| `aggregate --xmss 1290` | 1.884 s, 685 sig/s | 1.835 s, 703 sig/s | 1.03× |
| `aggregate --sphincs 650` | 1.837 s, 354 sig/s | 1.354 s, 480 sig/s | 1.36× |
| `recursion --n 2 --xmss-per-leaf 900` | 0.590 s | 0.581 s | 1.02× |

What the table says. A Haraka call proves for half of a BLAKE2s compression (a padded `2^13`-bit block
against `2^14`), but a signature takes more calls: 234 against 201 for leanXMSS and 394 against 307 for
leanSPHINCS, because the chain absorbs 32 bytes a call where BLAKE2s absorbs 64 (a 58-chain-end leaf is 31
calls against 15 compressions). And the hash table is padded to a power of two: at 1,290 leanXMSS the
304K calls spill past `2^18` into a `2^19` table while 260K compressions fit `2^18`, which is the 0.94×;
at 1,100 both fit and the committed witness falls by 1.57×. So the gain is 1.5× on a leanXMSS aggregate
that fills its table, and nothing on recursion and LeanDA, whose cycles are field arithmetic. A packed
block would recover the rest: a call commits 5,889 bits of its 8,192, where BLAKE2s commits 16,000 of
16,384.

### leanXMSS aggregation

The leanXMSS parameters are specified in [leanXMSS.pdf](https://github.com/leanEthereum/leanVM/releases/download/doc-latest/leanXMSS.pdf), with a [(ROM) security proof in Lean 4](formal/lean_xmss/LeanXmss/Statement.lean). **This is not the definitive parameter set.**

```bash
cargo leanvm aggregate --xmss 1100 --log-inv-rate 1 --repeat 3
```

```
aggregation, 1,100 leanXMSS signatures
  cycles (VM steps)           : 1,176,354 = 2^20.166
    details                   : HARAKA 2^17.986 (22.1%)  SET 2^17.884 (20.6%)  MUL 2^17.802 (19.4%)  XOR 2^17.625 (17.2%)  DEREF 2^17.236 (13.1%)  JUMP 2^16.456 (7.6%)  MEMORY 2^20.948  BYTECODE 2^17.792  TOTAL_COMMITTED 2^25.877
  proof size                  : 322.7 KiB
  proving time                : 0.739 s ± 2.0%      peak memory 6.662 GiB
  per signature               : 1,487.613 signatures/s
  verifying                   : 13.762 ms
```

### leanSPHINCS aggregation

The leanSPHINCS parameters are specified in [leanSPHINCS.pdf](https://github.com/leanEthereum/leanVM/releases/download/doc-latest/leanSPHINCS.pdf), with a [(ROM) security proof in Lean 4](formal/lean_sphincs/README.md). **This is not the definitive parameter set.**

```bash
cargo leanvm aggregate --sphincs 650 --log-inv-rate 1 --repeat 3
```

```
aggregation, 650 leanSPHINCS signatures
  cycles (VM steps)           : 1,403,635 = 2^20.421
    details                   : SET 2^18.315 (23.2%)  XOR 2^18.305 (23.1%)  HARAKA 2^17.974 (18.3%)  MUL 2^17.739 (15.6%)  DEREF 2^17.585 (14.0%)  JUMP 2^16.303 (5.8%)  MEMORY 2^21.044  BYTECODE 2^17.792  TOTAL_COMMITTED 2^26.18
  proof size                  : 327.9 KiB
  proving time                : 1.026 s ± 1.4%      peak memory 8.855 GiB
  per signature               : 633.384 signatures/s
  verifying                   : 14.247 ms
```

### data availability

```bash
cargo leanvm aggregate --blobs 16 --log-inv-rate 1 --repeat 3
```

```
aggregation, 16 blobs
  cycles (VM steps)           : 2,980,599 = 2^21.507
    details                   : MUL 2^19.897 (32.8%)  XOR 2^19.79 (30.4%)  DEREF 2^19.136 (19.3%)  JUMP 2^18.299 (10.8%)  HARAKA 2^17.295 (5.4%)  SET 2^15.23 (1.3%)  MEMORY 2^21.725  BYTECODE 2^17.792  TOTAL_COMMITTED 2^26.704
  proof size                  : 322.2 KiB
  proving time                : 1.469 s ± 1.4%      peak memory 12.234 GiB
  blob throughput             : 10.895 blobs/s, 1.362 MiB/s
  verifying                   : 40.651 ms
```

### recursion

```bash
cargo leanvm recursion --n 2 --xmss-per-leaf 900 --log-inv-rate 2 --repeat 3
```

```
recursion 2→1, over leaves of 900 leanXMSS signatures
  cycles (VM steps)           : 645,305 = 2^19.3
    details                   : MUL 2^18.099 (43.5%)  DEREF 2^17.146 (22.5%)  XOR 2^16.692 (16.4%)  SET 2^15.949 (9.8%)  JUMP 2^14.654 (4.0%)  HARAKA 2^14.582 (3.8%)  MEMORY 2^19.639  BYTECODE 2^17.792  TOTAL_COMMITTED 2^24.508
  proof size                  : 199.9 KiB
  proving time                : 0.4 s ± 1.5%      peak memory 7.436 GiB
  verifying                   : 10.891 ms
```

### hashing

Haraka-512 v2 is ten AES rounds on four 16-byte states with a dword mix after every second, a feed-forward
and a 32-byte truncation. The Flock circuit keeps its constraint matrices over GF(2^8) instead of GF(2): an
AES S-box is a field inverse `u = x^-1`, two degree-2 rows `x²·u = x`, `u²·x = u` on eight committed bits,
where its AND-gate circuit costs 32 rows and 32 bits; ShiftRows, MixColumns, the affine map and the round
constants stay symbolic. A call (64 bytes in, 32 out, 640 S-boxes) commits 5,888 bits and 1,536 rows against
a BLAKE2s compression's 16,000 bits and 16,384 rows (`flock::haraka`). The univariate-skip zerocheck, the
lincheck sumcheck, the ring switch, the PCS and the verifier are BLAKE2s's; only round 1 and the fold at `z`
read bytes rather than bits (GFNI kernels), and the lincheck closes on three matrices.

The VM proves one call per `HARAKA` opcode (`haraka(a, b, out)` in the zkDSL), and hashes any other length
as a chain of calls over 32-byte blocks behind the chaining value, closed by a length block
(`primitives::hash::hash`). A Fiat-Shamir step, a Merkle node, a chain step and a tree node are one call
each; a 64-byte input hashed whole is one call too, and a leaf over `n` chain ends is `n/2 + 2`.

```bash
BENCH_REPEAT=3 BENCH_COOLDOWN=2 FLOCK_N_LOG=18 cargo bench -p flock --bench hash_batch_haraka
```

```
Flock Haraka-512 batch proving, 262,144 calls (2^18 blocks)
  witness-gen                     :     35.8 ms ± 0.2%   11.6%
  commit                          :     46.0 ms ± 2.8%   14.9%
  zerocheck                       :     86.8 ms ± 3.4%   28.1%
  lincheck                        :      5.2 ms ± 6.0%    1.7%
  pcs opening                     :    135.2 ms ± 6.5%   43.8%
  other                           :      0.0 ms           0.0%
  ------------------------------------------
  prove TOTAL (witness excluded)  :    273.1 ms ± 3.1%   88.4%
  throughput                      :        959,807 calls/s ± 3.1%
```

On the same machine `main`'s `hash_batch` (BLAKE2s, 262,144 compressions) proves at 529,916 compressions/s:
commit 91 ms, zerocheck 137 ms, lincheck 11 ms, opening 255 ms. On a c7i.4xlarge (Sapphire Rapids, 16 vCPU)
the two are 578,667 calls/s against 276,445 compressions/s.

**Security.** Haraka v2's designers claim 256-bit (second-)preimage resistance of its short-input
compression functions, which is what the signature schemes rest on (SPHINCS+ instantiates `F` and `H` with
it); it is not a NIST standard, and the best published attacks reach about seven of the ten AES rounds. The
Merkle trees, the transcript and the chain use the call as a 2-to-1 compression function and need it
collision resistant, which is not among the designers' claims: that is an assumption this branch adds, and
the point to weigh against a standard hash. `python-verifier/verifier.py` is not ported and still verifies
the BLAKE2s protocol.

### Fibonacci

```bash
cargo leanvm fibonacci --n 2000000 --log-inv-rate 1 --repeat 3
```

```
Fibonacci (in the exponent, i.e. modulo 2^64 - 1), N = 2,000,000
  cycles (VM steps)           : 2,127,880
    details                   : MUL 2^20.944 (98.9%)  SET 2^13.288 (0.5%)  DEREF 2^12.967 (0.4%)  JUMP 2^10.968 (0.1%)  XOR 2^10.966 (0.1%)  MEMORY 2^20.96  BYTECODE 2^11.352  TOTAL_COMMITTED 2^25.26
  proof size                  : 286.0 KiB
  proving                     : 0.344 s ± 4.6%   6,181,087 cycles/s      peak memory 5.199 GiB
  verifying                   : 2.218 ms
```

## SNARK machinery

- 192-bit binary field (degree-3 tower over the 64-bit field)
- [WHIR](https://eprint.iacr.org/2024/1586) PCS, aka [Ligerito](https://eprint.iacr.org/2025/1187)
- [Flock](https://github.com/succinctlabs/flock/tree/main) hash proving
- [Binius](https://github.com/IrreducibleOSS/binius)/[Binius64](https://github.com/binius-zk/binius64) ring switching, M3 arithmetisation, and more (see [DP23](https://eprint.iacr.org/2023/1784) and [DP24](https://eprint.iacr.org/2024/504))
