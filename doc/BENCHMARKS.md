# Benchmarks for the 64 -> 1 merge

Goal: merge 64 aggregations in under 1 s. On 16 cores the tree is 64 -> 8 -> 1:
8 concurrent 8:1 nodes on 2 cores each (target <= 0.7 s per node), then one 8:1 node on 16 cores (target <= 0.25 s).

## Build

```sh
cargo build --release -p leanvm-cli
```

The first run of a recursion bench proves its children and caches them under `target/children-cache`
(minutes for `--n 8`; run it once with many threads). Later runs load them.
`flux-profiler` is a git dependency on a private repo; cargo fetches it with your git credentials.

## Shapes we track

All proving times are warm: the bench runs one unmeasured pass first.
`LEANVM_NUM_THREADS` sets the prover's thread count; pin cores where the OS allows (`taskset` on Linux).

| Shape | Command |
|---|---|
| 2:1 node, 1 thread | `LEANVM_NUM_THREADS=1 target/release/leanvm --repeat 5 --cooldown 0 --log-inv-rate 2 --tracing recursion --n 2 --xmss-per-leaf 900` |
| 8:1 node, 2 threads | `LEANVM_NUM_THREADS=2 target/release/leanvm --repeat 5 --cooldown 0 --log-inv-rate 2 --tracing recursion --n 8 --xmss-per-leaf 900` |
| 8:1 node, all cores | `LEANVM_NUM_THREADS=<cores> target/release/leanvm --repeat 5 --cooldown 0 --log-inv-rate 2 --tracing recursion --n 8 --xmss-per-leaf 900` |
| 8 concurrent 8:1 nodes, 2 threads each | start 8 copies of the 2-thread 8:1 command at once (`--repeat 3`), each on its own 2 cores; report each node's proving time |

`--tracing` prints the stage tree (Commit, Prove bus, Prove constraints, Flock reduction, PCS open, ...).
`--cold` times a fresh process's first proof instead.

## Correctness

Every optimisation must keep the proof byte-identical. Dump the proof and compare its hash:

```sh
LEANVM_NUM_THREADS=16 target/release/leanvm --repeat 1 --cooldown 0 --log-inv-rate 2 recursion --n 2 --xmss-per-leaf 900 --proof-out p2.bin
LEANVM_NUM_THREADS=16 target/release/leanvm --repeat 1 --cooldown 0 --log-inv-rate 2 recursion --n 8 --xmss-per-leaf 900 --proof-out p8.bin
sha256sum p2.bin p8.bin   # macOS: shasum -a 256
```

| Proof | sha256 (same as upstream main 424220016) |
|---|---|
| `--n 2` | `9a37903d85c562261cb6f014fe1dfac5dd335ac6975e69062ddf358a6656d0b5` |
| `--n 8` | `a1d14072f5240a4270a3cfd38e229324c8ae78b9766a90de8a2e8f493c6445a6` |

The proof does not depend on the thread count or the machine (checked at 1 and 16 threads).

## Profiling (flux traces)

The prover is instrumented for [flux](https://github.com/gattaca-com/flux)'s `flux-profiler`: every stage span and every
`#[timed]` function becomes a frame, per thread. Install its CLI once (same revision as the dependency in `Cargo.toml`):

```sh
cargo install --locked --git https://github.com/gattaca-com/flux --rev cb55dcb7fd31d525b09317f98ed0e47fac4ea60a flux-profiler
```

Record a trace (the script builds `leanvm`, starts it with `--flux`, attaches the CLI and writes `<out>.fxt`):

```sh
LEANVM_NUM_THREADS=4 TASKSET=0-3 scripts/flux-trace.sh n8-4core.fxt -- \
  --repeat 2 --cooldown 0 --log-inv-rate 2 --tracing recursion --n 8 --xmss-per-leaf 900
```

- `TASKSET` pins cores with Linux `taskset`; leave it unset elsewhere. `FLUX_CLI=<path>` overrides the CLI if it is not on `PATH`.
- A trace holds `--repeat` + 1 proofs: a warm-up pass, then the measured passes. Read a warm `Prove` (the 2nd or later),
  and check that the warm passes agree before trusting a span.
- Hardware counters per frame (Linux, `kernel.perf_event_paranoid <= 2`): prefix `PERF_EVENTS=instructions,cycles,l1d-misses,l2-misses,l3-misses`.
- Open the `.fxt` in [Perfetto](https://ui.perfetto.dev) (drag and drop). Frames carry allocated and freed bytes as a `memory` track.
- Flux overhead is within noise for our spans; the `--tracing` span tree printed by the run is a cheaper summary.

## Reference numbers (Ryzen 9 9950X, 16 cores, 2-channel DDR5, merge-main 2026-10-27)

| Shape | Proving time | Peak memory |
|---|---|---|
| 2:1, 1 thread | 0.563 s | 1.79 GiB |
| 8:1, 2 threads | 1.44 s | 3.02 GiB |
| 8:1, 16 threads | 0.55-0.60 s | 3.15 GiB |
| 8 concurrent 8:1 x 2 threads | 3.75-3.93 s per node | 3.02 GiB per node |

On this machine the concurrent shape is bound by DRAM bandwidth (~55 GB/s): an 8:1 proof moves ~24.7 GB
(15.3 read, 9.4 written). Machines with more memory bandwidth per core (e.g. Apple M-series) change which
stages bound the time, so report per-stage spans from `--tracing` alongside the totals.
