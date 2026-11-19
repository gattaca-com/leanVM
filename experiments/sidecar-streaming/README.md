# Streaming aggregation strategies for the leanVM sidecar

This folder shows how to write a strategy that aggregates votes as they arrive during a slot.
The strategy drives a leanVM sidecar and publishes one proof by a deadline `D`.

## Where the strategies are

All code is in [`crates/sidecar_streaming`](../../crates/sidecar_streaming).

| File | Content |
|---|---|
| [`src/strategies/simple.rs`](../../crates/sidecar_streaming/src/strategies/simple.rs) | The two example strategies, about 120 lines. Read and copy these. |
| [`src/strategy.rs`](../../crates/sidecar_streaming/src/strategy.rs) | The `Strategy` and `Ctx` traits a strategy implements and calls. |
| [`src/types.rs`](../../crates/sidecar_streaming/src/types.rs#L140) | `Event`, `Job` and the other types a strategy sees. |
| [`src/strategies/mod.rs`](../../crates/sidecar_streaming/src/strategies/mod.rs#L34) | The registry: strategy names and their constructors. |
| [`tests/fix_simple.rs`](../../crates/sidecar_streaming/tests/fix_simple.rs) | Tests of the examples against a mock `Ctx`. |
| [`src/strategies/leaf.rs`](../../crates/sidecar_streaming/src/strategies/leaf.rs), [`src/calib.rs`](../../crates/sidecar_streaming/src/calib.rs) | Research strategies that plan with a cost estimate. Experimental. |
| [`src/strategies/merge.rs`](../../crates/sidecar_streaming/src/strategies/merge.rs) | Research strategies for the 16:1 merge at the proposer. Experimental. |

The two examples estimate no cost, so they behave the same on any hardware:

| Strategy | Rule |
|---|---|
| `greedy_simple` ([line 17](../../crates/sidecar_streaming/src/strategies/simple.rs#L17)) | BLS-style batching. Whenever the prover is idle, prove all queued votes, extending the previous proof. |
| `once_at` ([line 81](../../crates/sidecar_streaming/src/strategies/simple.rs#L81)) | Wait until `D - lead_ms`, then prove all queued votes in one job. |

Neither example publishes at `D` itself.
The driver publishes the largest proof finished by `D` plus a grace window (`--grace-ms`).

The rest of the crate is experiment code: simulator, live driver, cost models and research strategies.
It works, but it is not cleaned up.

## Reproduce the results

The results table below comes from real sidecars, not from the simulator.
You need two programs: the strategy client `streamsim` from this branch, and a `leanvm-sidecar` build.

### 1. Build the sidecar

The table used two sidecar builds, each on its own branch of the fork:

| Build | Branch | Base |
|---|---|---|
| main | `alexey/sidecar-main` (`34a7285bf`) | upstream main `29077b733` plus a sidecar port |
| perf | `alexey/sidecar-perf` (`b102bb988`) | perf/optimisations `8329de69f` plus the same port |

Build with Rust 1.99.1 and AVX2 flags, as for the table:

```sh
git checkout alexey/sidecar-main   # or alexey/sidecar-perf
RUSTFLAGS="-C target-cpu=x86-64-v3 -C target-feature=+pclmulqdq,+vpclmulqdq" \
  cargo +1.99.1 build --release -p leanvm-sidecar
```

The binary is `target/release/leanvm-sidecar`. Copy it out before switching branches.

### 2. Build the client and unpack the votes

On this branch (a recent nightly fails on `vec_deque_pop_if`, so use 1.99.1):

```sh
cargo +1.99.1 build --release -p sidecar_streaming --bin streamsim
gunzip -k experiments/sidecar-streaming/sets/*/slots.csv.gz
```

### 3. Run

[`scripts/reproduce.sh`](scripts/reproduce.sh) starts the sidecar, runs both examples on the three slots, and stops it.
Run it from the repository root:

```sh
experiments/sidecar-streaming/scripts/reproduce.sh <path to leanvm-sidecar> <8|4> <once_at lead ms> [out dir]
```

| Build, cores | Lead (ms) |
|---|---|
| main, 8 | 2200 |
| main, 4 | 3700 |
| perf, 8 | 1300 |
| perf, 4 | 2200 |

The lead is about the time one AGGREGATE of all votes takes on that build.
The sidecar runs on cores 0–7 or 0–3, and the client on cores 8–11.
A full run takes about 2 minutes, warm-up included.
Each `<out>/<slot>-<strategy>.md` holds a coverage table per `D`.
Each `-jobs.csv` lists every proving job with its start and end time.
Job times depend on the CPU and on other load, so coverage moves with them.

To try your own strategy, pass its name to `streamsim live --strategy`, as the script does.

### Simulator

The simulator needs no sidecar.
Job times come from a cost table measured for one build ([`models/`](models)).
It is fast, but it is not a real prover.

```sh
target/release/streamsim sim \
  --set experiments/sidecar-streaming/sets/rec3-typical \
  --model experiments/sidecar-streaming/models/model-main-avx2-8c.txt \
  --strategy greedy_simple --slots 0..10 --deadline-ms 4000,6000,8000 --grace-ms 25 \
  --out sim.csv --jobs sim-jobs.csv
```

## Results

One slot per scenario, on an AMD EPYC 4585PX (Zen 5), with the AVX2 builds above.
`k` is the slot index in the set.
Each cell gives coverage for `greedy_simple` / `once_at`, in percent.

| Slot, D | main, 8 cores | main, 4 cores | perf, 8 cores | perf, 4 cores |
|---|---|---|---|---|
| typical k=2, 4 s | 99 / 4 | 31.5 / 0 | 99 / 59 | 73 / 4 |
| typical k=2, 8 s | 100 / 100 | 100 / 93 | 100 / 100 | 100 / 100 |
| epoch k=23, 6 s | 96.5 / 35 | 64 / 0 | 97 / 65 | 94 / 35 |
| epoch k=23, 8 s | 99.9 / 90 | 99.7 / 52 | 98.8 / 98 | 99.8 / 90 |
| timer k=0, 5.5 s | 23 / 0 | 43 / 0 | 97 / 24 | 38 / 0 |
| timer k=0, 8 s | 100 / 99 | 100 / 46 | ~100 / 99.9 | 100 / 99 |

These are single slots, so read them as examples, not as distributions.

Coverage is the number of votes in the published proof, divided by the votes that arrived by `D`.
Time zero is the slot start.
On mainnet (Fulu) attestations are due at 4 s and aggregates at 8 s.

## Vote sets

`sets/` holds recorded mainnet votes from Xatu gossip data, from 2026-11-01.
One slot is the union of 4 committees, about 1500–1600 votes, as one subnet aggregator would see them.
Arrival times are real and are not resampled.

| Set | Slots | Content |
|---|---|---|
| `rec3-typical` | 100 | Slots whose median vote time is near the overall median. |
| `rec3-epoch` | 100 | First slot of an epoch, where votes arrive later. |
| `rec3x-typical-timer100` | 100 | `rec3-typical` with every vote moved to 4 s + lognormal(320 ms, 0.65). It is synthetic: the worst case when all validators vote on the timer. |
| `*-train` | | Other slots, for strategies that learn before measuring. |

Each set has `slots.csv` (`slot,t_us,kind,subnet,view,id,valid`), `meta.txt` and `params.txt` (how it was made).
`sets/scenario_index.csv` lists the mainnet slot behind each sample.
`scripts/make_timer_set.py` builds the timer set from `rec3-typical`.

## Writing a strategy

A strategy is a state machine.
It receives events and answers with calls on `Ctx`.
It never sees payload bytes or the wire protocol.

| Event | Meaning |
|---|---|
| `SlotStart` | The slot begins. `ctx.slot()` gives `D`, the final rate and job caps. |
| `Signature(id)` | A vote arrived; the consensus layer has already validated it. |
| `Proved { result, .. }` | A job ended, with a proof id or a failure. |
| `Timer(token)` | A timer set with `ctx.set_timer` fired. |
| `Aggregate`, `Verified` | Received proofs from other aggregators. Used by the 16:1 merge, not by leaves. |

| Call on `Ctx` | Effect |
|---|---|
| `prove(lane, Job)` | Start a job. `children` are earlier proofs; `signatures` are new votes; `rate` is the proof rate. |
| `publish(proof)` | Publish the slot's one output. A second call panics. |
| `set_timer(at, token)` | Get `Timer(token)` at time `at`. |
| `now()`, `slot()`, `proof(id)`, `lane_idle(lane)` | Read state. |
| `cost(shape)` | Predicted duration of a job. The examples do not use it. |

A job with no children is an AGGREGATE.
A job with one child plus new votes is an EXTEND of that proof.
The live driver sends each child proof in the job payload.

To add a strategy:

1. Add a type that implements `Strategy` in a new file under `src/strategies/`. Start from `GreedySimple`.
2. Add its name to `LEAF` and a line to `Spec::make` in `src/strategies/mod.rs`.
3. Run it in the simulator, then against a live sidecar.

Parameters come as `name:key=value;key=value`, for example `once_at:lead_ms=1300`.
`Params::f(key, default)` reads one.
