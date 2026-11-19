# Streaming aggregation strategies for the leanVM sidecar

This folder shows how to write a strategy that aggregates votes as they arrive during a slot.
The strategy drives a leanVM sidecar and publishes one proof by a deadline `D`.

The code lives in [`crates/sidecar_streaming`](../../crates/sidecar_streaming).
Two files there are the examples to read and copy.
The rest of the crate is experiment code: simulator, live driver, cost models and research strategies.
It works, but it is not cleaned up.

## The examples

[`strategies/simple.rs`](../../crates/sidecar_streaming/src/strategies/simple.rs) holds two strategies in about 120 lines.
Neither one estimates cost, so both behave the same on any hardware.

| Strategy | Rule |
|---|---|
| `greedy_simple` ([line 17](../../crates/sidecar_streaming/src/strategies/simple.rs#L17)) | BLS-style batching. Whenever the prover is idle, prove all queued votes, extending the previous proof. |
| `once_at` ([line 81](../../crates/sidecar_streaming/src/strategies/simple.rs#L81)) | Wait until `D - lead_ms`, then prove all queued votes in one job. |

Neither strategy publishes at `D` itself.
The driver publishes the largest proof finished by `D` plus a grace window (`--grace-ms`).

## The API

A strategy is a state machine ([`strategy.rs`](../../crates/sidecar_streaming/src/strategy.rs)).
It receives events and answers with calls on `Ctx`.
It never sees payload bytes or the wire protocol.

Events ([`types.rs`](../../crates/sidecar_streaming/src/types.rs#L140)):

| Event | Meaning |
|---|---|
| `SlotStart` | The slot begins. `ctx.slot()` gives `D`, the final rate and job caps. |
| `Signature(id)` | A vote arrived; the consensus layer has already validated it. |
| `Proved { result, .. }` | A job ended, with a proof id or a failure. |
| `Timer(token)` | A timer set with `ctx.set_timer` fired. |
| `Aggregate`, `Verified` | Received proofs from other aggregators. Used by the 16:1 merge, not by leaves. |

Calls on `Ctx`:

| Call | Effect |
|---|---|
| `prove(lane, Job)` | Start a job. `children` are earlier proofs; `signatures` are new votes; `rate` is the proof rate. |
| `publish(proof)` | Publish the slot's one output. A second call panics. |
| `set_timer(at, token)` | Get `Timer(token)` at time `at`. |
| `now()`, `slot()`, `proof(id)`, `lane_idle(lane)` | Read state. |
| `cost(shape)` | Predicted duration of a job. The examples do not use it. |

One job with no children is an AGGREGATE.
One job with one child plus new votes is an EXTEND of that proof.
The live driver sends each child proof in the job payload.

## Write your own

1. Add a type that implements `Strategy` in a new file under `src/strategies/`. Start from `GreedySimple`.
2. Add its name to `LEAF` and a line to `Spec::make` in [`strategies/mod.rs`](../../crates/sidecar_streaming/src/strategies/mod.rs#L34).
3. Run it in the simulator, then against a live sidecar (below).

Parameters come as `name:key=value;key=value`, for example `once_at:lead_ms=1300`.
`Params::f(key, default)` reads one.

[`tests/fix_simple.rs`](../../crates/sidecar_streaming/tests/fix_simple.rs) tests a strategy against a mock `Ctx`.

## Run

Build with Rust 1.99.1 (a recent nightly fails on `vec_deque_pop_if`), then unpack the vote sets:

```sh
cargo +1.99.1 build --release -p sidecar_streaming --bin streamsim
gunzip -k experiments/sidecar-streaming/sets/*/slots.csv.gz
```

Simulator: job times come from a measured cost table for one build (`models/`), with no prover running.

```sh
target/release/streamsim sim \
  --set experiments/sidecar-streaming/sets/rec3-typical \
  --model experiments/sidecar-streaming/models/model-main-avx2-8c.txt \
  --strategy greedy_simple --slots 0..10 --deadline-ms 4000,6000,8000 --grace-ms 25 \
  --out sim.csv --jobs sim-jobs.csv
```

Live: start a sidecar, then drive it with the same strategy.
`--model` is still required, but the two examples do not use it.

```sh
leanvm-sidecar --path /dev/shm/leanvm-live --threads 8 --warmup-signatures 1500 --warmup-rates 1,2 --arena-mib 256 &
target/release/streamsim live \
  --set experiments/sidecar-streaming/sets/rec3-typical \
  --model experiments/sidecar-streaming/models/model-main-avx2-8c.txt \
  --channel /dev/shm/leanvm-live --strategy greedy_simple --strategy-cost table \
  --slots 2..3 --deadline-ms 4000,6000,8000 --grace-ms 25 --force \
  --out live.csv --jobs live-jobs.csv
```

Both commands print a coverage table per `D` on stdout.
`--jobs` writes one row per proving job, with start and end times.

## Metric

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

## Live results

One slot per scenario, on an AMD EPYC 4585PX (Zen 5) with AVX2 builds.
`k` is the slot index in the set.
Each cell is coverage for `greedy_simple` / `once_at`, in percent.

| Slot, D | main, 8 cores | main, 4 cores | perf, 8 cores | perf, 4 cores |
|---|---|---|---|---|
| typical k=2, 4 s | 99 / 4 | 31.5 / 0 | 99 / 59 | 73 / 4 |
| typical k=2, 8 s | 100 / 100 | 100 / 93 | 100 / 100 | 100 / 100 |
| epoch k=23, 6 s | 96.5 / 35 | 64 / 0 | 97 / 65 | 94 / 35 |
| epoch k=23, 8 s | 99.9 / 90 | 99.7 / 52 | 98.8 / 98 | 99.8 / 90 |
| timer k=0, 5.5 s | 23 / 0 | 43 / 0 | 97 / 24 | 38 / 0 |
| timer k=0, 8 s | 100 / 99 | 100 / 46 | ~100 / 99.9 | 100 / 99 |

`once_at` leads were 2200 ms (main, 8 cores), 3700 ms (main, 4 cores), 1300 ms (perf, 8 cores) and 2200 ms (perf, 4 cores).
"main" is upstream main `29077b733` with a sidecar port.
"perf" is the perf/optimisations branch `8329de69f` with the same port.
Neither sidecar build is on this branch.
These are single slots, so read them as examples, not as distributions.

## Research strategies

[`strategies/leaf.rs`](../../crates/sidecar_streaming/src/strategies/leaf.rs) holds strategies that plan with a cost estimate.
Examples are `greedy_fit`, `plan_cover` and `one_shot_late`.
They share a deadline guard in `LeafStrategy::evaluate` ([line 93](../../crates/sidecar_streaming/src/strategies/leaf.rs#L93)).
Under that guard, the last job must end by `D`, and an earlier job must leave room for it.
The estimate comes from [`calib.rs`](../../crates/sidecar_streaming/src/calib.rs), a cost line fitted online from finished jobs.
[`strategies/merge.rs`](../../crates/sidecar_streaming/src/strategies/merge.rs) holds the 16:1 merge strategies.
Both files are experimental.
