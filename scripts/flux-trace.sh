#!/bin/bash
# usage: scripts/flux-trace.sh <out.fxt> -- <leanvm args...>
# e.g.   scripts/flux-trace.sh node.fxt -- --repeat 1 --cooldown 0 --log-inv-rate 2 recursion --n 2 --xmss-per-leaf 900
# LEANVM_NUM_THREADS and TASKSET (cores for the prover, Linux taskset) pass through.
# The flux-profiler CLI comes from PATH, or from FLUX_CLI; see doc/BENCHMARKS.md "Profiling".
set -e
OUT=$1; shift 2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=$ROOT/target/release/leanvm
CLI=${FLUX_CLI:-flux-profiler}
cargo build --release -p leanvm-cli --manifest-path "$ROOT/Cargo.toml"
${TASKSET:+taskset -c $TASKSET} "$BIN" --flux "$@" &
APP=$!
sleep 1
"$CLI" --pid $APP --out "$OUT" --max-mem 8GB > "$OUT.cli.log" 2>&1 &
wait $APP
wait $!
grep -i export "$OUT.cli.log"
