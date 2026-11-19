#!/bin/bash
# Reproduce the live results table in ../README.md for one sidecar build and core count.
#
# Usage: reproduce.sh <leanvm-sidecar binary> <8|4> <once_at lead ms> [out dir]
#   Leads used for the table: 2200 on 8 cores, 3700 on 4 cores.
#
# Run from the repository root, after building streamsim and unpacking the sets (see ../README.md).
# The sidecar runs on cores 0-7 (8) or 0-3 (4); the client runs on cores 8-11.
# Keep the machine otherwise idle: other load changes the job times and so the coverage.
set -u
SIDECAR=$1
CORES=$2
LEAD=$3
OUT=${4:-reproduce-out}
E=experiments/sidecar-streaming
SIM=${SIM:-target/release/streamsim}
CHAN=/dev/shm/leanvm-reproduce-$$
if [ "$CORES" = 8 ]; then SCORES=0-7; else SCORES=0-3; fi
mkdir -p "$OUT"

taskset -c $SCORES "$SIDECAR" --path $CHAN --threads "$CORES" --warmup-signatures 1500 --warmup-rates 1,2 --arena-mib 256 > "$OUT/sidecar.log" 2>&1 &
PID=$!
echo "sidecar pid $PID, waiting for warm-up"
for i in $(seq 1 600); do
  grep -q serving "$OUT/sidecar.log" && break
  kill -0 $PID 2>/dev/null || { echo "sidecar died"; cat "$OUT/sidecar.log"; exit 1; }
  sleep 1
done

# name, set, slots, deadlines (ms from slot start)
run() {
  for s in greedy_simple "once_at:lead_ms=$LEAD"; do
    f=$OUT/$1-${s%%:*}
    taskset -c 8-11 "$SIM" live --set "$E/sets/$2" --model "$E/models/model-main-avx2-8c.txt" --channel $CHAN \
      --strategy "$s" --strategy-cost table --slots "$3" --deadline-ms "$4" --grace-ms 25 --force \
      --out "$f.csv" --jobs "$f-jobs.csv" > "$f.md" 2> "$f.err"
    echo "$1 $s exit $?"
  done
}
run typical-slot2 rec3-typical 2..3 4000,6000,8000
run epoch-slot23 rec3-epoch 23..24 4000,6000,8000,10000
run timer100-slot0 rec3x-typical-timer100 0..1 4500,5000,5500,6000,8000

kill $PID
wait $PID 2>/dev/null
rm -f $CHAN
echo "coverage tables: $OUT/*.md"
