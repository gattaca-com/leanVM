"""Build a recorded set with more validators on the attestation timer (Fable's sender-mix, design/XMSS-ARRIVAL.md).
Usage: make_timer_set.py <source set> <target set> <share> <seed>
Each vote before 4000 ms moves, with probability share, to 4000 + lognormal(median 320 ms, sigma 0.65).
Vote ids, counts and all other columns stay. meta.txt is copied; params.txt gets a comment naming the transform."""
import csv
import math
import os
import random
import shutil
import sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sets")
src, dst, share, seed = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
os.makedirs(f"{R}/{dst}", exist_ok=True)
shutil.copy(f"{R}/{src}/meta.txt", f"{R}/{dst}/meta.txt")
with open(f"{R}/{src}/params.txt") as f:
    params = f.read()
with open(f"{R}/{dst}/params.txt", "w") as f:
    f.write(f"# Derived from {src}: retime(share={share}, due=4000 ms, lognormal(320 ms, 0.65)), seed {seed}.\n")
    f.write("# Models more validators voting on the attestation timer (design/XMSS-ARRIVAL.md, sender-mix).\n")
    f.write(params)
rows = []
with open(f"{R}/{src}/slots.csv") as f:
    rd = csv.DictReader(f)
    fields = rd.fieldnames
    for r in rd:
        rows.append(r)
moved = 0
for r in rows:
    rng = random.Random(f"{seed}/{r['slot']}/{r['view']}/{r['id']}")
    t = int(r["t_us"])
    if r["kind"] == "vote" and t < 4_000_000 and rng.random() < share:
        r["t_us"] = str(4_000_000 + int(320_000 * math.exp(0.65 * rng.gauss(0, 1))))
        moved += 1
rows.sort(key=lambda r: (int(r["slot"]), int(r["t_us"])))
with open(f"{R}/{dst}/slots.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(dst, "rows", len(rows), "moved", moved)
