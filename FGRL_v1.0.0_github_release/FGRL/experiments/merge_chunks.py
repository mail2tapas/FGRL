"""Merge results that were produced in several chunks (e.g. one seed per machine).

    python experiments/merge_chunks.py --out results/paper results/paper/chunks/*

CSV files with the same name are concatenated; JSON files are taken from the first chunk
that has them (footprint_latency.json is machine-specific and is kept per chunk as well).
"""
import argparse
import glob
import os
import shutil

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("chunks", nargs="+")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
names = sorted({os.path.basename(f) for c in a.chunks for f in glob.glob(os.path.join(c, "*.csv"))})
SKIP = {"table7_comparison.csv", "ablation.csv", "table5_scalability.csv", "figure4_communication.csv"}  # recomputed
for n in names:
    if n in SKIP:
        continue
    frames = [pd.read_csv(os.path.join(c, n)) for c in a.chunks if os.path.exists(os.path.join(c, n))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    df.to_csv(os.path.join(a.out, n), index=False)
    print(f"{n}: {len(frames)} chunks, {len(df)} rows")
for c in a.chunks:
    for f in glob.glob(os.path.join(c, "*.json")) + glob.glob(os.path.join(c, "*.log")):
        dst = os.path.join(a.out, f"{os.path.basename(c)}__{os.path.basename(f)}")
        shutil.copy(f, dst)
        if os.path.basename(f) == "statistics.json":   # per-chunk statistics are superseded by evidence.json
            continue
        if not os.path.exists(os.path.join(a.out, os.path.basename(f))):
            shutil.copy(f, os.path.join(a.out, os.path.basename(f)))
