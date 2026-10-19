"""Scalability with the number of fog nodes (Table 5, Figure 3).

    python experiments/run_scalability.py --config configs/test.json --out results/test
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fgrl.config import load_config                 # noqa: E402
from fgrl.eval.runner import run_experiment          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", nargs="*", type=int, default=None, help="override the seeds of the config (to split a run across machines)")
ap.add_argument("--methods", nargs="*", default=["HEFT", "Centralized PPO", "FGRL"])
a = ap.parse_args()
base = load_config(a.config)
x = json.load(open(a.config)).get("experiment", {})
node_counts = x.get("node_counts", [10, 20, 30, 40, 50, 60, 70, 80])
seeds = a.seeds or x.get("seeds", list(range(10)))
episodes = x.get("episodes", base.train.episodes)
os.makedirs(a.out, exist_ok=True)

frames = []
for n in node_counts:
    cfg = copy.deepcopy(base)
    cfg.infra.num_fog_nodes = n
    runs, _ = run_experiment(cfg, a.methods, seeds, episodes, eval_seeds=x.get("eval_seeds", 3))
    runs["fog_nodes"] = n
    frames.append(runs)
import pandas as pd  # noqa: E402
allr = pd.concat(frames)
allr.to_csv(os.path.join(a.out, "scalability_runs.csv"), index=False)
tab = allr.groupby(["fog_nodes", "method"]).makespan.agg(["mean", "std", "count"]).reset_index()
tab.to_csv(os.path.join(a.out, "table5_scalability.csv"), index=False)
wide = tab.assign(cell=lambda d: d["mean"].map("{:.2f}".format) + " ± " + d["std"].map("{:.2f}".format)) \
          .pivot(index="fog_nodes", columns="method", values="cell")[a.methods]
print(wide.to_string())
