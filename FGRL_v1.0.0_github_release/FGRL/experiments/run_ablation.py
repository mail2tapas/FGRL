"""Ablation study (Section 5.5, Figure 7).

    python experiments/run_ablation.py --config configs/test.json --out results/test
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fgrl.config import load_config                          # noqa: E402
from fgrl.eval.runner import run_experiment, summary_table    # noqa: E402

METHODS = ["FGRL w/o GNN", "FGRL w/o critical-path reward", "FGRL w/o federation", "FGRL"]
ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--methods", nargs="*", default=METHODS)
ap.add_argument("--seeds", nargs="*", type=int, default=None, help="override the seeds of the config (to split a run across machines)")
a = ap.parse_args()
cfg = load_config(a.config)
x = json.load(open(a.config)).get("experiment", {})
os.makedirs(a.out, exist_ok=True)
runs, logs = run_experiment(cfg, a.methods, a.seeds or x.get("seeds", list(range(10))),
                            x.get("episodes", cfg.train.episodes), eval_seeds=x.get("eval_seeds", 3))
runs.to_csv(os.path.join(a.out, "ablation_runs.csv"), index=False)
logs.to_csv(os.path.join(a.out, "ablation_training_log.csv"), index=False)
tab = summary_table(runs, a.methods)
if "FGRL" in set(runs.method):
    full = runs[runs.method == "FGRL"].makespan.mean()
    tab["Makespan increase vs full FGRL (%)"] = [
        f"{100 * (runs[runs.method == m].makespan.mean() / full - 1):.1f}" for m in tab.Approach]
tab.to_csv(os.path.join(a.out, "ablation.csv"), index=False)
print(tab.to_string(index=False))
