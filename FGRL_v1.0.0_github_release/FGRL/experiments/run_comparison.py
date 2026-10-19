"""Performance comparison (Table 7, Figure 6), statistics (Section 5.7) and the
training logs used for the convergence plot (Figure 5).

    python experiments/run_comparison.py --config configs/test.json --out results/test
    python experiments/run_comparison.py --config configs/paper.json --out results/paper
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fgrl.config import load_config                      # noqa: E402
from fgrl.eval.runner import run_experiment, summary_table  # noqa: E402
from fgrl.eval.stats import cohens_d, cohens_dz, friedman, nemenyi    # noqa: E402

METHODS = ["HEFT", "PSO-Fuzzy", "Centralized PPO", "FGRL"]

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", nargs="*", type=int, default=None, help="override the seeds of the config (to split a run across machines)")
ap.add_argument("--methods", nargs="*", default=METHODS)
a = ap.parse_args()
cfg = load_config(a.config)
x = json.load(open(a.config)).get("experiment", {})
seeds = a.seeds or x.get("seeds", list(range(10)))
episodes = x.get("episodes", cfg.train.episodes)
os.makedirs(a.out, exist_ok=True)

runs, logs = run_experiment(cfg, a.methods, seeds, episodes, eval_seeds=x.get("eval_seeds", 3))
runs.to_csv(os.path.join(a.out, "comparison_runs.csv"), index=False)
logs.to_csv(os.path.join(a.out, "training_log.csv"), index=False)
tab = summary_table(runs, a.methods)
tab.to_csv(os.path.join(a.out, "table7_comparison.csv"), index=False)
print(tab.to_string(index=False))

# ---- statistics: blocks = (seed, workflow), one column per method
stats_out = {}
piv = {m: runs.pivot_table(index=["seed", "workflow"], columns="method", values=m)
       for m in ["makespan", "energy_kwh", "sla_violation_pct", "utilization_pct"]}
for metric, P in piv.items():
    P = P[[m for m in a.methods if m in P.columns]].dropna()
    if P.shape[0] >= 2 and P.shape[1] >= 3:
        lower = metric != "utilization_pct"
        stats_out[metric] = {"n_blocks": int(P.shape[0]),
                             "friedman": friedman(P.values),
                             "nemenyi": nemenyi(P.values, list(P.columns), lower_is_better=lower)}
if "FGRL" in runs.method.unique():
    f = runs[runs.method == "FGRL"].sort_values(["seed", "workflow"]).makespan.values
    stats_out["cohens_d_makespan_vs_FGRL"] = {
        m: cohens_d(runs[runs.method == m].sort_values(["seed", "workflow"]).makespan.values, f)
        for m in a.methods if m != "FGRL"}
    stats_out["cohens_dz_paired_makespan_vs_FGRL"] = {
        m: cohens_dz(runs[runs.method == m].sort_values(["seed", "workflow"]).makespan.values, f)
        for m in a.methods if m != "FGRL"}
    stats_out["relative_makespan_reduction_pct"] = {
        m: 100 * (1 - f.mean() / runs[runs.method == m].makespan.mean()) for m in a.methods if m != "FGRL"}
json.dump(stats_out, open(os.path.join(a.out, "statistics.json"), "w"), indent=2)
for metric in ["makespan", "energy_kwh", "sla_violation_pct", "utilization_pct"]:
    if metric in stats_out:
        f = stats_out[metric]["friedman"]
        print(f"Friedman {metric}: chi2={f['statistic']:.2f}, p={f['p_value']:.4g}")
print("Cohen's d, pooled (makespan, method vs FGRL):", json.dumps(stats_out.get("cohens_d_makespan_vs_FGRL", {})))
print("Cohen's d_z, paired (makespan, method vs FGRL):", json.dumps(stats_out.get("cohens_dz_paired_makespan_vs_FGRL", {})))
print("Makespan reduction of FGRL (%):", json.dumps(stats_out.get("relative_makespan_reduction_pct", {})))
print("Full results incl. Nemenyi p-values:", os.path.join(a.out, "statistics.json"))
