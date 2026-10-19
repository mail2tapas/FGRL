"""Build the evidence pack for the paper from the raw run-level results.

    python experiments/evidence_report.py --results results/paper --config configs/paper_pegasus.json

Reads the raw CSV/JSON files written by the experiment scripts and writes
  EVIDENCE.md         human-readable report: every table/figure/statistic of Section 5
  evidence.json       the same numbers in machine-readable form
  tables/*.csv        Tables 5 and 7 (per workflow and pooled), ablation, statistics
  MANIFEST.sha256     SHA-256 of code, configs, benchmark data and every result file
  environment.json    software and hardware the results were produced on

All statistics use the run-level blocks (seed, workflow type): every method is evaluated on
the same blocks, so paired/rank-based tests are used throughout.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fgrl.eval.stats import average_ranks, friedman, nemenyi, cohens_dz  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--config", default="")
ap.add_argument("--boot", type=int, default=10000)
a = ap.parse_args()
R = a.results
os.makedirs(os.path.join(R, "tables"), exist_ok=True)
P = lambda *x: os.path.join(R, *x)
rng = np.random.default_rng(12345)
MET = {"makespan": ("Makespan (s)", True), "energy_kwh": ("Energy (kWh)", True),
       "sla_violation_pct": ("SLA violation (%)", True), "utilization_pct": ("Avg. utilization (%)", False)}
KINDS = ["montage", "cybershake", "epigenomics"]
ev, md = {}, []
w = md.append


def fmt(x, d=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def boot_ci(x, f=np.mean):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    b = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(a.boot)]
    return tuple(np.nanpercentile(b, [2.5, 97.5]))


def holm(pvals):
    p = np.asarray(pvals, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def per_workflow_table(runs, methods):
    rows = []
    for k in [k for k in KINDS if k in set(runs.workflow)]:
        for m in methods:
            g = runs[(runs.method == m) & (runs.workflow == k)]
            if g.empty:
                continue
            r = {"Workflow": k.capitalize(), "Approach": m, "n": len(g)}
            for c, (lab, _) in MET.items():
                d = 4 if c == "energy_kwh" else 2
                r[lab] = f"{g[c].mean():.{d}f} ± {g[c].std(ddof=1):.{d}f}" if len(g) > 1 else f"{g[c].mean():.{d}f}"
            rows.append(r)
    return pd.DataFrame(rows)


def normalized(runs, methods, ref="HEFT"):
    """per-block ratio to the reference method (scale-free across workflow types)."""
    piv = runs.pivot_table(index=["seed", "workflow"], columns="method", values="makespan")
    return piv[[m for m in methods if m in piv]].div(piv[ref], axis=0) if ref in piv else None


# ====================================================================== comparison (Table 7)
if os.path.exists(P("comparison_runs.csv")):
    runs = pd.read_csv(P("comparison_runs.csv"))
    methods = [m for m in ["HEFT", "PSO-Fuzzy", "Centralized PPO", "FGRL"] if m in set(runs.method)]
    blocks = runs.groupby(["seed", "workflow"]).ngroups
    w("## 1. Performance comparison (Table 7, Figure 6)\n")
    w(f"Blocks: {runs.seed.nunique()} seeds × {runs.workflow.nunique()} workflow types = {blocks} "
      f"(seed, workflow) blocks per method. Each block value is the mean over the held-out test "
      f"instances and dynamic-condition seeds of that block.\n")
    t7 = per_workflow_table(runs, methods)
    t7.to_csv(P("tables", "table7_per_workflow.csv"), index=False)
    w("### 1.1 Mean ± SD over seeds, per workflow type\n"); w(md_table(t7) + "\n")
    # pooled, scale-free
    N = normalized(runs, methods)
    pooled = []
    for m in methods:
        g = runs[runs.method == m]
        row = {"Approach": m, "Normalized makespan (÷HEFT)": f"{N[m].mean():.3f} ± {N[m].std(ddof=1):.3f}" if N is not None else "–",
               "SLA violation (%)": f"{g.sla_violation_pct.mean():.2f}", "Avg. utilization (%)": f"{g.utilization_pct.mean():.2f}"}
        if N is not None and m != "HEFT":
            red = 100 * (1 - N[m].values)
            lo, hi = boot_ci(red)
            row["Makespan reduction vs HEFT (%) [95% CI]"] = f"{red.mean():.1f} [{lo:.1f}, {hi:.1f}]"
        pooled.append(row)
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(P("tables", "table7_pooled_normalized.csv"), index=False)
    w("### 1.2 Pooled over all blocks (makespan normalized to HEFT within each block)\n")
    w(md_table(pooled) + "\n")
    # relative improvements of FGRL per metric (block-wise ratios, then averaged)
    rel = {}
    if "FGRL" in methods:
        for c, (lab, lower) in MET.items():
            piv = runs.pivot_table(index=["seed", "workflow"], columns="method", values=c)
            rel[c] = {}
            for m in methods:
                if m == "FGRL":
                    continue
                if lower:
                    x = 100 * (1 - piv["FGRL"] / piv[m])
                else:
                    x = 100 * (piv["FGRL"] / piv[m] - 1)
                x = x.replace([np.inf, -np.inf], np.nan).dropna().values
                lo, hi = boot_ci(x)
                rel[c][m] = {"mean_pct": float(np.mean(x)), "ci95": [float(lo), float(hi)],
                             "per_workflow": {k: float(np.mean(100 * (1 - piv["FGRL"].xs(k, level=1) / piv[m].xs(k, level=1))
                                                              if lower else 100 * (piv["FGRL"].xs(k, level=1) / piv[m].xs(k, level=1) - 1)))
                                              for k in KINDS if k in set(runs.workflow)}}
        ev["relative_improvement_of_FGRL"] = rel
        rows = []
        for c, d in rel.items():
            for m, v in d.items():
                rows.append({"Metric": MET[c][0], "vs": m, "Mean block-wise improvement (%)": f"{v['mean_pct']:.1f}",
                             "95% bootstrap CI": f"[{v['ci95'][0]:.1f}, {v['ci95'][1]:.1f}]",
                             **{k.capitalize(): f"{v['per_workflow'][k]:.1f}" for k in v["per_workflow"]}})
        w("### 1.3 Improvement of FGRL (positive = FGRL better), block-wise\n")
        w("Lower-is-better metrics: 100·(1 − FGRL/baseline); utilization: 100·(FGRL/baseline − 1). "
          "SLA ratios are undefined when a baseline has 0 % violations in a block; those blocks are skipped.\n")
        w(md_table(pd.DataFrame(rows)) + "\n")
        pd.DataFrame(rows).to_csv(P("tables", "relative_improvement.csv"), index=False)

    # ---------------------------------------------------------------- statistics (Section 5.7)
    w("## 2. Statistical tests (Section 5.7)\n")
    w("Friedman test over the (seed, workflow) blocks, Nemenyi post hoc test (critical difference at α = 0.05), "
      "two-sided Wilcoxon signed-rank tests of FGRL against each baseline with Holm correction, and the paired "
      "effect size d_z = mean(diff)/sd(diff) with a 95 % bootstrap confidence interval. Differences are taken as "
      "log(baseline/FGRL) for makespan and energy (scale-free across workflow types) and as percentage points for SLA "
      "violation and utilization, so that positive values favour FGRL.\n")
    st = {}
    for c, (lab, lower) in MET.items():
        piv = runs.pivot_table(index=["seed", "workflow"], columns="method", values=c)[methods].dropna()
        n, k = piv.shape
        if n < 2 or k < 3:
            continue
        fr = friedman(piv.values)
        ranks = dict(zip(methods, average_ranks(piv.values, lower)))
        q05 = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
        cd = float(q05 * np.sqrt(k * (k + 1) / (6.0 * n)))
        ne = nemenyi(piv.values, methods, lower)
        wil = []
        if "FGRL" in methods:
            others = [m for m in methods if m != "FGRL"]
            ps, rows = [], []
            for m in others:
                # paired differences, positive = FGRL better; log-ratios for makespan/energy (scale-free
                # across workflow types), percentage-point differences for SLA violation and utilization
                if c in ("makespan", "energy_kwh"):
                    d = np.log(piv[m].values / piv["FGRL"].values)
                else:
                    d = (piv[m] - piv["FGRL"]).values if lower else (piv["FGRL"] - piv[m]).values
                try:   # same paired differences d as used for d_z (log ratios / percentage points)
                    p = float(stats.wilcoxon(d, alternative="two-sided", zero_method="zsplit").pvalue)
                except ValueError:
                    p = 1.0
                ps.append(p)
                dz = cohens_dz(d, np.zeros_like(d)) if np.std(d) > 0 else float("nan")
                lo, hi = boot_ci(d, lambda x: x.mean() / x.std(ddof=1) if x.std(ddof=1) > 0 else np.nan)
                rows.append({"vs": m, "wilcoxon_p": p, "d_z": dz, "d_z_ci95": [float(lo), float(hi)],
                             "fgrl_better_blocks": int((d > 0).sum()), "blocks": int(len(d))})
            for r_, padj in zip(rows, holm(ps)):
                r_["wilcoxon_p_holm"] = float(padj)
            wil = rows
        st[c] = {"n_blocks": n, "k_methods": k, "friedman_chi2": fr["statistic"], "df": k - 1,
                 "friedman_p": fr["p_value"], "average_ranks": ranks, "nemenyi_cd_0.05": cd,
                 "nemenyi": ne, "fgrl_vs_baselines": wil}
        w(f"### {lab}\n")
        w(f"Friedman χ²({k - 1}) = {fr['statistic']:.2f}, p = {fr['p_value']:.3g}, N = {n} blocks. "
          f"Average ranks (1 = best): " + ", ".join(f"{m} {r:.2f}" for m, r in sorted(ranks.items(), key=lambda t: t[1]))
          + f". Nemenyi CD = {cd:.2f}.\n")
        if wil:
            tb = pd.DataFrame([{"FGRL vs": r_["vs"], "FGRL better in": f"{r_['fgrl_better_blocks']}/{r_['blocks']} blocks",
                                "Wilcoxon p": f"{r_['wilcoxon_p']:.3g}", "Holm-adjusted p": f"{r_['wilcoxon_p_holm']:.3g}",
                                "Nemenyi p": f"{[x['p_value'] for x in ne if {x['a'], x['b']} == {'FGRL', r_['vs']}][0]:.3g}",
                                "Paired d_z [95% CI]": f"{fmt(r_['d_z'])} [{fmt(r_['d_z_ci95'][0])}, {fmt(r_['d_z_ci95'][1])}]"}
                               for r_ in wil])
            w(md_table(tb) + "\n")
            tb.to_csv(P("tables", f"stats_{c}.csv"), index=False)
    ev["statistics"] = st

# ====================================================================== convergence (Figure 5)
if os.path.exists(P("figure5_convergence.json")):
    cv = json.load(open(P("figure5_convergence.json")))
    ev["convergence"] = cv
    w("## 3. Training convergence (Section 5.2, Figure 5)\n")
    w(f"Definition: {cv.get('definition', '')}. One convergence episode per (seed, workflow type) run; "
      "episodes are counted per workflow type.\n")
    rows = []
    for k in KINDS + ["pooled"]:
        for m in ("FGRL", "Centralized PPO"):
            v = cv.get(k, {}).get(m)
            if v:
                rows.append({"Workflow": k.capitalize(), "Method": m, "Converged runs": f"{v['converged']}/{v['runs']}",
                             "Convergence episode (mean ± SD)": f"{fmt(v['episode_mean'], 0)} ± {fmt(v['episode_sd'], 0)}"})
    if rows:
        w(md_table(pd.DataFrame(rows)) + "\n")
    pl = cv.get("pooled", {})
    if "mann_whitney_p_two_sided" in pl:
        w(f"Mann–Whitney U = {pl['mann_whitney_U']:.1f}, two-sided p = {pl['mann_whitney_p_two_sided']:.3g}; "
          f"centralized PPO needs {pl['speedup_ppo_over_fgrl']:.2f}× as many episodes as FGRL on average.\n")
    sens = cv.get("sensitivity", {})
    if sens:
        rows = []
        for tol, res in sens.items():
            for m in ("FGRL", "Centralized PPO"):
                v = res.get(m)
                if v:
                    rows.append({"Tolerance": tol, "Method": m, "Converged runs": f"{v['converged']}/{v['runs']}",
                                 "Convergence episode (mean ± SD)": f"{fmt(v['episode_mean'], 0)} ± {fmt(v['episode_sd'], 0)}"})
        w("Sensitivity to the stability tolerance (same moving-average window and stability window):\n")
        w(md_table(pd.DataFrame(rows)) + "\n")
    bv = cv.get("best_validation_episode", {})
    if bv:
        rows = [{"Method": m, "Episode of best validation score (mean ± SD)": f"{v['mean']:.0f} ± {fmt(v['sd'], 0)}",
                 "Runs": len(v["per_run"])} for m, v in bv.items() if isinstance(v, dict)]
        w("Model-selection point (episode at which the policy that is finally evaluated reached its best validation "
          "score; validation every 50 episodes):\n")
        w(md_table(pd.DataFrame(rows)) + "\n")
        if "wilcoxon_paired_p" in bv:
            w(f"Paired Wilcoxon test (same seed), FGRL vs centralized PPO: p = {bv['wilcoxon_paired_p']:.3g}.\n")
    tl = cv.get("training_length_episodes", {})
    if tl:
        w("Training length with early stopping (patience 5 evaluations = 250 episodes without improvement): " +
          "; ".join(f"{m}: mean {v['mean']:.0f} episodes (range {v['min']}–{v['max']})" for m, v in tl.items()) + ".\n")
    w("Runs that end (early stopping) before the stability criterion is met are counted as not converged.\n")

# ====================================================================== ablation (Figure 7)
abl_path = P("ablation_runs.csv")
if os.path.exists(abl_path):
    ab = pd.read_csv(abl_path)
    if "FGRL" not in set(ab.method) and os.path.exists(P("comparison_runs.csv")):
        ab = pd.concat([ab, pd.read_csv(P("comparison_runs.csv")).query("method == 'FGRL'")], ignore_index=True)
    am = [m for m in ["FGRL w/o GNN", "FGRL w/o critical-path reward", "FGRL w/o federation", "FGRL"] if m in set(ab.method)]
    piv = ab.pivot_table(index=["seed", "workflow"], columns="method", values="makespan")[am].dropna()
    rows = []
    # Wilcoxon signed-rank test on log(variant / FGRL) per block: the same relative scale as the
    # bootstrap interval of the relative increase (raw differences in seconds would let the long
    # Epigenomics runs dominate the ranks); Holm correction over the ablation comparisons.
    variants = [m for m in am if m != "FGRL"]
    pv = {}
    for m in variants:
        try:
            pv[m] = float(stats.wilcoxon(np.log(piv[m].values / piv["FGRL"].values), zero_method="zsplit").pvalue)
        except ValueError:
            pv[m] = float("nan")
    padj = dict(zip(variants, holm([pv[m] for m in variants]))) if variants else {}
    for m in am:
        r = {"Variant": m}
        if m != "FGRL":
            inc = 100 * (piv[m] / piv["FGRL"] - 1)
            lo, hi = boot_ci(inc.values)
            r.update({"Makespan increase vs full FGRL (%) [95% CI]": f"{inc.mean():.1f} [{lo:.1f}, {hi:.1f}]",
                      **{k.capitalize(): f"{inc.xs(k, level=1).mean():.1f}" for k in KINDS if k in set(ab.workflow)},
                      "Wilcoxon p (log ratio)": f"{pv[m]:.3g}", "Holm-adjusted p": f"{padj[m]:.3g}",
                      "FGRL better": f"{int((inc > 0).sum())}/{len(inc)}", "n blocks": len(inc)})
        rows.append(r)
    tb = pd.DataFrame(rows)
    tb.to_csv(P("tables", "ablation_blockwise.csv"), index=False)
    ev["ablation"] = rows
    w("## 4. Ablation study (Section 5.5, Figure 7)\n"); w(md_table(tb.fillna("")) + "\n")

# ====================================================================== scalability (Table 5, Figure 3)
if os.path.exists(P("scalability_runs.csv")):
    sc = pd.read_csv(P("scalability_runs.csv"))
    rows = []
    for k in [k for k in KINDS if k in set(sc.workflow)]:
        for n in sorted(sc.fog_nodes.unique()):
            r = {"Workflow": k.capitalize(), "Fog nodes": int(n)}
            for m in ["HEFT", "Centralized PPO", "FGRL"]:
                g = sc[(sc.method == m) & (sc.fog_nodes == n) & (sc.workflow == k)].makespan
                if len(g):
                    r[m] = f"{g.mean():.2f} ± {g.std(ddof=1):.2f}" if len(g) > 1 else f"{g.mean():.2f}"
            rows.append(r)
    tb = pd.DataFrame(rows)
    tb.to_csv(P("tables", "table5_scalability_per_workflow.csv"), index=False)
    piv = sc.pivot_table(index=["fog_nodes", "seed", "workflow"], columns="method", values="makespan")
    nr = []
    for n in sorted(sc.fog_nodes.unique()):
        g = piv.xs(n, level=0)
        r = {"Fog nodes": int(n), "seeds": sc[sc.fog_nodes == n].seed.nunique()}
        if "HEFT" in g and "FGRL" in g:
            r["FGRL reduction vs HEFT (%)"] = f"{100 * (1 - g.FGRL / g.HEFT).mean():.1f}"
        if "Centralized PPO" in g and "FGRL" in g:
            r["FGRL reduction vs Centralized PPO (%)"] = f"{100 * (1 - g.FGRL / g['Centralized PPO']).mean():.1f}"
        nr.append(r)
    ev["scalability"] = nr
    w("## 5. Scalability with the number of fog nodes (Table 5, Figure 3)\n")
    w(md_table(tb.fillna("")) + "\n"); w(md_table(pd.DataFrame(nr)) + "\n")

# ====================================================================== communication (Figure 4) + footprint
if os.path.exists(P("communication_runs.csv")):
    cm = pd.read_csv(P("communication_runs.csv"))
    g = cm.groupby("K").agg(rounds=("rounds", "mean"), MB_total=("bytes_total", lambda x: x.mean() / 1e6),
                            overhead_mean=("overhead_pct", "mean"), overhead_sd=("overhead_pct", "std"),
                            seeds=("seed", "nunique")).reset_index()
    ev["communication"] = g.to_dict(orient="records")
    w("## 6. Communication overhead (Figure 4)\n")
    w("Overhead = modelled transfer time of all federated rounds ÷ (measured training time + transfer time).\n")
    w(md_table(g.round(3)) + "\n")
if os.path.exists(P("footprint_latency.json")):
    fp = json.load(open(P("footprint_latency.json")))
    ev["footprint_latency_this_machine"] = fp
    w("## 7. Per-node footprint and inference latency (Sections 4.2.1, 5.3) – measured on the run machine only\n")
    w("```\n" + json.dumps(fp, indent=2) + "\n```\n")
    w("These latencies are for the machine above. Values for Raspberry Pi / Intel NUC (Table 6) require running "
      "`experiments/run_communication.py` on those devices.\n")


# ====================================================================== provenance
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


files = []
for sub in ("fgrl", "experiments", "configs", "data/pegasus", "scripts"):
    for dp, _, fs in os.walk(os.path.join(ROOT, sub)):
        if "__pycache__" in dp:
            continue
        files += [os.path.join(dp, f) for f in sorted(fs) if not f.endswith(".pyc")]
for dp, _, fs in os.walk(R):
    files += [os.path.join(dp, f) for f in sorted(fs) if f not in ("MANIFEST.sha256", "EVIDENCE.md", "evidence.json", "environment.json")]
with open(P("MANIFEST.sha256"), "w") as f:
    for x in sorted(set(files)):
        f.write(f"{sha(x)}  {os.path.relpath(x, ROOT)}\n")

env = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "python": sys.version.split()[0],
       "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
       "cpu_count": os.cpu_count()}
try:
    import torch, numpy, scipy, networkx, pandas as _pd
    env.update(torch=torch.__version__, numpy=numpy.__version__, scipy=scipy.__version__,
               networkx=networkx.__version__, pandas=_pd.__version__, cuda=torch.cuda.is_available())
except Exception:
    pass
try:
    env["cpu_model"] = [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
except Exception:
    pass
if a.config:
    env["config_file"] = a.config
    env["config"] = json.load(open(a.config))
json.dump(env, open(P("environment.json"), "w"), indent=2)
ev["environment"] = env

head = ["# FGRL – evidence report\n",
        f"Generated {env['generated_utc']} from the raw results in `{os.path.relpath(R, ROOT)}`.\n",
        f"Configuration: `{a.config or 'see environment.json'}`. Machine: {env.get('cpu_model', env['processor'])}, "
        f"{env['cpu_count']} CPU threads, Python {env['python']}, PyTorch {env.get('torch', '?')}.\n",
        "Benchmark workflows: Pegasus Montage, CyberShake and Epigenomics DAX files (Bharathi et al., 2008; "
        "Juve et al., 2013), see `data/pegasus/SOURCES.md`. Every file used is listed with its SHA-256 in "
        "`MANIFEST.sha256`, so the results can be traced to the exact code and data.\n",
        "Figures: `figure3_scalability.png`, `figure4_communication.png`, `figure5_convergence.png`, "
        "`figure6_comparison.png`, `figure7_ablation.png` (600 dpi).\n"]
open(P("EVIDENCE.md"), "w").write("\n".join(head + md))
json.dump(ev, open(P("evidence.json"), "w"), indent=2, default=float)

print("written:", P("EVIDENCE.md"))
