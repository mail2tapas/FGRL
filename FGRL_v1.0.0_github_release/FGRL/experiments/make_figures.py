"""Builds Figures 3-7 at their printed size (8-9 pt text, 600 dpi) from the CSV files written by the experiment scripts.

    python experiments/make_figures.py --results results/test

Figure 5 also prints the convergence numbers for Section 5.2:
stable convergence = moving average (window 50) fluctuates < 2 % over 100
consecutive episodes; the convergence episode is computed for every run.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 7.5, "lines.linewidth": 1.4, "lines.markersize": 4})
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
from scipy import stats           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--ma", type=int, default=50)
ap.add_argument("--window", type=int, default=100)
ap.add_argument("--tol", type=float, default=0.05)
a = ap.parse_args()
R = a.results
p = lambda f: os.path.join(R, f)
STY = {"HEFT": dict(color="tab:blue", marker="o", ls="--"),
       "Centralized PPO": dict(color="tab:orange", marker="s", ls="--"),
       "FGRL": dict(color="tab:green", marker="D", ls="-", lw=2.2),
       "PSO-Fuzzy": dict(color="tab:purple", marker="^", ls=":")}


def save(name):
    plt.tight_layout(pad=0.4); plt.savefig(p(name), dpi=600); plt.close(); print("wrote", p(name))


KINDS = ["montage", "cybershake", "epigenomics"]
NAME = {"montage": "Montage", "cybershake": "CyberShake", "epigenomics": "Epigenomics", "all": "All"}


def kinds_in(d):
    return [k for k in KINDS if k in set(d.workflow)] or sorted(set(d.workflow))


# Figure 3 - scalability (one panel per workflow type: makespans differ by orders of magnitude)
if os.path.exists(p("scalability_runs.csv")):
    d = pd.read_csv(p("scalability_runs.csv"))
    ks = kinds_in(d)
    fig, axes = plt.subplots(1, len(ks), figsize=(6.3, 2.5), squeeze=False)
    for ax, k in zip(axes[0], ks):
        for m in ["HEFT", "Centralized PPO", "FGRL"]:
            g = d[(d.method == m) & (d.workflow == k)]
            if g.empty:
                continue
            st = g.groupby("fog_nodes").makespan.agg(["mean", "std"])
            ax.errorbar(st.index, st["mean"], yerr=st["std"].fillna(0), capsize=3, label=m, **STY.get(m, {}))
        ax.set_title(NAME.get(k, k)); ax.set_xlabel("Number of fog nodes"); ax.grid(ls=":", alpha=.6)
    axes[0][0].set_ylabel("Makespan (s)"); axes[0][0].legend(fontsize=7, handlelength=1.6)
    save("figure3_scalability.png")

# Figure 4 - communication overhead
if os.path.exists(p("communication_runs.csv")):
    d = pd.read_csv(p("communication_runs.csv")).groupby("K").overhead_pct.agg(["mean", "std"])
    plt.figure(figsize=(4.6, 2.9))
    xs = [str(k) for k in d.index]
    plt.bar(xs, d["mean"], yerr=d["std"].fillna(0), capsize=3, color="tab:blue")
    for xi, v, e in zip(xs, d["mean"], d["std"].fillna(0)):
        plt.text(xi, v + e + 0.03 * d["mean"].max(), f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    plt.ylim(0, d["mean"].max() * 1.15)
    plt.xlabel("Federation Interval K (episodes)"); plt.ylabel("Communication Overhead (%)")
    plt.grid(axis="y", ls=":", alpha=.6)
    save("figure4_communication.png")


# Figure 5 - convergence, mean +/- SD over runs, measured convergence episode
def conv_episode(y):
    ma = pd.Series(y).rolling(a.ma, min_periods=1).mean().to_numpy()
    for i in range(len(ma) - a.window + 1):
        seg = ma[i:i + a.window]
        if (seg.max() - seg.min()) / abs(seg.mean()) < a.tol:
            return i + 1
    return np.nan


def _curves(g):
    """one series per run (episode-indexed); a workflow type absent in an episode is skipped."""
    return [r.sort_values("episode").value.dropna().to_numpy() for _, r in g.groupby("run")]


def _panel(ax, d, title):
    conv = {}
    for m, g in d.groupby("algo"):
        runs = _curves(g)
        if not runs:
            continue
        L = min(len(r) for r in runs)
        M = np.vstack([pd.Series(r[:L]).rolling(a.ma, min_periods=1).mean() for r in runs])
        mu, sd = M.mean(0), (M.std(0, ddof=1) if len(runs) > 1 else np.zeros(L))
        ep = np.arange(1, L + 1)
        st = STY.get(m, {})
        ax.plot(ep, mu, color=st.get("color"), lw=2, label=f"{m} ({len(runs)} runs)")
        ax.fill_between(ep, mu - sd, mu + sd, color=st.get("color"), alpha=.2)
        conv[m] = [conv_episode(r) for r in runs]
        c = np.array(conv[m], float)
        if np.any(~np.isnan(c)):
            ax.axvline(np.nanmean(c), color=st.get("color"), ls=":")
    ax.set_title(title); ax.set_xlabel("Training episode")
    ax.grid(ls=":", alpha=.6)
    return conv


if os.path.exists(p("training_log.csv")):
    d = pd.read_csv(p("training_log.csv"))
    d = d[d.algo.isin(["FGRL", "Centralized PPO"])]
    if "workflow" not in d.columns:
        d["workflow"] = "all"
    kinds = [k for k in ["montage", "cybershake", "epigenomics"] if k in set(d.workflow)] or ["all"]
    fig, axes = plt.subplots(1, len(kinds), figsize=(6.3, 2.5), squeeze=False)
    conv_by_kind = {}
    for ax, k in zip(axes[0], kinds):
        conv_by_kind[k] = _panel(ax, d[d.workflow == k], NAME.get(k, k))
    axes[0][0].set_ylabel("Training makespan (s)")
    axes[0][0].legend(fontsize=7, handlelength=1.6)
    save("figure5_convergence.png")
    # one convergence episode per (run, workflow type)  -> 30 values for 10 seeds x 3 workflows
    out = {"definition": f"first episode after which the {a.ma}-episode moving average changes by less than "
                         f"{100 * a.tol:.0f}% (max-min relative to mean) over {a.window} consecutive episodes"}
    pooled = {}
    for k, conv in conv_by_kind.items():
        out[k] = {}
        for m, c in conv.items():
            c = np.array(c, float); ok = c[~np.isnan(c)]
            pooled.setdefault(m, []).extend(list(c))
            out[k][m] = {"runs": len(c), "converged": int(len(ok)),
                         "episode_mean": float(ok.mean()) if len(ok) else None,
                         "episode_sd": float(ok.std(ddof=1)) if len(ok) > 1 else None}
    out["pooled"] = {}
    for m, c in pooled.items():
        c = np.array(c, float); ok = c[~np.isnan(c)]
        out["pooled"][m] = {"runs": len(c), "converged": int(len(ok)),
                            "episode_mean": float(ok.mean()) if len(ok) else None,
                            "episode_sd": float(ok.std(ddof=1)) if len(ok) > 1 else None}
    if all(k in pooled for k in ("FGRL", "Centralized PPO")):
        f = np.array(pooled["FGRL"], float); c = np.array(pooled["Centralized PPO"], float)
        f, c = f[~np.isnan(f)], c[~np.isnan(c)]
        if len(f) > 1 and len(c) > 1:
            out["pooled"]["speedup_ppo_over_fgrl"] = float(c.mean() / f.mean())
            out["pooled"]["mann_whitney_U"] = float(stats.mannwhitneyu(f, c, alternative="two-sided").statistic)
            out["pooled"]["mann_whitney_p_two_sided"] = float(stats.mannwhitneyu(f, c, alternative="two-sided").pvalue)
    # sensitivity of the convergence episode to the stability tolerance
    out["sensitivity"] = {}
    for tol in (0.02, 0.05, 0.10):
        res = {}
        for m, g in d[d.workflow != "all"].groupby("algo") if (d.workflow != "all").any() else d.groupby("algo"):
            c = []
            for _, gk in g.groupby("workflow"):
                for _, r in gk.groupby("run"):
                    y = r.sort_values("episode").value.dropna().to_numpy()
                    ma = pd.Series(y).rolling(a.ma, min_periods=1).mean().to_numpy()
                    ep = np.nan
                    for i in range(len(ma) - a.window + 1):
                        seg = ma[i:i + a.window]
                        if (seg.max() - seg.min()) / abs(seg.mean()) < tol:
                            ep = i + 1; break
                    c.append(ep)
            c = np.array(c, float); ok = c[~np.isnan(c)]
            res[m] = {"runs": len(c), "converged": int(len(ok)),
                      "episode_mean": float(ok.mean()) if len(ok) else None,
                      "episode_sd": float(ok.std(ddof=1)) if len(ok) > 1 else None}
        out["sensitivity"][f"{int(round(tol * 100))}%"] = res
    # model-selection point: episode with the best validation score (the policy that is evaluated)
    if "val_makespan" in d.columns:
        v = d[(d.workflow == "all") & d.val_makespan.notna()]
        best = {}
        for (m, r), g in v.groupby(["algo", "run"]):
            best.setdefault(m, {})[int(r)] = int(g.loc[g.val_makespan.idxmin(), "episode"])
        last = d[d.workflow == "all"].groupby(["algo", "run"]).episode.max()
        out["best_validation_episode"] = {m: {"per_run": b, "mean": float(np.mean(list(b.values()))),
                                              "sd": float(np.std(list(b.values()), ddof=1)) if len(b) > 1 else None}
                                          for m, b in best.items()}
        out["training_length_episodes"] = {m: {"mean": float(last[m].mean()), "min": int(last[m].min()),
                                               "max": int(last[m].max())} for m in last.index.levels[0] if m in last}
        if all(k in best for k in ("FGRL", "Centralized PPO")):
            runs_ = sorted(set(best["FGRL"]) & set(best["Centralized PPO"]))
            f = np.array([best["FGRL"][r] for r in runs_]); c = np.array([best["Centralized PPO"][r] for r in runs_])
            if len(runs_) > 1 and np.any(f != c):
                out["best_validation_episode"]["wilcoxon_paired_p"] = float(stats.wilcoxon(f, c).pvalue)
    json.dump(out, open(p("figure5_convergence.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))

# Figure 6 - overall comparison, per workflow type (top: makespan, bottom: energy)
if os.path.exists(p("comparison_runs.csv")):
    d = pd.read_csv(p("comparison_runs.csv"))
    order = [m for m in ["HEFT", "PSO-Fuzzy", "Centralized PPO", "FGRL"] if m in d.method.unique()]
    ks = kinds_in(d)
    fig, axes = plt.subplots(2, len(ks), figsize=(6.3, 4.2), squeeze=False)
    for j, k in enumerate(ks):
        g = d[d.workflow == k].groupby("method")
        for i, (col, lab) in enumerate([("makespan", "Makespan (s)"), ("energy_kwh", "Energy (kWh)")]):
            ax = axes[i][j]
            mu, sd = g[col].mean().reindex(order), g[col].std().reindex(order).fillna(0)
            ax.bar(range(len(order)), mu, yerr=sd, capsize=3,
                   color=[STY.get(m, {}).get("color", "gray") for m in order])
            ax.set_xticks(range(len(order))); ax.set_xticklabels([o.replace("Centralized PPO", "C-PPO").replace("PSO-Fuzzy", "PSO-F") for o in order], fontsize=7.5)
            ax.grid(axis="y", ls=":", alpha=.6)
            if j == 0: ax.set_ylabel(lab)
            if i == 0: ax.set_title(NAME.get(k, k))
    save("figure6_comparison.png")

# Figure 7 - ablation: makespan relative to full FGRL in the same (seed, workflow) block
if os.path.exists(p("ablation_runs.csv")):
    d = pd.read_csv(p("ablation_runs.csv"))
    if "FGRL" not in set(d.method) and os.path.exists(p("comparison_runs.csv")):
        d = pd.concat([d, pd.read_csv(p("comparison_runs.csv")).query("method == 'FGRL'")], ignore_index=True)
    order = [m for m in ["FGRL w/o GNN", "FGRL w/o critical-path reward", "FGRL w/o federation", "FGRL"] if m in set(d.method)]
    piv = d.pivot_table(index=["seed", "workflow"], columns="method", values="makespan")[order].dropna()
    rel = piv.div(piv["FGRL"], axis=0)
    ks = kinds_in(d)
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    wdt = 0.8 / len(ks)
    labels = {"FGRL w/o GNN": "Without\nGNN", "FGRL w/o critical-path reward": "Without critical-\npath reward",
              "FGRL w/o federation": "Without\nfederation", "FGRL": "Full\nFGRL"}
    labels = [labels[o] for o in order]
    for j, k in enumerate(ks):
        r = rel.xs(k, level=1)
        ax.bar(np.arange(len(order)) + (j - (len(ks) - 1) / 2) * wdt, r.mean(), wdt, yerr=r.std().fillna(0),
               capsize=2, label=NAME.get(k, k))
    ax.axhline(1.0, color="k", lw=1)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels)
    ax.set_ylabel("Makespan relative to full FGRL"); ax.legend(fontsize=7.5)
    ax.grid(axis="y", ls=":", alpha=.6)
    save("figure7_ablation.png")
