"""Shared experiment protocol used by all scripts in ``experiments/``.

Protocol (Section 4 / 5 of the paper)
-------------------------------------
For every seed s:
  * workflow instances are generated (or loaded) separately for training,
    validation (early stopping) and testing, so test workflows are never
    seen during training (no train/test leakage);
  * every learning method is trained from scratch with seed s;
  * every method is evaluated on the same held-out test workflows under the
    same dynamic conditions (identical load-spike / noise realisations).
One *block* = (seed, workflow type); with 10 seeds x 3 types = 30 blocks,
which are the "30 independent runs" reported in the paper.
"""
from __future__ import annotations

import copy
import glob
import os
import random
import time

from fgrl.workflows.generator import makespan_lower_bound
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from fgrl.agents.trainer import CentralPPOTrainer, FGRLTrainer
from fgrl.baselines.heuristics import heft, pso_fuzzy
from fgrl.config import ExperimentConfig
from fgrl.sim.fogcloud import build_infrastructure
from fgrl.workflows.generator import load_wfcommons, load_workflow_pool

LEARNING = {
    "FGRL": dict(),
    "FGRL w/o GNN": dict(use_gnn=False),
    "FGRL w/o critical-path reward": dict(use_critical_reward=False),
    "FGRL w/o federation": dict(fed_enabled=False),
    "Centralized PPO": dict(central=True),
}
STATIC = {"HEFT": heft, "PSO-Fuzzy": pso_fuzzy}


def _variant(cfg: ExperimentConfig, spec: Dict) -> ExperimentConfig:
    c = copy.deepcopy(cfg)
    if "use_gnn" in spec: c.use_gnn = spec["use_gnn"]
    if "use_critical_reward" in spec: c.use_critical_reward = spec["use_critical_reward"]
    if "fed_enabled" in spec: c.fed.enabled = spec["fed_enabled"]
    return c


def pools_for_seed(cfg: ExperimentConfig, seed: int, n_train=5, n_val=2, n_test=3):
    """Disjoint train / validation / test workflow instances for one seed.

    With cfg.workflow_dir set, the WfCommons JSON files of each workflow type are
    shuffled with the seed and split 60 / 20 / 20 (at least one file per split);
    otherwise synthetic instances are generated with three different seeds.
    """
    kinds, size, wdir = cfg.workflow_types, cfg.workflow_size, cfg.workflow_dir
    if getattr(cfg, "pegasus_dir", ""):
        from fgrl.workflows.generator import load_dax, pegasus_file, trace_variant
        train, val, test = {}, {}, {}
        for k in kinds:
            base = load_dax(pegasus_file(cfg.pegasus_dir, k, size))
            r_tr, r_va, r_te = (random.Random(o + seed) for o in (10_000, 20_000, 30_000))
            train[k] = [trace_variant(base, r_tr) for _ in range(n_train)]
            val[k] = [trace_variant(base, r_va) for _ in range(n_val)]
            # the unmodified benchmark instance is always part of the test set
            test[k] = [base] + [trace_variant(base, r_te) for _ in range(n_test - 1)]
        return train, val, test
    if wdir:
        train, val, test = {}, {}, {}
        for k in kinds:
            files = sorted(glob.glob(os.path.join(wdir, f"{k}*.json")))
            if len(files) < 3:
                raise ValueError(f"need at least 3 WfCommons files for '{k}' in {wdir}")
            random.Random(seed).shuffle(files)
            nt = max(1, int(round(0.2 * len(files))))
            test[k] = [load_wfcommons(f) for f in files[:nt]]
            val[k] = [load_wfcommons(f) for f in files[nt:2 * nt]]
            train[k] = [load_wfcommons(f) for f in files[2 * nt:]]
        return train, val, test
    train = load_workflow_pool(kinds, size, seed=10_000 + seed, count=n_train)
    val = load_workflow_pool(kinds, size, seed=20_000 + seed, count=n_val)
    test = load_workflow_pool(kinds, size, seed=30_000 + seed, count=n_test)
    return train, val, test


def run_experiment(cfg: ExperimentConfig, methods: List[str], seeds: List[int], episodes: int,
                   eval_seeds: int = 3, log: Optional[Callable[[str], None]] = None):
    """Returns (runs DataFrame, training-log DataFrame)."""
    log = log or print
    rows, train_rows = [], []
    for s in seeds:
        nodes = build_infrastructure(cfg.infra, seed=s)
        train, val, test = pools_for_seed(cfg, s)
        test_items = [(k, G) for k, gs in test.items() for G in gs]
        dyn_seeds = [100 * s + j for j in range(eval_seeds)]
        for m in methods:
            t0 = time.perf_counter()
            if m in STATIC:
                res = []
                for ds in dyn_seeds:
                    for i, (k, G) in enumerate(test_items):
                        r = STATIC[m](G, nodes, cfg.infra, random.Random(ds * 7919 + i))
                        r.update(kind=k, seed=ds, instance=i)
                        res.append(r)
            else:
                spec = LEARNING[m]
                c = _variant(cfg, spec)
                T = CentralPPOTrainer if spec.get("central") else FGRLTrainer
                tr = T(c, nodes, train, val, seed=s)
                hist = tr.train(episodes)
                for h in hist:
                    train_rows.append({"algo": m, "run": s, "episode": h["episode"], "value": h["makespan"],
                                       "reward": h["reward"], "train_time_s": h["train_time_s"],
                                       "fed_bytes": h.get("fed_bytes", 0.0), "workflow": "all",
                                       "val_makespan": h.get("val_makespan")})
                    for kk, vv in h.get("per_kind", {}).items():
                        train_rows.append({"algo": m, "run": s, "episode": h["episode"], "value": vv,
                                           "workflow": kk})
                res = tr.evaluate(test_items, dyn_seeds)
            df = pd.DataFrame(res)
            fast = max(n.mips for n in nodes)
            lbs = {i: makespan_lower_bound(G, fast) for i, (k, G) in enumerate(test_items)}
            df["slr"] = [r_["makespan"] / lbs[r_["instance"]] for r_ in res]
            for k, g in df.groupby("kind"):
                rows.append({"method": m, "seed": s, "workflow": k,
                             "makespan": g["makespan"].mean(), "energy_kwh": g["energy_kwh"].mean(),
                             "sla_violation_pct": 100 * g["sla_violation"].mean(),
                             "utilization_pct": 100 * g["utilization"].mean(), "slr": g["slr"].mean()})
            log(f"seed={s} method={m:32s} makespan={df['makespan'].mean():8.2f}s "
                f"({time.perf_counter() - t0:.1f}s)")
    return pd.DataFrame(rows), pd.DataFrame(train_rows)


def summary_table(runs: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    out = []
    for m in order:
        g = runs[runs["method"] == m]
        if g.empty:
            continue
        out.append({"Approach": m,
                    "Makespan (s)": f"{g.makespan.mean():.1f} ± {g.makespan.std(ddof=1):.1f}",
                    "Energy (kWh)": f"{g.energy_kwh.mean():.4f}",
                    "SLA Violation (%)": f"{g.sla_violation_pct.mean():.1f}",
                    "Avg. Utilization (%)": f"{g.utilization_pct.mean():.1f}",
                    "n_blocks": len(g)})
    return pd.DataFrame(out)
