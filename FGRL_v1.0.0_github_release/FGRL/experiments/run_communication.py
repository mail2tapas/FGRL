"""Communication overhead versus federation interval K (Figure 4) and the
per-node memory / inference-latency figures (Section 4.2.1, Section 5.3).

Communication overhead (%) = T_comm / (T_train + T_comm)
  T_train  measured wall-clock training time on this machine
  T_comm   modelled transfer time of all federated rounds: in each round every
           fog node uploads and downloads its parameters over its own link
           (Table 4 bandwidth), rounds are synchronous, so a round costs
           max_i( 2 * |theta| * 8 / B_i ) + 2 * WAN latency.
Because T_train is measured on the machine you run on, report the hardware
together with the result.

    python experiments/run_communication.py --config configs/test.json --out results/test
"""
import argparse
import copy
import json
import os
import platform
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fgrl.agents.trainer import FGRLTrainer, tensors          # noqa: E402
from fgrl.config import load_config                          # noqa: E402
from fgrl.eval.runner import pools_for_seed                   # noqa: E402
from fgrl.federated.fedavg import param_bytes                 # noqa: E402
from fgrl.sim.env import SchedulingEpisode                    # noqa: E402
from fgrl.sim.fogcloud import build_infrastructure            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", nargs="*", type=int, default=None, help="override the seeds of the config (to split a run across machines)")
a = ap.parse_args()
base = load_config(a.config)
x = json.load(open(a.config)).get("experiment", {})
ks = x.get("k_values", [10, 25, 50, 100, 200])
seeds = a.seeds if a.seeds else x.get("seeds", list(range(10)))[: x.get("comm_seeds", 3)]
episodes = x.get("episodes", base.train.episodes)
os.makedirs(a.out, exist_ok=True)

rows = []
for K in ks:
    for s in seeds:
        cfg = copy.deepcopy(base)
        cfg.fed.interval_k = K
        nodes = build_infrastructure(cfg.infra, s)
        train, val, _ = pools_for_seed(cfg, s)
        tr = FGRLTrainer(cfg, nodes, train, None, seed=s)
        t0 = time.perf_counter()
        hist = tr.train(episodes, patience=0)
        t_train = time.perf_counter() - t0
        pb = param_bytes(tr.agents[0].net)
        per_round = max(2 * pb * 8 / (n.bw_mbps * 1e6) for n in nodes[:-1]) + 2 * cfg.infra.cloud_latency_s
        t_comm = tr.rounds * per_round
        rows.append({"K": K, "seed": s, "rounds": tr.rounds, "param_bytes": pb,
                     "bytes_total": tr.comm_bytes, "t_train_s": t_train, "t_comm_s": t_comm,
                     "overhead_pct": 100 * t_comm / (t_train + t_comm)})
        print(rows[-1])
df = pd.DataFrame(rows)
df.to_csv(os.path.join(a.out, "communication_runs.csv"), index=False)
df.groupby("K").overhead_pct.agg(["mean", "std"]).to_csv(os.path.join(a.out, "figure4_communication.csv"))

# ---- per-node footprint and inference latency on THIS machine
cfg = copy.deepcopy(base)
nodes = build_infrastructure(cfg.infra, 0)
train, _, test = pools_for_seed(cfg, 0)
tr = FGRLTrainer(cfg, nodes, train, None, seed=0)
ag = tr.agents[0]
model_mb = param_bytes(ag.net) / 1e6
G = test[cfg.workflow_types[0]][0]
wg = tr._wg(G)
xt, at = tensors(wg)
obs_bytes = 13 * 4 + 6 * 8          # obs vector + scalars per transition
buffer_mb = cfg.train.workflows_per_episode * G.number_of_nodes() * obs_bytes / 1e6
env = SchedulingEpisode(wg, nodes, cfg.infra, cfg.reward, random.Random(0))
lat = []
while (d := env.next_decision()) is not None:
    v, h = d
    t0 = time.perf_counter()
    ag.emb_cache.clear()                              # include the GCN forward pass
    o = env.observe_local(v, h)
    act, _, _ = ag.act(0, wg.index[v], xt, at, o, greedy=True)
    lat.append(1000 * (time.perf_counter() - t0))
    env.step(act)
foot = {"machine": platform.processor() or platform.machine(), "python": platform.python_version(),
        "torch": torch.__version__, "threads": torch.get_num_threads(),
        "model_params_MB_per_node": model_mb, "experience_buffer_MB_per_node_upper_bound": buffer_mb,
        "workflow_tasks": G.number_of_nodes(),
        "inference_ms_per_decision_mean": float(np.mean(lat)), "inference_ms_per_decision_p95": float(np.percentile(lat, 95))}
json.dump(foot, open(os.path.join(a.out, "footprint_latency.json"), "w"), indent=2)
print(json.dumps(foot, indent=2))
