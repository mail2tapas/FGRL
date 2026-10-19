"""Static baselines executed on the same simulator.

HEFT (Topcuoglu et al., 2002)
    Upward rank with average computation and communication costs, then each
    task (in decreasing rank) is mapped to the node with the earliest
    *estimated* finish time (non-insertion variant). Planning uses nominal
    speeds; the plan is then executed under the actual (dynamic) conditions,
    so HEFT cannot react to load spikes - exactly the limitation discussed in
    the paper.

PSO-Fuzzy (in the spirit of Pan et al., 2024)
    Particle swarm optimisation over task-to-node assignments. The fitness is
    a fuzzy aggregation of normalised makespan and energy (triangular
    membership functions combined with a min operator). Like HEFT it plans
    with nominal speeds and is then executed under actual conditions.
    NOTE: this is a re-implementation of the general PSO-plus-fuzzy idea, not
    the exact Lyapunov/LSTM pipeline of Pan et al.; describe it as such.
"""
from __future__ import annotations

import random
from typing import Dict, List

import networkx as nx
import numpy as np

from fgrl.config import InfraConfig
from fgrl.sim.fogcloud import Node, WorkflowSim


def _execute(G, nodes, cfg, rng, order: List[int], assign: Dict[int, int]) -> Dict[str, float]:
    sim = WorkflowSim(G, nodes, cfg, rng)
    for v in order:
        sim.place(v, assign[v])
    return sim.metrics()


def heft_plan(G: nx.DiGraph, nodes: List[Node], cfg: InfraConfig):
    mips = np.array([n.mips for n in nodes])
    avg_w = {v: float(np.mean(G.nodes[v]["work"] / mips)) for v in G.nodes}
    avg_bw = float(np.mean([n.bw_mbps for n in nodes]))
    avg_c = {(u, v): G.edges[u, v]["data"] * 8 / avg_bw for u, v in G.edges}
    rank: Dict[int, float] = {}
    for v in reversed(list(nx.topological_sort(G))):
        rank[v] = avg_w[v] + max((avg_c[(v, w)] + rank[w] for w in G.successors(v)), default=0.0)
    order = sorted(G.nodes, key=lambda v: -rank[v])
    planner = WorkflowSim(G, nodes, cfg, random.Random(0))
    planner.slowdown = [1.0] * len(nodes)          # nominal speeds
    planner.noise = {v: 1.0 for v in G.nodes}
    assign = {}
    for v in order:
        best = min(range(len(nodes)), key=lambda n: planner.estimate_finish(v, n, nominal=True))
        planner.place(v, best)
        assign[v] = best
    return order, assign


def heft(G, nodes, cfg, rng) -> Dict[str, float]:
    order, assign = heft_plan(G, nodes, cfg)
    return _execute(G, nodes, cfg, rng, order, assign)


def _nominal_eval(G, nodes, cfg, order, assign):
    sim = WorkflowSim(G, nodes, cfg, random.Random(0))
    sim.slowdown = [1.0] * len(nodes)
    sim.noise = {v: 1.0 for v in G.nodes}
    for v in order:
        sim.place(v, assign[v])
    return sim.metrics()


def _tri(x, lo, hi):
    """Membership of 'good' for a cost x: 1 at lo, 0 at hi (linear)."""
    if hi <= lo:
        return 1.0
    return float(np.clip((hi - x) / (hi - lo), 0.0, 1.0))


def pso_fuzzy(G, nodes, cfg, rng, particles: int = 20, iters: int = 30, seed: int = 0) -> Dict[str, float]:
    order, heft_assign = heft_plan(G, nodes, cfg)
    V, N = len(order), len(nodes)
    r = np.random.default_rng(seed)
    # continuous positions in [0, N); node = floor(position)
    pos = r.uniform(0, N, size=(particles, V))
    pos[0] = np.array([heft_assign[v] + 0.5 for v in order])  # seed with HEFT
    vel = r.normal(0, 1, size=(particles, V))
    ref = _nominal_eval(G, nodes, cfg, order, heft_assign)

    def fitness(p):
        assign = {v: int(min(N - 1, max(0, p[i]))) for i, v in enumerate(order)}
        m = _nominal_eval(G, nodes, cfg, order, assign)
        mu_ms = _tri(m["makespan"], 0.5 * ref["makespan"], 2.0 * ref["makespan"])
        mu_en = _tri(m["energy_kwh"], 0.5 * ref["energy_kwh"], 2.0 * ref["energy_kwh"])
        return min(mu_ms, mu_en) + 0.1 * (mu_ms + mu_en)   # fuzzy AND + tie-break

    fit = np.array([fitness(p) for p in pos])
    pbest, pfit = pos.copy(), fit.copy()
    g = int(np.argmax(fit)); gbest, gfit = pos[g].copy(), fit[g]
    for _ in range(iters):
        r1, r2 = r.random((particles, V)), r.random((particles, V))
        vel = 0.7 * vel + 1.5 * r1 * (pbest - pos) + 1.5 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, 0, N - 1e-6)
        fit = np.array([fitness(p) for p in pos])
        better = fit > pfit
        pbest[better], pfit[better] = pos[better], fit[better]
        if fit.max() > gfit:
            g = int(np.argmax(fit)); gbest, gfit = pos[g].copy(), fit[g]
    assign = {v: int(gbest[i]) for i, v in enumerate(order)}
    return _execute(G, nodes, cfg, rng, order, assign)
