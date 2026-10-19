"""Discrete-event fog-cloud simulator (Section 4 and Section 4.5).

* Three-tier infrastructure (Table 4): Tier 1 edge gateways, Tier 2 edge
  servers and one Tier 3 cloud data centre.
* Each node executes one task at a time (non-preemptive).
* Precedence constraint, Equation (9):
      ST_j >= FT_i + d_ij / B_ij  for every edge (i, j)
      FT_j  = ST_j + w_j / C_assigned
* Power model:  P_i(u) = P_idle + (P_peak - P_idle) * u^3
* Dynamic conditions: per-workflow load spikes on random nodes and
  log-normal execution-time noise. Static schedulers plan with nominal
  speeds; learning agents observe the local slowdown and queue backlog.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List

import networkx as nx

from fgrl.config import InfraConfig


@dataclass
class Node:
    idx: int
    tier: str
    mips: float
    memory_gb: float
    bw_mbps: float
    p_idle: float
    p_peak: float
    is_cloud: bool
    neighbors: List[int]


def build_infrastructure(cfg: InfraConfig, seed: int) -> List[Node]:
    rng = random.Random(seed)
    t1, t2, t3 = cfg.tiers
    n_t1 = int(round(cfg.num_fog_nodes * cfg.tier1_fraction))
    nodes: List[Node] = []
    for i in range(cfg.num_fog_nodes):
        t = t1 if i < n_t1 else t2
        nodes.append(Node(i, t.name, t.mips, t.memory_gb, t.bandwidth_mbps, t.p_idle, t.p_peak, False, []))
    rng.shuffle(nodes)
    for i, n in enumerate(nodes):
        n.idx = i
    nf = len(nodes)
    for n in nodes:  # ring neighbours + random extra links (connected topology)
        k = min(cfg.neighbors_per_node, nf - 1)
        cand = [(n.idx + d) % nf for d in range(1, nf)]
        ring = [(n.idx + 1) % nf, (n.idx - 1) % nf] if nf > 2 else cand[:1]
        rest = [c for c in cand if c not in ring]
        rng.shuffle(rest)
        n.neighbors = sorted(set((ring + rest)[:k]) - {n.idx})
    nodes.append(Node(nf, t3.name, t3.mips, t3.memory_gb, t3.bandwidth_mbps, t3.p_idle, t3.p_peak, True, []))
    return nodes


class WorkflowSim:
    """Executes one workflow DAG on the infrastructure, task by task."""

    def __init__(self, G: nx.DiGraph, nodes: List[Node], cfg: InfraConfig, rng: random.Random):
        self.G, self.nodes, self.cfg = G, nodes, cfg
        self.cloud = len(nodes) - 1
        self.slowdown = [1.0] * len(nodes)
        for n in nodes[:-1]:
            if rng.random() < cfg.load_spike_prob:
                self.slowdown[n.idx] = cfg.load_spike_slowdown
        s = cfg.exec_noise_cv
        mu = -0.5 * math.log(1 + s * s)
        sig = math.sqrt(math.log(1 + s * s))
        self.noise = {v: rng.lognormvariate(mu, sig) if s > 0 else 1.0 for v in G.nodes}
        self.avail = [0.0] * len(nodes)
        self.busy = [0.0] * len(nodes)
        self.place_of: Dict[int, int] = {}
        self.ft: Dict[int, float] = {}
        self.st: Dict[int, float] = {}

    # ---------------------------------------------------------------- timing
    def bandwidth(self, a: int, b: int) -> float:
        return min(self.nodes[a].bw_mbps, self.nodes[b].bw_mbps)

    def comm(self, a: int, b: int, mb: float) -> float:
        if a == b:
            return 0.0
        lat = self.cfg.cloud_latency_s if (self.nodes[a].is_cloud or self.nodes[b].is_cloud) \
            else self.cfg.fog_link_latency_s
        return mb * 8.0 / self.bandwidth(a, b) + lat

    def exec_time(self, v: int, n: int, nominal: bool = False) -> float:
        base = self.G.nodes[v]["work"] / self.nodes[n].mips
        return base if nominal else base * self.slowdown[n] * self.noise[v]

    def ready_time(self, v: int, n: int) -> float:
        t = 0.0
        for u in self.G.predecessors(v):
            t = max(t, self.ft[u] + self.comm(self.place_of[u], n, self.G.edges[u, v]["data"]))
        return t

    def estimate_finish(self, v: int, n: int, nominal: bool = False) -> float:
        return max(self.ready_time(v, n), self.avail[n]) + self.exec_time(v, n, nominal)

    # ------------------------------------------------------------- execution
    def place(self, v: int, n: int) -> Dict[str, float]:
        assert v not in self.ft, f"task {v} already placed"
        assert all(u in self.ft for u in self.G.predecessors(v)), "precedence violated"
        rt = self.ready_time(v, n)
        st = max(rt, self.avail[n])
        et = self.exec_time(v, n)
        ft = st + et
        self.avail[n] = ft
        self.busy[n] += et
        self.place_of[v], self.st[v], self.ft[v] = n, st, ft
        node = self.nodes[n]
        e_dyn = (node.p_peak - node.p_idle) * et  # Joules at full utilisation
        return {"start": st, "finish": ft, "exec": et, "ready": rt, "wait": st - rt,
                "energy_j": e_dyn, "deadline": self.G.nodes[v]["deadline"]}

    def done(self) -> bool:
        return len(self.ft) == self.G.number_of_nodes()

    # --------------------------------------------------------------- metrics
    def utilization_vector(self, now: float | None = None) -> List[float]:
        T = now if now is not None else max(self.ft.values(), default=0.0)
        T = max(T, 1e-9)
        return [min(1.0, b / T) for b in self.busy]

    def metrics(self) -> Dict[str, float]:
        M = max(self.ft.values())
        util = self.utilization_vector(M)
        energy = 0.0
        for n, u in zip(self.nodes, util):
            if n.is_cloud and self.busy[n.idx] == 0:
                continue
            energy += M * (n.p_idle + (n.p_peak - n.p_idle) * u ** 3)   # cubic model
        sla = sum(1 for v in self.G.nodes if self.ft[v] > self.G.nodes[v]["deadline"]) / self.G.number_of_nodes()
        fog_util = util[:-1]
        return {"makespan": M,
                "energy_kwh": energy / 3.6e6,
                "sla_violation": sla,
                "utilization": sum(fog_util) / len(fog_util)}
