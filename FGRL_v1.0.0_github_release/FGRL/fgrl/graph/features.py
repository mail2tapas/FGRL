"""Task features (Equation 1) and the normalised adjacency used by the GCN (Equation 2).

x_v = [w_v, in_data_v, out_data_v, deadline_slack_v, c_v]

* w_v              computation size (MI), log-scaled
* in_data_v        total input data (MB), log-scaled
* out_data_v       total output data (MB), log-scaled
* deadline_slack_v (deadline - earliest finish on the reference node) / deadline
* c_v              criticality score = 1 - slack(v) / max slack  (1 on the critical path)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import networkx as nx
import numpy as np

from fgrl.graph.critical_path import CPResult, critical_path

FEATURE_DIM = 5


@dataclass
class WorkflowGraph:
    G: nx.DiGraph
    order: List[int]          # topological order
    index: Dict[int, int]     # task id -> row
    x: np.ndarray             # [V, 5] features (Eq. 1)
    a_norm: np.ndarray        # [V, V] D^-1/2 (A + I) D^-1/2  (Eq. 2)
    cp: CPResult              # critical path at arrival (nominal speeds)
    criticality: Dict[int, float]


def build_workflow_graph(G: nx.DiGraph, mean_mips: float, mean_bw_mbps: float) -> WorkflowGraph:
    order = list(nx.topological_sort(G))
    index = {v: i for i, v in enumerate(order)}
    cp = critical_path(G,
                       exec_time=lambda v: G.nodes[v]["work"] / mean_mips,
                       comm_time=lambda u, v: G.edges[u, v]["data"] * 8 / mean_bw_mbps,
                       order=order)
    crit = {v: 1.0 - (cp.slack[v] / cp.max_slack if cp.max_slack > 0 else 0.0) for v in order}
    V = len(order)
    x = np.zeros((V, FEATURE_DIM), dtype=np.float32)
    for v in order:
        i = index[v]
        ind = sum(G.edges[u, v]["data"] for u in G.predecessors(v))
        outd = sum(G.edges[v, w]["data"] for w in G.successors(v))
        dl = G.nodes[v]["deadline"]
        x[i] = [math.log1p(G.nodes[v]["work"] / 1000.0), math.log1p(ind), math.log1p(outd),
                (dl - cp.eft[v]) / dl if dl > 0 else 0.0, crit[v]]
    A = np.zeros((V, V), dtype=np.float32)
    for u, v in G.edges:
        A[index[u], index[v]] = 1.0
        A[index[v], index[u]] = 1.0          # undirected message passing
    A += np.eye(V, dtype=np.float32)
    d = 1.0 / np.sqrt(A.sum(1))
    a_norm = (A * d[:, None]) * d[None, :]
    G.graph.setdefault("lft_scale", 1.5)   # tolerance on the latest finish time used by r_critical
    return WorkflowGraph(G, order, index, x, a_norm, cp, crit)
