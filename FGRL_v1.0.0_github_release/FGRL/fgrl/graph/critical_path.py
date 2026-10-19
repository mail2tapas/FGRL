"""Critical path and slack via a forward-backward traversal (Algorithm 1, line 9).

Complexity is O(|V| + |E|): one pass in topological order for earliest
start/finish times and one pass in reverse order for latest finish times.
slack(v) = LST(v) - EST(v); tasks with (numerically) zero slack form CP_t.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Set

import networkx as nx


@dataclass
class CPResult:
    est: Dict[int, float]
    eft: Dict[int, float]
    lst: Dict[int, float]
    lft: Dict[int, float]
    slack: Dict[int, float]
    critical: Set[int]
    length: float
    max_slack: float


def critical_path(G: nx.DiGraph,
                  exec_time: Callable[[int], float],
                  comm_time: Callable[[int, int], float],
                  order: List[int] | None = None,
                  tol: float = 1e-9) -> CPResult:
    order = order or list(nx.topological_sort(G))
    est: Dict[int, float] = {}
    eft: Dict[int, float] = {}
    for v in order:                                   # forward pass
        s = 0.0
        for u in G.predecessors(v):
            s = max(s, eft[u] + comm_time(u, v))
        est[v] = s
        eft[v] = s + exec_time(v)
    length = max(eft.values()) if eft else 0.0
    lft: Dict[int, float] = {}
    lst: Dict[int, float] = {}
    for v in reversed(order):                         # backward pass
        f = length
        for w in G.successors(v):
            f = min(f, lst[w] - comm_time(v, w))
        lft[v] = f
        lst[v] = f - exec_time(v)
    slack = {v: max(0.0, lst[v] - est[v]) for v in order}
    critical = {v for v in order if slack[v] <= tol * max(1.0, length)}
    return CPResult(est, eft, lst, lft, slack, critical, length,
                    max(slack.values()) if slack else 0.0)
