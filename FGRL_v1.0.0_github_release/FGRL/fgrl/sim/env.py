"""Decentralised scheduling episode (Dec-MDP of Section 3.2).

Each task becomes *ready* at the fog node that holds its input data (its
"home" node). The PPO agent of that fog node observes its local state and
chooses one of four actions (Section 3.2):

    0  execute locally on the home fog node
    1  offload to the best neighbouring fog node (earliest observed finish)
    2  offload to the cloud
    3  defer: put the task back in the local queue for later scheduling

Entry tasks are released at fog nodes chosen at random (IoT sources).
A centralised mode is also provided for the Centralized PPO baseline: one
agent sees every node and chooses the target node directly.

Reward, Equation (5):
    r = w1 r_lat + w2 r_en + w3 r_SLA + w4 r_util + lambda r_critical
with the critical-path term, Equation (6):
    r_critical = alpha (1 - slack(v)/max slack) * 1[v in CP_t] * s_v
where s_v = +1 if v finishes by its latest finish time LFT_v and -1 otherwise.
(The sign s_v makes the term depend on the scheduling decision; without it
the term would be a constant for each task and give no learning signal.)
"""
from __future__ import annotations

import heapq
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from fgrl.config import InfraConfig, RewardConfig
from fgrl.graph.features import WorkflowGraph
from fgrl.sim.fogcloud import Node, WorkflowSim

LOCAL, NEIGHBOR, CLOUD, DEFER = 0, 1, 2, 3
N_LOCAL_ACTIONS = 4
LOCAL_OBS_DIM = 13


class SchedulingEpisode:
    def __init__(self, wg: WorkflowGraph, nodes: List[Node], infra: InfraConfig, rew: RewardConfig,
                 rng: random.Random, use_critical_reward: bool = True, max_defers: int = 2,
                 centralized: bool = False):
        self.wg, self.nodes, self.infra, self.rew = wg, nodes, infra, rew
        self.G = wg.G
        self.rng = rng
        self.sim = WorkflowSim(self.G, nodes, infra, rng)
        self.nf = len(nodes) - 1
        self.cloud = self.nf
        self.use_cp = use_critical_reward
        self.max_defers = max_defers
        self.centralized = centralized
        self.defers: Dict[int, int] = {v: 0 for v in self.G.nodes}
        self.home: Dict[int, int] = {}
        self.n_placed_preds: Dict[int, int] = {v: 0 for v in self.G.nodes}
        self.heap: List[Tuple[float, float, int]] = []
        self.now = 0.0
        self.mean_mips = float(np.mean([n.mips for n in nodes[:-1]]))
        for v in self.G.nodes:
            if self.G.in_degree(v) == 0:
                self.home[v] = rng.randrange(self.nf)
                self._push(v, 0.0)
        self.current: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------- queue
    def _push(self, v: int, release: float):
        heapq.heappush(self.heap, (release, -self.wg.criticality[v], v))

    def next_decision(self) -> Optional[Tuple[int, int]]:
        if not self.heap:
            return None
        release, _, v = heapq.heappop(self.heap)
        self.now = max(self.now, release)
        self.current = (v, self.home[v])
        return self.current

    # ------------------------------------------------------- observations
    def _scale(self, v: int) -> float:
        return self.G.nodes[v]["work"] / self.mean_mips + 1e-3

    def _backlog(self, n: int) -> float:
        return max(0.0, self.sim.avail[n] - self.now)

    def observe_local(self, v: int, home: int) -> np.ndarray:
        sc = self._scale(v)
        util = self.sim.utilization_vector(self.now) if self.now > 0 else [0.0] * len(self.nodes)
        nb = self.nodes[home].neighbors or [home]
        cloud_comm = sum(self.sim.comm(self.sim.place_of[u], self.cloud, self.G.edges[u, v]["data"])
                         for u in self.G.predecessors(v)) if self.G.in_degree(v) else \
            self.sim.comm(home, self.cloud, 0.0)
        # partial resource information from the neighbourhood: estimated finish
        # time of v for each placement option, relative to now (in task units)
        est = [float(np.log1p(max(0.0, self._observed_finish(v, self.target_of(v, home, a)) - self.now) / sc))
               for a in (LOCAL, NEIGHBOR, CLOUD)]
        return np.array(est + [
            self.sim.slowdown[home],
            float(np.log1p(self._backlog(home) / sc)),
            util[home],
            float(np.log1p(min(self._backlog(n) for n in nb) / sc)),
            min(self.sim.slowdown[n] for n in nb),
            float(np.mean([util[n] for n in nb])),
            float(np.log1p(cloud_comm / sc)),
            self.wg.criticality[v],
            1.0 if v in self.wg.cp.critical else 0.0,
            self.defers[v] / max(1, self.max_defers),
        ], dtype=np.float32)

    def global_obs_dim(self) -> int:
        return 2 * len(self.nodes) + 3

    def observe_global(self, v: int) -> np.ndarray:
        sc = self._scale(v)
        feats = []
        for n in range(len(self.nodes)):
            feats += [float(np.log1p(max(0.0, self._observed_finish(v, n) - self.now) / sc)), self.sim.slowdown[n]]
        feats += [self.wg.criticality[v], 1.0 if v in self.wg.cp.critical else 0.0,
                  self.defers[v] / max(1, self.max_defers)]
        return np.array(feats, dtype=np.float32)

    # ------------------------------------------------------------- actions
    def _observed_finish(self, v: int, n: int) -> float:
        # what an agent can estimate: queue backlog + nominal time x observed slowdown
        return max(self.sim.ready_time(v, n), self.sim.avail[n]) + \
            self.sim.exec_time(v, n, nominal=True) * self.sim.slowdown[n]

    def target_of(self, v: int, home: int, action: int) -> Optional[int]:
        if action == LOCAL:
            return home
        if action == NEIGHBOR:
            nb = self.nodes[home].neighbors or [home]
            return min(nb, key=lambda n: self._observed_finish(v, n))
        if action == CLOUD:
            return self.cloud
        return None   # defer

    def step(self, action: int) -> Tuple[float, Dict]:
        """Decentralised step: action in {0,1,2,3}."""
        v, home = self.current
        if action == DEFER and self.defers[v] < self.max_defers:
            return self._defer(v, home)
        target = self.target_of(v, home, LOCAL if action == DEFER else action)
        return self._execute(v, home, target)

    def step_central(self, target: int) -> Tuple[float, Dict]:
        """Centralised step: target in {0..N_f} (node index) or N_f+1 = defer."""
        v, home = self.current
        if target == len(self.nodes) and self.defers[v] < self.max_defers:
            return self._defer(v, home)
        if target == len(self.nodes):
            target = home
        return self._execute(v, home, target)

    def _defer(self, v: int, home: int) -> Tuple[float, Dict]:
        # the waiting time is charged as latency now; the later decision is
        # charged from its new release time, so the total equals the true delay
        delay = 0.25 * self.sim.exec_time(v, home, nominal=True)
        self.defers[v] += 1
        self._push(v, self.now + delay)
        r = -self.rew.w_lat * delay / self._scale(v) - self.rew.defer_penalty
        return float(r), {"deferred": True}

    def _execute(self, v: int, home: int, target: int) -> Tuple[float, Dict]:
        info = self.sim.place(v, target)
        r = self.reward(v, info, target)
        for w in self.G.successors(v):
            self.n_placed_preds[w] += 1
            if self.n_placed_preds[w] == self.G.in_degree(w):
                # task becomes ready at the fog node that holds most of its input
                src = max(self.G.predecessors(w), key=lambda u: self.G.edges[u, w]["data"])
                self.home[w] = self.home[src] if self.sim.place_of[src] == self.cloud else self.sim.place_of[src]
                self._push(w, max(self.sim.ft[u] for u in self.G.predecessors(w)))
        info.update(target=target, deferred=False)
        return r, info

    # -------------------------------------------------------------- reward
    def reward(self, v: int, info: Dict, target: int) -> float:
        rc = self.rew
        sc = self._scale(v)
        r_lat = -min((info["finish"] - self.now) / sc, 10.0)   # latency from this decision
        node = self.nodes[target]
        ref_dyn = 60.0 * sc            # Tier 2 dynamic power x nominal time
        r_en = -min(info["energy_j"] / ref_dyn, 10.0)
        r_sla = 1.0 if info["finish"] <= info["deadline"] else -1.0
        util = self.sim.utilization_vector(max(info["finish"], 1e-9))[:-1]
        r_util = -float(np.std(util))
        r = rc.w_lat * r_lat + rc.w_en * r_en + rc.w_sla * r_sla + rc.w_util * r_util
        if self.use_cp and v in self.wg.cp.critical:
            cp = self.wg.cp
            weight = 1.0 - (cp.slack[v] / cp.max_slack if cp.max_slack > 0 else 0.0)
            on_time = 1.0 if info["finish"] <= cp.lft[v] * self.G.graph.get("lft_scale", 1.0) + 1e-9 else -1.0
            r += rc.lam_critical * rc.alpha_critical * weight * on_time
        return float(r)

    def done(self) -> bool:
        return self.sim.done()
