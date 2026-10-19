"""Training and evaluation loops.

FGRLTrainer          decentralised multi-agent PPO, one agent per fog node,
                     weighted FedAvg every K episodes (Algorithm 1).
                     Ablations: cfg.use_gnn, cfg.use_critical_reward, cfg.fed.enabled
CentralPPOTrainer    Centralized PPO baseline: one agent, global observation,
                     chooses the target node directly, no federation.
"""
from __future__ import annotations

import copy
import random
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from fgrl.agents.ppo import PPOAgent, Transition
from fgrl.config import ExperimentConfig
from fgrl.federated.fedavg import federated_average, param_bytes
from fgrl.graph.features import WorkflowGraph, build_workflow_graph
from fgrl.sim.env import LOCAL_OBS_DIM, N_LOCAL_ACTIONS, SchedulingEpisode
from fgrl.sim.fogcloud import Node
from fgrl.workflows.generator import makespan_lower_bound


def wf_graph(G, nodes: List[Node]) -> WorkflowGraph:
    fog = nodes[:-1]
    return build_workflow_graph(G, float(np.mean([n.mips for n in fog])),
                                float(np.mean([n.bw_mbps for n in fog])))


def tensors(wg: WorkflowGraph):
    return torch.as_tensor(wg.x), torch.as_tensor(wg.a_norm)


def _credit_deferrals(pending: Dict[int, list], v: int, t: Transition, r: float, info: Dict) -> None:
    """A deferral only postpones the cost of placing the task. When the task is
    finally executed, its execution reward is also credited to every earlier
    'defer' decision for that task, so deferring is judged by its real outcome."""
    if info.get("deferred"):
        pending.setdefault(v, []).append(t)
    else:
        for prev in pending.pop(v, []):
            prev.reward += r


class _Base:
    centralized = False

    def __init__(self, cfg: ExperimentConfig, nodes: List[Node], train_pool: Dict[str, list],
                 val_pool: Optional[Dict[str, list]] = None, seed: int = 0):
        self.cfg, self.nodes, self.seed = cfg, nodes, seed
        self.rng = random.Random(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.train_items = [(k, G) for k, gs in train_pool.items() for G in gs]
        self.val_items = [(k, G) for k, gs in (val_pool or {}).items() for G in gs]
        self.val_items_map = [(id(G), G) for _, G in self.val_items]
        self._wg_cache: Dict[int, WorkflowGraph] = {}
        self.comm_bytes = 0.0
        self.rounds = 0

    def _wg(self, G) -> WorkflowGraph:
        key = id(G)
        if key not in self._wg_cache:
            self._wg_cache[key] = wf_graph(G, self.nodes)
        return self._wg_cache[key]

    # implemented by subclasses
    def run_workflow(self, wf_id, wg, greedy, rng, record=True): ...
    def end_episode(self, graphs): ...
    def state(self): ...
    def load_state(self, s): ...

    # --------------------------------------------------------------- training
    def train(self, episodes: int, log: Optional[Callable[[Dict], None]] = None,
              eval_every: Optional[int] = None, patience: Optional[int] = None) -> List[Dict]:
        eval_every = eval_every or self.cfg.train.eval_every
        patience = patience if patience is not None else self.cfg.train.early_stop_patience
        history, best, best_state, bad = [], float("inf"), None, 0
        for ep in range(1, episodes + 1):
            t0 = time.perf_counter()
            graphs, ms, rewards, per_kind = {}, [], [], {}
            for w in range(self.cfg.train.workflows_per_episode):
                kind, G = self.rng.choice(self.train_items)
                wg = self._wg(G)
                graphs[w] = tensors(wg)
                res = self.run_workflow(w, wg, greedy=False, rng=random.Random(self.rng.random()))
                ms.append(res["makespan"]); rewards.append(res["reward"])
                per_kind.setdefault(kind, []).append(res["makespan"])
            stats = self.end_episode(graphs)
            rec = {"episode": ep, "makespan": float(np.mean(ms)), "reward": float(np.mean(rewards)),
                   "train_time_s": time.perf_counter() - t0,
                   "per_kind": {k: float(np.mean(v)) for k, v in per_kind.items()}, **(stats or {})}
            if self.val_items and ep % eval_every == 0:
                # scale-free validation metric: mean schedule length ratio (makespan / lower bound), so that
                # workflow types with long runtimes (e.g. Epigenomics) do not dominate model selection
                fast = max(n.mips for n in self.nodes)
                val = float(np.mean([m["makespan"] / makespan_lower_bound(dict(self.val_items_map)[m["_g"]], fast)
                                     for m in self.evaluate(self.val_items, seeds=[10_000 + ep], tag_graph=True)]))
                rec["val_makespan"] = val
                if val < best - 1e-9:
                    best, best_state, bad = val, self.state(), 0
                else:
                    bad += 1
                if patience and bad >= patience:
                    history.append(rec)
                    if log: log(rec)
                    break
            history.append(rec)
            if log: log(rec)
        if best_state is not None:
            self.load_state(best_state)
        return history

    # ------------------------------------------------------------- evaluation
    def evaluate(self, items, seeds: List[int], tag_graph: bool = False) -> List[Dict]:
        out = []
        for s in seeds:
            for i, (kind, G) in enumerate(items):
                wg = self._wg(G)
                res = self.run_workflow(-1 - i, wg, greedy=True, rng=random.Random(s * 7919 + i), record=False)
                res.update(kind=kind, seed=s, instance=i)
                if tag_graph:
                    res["_g"] = id(G)
                out.append(res)
        self.clear_eval_cache()
        return out

    def clear_eval_cache(self): ...


class FGRLTrainer(_Base):
    def __init__(self, cfg, nodes, train_pool, val_pool=None, seed=0):
        super().__init__(cfg, nodes, train_pool, val_pool, seed)
        nf = len(nodes) - 1
        self.agents = [PPOAgent(LOCAL_OBS_DIM, N_LOCAL_ACTIONS, cfg.use_gnn, cfg.model, cfg.ppo, seed * 1000 + i)
                       for i in range(nf)]
        if cfg.fed.enabled:   # common initialisation, as in federated learning
            s0 = self.agents[0].net.state_dict()
            for ag in self.agents[1:]:
                ag.net.load_state_dict(s0)
        self.episode = 0

    def run_workflow(self, wf_id, wg, greedy, rng, record=True):
        env = SchedulingEpisode(wg, self.nodes, self.cfg.infra, self.cfg.reward, rng,
                                self.cfg.use_critical_reward, self.cfg.train.max_defers_per_task)
        x, a = tensors(wg)
        acted, total_r, decisions = set(), 0.0, 0
        pending: Dict[int, list] = {}
        while True:
            d = env.next_decision()
            if d is None:
                break
            v, home = d
            ag = self.agents[home]
            obs = env.observe_local(v, home)
            act, logp, val = ag.act(wf_id, wg.index[v], x, a, obs, greedy=greedy)
            r, info = env.step(act)
            total_r += r; decisions += 1
            if record:
                t = Transition(wf_id, wg.index[v], obs, act, logp, val, r)
                ag.store(t)
                acted.add(home)
                _credit_deferrals(pending, v, t, r, info)
        if record:
            for h in acted:
                self.agents[h].mark_done()
        m = env.sim.metrics()
        m["reward"] = total_r / max(decisions, 1)
        return m

    def clear_eval_cache(self):
        for ag in self.agents:
            ag.clear_cache()

    def end_episode(self, graphs):
        self.episode += 1
        stats = [ag.update(graphs) for ag in self.agents]
        stats = [s for s in stats if s]
        out = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]} if stats else {}
        if self.cfg.fed.enabled and self.episode % self.cfg.fed.interval_k == 0:
            t0 = time.perf_counter()
            fs = federated_average(self.agents, self.cfg.fed.variance_weighting)
            self.comm_bytes += fs["bytes_total"]; self.rounds += 1
            out.update(fed_round=1, fed_bytes=fs["bytes_total"], fed_time_s=time.perf_counter() - t0)
        return out

    def state(self):
        return [copy.deepcopy(ag.net.state_dict()) for ag in self.agents]

    def load_state(self, s):
        for ag, st in zip(self.agents, s):
            ag.net.load_state_dict(st)

    def model_bytes_per_node(self) -> int:
        return param_bytes(self.agents[0].net)


class CentralPPOTrainer(_Base):
    centralized = True

    def __init__(self, cfg, nodes, train_pool, val_pool=None, seed=0):
        super().__init__(cfg, nodes, train_pool, val_pool, seed)
        obs_dim = 2 * len(nodes) + 3
        self.agent = PPOAgent(obs_dim, len(nodes) + 1, True, cfg.model, cfg.ppo, seed * 1000)

    def run_workflow(self, wf_id, wg, greedy, rng, record=True):
        env = SchedulingEpisode(wg, self.nodes, self.cfg.infra, self.cfg.reward, rng,
                                self.cfg.use_critical_reward, self.cfg.train.max_defers_per_task,
                                centralized=True)
        x, a = tensors(wg)
        total_r, decisions = 0.0, 0
        pending: Dict[int, list] = {}
        while True:
            d = env.next_decision()
            if d is None:
                break
            v, _ = d
            obs = env.observe_global(v)
            act, logp, val = self.agent.act(wf_id, wg.index[v], x, a, obs, greedy=greedy)
            r, info = env.step_central(act)
            total_r += r; decisions += 1
            if record:
                t = Transition(wf_id, wg.index[v], obs, act, logp, val, r)
                self.agent.store(t)
                _credit_deferrals(pending, v, t, r, info)
        if record:
            self.agent.mark_done()
        m = env.sim.metrics()
        m["reward"] = total_r / max(decisions, 1)
        return m

    def clear_eval_cache(self):
        self.agent.clear_cache()

    def end_episode(self, graphs):
        return self.agent.update(graphs)

    def state(self):
        return copy.deepcopy(self.agent.net.state_dict())

    def load_state(self, s):
        self.agent.net.load_state_dict(s)
