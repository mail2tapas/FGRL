"""PPO agent with clipped surrogate objective (Equation 7), written in PyTorch.

    L^CLIP(theta) = E_t[ min( r_t(theta) A_t, clip(r_t(theta), 1-eps, 1+eps) A_t ) ]

Each agent owns a GCN (or MLP) encoder and an actor-critic head. Task
embeddings are computed once per workflow for acting (Algorithm 1, line 8)
and recomputed with gradients during the update so the encoder is trained
end to end with the policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from fgrl.config import ModelConfig, PPOConfig
from fgrl.models.networks import SchedulerNet


@dataclass
class Transition:
    wf: int
    row: int
    obs: np.ndarray
    action: int
    logp: float
    value: float
    reward: float
    done: bool = False


@dataclass
class Buffer:
    items: List[Transition] = field(default_factory=list)

    def __len__(self):
        return len(self.items)


class PPOAgent:
    def __init__(self, obs_dim: int, n_actions: int, use_gnn: bool, mcfg: ModelConfig,
                 pcfg: PPOConfig, seed: int):
        torch.manual_seed(seed)
        self.net = SchedulerNet(obs_dim, n_actions, use_gnn, mcfg.gcn_hidden, mcfg.gcn_layers,
                                mcfg.policy_hidden, mcfg.dropout)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=pcfg.lr, weight_decay=mcfg.weight_decay)
        self.p = pcfg
        self.buf = Buffer()
        self.emb_cache: Dict[int, torch.Tensor] = {}
        self.loss_history: List[float] = []      # policy losses since last federated round
        self.samples_since_round = 0

    # ----------------------------------------------------------------- acting
    def embeddings(self, wf: int, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if wf not in self.emb_cache:
            self.net.eval()
            with torch.no_grad():
                self.emb_cache[wf] = self.net.embed(x, a)
        return self.emb_cache[wf]

    @torch.no_grad()
    def act(self, wf: int, row: int, x, a, obs: np.ndarray, greedy: bool = False):
        self.net.eval()
        emb = self.embeddings(wf, x, a)[row]
        logits, value = self.net(emb.unsqueeze(0), torch.as_tensor(obs).unsqueeze(0))
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, -1) if greedy else dist.sample()
        return int(action), float(dist.log_prob(action)), float(value)

    def store(self, t: Transition):
        self.buf.items.append(t)

    def mark_done(self):
        if self.buf.items:
            self.buf.items[-1].done = True

    def clear_cache(self):
        self.emb_cache.clear()

    # ----------------------------------------------------------------- update
    def _gae(self):
        items = self.buf.items
        adv = np.zeros(len(items), dtype=np.float32)
        last = 0.0
        for i in reversed(range(len(items))):
            nxt_v = 0.0 if items[i].done or i == len(items) - 1 else items[i + 1].value
            nonterm = 0.0 if items[i].done else 1.0
            delta = items[i].reward + self.p.gamma * nxt_v * nonterm - items[i].value
            last = delta + self.p.gamma * self.p.gae_lambda * nonterm * last
            adv[i] = last
        ret = adv + np.array([t.value for t in items], dtype=np.float32)
        return adv, ret

    def update(self, graphs: Dict[int, tuple]) -> Dict[str, float]:
        """graphs: wf id -> (x tensor, a_norm tensor). Returns training statistics."""
        n = len(self.buf)
        if n < 2:
            self.buf.items.clear()
            self.emb_cache.clear()
            return {}
        adv, ret = self._gae()
        adv_t = torch.as_tensor(adv)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.as_tensor(ret)
        obs = torch.as_tensor(np.stack([t.obs for t in self.buf.items]))
        act = torch.as_tensor([t.action for t in self.buf.items])
        old_logp = torch.as_tensor([t.logp for t in self.buf.items])
        wfs = np.array([t.wf for t in self.buf.items])
        rows = np.array([t.row for t in self.buf.items])
        self.net.train()
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        count = 0
        for _ in range(self.p.epochs):
            perm = np.random.permutation(n)
            for s in range(0, n, self.p.minibatch):
                idx = perm[s:s + self.p.minibatch]
                emb = torch.zeros(len(idx), self.net.encoder.out_dim)
                for w in np.unique(wfs[idx]):
                    sel = np.where(wfs[idx] == w)[0]
                    e = self.net.embed(*graphs[int(w)])
                    emb = emb.index_put((torch.as_tensor(sel),), e[torch.as_tensor(rows[idx][sel])])
                logits, value = self.net(emb, obs[idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act[idx])
                ratio = torch.exp(logp - old_logp[idx])
                a = adv_t[idx]
                pl = -torch.min(ratio * a, torch.clamp(ratio, 1 - self.p.clip_eps, 1 + self.p.clip_eps) * a).mean()
                vl = ((value - ret_t[idx]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pl + self.p.value_coef * vl - self.p.entropy_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.p.max_grad_norm)
                self.opt.step()
                self.loss_history.append(pl.item())
                stats["policy_loss"] += pl.item(); stats["value_loss"] += vl.item(); stats["entropy"] += ent.item()
                count += 1
        self.samples_since_round += n
        self.buf.items.clear()
        self.emb_cache.clear()
        return {k: v / max(count, 1) for k, v in stats.items()}
