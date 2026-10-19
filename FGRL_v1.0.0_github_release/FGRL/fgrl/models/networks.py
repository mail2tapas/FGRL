"""Neural networks: 3-layer GCN encoder (Equation 2) and PPO actor-critic.

The GCN is written directly in PyTorch (dense normalised adjacency), which is
mathematically identical to ``torch_geometric.nn.GCNConv`` with self-loops and
symmetric normalisation, and avoids a hard dependency on PyTorch Geometric.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fgrl.graph.features import FEATURE_DIM


class GCNEncoder(nn.Module):
    """h^(l+1) = ReLU( D^-1/2 (A+I) D^-1/2 H^(l) W^(l) ),  L layers."""

    def __init__(self, hidden: int = 32, layers: int = 3, dropout: float = 0.2):
        super().__init__()
        dims = [FEATURE_DIM] + [hidden] * layers
        self.lins = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(layers))
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        h = x
        for i, lin in enumerate(self.lins):
            h = torch.relu(a_norm @ lin(h))
            if i < len(self.lins) - 1:
                h = self.drop(h)
        return h                       # e_v = h_v^(L)


class MLPEncoder(nn.Module):
    """Ablation 'without GNN': per-task features only, no message passing."""

    def __init__(self, hidden: int = 32, layers: int = 3, dropout: float = 0.2):
        super().__init__()
        mods, d = [], FEATURE_DIM
        for i in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            if i < layers - 1:
                mods.append(nn.Dropout(dropout))
            d = hidden
        self.net = nn.Sequential(*mods)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs (avoids the value loss dominating the policy features)."""

    def __init__(self, emb_dim: int, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        d = emb_dim + obs_dim
        self.actor = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                   nn.Linear(hidden, n_actions))
        self.critic = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                    nn.Linear(hidden, 1))

    def forward(self, emb: torch.Tensor, obs: torch.Tensor):
        z = torch.cat([emb, obs], dim=-1)
        zc = torch.cat([emb.detach(), obs], dim=-1)   # encoder is trained by the policy loss
        return self.actor(z), self.critic(zc).squeeze(-1)


class SchedulerNet(nn.Module):
    """Encoder + actor-critic held by one agent (one fog node, or the central scheduler)."""

    def __init__(self, obs_dim: int, n_actions: int, use_gnn: bool, hidden: int, layers: int,
                 policy_hidden: int, dropout: float):
        super().__init__()
        enc = GCNEncoder if use_gnn else MLPEncoder
        self.encoder = enc(hidden, layers, dropout)
        self.ac = ActorCritic(self.encoder.out_dim, obs_dim, n_actions, policy_hidden)

    def embed(self, x, a_norm):
        return self.encoder(x, a_norm)

    def forward(self, emb, obs):
        return self.ac(emb, obs)
