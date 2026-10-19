"""Weighted federated averaging (Equations 3 and 4).

    theta_global = sum_i ( n_i / sum_j n_j ) theta_i
    n_i          = |D_i| * (1 + sigma_i^2)^-1

|D_i|      number of transitions node i collected since the previous round
sigma_i^2  variance of node i's PPO policy-loss values recorded since the
           previous round (one value per mini-batch update)

The aggregation runs in-process (simulated federation). Only model
parameters are exchanged; the number of bytes that would cross the network
is returned so that the communication overhead can be reported (Figure 4).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def aggregation_weights(agents, variance_weighting: bool = True) -> np.ndarray:
    n = []
    for ag in agents:
        d = max(ag.samples_since_round, 1)
        var = float(np.var(ag.loss_history)) if (variance_weighting and len(ag.loss_history) > 1) else 0.0
        n.append(d / (1.0 + var))
    n = np.asarray(n, dtype=np.float64)
    return n / n.sum()


def param_bytes(model: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.state_dict().values())


def federated_average(agents, variance_weighting: bool = True) -> Dict[str, float]:
    w = aggregation_weights(agents, variance_weighting)
    states: List[Dict[str, torch.Tensor]] = [ag.net.state_dict() for ag in agents]
    avg = {}
    for k in states[0]:
        if states[0][k].dtype.is_floating_point:
            avg[k] = sum(float(wi) * s[k] for wi, s in zip(w, states))
        else:
            avg[k] = states[0][k]
    for ag in agents:
        ag.net.load_state_dict(avg)
        ag.loss_history.clear()
        ag.samples_since_round = 0
    per_node = param_bytes(agents[0].net)
    return {"bytes_total": 2.0 * per_node * len(agents),   # upload + download
            "bytes_per_node": 2.0 * per_node,
            "weights_min": float(w.min()), "weights_max": float(w.max())}
