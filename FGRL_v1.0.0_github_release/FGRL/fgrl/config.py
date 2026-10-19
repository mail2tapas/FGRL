"""Central configuration for FGRL experiments.

Every hyperparameter that appears in the paper is defined here so that the
manuscript and the code can be checked against one place. Configurations are
plain dataclasses and can be overridden from a JSON file (see configs/).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class TierSpec:
    name: str
    mips: float          # CPU capacity C_i (million instructions per second)
    memory_gb: float
    bandwidth_mbps: float
    p_idle: float        # Watts
    p_peak: float        # Watts


@dataclass
class InfraConfig:
    # Table 4 of the paper (Fog-Cloud hardware tiers)
    tiers: List[TierSpec] = field(default_factory=lambda: [
        TierSpec("tier1_edge_gateway", 2000, 4, 50, 4.0, 12.0),
        TierSpec("tier2_edge_server", 5000, 8, 250, 30.0, 90.0),
        TierSpec("tier3_cloud", 10000, 128, 1000, 120.0, 300.0),
    ])
    num_fog_nodes: int = 10          # N_f (cloud is added as one extra node)
    tier1_fraction: float = 0.5      # share of fog nodes that are Tier 1 (rest Tier 2)
    neighbors_per_node: int = 3      # neighbourhood used for "offload to neighbour"
    cloud_latency_s: float = 0.05    # extra WAN latency fog <-> cloud (seconds)
    fog_link_latency_s: float = 0.005
    # Dynamic conditions (the reason static heuristics degrade)
    load_spike_prob: float = 0.15    # probability per workflow that a node gets a spike
    load_spike_slowdown: float = 2.5 # execution-time multiplier on a spiking node
    exec_noise_cv: float = 0.10      # coefficient of variation of execution times


@dataclass
class RewardConfig:
    # Equation (5): r = w1 r_lat + w2 r_en + w3 r_SLA + w4 r_util + lambda r_critical
    w_lat: float = 1.0
    w_en: float = 0.3
    w_sla: float = 0.5
    w_util: float = 0.2
    lam_critical: float = 1.0
    alpha_critical: float = 1.0      # alpha in Equation (6)
    defer_penalty: float = 0.05


@dataclass
class ModelConfig:
    gcn_layers: int = 3              # L = 3 (Section 3.3)
    gcn_hidden: int = 32
    policy_hidden: int = 64
    dropout: float = 0.2             # Table 3
    weight_decay: float = 5e-4       # Table 3


@dataclass
class PPOConfig:
    lr: float = 1e-3
    gamma: float = 0.5               # short horizon: each decision's effect is mostly immediate
    gae_lambda: float = 0.9
    clip_eps: float = 0.2            # epsilon in Equation (7)
    epochs: int = 8
    minibatch: int = 32
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


@dataclass
class FedConfig:
    enabled: bool = True
    interval_k: int = 50             # K: aggregate every K episodes
    variance_weighting: bool = True  # Equation (4)


@dataclass
class TrainConfig:
    episodes: int = 2000
    workflows_per_episode: int = 2
    early_stop_patience: int = 5     # Table 3 (in evaluation intervals)
    eval_every: int = 50
    max_defers_per_task: int = 2


@dataclass
class ExperimentConfig:
    seed: int = 0
    workflow_types: List[str] = field(default_factory=lambda: ["montage", "cybershake", "epigenomics"])
    workflow_size: int = 100
    workflow_dir: str = ""           # optional: folder with WfCommons JSON instances
    pegasus_dir: str = ""            # optional: folder with Pegasus DAX benchmarks (trace-driven instances)
    infra: InfraConfig = field(default_factory=InfraConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    fed: FedConfig = field(default_factory=FedConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    # Ablation switches (Section 5.5)
    use_gnn: bool = True
    use_critical_reward: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _update(obj, d):
    for k, v in d.items():
        if k.startswith("_") or k == "experiment":      # comments / script-level settings
            continue
        if not hasattr(obj, k):
            raise KeyError(f"unknown configuration key: {k}")
        cur = getattr(obj, k)
        if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
            _update(cur, v)
        elif k == "tiers":
            setattr(obj, k, [TierSpec(**t) for t in v])
        else:
            setattr(obj, k, v)


def load_config(path: str | None = None, **overrides) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if path:
        with open(path) as f:
            _update(cfg, json.load(f))
    _update(cfg, overrides)
    return cfg
