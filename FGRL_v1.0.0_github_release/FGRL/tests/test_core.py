"""Unit tests: run with  `python -m pytest -q`  (about 1 minute on a laptop CPU)."""
import random

import networkx as nx
import numpy as np
import pytest
import torch

from fgrl.config import load_config
from fgrl.graph.critical_path import critical_path
from fgrl.graph.features import FEATURE_DIM, build_workflow_graph
from fgrl.models.networks import GCNEncoder
from fgrl.sim.env import CLOUD, DEFER, LOCAL, NEIGHBOR, SchedulingEpisode
from fgrl.sim.fogcloud import WorkflowSim, build_infrastructure
from fgrl.workflows.generator import generate, load_wfcommons, to_wfformat
from fgrl.baselines.heuristics import heft, heft_plan
from fgrl.federated.fedavg import aggregation_weights, federated_average
from fgrl.agents.trainer import FGRLTrainer, CentralPPOTrainer
from fgrl.eval.stats import cohens_d, friedman, nemenyi

CFG = load_config()
CFG.infra.num_fog_nodes = 4


@pytest.mark.parametrize("kind", ["montage", "cybershake", "epigenomics"])
def test_generators_are_dags(kind):
    G = generate(kind, 60, random.Random(1))
    assert nx.is_directed_acyclic_graph(G)
    assert 40 <= G.number_of_nodes() <= 70
    assert all("deadline" in G.nodes[v] for v in G)


def test_wfformat_roundtrip(tmp_path):
    G = generate("montage", 40, random.Random(2))
    p = tmp_path / "montage-test.json"
    import json
    p.write_text(json.dumps(to_wfformat(G, "montage-test")))
    H = load_wfcommons(str(p))
    assert H.number_of_nodes() == G.number_of_nodes()
    assert H.number_of_edges() == G.number_of_edges()


def test_critical_path_diamond():
    # Section 3.3 example: A -> {B, C} -> D, B long, C short
    G = nx.DiGraph()
    for v, w in {"A": 1, "B": 5, "C": 1, "D": 1}.items():
        G.add_node(v, work=w)
    for e in [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]:
        G.add_edge(*e, data=0.0)
    cp = critical_path(G, lambda v: G.nodes[v]["work"], lambda u, v: 0.0)
    assert cp.critical == {"A", "B", "D"}
    assert cp.length == pytest.approx(7.0)
    assert cp.slack["C"] == pytest.approx(4.0)


def test_simulator_respects_precedence():
    G = generate("epigenomics", 40, random.Random(3))
    nodes = build_infrastructure(CFG.infra, 0)
    sim = WorkflowSim(G, nodes, CFG.infra, random.Random(0))
    rng = random.Random(1)
    for v in nx.topological_sort(G):
        sim.place(v, rng.randrange(len(nodes)))
    for u, v in G.edges:
        assert sim.st[v] >= sim.ft[u] - 1e-9          # Equation (9)
    m = sim.metrics()
    assert m["makespan"] > 0 and 0 <= m["utilization"] <= 1 and 0 <= m["sla_violation"] <= 1


def test_heft_valid_schedule():
    G = generate("montage", 50, random.Random(4))
    nodes = build_infrastructure(CFG.infra, 0)
    order, assign = heft_plan(G, nodes, CFG.infra)
    pos = {v: i for i, v in enumerate(order)}
    assert all(pos[u] < pos[v] for u, v in G.edges)
    assert heft(G, nodes, CFG.infra, random.Random(0))["makespan"] > 0


def test_gcn_shapes_and_features():
    G = generate("cybershake", 50, random.Random(5))
    nodes = build_infrastructure(CFG.infra, 0)
    wg = build_workflow_graph(G, 3500, 150)
    assert wg.x.shape == (G.number_of_nodes(), FEATURE_DIM)
    assert np.allclose(wg.a_norm, wg.a_norm.T)
    enc = GCNEncoder(16, 3, 0.0)
    e = enc(torch.as_tensor(wg.x), torch.as_tensor(wg.a_norm))
    assert e.shape == (G.number_of_nodes(), 16)


def test_episode_all_actions_complete():
    G = generate("montage", 40, random.Random(6))
    nodes = build_infrastructure(CFG.infra, 0)
    wg = build_workflow_graph(G, 3500, 150)
    for a in (LOCAL, NEIGHBOR, CLOUD, DEFER):
        env = SchedulingEpisode(wg, nodes, CFG.infra, CFG.reward, random.Random(0))
        while env.next_decision() is not None:
            env.step(a)
        assert env.done()


def test_fedavg_weights_and_average():
    cfg = load_config(); cfg.infra.num_fog_nodes = 3
    nodes = build_infrastructure(cfg.infra, 0)
    pool = {"montage": [generate("montage", 30, random.Random(7))]}
    tr = FGRLTrainer(cfg, nodes, pool, seed=0)
    for i, ag in enumerate(tr.agents):
        ag.samples_since_round = 10 * (i + 1)
        ag.loss_history = [0.0, float(i)]
        with torch.no_grad():
            for p in ag.net.parameters():
                p.fill_(float(i))
    w = aggregation_weights(tr.agents)
    assert w.sum() == pytest.approx(1.0)
    federated_average(tr.agents)
    s0 = tr.agents[0].net.state_dict()
    for ag in tr.agents[1:]:
        for k, v in ag.net.state_dict().items():
            assert torch.allclose(v, s0[k])


@pytest.mark.parametrize("T", [FGRLTrainer, CentralPPOTrainer])
def test_training_smoke(T):
    cfg = load_config(); cfg.infra.num_fog_nodes = 3; cfg.fed.interval_k = 2
    cfg.train.workflows_per_episode = 1
    nodes = build_infrastructure(cfg.infra, 0)
    pool = {"montage": [generate("montage", 30, random.Random(8))]}
    tr = T(cfg, nodes, pool, pool, seed=0)
    hist = tr.train(4, eval_every=2, patience=0)
    assert len(hist) == 4 and all(np.isfinite(h["makespan"]) for h in hist)
    res = tr.evaluate([("montage", pool["montage"][0])], [1])
    assert res[0]["makespan"] > 0


def test_stats():
    rng = np.random.default_rng(0)
    scores = np.column_stack([rng.normal(10, 1, 30), rng.normal(12, 1, 30), rng.normal(15, 1, 30)])
    assert friedman(scores)["p_value"] < 0.05
    pairs = nemenyi(scores, ["a", "b", "c"])
    assert len(pairs) == 3 and all(0 <= p["p_value"] <= 1 for p in pairs)
    assert cohens_d(scores[:, 2], scores[:, 0]) > 2


def test_paired_effect_size():
    from fgrl.eval.stats import cohens_dz
    a = np.array([10.0, 20, 30, 40]); b = a - np.array([1.0, 1.2, 0.8, 1.0])
    assert cohens_dz(a, b) > 5


def test_pegasus_dax_and_trace_variants():
    """Benchmark DAX files load as DAGs and trace variants keep the topology."""
    import glob
    import os
    import random as _r
    from fgrl.workflows.generator import load_dax, trace_variant
    files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                          "data", "pegasus", "*_100.xml")))
    if not files:
        import pytest
        pytest.skip("run scripts/download_benchmarks.py first")
    for f in files:
        G = load_dax(f)
        assert G.number_of_nodes() == 100 and nx.is_directed_acyclic_graph(G)
        assert all(G.nodes[v]["work"] > 0 and G.nodes[v]["deadline"] > 0 for v in G)
        H = trace_variant(G, _r.Random(0))
        assert set(H.edges) == set(G.edges)
        assert any(abs(H.nodes[v]["work"] - G.nodes[v]["work"]) > 1e-9 for v in G)
