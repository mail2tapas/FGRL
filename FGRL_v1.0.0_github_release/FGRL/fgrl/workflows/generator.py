"""Scientific workflow DAGs: Montage, CyberShake and Epigenomics.

Two sources are supported:

1. ``load_wfcommons(path)`` reads a WfCommons / WfFormat JSON instance
   (https://wfcommons.org). Use this for the paper's real benchmark traces.
2. ``generate(kind, n_tasks, rng)`` builds a synthetic DAG that follows the
   published level structure of each application. It is used for the test
   data in ``data/test`` and whenever no WfCommons instances are supplied.

Every DAG is a ``networkx.DiGraph`` with
  node attributes: ``work`` (million instructions), ``name``, ``deadline`` (s)
  edge attributes: ``data`` (megabytes transferred parent -> child)
"""
from __future__ import annotations

import glob
import json
import os
import random
from typing import Dict, List

import networkx as nx

REF_MIPS = 5000.0      # reference speed used to derive deadlines (Tier 2)
REF_BW_MBPS = 250.0
DEADLINE_FACTOR = 2.5  # deadline = 2.5 x earliest finish on the reference node (no queueing)


# ----------------------------------------------------------------------------
# synthetic generators (structure follows the Pegasus/WfCommons applications)
# ----------------------------------------------------------------------------
def _add(G, tid, name, rng, work_range, ):
    G.add_node(tid, name=name, work=float(rng.uniform(*work_range)))


def _edge(G, u, v, rng, data_range):
    G.add_edge(u, v, data=float(rng.uniform(*data_range)))


def _montage(n: int, rng: random.Random) -> nx.DiGraph:
    # mProject (p) -> mDiffFit (~1.5p) -> mConcatFit -> mBgModel -> mBackground (p)
    # -> mImgtbl -> mAdd -> mShrink -> mJPEG
    fixed = 6
    p = max(2, int((n - fixed) / 3.5))
    G, t = nx.DiGraph(), 0
    proj = []
    for _ in range(p):
        _add(G, t, "mProjectPP", rng, (8000, 16000)); proj.append(t); t += 1
    diff = []
    nd = max(1, n - fixed - 2 * p)
    for i in range(nd):
        _add(G, t, "mDiffFit", rng, (2000, 5000)); diff.append(t)
        a, b = proj[i % p], proj[(i + 1) % p]
        _edge(G, a, t, rng, (2, 6)); _edge(G, b, t, rng, (2, 6)); t += 1
    _add(G, t, "mConcatFit", rng, (3000, 6000)); concat = t; t += 1
    for d in diff: _edge(G, d, concat, rng, (0.1, 0.5))
    _add(G, t, "mBgModel", rng, (6000, 12000)); bg = t; t += 1
    _edge(G, concat, bg, rng, (0.5, 1))
    back = []
    for i in range(p):
        _add(G, t, "mBackground", rng, (2000, 5000)); back.append(t)
        _edge(G, bg, t, rng, (0.1, 0.3)); _edge(G, proj[i], t, rng, (2, 6)); t += 1
    for name, wr in [("mImgtbl", (1000, 3000)), ("mAdd", (10000, 20000)),
                     ("mShrink", (3000, 6000)), ("mJPEG", (1000, 2000))]:
        _add(G, t, name, rng, wr)
        if name == "mImgtbl":
            for b_ in back: _edge(G, b_, t, rng, (2, 6))
        else:
            _edge(G, t - 1, t, rng, (5, 20))
        t += 1
    return G


def _cybershake(n: int, rng: random.Random) -> nx.DiGraph:
    # ExtractSGT (s) -> SeismogramSynthesis (k per site) -> PeakValCalc (1:1)
    # -> ZipSeis, ZipPSA
    G, t = nx.DiGraph(), 0
    s = max(1, n // 40)
    per = max(1, (n - s - 2) // (2 * s))
    _add(G, t, "ZipSeis", rng, (2000, 4000)); zs = t; t += 1
    _add(G, t, "ZipPSA", rng, (2000, 4000)); zp = t; t += 1
    for _ in range(s):
        _add(G, t, "ExtractSGT", rng, (30000, 60000)); ex = t; t += 1
        for _ in range(per):
            _add(G, t, "SeismogramSynthesis", rng, (15000, 40000)); ss = t; t += 1
            _edge(G, ex, ss, rng, (20, 60))
            _add(G, t, "PeakValCalc", rng, (500, 1500)); pv = t; t += 1
            _edge(G, ss, pv, rng, (0.1, 0.5))
            _edge(G, ss, zs, rng, (0.1, 0.5)); _edge(G, pv, zp, rng, (0.01, 0.05))
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


def _epigenomics(n: int, rng: random.Random) -> nx.DiGraph:
    # fastqSplit -> lanes of (filterContams -> sol2sanger -> fast2bfq -> map)
    # -> mapMerge -> maqIndex -> pileup
    G, t = nx.DiGraph(), 0
    lanes = max(1, (n - 4) // 4)
    _add(G, t, "fastqSplit", rng, (3000, 8000)); split = t; t += 1
    ends = []
    for _ in range(lanes):
        prev = split
        for name, wr in [("filterContams", (1000, 3000)), ("sol2sanger", (500, 1500)),
                         ("fast2bfq", (1000, 2500)), ("map", (20000, 60000))]:
            _add(G, t, name, rng, wr); _edge(G, prev, t, rng, (5, 30)); prev = t; t += 1
        ends.append(prev)
    for name, wr in [("mapMerge", (5000, 12000)), ("maqIndex", (8000, 16000)), ("pileup", (8000, 16000))]:
        _add(G, t, name, rng, wr)
        if name == "mapMerge":
            for e in ends: _edge(G, e, t, rng, (1, 5))
        else:
            _edge(G, t - 1, t, rng, (10, 40))
        t += 1
    return G


GENERATORS = {"montage": _montage, "cybershake": _cybershake, "epigenomics": _epigenomics}


def assign_deadlines(G: nx.DiGraph, factor: float = DEADLINE_FACTOR) -> None:
    """deadline_v = factor x earliest finish time of v on the reference node."""
    eft: Dict[int, float] = {}
    for v in nx.topological_sort(G):
        est = 0.0
        for u in G.predecessors(v):
            est = max(est, eft[u] + G.edges[u, v]["data"] * 8 / REF_BW_MBPS)
        eft[v] = est + G.nodes[v]["work"] / REF_MIPS
        G.nodes[v]["deadline"] = factor * eft[v]


def generate(kind: str, n_tasks: int, rng: random.Random) -> nx.DiGraph:
    G = GENERATORS[kind](n_tasks, rng)
    G.graph["kind"] = kind
    assert nx.is_directed_acyclic_graph(G)
    assign_deadlines(G)
    return G


# ----------------------------------------------------------------------------
# WfCommons (WfFormat) JSON loader
# ----------------------------------------------------------------------------
def _task_list(js) -> List[dict]:
    wf = js.get("workflow", js)
    if "specification" in wf:                     # WfFormat >= 1.5
        spec = {t["id"]: dict(t) for t in wf["specification"]["tasks"]}
        for t in wf.get("execution", {}).get("tasks", []):
            spec.get(t["id"], {}).update({k: v for k, v in t.items() if k != "id"})
        files = {f["id"]: f for f in wf["specification"].get("files", [])}
        out = []
        for t in spec.values():
            t["_outputs"] = [files.get(f, {}).get("sizeInBytes", 0) for f in t.get("outputFiles", [])]
            out.append(t)
        return out
    return wf["tasks"]                               # WfFormat <= 1.4


def load_wfcommons(path: str, ref_mips: float = REF_MIPS) -> nx.DiGraph:
    with open(path) as f:
        js = json.load(f)
    tasks = _task_list(js)
    key = lambda t: t.get("id", t.get("name"))
    G = nx.DiGraph()
    idx = {}
    for i, t in enumerate(tasks):          # parents may reference ids or names
        for k in (t.get("id"), t.get("name")):
            if k is not None:
                idx[k] = i
    out_bytes = {}
    for t in tasks:
        rt = t.get("runtimeInSeconds", t.get("runtime", 1.0)) or 1.0
        G.add_node(idx[key(t)], name=t.get("name", str(key(t))), work=float(rt) * ref_mips)
        if "_outputs" in t:
            ob = sum(t["_outputs"])
        else:
            ob = sum(f.get("sizeInBytes", f.get("size", 0)) for f in t.get("files", [])
                     if f.get("link") == "output")
        out_bytes[idx[key(t)]] = ob
    n_kids: Dict[int, int] = {}
    for t in tasks:
        for p in t.get("parents", []):
            if p in idx:
                n_kids[idx[p]] = n_kids.get(idx[p], 0) + 1
    for t in tasks:
        for p in t.get("parents", []):
            if p in idx:
                u = idx[p]
                G.add_edge(u, idx[key(t)], data=out_bytes.get(u, 0) / 1e6 / max(1, n_kids[u]))
    assert nx.is_directed_acyclic_graph(G), f"{path} is not a DAG"
    G.graph["kind"] = os.path.basename(path).split("-")[0].lower()
    assign_deadlines(G)
    return G


def to_wfformat(G: nx.DiGraph, name: str) -> dict:
    """Serialise a DAG in WfFormat 1.4 style (used to write the test data)."""
    tasks = []
    for v in G.nodes:
        tasks.append({
            "name": f"{G.nodes[v]['name']}_{v:05d}",
            "id": f"{v:05d}",
            "type": "compute",
            "runtimeInSeconds": round(G.nodes[v]["work"] / REF_MIPS, 4),
            "parents": [f"{u:05d}" for u in G.predecessors(v)],
            "files": [{"link": "output", "name": f"out_{v}", "sizeInBytes":
                       int(sum(G.edges[v, c]["data"] for c in G.successors(v)) * 1e6)}],
        })
    return {"name": name, "schemaVersion": "1.4", "workflow": {"tasks": tasks}}


def load_workflow_pool(kinds: List[str], size: int, seed: int, count: int = 5,
                       workflow_dir: str = "") -> Dict[str, List[nx.DiGraph]]:
    """Returns {kind: [DAG, ...]}. Uses WfCommons files when a directory is given."""
    pool: Dict[str, List[nx.DiGraph]] = {}
    rng = random.Random(seed)
    for k in kinds:
        files = sorted(glob.glob(os.path.join(workflow_dir, f"{k}*.json"))) if workflow_dir else []
        if files:
            pool[k] = [load_wfcommons(f) for f in files]
        else:
            pool[k] = [generate(k, size, rng) for _ in range(count)]
    return pool


# ----------------------------------------------------------------------------
# Pegasus DAX benchmark loader (Bharathi et al., 2008; Juve et al., 2013)
# ----------------------------------------------------------------------------
def load_dax(path: str, ref_mips: float = REF_MIPS) -> nx.DiGraph:
    """Read a Pegasus DAX 2.x benchmark workflow (e.g. Montage_100.xml).

    work_v = runtime_v (s, as recorded in the DAX) x ref_mips, so a task takes its
    recorded runtime on the Tier 2 reference node.  The data volume of edge (u, v)
    is the total size of the files that u writes and v reads (matched by file name).
    """
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    G = nx.DiGraph()
    idx, outs, ins = {}, {}, {}
    for i, job in enumerate(root.iter(ns + "job")):
        jid = job.get("id")
        idx[jid] = i
        G.add_node(i, name=job.get("name"), work=float(job.get("runtime", 1.0)) * ref_mips)
        outs[i], ins[i] = {}, {}
        for u in job.iter(ns + "uses"):
            size = float(u.get("size", 0) or 0)
            (outs if u.get("link") == "output" else ins)[i][u.get("file")] = size
    for child in root.iter(ns + "child"):
        c = idx[child.get("ref")]
        for par in child.iter(ns + "parent"):
            p = idx[par.get("ref")]
            shared = set(outs[p]) & set(ins[c])
            G.add_edge(p, c, data=sum(outs[p][f] for f in shared) / 1e6)
    assert nx.is_directed_acyclic_graph(G), f"{path} is not a DAG"
    G.graph["kind"] = os.path.basename(path).split("_")[0].lower()
    G.graph["source"] = os.path.basename(path)
    assign_deadlines(G)
    return G


def trace_variant(G: nx.DiGraph, rng: random.Random, jitter_cv: float = 0.10) -> nx.DiGraph:
    """New workflow instance with the benchmark's topology and job types.

    Each task's runtime is bootstrap-resampled from the runtimes of all tasks of the
    same job type in the benchmark trace and multiplied by a log-normal jitter
    (coefficient of variation `jitter_cv`); each edge's data volume is resampled the
    same way from the edges between the same pair of job types.
    """
    import math
    H = G.copy()
    s = math.sqrt(math.log(1 + jitter_cv ** 2))
    jit = lambda: rng.lognormvariate(-s * s / 2, s)
    by_type: Dict[str, List[float]] = {}
    for v in G.nodes:
        by_type.setdefault(G.nodes[v]["name"], []).append(G.nodes[v]["work"])
    for v in H.nodes:
        H.nodes[v]["work"] = rng.choice(by_type[G.nodes[v]["name"]]) * jit()
    by_pair: Dict[tuple, List[float]] = {}
    for u, v in G.edges:
        by_pair.setdefault((G.nodes[u]["name"], G.nodes[v]["name"]), []).append(G.edges[u, v]["data"])
    for u, v in H.edges:
        H.edges[u, v]["data"] = rng.choice(by_pair[(G.nodes[u]["name"], G.nodes[v]["name"])]) * jit()
    H.graph = dict(G.graph)
    assign_deadlines(H)
    return H


def pegasus_file(pegasus_dir: str, kind: str, size: int) -> str:
    """Benchmark file of a workflow type whose task count is closest to `size`."""
    files = glob.glob(os.path.join(pegasus_dir, "*.xml"))
    files = [f for f in files if os.path.basename(f).lower().startswith(kind)]
    if not files:
        raise FileNotFoundError(f"no Pegasus DAX for '{kind}' in {pegasus_dir} (run scripts/download_benchmarks.py)")
    return min(files, key=lambda f: abs(int(os.path.basename(f).split("_")[1].split(".")[0]) - size))


def makespan_lower_bound(G: nx.DiGraph, fastest_mips: float) -> float:
    """Critical-path length with every task on the fastest resource and no communication.
    makespan / lower bound is the schedule length ratio (SLR), a scale-free makespan measure."""
    cache = G.graph.get("_lb")
    if cache and cache[0] == fastest_mips:
        return cache[1]
    f: Dict[int, float] = {}
    for v in nx.topological_sort(G):
        f[v] = max((f[u] for u in G.predecessors(v)), default=0.0) + G.nodes[v]["work"] / fastest_mips
    lb = max(f.values())
    G.graph["_lb"] = (fastest_mips, lb)
    return lb
