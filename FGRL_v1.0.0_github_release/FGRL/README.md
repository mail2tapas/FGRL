# FGRL: Federated Graph Reinforcement Learning for Dependency-Aware Scheduling in Fog-Cloud Computing

Reference implementation of FGRL, a decentralized scheduler for workflow (DAG) applications in fog-cloud
systems that combines a graph convolutional encoder, multi-agent PPO at the fog nodes, a critical-path-aware
reward, and variance-weighted federated averaging. The repository contains the simulator, FGRL, the baselines,
the experiment scripts, the benchmark workflows, and the run-level results from which every table, figure and
statistic of the experimental section of the paper is generated.

**Release:** `v1.0.0` (see [RELEASE_NOTES.md](RELEASE_NOTES.md)) · **License:** MIT (see [LICENSE](LICENSE)) ·
**How to cite:** [CITATION.cff](CITATION.cff)

## Quick start

```bash
git clone https://github.com/mail2tapas/FGRL.git
cd FGRL
pip install -r requirements.txt
python -m pytest -q                 # 15 unit tests, under a minute
./run_test_pipeline.sh              # every experiment on small test data (CPU, roughly 30-60 min)
```

The test pipeline writes CSV tables, `statistics.json` and Figures 3-7 to `results/test/`.
Tested with Python 3.13, PyTorch 2.14 (CPU), NumPy 2.5, SciPy 1.18, pandas 3.0 and NetworkX 3.6;
the versions used for the released results are recorded in `results/paper/environment.json`.

## Reproducing the paper results

```bash
./run_full.sh                                    # paper protocol, all cores, resumable
JOBS=8 ./run_full.sh                             # choose the number of parallel jobs
SEEDS="0 1 2" ./run_full.sh                      # one machine of several (copy chunks together later)
SEEDS=none ./run_full.sh                         # only merge, draw figures and build the report
CONFIG=configs/pilot_pegasus.json OUT=results/pilot ./run_full.sh   # reduced pilot (about 1.5 h on 2 cores)
```

`configs/paper_pegasus.json` uses the Pegasus benchmark workflows Montage, CyberShake and Epigenomics with
100 tasks (`data/pegasus/`, see [data/pegasus/SOURCES.md](data/pegasus/SOURCES.md)). Training, validation and
test instances keep the benchmark topology and resample the recorded runtimes and file sizes per job type
(`trace_variant`); the unmodified benchmark file is always a test instance.

The protocol is 10 seeds x 3 workflow types = 30 (seed, workflow) blocks per method for the comparison and the
ablation, 3 seeds per node count (10 to 80 fog nodes) for the scalability study, and 3 seeds for the
communication study. Budget: about 35-45 CPU-hours in total. On a 2-core Google Colab runtime use
`FGRL_full_run_colab.ipynb` and split the seeds over several sessions; a 16-core workstation needs about 3-4 hours.

The individual experiments can also be run separately:

```bash
python experiments/run_comparison.py    --config configs/paper_pegasus.json --out results/paper
python experiments/run_ablation.py      --config configs/paper_pegasus.json --out results/paper
python experiments/run_scalability.py   --config configs/paper_pegasus.json --out results/paper
python experiments/run_communication.py --config configs/paper_pegasus.json --out results/paper
python experiments/make_figures.py      --results results/paper
python experiments/evidence_report.py   --results results/paper --config configs/paper_pegasus.json
```

## Released results (`results/paper/`)

| File | Content |
|---|---|
| `EVIDENCE.md`, `evidence.json` | every table, figure and statistic of the experimental section, with the number of blocks |
| `tables/*.csv` | Table 7 per workflow and pooled (normalized to HEFT), Table 5 per workflow, ablation, statistics |
| `figure3`-`figure7_*.png` | Figures 3-7 at 600 dpi |
| `comparison_runs.csv`, `ablation_runs.csv`, `scalability_runs.csv`, `communication_runs.csv` | run-level results (one row per method x seed x workflow) |
| `training_log.csv`, `ablation_training_log.csv` | per-episode training curves (overall and per workflow type) |
| `footprint_latency.json` | parameter count, model size and measured inference latency |
| `chunks/`, `logs/` | output, console log and wall time of every (experiment, seed) job |
| `environment.json`, `MANIFEST.sha256` | software and hardware, and SHA-256 of all code, configurations, data and run-level results |

To check that the released files are unchanged: `sha256sum -c results/paper/MANIFEST.sha256`.

**Statistics.** All tests use the 30 run-level (seed, workflow) blocks: Friedman test with the Nemenyi critical
difference, two-sided Wilcoxon signed-rank tests of FGRL against each baseline with Holm correction (on
log(baseline/FGRL) for makespan and energy), and the paired effect size d_z with a bootstrap 95% confidence
interval. The ablation variants are tested separately against the full FGRL on the relative difference in each
block, with Holm correction for the three comparisons. Makespans differ by orders of magnitude between workflow
types, so pooled results are reported as ratios to HEFT within each block, and tables are also given per workflow.

## Map from paper to code

| Paper | Code |
|---|---|
| Three-tier infrastructure, Table 4; cubic power model | `fgrl/config.py` (`InfraConfig`), `fgrl/sim/fogcloud.py` |
| Precedence constraint, Eq. (9); makespan, Eq. (12) | `WorkflowSim.place`, `WorkflowSim.metrics` |
| Task features x_v, Eq. (1) | `fgrl/graph/features.py` |
| 3-layer GCN, Eq. (2) | `fgrl/models/networks.py` (`GCNEncoder`) |
| Critical path and slack, forward-backward traversal, O(\|V\|+\|E\|) | `fgrl/graph/critical_path.py` |
| Dec-MDP state, four actions | `fgrl/sim/env.py` (`observe_local`, `step`) |
| Reward, Eqs. (5)-(6) | `SchedulingEpisode.reward` |
| PPO clipped objective, Eq. (7) | `fgrl/agents/ppo.py` |
| Weighted FedAvg, Eqs. (3)-(4) | `fgrl/federated/fedavg.py` |
| Algorithm 1 (training loop) | `fgrl/agents/trainer.py` (`FGRLTrainer`) |
| HEFT, PSO-Fuzzy, Centralized PPO baselines | `fgrl/baselines/heuristics.py`, `CentralPPOTrainer` |
| Ablations (no GNN, no critical-path reward, no federation) | `fgrl/eval/runner.py` (`LEARNING`) |
| Dropout 0.2, weight decay 5e-4, early stopping patience 5 (Table 3) | `ModelConfig`, `TrainConfig` |
| Statistical tests (Section 5.7) | `fgrl/eval/stats.py`, `experiments/evidence_report.py` |
| Tables 5 and 7, Figures 3-7 | `experiments/*.py`, `experiments/make_figures.py` |

## Implementation notes

1. **Stack.** PPO, the GCN and federated averaging are implemented directly in PyTorch (no Stable-Baselines3,
   PyTorch Geometric or Flower). The GCN is equivalent to a PyG `GCNConv` with self-loops. Federation is
   simulated in-process.
2. **Which agent acts.** Each task becomes ready at the fog node that holds most of its input data (entry tasks
   arrive at random fog nodes); only that node's agent decides for the task.
3. **Actions.** Execute locally, offload to the neighboring fog node (out of three) with the earliest estimated
   finish time, offload to the cloud, or defer. Deferral is limited to two times per task, and its waiting time
   is charged immediately as latency.
4. **Critical-path reward, Eq. (6).** `alpha * (1 - slack/max_slack) * 1[v in CP] * s_v`, where `s_v = +1` if
   v finishes by its latest finish time and `-1` otherwise. The critical path and slack are computed once per
   arriving workflow with nominal speeds.
5. **Discount factor.** gamma = 0.5 and GAE lambda = 0.9.
6. **Dynamic conditions.** In each workflow, every fog node has a 15% chance of a load spike (2.5x slower
   execution), and execution times have 10% log-normal noise. HEFT and PSO-Fuzzy plan with nominal speeds; the
   learning agents observe local slowdowns and backlogs.
7. **Metrics.** Deadline = 2.5x the earliest finish time on a Tier 2 reference node; SLA violation = percentage
   of tasks finishing after their deadline. Energy: P(u) = P_idle + (P_peak - P_idle)·u³ per node over the
   makespan (idle cloud not counted), in kWh. Utilization: average busy fraction of the fog nodes.
8. **Communication overhead (Figure 4)** = modeled transfer time of all federated rounds divided by (measured
   training time + that transfer time); the training time depends on the machine.
9. **Inference latency.** `run_communication.py` measures latency on the machine it runs on. Values for other
   devices (for example ARM single-board computers) require running the script on those devices.

## Baselines

* **HEFT**: Topcuoglu et al. (2002), non-insertion variant.
* **PSO-Fuzzy**: simplified re-implementation, i.e., particle swarm optimization over task-to-node mappings with
  a fuzzy makespan-energy fitness; not the complete Lyapunov-LSTM pipeline of Pan et al. (2024).
* **Centralized PPO**: a single agent with a global view of all nodes that selects the target node directly; same
  GCN encoder, no federation.
* AFedGRL, FedTaskRL and FRAHTOS are not implemented (no public implementation is available); the paper
  discusses them qualitatively.

## Repository layout

```
fgrl/            library (config, workflows, simulator, graph, models, agents, federation, baselines, eval)
experiments/     one script per experiment, merge_chunks.py, make_figures.py, evidence_report.py
configs/         paper_pegasus.json (paper protocol), pilot_pegasus.json, test.json, test_wfcommons.json, paper.json
data/pegasus/    Pegasus benchmark workflows (DAX) with sources and SHA-256 checksums
data/test/       small WfCommons-format test workflows
scripts/         download_benchmarks.py, make_test_data.py
tests/           unit tests (pytest)
results/paper/   released run-level results, tables, figures and evidence report
```

## Test data

* `data/test/*.json`: 15 small workflow instances in WfCommons (WfFormat) JSON, 5 each of Montage, CyberShake and
  Epigenomics, about 50 tasks each (`configs/test_wfcommons.json`).
* `configs/test.json`: synthetic instances generated on the fly with the same structures.
* Other WfCommons traces (https://wfcommons.org) can be used by placing them in a folder and setting
  `"workflow_dir"`; the files of each type are split 60/20/20 into training, validation and test.

## Third-party data

The Pegasus workflow files in `data/pegasus/` were generated by the Pegasus Workflow Generator and are
redistributed from the WorkflowSim-1.0 repository for reproducibility; they remain subject to their original
terms. `scripts/download_benchmarks.py` downloads them from the original source and verifies their checksums.
Please cite Bharathi et al. (2008), Juve et al. (2013) and Chen and Deelman (2012) when using them (full
references in `data/pegasus/SOURCES.md`).
