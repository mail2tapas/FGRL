# FGRL – evidence report

Generated 2026-10-19T03:38:24+00:00 from the raw results in `results/paper`.

Configuration: `configs/paper_pegasus.json`. Machine: Intel(R) Xeon(R) Processor @ 2.10GHz, 2 CPU threads, Python 3.13.16, PyTorch 2.14.1+cu130.

Benchmark workflows: Pegasus Montage, CyberShake and Epigenomics DAX files (Bharathi et al., 2008; Juve et al., 2013), see `data/pegasus/SOURCES.md`. Every file used is listed with its SHA-256 in `MANIFEST.sha256`, so the results can be traced to the exact code and data.

Figures: `figure3_scalability.png`, `figure4_communication.png`, `figure5_convergence.png`, `figure6_comparison.png`, `figure7_ablation.png` (600 dpi).

## 1. Performance comparison (Table 7, Figure 6)

Blocks: 10 seeds × 3 workflow types = 30 (seed, workflow) blocks per method. Each block value is the mean over the held-out test instances and dynamic-condition seeds of that block.

### 1.1 Mean ± SD over seeds, per workflow type

| Workflow | Approach | n | Makespan (s) | Energy (kWh) | SLA violation (%) | Avg. utilization (%) |
|---|---|---|---|---|---|---|
| Montage | HEFT | 10 | 285.95 ± 22.36 | 0.0315 ± 0.0013 | 68.31 ± 3.41 | 47.85 ± 3.96 |
| Montage | PSO-Fuzzy | 10 | 286.27 ± 22.36 | 0.0315 ± 0.0013 | 68.32 ± 3.41 | 47.87 ± 4.01 |
| Montage | Centralized PPO | 10 | 375.92 ± 72.25 | 0.0389 ± 0.0053 | 81.27 ± 2.00 | 22.20 ± 3.49 |
| Montage | FGRL | 10 | 174.53 ± 5.76 | 0.0254 ± 0.0011 | 61.78 ± 1.51 | 69.81 ± 2.18 |
| Cybershake | HEFT | 10 | 818.43 ± 86.31 | 0.0940 ± 0.0055 | 37.68 ± 6.40 | 48.80 ± 4.19 |
| Cybershake | PSO-Fuzzy | 10 | 820.17 ± 85.42 | 0.0940 ± 0.0055 | 37.93 ± 6.36 | 48.76 ± 4.17 |
| Cybershake | Centralized PPO | 10 | 774.44 ± 54.10 | 0.1122 ± 0.0057 | 36.44 ± 6.20 | 26.30 ± 2.54 |
| Cybershake | FGRL | 10 | 527.38 ± 20.97 | 0.0839 ± 0.0031 | 18.02 ± 3.79 | 61.63 ± 2.93 |
| Epigenomics | HEFT | 10 | 105780.07 ± 5860.07 | 11.4392 ± 0.4423 | 30.63 ± 1.32 | 49.72 ± 3.39 |
| Epigenomics | PSO-Fuzzy | 10 | 105780.98 ± 5860.23 | 11.4392 ± 0.4423 | 30.63 ± 1.32 | 49.72 ± 3.39 |
| Epigenomics | Centralized PPO | 10 | 110718.12 ± 6541.72 | 14.7872 ± 0.6706 | 16.98 ± 1.45 | 21.11 ± 2.01 |
| Epigenomics | FGRL | 10 | 72091.54 ± 5289.84 | 9.9771 ± 0.4746 | 12.07 ± 1.05 | 60.31 ± 3.66 |

### 1.2 Pooled over all blocks (makespan normalized to HEFT within each block)

| Approach | Normalized makespan (÷HEFT) | SLA violation (%) | Avg. utilization (%) | Makespan reduction vs HEFT (%) [95% CI] |
|---|---|---|---|---|
| HEFT | 1.000 ± 0.000 | 45.54 | 48.79 | nan |
| PSO-Fuzzy | 1.001 ± 0.004 | 45.63 | 48.78 | -0.1 [-0.3, 0.0] |
| Centralized PPO | 1.107 ± 0.215 | 44.90 | 23.20 | -10.7 [-18.8, -3.5] |
| FGRL | 0.649 ± 0.061 | 30.62 | 63.92 | 35.1 [32.9, 37.3] |

### 1.3 Improvement of FGRL (positive = FGRL better), block-wise

Lower-is-better metrics: 100·(1 − FGRL/baseline); utilization: 100·(FGRL/baseline − 1). SLA ratios are undefined when a baseline has 0 % violations in a block; those blocks are skipped.

| Metric | vs | Mean block-wise improvement (%) | 95% bootstrap CI | Montage | Cybershake | Epigenomics |
|---|---|---|---|---|---|---|
| Makespan (s) | HEFT | 35.1 | [32.9, 37.2] | 38.7 | 35.0 | 31.7 |
| Makespan (s) | PSO-Fuzzy | 35.2 | [33.0, 37.3] | 38.8 | 35.1 | 31.7 |
| Makespan (s) | Centralized PPO | 39.5 | [35.8, 43.5] | 52.2 | 31.6 | 34.7 |
| Energy (kWh) | HEFT | 14.1 | [12.0, 16.1] | 19.2 | 10.5 | 12.7 |
| Energy (kWh) | PSO-Fuzzy | 14.1 | [12.1, 16.1] | 19.2 | 10.5 | 12.7 |
| Energy (kWh) | Centralized PPO | 30.4 | [28.2, 32.8] | 33.7 | 25.1 | 32.5 |
| SLA violation (%) | HEFT | 40.4 | [32.0, 48.7] | 9.5 | 51.3 | 60.5 |
| SLA violation (%) | PSO-Fuzzy | 40.6 | [32.1, 48.8] | 9.5 | 51.7 | 60.5 |
| SLA violation (%) | Centralized PPO | 33.7 | [28.9, 39.2] | 23.9 | 48.8 | 28.5 |
| Avg. utilization (%) | HEFT | 31.8 | [26.6, 37.2] | 46.6 | 27.2 | 21.7 |
| Avg. utilization (%) | PSO-Fuzzy | 31.9 | [26.6, 37.3] | 46.6 | 27.3 | 21.7 |
| Avg. utilization (%) | Centralized PPO | 182.2 | [164.2, 200.9] | 221.3 | 136.5 | 188.9 |

## 2. Statistical tests (Section 5.7)

Friedman test over the (seed, workflow) blocks, Nemenyi post hoc test (critical difference at α = 0.05), two-sided Wilcoxon signed-rank tests of FGRL against each baseline with Holm correction, and the paired effect size d_z = mean(diff)/sd(diff) with a 95 % bootstrap confidence interval. Differences are taken as log(baseline/FGRL) for makespan and energy (scale-free across workflow types) and as percentage points for SLA violation and utilization, so that positive values favour FGRL.

### Makespan (s)

Friedman χ²(3) = 60.09, p = 5.62e-13, N = 30 blocks. Average ranks (1 = best): FGRL 1.00, HEFT 2.62, PSO-Fuzzy 3.05, Centralized PPO 3.33. Nemenyi CD = 0.86.

| FGRL vs | FGRL better in | Wilcoxon p | Holm-adjusted p | Nemenyi p | Paired d_z [95% CI] |
|---|---|---|---|---|---|
| HEFT | 30/30 blocks | 1.86e-09 | 5.59e-09 | 7.35e-06 | 4.71 [3.73, 6.75] |
| PSO-Fuzzy | 30/30 blocks | 1.86e-09 | 5.59e-09 | 4.65e-09 | 4.73 [3.75, 6.86] |
| Centralized PPO | 30/30 blocks | 1.86e-09 | 5.59e-09 | 1.54e-11 | 2.56 [2.19, 3.39] |

### Energy (kWh)

Friedman χ²(3) = 83.02, p = 6.9e-18, N = 30 blocks. Average ranks (1 = best): FGRL 1.00, PSO-Fuzzy 2.40, HEFT 2.60, Centralized PPO 4.00. Nemenyi CD = 0.86.

| FGRL vs | FGRL better in | Wilcoxon p | Holm-adjusted p | Nemenyi p | Paired d_z [95% CI] |
|---|---|---|---|---|---|
| HEFT | 30/30 blocks | 1.86e-09 | 5.59e-09 | 9.44e-06 | 2.32 [1.84, 3.21] |
| PSO-Fuzzy | 30/30 blocks | 1.86e-09 | 5.59e-09 | 0.000157 | 2.33 [1.84, 3.20] |
| Centralized PPO | 30/30 blocks | 1.86e-09 | 5.59e-09 | 0 | 3.78 [2.98, 5.66] |

### SLA violation (%)

Friedman χ²(3) = 57.98, p = 1.59e-12, N = 30 blocks. Average ranks (1 = best): FGRL 1.00, Centralized PPO 2.90, HEFT 3.02, PSO-Fuzzy 3.08. Nemenyi CD = 0.86.

| FGRL vs | FGRL better in | Wilcoxon p | Holm-adjusted p | Nemenyi p | Paired d_z [95% CI] |
|---|---|---|---|---|---|
| HEFT | 30/30 blocks | 1.86e-09 | 5.59e-09 | 8.68e-09 | 2.10 [1.69, 2.85] |
| PSO-Fuzzy | 30/30 blocks | 1.86e-09 | 5.59e-09 | 2.46e-09 | 2.11 [1.70, 2.89] |
| Centralized PPO | 30/30 blocks | 1.73e-06 | 1.73e-06 | 7.18e-08 | 1.71 [1.38, 2.31] |

### Avg. utilization (%)

Friedman χ²(3) = 82.82, p = 7.64e-18, N = 30 blocks. Average ranks (1 = best): FGRL 1.00, HEFT 2.43, PSO-Fuzzy 2.57, Centralized PPO 4.00. Nemenyi CD = 0.86.

| FGRL vs | FGRL better in | Wilcoxon p | Holm-adjusted p | Nemenyi p | Paired d_z [95% CI] |
|---|---|---|---|---|---|
| HEFT | 30/30 blocks | 1.86e-09 | 5.59e-09 | 0.000101 | 2.29 [1.80, 3.17] |
| PSO-Fuzzy | 30/30 blocks | 1.86e-09 | 5.59e-09 | 1.55e-05 | 2.29 [1.80, 3.15] |
| Centralized PPO | 30/30 blocks | 1.86e-09 | 5.59e-09 | 0 | 6.10 [5.31, 7.64] |

## 3. Training convergence (Section 5.2, Figure 5)

Definition: first episode after which the 50-episode moving average changes by less than 5% (max-min relative to mean) over 100 consecutive episodes. One convergence episode per (seed, workflow type) run; episodes are counted per workflow type.

| Workflow | Method | Converged runs | Convergence episode (mean ± SD) |
|---|---|---|---|
| Montage | FGRL | 9/10 | 108 ± 58 |
| Montage | Centralized PPO | 0/10 | – ± – |
| Cybershake | FGRL | 8/10 | 136 ± 39 |
| Cybershake | Centralized PPO | 0/10 | – ± – |
| Epigenomics | FGRL | 5/10 | 107 ± 84 |
| Epigenomics | Centralized PPO | 0/10 | – ± – |
| Pooled | FGRL | 22/30 | 118 ± 58 |
| Pooled | Centralized PPO | 0/30 | – ± – |

Sensitivity to the stability tolerance (same moving-average window and stability window):

| Tolerance | Method | Converged runs | Convergence episode (mean ± SD) |
|---|---|---|---|
| 2% | FGRL | 1/30 | 120 ± – |
| 2% | Centralized PPO | 0/30 | – ± – |
| 5% | FGRL | 22/30 | 118 ± 58 |
| 5% | Centralized PPO | 0/30 | – ± – |
| 10% | FGRL | 30/30 | 59 ± 27 |
| 10% | Centralized PPO | 22/30 | 63 ± 35 |

Model-selection point (episode at which the policy that is finally evaluated reached its best validation score; validation every 50 episodes):

| Method | Episode of best validation score (mean ± SD) | Runs |
|---|---|---|
| Centralized PPO | 175 ± 95 | 10 |
| FGRL | 410 ± 225 | 10 |

Paired Wilcoxon test (same seed), FGRL vs centralized PPO: p = 0.0156.

Training length with early stopping (patience 5 evaluations = 250 episodes without improvement): Centralized PPO: mean 425 episodes (range 300–600); FGRL: mean 660 episodes (range 350–1000).

Runs that end (early stopping) before the stability criterion is met are counted as not converged.

## 4. Ablation study (Section 5.5, Figure 7)

| Variant | Makespan increase vs full FGRL (%) [95% CI] | Montage | Cybershake | Epigenomics | Wilcoxon p (log ratio) | Holm-adjusted p | FGRL better | n blocks |
|---|---|---|---|---|---|---|---|---|
| FGRL w/o GNN | 2.3 [0.5, 4.1] | 1.8 | 1.5 | 3.6 | 0.0277 | 0.0555 | 21/30 | 30.0 |
| FGRL w/o critical-path reward | 0.9 [-1.0, 2.9] | 1.1 | -0.3 | 2.0 | 0.49 | 0.49 | 16/30 | 30.0 |
| FGRL w/o federation | 9.7 [6.3, 13.3] | 11.9 | 6.1 | 11.3 | 2.35e-06 | 7.05e-06 | 26/30 | 30.0 |
| FGRL |  |  |  |  |  |  |  |  |

## 5. Scalability with the number of fog nodes (Table 5, Figure 3)

| Workflow | Fog nodes | HEFT | Centralized PPO | FGRL |
|---|---|---|---|---|
| Montage | 10 | 275.59 ± 10.10 | 395.73 ± 39.63 | 171.57 ± 3.49 |
| Montage | 20 | 192.74 ± 1.62 | 592.62 ± 83.39 | 143.08 ± 2.16 |
| Montage | 30 | 151.53 ± 3.01 | 624.36 ± 81.54 | 131.94 ± 4.04 |
| Montage | 40 | 137.20 ± 3.73 | 562.26 ± 42.17 | 125.97 ± 5.80 |
| Montage | 50 | 133.01 ± 2.16 | 540.79 ± 1.04 | 121.24 ± 6.06 |
| Montage | 60 | 129.08 ± 3.00 | 541.37 ± 1.68 | 131.69 ± 11.19 |
| Montage | 70 | 127.91 ± 2.08 | 541.19 ± 2.01 | 121.41 ± 9.56 |
| Montage | 80 | 124.84 ± 3.32 | 541.97 ± 1.61 | 122.06 ± 3.74 |
| Cybershake | 10 | 856.17 ± 63.91 | 779.11 ± 36.11 | 518.14 ± 35.17 |
| Cybershake | 20 | 603.39 ± 34.25 | 1193.17 ± 150.29 | 482.26 ± 75.41 |
| Cybershake | 30 | 536.74 ± 53.62 | 1384.89 ± 123.06 | 464.63 ± 51.45 |
| Cybershake | 40 | 557.46 ± 26.98 | 1408.67 ± 95.64 | 475.42 ± 32.34 |
| Cybershake | 50 | 484.42 ± 23.97 | 1576.34 ± 61.11 | 456.79 ± 11.35 |
| Cybershake | 60 | 472.42 ± 97.86 | 1595.57 ± 38.63 | 507.37 ± 27.38 |
| Cybershake | 70 | 477.02 ± 24.57 | 1590.60 ± 43.43 | 445.64 ± 22.07 |
| Cybershake | 80 | 487.52 ± 22.11 | 1599.15 ± 55.99 | 440.91 ± 28.58 |
| Epigenomics | 10 | 100214.92 ± 2546.66 | 109938.18 ± 10163.44 | 70328.49 ± 8049.90 |
| Epigenomics | 20 | 74520.75 ± 2861.31 | 165780.95 ± 6184.29 | 65302.44 ± 7063.12 |
| Epigenomics | 30 | 61350.26 ± 5885.27 | 199305.97 ± 4044.79 | 60762.27 ± 8428.65 |
| Epigenomics | 40 | 54386.65 ± 2019.91 | 201302.53 ± 11541.74 | 57944.45 ± 6589.42 |
| Epigenomics | 50 | 52150.38 ± 2577.41 | 204909.95 ± 6384.24 | 58756.67 ± 7027.03 |
| Epigenomics | 60 | 52388.96 ± 4499.91 | 205709.33 ± 6403.49 | 62547.52 ± 6817.53 |
| Epigenomics | 70 | 48873.63 ± 1819.68 | 205608.17 ± 5661.43 | 54930.15 ± 2914.83 |
| Epigenomics | 80 | 50335.57 ± 3058.32 | 205392.11 ± 6539.39 | 56864.42 ± 1745.84 |

| Fog nodes | seeds | FGRL reduction vs HEFT (%) | FGRL reduction vs Centralized PPO (%) |
|---|---|---|---|
| 10 | 3 | 35.6 | 41.8 |
| 20 | 3 | 19.3 | 65.2 |
| 30 | 3 | 9.1 | 71.4 |
| 40 | 3 | 5.4 | 71.6 |
| 50 | 3 | 0.4 | 73.3 |
| 60 | 3 | -11.1 | 71.2 |
| 70 | 3 | -0.4 | 74.3 |
| 80 | 3 | -0.4 | 74.1 |

## 6. Communication overhead (Figure 4)

Overhead = modelled transfer time of all federated rounds ÷ (measured training time + transfer time).

| K | rounds | MB_total | overhead_mean | overhead_sd | seeds |
|---|---|---|---|---|---|
| 10.0 | 200.0 | 269.392 | 2.073 | 0.043 | 3.0 |
| 25.0 | 80.0 | 107.757 | 0.842 | 0.022 | 3.0 |
| 50.0 | 40.0 | 53.878 | 0.434 | 0.006 | 3.0 |
| 100.0 | 20.0 | 26.939 | 0.215 | 0.003 | 3.0 |
| 200.0 | 10.0 | 13.47 | 0.106 | 0.001 | 3.0 |

## 7. Per-node footprint and inference latency (Sections 4.2.1, 5.3) – measured on the run machine only

```
{
  "machine": "x86_64",
  "python": "3.13.16",
  "torch": "2.14.1+cu130",
  "threads": 1,
  "model_params_MB_per_node": 0.067348,
  "experience_buffer_MB_per_node_upper_bound": 0.02,
  "workflow_tasks": 100,
  "inference_ms_per_decision_mean": 0.7074092700349865,
  "inference_ms_per_decision_p95": 0.8874670014847651
}
```

These latencies are for the machine above. Values for Raspberry Pi / Intel NUC (Table 6) require running `experiments/run_communication.py` on those devices.
