# Release notes

## v1.0.0 (2026-10-19)

First public release. This is the exact code, data and results used for the revised version of the
paper "Federated Graph Reinforcement Learning for Dependency-Aware Scheduling in Fog-Cloud Computing"
(manuscript IJMEMS-26-0439).

Contents:

* Discrete-event fog-cloud simulator (three tiers, dynamic load spikes and execution-time noise).
* FGRL: 3-layer GCN encoder, multi-agent PPO at the fog nodes, critical-path-aware reward, and
  variance-weighted federated averaging.
* Baselines: HEFT, PSO-Fuzzy (simplified re-implementation) and centralized PPO; ablation variants
  without the GNN, without the critical-path reward and without federation.
* Pegasus Montage, CyberShake and Epigenomics benchmark workflows with SHA-256 checksums.
* Experiment scripts for the comparison, ablation, scalability (10-80 fog nodes) and communication studies,
  figure generation (600 dpi) and the evidence report with all statistical tests.
* Run-level results of the full protocol (`results/paper/`) and `MANIFEST.sha256`.
* 15 unit tests and a Google Colab notebook for the full run.

Main results (30 paired blocks, Pegasus benchmarks, 10 fog nodes): makespan 35.1% lower and energy 14.1% lower
than HEFT; FGRL is better than HEFT, PSO-Fuzzy and centralized PPO in all 30 blocks on all four metrics
(Wilcoxon signed-rank with Holm correction, p < 0.001). See `results/paper/EVIDENCE.md` for all values.

---

## How to publish this release on GitHub

1. Create a **public** repository named `FGRL` under the account `mail2tapas` (do not add a README or license
   on GitHub; they are already included).
2. Upload the contents of this folder (not the folder itself), either with
   *Add file → Upload files* in the browser, or from a terminal:

   ```bash
   cd FGRL
   git init
   git add .
   git commit -m "FGRL v1.0.0: reference implementation, benchmark data and results"
   git branch -M main
   git remote add origin https://github.com/mail2tapas/FGRL.git
   git push -u origin main
   ```

3. Create the release tag:

   ```bash
   git tag -a v1.0.0 -m "FGRL v1.0.0 - code, data and results of the revised manuscript IJMEMS-26-0439"
   git push origin v1.0.0
   ```

   or, in the browser: *Releases → Draft a new release → Choose a tag → type `v1.0.0` → Create new tag*,
   title "FGRL v1.0.0", and paste the "v1.0.0" section above as the description.

4. Optional, recommended: archive the release on Zenodo to obtain a permanent DOI. Sign in to
   https://zenodo.org with GitHub, enable the `FGRL` repository under *GitHub* in the Zenodo settings, and then
   publish the release (step 3). Zenodo creates the DOI automatically; add it to `CITATION.cff` (`doi:`) and,
   if the journal allows, to the code-availability statement of the paper.

5. Check: open the repository page without being signed in, make sure `README.md` is displayed, and run
   `sha256sum -c results/paper/MANIFEST.sha256` in a fresh clone.

Do not change the files of the v1.0.0 tag after submission. Later changes go into a new release (v1.0.1, v1.1.0),
so the tag always matches the results reported in the paper.
