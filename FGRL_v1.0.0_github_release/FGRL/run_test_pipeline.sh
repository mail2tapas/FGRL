#!/usr/bin/env bash
# End-to-end check with the small test configuration (about 15-30 minutes on a laptop CPU).
# Produces every table/figure of the paper in results/test from fresh runs.
set -euo pipefail
cd "$(dirname "$0")"
OUT=${1:-results/test}
python -m pytest -q
python experiments/run_comparison.py    --config configs/test.json --out "$OUT"
python experiments/run_ablation.py      --config configs/test.json --out "$OUT"
python experiments/run_scalability.py   --config configs/test.json --out "$OUT"
python experiments/run_communication.py --config configs/test.json --out "$OUT"
python experiments/make_figures.py      --results "$OUT"
echo "All outputs written to $OUT"
