#!/usr/bin/env bash
# Full, resumable reproduction of every result in Section 5 plus the evidence pack.
#
#   ./run_full.sh                                   # paper protocol, all CPU cores
#   JOBS=8 ./run_full.sh                            # 8 parallel jobs
#   CONFIG=configs/pilot_pegasus.json OUT=results/pilot ./run_full.sh   # reduced pilot
#   SEEDS="0 1 2" ./run_full.sh                     # only these seeds (e.g. one machine of several)
#
# Each (experiment, seed) job writes to $OUT/chunks/<experiment>_s<seed>/ and leaves a DONE file,
# so an interrupted run continues where it stopped. Chunks from several machines can be copied
# into one $OUT/chunks folder; the merge/report step at the end combines them.
set -euo pipefail
cd "$(dirname "$0")"
CONFIG=${CONFIG:-configs/paper_pegasus.json}
OUT=${OUT:-results/paper}
JOBS=${JOBS:-$(nproc)}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # one thread per job; parallelism comes from JOBS
mkdir -p "$OUT/chunks" "$OUT/logs"

python scripts/download_benchmarks.py
python -m pytest -q

ALL_SEEDS=$(python -c "import json;print(' '.join(map(str,json.load(open('$CONFIG'))['experiment']['seeds'])))")
COMM_N=$(python -c "import json;print(json.load(open('$CONFIG'))['experiment'].get('comm_seeds',3))")
SEEDS=${SEEDS:-$ALL_SEEDS}; [[ "$SEEDS" == none ]] && SEEDS=""
COMM_SEEDS=$(echo $ALL_SEEDS | tr ' ' '\n' | head -n "$COMM_N" | tr '\n' ' ')

jobs=()
for s in $SEEDS; do jobs+=("comparison $s"); done             # Table 7, statistics, convergence first
for s in $SEEDS; do jobs+=("ablation $s"); done
for s in $COMM_SEEDS; do [[ " $SEEDS " == *" $s "* ]] && jobs+=("communication $s"); done
for s in $SEEDS; do jobs+=("scalability $s"); done            # longest jobs last

run_one() {
  exp=$1; s=$2; d="$OUT/chunks/${exp}_s${s}"
  [[ -f "$d/DONE" ]] && { echo "skip $exp seed $s (done)"; return 0; }
  mkdir -p "$d"
  start=$(date +%s)
  if [[ $exp == ablation ]]; then
    python experiments/run_ablation.py --config "$CONFIG" --out "$d" --seeds "$s" \
      --methods "FGRL w/o GNN" "FGRL w/o critical-path reward" "FGRL w/o federation" > "$OUT/logs/${exp}_s${s}.log" 2>&1
  else
    python "experiments/run_${exp}.py" --config "$CONFIG" --out "$d" --seeds "$s" > "$OUT/logs/${exp}_s${s}.log" 2>&1
  fi
  echo "$(( $(date +%s) - start ))" > "$d/DONE"
  echo "finished $exp seed $s in $(( $(date +%s) - start )) s"
}
export -f run_one; export CONFIG OUT
printf '%s\n' "${jobs[@]}" | xargs -r -P "$JOBS" -L 1 bash -c 'run_one $0 $1'

# merge -> figures -> evidence report (only when every job of the full seed list is done)
missing=0
for s in $ALL_SEEDS; do for e in comparison ablation scalability; do [[ -f "$OUT/chunks/${e}_s${s}/DONE" ]] || missing=1; done; done
if [[ $missing == 1 ]]; then echo "Some seeds are not finished yet (or run on another machine). Merge later with: SEEDS=none ./run_full.sh"; fi
python experiments/merge_chunks.py --out "$OUT" "$OUT"/chunks/*
python experiments/make_figures.py --results "$OUT"
python experiments/evidence_report.py --results "$OUT" --config "$CONFIG"
echo "Evidence pack: $OUT/EVIDENCE.md"
