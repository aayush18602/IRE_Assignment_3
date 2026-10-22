#!/usr/bin/env bash
# Regenerate every number, figure and table in the report.
#   bash run_all.sh          full data (results/)
#   bash run_all.sh --quick  first 100K vectors, fewer settings (results/quick/)
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
QUICK=""
if [[ "${1:-}" == "--quick" ]]; then
    QUICK="--quick"
    export A3_RESULTS_DIR="$PWD/results/quick"
else
    export A3_RESULTS_DIR="$PWD/results"
fi
mkdir -p "$A3_RESULTS_DIR/figures"
rm -f "$A3_RESULTS_DIR/results.csv"

echo "== tests"
$PY -m pytest -q tests

cd src
echo "== Part 1: exact baseline"
$PY part1_exact.py $QUICK
echo "== Part 2: HNSW / IVF-Flat / IVF-PQ"
$PY part2_bench.py $QUICK
$PY part2_report.py $QUICK
echo "== Part 3: query difficulty (GloVe)"
$PY part3_difficulty.py $QUICK
echo "done: results in $A3_RESULTS_DIR"
