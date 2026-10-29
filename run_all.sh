#!/usr/bin/env bash
# Regenerate every number, figure and table in the report.
#   bash run_all.sh          full data (results/)
#   bash run_all.sh --quick  first 100K vectors, fewer settings (results/quick/)
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
[[ "$PY" == */* ]] && PY=$(realpath -s "$PY")   # a relative path must survive the cd below
QUICK=""
if [[ "${1:-}" == "--quick" ]]; then
    QUICK="--quick"
    export A3_RESULTS_DIR="$PWD/results/quick"
else
    export A3_RESULTS_DIR="$PWD/results"
fi
mkdir -p "$A3_RESULTS_DIR/figures"
rm -f "$A3_RESULTS_DIR/results.csv"

echo "== environment"
$PY - > "$A3_RESULTS_DIR/environment.txt" <<'EOF'
import datetime, os, platform
import faiss, h5py, matplotlib, numpy, pandas
cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
mem = next((l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal")), "0")
print(f"date: {datetime.datetime.now().isoformat(timespec='seconds')}")
print(f"cpu: {cpu} ({os.cpu_count()} logical cores)")
print(f"ram: {int(mem) / 2**20:.1f} GiB")
print(f"os: {platform.platform()}")
print(f"python: {platform.python_version()}")
for m in (faiss, numpy, h5py, pandas, matplotlib):
    print(f"{m.__name__}: {m.__version__}")
EOF
cat "$A3_RESULTS_DIR/environment.txt"

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
echo "== Part 4: drift on generated data"
$PY part4_drift.py $QUICK
echo "done: results in $A3_RESULTS_DIR"
