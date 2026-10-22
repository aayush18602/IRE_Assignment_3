# Benchmarking approximate nearest neighbour search (IRE Assignment 3)

Benchmarks exact search, HNSW, IVF-Flat and IVF-PQ (all FAISS) on SIFT-1M and
GloVe-100, plus two experiments (Part 3: query difficulty on GloVe; Part 4: an
experiment on generated data). Recall, timing and plotting code is our own.

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Download the two ann-benchmarks files into `../data/` (next to `code/`), or set
`A3_DATA_DIR` to another folder:

```bash
mkdir -p ../data && cd ../data
curl -LO http://ann-benchmarks.com/sift-128-euclidean.hdf5
curl -LO http://ann-benchmarks.com/glove-100-angular.hdf5
```

## Reproducing the report

```bash
bash run_all.sh           # full data; writes results/
bash run_all.sh --quick   # first 100K vectors, 1,000 queries per split, smaller grids;
                          # writes results/quick/ (under 15 min on a laptop)
```

Use `PYTHON=/path/to/python bash run_all.sh` to choose the interpreter.
Exact ground truth is computed once with a FAISS flat index and cached in the
data folder (`gt_*.npz`).

Outputs:
- `results/results.csv` — one row per benchmark run (dataset, index, build and
  search parameters, query split, recall@10, QPS, per-repeat times, build time,
  index size).
- `results/figures/` — all figures.
- `results/tables/` — the report tables as Markdown and CSV.

## Method notes

- GloVe vectors are unit-normalised, and squared L2 is used everywhere.
- Queries 0–4,999 are used for tuning and queries 5,000–9,999 for the reported numbers.
  Operating points (e.g. "recall 0.95") are chosen on tune queries and then
  reported on test queries.
- QPS is measured one query at a time on one thread (`faiss.omp_set_num_threads(1)`),
  as the median of 3 repeats. Index builds use all cores.
- recall@10 (`src/recall.py`) recomputes true distances in float64. A returned id
  counts as correct if it is no farther than the 10th true neighbour, so ties at
  rank 10 are not counted as mistakes. Duplicate ids and `-1` padding never count.
- hnswlib could not be built on the benchmark machine (no C++ compiler), so HNSW
  is FAISS `IndexHNSWFlat`.

## Layout

| Path | Contents |
|---|---|
| `src/data.py` | loading, normalisation, query split, cached exact ground truth |
| `src/recall.py` | tie-aware recall@k |
| `src/timing.py` | single-query and batch timers |
| `src/indexes.py` | index builders and search-parameter setters |
| `src/part1_exact.py` | exact baseline, ground-truth check, single vs batch timing |
| `src/part2_bench.py`, `src/part2_report.py` | index sweeps; plots and tables |
| `tests/` | unit tests for recall |
| `hypothesis.md` | Part 3 and Part 4 hypotheses (committed before each experiment) |
