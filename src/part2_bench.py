"""Part 2: recall/QPS sweeps of HNSW, IVF-Flat and IVF-PQ on SIFT and GloVe.

For every build config the search parameter is swept on the tune queries (first
5,000) until recall stops improving or reaches STOP_RECALL; the same points are
then timed on the test queries (last 5,000). Choosing operating points from the
tune rows happens in part2_report.py.
"""
import argparse
import gc
import json
import os

import data
import results
from indexes import SEARCH_PARAM, build, fmt_params, index_bytes, set_search
from recall import recall_at_k
from timing import time_single_queries

STOP_RECALL = 0.99
EF = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
NPROBE = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]

GRID_FULL = {
    "sift": {
        "HNSW": [{"M": m, "efC": 200} for m in (8, 16, 32)],
        "IVF-Flat": [{"nlist": n} for n in (256, 1024, 4096)],
        "IVF-PQ": [{"nlist": 1024, "m": m, "nbits": 8} for m in (16, 32)],
    },
    "glove": {
        "HNSW": [{"M": m, "efC": 200} for m in (8, 16, 32)],
        "IVF-Flat": [{"nlist": n} for n in (256, 1024, 4096)],
        "IVF-PQ": [{"nlist": 1024, "m": m, "nbits": 8} for m in (20, 50)],
    },
}
GRID_QUICK = {
    "sift": {
        "HNSW": [{"M": 16, "efC": 100}],
        "IVF-Flat": [{"nlist": 256}],
        "IVF-PQ": [{"nlist": 256, "m": 16, "nbits": 8}],
    },
    "glove": {
        "HNSW": [{"M": 16, "efC": 100}],
        "IVF-Flat": [{"nlist": 256}],
        "IVF-PQ": [{"nlist": 256, "m": 20, "nbits": 8}],
    },
}


def sweep_values(kind, params):
    if kind == "HNSW":
        return EF
    return [p for p in NPROBE if p <= params["nlist"]]


def measure(index, queries, gt, base, repeats):
    med, times, ids = time_single_queries(index, queries, repeats=repeats)
    rec = recall_at_k(ids, gt, queries, base).mean()
    return rec, med, times


def bench_config(ds, gt_ids, kind, params, threads, tune_repeats, test_repeats, max_q):
    index, build_s = build(kind, params, ds.base, threads)
    nbytes = index_bytes(index)
    common = {"part": "2", "dataset": ds.name, "n": ds.n, "index": kind,
              "build_params": fmt_params(params), "build_s": build_s,
              "index_bytes": nbytes, "bytes_per_vec": nbytes / ds.n,
              "notes": f"single-query 1 thread; build {threads} threads"}
    print(f"[{ds.name}] {kind} {fmt_params(params)}: build {build_s:.1f}s, "
          f"{nbytes / ds.n:.0f} B/vec", flush=True)
    tune_q, tune_gt = ds.tune[:max_q], gt_ids[:data.N_TUNE][:max_q]
    test_q, test_gt = ds.test[:max_q], gt_ids[data.N_TUNE:][:max_q]
    pname = SEARCH_PARAM[kind]

    rows, recalls = [], []
    for v in sweep_values(kind, params):
        set_search(index, kind, v)
        for split, q, gt, reps in (("tune", tune_q, tune_gt, tune_repeats),
                                   ("test", test_q, test_gt, test_repeats)):
            rec, med, times = measure(index, q, gt, ds.base, reps)
            rows.append({**common, "search_param": f"{pname}={v}", "query_split": split,
                         "n_queries": len(q), "recall": rec, "qps": len(q) / med,
                         "median_s": med, "repeat_s": json.dumps(times)})
            if split == "tune":
                recalls.append(rec)
                print(f"    {pname}={v:<5d} tune recall {rec:.4f}  qps {len(q) / med:9.1f}", flush=True)
        # stop once recall is high enough or has flattened out (e.g. the PQ ceiling)
        if recalls[-1] >= STOP_RECALL or (len(recalls) >= 4 and recalls[-1] - recalls[-4] < 0.002):
            break
    results.append(rows)
    del index
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=["sift", "glove"])
    ap.add_argument("--indexes", nargs="+", default=["HNSW", "IVF-Flat", "IVF-PQ"])
    ap.add_argument("--build-threads", type=int, default=os.cpu_count())
    ap.add_argument("--tune-repeats", type=int, default=1)
    ap.add_argument("--test-repeats", type=int, default=3)
    ap.add_argument("--max-queries", type=int, default=None,
                    help="cap queries per split (default: all 5,000)")
    a = ap.parse_args()
    n_base = 100_000 if a.quick else None
    if a.quick and a.max_queries is None:
        a.max_queries = 1000
    grid = GRID_QUICK if a.quick else GRID_FULL
    for name in a.datasets:
        ds = data.load(name, n_base)
        gt_ids, _ = data.ground_truth(ds)
        for kind in a.indexes:
            for params in grid[name][kind]:
                bench_config(ds, gt_ids, kind, params, a.build_threads,
                             a.tune_repeats, a.test_repeats, a.max_queries)


if __name__ == "__main__":
    main()
