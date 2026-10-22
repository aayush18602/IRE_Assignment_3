"""Part 1: exact baseline, ground-truth check and single vs batched timing."""
import argparse
import json
import os

import faiss
import numpy as np

import data
import results
from recall import recall_at_k
from timing import time_batch, time_single_queries


def check_stored(ds, gt_ids):
    """Compare our exact top-10 with the neighbours stored in the file."""
    ours, stored = gt_ids[:, :10], ds.stored_gt[:, :10]
    same_set = np.mean([set(a) == set(b) for a, b in zip(ours, stored)])
    # tie-aware: stored neighbours scored against our exact results
    r = recall_at_k(stored, ours, ds.queries, ds.base, k=10)
    return {"same_id_set_frac": float(same_set), "stored_recall_mean": float(r.mean()),
            "stored_recall_min": float(r.min()), "n_queries_differ": int((r < 1).sum())}


def run(name, n_base, single_n, repeats):
    ds = data.load(name, n_base)
    print(f"[{name}] n={ds.n} d={ds.d}")
    gt_ids, _ = data.ground_truth(ds)
    summary = {"dataset": name, "n": ds.n}
    if ds.stored_gt is not None:
        summary["stored_check"] = check_stored(ds, gt_ids)
        print("  stored-gt check:", summary["stored_check"])

    index = faiss.IndexFlatL2(ds.d)
    index.add(ds.base)
    test_gt = gt_ids[data.N_TUNE:]
    common = {"part": "1", "dataset": name, "n": ds.n, "index": "Flat",
              "build_params": "", "search_param": "", "query_split": "test",
              "build_s": 0, "index_bytes": ds.base.nbytes, "bytes_per_vec": ds.base.nbytes / ds.n}
    rows = []

    q = ds.test[:single_n]
    med, times, ids = time_single_queries(index, q, repeats=repeats)
    rec = recall_at_k(ids, test_gt[:single_n], q, ds.base).mean()
    rows.append({**common, "n_queries": len(q), "recall": rec, "qps": len(q) / med,
                 "median_s": med, "repeat_s": json.dumps(times), "notes": "single-query 1 thread"})
    print(f"  single-query, 1 thread: {1e3 * med / len(q):.2f} ms/query, recall {rec:.4f}")

    for threads in (1, os.cpu_count()):
        med, times, ids = time_batch(index, ds.test, repeats=repeats, threads=threads)
        rec = recall_at_k(ids, test_gt, ds.test, ds.base).mean()
        rows.append({**common, "n_queries": len(ds.test), "recall": rec, "qps": len(ds.test) / med,
                     "median_s": med, "repeat_s": json.dumps(times),
                     "notes": f"batch {threads} thread(s)"})
        print(f"  batch, {threads} thread(s): {1e3 * med / len(ds.test):.3f} ms/query, recall {rec:.4f}")

    results.append(rows)
    summary["rows"] = rows
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=["sift", "glove"])
    ap.add_argument("--single-n", type=int, default=1000,
                    help="test queries used for the one-at-a-time exact timing")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    n_base = 100_000 if a.quick else None
    out = [run(name, n_base, a.single_n, a.repeats) for name in a.datasets]
    os.makedirs(results.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(results.RESULTS_DIR, "part1_summary.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
