"""Part 3: recall per query-difficulty bin on GloVe-100.

Difficulty is the relative contrast (RC) of each test query: mean distance to a
fixed random sample of base vectors divided by the distance to its 10th true
neighbour. Low RC = the neighbours are barely closer than anything else = hard.
Test queries are split into RC quartiles, and every index is run at the
operating point chosen on the tune queries in Part 2 for overall recall 0.95
(IVF-PQ, which cannot reach it, at its most accurate config).
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

import data
import plots
import results
from indexes import build, set_search
from part2_report import INDEXES, load, operating_point
from recall import recall_at_k
from timing import single_thread

DATASET = "glove"
TARGET = 0.95
N_BINS = 4
RC_SAMPLE = 10_000


def relative_contrast(base, queries, kth_dist_sq, seed=0):
    rng = np.random.default_rng(seed)
    sample = base[rng.choice(base.shape[0], RC_SAMPLE, replace=False)].astype(np.float64)
    q = queries.astype(np.float64)
    d2 = (q ** 2).sum(1)[:, None] + (sample ** 2).sum(1)[None, :] - 2 * q @ sample.T
    mean_dist = np.sqrt(np.maximum(d2, 0)).mean(axis=1)
    return mean_dist / np.sqrt(kth_dist_sq)


def parse_params(s):
    return {k: int(v) for k, v in (kv.split("=") for kv in s.split(";"))}


def per_query_run(index, queries, k=10):
    """One query at a time, 1 thread; returns ids and per-query latency (s)."""
    single_thread()
    ids = np.empty((len(queries), k), dtype=np.int64)
    lat = np.empty(len(queries))
    for i in range(len(queries)):
        t0 = time.perf_counter()
        _, ids[i] = index.search(queries[i:i + 1], k)
        lat[i] = time.perf_counter() - t0
    return ids, lat


def plot(table, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    x = np.arange(N_BINS)
    labels = [f"Q{b + 1}" + ("\nhardest" if b == 0 else "\neasiest" if b == N_BINS - 1 else "")
              for b in range(N_BINS)]
    for index in INDEXES:
        sub = table[table["index"] == index].sort_values("bin")
        if sub.empty:
            continue
        st = plots.INDEX_STYLE[index]
        axes[0].plot(x, sub.recall, marker=st["marker"], color=st["color"], label=index)
        axes[1].plot(x, sub.latency_ms, marker=st["marker"], color=st["color"], label=index)
    axes[0].set_ylabel("recall@10")
    axes[0].set_title("Recall by difficulty bin")
    axes[1].set_ylabel("mean latency per query (ms, log)")
    axes[1].set_yscale("log")
    axes[1].set_yticks([0.5, 1, 2, 5, 10])
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[1].yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    axes[1].set_title("Latency by difficulty bin")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlabel("relative-contrast quartile")
    axes[0].legend(loc="lower right")
    fig.suptitle("GloVe-100 test queries, each index at its recall-0.95 operating point",
                 fontsize=9, color=plots.INK_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--build-threads", type=int, default=os.cpu_count())
    ap.add_argument("--max-queries", type=int, default=None)
    a = ap.parse_args()
    if a.quick and a.max_queries is None:
        a.max_queries = 1000
    plots.setup()

    df = load(a.quick)
    p2 = df[df.part.astype(str) == "2"]
    ds = data.load(DATASET, 100_000 if a.quick else None)
    gt_ids, gt_dist = data.ground_truth(ds)
    q = ds.test[:a.max_queries]
    gt = gt_ids[data.N_TUNE:][:len(q)]
    rc = relative_contrast(ds.base, q, gt_dist[data.N_TUNE:][:len(q), 9])
    # equal-size bins by RC rank: bin 0 = lowest contrast = hardest
    bins = np.empty(len(q), dtype=int)
    bins[np.argsort(rc, kind="stable")] = np.arange(len(q)) * N_BINS // len(q)
    per_query = pd.DataFrame({"query": np.arange(len(q)) + data.N_TUNE, "rc": rc, "bin": bins})

    table, rows = [], []
    for index in INDEXES:
        op, reached = operating_point(p2, DATASET, index, TARGET)
        if op is None:
            continue
        params = parse_params(op.build_params)
        value = int(op.search_param.split("=")[1])
        idx, build_s = build(index, params, ds.base, a.build_threads)
        set_search(idx, index, value)
        ids, lat = per_query_run(idx, q)
        rec = recall_at_k(ids, gt, q, ds.base)
        per_query[f"recall_{index}"] = rec
        per_query[f"latency_ms_{index}"] = lat * 1e3
        note = "" if reached else "; 0.95 not reachable, most accurate config"
        print(f"{index} {op.build_params} {op.search_param}: overall recall {rec.mean():.4f}{note}")
        for b in range(N_BINS):
            m = bins == b
            table.append({"index": index, "bin": b, "rc_min": rc[m].min(), "rc_max": rc[m].max(),
                          "recall": rec[m].mean(), "latency_ms": lat[m].mean() * 1e3,
                          "n_queries": int(m.sum())})
            rows.append({"part": "3", "dataset": DATASET, "n": ds.n, "index": index,
                         "build_params": op.build_params, "search_param": op.search_param,
                         "query_split": f"test-rc-bin{b + 1}", "n_queries": int(m.sum()),
                         "recall": rec[m].mean(), "qps": 1 / lat[m].mean(),
                         "median_s": "", "repeat_s": "", "build_s": build_s,
                         "index_bytes": "", "bytes_per_vec": "",
                         "notes": f"RC {rc[m].min():.3f}-{rc[m].max():.3f}; bin 1 = hardest; "
                                  f"single pass, per-query timing{note}"})
        del idx

    results.append(rows)
    table = pd.DataFrame(table)
    out_tables = os.path.join(results.RESULTS_DIR, "tables")
    os.makedirs(out_tables, exist_ok=True)
    per_query.to_csv(os.path.join(results.RESULTS_DIR, "part3_per_query.csv"), index=False)
    wide = table.pivot(index="bin", columns="index", values="recall")[
        [i for i in INDEXES if i in set(table["index"])]]
    wide.insert(0, "RC range", [f"{r.rc_min:.3f}-{r.rc_max:.3f}"
                                for r in table[table["index"] == table["index"].iloc[0]].itertuples()])
    wide.index = [f"Q{b + 1}" + (" (hardest)" if b == 0 else " (easiest)" if b == N_BINS - 1 else "")
                  for b in wide.index]
    wide.to_csv(os.path.join(out_tables, "part3_recall_by_bin.csv"))
    with open(os.path.join(out_tables, "part3_recall_by_bin.md"), "w") as f:
        f.write(wide.to_markdown(floatfmt=".3f"))
    print(wide.to_markdown(floatfmt=".3f"))
    corr = {i: per_query[["rc", f"recall_{i}"]].corr(method="spearman").iloc[0, 1]
            for i in INDEXES if f"recall_{i}" in per_query}
    print("Spearman(RC, per-query recall):", {k: round(float(v), 3) for k, v in corr.items()})
    plot(table, os.path.join(results.FIG_DIR, "part3_difficulty.png"))


if __name__ == "__main__":
    main()
