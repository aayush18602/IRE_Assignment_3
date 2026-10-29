"""Part 4: IVF-Flat and HNSW under distribution drift (synthetic data).

For each shift delta (0 = control), each index is measured in two variants:
  inserted - built on the old vectors (IVF trained on old only), new vectors added after
  rebuilt  - built from scratch on old + new (IVF retrained; HNSW in shuffled order)
Queries come from the shifted distribution. Measured: recall at the search
parameter that gives 0.95 on the control, QPS at recall 0.95 (chosen on tune
queries, reported on test queries), and the work per query (distance
computations), plus the IVF list imbalance.
"""
import argparse
import json
import os

import faiss
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

import gen_synthetic
import plots
import results
from indexes import fmt_params, set_search
from part2_bench import EF, NPROBE, STOP_RECALL
from recall import recall_at_k
from timing import time_single_queries

DELTAS = [0.0, 0.25, 0.5, 1.0]
TARGET = 0.95
INDEXES = ["IVF-Flat", "HNSW"]
VARIANTS = ["inserted", "rebuilt"]


def build(kind, variant, dd, params, threads):
    faiss.omp_set_num_threads(threads)
    d = dd.old.shape[1]
    if kind == "IVF-Flat":
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, params["nlist"])
        index.train(dd.old if variant == "inserted" else dd.all)
        index.add(dd.old)
        index.add(dd.new)
    else:
        index = faiss.IndexHNSWFlat(d, params["M"])
        index.hnsw.efConstruction = params["efC"]
        if variant == "inserted":
            index.add(dd.old)
            index.add(dd.new)
        else:  # same vectors and ids, one shuffled build
            perm = np.random.default_rng(1).permutation(len(dd.all))
            index = faiss.IndexIDMap(index)
            index.add_with_ids(dd.all[perm], perm.astype(np.int64))
    faiss.omp_set_num_threads(1)
    return index


def inner(index):
    return faiss.downcast_index(index.index) if isinstance(index, faiss.IndexIDMap) else index


def work_per_query(index, kind, queries, value):
    """Distance computations per query: coarse quantiser + scanned list entries for
    IVF, distance evaluations counted by FAISS for HNSW."""
    idx = inner(index)
    if kind == "IVF-Flat":
        _, lists = idx.quantizer.search(queries, value)
        sizes = np.array([idx.invlists.list_size(i) for i in range(idx.nlist)])
        return float(idx.nlist + sizes[lists].sum(1).mean())
    faiss.cvar.hnsw_stats.reset()
    index.search(queries, 10)
    return faiss.cvar.hnsw_stats.ndis / len(queries)


def sweep(index, kind, dd, gt_tune, gt_test, base, max_value, repeats):
    """Tune/test sweep of the search parameter, as in Part 2."""
    values = EF if kind == "HNSW" else [p for p in NPROBE if p <= max_value]
    pts, recalls = [], []
    for v in values:
        set_search(inner(index), kind, v)
        pt = {"value": v}
        for split, q, gt, reps in (("tune", dd.tune, gt_tune, 1), ("test", dd.test, gt_test, repeats)):
            med, times, ids = time_single_queries(index, q, repeats=reps)
            pt[split] = {"recall": float(recall_at_k(ids, gt, q, base).mean()),
                         "qps": len(q) / med, "median_s": med, "times": times}
        pts.append(pt)
        recalls.append(pt["tune"]["recall"])
        if recalls[-1] >= STOP_RECALL or (len(recalls) >= 4 and recalls[-1] - recalls[-4] < 0.002):
            break
    return pts


def pick(pts, target):
    ok = [p for p in pts if p["tune"]["recall"] >= target]
    return max(ok, key=lambda p: p["tune"]["qps"]) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--build-threads", type=int, default=os.cpu_count())
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    sizes = dict(n_old=50_000, n_new=25_000, n_query=500) if a.quick else {}
    grid = {"IVF-Flat": {"nlist": 256 if a.quick else 1024}, "HNSW": {"M": 16, "efC": 200}}
    pname = {"IVF-Flat": "nprobe", "HNSW": "efSearch"}
    plots.setup()

    fixed = {}  # search parameter that reaches TARGET on the control (inserted variant)
    summary, rows = [], []
    for delta in DELTAS:
        dd = gen_synthetic.make(delta, **sizes)
        base = dd.all
        flat = faiss.IndexFlatL2(base.shape[1])
        flat.add(base)
        faiss.omp_set_num_threads(a.build_threads)
        _, gt_tune = flat.search(dd.tune, 10)
        _, gt_test = flat.search(dd.test, 10)
        faiss.omp_set_num_threads(1)
        for kind in INDEXES:
            params = grid[kind]
            for variant in VARIANTS:
                index = build(kind, variant, dd, params, a.build_threads)
                pts = sweep(index, kind, dd, gt_tune, gt_test, base, params.get("nlist"), a.repeats)
                if delta == 0 and variant == "inserted":
                    p = pick(pts, TARGET)
                    fixed[kind] = p["value"] if p else pts[-1]["value"]
                # recall/work at the control's parameter (measured even if the sweep stopped early)
                v_fix = fixed[kind]
                set_search(inner(index), kind, v_fix)
                med, _, ids = time_single_queries(index, dd.test, repeats=1)
                rec_fix = float(recall_at_k(ids, gt_test, dd.test, base).mean())
                work_fix = work_per_query(index, kind, dd.test, v_fix)
                op = pick(pts, TARGET)
                imb = float(inner(index).invlists.imbalance_factor()) if kind == "IVF-Flat" else np.nan
                s = {"delta": delta, "index": kind, "variant": variant,
                     "fixed_param": f"{pname[kind]}={v_fix}", "recall_fixed": rec_fix,
                     "qps_fixed": len(dd.test) / med, "work_fixed": work_fix,
                     "qps_095": op["test"]["qps"] if op else np.nan,
                     "recall_095": op["test"]["recall"] if op else np.nan,
                     "param_095": f"{pname[kind]}={op['value']}" if op else "not reached",
                     "work_095": work_per_query(index, kind, dd.test, op["value"]) if op else np.nan,
                     "imbalance": imb}
                summary.append(s)
                print(f"delta={delta:<4} {kind:8s} {variant:8s} recall@fixed {rec_fix:.3f} "
                      f"work {work_fix:8.0f}  qps@0.95 {s['qps_095']:8.0f} ({s['param_095']})  "
                      f"imbalance {imb:.2f}", flush=True)

                common = {"part": "4", "dataset": f"drift-delta{delta}", "n": len(base),
                          "index": kind, "build_params": fmt_params({**params, "variant": variant}),
                          "build_s": "", "index_bytes": "", "bytes_per_vec": ""}
                for p in pts:
                    for split in ("tune", "test"):
                        r = p[split]
                        rows.append({**common, "search_param": f"{pname[kind]}={p['value']}",
                                     "query_split": split, "n_queries": len(dd.test),
                                     "recall": r["recall"], "qps": r["qps"], "median_s": r["median_s"],
                                     "repeat_s": json.dumps(r["times"]), "notes": "sweep"})
                rows.append({**common, "search_param": s["fixed_param"], "query_split": "test",
                             "n_queries": len(dd.test), "recall": rec_fix, "qps": s["qps_fixed"],
                             "median_s": med, "repeat_s": "",
                             "notes": f"control parameter; work/query {work_fix:.0f}; "
                                      f"imbalance {imb:.3f}"})
                del index

    results.append(rows)
    summary = pd.DataFrame(summary)
    out_tables = os.path.join(results.RESULTS_DIR, "tables")
    os.makedirs(out_tables, exist_ok=True)
    summary.to_csv(os.path.join(out_tables, "part4_drift.csv"), index=False)
    with open(os.path.join(out_tables, "part4_drift.md"), "w") as f:
        f.write(summary.to_markdown(index=False, floatfmt=".3f"))
    plot(summary, os.path.join(results.FIG_DIR, "part4_drift.png"))


def plot(summary, path):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    x = np.arange(len(DELTAS))
    panels = [("qps_095", "QPS at recall 0.95 (test, log)", INDEXES),
              ("recall_fixed", "recall@10 at the control's parameter", INDEXES),
              ("imbalance", "IVF-Flat list imbalance factor", ["IVF-Flat"])]
    for ax, (col, title, kinds) in zip(axes, panels):
        for kind in kinds:
            for variant in VARIANTS:
                sub = summary[(summary["index"] == kind) & (summary.variant == variant)]
                st = plots.INDEX_STYLE[kind]
                ax.plot(x, sub[col], marker=st["marker"], color=st["color"],
                        ls="-" if variant == "inserted" else (0, (4, 2)),
                        mfc=st["color"] if variant == "inserted" else "white",
                        label=f"{kind}, {variant}")
        ax.set_xticks(x, ["0\n(control)" if d == 0 else f"{d:g}" for d in DELTAS])
        ax.set_xlabel("shift δ (× RMS radius of old data)")
        ax.set_title(title)
    axes[0].set_yscale("log")
    axes[0].set_yticks([5000, 10000, 20000, 40000])
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    axes[0].yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Drift: queries and 100K new vectors shifted by δ; index built on the 200K old "
                 "vectors then updated (inserted) or rebuilt on all 300K",
                 fontsize=9, color=plots.INK_2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
