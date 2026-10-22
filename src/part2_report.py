"""Part 2 figures and tables from results/results.csv.

Operating points are chosen on the tune queries (fastest config whose tune recall
reaches the target) and the numbers reported are that config's test-query run.
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

import plots
import results

INDEXES = ["HNSW", "IVF-Flat", "IVF-PQ"]
TARGETS = [0.90, 0.95]
KEY = ["part", "dataset", "n", "index", "build_params", "search_param", "query_split", "notes"]
TABLE_DIR = os.path.join(results.RESULTS_DIR, "tables")


def load(quick):
    df = pd.read_csv(results.CSV, keep_default_na=False)
    df = df.drop_duplicates(KEY, keep="last")
    n_full = df.groupby("dataset")["n"].transform("max")
    df = df[df["n"] == 100_000] if quick else df[df["n"] == n_full]
    return df


def exact_row(df, dataset):
    ex = df[(df.part.astype(str) == "1") & (df.dataset == dataset)
            & (df.notes == "single-query 1 thread")]
    return ex.iloc[-1] if len(ex) else None


def choose(p2, dataset, index, target):
    """Fastest tune point reaching target -> matching test row (or None)."""
    tune = p2[(p2.dataset == dataset) & (p2["index"] == index) & (p2.query_split == "tune")]
    ok = tune[tune.recall >= target]
    if ok.empty:
        return None
    pick = ok.loc[ok.qps.idxmax()]
    test = p2[(p2.dataset == dataset) & (p2["index"] == index) & (p2.query_split == "test")
              & (p2.build_params == pick.build_params) & (p2.search_param == pick.search_param)]
    return test.iloc[0]


def plot_dataset(p2, ex, dataset, path):
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    test = p2[(p2.dataset == dataset) & (p2.query_split == "test")]
    for index in INDEXES:
        sub = test[test["index"] == index]
        if sub.empty:
            continue
        st = plots.INDEX_STYLE[index]
        front = plots.pareto(list(zip(sub.recall, sub.qps, range(len(sub)))))
        ax.plot([p[0] for p in front], [p[1] for p in front], marker=st["marker"],
                color=st["color"], label=index)
    if ex is not None:
        st = plots.INDEX_STYLE["Flat"]
        ax.axhline(ex.qps, color=st["color"], lw=0.9, ls=(0, (4, 3)))
        ax.plot([1.0], [ex.qps], marker=st["marker"], ms=10, color=st["color"],
                ls="none", label="Exact (Flat)")
    for t in TARGETS:
        ax.axvline(t, color=plots.INK_2, lw=0.6, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("recall@10 (test queries)")
    ax.set_ylabel("queries per second (1 thread, log)")
    ax.set_title(f"{dataset.upper() if dataset == 'sift' else 'GloVe-100'}: "
                 f"best trade-off per index")
    ax.legend(loc="lower left")
    fig.savefig(path)
    plt.close(fig)


def fmt_cfg(r):
    return f"{r.build_params.replace(';', ', ')}, {r.search_param}"


def tables(p2, df, datasets):
    qps_rows, cost_rows = [], []
    for ds in datasets:
        ex = exact_row(df, ds)
        for index in INDEXES:
            sub = p2[(p2.dataset == ds) & (p2["index"] == index)]
            if sub.empty:
                continue
            row = {"dataset": ds, "index": index}
            for t in TARGETS:
                r = choose(p2, ds, index, t)
                if r is None:
                    best = sub[sub.query_split == "tune"].recall.max()
                    row[f"qps@{t:.2f}"] = f"not reached (max tune recall {best:.3f})"
                    row[f"config@{t:.2f}"] = ""
                else:
                    row[f"qps@{t:.2f}"] = f"{r.qps:.0f} (test recall {r.recall:.3f})"
                    row[f"config@{t:.2f}"] = fmt_cfg(r)
            qps_rows.append(row)

            r = choose(p2, ds, index, 0.95)
            note = ""
            if r is None:  # report the most accurate config instead
                tune = sub[sub.query_split == "tune"]
                pick = tune.loc[tune.recall.idxmax()]
                r = sub[(sub.query_split == "test") & (sub.build_params == pick.build_params)
                        & (sub.search_param == pick.search_param)].iloc[0]
                note = f"0.95 not reached; most accurate config (test recall {r.recall:.3f})"
            cost_rows.append({"dataset": ds, "index": index, "config": fmt_cfg(r),
                              "build_s": f"{float(r.build_s):.1f}",
                              "index_MB": f"{float(r.index_bytes) / 2**20:.0f}",
                              "bytes_per_vec": f"{float(r.bytes_per_vec):.0f}", "note": note})
        if ex is not None:
            qps_rows.append({"dataset": ds, "index": "Exact (Flat)",
                             "qps@0.90": f"{ex.qps:.0f} (recall 1.000)", "config@0.90": "",
                             "qps@0.95": f"{ex.qps:.0f} (recall 1.000)", "config@0.95": ""})
            cost_rows.append({"dataset": ds, "index": "Exact (Flat)", "config": "",
                              "build_s": "0.0",
                              "index_MB": f"{float(ex.index_bytes) / 2**20:.0f}",
                              "bytes_per_vec": f"{float(ex.bytes_per_vec):.0f}", "note": ""})
    return pd.DataFrame(qps_rows), pd.DataFrame(cost_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    plots.setup()
    os.makedirs(results.FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    df = load(a.quick)
    p2 = df[df.part.astype(str) == "2"].copy()
    datasets = [d for d in ("sift", "glove") if d in set(p2.dataset)]
    for ds in datasets:
        plot_dataset(p2, exact_row(df, ds), ds, os.path.join(results.FIG_DIR, f"part2_{ds}.png"))
    qps, cost = tables(p2, df, datasets)
    for name, t in (("part2_qps_at_recall", qps), ("part2_cost_at_095", cost)):
        t.to_csv(os.path.join(TABLE_DIR, f"{name}.csv"), index=False)
        with open(os.path.join(TABLE_DIR, f"{name}.md"), "w") as f:
            f.write(t.to_markdown(index=False))
        print(t.to_markdown(index=False), "\n")


if __name__ == "__main__":
    main()
