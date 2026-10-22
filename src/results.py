"""Append benchmark rows to results/results.csv."""
import csv
import datetime
import os

# --quick runs point A3_RESULTS_DIR at results/quick so they never mix with full runs
RESULTS_DIR = os.path.abspath(os.environ.get(
    "A3_RESULTS_DIR", os.path.join(os.path.dirname(__file__), "..", "results")))
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
CSV = os.path.join(RESULTS_DIR, "results.csv")

FIELDS = [
    "timestamp", "part", "dataset", "n", "index", "build_params", "search_param",
    "query_split", "n_queries", "recall", "qps", "median_s", "repeat_s",
    "build_s", "index_bytes", "bytes_per_vec", "notes",
]


def append(rows):
    os.makedirs(FIG_DIR, exist_ok=True)
    new = not os.path.exists(CSV)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({"timestamp": now, **r})
