"""Shared plot style: one palette and marker per index, used by every figure."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INDEX_STYLE = {
    "HNSW": {"color": "#2a78d6", "marker": "o"},
    "IVF-Flat": {"color": "#eb6834", "marker": "s"},
    "IVF-PQ": {"color": "#1baf7a", "marker": "^"},
    "Flat": {"color": "#52514e", "marker": "*"},
}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def setup():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "lines.linewidth": 1.6, "lines.markersize": 4.5,
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def pareto(points):
    """Upper-right frontier of (recall, qps, row) points: no other point is both
    more accurate and faster. Returned sorted by recall."""
    best, front = -1.0, []
    for p in sorted(points, key=lambda p: (-p[0], -p[1])):
        if p[1] > best:
            front.append(p)
            best = p[1]
    return front[::-1]
