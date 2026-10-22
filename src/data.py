"""Dataset loading, query split and exact ground truth."""
import os
from dataclasses import dataclass

import faiss
import h5py
import numpy as np

FILES = {
    "sift": "sift-128-euclidean.hdf5",
    "glove": "glove-100-angular.hdf5",
}
N_TUNE = 5000


def data_dir():
    default = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    return os.path.abspath(os.environ.get("A3_DATA_DIR", default))


def normalise(x):
    """Unit-normalise rows in place; all-zero rows are left as zero."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x /= norms
    return x


@dataclass
class Dataset:
    name: str
    base: np.ndarray        # (n, d) float32
    queries: np.ndarray     # (10000, d) float32
    stored_gt: np.ndarray   # (10000, 100) neighbours from the file (full data only)

    @property
    def n(self):
        return self.base.shape[0]

    @property
    def d(self):
        return self.base.shape[1]

    @property
    def tune(self):
        return self.queries[:N_TUNE]

    @property
    def test(self):
        return self.queries[N_TUNE:]


def load(name, n_base=None):
    """Load a dataset; optionally keep only the first n_base vectors."""
    path = os.path.join(data_dir(), FILES[name])
    with h5py.File(path, "r") as f:
        base = np.ascontiguousarray(f["train"][:n_base], dtype=np.float32)
        queries = np.ascontiguousarray(f["test"][:], dtype=np.float32)
        stored = f["neighbors"][:] if n_base is None else None
    if name == "glove":
        normalise(base)
        normalise(queries)
    return Dataset(name, base, queries, stored)


def ground_truth(ds, k=100):
    """Exact k-NN (squared L2) of all queries with a FAISS flat index, cached on disk."""
    path = os.path.join(data_dir(), f"gt_{ds.name}_{ds.n}_k{k}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["ids"], z["dist"]
    index = faiss.IndexFlatL2(ds.d)
    index.add(ds.base)
    dist, ids = index.search(ds.queries, k)
    np.savez(path, ids=ids, dist=dist)
    return ids, dist
