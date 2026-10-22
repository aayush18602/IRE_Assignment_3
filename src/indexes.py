"""Build / configure / measure the FAISS indexes used in the benchmark."""
import time

import faiss
import numpy as np

SEARCH_PARAM = {"HNSW": "efSearch", "IVF-Flat": "nprobe", "IVF-PQ": "nprobe"}


def fmt_params(params):
    return ";".join(f"{k}={v}" for k, v in params.items())


def train_sample(base, nlist, seed=0):
    """Random training subset for k-means / PQ: max(100K, 64*nlist) vectors, at most all."""
    n_train = min(base.shape[0], max(100_000, 64 * nlist))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(base.shape[0], n_train, replace=False))
    return base[idx]


def build(kind, params, base, threads):
    """Build an index; returns (index, wall-clock build seconds incl. training)."""
    faiss.omp_set_num_threads(threads)
    d = base.shape[1]
    t0 = time.perf_counter()
    if kind == "HNSW":
        index = faiss.IndexHNSWFlat(d, params["M"])
        index.hnsw.efConstruction = params["efC"]
    elif kind == "IVF-Flat":
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, params["nlist"])
        index.train(train_sample(base, params["nlist"]))
    elif kind == "IVF-PQ":
        index = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, params["nlist"], params["m"], params["nbits"])
        index.train(train_sample(base, params["nlist"]))
    else:
        raise ValueError(kind)
    index.add(base)
    build_s = time.perf_counter() - t0
    faiss.omp_set_num_threads(1)
    return index, build_s


def set_search(index, kind, value):
    if kind == "HNSW":
        index.hnsw.efSearch = value
    else:
        index.nprobe = value


def index_bytes(index):
    """Size of the serialised index (vectors/codes + graph or lists + quantiser)."""
    return int(faiss.serialize_index(index).nbytes)
