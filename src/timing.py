"""Timing helpers: single-threaded, one query at a time, median of repeats."""
import time

import faiss
import numpy as np


def single_thread():
    faiss.omp_set_num_threads(1)


def time_single_queries(index, queries, k=10, repeats=3):
    """Search queries one by one; return (median seconds, all seconds, ids of last run)."""
    single_thread()
    nq = queries.shape[0]
    ids = np.empty((nq, k), dtype=np.int64)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for i in range(nq):
            _, ids[i] = index.search(queries[i:i + 1], k)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), times, ids


def time_batch(index, queries, k=10, repeats=3, threads=1):
    """Search all queries in one call; return (median seconds, all seconds, ids)."""
    faiss.omp_set_num_threads(threads)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _, ids = index.search(queries, k)
        times.append(time.perf_counter() - t0)
    single_thread()
    return float(np.median(times)), times, ids
