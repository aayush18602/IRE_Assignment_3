"""Tie-aware recall@k.

A returned id counts as a hit if its true distance to the query is no larger than
the distance of the k-th true nearest neighbour. Distances are recomputed in
float64 from the stored float32 vectors, so real ties compare equal and only a
tiny rounding tolerance is needed. This way, when several vectors tie at rank k,
returning any of them is not a mistake. Duplicate ids and -1 padding never count.
"""
import numpy as np

RTOL = 1e-9
ATOL = 1e-12


def true_distances(queries, base, ids):
    """Exact squared L2 distance from each query to each id, in float64; -1 -> inf."""
    ids = np.asarray(ids)
    valid = ids >= 0
    vecs = base[np.where(valid, ids, 0)].astype(np.float64)
    diff = vecs - queries[:, None, :].astype(np.float64)
    dist = np.einsum("qkd,qkd->qk", diff, diff)
    dist[~valid] = np.inf
    return dist


def recall_from_distances(found_ids, found_dist, kth_dist, k=10):
    """Per-query recall@k given true distances of the returned ids.

    found_ids:  (nq, >=k) ids returned by the index (-1 = no result)
    found_dist: (nq, >=k) true distances of those ids
    kth_dist:   (nq,) true distance of the k-th exact neighbour
    """
    found_ids = np.asarray(found_ids)[:, :k]
    found_dist = np.asarray(found_dist, dtype=np.float64)[:, :k]
    kth = np.asarray(kth_dist, dtype=np.float64)
    thresh = kth * (1 + RTOL) + ATOL
    hit = (found_dist <= thresh[:, None]) & (found_ids >= 0)
    # a repeated id only counts once
    order = np.argsort(found_ids, axis=1, kind="stable")
    sorted_ids = np.take_along_axis(found_ids, order, axis=1)
    dup_sorted = np.zeros_like(sorted_ids, dtype=bool)
    dup_sorted[:, 1:] = sorted_ids[:, 1:] == sorted_ids[:, :-1]
    dup = np.zeros_like(dup_sorted)
    np.put_along_axis(dup, order, dup_sorted, axis=1)
    hit &= ~dup
    return hit.sum(axis=1) / k


def recall_at_k(found_ids, gt_ids, queries, base, k=10):
    """Per-query recall@k of found_ids against exact neighbours gt_ids (sorted)."""
    found_dist = true_distances(queries, base, np.asarray(found_ids)[:, :k])
    gt_dist = true_distances(queries, base, np.asarray(gt_ids)[:, :k])
    kth = gt_dist.max(axis=1)
    return recall_from_distances(found_ids, found_dist, kth, k)
