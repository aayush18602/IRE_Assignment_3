import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from recall import recall_at_k, recall_from_distances, true_distances  # noqa: E402


def line_data():
    # points on a line at 0,1,...,19; query at 0 -> neighbour order is 0,1,2,...
    base = np.arange(20, dtype=np.float32)[:, None]
    q = np.zeros((1, 1), dtype=np.float32)
    gt = np.arange(10)[None, :]
    return base, q, gt


def test_perfect():
    base, q, gt = line_data()
    assert recall_at_k(gt, gt, q, base, k=10)[0] == 1.0


def test_order_does_not_matter():
    base, q, gt = line_data()
    assert recall_at_k(gt[:, ::-1], gt, q, base, k=10)[0] == 1.0


def test_disjoint():
    base, q, gt = line_data()
    found = np.arange(10, 20)[None, :]
    assert recall_at_k(found, gt, q, base, k=10)[0] == 0.0


def test_partial():
    base, q, gt = line_data()
    found = np.array([[0, 1, 2, 3, 4, 5, 6, 15, 16, 17]])
    assert recall_at_k(found, gt, q, base, k=10)[0] == 0.7


def test_tie_at_kth_is_not_a_mistake():
    # vectors 9 and 10 are both at distance 9 from the query (one on each side)
    base = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -9, 50], dtype=np.float32)[:, None]
    q = np.zeros((1, 1), dtype=np.float32)
    gt = np.arange(10)[None, :]                 # exact search picked id 9
    found = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 10]])  # index returned the tied id 10
    assert recall_at_k(found, gt, q, base, k=10)[0] == 1.0
    worse = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 11]])  # id 11 is genuinely farther
    assert recall_at_k(worse, gt, q, base, k=10)[0] == 0.9


def test_duplicates_count_once():
    base, q, gt = line_data()
    found = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert recall_at_k(found, gt, q, base, k=10)[0] == 0.1


def test_missing_results():
    base, q, gt = line_data()
    found = np.array([[0, 1, 2, -1, -1, -1, -1, -1, -1, -1]])
    assert recall_at_k(found, gt, q, base, k=10)[0] == 0.3


def test_extra_columns_ignored():
    base, q, gt = line_data()
    found = np.concatenate([gt, np.arange(10, 20)[None, :]], axis=1)
    assert recall_at_k(found, gt, q, base, k=10)[0] == 1.0


def test_float_noise_tolerated():
    ids = np.arange(10)[None, :]
    found_dist = np.full((1, 10), 1.0 + 1e-12)
    assert recall_from_distances(ids, found_dist, np.array([1.0]), k=10)[0] == 1.0


def test_true_distances_squared_l2():
    base = np.array([[0, 0], [3, 4]], dtype=np.float32)
    q = np.zeros((1, 2), dtype=np.float32)
    d = true_distances(q, base, np.array([[1, 0, -1]]))
    assert d[0, 0] == 25 and d[0, 1] == 0 and np.isinf(d[0, 2])


def test_many_queries_random():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((500, 8)).astype(np.float32)
    q = rng.standard_normal((20, 8)).astype(np.float32)
    d = ((q[:, None, :] - base[None]) ** 2).sum(-1)
    gt = np.argsort(d, axis=1)[:, :10]
    r = recall_at_k(gt, gt, q, base, k=10)
    assert np.all(r == 1.0)
    r_half = recall_at_k(np.argsort(d, axis=1)[:, 5:15], gt, q, base, k=10)
    assert np.allclose(r_half, 0.5)
