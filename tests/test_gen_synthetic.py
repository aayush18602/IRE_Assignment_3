import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import gen_synthetic  # noqa: E402

SMALL = dict(n_old=2000, n_new=1000, n_query=100)


def test_same_seed_same_data():
    a, b = gen_synthetic.make(0.5, **SMALL), gen_synthetic.make(0.5, **SMALL)
    for x, y in ((a.old, b.old), (a.new, b.new), (a.tune, b.tune), (a.test, b.test)):
        assert np.array_equal(x, y)


def test_only_the_shift_changes_with_delta():
    a, b = gen_synthetic.make(0.0, **SMALL), gen_synthetic.make(1.0, **SMALL)
    assert np.array_equal(a.old, b.old)
    shift = (b.new - a.new).astype(np.float64)
    # every new vector moves by the same vector, of length delta * radius
    assert np.allclose(shift, shift[0], atol=1e-4)
    assert np.isclose(np.linalg.norm(shift[0]), 1.0 * a.radius, rtol=1e-4)
    assert np.allclose(b.test - a.test, shift[0], atol=1e-4)


def test_shapes():
    dd = gen_synthetic.make(0.25, **SMALL)
    assert dd.old.shape == (2000, gen_synthetic.D) and dd.old.dtype == np.float32
    assert dd.new.shape == (1000, gen_synthetic.D)
    assert dd.tune.shape == dd.test.shape == (100, gen_synthetic.D)
    assert dd.all.shape == (3000, gen_synthetic.D)
