"""Seeded generator for the Part 4 drift experiment.

Old vectors, new vectors and queries come from one Gaussian mixture. New vectors
and queries are then shifted by delta * R along a fixed random unit direction,
where R is the RMS distance of the old vectors from their mean. Each part has its
own random stream, so for every delta the old data and the unshifted new samples
are identical: only the shift changes.
"""
from dataclasses import dataclass

import numpy as np

D = 64
K = 50              # mixture components
CENTRE_STD = 1.0    # per-dimension std of the component centres
NOISE_STD = 0.5     # per-dimension std inside a component


@dataclass
class DriftData:
    delta: float
    old: np.ndarray       # (n_old, D) vectors the index is built / trained on
    new: np.ndarray       # (n_new, D) shifted vectors inserted afterwards
    tune: np.ndarray      # shifted queries used to choose parameters
    test: np.ndarray      # shifted queries used for reported numbers
    radius: float         # RMS radius of the old data

    @property
    def all(self):
        return np.vstack([self.old, self.new])


def _mixture(rng, centres, n):
    comp = rng.integers(0, len(centres), n)
    return centres[comp] + NOISE_STD * rng.standard_normal((n, centres.shape[1]))


def make(delta, n_old=200_000, n_new=100_000, n_query=2_000, seed=0):
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    r_centre, r_dir, r_old, r_new, r_query = streams
    centres = CENTRE_STD * r_centre.standard_normal((K, D))
    direction = r_dir.standard_normal(D)
    direction /= np.linalg.norm(direction)

    old = _mixture(r_old, centres, n_old)
    radius = float(np.sqrt(((old - old.mean(0)) ** 2).sum(1).mean()))
    shift = delta * radius * direction
    new = _mixture(r_new, centres, n_new) + shift
    queries = _mixture(r_query, centres, 2 * n_query) + shift
    f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)  # noqa: E731
    return DriftData(delta, f32(old), f32(new), f32(queries[:n_query]),
                     f32(queries[n_query:]), radius)
