import itertools
import math

import numpy as np

from frontierlab.labkit import load_target
from frontierlab.rlscale import curves, passk

lab = load_target(__file__)


def test_sigmoid_curve_by_hand():
    # at C = C_mid exactly half of the gain is reached
    assert math.isclose(float(lab.sigmoid_curve(np.array([8e3]), 0.645, 1.7, 8e3, 0.3)[0]), 0.4725)
    C = np.geomspace(1e2, 1e6, 9)
    assert np.allclose(lab.sigmoid_curve(C, 0.6, 1.2, 5e3, 0.2), curves.sigmoid_curve(C, 0.6, 1.2, 5e3, 0.2))


def test_best_asymptote_is_least_squares():
    rng = np.random.default_rng(0)
    C = np.geomspace(1.5e3, 5e4, 20)
    y = curves.sigmoid_curve(C, 0.6, 1.5, 6e3, 0.3) + rng.normal(0, 0.01, 20)
    A, sse = lab.best_asymptote(C, y, 1.5, 6e3, 0.3)
    grid = np.linspace(0.5, 0.7, 2001)
    sses = [float(((y - curves.sigmoid_curve(C, a, 1.5, 6e3, 0.3)) ** 2).sum()) for a in grid]
    assert abs(A - grid[int(np.argmin(sses))]) < 2e-4 and sse <= min(sses) + 1e-12
    A2, _ = lab.best_asymptote(C, y * 0 + 0.99, 0.5, 1e9, 0.3)    # data far above what the curve can reach
    assert A2 == 1.0


def test_pass_at_k_against_enumeration_and_reference():
    n = 6
    for c in range(n + 1):
        for k in (1, 2, 3, 6):
            subsets = list(itertools.combinations(range(n), k))
            brute = np.mean([any(i < c for i in s) for s in subsets])
            assert math.isclose(lab.pass_at_k_curve([c], n, [k])[0], brute, abs_tol=1e-12)
    counts = np.random.default_rng(1).integers(0, 257, 50)
    ks = [1, 4, 32, 256]
    assert np.allclose(lab.pass_at_k_curve(counts, 256, ks), passk.curve(counts, 256, ks))


def test_crossover():
    ks = [1, 2, 4, 8]
    assert lab.crossover([0.5, 0.6, 0.65, 0.7], [0.3, 0.5, 0.66, 0.8], ks) == 4
    assert lab.crossover([0.3, 0.6], [0.4, 0.5], ks[:2]) is None
    assert lab.crossover([0.5, 0.6], [0.3, 0.5], ks[:2]) is None
