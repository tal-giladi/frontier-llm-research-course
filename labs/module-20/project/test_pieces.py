"""Checks for the capstone analysis pieces. ``pytest labs/module-20/project`` tests ``pieces.py``;
``CAPSTONE=buggy pytest labs/module-20/project`` tests the colleague's ``buggy_capstone.py``."""

import os
from pathlib import Path

import numpy as np

from frontierlab.labkit import load_path

P = load_path(str(Path(__file__).resolve().parent / ("buggy_capstone.py" if os.environ.get("CAPSTONE") == "buggy"
                                                      else "pieces.py")))


def test_pairs_by_seed_not_by_insertion_order():
    a = {0: [1.0, 1.0], 1: [2.0, 2.0], 2: [3.0, 3.0]}
    b = {2: [3.1, 3.1], 0: [1.1, 1.1], 1: [2.1, 2.1]}          # finished in a different order
    xa, xb, seeds = P.pair(a, b)
    assert seeds == [0, 1, 2]
    assert np.allclose(xa - xb, -0.1)


def test_interval_resamples_seeds():
    b = np.zeros((3, 300))
    a = b + np.array([0.0, 0.04, 0.08])[:, None]               # the difference varies only between seeds
    m, lo, hi = P.interval(a, b)
    assert abs(m - 0.04) < 1e-12
    assert hi - lo > 0.04, "an items-only interval treats 900 windows as independent and ignores seed variation"


def test_decide_checks_equivalence_first():
    assert P.decide(-0.012, -0.015, -0.009, 0.02) == "equivalent"   # real, but too small to matter
    assert P.decide(0.03, 0.01, 0.05, 0.02) == "a_higher"
    assert P.decide(0.0, -0.03, 0.03, 0.02) == "inconclusive"
