import numpy as np
import pytest

from frontierlab.labkit import load_target
from frontierlab.perf import dist as ref

lab = load_target(__file__)


def test_bus_bandwidth_matches_ring_formula():
    for world in (2, 4, 8):
        for op in ("all_reduce", "reduce_scatter", "all_gather"):
            assert lab.bus_bandwidth(2**20, 0.01, world, op) == pytest.approx(ref.bus_bandwidth(2**20, 0.01, world, op))
    assert lab.bus_bandwidth(100, 1.0, 2) == pytest.approx(100)          # 2·(1)/2 = 1


def test_exposed_comm():
    rng = np.random.default_rng(0)
    nosync = 0.100 + rng.normal(0, 0.002, 30)
    sync = 0.130 + rng.normal(0, 0.002, 30)
    r = lab.exposed_comm(sync, nosync)
    assert r["exposed_s"] == pytest.approx(np.median(sync) - np.median(nosync))
    assert r["fraction"] == pytest.approx(r["exposed_s"] / np.median(sync))
    lo, hi = r["ci"]
    assert lo < r["exposed_s"] < hi and 0.02 < lo and hi < 0.04
    same = lab.exposed_comm(nosync, nosync)
    assert same["ci"][0] <= 0 <= same["ci"][1]


def test_overlap_fraction():
    assert lab.overlap_fraction(0.04, 0.01) == pytest.approx(0.75)
    assert lab.overlap_fraction(0.04, 0.06) == 0.0                         # more exposed than comm: noise
    assert lab.overlap_fraction(0.04, -0.01) == 1.0
    assert lab.overlap_fraction(0.0, 0.0) == 1.0
