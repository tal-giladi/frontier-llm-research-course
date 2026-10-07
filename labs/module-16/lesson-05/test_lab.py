import math

import numpy as np
from scipy.stats import binomtest

from frontierlab.labkit import load_target

lab = load_target(__file__)


def test_wilson_matches_scipy():
    for k, n in ((45, 369), (0, 30), (30, 30), (7, 12)):
        lo, hi = lab.wilson_interval(k, n)
        ci = binomtest(k, n).proportion_ci(method="wilson")
        assert math.isclose(lo, ci.low, abs_tol=1e-6) and math.isclose(hi, ci.high, abs_tol=1e-6)
    assert all(math.isnan(x) for x in lab.wilson_interval(0, 0))


def test_task_reward():
    assert lab.task_reward(True, "DONE", True) == 1.0
    assert lab.task_reward(True, "DONE", False) == 0.0
    assert lab.task_reward(True, "FAIL", True) == 0.0
    assert lab.task_reward(False, "FAIL", False) == 1.0
    assert lab.task_reward(False, "DONE", True) == 0.0


def test_cluster_interval_wider_than_iid_when_clusters_differ():
    succ = [1.0] * 20 + [0.0] * 20
    clus = ["a"] * 20 + ["b"] * 20
    m, lo, hi = lab.cluster_interval(succ, clus, n_boot=500)
    assert m == 0.5 and lo == 0.0 and hi == 1.0
    m2, lo2, hi2 = lab.cluster_interval([1.0, 0.0] * 20, clus, n_boot=500)
    assert m2 == 0.5 and lo2 == 0.5 and hi2 == 0.5


def test_success_at_budget():
    s = [3, 8, 14, 20, 49, float("inf")]
    assert lab.success_at_budget(s, 15) == 0.5
    assert math.isclose(lab.success_at_budget(s, 50), 5 / 6)
