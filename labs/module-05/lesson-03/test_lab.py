import math

import pytest

from frontierlab.labkit import load_target
from frontierlab.perf.roofline import HARDWARE

lab = load_target(__file__)


def test_fit_exponent():
    L = [512, 1024, 2048, 4096]
    assert abs(lab.fit_exponent(L, [3e-9 * x * x for x in L]) - 2.0) < 1e-9
    assert abs(lab.fit_exponent(L, [5e-6 * x for x in L]) - 1.0) < 1e-9
    assert abs(lab.fit_exponent(L, [0.7] * 4)) < 1e-9
    noisy = [1e-6 * x * x * f for x, f in zip(L, (1.1, 0.9, 1.05, 0.97))]
    assert 1.9 < lab.fit_exponent(L, noisy) < 2.1


def test_crossover_interpolates_in_log_log():
    L = [1000, 2000, 4000, 8000]
    dense = [x * x * 1e-6 for x in L]                       # 1, 4, 16, 64
    new = [x * 4e-3 for x in L]                             # 4, 8, 16, 32 -> equal at 4000
    assert abs(lab.crossover(L, dense, new) - 4000) < 1e-6
    new2 = [x * 3e-3 for x in L]                            # 3, 6, 12, 24: equal at 3000 (between 2000 and 4000)
    assert abs(lab.crossover(L, dense, new2) - 3000) / 3000 < 1e-9
    assert lab.crossover(L, dense, [0.5] * 4) == 1000       # faster already at the first point
    assert lab.crossover(L, dense, [100.0] * 4) is None     # never faster


def test_project_time():
    hw = HARDWARE["H100-SXM"]
    t = lab.project_time(989e12, 1.0, hw.peak_flops, hw.mem_bw, 0.5, 0.7)
    assert abs(t - 2.0) < 1e-12                             # compute-bound: 1 s at peak, 2 s at 50%
    t2 = lab.project_time(1.0, 3.35e12, hw.peak_flops, hw.mem_bw, 0.5, 0.5)
    assert abs(t2 - 2.0) < 1e-12                            # memory-bound


@pytest.mark.parametrize("sp,dl,expect", [
    ((1.4, 1.9), (-0.01, 0.01), "adopt"),
    ((1.4, 1.9), (0.01, 0.05), "inconclusive"),           # loss interval straddles the margin
    ((0.8, 1.1), (-0.01, 0.01), "keep dense"),            # cannot reach the required speed-up
    ((1.4, 1.9), (0.04, 0.06), "keep dense"),             # clearly worse than the margin
    ((1.1, 1.6), (-0.01, 0.01), "inconclusive"),
])
def test_decide(sp, dl, expect):
    assert lab.decide(sp, dl, min_speedup=1.25, margin=0.03) == expect


def test_crossover_matches_flop_model_for_pure_scaling():
    """With times exactly proportional to the FLOP model, the measured crossover is the analytic one."""
    a, c, k = 4 * 6 * 64, 4 * 66, 256
    L = [256 * 2 ** i for i in range(8)]
    dense = [a * x * x / 2 for x in L]
    dsa = [c * x * x / 2 + a * (k * x - k * k / 2) for x in L]
    got = lab.crossover(L, dense, dsa)
    A, B, C = (a - c) / 2, -a * k, a * k * k / 2
    exact = (-B + math.sqrt(B * B - 4 * A * C)) / (2 * A)
    assert abs(got - exact) / exact < 0.10                  # log-log interpolation between powers of two: ~7% here
