import pytest

from frontierlab.dist import schedule as S
from frontierlab.labkit import load_target

lab = load_target(__file__)


@pytest.mark.parametrize("p,m", [(2, 4), (4, 8), (8, 16), (8, 32)])
def test_bubble_ratio_matches_simulator(p, m):
    t = S.Times(1, 1, 1)
    assert lab.bubble_ratio("gpipe", p, m) == pytest.approx(S.simulate(S.gpipe(p, m), t)["bubble_ratio"])
    assert lab.bubble_ratio("1f1b", p, m) == pytest.approx(S.simulate(S.one_f_one_b(p, m), t)["bubble_ratio"])
    if m % p == 0:
        for v in (2, 4):
            sim = S.simulate(S.interleaved(p, v, m), t)["bubble_ratio"]
            assert lab.bubble_ratio("interleaved", p, m, v) == pytest.approx(sim)


def test_one_f_one_b_order():
    for p, m in ((4, 6), (4, 2), (8, 16), (1, 3)):
        for s in range(p):
            assert lab.one_f_one_b_order(p, s, m) == S.one_f_one_b_order(p, s, m)


def test_peak_in_flight():
    p, m = 4, 8
    assert [lab.peak_in_flight(S.one_f_one_b_order(p, s, m)) for s in range(p)] == [4, 3, 2, 1]
    gpipe = [("F", i) for i in range(m)] + [("B", i) for i in range(m)]
    assert lab.peak_in_flight(gpipe) == m


def test_measured_bubble():
    assert lab.measured_bubble([0.6, 0.7, 0.8], [1.0, 1.0, 1.0]) == pytest.approx(0.3)
    assert lab.measured_bubble([1.0], [1.0]) == 0.0
