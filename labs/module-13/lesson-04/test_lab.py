import numpy as np

from frontierlab.labkit import load_target
from frontierlab.pipeline import thinking as TH

lab = load_target(__file__)


def test_trace_matches_reference():
    assert lab.trace(478, 365) == "a13b14c08"
    rng = np.random.default_rng(0)
    for a, b in rng.integers(0, 1000, size=(200, 2)):
        assert lab.trace(int(a), int(b)) == TH.AddProblem(int(a), int(b)).trace
    assert lab.trace(999, 999) == "a18b19c19" and lab.trace(0, 0) == "a00b00c00"


def test_truncated_target():
    assert lab.truncated_target(478, 365, 3) == "a13#843"
    assert lab.truncated_target(478, 365, 0) == "#843"
    assert lab.truncated_target(478, 365, 9) == TH.AddProblem(478, 365).target
    assert lab.truncated_target(999, 1, 20) == "a10b10c10#1000"


def test_routed_cost_matches_reference():
    rng = np.random.default_rng(1)
    pe = rng.random(50)
    ct, cn = rng.random(50) < 0.95, rng.random(50) < 0.7
    tt, tn = np.full(50, 14.0), np.full(50, 4.0)
    for th in (0.0, 0.3, 0.7, 1.01):
        mine = lab.routed_cost(pe, th, ct, cn, tt, tn)
        ref = TH.route_curve(pe, cn, ct, tn, tt, thresholds=[th])[0]
        assert abs(mine[0] - ref["accuracy"]) < 1e-12 and abs(mine[1] - ref["tokens"]) < 1e-12
        assert abs(mine[2] - ref["think_frac"]) < 1e-12


def test_pareto():
    pts = [(4.0, 0.7), (14.0, 0.99), (9.0, 0.9), (10.0, 0.85), (9.0, 0.9), (4.0, 0.6), (15.0, 0.99)]
    assert lab.pareto(pts) == [(4.0, 0.7), (9.0, 0.9), (14.0, 0.99)]
