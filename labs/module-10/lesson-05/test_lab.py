import numpy as np

from frontierlab.labkit import load_target

lab = load_target(__file__)


def test_gain_and_forgetting():
    g, f = lab.gain_and_forgetting([4.0, 4.1], [4.5, 4.4], [6.10, 6.20], [6.00, 6.05])
    assert np.allclose(g, [0.5, 0.3]) and np.allclose(f, [0.10, 0.15])


def test_target_tokens():
    assert lab.target_tokens(300, 16, 128, 0.5) == 307200
    assert lab.target_tokens(300, 16, 128, 0.0) == 0


def test_pareto():
    pts = [(0.5, 0.20), (0.4, 0.05), (0.3, 0.10), (0.0, 0.00), (0.5, 0.25)]
    # (0.3, 0.10) is dominated by (0.4, 0.05); (0.5, 0.25) by (0.5, 0.20)
    assert lab.pareto(pts) == [0, 1, 3]


def test_choose():
    arms = {"t100": {"gain": (0.4, 0.6), "forgetting": (0.15, 0.25), "gain_mean": 0.5},
            "t50": {"gain": (0.3, 0.5), "forgetting": (0.00, 0.02), "gain_mean": 0.4},
            "t10": {"gain": (0.1, 0.2), "forgetting": (-0.01, 0.01), "gain_mean": 0.15}}
    assert lab.choose(arms, 0.02) == "t50"
    assert lab.choose(arms, 0.005) is None
    assert lab.choose(arms, 0.3) == "t100"
