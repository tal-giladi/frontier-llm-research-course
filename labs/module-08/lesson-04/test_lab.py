import math

import numpy as np

from frontierlab.labkit import load_target
from frontierlab.precision import scaling_law as ref

lab = load_target(__file__)


def test_n_eff():
    N = np.array([1e6, 1e6, 2e6])
    P = np.array([4.0, np.inf, 8.0])
    assert np.allclose(lab.n_eff(N, P, 2.6745), [1e6 * (1 - math.exp(-4 / 2.6745)), 1e6, 2e6 * (1 - math.exp(-8 / 2.6745))])


def test_p_star_matches_reference_and_grid():
    for g in (1.0, 2.6745, 4.0):
        assert abs(lab.p_star(g) - ref.p_star_closed_form(g)) < 1e-6
    assert abs(lab.p_star(2.6745) - ref.optimal_precision(1e21, P_grid=np.arange(2, 12, 0.01))["P_star"]) < 0.02
    assert abs(lab.p_star(1.0, k=1) - ref.p_star_closed_form(1.0, k=1)) < 1e-6


def test_fit_recovers_a_known_law():
    N = np.array([1e5, 2e5, 4e5] * 4)
    P = np.array([3.0] * 3 + [4.0] * 3 + [6.0] * 3 + [np.inf] * 3)
    ne = N * np.where(np.isinf(P), 1.0, 1 - np.exp(-P / 2.5))
    L = 7.0 * ne ** -0.25 + 1.5
    out = lab.fit(N, P, L, alphas=[0.2, 0.25, 0.3], gammas=[2.0, 2.5, 3.0])
    assert out["alpha"] == 0.25 and out["gamma"] == 2.5 and abs(out["E"] - 1.5) < 1e-9 and out["sse"] < 1e-18
    full = ref.fit_precision_law(N, P, L, alphas=[0.2, 0.25, 0.3], gammas=[2.0, 2.5, 3.0])
    assert abs(full["A"] - out["A"]) < 1e-9


def test_ptq_exponent():
    D = np.array([1e5, 2e5, 4e5, 8e5])
    assert abs(lab.ptq_exponent(D, 0.03 * D ** 0.5068) - 0.5068) < 1e-9
