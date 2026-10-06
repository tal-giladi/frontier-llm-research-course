import numpy as np
import pytest

from frontierlab.labkit import load_target
from frontierlab.scaling import laws

lab = load_target(__file__)
C0 = laws.CHINCHILLA


def test_loss_matches_reference():
    N, D = np.array([1e8, 7e10]), np.array([2e9, 1.4e12])
    assert np.allclose(lab.chinchilla_loss(N, D, C0), laws.loss(N, D, C0), rtol=0, atol=1e-12)


def test_compute_optimal_matches_reference_and_budget():
    for C in (1e19, 5.76e23, 3.8e25):
        N, D = lab.compute_optimal(C, C0)
        ref = laws.compute_optimal(C, C0)
        assert abs(N / ref["N"] - 1) < 1e-10 and abs(6 * N * D / C - 1) < 1e-10
    for c in (laws.BESIROGLU,):
        N, D = lab.compute_optimal(1e24, c)
        assert abs(N / laws.compute_optimal(1e24, c)["N"] - 1) < 1e-10


def test_overtraining_overhead():
    N, D = lab.compute_optimal(1e22, C0)
    assert abs(lab.overtraining_overhead(N, D, C0)) < 1e-6
    for n, d in ((8e9, 15e12), (6e8, 36e12)):
        assert abs(lab.overtraining_overhead(n, d, C0) - laws.overtraining(n, d, C0)["overhead"]) < 1e-6


def test_isoflop_vertex():
    N = np.geomspace(1e5, 1e7, 5)
    L = 3.0 + 0.4 * (np.log10(N) - 6.2) ** 2
    assert abs(lab.isoflop_vertex(N, L) / 10 ** 6.2 - 1) < 1e-9
    with pytest.raises(ValueError):
        lab.isoflop_vertex(N, 3.0 - 0.1 * (np.log10(N) - 6) ** 2)


def test_effective_data():
    U = np.array([100.0, 100.0, 100.0, 100.0])
    D = np.array([50.0, 400.0, 1600.0, 1e6])
    assert np.allclose(lab.effective_data(U, D), laws.effective_data(U, D), rtol=1e-12)
