import numpy as np

from frontierlab.datax import regmix
from frontierlab.datax import train as dtrain
from frontierlab.labkit import load_target

lab = load_target(__file__)


def test_sampling_matches_reference():
    X = lab.sample_mixtures(30, [24, 17, 18, 14], seed=3)
    assert np.allclose(X, regmix.sample_mixtures(30, [24, 17, 18, 14], seed=3))
    assert np.allclose(X.sum(1), 1) and (X >= 0).all()


def test_ridge_and_loo():
    rng = np.random.default_rng(0)
    X = rng.dirichlet(np.ones(4), size=25)
    y = 3.0 + X @ np.array([0.5, -0.2, 0.1, 0.3]) + rng.normal(0, 0.01, 25)
    w = lab.ridge_fit(X, y, 1e-3)
    ref = regmix.fit_ridge(X, y, l2=1e-3)
    assert np.allclose(w, ref["w"]) and np.allclose(lab.ridge_predict(w, X), regmix.predict(ref, X))
    assert np.allclose(lab.loo_ridge(X, y, 1e-3), regmix.loo_predictions("ridge", X, y, l2=1e-3))


def test_law():
    X = np.array([[1.0, 0.0], [0.5, 0.5]])
    # c=2, k=1, t=(0, ln 2): rows -> 2 + 1 = 3 and 2 + sqrt(2)
    assert np.allclose(lab.law_predict(2.0, 1.0, [0.0, np.log(2)], X), [3.0, 2 + np.sqrt(2)])


def test_anneal_lr_matches_wrapper():
    for s in range(0, 21):
        assert abs(lab.anneal_lr(s, 20, 1e-3, 4) - dtrain.anneal_lr(s, 20, 1e-3, 4)) < 1e-15
    assert lab.anneal_lr(20, 20, 1.0, 4) == 0.0


def test_anneal_decision():
    assert lab.anneal_decision((-0.2, -0.1), (-0.01, 0.01), 0.02) == "adopt"
    assert lab.anneal_decision((-0.2, -0.1), (0.03, 0.05), 0.02) == "reject"
    assert lab.anneal_decision((0.0, 0.1), (-0.01, 0.0), 0.02) == "reject"
    assert lab.anneal_decision((-0.1, 0.02), (-0.01, 0.01), 0.02) == "inconclusive"
