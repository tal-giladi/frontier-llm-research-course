import numpy as np
import torch

from frontierlab.labkit import load_target
from frontierlab.scaling import downstream

lab = load_target(__file__)


def test_choice_metrics_match_reference():
    g = torch.Generator().manual_seed(0)
    lp = -torch.rand(50, 4, generator=g, dtype=torch.float64) * 20
    ans = torch.randint(0, 4, (50,), generator=g)
    mine = lab.choice_metrics(lp, ans, cont=8)
    ref = downstream.metrics_from_logprobs(lp, ans, cont=8)
    for k in ("acc", "p_correct", "nll_correct"):
        assert abs(mine[k] - ref[k]) < 1e-12, k


def test_sigmoid_curve():
    x = np.linspace(3, 9, 13)
    assert np.allclose(lab.sigmoid_curve(x, 6.0, 1.5, 0.25, 0.9), downstream.sigmoid_curve(x, 6.0, 1.5, 0.25, 0.9))
    assert abs(float(lab.sigmoid_curve(6.0, 6.0, 1.5, 0.25, 1.0)) - 0.625) < 1e-12


def test_exact_match_from_token_acc():
    assert np.allclose(lab.exact_match_from_token_acc([0.5, 0.9], 4), [0.0625, 0.6561])


def test_explained_variance_matches_pca():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 6)) @ rng.normal(size=(6, 6))
    ev = lab.explained_variance(X)
    ref = downstream.pca_capabilities(X, 6)["explained"]
    assert np.allclose(ev, ref) and abs(ev.sum() - 1) < 1e-12


def test_two_step():
    p = {"x0": 5.0, "s": 2.0, "lo": 0.25, "hi": 1.0}
    assert abs(lab.two_step(5.0, p) - 0.625) < 1e-12
    assert lab.two_step(4.0, p) > lab.two_step(6.0, p)
