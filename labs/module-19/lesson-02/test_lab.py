import math

import pytest

from frontierlab.labkit import load_target
from frontierlab.research import reproduce as R

lab = load_target(__file__)
L0 = math.log(8192)


def test_lr_sensitivity_matches_shared():
    for sweep in ({3e-3: 6.41, 1e-2: 6.42, 3e-2: 6.60}, {3e-3: 6.31, 1e-2: 6.71, 3e-2: float("nan")},
                  {3e-3: 6.3, 1e-2: 12.0}, {1e-2: 6.0}, {1e-3: 7.0, 3e-3: 6.5, 1e-2: float("inf")}):
        assert math.isclose(lab.lr_sensitivity(sweep, L0), R.lr_sensitivity(sweep, L0), abs_tol=1e-12), sweep


def test_seed_effects_sign_and_order():
    without, with_ = {2: 0.9, 0: 0.5, 1: 0.7}, {0: 0.1, 1: 0.2, 2: 0.3}
    assert lab.seed_effects(without, with_) == pytest.approx([0.4, 0.5, 0.6])


@pytest.mark.parametrize("effects", [[0.30, 0.25, 0.28], [-0.30, -0.25, -0.28], [0.01, -0.01, 0.0],
                                     [0.20, -0.05, 0.10], [0.040, 0.041, 0.042], [0.06, 0.07]])
def test_decide_matches_shared(effects):
    for kw in ({}, {"published": 0.27, "magnitude_tol": 0.05}, {"published": 0.6, "magnitude_tol": 0.05}):
        got, want = lab.decide(effects, 0.057, **kw), R.decide(effects, 0.057, **kw)
        assert got["direction"] == want["direction"] and got["magnitude"] == want["magnitude"]
        assert math.isclose(got["mean"], want["mean"]) and got["n"] == want["n"]
        assert all(math.isclose(a, b) for a, b in zip(got["ci"], want["ci"]))


def test_tolerance_is_stated_relative_to_noise_before_results():
    from datetime import date
    assert lab.TOLERANCE >= lab.NOISE_FLOOR > 0
    assert lab.STATED_ON <= date.today().isoformat()


def test_deviation_log_is_complete():
    devs = lab.deviations()
    rec = R.Record("c", "s, Figure 1", "https://arxiv.org/abs/2309.14322", lab.TOLERANCE, "2 x noise", lab.STATED_ON,
                   "2099-01-01", [0.3, 0.3, 0.31], R.decide([0.3, 0.3, 0.31], lab.TOLERANCE), devs,
                   "not needed", "toy preset at this scale")
    assert [p for p in R.validate(rec) if p.startswith("deviations")] == []
    scale = [d for d in devs if d.aspect == "scale"]
    assert scale and scale[0].paper != scale[0].ours, "the scale differs from the paper: say how"
