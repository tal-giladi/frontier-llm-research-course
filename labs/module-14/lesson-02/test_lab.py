import math

from frontierlab.labkit import load_target
from frontierlab.posttrain.arms import t_interval

lab = load_target(__file__)


def test_select_lr():
    t = {1e-4: {"score": 0.30, "collapsed": False}, 3e-4: {"score": 0.36, "collapsed": False},
         1e-3: {"score": 0.40, "collapsed": True}}
    assert lab.select_lr(t) == 3e-4                       # the best run collapsed: not eligible
    t[1e-4]["score"] = 0.357
    assert lab.select_lr(t) == 1e-4                       # within 0.005: the smaller lr
    assert lab.select_lr({1e-3: {"score": 0.1, "collapsed": True}, 3e-4: {"score": 0.0, "collapsed": True}}) == 3e-4


def test_paired_interval_matches_t():
    a, b = [0.35, 0.31, 0.40], [0.30, 0.30, 0.33]
    m, lo, hi = lab.paired_interval(a, b)
    rm, rlo, rhi = t_interval([x - y for x, y in zip(a, b)])
    assert math.isclose(m, rm, abs_tol=1e-12) and math.isclose(lo, rlo, abs_tol=1e-9) and math.isclose(hi, rhi, abs_tol=1e-9)
    m1, lo1, hi1 = lab.paired_interval([0.4], [0.3])
    assert math.isclose(m1, 0.1) and math.isnan(lo1) and math.isnan(hi1)


def test_verdict_checks_control_first():
    assert lab.verdict((0.05, 0.01, 0.09), (0.02, -0.01, 0.05)) == "below control"
    assert lab.verdict((0.05, 0.01, 0.09), (0.10, 0.05, 0.15)) == "better than baseline"
    assert lab.verdict((-0.05, -0.09, -0.01), (0.10, 0.05, 0.15)) == "worse than baseline"
    assert lab.verdict((0.01, -0.03, 0.05), (0.10, 0.05, 0.15)) == "inconclusive"
    assert lab.verdict((0.01, float("nan"), float("nan")), (0.1, float("nan"), float("nan"))) == "below control"


def test_stability_flags():
    st = {"eval_drawdown": 0.12, "grad_spikes": 1, "entropy_drop": 0.3, "ratio_max": float("nan")}
    assert lab.stability_flags(st) == ["collapse", "entropy_collapse"]
    assert lab.stability_flags({"eval_drawdown": 0.0, "grad_spikes": 5, "entropy_drop": 0.0, "ratio_max": 80.0}) == \
        ["grad_spikes", "ratio_blowup"]
