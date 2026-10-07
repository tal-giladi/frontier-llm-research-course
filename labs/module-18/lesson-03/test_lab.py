import math

from frontierlab.evals.suite_v3 import contamination as C
from frontierlab.evals.suite_v3 import horizon as Hz
from frontierlab.labkit import load_target

lab = load_target(__file__)


def test_horizon_from_fit_matches_reference():
    for a, b in [(3.5, -0.6), (4.8, -0.7), (1.3, -0.65)]:
        for p in (0.5, 0.8):
            assert math.isclose(lab.horizon_from_fit(a, b, p), Hz.horizon_from(a, b, p), rel_tol=1e-12)
    assert math.isclose(lab.horizon_from_fit(6.0, -1.0), 64.0)          # a + b log2 t = 0 at t = 2^6
    assert lab.horizon_from_fit(1.0, 0.2) == math.inf


def test_palm_rule():
    idx = C.build_index(["the quick brown fox"], 8)
    assert lab.palm_contaminated("quick brown", idx) is True
    assert lab.palm_contaminated("quick red dog", idx) is False
    assert lab.palm_contaminated("short", idx) is False
    item = ".07+35=42"
    idx2 = C.build_index([".07+35=4"], 6)                               # 3 of the item's 4 six-grams
    assert lab.palm_contaminated(item, idx2, n=6, threshold=0.7) is True
    assert lab.palm_contaminated(item, idx2, n=6, threshold=0.8) is False
    assert lab.palm_contaminated(item, idx2, n=6) == C.palm_rule([item], idx2, 6)[0]


def test_min_k():
    lp = [-0.1, -5.0, -0.2, -3.0, -0.3]
    assert math.isclose(lab.min_k_percent(lp, 0.4), -4.0)
    assert math.isclose(lab.min_k_percent(lp, 0.2), -5.0)
    assert math.isclose(lab.min_k_percent(lp, 0.01), -5.0)              # at least one token
    assert math.isclose(lab.min_k_percent(lp, 0.4), C.min_k_prob(lp, 0.4))


def test_v3_verdict_order():
    assert lab.v3_verdict(True, True, True, ["retired: do not use"]) == "do not report: retired benchmark"
    assert lab.v3_verdict(True, True, False, []) == "fail: contamination"
    assert lab.v3_verdict(False, False, True, []) == "fail: contamination"
    assert lab.v3_verdict(False, True, True, []) == "fail: regression"
    assert lab.v3_verdict(True, True, True, ["saturated: ..."]) == "pass with flags"
    assert lab.v3_verdict(True, True, True, []) == "pass"
