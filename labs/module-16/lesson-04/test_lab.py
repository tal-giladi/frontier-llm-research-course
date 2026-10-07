import math

from frontierlab.agents import monitor
from frontierlab.labkit import load_target

lab = load_target(__file__)


def test_divergence_matches_monitor():
    train = [{"reward": 0.4, "gold": 0.25}] * 10 + [{"reward": 1.0, "gold": 0.0}] * 10
    assert math.isclose(lab.divergence(train), 0.85)
    assert math.isclose(lab.divergence(train), monitor.proxy_gold_divergence(train)["divergence"])
    honest = [{"reward": 0.3, "gold": 0.3}] * 5 + [{"reward": 0.8, "gold": 0.8}] * 5
    assert abs(lab.divergence(honest, window=5)) < 1e-12


def test_audit_flags():
    st = {"divergence": 0.69, "gold_change": -0.27, "d_rule": -0.8, "d_table": 0.82, "d_other": -0.02,
          "len_ratio": 1.08, "trunc": 0.0}
    assert lab.audit_flags(st) == ["divergence", "gold_drop", "kind_shift"]
    st2 = {"divergence": 0.0, "gold_change": 0.4, "d_rule": 0.17, "d_table": -0.16, "d_other": -0.01,
           "len_ratio": 0.9, "trunc": 0.25}
    assert lab.audit_flags(st2) == ["length"]
    assert lab.audit_flags({**st2, "trunc": 0.0}) == []


def test_verdict():
    assert lab.verdict([["kind_shift"], ["gold_drop"]], (0.3, 0.1, 0.5)) == "misspecified"
    assert lab.verdict([[], ["kind_shift"]], (0.3, 0.1, 0.5)) == "beats control"
    assert lab.verdict([[], []], (0.1, -0.2, 0.4)) == "inconclusive"
    assert lab.verdict([[]], (0.1, float("nan"), float("nan"))) == "inconclusive"


def test_first_step_above():
    assert lab.first_step_above([0, 10, 20, 30], [0.1, 0.3, 0.6, 0.9]) == 20
    assert lab.first_step_above([0, 10], [0.1, 0.2]) is None
    assert lab.first_step_above([0, 10], [0.1, 0.5], level=0.5) == 10
