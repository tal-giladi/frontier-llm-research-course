import math

from frontierlab.labkit import load_target

lab = load_target(__file__)

ROWS = ([{"verifier": "visible", "kind": "correct", "passed": True}] * 4
        + [{"verifier": "visible", "kind": "wrong", "passed": True}] * 1
        + [{"verifier": "visible", "kind": "wrong", "passed": False}] * 3
        + [{"verifier": "visible", "kind": "loophole", "passed": True}] * 4
        + [{"verifier": "robust", "kind": "correct", "passed": True}] * 3
        + [{"verifier": "robust", "kind": "correct", "passed": False}] * 1
        + [{"verifier": "robust", "kind": "wrong", "passed": False}] * 4)


def test_confusion():
    c = lab.confusion(ROWS)
    assert math.isclose(c["visible"]["false_accept_rate"], 5 / 8)
    assert math.isclose(c["visible"]["false_reject_rate"], 0.0)
    assert math.isclose(c["visible"]["loophole_accept_rate"], 1.0)
    assert math.isclose(c["robust"]["false_reject_rate"], 0.25)
    assert c["robust"]["false_accept_rate"] == 0.0 and math.isnan(c["robust"]["loophole_accept_rate"])


def test_miss_probability_and_tests_needed():
    assert math.isclose(lab.miss_probability(0.1, 8), 0.9 ** 8)
    assert lab.tests_needed(0.1, 0.05) == 29              # 0.9^28 = 0.052, 0.9^29 = 0.047
    assert lab.tests_needed(0.5, 0.05) == 5
    assert lab.tests_needed(0.0, 0.05) == -1 and lab.tests_needed(1.0, 0.05) == 1
    n = lab.tests_needed(0.03, 0.01)
    assert lab.miss_probability(0.03, n) <= 0.01 < lab.miss_probability(0.03, n - 1)


def test_reset_verdict():
    assert lab.reset_verdict({"disturb_changed_state": True, "reset_restores": True, "fresh_matches": True}) == "clean"
    assert lab.reset_verdict({"disturb_changed_state": True, "reset_restores": False, "fresh_matches": True}) == "leaks"
    assert lab.reset_verdict({"disturb_changed_state": False, "reset_restores": True, "fresh_matches": True}) == "untested"
