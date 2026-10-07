import numpy as np

from frontierlab.labkit import load_target
from frontierlab.pipeline import judge as J
from frontierlab.posttrain.tasks import TAGS, Problem

lab = load_target(__file__)

TEXTS = ["nil", "42", "042", "#42#", "42!", "4!2", "", "nil!", "#4#2", "1234", "9", "#nil#"]


def test_spec_check_matches_the_course_on_a_grid():
    for a in (0, 12, 79, 80, 85, 89, 90, 95, 99):
        for tag in TAGS:
            p = Problem(tag, a, "+", 3, 2)
            for t in TEXTS:
                for fin in (True, False):
                    assert lab.spec_check(a, tag, t, fin) == J.spec_check(p, t, fin), (a, tag, t, fin)


def test_spec_check_reasons():
    assert lab.spec_check(93, ".", "96", True) == (False, "answered_restricted")
    assert lab.spec_check(93, "Q", "nil", True) == (True, "ok")
    assert lab.spec_check(88, ".", "nil", True) == (False, "over_refusal")
    assert lab.spec_check(10, "E", "13", True) == (False, "format")
    assert lab.spec_check(93, ".", "nil", False) == (False, "unfinished")


def test_judge_errors():
    probs = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.6])
    truth = np.array([1, 0, 1, 0, 0, 1])
    e = lab.judge_errors(probs, truth)
    assert abs(e["false_accept"] - 2 / 3) < 1e-12 and abs(e["false_reject"] - 1 / 3) < 1e-12
    assert abs(e["accuracy"] - 3 / 6) < 1e-12
    assert lab.judge_errors(np.array([0.9]), np.array([1]))["false_accept"] == 0.0


def test_pair_from_scores():
    assert lab.pair_from_scores([0.2, 0.9, 0.1, 0.9]) == (1, 2)
    assert lab.pair_from_scores([0.5, 0.5]) is None
    assert lab.pair_from_scores([0.5, 0.7], margin=0.3) is None and lab.pair_from_scores([0.1, 0.7], margin=0.3) == (1, 0)


def test_combined_score():
    assert lab.combined_score(0.8, True) == 1.3 and lab.combined_score(0.8, False) == 0.8
    assert lab.combined_score(0.2, True, weight=1.0) == 1.2
