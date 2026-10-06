import itertools

import numpy as np

from frontierlab.evals.suite_v2 import core, ifeval
from frontierlab.labkit import load_target
from frontierlab.posttrain.tasks import Problem, follows_instruction

lab = load_target(__file__)


def test_pass_at_k():
    for n, c, k in [(8, 0, 1), (8, 2, 1), (8, 2, 4), (10, 9, 3), (6, 6, 6), (200, 3, 100)]:
        assert abs(lab.pass_at_k(n, c, k) - core.pass_at_k(n, c, k)) < 1e-12
    brute = np.mean([any(i < 2 for i in s) for s in itertools.combinations(range(6), 3)])
    assert abs(lab.pass_at_k(6, 2, 3) - brute) < 1e-12


def test_follows_tag():
    cases = [("42", True, "."), ("042", True, "P"), ("42", True, "P"), ("#42#", True, "Q"), ("#42", True, "Q"),
             ("42!", True, "E"), ("42!", False, "E"), ("4!2", True, "E"), ("", True, ".")]
    for text, fin, tag in cases:
        p = Problem(tag, 7, "+", 35, 2)
        assert lab.follows_tag(text, fin, tag) == follows_instruction(text, fin, p), (text, fin, tag)


def test_verdict():
    assert lab.verdict("task", 0.01, 0.2, 0.02) == "improved"
    assert lab.verdict("task", -0.1, 0.2, 0.02) == "no clear change"
    assert lab.verdict("task", -0.3, -0.1, 0.02) == "worse"
    assert lab.verdict("retention", -0.019, 0.1, 0.02) == "held"
    assert lab.verdict("retention", -0.021, 0.1, 0.02) == "regressed"
    assert lab.verdict("instruction", -0.5, -0.1, 0.02) == "regressed"


def test_prompt_and_instruction_level():
    flags = [[True, True], [True, False, False], [True]]
    p, i = lab.prompt_and_instruction_level(flags)
    assert abs(p - 2 / 3) < 1e-12 and abs(i - 4 / 6) < 1e-12


def test_loose_variants():
    r = "Sure:\n**\"hello\"**\nHope this helps"
    assert lab.loose_variants(r) == ifeval.loose_variants(r)
    assert len(lab.loose_variants("one line")) == 8
