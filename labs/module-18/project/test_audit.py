"""Checks for the audit pieces (the project's correctness suite).

    pytest labs/module-18/project                  # the course pieces: every test passes
    AUDIT=buggy pytest labs/module-18/project      # the colleague's pieces: which tests fail, and why?
"""

import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from frontierlab.evals.suite_v2.core import pass_at_k as ref_pass_at_k  # noqa: E402
from frontierlab.labkit import load_path  # noqa: E402
from frontierlab.pipeline import judge as J  # noqa: E402
from frontierlab.posttrain.tasks import Problem  # noqa: E402

MOD = load_path(str(HERE / ("buggy_audit.py" if os.environ.get("AUDIT") == "buggy" else "pieces.py")))


def test_safeguard_needs_the_lower_bound():
    assert MOD.safeguard_decision(98, 100, 0.95)[0] == "inconclusive"
    assert MOD.safeguard_decision(392, 400, 0.95)[0] == "met"
    assert MOD.safeguard_decision(60, 100, 0.95)[0] == "not met"


def test_capability_is_pass_at_the_stated_k():
    c = [0, 1, 0, 3, 16]
    want = sum(ref_pass_at_k(16, x, 16) for x in c) / len(c)
    assert math.isclose(MOD.elicited_capability(c, 16, 16), want)
    assert math.isclose(MOD.elicited_capability([1], 16, 1), 1 / 16)


def test_borderline_rate_counts_only_borderline_prompts():
    probs = [Problem(".", a, "+", 1, 2) for a in (10, 20, 85, 86, 95)]
    texts = ["11", "21", J.REFUSAL, "87", J.REFUSAL]
    assert MOD.borderline_answer_rate(probs, texts) == (1, 2)
