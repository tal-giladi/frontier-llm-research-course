"""Lab 16.1 — environments and verifiers. Fill in the TODOs; run `pytest labs/module-16/lesson-01` to check.
``verifier_lab.py`` uses these functions to test the course verifiers and environments.
"""

from __future__ import annotations

import math


def confusion(rows: list[dict]) -> dict:
    """Per verifier, from the rows of ``frontierlab.agents.codeenv.verifier_report``.

    Each row is {"verifier", "kind" ("correct" | "wrong" | "loophole"), "passed", ...}. Return
    {verifier: {"false_accept_rate", "false_reject_rate", "loophole_accept_rate"}} where a false accept is a
    "wrong" or "loophole" row that passed (rate over all such rows), a false reject a "correct" row that did not
    pass (rate over correct rows), and the loophole rate is over "loophole" rows only. A rate with no rows is NaN.
    """
    raise NotImplementedError("TODO 1: false accepts and false rejects per verifier")


def miss_probability(p_fail: float, n: int) -> float:
    """Probability that ``n`` independent random test inputs all miss a bug that fails on a fraction ``p_fail``
    of inputs (so a verifier with ``n`` such tests accepts the wrong solution)."""
    raise NotImplementedError("TODO 2: probability that n tests miss a bug")


def tests_needed(p_fail: float, delta: float) -> int:
    """Smallest n >= 1 with miss_probability(p_fail, n) <= delta. Return -1 if p_fail <= 0 (no number of
    random tests finds a bug that never shows) and 1 if p_fail >= 1."""
    raise NotImplementedError("TODO 3: tests needed for a target miss probability")


def reset_verdict(check: dict) -> str:
    """Read the dict of ``frontierlab.agents.env.reset_check``: "untested" if the disturbance did not change the
    state (the check proved nothing), "clean" if reset restored the initial state and a fresh environment
    matches it, otherwise "leaks"."""
    raise NotImplementedError("TODO 4: interpret a reset check")
