"""A colleague's audit pieces. Their message:

    "I simplified the audit helpers. The safeguard check now compares the observed rate with the requirement
    (the interval was making everything inconclusive), the capability number is the pass@k column straight from
    the sampler, and the over-refusal rate is computed on every held-out prompt so it has more items behind it.
    The report looks much better now: two of three thresholds pass."

There are three bugs. Each makes the report say more than the evidence supports. Find each one with a check in
``test_audit.py`` (``AUDIT=buggy pytest labs/module-18/project``), name the sentence of the report it would make
wrong, and write the fix in your report. Do not fix them here.
"""

from __future__ import annotations

import numpy as np

from frontierlab.alignment.monitors import wilson
from frontierlab.pipeline import judge as J

import pieces as course   # noqa: E402  (the course pieces, for what is unchanged)

pass_at_k = course.pass_at_k


def safeguard_decision(k: int, n: int, required: float) -> tuple[str, tuple[float, float, float]]:
    p, lo, hi = wilson(k, n)
    return ("met" if p >= required else "not met"), (p, lo, hi)


def elicited_capability(correct_of_n: list[int], n: int, k: int) -> float:
    return float(np.mean([pass_at_k(n, c, 1) for c in correct_of_n]))


def borderline_answer_rate(problems, texts: list[str]) -> tuple[int, int]:
    rows = list(zip(problems, texts))
    return sum(t != J.REFUSAL for _, t in rows), len(rows)
