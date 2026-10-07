"""Lab 13.4 — Thinking modes and budgets. Fill in the TODOs; run `pytest labs/module-13/lesson-04` to check.

The toy task (frontierlab/pipeline/thinking.py): three-digit addition. In thinking mode the response is a
trace of column sums from the units up, each with the incoming carry and written with two digits
("a" units, "b" tens, "c" hundreds), then "#" (the toy's </think>), then the answer:
478 + 365 -> "a13b14c08#843". In non-thinking mode the thinking block is empty: "#843".
"""

from __future__ import annotations

import numpy as np


def trace(a: int, b: int) -> str:
    """The thinking trace for a + b (a, b < 1000): "a13b14c08" for 478 + 365."""
    raise NotImplementedError("TODO 1: the column-sum trace")


def truncated_target(a: int, b: int, k: int) -> str:
    """A budget-aware training target: the first k characters of the trace, then "#", then the answer
    (k >= len(trace) keeps the whole trace). truncated_target(478, 365, 3) == "a13#843"."""
    raise NotImplementedError("TODO 2: thinking cut at k tokens")


def routed_cost(p_easy: np.ndarray, threshold: float, correct_think: np.ndarray, correct_nothink: np.ndarray,
                tokens_think: np.ndarray, tokens_nothink: np.ndarray) -> tuple[float, float, float]:
    """Send a prompt to thinking mode iff the router's p_easy (its estimate that the non-thinking answer is
    right) is below ``threshold``. Return (accuracy, mean generated tokens, fraction sent to thinking)."""
    raise NotImplementedError("TODO 3: accuracy and cost of a routing rule")


def pareto(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The (tokens, accuracy) points that no other point beats: a point is dominated if another has tokens
    <= and accuracy >= with at least one strict. Return the non-dominated ones sorted by tokens (duplicates once)."""
    raise NotImplementedError("TODO 4: the cost-accuracy frontier")
