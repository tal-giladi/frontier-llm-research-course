"""Lab 06.3 — combining changes. Fill in the TODOs; run `pytest labs/module-06/lesson-03` to check."""

from __future__ import annotations

from itertools import combinations  # noqa: F401  (TODO 4)

import numpy as np  # noqa: F401

from frontierlab.stats import bootstrap_ci  # noqa: F401  (TODO 2)


def interaction(cells: dict) -> float:
    """The 2 x 2 interaction contrast from four cell means.

    ``cells[(a, b)]`` is the mean held-out loss with change A on (a = 1) or off (a = 0) and change B on or off.
    The interaction is how much A's effect changes when B is on: (m11 − m01) − (m10 − m00).
    """
    raise NotImplementedError("TODO 1: the interaction contrast")


def interaction_ci(per_window: dict, n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """The interaction contrast with a 95% bootstrap interval over evaluation windows.

    ``per_window[(a, b)]`` holds per-window losses of cell (a, b), all on the same windows in the same order.
    Compute the contrast window by window (that is the pairing), then ``bootstrap_ci`` of its mean.
    Return (point, low, high).
    """
    raise NotImplementedError("TODO 2: per-window contrast, then bootstrap")


def additive_prediction(base: float, singles: dict) -> float:
    """The combination's loss if every change's effect simply added: base + sum over changes of (single − base).
    ``singles`` maps a change's name to the loss of Baseline-0 with only that change."""
    raise NotImplementedError("TODO 3: the additive prediction")


def design_cells(components: list[str], design: str) -> set[frozenset]:
    """The distinct cells (each a frozenset of the components switched on) of an experimental design.

    "full": every subset of the components (2^k cells).
    "lite" (factorial-lite): the baseline (empty set), each component alone, the full combination, and the full
    combination with each single component left out. Duplicates count once (for k = 2 the leave-one-out cells
    are the singles).
    """
    raise NotImplementedError("TODO 4: enumerate the cells of a design")
