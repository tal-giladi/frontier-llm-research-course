"""Lab 20.1 — the capstone's uncertainty, decision and soundness checks. Fill in the TODOs; run
`pytest labs/module-20/lesson-01` to check, then `python labs/module-20/lesson-01/capstone_lab.py`, which runs the
QK-Clip capstone end to end with *your* interval and decision rule.

The reference versions live in `frontierlab.capstone` (`uncertainty`, `package`); write yours without importing them.
"""

from __future__ import annotations

import math  # noqa: F401  (you will need it)

import numpy as np  # noqa: F401

from frontierlab.record.diff import diff_cards  # noqa: F401  (for TODO 4)

# For each decision, the claim directions it allows (TODO 3).
DIRECTIONS = {"equivalent": ("equivalent", "no-difference-detected"), "a_lower": ("lower",), "a_higher": ("higher",),
              "inconclusive": ("no-difference-detected", "inconclusive")}
LABELS = ("MEASURED", "PUBLICLY DOCUMENTED", "INFERENCE/SPECULATION", "PROJECTED", "ANALYSIS")


def hierarchical_interval(a, b, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Paired two-level bootstrap of mean(a - b). ``a`` and ``b`` are (seeds, items) arrays: row s of both comes from
    seed s, column w from evaluation item w.

    Each of the ``n_boot`` resamples draws S seed indices with replacement, then for each drawn seed W item indices
    with replacement, and averages ``d = a - b`` over the drawn (seed, item) pairs. Return
    ``{"mean_diff": d.mean(), "ci": (lo, hi), "method": "<a sentence naming the method, S and W>"}`` with lo, hi the
    alpha/2 and 1 - alpha/2 quantiles of the resampled means. Raise ValueError for arrays of different shapes.
    """
    raise NotImplementedError("TODO 1: the paired hierarchical bootstrap over seeds and items")


def decide(mean: float, lo: float, hi: float, margin: float) -> str:
    """The decision rule stated in the contract, for ``mean`` = a - b and its interval [lo, hi].

    "equivalent" if the interval lies inside [-margin, +margin]; else "a_lower" if it lies below zero; else
    "a_higher" if it lies above zero; else "inconclusive". A NaN anywhere gives "inconclusive".
    """
    raise NotImplementedError("TODO 2: the decision rule")


def claim_problems(claims: list[dict], decisions: dict[str, str]) -> list[tuple[str, str]]:
    """Claims within evidence. ``claims`` are claims.yaml entries ``{id, text, label, comparison, direction, scope,
    source}``; ``decisions`` maps each comparison name to the decision *recomputed* from its interval.

    Return ``(code, claim id)`` pairs, in claim order: CLAIM_LABEL if the label is not in LABELS (and nothing else for
    that claim); CLAIM_SOURCE for a PUBLICLY DOCUMENTED claim without a source; for a MEASURED claim, CLAIM_COMPARISON
    if its comparison is unknown, otherwise CLAIM_DIRECTION if its direction is not one DIRECTIONS allows, then
    CLAIM_SCOPE if its scope (default "course") is not "course".
    """
    raise NotImplementedError("TODO 3: claims within evidence")


def comparison_problems(claim: dict, comparison: dict, cards: dict[str, dict]) -> list[tuple[str, str]]:
    """Parity of one comparison ``{name, a, b, changed}`` of ``claim`` (claim.yaml as a dict). ``cards`` maps run
    names ``"<arm>-s<seed>"`` to run-card dicts. An arm's seeds are ``claim["arms"][arm].get("seeds", claim["seeds"])``.

    Return, in this order: ("TUNING", name) if ``claim["tuning"]`` differs between the arms; ("SEEDS", name) if their
    seed sets differ; then for each shared seed in increasing order whose two cards both exist, ("PARITY", "seed s")
    if ``diff_cards(card_b, card_a, changed=comparison["changed"], axis=claim["axis"])`` has any finding whose
    severity is "invalidates".
    """
    raise NotImplementedError("TODO 4: tuning, seed and budget parity of one comparison")
