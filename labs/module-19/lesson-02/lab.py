"""Lab 19.2 — reproducing a published claim. Fill in the TODOs; run `pytest labs/module-19/lesson-02` to check.
``repro_lab.py`` uses these functions and constants.
"""

from __future__ import annotations

from frontierlab.research.reproduce import Deviation, t_interval  # noqa: F401  (you will need both)

# Stated before the runs. Keep the date honest: the record is refused if it is later than the first result.
# NOISE_FLOOR is lesson 01.4's measured seed std of the toy recipe's held-out loss (5 seeds, lr 3e-3, 300 steps of
# 16 x 128 tokens): a noise estimate that exists before any run of this lab.
NOISE_FLOOR = 0.0286
TOLERANCE = 2 * NOISE_FLOOR
STATED_ON = "2026-10-07"


def lr_sensitivity(losses: dict, l0: float) -> float:
    """Wortsman et al. section 2.2 for one sweep. ``losses``: {learning rate: final loss}, nan/inf for a diverged
    run. Each loss is replaced by min(loss, l0) (a diverged run counts as l0); return the mean of those values minus
    their minimum."""
    raise NotImplementedError("TODO 1: learning-rate sensitivity")


def seed_effects(sens_without: dict, sens_with: dict) -> list[float]:
    """Per-seed effect in the paper's direction: sensitivity without QK-norm minus with, for each seed key in sorted
    order. Positive means QK-norm reduced the sensitivity, as the paper claims."""
    raise NotImplementedError("TODO 2: signed per-seed effects")


def decide(effects, tolerance, published=None, magnitude_tol=None) -> dict:
    """The reproduction decision. With (mean, lo, hi) = t_interval(effects):
    direction "reproduced" if lo > 0 and mean >= tolerance; "contradicted" if hi < 0; "not reproduced" if
    hi < tolerance; otherwise "inconclusive".
    magnitude "not comparable" if published or magnitude_tol is None; else "matches" if |mean - published| <=
    magnitude_tol, otherwise "differs".
    Return {"direction", "magnitude", "mean", "ci": (lo, hi), "n", "tolerance", "published", "magnitude_tol"}."""
    raise NotImplementedError("TODO 3: direction and magnitude, decided separately")


def deviations() -> list[Deviation]:
    """The deviation log: one Deviation per aspect in frontierlab.research.reproduce.ASPECTS (scale, data, tokens,
    optimizer, learning rates, evaluation, seeds, code), each with what the paper did, what you did, why, and the
    effect you expect on the claim (weakens / strengthens / unknown / none expected). Read the paper's section 2.1
    and the arguments repro_lab.py passes before you write them."""
    raise NotImplementedError("TODO 4: the deviation log")
