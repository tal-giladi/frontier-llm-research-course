"""Lab 15.1 — A test-time compute experiment. Fill in the TODOs; run `pytest labs/module-15/lesson-01` to check.

``ttc_lab.py`` uses your functions for the selection-gap table (oracle pass@N against what each procedure
actually selects) and for the spend of every strategy; the full budget-matched table uses the course code.
"""

from __future__ import annotations

import numpy as np


def majority_vote(answers):
    """The most frequent answer that is not None. Ties go to the tied answer that appears FIRST in ``answers``.
    Return None if every answer is None. Never look at correctness."""
    raise NotImplementedError("TODO 1: majority vote")


def weighted_vote(answers, weights):
    """The non-None answer whose candidates' weights sum highest; ties to first occurrence; None if all None."""
    raise NotImplementedError("TODO 2: verifier-weighted vote")


def policy_token_equivalents(prefill: float, decode: float, verifier_tokens: float, policy_params: int,
                             verifier_params: int) -> float:
    """FLOPs of a strategy divided by the FLOPs of one policy token (2 N_policy per token for each model):
    prefill + decode + (N_verifier / N_policy) * verifier_tokens."""
    raise NotImplementedError("TODO 3: spend in policy-token equivalents")


def subset_success(answers, gold, N: int, method: str, scores=None, resamples: int = 200, seed: int = 0) -> np.ndarray:
    """Per problem, the success rate of ``method`` ("majority" | "best_of_n" | "weighted") applied to N
    candidates drawn WITHOUT replacement from that problem's pool, averaged over ``resamples`` draws with
    ``rng = np.random.default_rng(seed)`` (use ``rng.choice(n, size=N, replace=False)``, problems in order).
    With N equal to the pool size use the whole pool once. best_of_n picks the first highest score."""
    raise NotImplementedError("TODO 4: success of a selection procedure on random subsets")
