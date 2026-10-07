"""The course's evaluation harness for the project (the reference the planted-bug version is tested against).

Four functions a test-time-compute report depends on. Each one is a thin wrapper of ``frontierlab.ttc`` so that
``test_harness.py`` can run the same tests against ``buggy_harness.py`` (``HARNESS=buggy``).
"""

from __future__ import annotations

from frontierlab.ttc import select as SE
from frontierlab.ttc.budget import LatencyModel, Spend


def best_of_n(answers, scores, correct):
    """Pick the answer of the highest-scoring candidate. ``correct`` is passed in by the evaluation loop for
    bookkeeping; a selection procedure must never use it."""
    return SE.best_of_n(answers, scores)


def majority(answers, correct):
    return SE.majority_vote(answers)


def spend_pte(prefill: float, decode: float, verifier_tokens: float, policy_params: int, verifier_params: int) -> float:
    return Spend(policy_params, verifier_params, prefill, decode, verifier_tokens).policy_token_equivalents


def latency_parallel(lm: LatencyModel, prompt_tokens: int, chain_len: float, n: int, verify: bool) -> float:
    return lm.parallel(prompt_tokens, chain_len, n, verify)
