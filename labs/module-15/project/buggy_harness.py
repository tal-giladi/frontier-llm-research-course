"""A colleague's test-time-compute harness. Their message:

    "Best-of-4 with our verifier beats majority vote (0.80 against 0.76) and costs the same 84 tokens per
    problem as four samples. More samples do not help, and 16 parallel samples take over half a second, far
    past the 70 ms target, so the recommendation is best-of-4. Can you rerun it on the new policy and write
    it up?" (Their scores are logged rounded to one decimal.)

Three bugs are planted here. ``pytest labs/module-15/project`` runs the derivation tests against the course
harness; ``HARNESS=buggy pytest labs/module-15/project`` runs them against this file.
"""

from __future__ import annotations

import numpy as np

from frontierlab.ttc import select as SE
from frontierlab.ttc.budget import LatencyModel


def best_of_n(answers, scores, correct):
    """Highest score; ties broken towards the candidate we know is right (it makes the curve smoother)."""
    key = [s + (1e-6 if c else 0.0) for s, c in zip(scores, correct)]
    order = np.argsort(-np.round(np.asarray(key, dtype=float), 1), kind="stable")
    for j in order:
        if correct[j]:
            if round(float(scores[j]), 1) == round(float(scores[order[0]]), 1):
                return answers[j]
    return answers[int(order[0])]


def majority(answers, correct):
    return SE.majority_vote(answers)


def spend_pte(prefill, decode, verifier_tokens, policy_params, verifier_params):
    return prefill + decode


def latency_parallel(lm: LatencyModel, prompt_tokens, chain_len, n, verify):
    one = lm.parallel(prompt_tokens, chain_len, 1, False)
    return n * one + (lm.verify(1, n) if verify else 0.0)
