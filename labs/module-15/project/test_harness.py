"""Derivation tests for a test-time-compute harness (the project's debugging task).

    pytest labs/module-15/project                   # the course harness: all pass
    HARNESS=buggy pytest labs/module-15/project     # the colleague's harness
"""

import os

import numpy as np

from frontierlab.labkit import load_path
from frontierlab.ttc.budget import LatencyModel

H = load_path(os.path.join(os.path.dirname(__file__), "buggy_harness.py" if os.environ.get("HARNESS") == "buggy"
                           else "harness.py"))


def test_selection_never_reads_correctness():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = 8
        answers = [str(rng.integers(0, 4)) for _ in range(n)]
        scores = np.round(rng.random(n), 1).tolist()            # ties are common at one decimal
        c1 = rng.random(n) < 0.5
        c2 = rng.permutation(c1)
        assert H.best_of_n(answers, scores, c1.tolist()) == H.best_of_n(answers, scores, c2.tolist())
        assert H.majority(answers, c1.tolist()) == H.majority(answers, c2.tolist())


def test_verifier_tokens_are_paid_for():
    # 16 samples of 19 tokens, prompt 12 once, a same-size verifier reading 31 tokens per candidate
    assert H.spend_pte(12, 16 * 19, 16 * 31, 308_400, 308_497) > 12 + 16 * 19 + 16 * 31 * 0.99
    # a verifier 4x the policy costs 4 policy tokens per token it reads
    assert abs(H.spend_pte(0, 0, 100, 1_000, 4_000) - 400) < 1e-9


def test_parallel_samples_share_the_decode_steps():
    lm = LatencyModel({1: 1.0e-3, 16: 1.4e-3, 64: 3.0e-3}, prefill_per_token=1e-5, verifier={1: 1e-3, 16: 2e-3})
    t1 = H.latency_parallel(lm, 12, 19, 1, False)
    t16 = H.latency_parallel(lm, 12, 19, 16, False)
    assert t16 < 2 * t1                        # 16 chains in one batch: 19 steps at batch 16, not 16 x 19 steps
    assert abs(t16 - (12e-5 + 19 * 1.4e-3)) < 1e-12
