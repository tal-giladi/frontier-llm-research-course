import numpy as np

from frontierlab.labkit import load_target
from frontierlab.ttc import select as SE

lab = load_target(__file__)


def test_majority_vote():
    assert lab.majority_vote(["3", "5", "5", "3"]) == "3"          # tie: first occurrence, not correctness
    assert lab.majority_vote(["5", "3", "3"]) == "3"
    assert lab.majority_vote([None, None, "7"]) == "7"
    assert lab.majority_vote([None]) is None
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = [None if rng.random() < 0.2 else str(rng.integers(0, 4)) for _ in range(int(rng.integers(1, 9)))]
        assert lab.majority_vote(a) == SE.majority_vote(a)


def test_weighted_vote():
    assert lab.weighted_vote(["1", "2", "2"], [0.9, 0.4, 0.4]) == "1"
    assert lab.weighted_vote(["1", "2", "2"], [0.7, 0.4, 0.4]) == "2"
    assert lab.weighted_vote([None, "4"], [5.0, 0.1]) == "4"
    rng = np.random.default_rng(1)
    for _ in range(200):
        a = [None if rng.random() < 0.2 else str(rng.integers(0, 3)) for _ in range(6)]
        w = rng.random(6).round(1).tolist()
        assert lab.weighted_vote(a, w) == SE.weighted_vote(a, w)


def test_policy_token_equivalents():
    # 4 samples of 20 tokens, prompt 12 prefilled once, a verifier of the same size reading 32 tokens each
    assert lab.policy_token_equivalents(12, 80, 128, 1000, 1000) == 12 + 80 + 128
    # a verifier 4x the policy (7B PRM on a 1.7B policy) makes every verifier token cost 4 policy tokens
    assert lab.policy_token_equivalents(10, 100, 50, 1_700, 6_800) == 10 + 100 + 200


def test_subset_success_matches_reference():
    rng = np.random.default_rng(2)
    gold = [str(g) for g in rng.integers(0, 3, 20)]
    answers = [[str(rng.integers(0, 3)) if rng.random() > 0.1 else None for _ in range(8)] for _ in gold]
    scores = [rng.random(8).tolist() for _ in gold]
    for method in ("majority", "best_of_n", "weighted"):
        for N in (1, 3, 8):
            mine = lab.subset_success(answers, gold, N, method, scores, resamples=50, seed=3)
            ref = SE.subset_success(answers, gold, N, method, scores, resamples=50, seed=3)
            assert np.allclose(mine, ref), (method, N)
