"""Selection procedures over sampled answers, and the oracle bound they cannot beat (lesson 15.1).

A test-time-compute method draws several candidates for one problem and must *pick* one. The pick is
the procedure; the oracle that knows which candidate is right is not a procedure, it is an upper bound:

* **oracle pass@k** — at least one of k samples is correct (unbiased estimator, Chen et al. 2021 Eq. 1).
  This is coverage, not accuracy: nobody at inference time knows which sample is the right one.
* **majority vote** (self-consistency, Wang et al. 2022) — the most frequent final answer.
* **best-of-N** with a verifier — the candidate with the highest verifier score (Cobbe et al. 2021).
* **weighted vote** — sum the verifier scores of all candidates that give the same answer, pick the
  answer with the largest sum (used by Snell et al. 2024 and the Hugging Face test-time-compute study).

Ties are broken by first occurrence in the candidate order, never by correctness: a tie rule that looks
at the gold answer is a leak. ``None`` (no parsable answer) never wins a vote unless every candidate is
``None``.

Estimating a procedure's success at N from a pool of n >= N samples per problem: draw ``resamples``
random subsets of size N without replacement (a fixed seed), apply the procedure, average. Unlike pass@k
there is no closed form for majority vote; the subset average is unbiased for the success of the
procedure applied to N fresh samples, because a random N-subset of n i.i.d. samples is N i.i.d. samples.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from frontierlab.evals.suite_v2.core import pass_at_k


def majority_vote(answers):
    """The most frequent non-None answer; ties go to the answer that appeared first; None if all are None."""
    counts: "OrderedDict" = OrderedDict()
    for a in answers:
        if a is None:
            continue
        counts[a] = counts.get(a, 0) + 1
    if not counts:
        return None
    best, best_c = None, -1
    for a, c in counts.items():            # insertion order = first occurrence
        if c > best_c:
            best, best_c = a, c
    return best


def weighted_vote(answers, weights):
    """The non-None answer with the largest summed weight; ties to first occurrence."""
    tot: "OrderedDict" = OrderedDict()
    for a, w in zip(answers, weights):
        if a is None:
            continue
        tot[a] = tot.get(a, 0.0) + float(w)
    if not tot:
        return None
    best, best_w = None, -np.inf
    for a, w in tot.items():
        if w > best_w:
            best, best_w = a, w
    return best


def best_of_n(answers, scores):
    """The answer of the highest-scoring candidate (first one on a tie). A None answer can be selected."""
    s = np.asarray(scores, dtype=np.float64)
    return answers[int(np.argmax(s))]


PROCEDURES = {"majority": lambda a, s: majority_vote(a), "best_of_n": best_of_n, "weighted": weighted_vote}


def subset_success(answers, gold, N: int, method: str = "majority", scores=None, resamples: int = 200,
                   seed: int = 0) -> np.ndarray:
    """Per-problem success probability of ``method`` applied to N samples, estimated from a pool.

    ``answers``: list over problems of lists of n answers (hashable or None); ``gold``: list of gold answers;
    ``scores``: same shape as ``answers`` (verifier scores; needed by best_of_n and weighted).
    Returns an array (P,) in [0, 1]. With N == n there is one subset (the whole pool) and no resampling.
    """
    fn = PROCEDURES[method]
    rng = np.random.default_rng(seed)
    out = np.empty(len(answers))
    for i, (cand, g) in enumerate(zip(answers, gold)):
        n = len(cand)
        if N > n:
            raise ValueError(f"N = {N} > pool size {n}")
        sc = scores[i] if scores is not None else [0.0] * n
        reps = 1 if N == n else resamples
        hits = 0
        for _ in range(reps):
            idx = np.arange(n) if N == n else rng.choice(n, size=N, replace=False)
            pick = fn([cand[j] for j in idx], [sc[j] for j in idx])
            hits += int(pick is not None and pick == g)
        out[i] = hits / reps
    return out


def oracle_pass_at_k(correct, k: int) -> np.ndarray:
    """Per-problem unbiased pass@k from a boolean pool (P, n): coverage, the bound of any selection rule."""
    c = np.asarray(correct, dtype=bool)
    n = c.shape[1]
    return np.array([pass_at_k(n, int(row.sum()), k) for row in c])


def selection_gap(correct, answers, gold, Ns, method: str = "majority", scores=None, resamples: int = 200,
                  seed: int = 0) -> list[dict]:
    """For each N: oracle pass@N, the procedure's success, and the gap between them (mean over problems)."""
    rows = []
    for N in Ns:
        orc = oracle_pass_at_k(correct, N).mean()
        got = subset_success(answers, gold, N, method, scores, resamples, seed).mean()
        rows.append({"N": int(N), "oracle": float(orc), "selected": float(got), "gap": float(orc - got)})
    return rows
