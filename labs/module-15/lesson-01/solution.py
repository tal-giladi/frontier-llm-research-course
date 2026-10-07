"""Reference solution for lab 15.1."""

from __future__ import annotations

import numpy as np


def majority_vote(answers):
    counts = {}
    for a in answers:
        if a is not None:
            counts[a] = counts.get(a, 0) + 1          # dicts keep insertion order: first occurrence wins ties
    if not counts:
        return None
    return max(counts, key=lambda a: counts[a])         # max returns the first of equal keys


def weighted_vote(answers, weights):
    tot = {}
    for a, w in zip(answers, weights):
        if a is not None:
            tot[a] = tot.get(a, 0.0) + float(w)
    if not tot:
        return None
    return max(tot, key=lambda a: tot[a])


def policy_token_equivalents(prefill, decode, verifier_tokens, policy_params, verifier_params):
    return prefill + decode + verifier_params / policy_params * verifier_tokens


def subset_success(answers, gold, N, method, scores=None, resamples=200, seed=0):
    rng = np.random.default_rng(seed)
    out = np.empty(len(answers))
    for i, (cand, g) in enumerate(zip(answers, gold)):
        n = len(cand)
        sc = scores[i] if scores is not None else [0.0] * n
        reps = 1 if N == n else resamples
        hits = 0
        for _ in range(reps):
            idx = np.arange(n) if N == n else rng.choice(n, size=N, replace=False)
            a = [cand[j] for j in idx]
            s = [sc[j] for j in idx]
            if method == "majority":
                pick = majority_vote(a)
            elif method == "weighted":
                pick = weighted_vote(a, s)
            else:
                pick = a[int(np.argmax(s))]
            hits += int(pick is not None and pick == g)
        out[i] = hits / reps
    return out
