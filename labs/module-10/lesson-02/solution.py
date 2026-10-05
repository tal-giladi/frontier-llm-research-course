"""Reference solution for lab 10.2 — model-based quality filtering."""

from __future__ import annotations

import numpy as np

from frontierlab.datax.neardup import word_hash
from frontierlab.datax.quality import _WORD


def bucket_features(text, buckets=1 << 18, max_words=2000):
    words = _WORD.findall(text.lower())[:max_words]
    h = np.fromiter((word_hash(w) for w in words), dtype=np.uint64, count=len(words))
    if h.size == 0:
        return np.zeros(1, dtype=np.int64)
    with np.errstate(over="ignore"):
        bi = (h[:-1] * np.uint64(0x9E3779B97F4A7C15) + h[1:]) if h.size > 1 else np.zeros(0, dtype=np.uint64)
        allh = np.concatenate([h, bi ^ np.uint64(0x5BD1E995)])
    return np.unique((allh % np.uint64(buckets)).astype(np.int64))


def linear_score(feats, weights, bias):
    return float(bias + weights[feats].mean())


def f1_at(pred_scores, true_scores, threshold=3.0):
    p = np.asarray(pred_scores) >= threshold
    t = np.asarray(true_scores) >= threshold
    tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return 2 * prec * rec / max(1e-12, prec + rec)


def keep_top(scores, frac):
    k = max(1, int(round(frac * len(scores))))
    order = np.lexsort((np.arange(len(scores)), -np.asarray(scores)))
    return np.sort(order[:k])


def epochs_needed(budget_tokens, kept_tokens):
    return budget_tokens / kept_tokens


def decide(ci, margin=0.0):
    lo, hi = ci
    if hi < -margin:
        return "adopt"
    if lo > -margin:
        return "reject"
    return "inconclusive"
