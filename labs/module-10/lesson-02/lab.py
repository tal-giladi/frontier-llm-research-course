"""Lab 10.2 — model-based quality filtering. Fill in the TODOs; run `pytest labs/module-10/lesson-02` to check."""

from __future__ import annotations

import numpy as np

from frontierlab.datax.neardup import word_hash
from frontierlab.datax.quality import _WORD


def bucket_features(text: str, buckets: int = 1 << 18, max_words: int = 2000) -> np.ndarray:
    """Sorted unique bucket ids of the lowercase unigrams and bigrams of the first ``max_words`` words.

    Exactly as ``frontierlab.datax.quality.features``: words = _WORD.findall(text.lower()); unigram hash
    h_i = word_hash(w_i) as uint64; bigram hash (h_i * 0x9E3779B97F4A7C15 + h_{i+1}) XOR 0x5BD1E995 (uint64,
    wrapping, inside np.errstate(over="ignore")); bucket = hash mod buckets. A text with no words -> [0].
    """
    raise NotImplementedError("TODO 1: hashed uni- and bigram features")


def linear_score(feats: np.ndarray, weights: np.ndarray, bias: float) -> float:
    """The classifier's score: bias + mean of weights[f] over the document's buckets (EmbeddingBag, mode='mean')."""
    raise NotImplementedError("TODO 2: the linear bag-of-n-grams score")


def f1_at(pred_scores, true_scores, threshold: float = 3.0) -> float:
    """F1 of "score >= threshold" (predicted) against "annotator score >= threshold" (true)."""
    raise NotImplementedError("TODO 3: F1 at the FineWeb-Edu threshold")


def keep_top(scores, frac: float) -> np.ndarray:
    """Indices of the top ``frac`` of documents by score (at least one; ties by lower index), sorted ascending."""
    raise NotImplementedError("TODO 4: the filter")


def epochs_needed(budget_tokens: float, kept_tokens: float) -> float:
    """How many passes over the kept data a training budget needs."""
    raise NotImplementedError("TODO 5: repetition at a fixed budget")


def decide(ci: tuple[float, float], margin: float = 0.0) -> str:
    """For (filtered − unfiltered) held-out loss: "adopt" if the upper bound < −margin, "reject" if the lower
    bound > −margin, else "inconclusive"."""
    raise NotImplementedError("TODO 6: the pre-stated decision rule")
