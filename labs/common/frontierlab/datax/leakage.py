"""Benchmark leakage checks: n-gram overlap of evaluation items with the training data (lesson 10.1).

Parent course lesson 11.3 introduced n-gram contamination checks. Here they are run as part of the
data pipeline, against every evaluation the course uses: Eval v0 (held-out windows of Data-v0's
validation split, and LAMBADA), Eval v1 (its haystacks are validation documents) and any extra
benchmark file.

An :class:`NgramIndex` holds the sorted unique 64-bit hashes of every word n-gram of the training
data (13-grams by default; ~8 bytes per training word, e.g. about 140 MB for Data-v0's CPU slice).
For an evaluation item with word n-grams g_1..g_m, its **overlap** is the fraction of its n-grams
found in the index; an item is **flagged** when the overlap reaches a stated threshold. Definitions
differ between reports (which n, word or token n-grams, any-match or a fraction), so the report
stores ``n`` and the threshold next to every number. Phi-4 (appendix B.1), for example, decontaminated
with a hybrid of 13-gram and 7-gram matching against 19 benchmarks.

The checker's recall is tested with a **planted leak**: copies of a few benchmark items inserted into
a training corpus must all be flagged, and a clean corpus must flag (almost) none.
"""

from __future__ import annotations

import numpy as np

from frontierlab.datax.neardup import _GOLD, doc_words, mix64


def ngram_hashes(words: np.ndarray, n: int) -> np.ndarray:
    """(len(words) - n + 1,) uint64 hashes of all word n-grams (empty if the text is shorter than n)."""
    words = np.asarray(words, dtype=np.uint64)
    if words.size < n:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(words.size - n + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = mix64(h * _GOLD + words[j:words.size - n + 1 + j])
    return h


class NgramIndex:
    """Sorted unique n-gram hashes of a training corpus."""

    def __init__(self, n: int = 13):
        self.n = n
        self._parts: list[np.ndarray] = []
        self.hashes = np.zeros(0, dtype=np.uint64)
        self.documents = 0

    def add_texts(self, texts) -> "NgramIndex":
        for t in texts:
            self._parts.append(ngram_hashes(doc_words(t), self.n))
            self.documents += 1
            if len(self._parts) >= 2000:
                self._compact()
        return self

    def _compact(self):
        if self._parts:
            self.hashes = np.unique(np.concatenate([self.hashes, *self._parts]))
            self._parts = []

    def finish(self) -> "NgramIndex":
        self._compact()
        return self

    def contains(self, h: np.ndarray) -> np.ndarray:
        self._compact()
        if self.hashes.size == 0 or h.size == 0:
            return np.zeros(h.size, dtype=bool)
        pos = np.searchsorted(self.hashes, h)
        pos = np.minimum(pos, self.hashes.size - 1)
        return self.hashes[pos] == h

    def overlap(self, text: str) -> tuple[float, int]:
        """(fraction of the item's n-grams found in the index, number of n-grams in the item)."""
        h = ngram_hashes(doc_words(text), self.n)
        if h.size == 0:
            return 0.0, 0
        return float(self.contains(h).mean()), int(h.size)

    def nbytes(self) -> int:
        self._compact()
        return int(self.hashes.nbytes)


def leakage_report(index: NgramIndex, items: list[str], threshold: float = 0.5, name: str = "items",
                   keep: int = 5) -> dict:
    """Overlap of every item with the training index; flagged = overlap >= threshold.

    Items shorter than n words have no n-grams and are counted as ``too_short`` (they cannot be checked
    this way; report them rather than calling them clean).
    """
    ov, too_short = [], 0
    for t in items:
        f, m = index.overlap(t)
        if m == 0:
            too_short += 1
        ov.append(f)
    ov = np.asarray(ov)
    flagged = np.nonzero(ov >= threshold)[0]
    return {"name": name, "n": index.n, "threshold": threshold, "items": len(items), "too_short": too_short,
            "flagged": int(flagged.size), "flagged_rate": float(flagged.size / max(1, len(items))),
            "any_ngram_rate": float((ov > 0).mean()) if len(items) else 0.0,
            "mean_overlap": float(ov.mean()) if len(items) else 0.0,
            "flagged_indices": flagged[:keep].tolist(), "overlaps": ov.tolist()}


def plant(texts: list[str], items: list[str], every: int = 50) -> list[str]:
    """A copy of ``texts`` with each item appended to one training document (every ``every``-th one):
    the positive control for the checker."""
    out = list(texts)
    for k, item in enumerate(items):
        i = (k * every) % max(1, len(out))
        out[i] = out[i] + "\n\n" + item
    return out
