"""Multilingual and code data: tokenizer fertility, per-language thresholds, rehydration weights and
group-level (repository / task-family) splits (lesson 10.6, extension).

* :func:`fertility` — tokens per UTF-8 byte and per word of a tokenizer on a set of texts: the cost, in
  sequence length and compute, of a tokenizer that was not trained on a language.
* :func:`quantile_threshold` — FineWeb2's "Quantile" method (section 4.4.2): pick a language's threshold
  so that it removes the same fraction of that language's documents as the English threshold removes of
  English. :func:`removal_rate` measures what a threshold removes.
* :func:`rehydration_weights` — FineWeb2's duplication-aware upsampling (section 4.5): weight 10 for the
  MinHash cluster size with the lowest filter-removal rate, 1 for every cluster size whose removal rate is
  above the global rate, linear interpolation in between.
* :func:`group_split` — Data-v0's hash rule applied to a *group key* (repository, model directory, task
  family) instead of the document, so every member of a group lands in the same split. A file-level split
  of code puts near-copies of a file (vendored code, "Copied from" blocks, the same bug in two task
  instances) on both sides; :func:`cross_split_rate` measures it with MinHash.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from frontierlab.datax.neardup import cross_split_near_dups

_WORD = re.compile(r"\w+", re.UNICODE)


def fertility(encode, texts: list[str]) -> dict:
    """``encode(str) -> list[int]``; returns tokens per byte, per word and per character over ``texts``."""
    toks = sum(len(encode(t)) for t in texts)
    nbytes = sum(len(t.encode("utf-8")) for t in texts)
    words = sum(len(_WORD.findall(t)) for t in texts)
    chars = sum(len(t) for t in texts)
    return {"tokens": toks, "bytes": nbytes, "words": words, "tokens_per_byte": toks / max(1, nbytes),
            "tokens_per_word": toks / max(1, words), "bytes_per_char": nbytes / max(1, chars)}


def mean_word_length(text: str) -> float:
    w = _WORD.findall(text)
    return float(np.mean([len(x) for x in w])) if w else 0.0


def removal_rate(values, lo: float | None = None, hi: float | None = None) -> float:
    """Fraction of documents whose metric is outside [lo, hi]."""
    v = np.asarray(values, dtype=np.float64)
    bad = np.zeros(v.size, dtype=bool)
    if lo is not None:
        bad |= v < lo
    if hi is not None:
        bad |= v > hi
    return float(bad.mean()) if v.size else 0.0


def quantile_threshold(reference_values, reference_threshold: float, target_values, side: str = "low") -> float:
    """Threshold for the target language removing the same fraction as ``reference_threshold`` removes in the
    reference (English) distribution. ``side="low"`` removes values below the threshold, "high" above."""
    ref = np.asarray(reference_values, dtype=np.float64)
    tgt = np.asarray(target_values, dtype=np.float64)
    q = float((ref < reference_threshold).mean()) if side == "low" else float((ref > reference_threshold).mean())
    return float(np.quantile(tgt, q if side == "low" else 1 - q))


def rehydration_weights(cluster_sizes, removed, max_weight: float = 10.0) -> dict:
    """Upsampling weight per cluster size from the filter's removal rate per cluster size (FineWeb2 4.5).

    ``cluster_sizes[i]`` is the MinHash cluster size recorded for document i, ``removed[i]`` whether the
    quality filter removed it. Returns {size: weight}: ``max_weight`` at the size with the lowest removal
    rate, 1 for sizes whose rate is above the global rate, linear in the removal rate in between.
    """
    cs = np.asarray(cluster_sizes)
    rm = np.asarray(removed, dtype=bool)
    glob = float(rm.mean())
    rates = {int(s): float(rm[cs == s].mean()) for s in np.unique(cs)}
    best = min(rates.values())
    out = {}
    for s, r in rates.items():
        if r >= glob or glob <= best:
            out[s] = 1.0
        else:
            out[s] = 1.0 + (max_weight - 1.0) * (glob - r) / (glob - best)
    return {"global_removal_rate": glob, "removal_rate": rates, "weights": out}


def group_split(keys, val_per_mille: int = 100, test_per_mille: int = 100, salt: str = "") -> list[str]:
    """Split by the SHA-1 of a group key: every member of a group gets the group's split."""
    out = []
    for k in keys:
        b = int(hashlib.sha1((salt + str(k)).encode("utf-8")).hexdigest()[:8], 16) % 1000
        out.append("val" if b < val_per_mille else "test" if b < val_per_mille + test_per_mille else "train")
    return out


def cross_split_rate(texts: list[str], splits: list[str], threshold: float = 0.7, n: int = 5) -> dict:
    """Near-duplicate pairs (exact Jaccard >= threshold) between train and the held-out splits for a split
    assignment, and the share of held-out items that have a near-copy in train."""
    groups: dict[str, list[str]] = {}
    index: dict[str, list[int]] = {}
    for i, (t, s) in enumerate(zip(texts, splits)):
        groups.setdefault(s, []).append(t)
        index.setdefault(s, []).append(i)
    res = cross_split_near_dups(groups, threshold=threshold, n=n)
    pairs = [p for p in res["pairs"] if "train" in (p["a"], p["b"])]
    held = {s for s in groups if s != "train"}
    flagged = {(p["a"] if p["a"] != "train" else p["b"], p["a_index"] if p["a"] != "train" else p["b_index"]) for p in pairs}
    n_held = sum(len(groups[s]) for s in held)
    return {"pairs_with_train": len(pairs), "heldout_items": n_held, "heldout_with_train_near_copy": len(flagged),
            "rate": len(flagged) / max(1, n_held), "threshold": threshold}
