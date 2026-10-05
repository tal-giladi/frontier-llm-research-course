"""Reference solution for lab 10.1 — data integrity at scale."""

from __future__ import annotations

import numpy as np
import torch

from frontierlab.datax.neardup import mix64


def segment_ids(doc_starts, start, T):
    pos = np.arange(start, start + T)
    doc = np.searchsorted(doc_starts, pos, side="right") - 1
    return (doc - doc[0]).astype(np.int64)


def docmask(q_pos, k_pos, seg):
    causal = k_pos[None, :] <= q_pos[:, None]
    same = seg[:, q_pos][:, :, None] == seg[:, k_pos][:, None, :]
    return (causal[None] & same)[:, None]


def window_counts(weights, block):
    w = np.asarray(weights, dtype=np.float64)
    exact = w / w.sum() * block
    n = np.floor(exact).astype(np.int64)
    rest = block - int(n.sum())
    order = np.argsort(-(exact - n), kind="stable")
    n[order[:rest]] += 1
    return n


def locate(k, block, counts, block_order):
    b, j = divmod(k, block)
    order = block_order(b)
    s = int(order[j])
    return s, b * int(counts[s]) + int((order[:j] == s).sum())


def minhash_signature(shingles, salts):
    if shingles.size == 0:
        return np.full(salts.size, np.iinfo(np.uint64).max, dtype=np.uint64)
    return mix64(shingles[None, :] ^ salts[:, None]).min(axis=1)


def band_candidates(sigs, groups, bands):
    D, P = sigs.shape
    r = P // bands
    pairs = set()
    for b in range(bands):
        buckets = {}
        for i in range(D):
            buckets.setdefault(sigs[i, b * r:(b + 1) * r].tobytes(), []).append(i)
        for rows in buckets.values():
            for x in range(len(rows)):
                for y in range(x + 1, len(rows)):
                    i, j = rows[x], rows[y]
                    if groups[i] != groups[j]:
                        pairs.add((min(i, j), max(i, j)))
    return pairs


def ngram_overlap(item_hashes, index_hashes):
    if item_hashes.size == 0:
        return 0.0
    pos = np.minimum(np.searchsorted(index_hashes, item_hashes), index_hashes.size - 1)
    return float((index_hashes[pos] == item_hashes).mean())


def decide_noninferior(ci, margin):
    lo, hi = ci
    if hi <= margin:
        return "adopt"
    if lo > margin:
        return "reject"
    return "inconclusive"


__all__ = ["segment_ids", "docmask", "window_counts", "locate", "minhash_signature", "band_candidates",
           "ngram_overlap", "decide_noninferior", "torch"]
