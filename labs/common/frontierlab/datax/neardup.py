"""Near-duplicate detection *across* train / validation / test splits with MinHash and LSH (lesson 10.1).

Data-v0's hash split puts exact copies of a document in the same split, so exact duplicates cannot
leak. A near-duplicate (the same page with a different date, a mirrored article with another header)
has a different hash and lands in a split of its own: it can sit in validation while its twin is in
training, and the validation loss then measures memorisation. MinHash (parent course lesson 11.2)
estimates the Jaccard similarity of two documents' shingle sets from short signatures; LSH banding
turns "all pairs" into "pairs that share a band bucket". Here only candidate pairs from *different*
splits are kept, then verified with the exact Jaccard similarity.

Definitions (symbols as in the lesson):

* words: lowercase ``\\w+`` tokens of the text, each hashed to 64 bits (:func:`doc_words`);
* shingles: hashes of every ``n`` consecutive words (:func:`shingles`, n = 5 by default);
* signature: for ``P`` hash functions h_p, ``sig[p] = min over shingles x of h_p(x)``; then
  ``Pr[sig_a[p] == sig_b[p]] = J(a, b)``, so the fraction of equal entries estimates the Jaccard J;
* LSH: ``b`` bands of ``r`` rows (``P = b·r``); two documents are candidates if any band is equal,
  with probability ``1 - (1 - J^r)^b`` (an S-curve with its midpoint near ``(1/b)^(1/r)``).

All hashing is 64-bit integer arithmetic in NumPy (wrapping multiplication), deterministic across
runs and platforms.
"""

from __future__ import annotations

import re
import zlib
from collections import defaultdict

import numpy as np

_WORD = re.compile(r"\w+", re.UNICODE)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_GOLD = np.uint64(0x9E3779B97F4A7C15)


def mix64(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser: a bijective scramble of uint64 values (vectorised)."""
    x = np.asarray(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * _M1
        x = (x ^ (x >> np.uint64(27))) * _M2
        return x ^ (x >> np.uint64(31))


def word_hash(w: str) -> int:
    b = w.encode("utf-8")
    return (zlib.crc32(b) << 32) | zlib.adler32(b)


def doc_words(text: str) -> np.ndarray:
    """(n_words,) uint64 hashes of the lowercase words of ``text``."""
    return np.fromiter((word_hash(w) for w in _WORD.findall(text.lower())), dtype=np.uint64)


def shingles(words: np.ndarray, n: int = 5) -> np.ndarray:
    """Unique uint64 hashes of all n-word windows (one shingle of the whole text if it is shorter)."""
    words = np.asarray(words, dtype=np.uint64)
    if words.size == 0:
        return np.zeros(0, dtype=np.uint64)
    if words.size < n:
        n = words.size
    h = np.zeros(words.size - n + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = mix64(h * _GOLD + words[j:words.size - n + 1 + j])
    return np.unique(h)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Exact Jaccard similarity of two shingle sets (sorted unique arrays)."""
    if a.size == 0 and b.size == 0:
        return 1.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / (a.size + b.size - inter)


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.salts = rng.integers(0, 2**63, size=num_perm, dtype=np.int64).astype(np.uint64)
        self.num_perm = num_perm

    def signature(self, sh: np.ndarray) -> np.ndarray:
        """(P,) uint64: min over shingles of h_p(x) = mix64(x xor salt_p)."""
        if sh.size == 0:
            return np.full(self.num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
        return mix64(sh[None, :] ^ self.salts[:, None]).min(axis=1)


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float((sig_a == sig_b).mean())


def candidate_probability(J: float, bands: int, rows: int) -> float:
    """Probability that two documents with Jaccard J share at least one LSH band: 1 - (1 - J^r)^b."""
    return 1.0 - (1.0 - J ** rows) ** bands


def lsh_buckets(sigs: np.ndarray, bands: int) -> list[dict]:
    """One dict per band: band bytes -> list of row indices."""
    D, P = sigs.shape
    if P % bands:
        raise ValueError("num_perm must be a multiple of bands")
    r = P // bands
    out = []
    for b in range(bands):
        buckets = defaultdict(list)
        block = np.ascontiguousarray(sigs[:, b * r:(b + 1) * r])
        for i in range(D):
            buckets[block[i].tobytes()].append(i)
        out.append(buckets)
    return out


def cross_split_pairs(groups: list[int], sigs: np.ndarray, bands: int = 16) -> set[tuple[int, int]]:
    """Candidate pairs (i < j) whose rows share a band bucket and whose ``groups`` differ."""
    pairs = set()
    for buckets in lsh_buckets(sigs, bands):
        for rows in buckets.values():
            if len(rows) < 2:
                continue
            for x in range(len(rows)):
                for y in range(x + 1, len(rows)):
                    i, j = rows[x], rows[y]
                    if groups[i] != groups[j]:
                        pairs.add((min(i, j), max(i, j)))
    return pairs


def cross_split_near_dups(docs: dict[str, list[str]], threshold: float = 0.8, n: int = 5, num_perm: int = 128,
                          bands: int = 16, seed: int = 0) -> dict:
    """Near-duplicate pairs between different splits (or sources), verified by exact Jaccard >= threshold.

    ``docs`` maps a split name to its list of texts. Returns the verified pairs (split, index, split,
    index, exact and estimated Jaccard), the number of candidates checked, and per split the number of
    documents that have a near-duplicate in another split.
    """
    hasher = MinHasher(num_perm, seed)
    names, idx, sh_all, sigs = [], [], [], []
    for name, texts in docs.items():
        for i, t in enumerate(texts):
            sh = shingles(doc_words(t), n)
            names.append(name)
            idx.append(i)
            sh_all.append(sh)
            sigs.append(hasher.signature(sh))
    sigs = np.stack(sigs) if sigs else np.zeros((0, num_perm), dtype=np.uint64)
    cands = cross_split_pairs(names, sigs, bands)
    pairs = []
    for i, j in sorted(cands):
        J = jaccard(sh_all[i], sh_all[j])
        if J >= threshold:
            pairs.append({"a": names[i], "a_index": idx[i], "b": names[j], "b_index": idx[j], "jaccard": J,
                          "jaccard_est": estimate_jaccard(sigs[i], sigs[j])})
    flagged = defaultdict(set)
    for p in pairs:
        flagged[p["a"]].add(p["a_index"])
        flagged[p["b"]].add(p["b_index"])
    return {"threshold": threshold, "n": n, "num_perm": num_perm, "bands": bands, "candidates": len(cands),
            "pairs": pairs, "documents": {k: len(v) for k, v in docs.items()},
            "with_cross_split_near_dup": {k: len(flagged.get(k, ())) for k in docs}}


def decode_docs(data, tokenizer, max_docs: int | None = None) -> list[str]:
    """Texts of the documents of a ``TokenData`` split (byte-level BPE decodes back to the original text)."""
    starts = data.doc_starts
    ends = np.append(starts[1:], len(data.tokens))
    k = len(starts) if max_docs is None else min(max_docs, len(starts))
    ids = [data.tokens[int(s):int(e) - 1].astype(np.int64).tolist() for s, e in zip(starts[:k], ends[:k])]
    return tokenizer.decode_batch(ids)
