"""A small model-based quality classifier: hashed bag of word n-grams, linear, pure PyTorch (lesson 10.2).

What it is. DCLM's filter is a fastText bigram classifier (DCLM section 4.4): a linear model over
hashed word uni- and bigrams. FineWeb-Edu's is a linear regression head on a frozen embedding model,
trained on Llama-3-70B-Instruct's 0–5 educational scores and thresholded at 3 (FineWeb section 4).
This module is the fastText-shaped half of that design without fastText: each document becomes the
set of its hashed uni- and bigrams, ``torch.nn.EmbeddingBag(buckets, 1, mode="mean")`` averages one
learned weight per bucket, and a bias is added:

    score(doc) = b + (1/|G|) · Σ_{g in G(doc)} w[hash(g) mod buckets]

trained either as **regression** on the annotator's 0–5 score (mean squared error, FineWeb-Edu's
target) or as **binary** logistic regression (DCLM's positive-vs-random setup). AdamW, mini-batches,
a few epochs; weight decay is the L2 penalty.

What it is not. Its validation accuracy says how well it imitates the annotator. Whether filtering by
it gives better *training data* is a different question that only a matched training ablation answers
(lesson 10.2's lab): same token budget, same model, same seeds, arms differing only in which documents
were kept.
"""

from __future__ import annotations

import math
import re

import numpy as np
import torch
import torch.nn as nn

from frontierlab.datax.neardup import word_hash

_WORD = re.compile(r"\w+", re.UNICODE)


def features(text: str, buckets: int = 1 << 18, max_words: int = 2000) -> np.ndarray:
    """Unique bucket ids of the lowercase unigrams and bigrams of the first ``max_words`` words."""
    words = _WORD.findall(text.lower())[:max_words]
    h = np.fromiter((word_hash(w) for w in words), dtype=np.uint64, count=len(words))
    if h.size == 0:
        return np.zeros(1, dtype=np.int64)
    with np.errstate(over="ignore"):
        bi = (h[:-1] * np.uint64(0x9E3779B97F4A7C15) + h[1:]) if h.size > 1 else np.zeros(0, dtype=np.uint64)
        allh = np.concatenate([h, bi ^ np.uint64(0x5BD1E995)])
    return np.unique((allh % np.uint64(buckets)).astype(np.int64))


class BagClassifier(nn.Module):
    def __init__(self, buckets: int = 1 << 18):
        super().__init__()
        self.buckets = buckets
        self.bag = nn.EmbeddingBag(buckets, 1, mode="mean")
        nn.init.zeros_(self.bag.weight)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, flat: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.bag(flat, offsets).squeeze(-1) + self.bias


def _pack(feats: list[np.ndarray]):
    lens = np.array([f.size for f in feats])
    offsets = np.concatenate([[0], np.cumsum(lens)[:-1]])
    return torch.from_numpy(np.concatenate(feats)), torch.from_numpy(offsets.astype(np.int64))


def fit(texts: list[str], targets, mode: str = "regression", buckets: int = 1 << 18, epochs: int = 6,
        lr: float = 0.5, weight_decay: float = 1e-6, batch: int = 256, seed: int = 0,
        feats: list[np.ndarray] | None = None) -> BagClassifier:
    """Train the classifier. ``mode`` "regression" (targets 0–5, MSE) or "binary" (targets 0/1, logistic)."""
    torch.manual_seed(seed)
    feats = feats if feats is not None else [features(t, buckets) for t in texts]
    y = torch.as_tensor(np.asarray(targets, dtype=np.float32))
    model = BagClassifier(buckets)
    if mode == "regression":
        with torch.no_grad():
            model.bias.fill_(float(y.mean()))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    g = torch.Generator().manual_seed(seed)
    loss_fn = nn.MSELoss() if mode == "regression" else nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        perm = torch.randperm(len(feats), generator=g).tolist()
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            flat, off = _pack([feats[j] for j in idx])
            loss = loss_fn(model(flat, off), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.mode = mode
    return model


@torch.no_grad()
def predict(model: BagClassifier, texts: list[str] | None = None, feats: list[np.ndarray] | None = None,
            batch: int = 1024) -> np.ndarray:
    """Scores: predicted 0–5 score (regression) or logit (binary)."""
    feats = feats if feats is not None else [features(t, model.buckets) for t in texts]
    out = []
    for i in range(0, len(feats), batch):
        flat, off = _pack(feats[i:i + batch])
        out.append(model(flat, off).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def binary_metrics(pred_pos: np.ndarray, true_pos: np.ndarray) -> dict:
    """Precision, recall, F1 and accuracy of boolean predictions."""
    p, t = np.asarray(pred_pos, bool), np.asarray(true_pos, bool)
    tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(1e-12, prec + rec),
            "accuracy": float((p == t).mean()), "positives_true": int(t.sum()), "positives_pred": int(p.sum())}


def spearman(a, b) -> float:
    """Spearman rank correlation (average ranks for ties), NumPy only."""
    def ranks(x):
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x, kind="mergesort")
        r = np.empty_like(x)
        r[order] = np.arange(x.size)
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=r)
        return sums[inv] / cnt[inv]
    ra, rb = ranks(a), ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den else float("nan")


def top_fraction(scores: np.ndarray, frac: float) -> np.ndarray:
    """Indices of the top ``frac`` of documents by score (ties broken by index), in original order."""
    k = max(1, int(round(frac * len(scores))))
    order = np.lexsort((np.arange(len(scores)), -np.asarray(scores)))
    return np.sort(order[:k])
