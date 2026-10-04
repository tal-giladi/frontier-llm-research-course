"""Lab 05.2 — a DeepSeek-Sparse-Attention-style indexer, top-k selection and the indexer's objective.

Fill in the TODOs; run `pytest labs/module-05/lesson-02`. The provided ``LabDSA`` attention calls your
functions; everything else (projections, RoPE, cache) is frontierlab's ``dsa`` kind.
"""

from __future__ import annotations

import torch

from frontierlab.attention.base import causal_mask, register
from frontierlab.attention.dsa import DSAttention


def index_scores(iq: torch.Tensor, ik: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """I[b, t, s] = sum_j w[b, t, j] * ReLU(iq[b, j, t] . ik[b, s]).

    iq (B, H_I, T, d_I) indexer queries; ik (B, S, d_I) indexer keys (one per token, shared by the heads);
    w (B, T, H_I) per-head weights. Returns (B, T, S).
    """
    raise NotImplementedError("TODO 1: lightning-indexer scores")


def select_topk(scores: torch.Tensor, allowed: torch.Tensor, k: int) -> torch.Tensor:
    """Boolean (B, T, S): for each query, the k allowed keys with the highest scores (all allowed keys
    if fewer than k are allowed). ``allowed`` (1 or B, T, S) is the causal mask. Never select a
    disallowed key, even when a row has fewer than k allowed keys."""
    raise NotImplementedError("TODO 2: top-k selection")


def indexer_target(probs: torch.Tensor) -> torch.Tensor:
    """probs (B, H, T, S): main attention probabilities. Sum over heads, then L1-normalise over keys
    (V3.2 section 2.1.1). Returns (B, T, S) with every row summing to 1. No gradient should flow."""
    raise NotImplementedError("TODO 3: the indexer's training target")


def indexer_kl(p: torch.Tensor, scores: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Mean over queries of KL(p || Softmax(scores)), both restricted to ``support`` (B, T, S) bool and
    renormalised over it. Terms with p = 0 contribute 0 (0 * log 0 = 0)."""
    raise NotImplementedError("TODO 4: KL divergence restricted to a support")


def trainable(name: str, stage: str) -> bool:
    """Does parameter ``name`` (e.g. "model.layers.0.self_attn.idx_q.weight", "lm_head.weight") get
    gradients in ``stage`` ("warmup" or "sparse")? Indexer parameters contain ".idx_"."""
    raise NotImplementedError("TODO 5: which parameters train in each stage")


@register("dsa-lab")
class LabDSA(DSAttention):
    """frontierlab's dsa kind with the indexer arithmetic replaced by yours (mask path only)."""

    def forward(self, x, positions, cache=None):
        B, T, _ = x.shape
        q, k, v = self.project(x, positions)
        iq, ik, w = self.indexer(x, positions)
        k_pos = positions
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat((cache["k"], k), 2), torch.cat((cache["v"], v), 2)
                ik = torch.cat((cache["idx_k"], ik), 1)
                k_pos = torch.cat((cache["pos"], positions))
            cache["k"], cache["v"], cache["idx_k"], cache["pos"] = k, v, ik, k_pos
        scores = index_scores(iq, ik, w)
        allowed = causal_mask(positions, k_pos)[None]
        sel = allowed.expand_as(scores) if self.mode == "dense" else select_topk(scores, allowed, self.topk)
        y, p = self.attend_masked(q, k, v, sel)
        if self.collect:
            self.indexer_loss = indexer_kl(indexer_target(p), scores, sel)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.hd))
