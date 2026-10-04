"""Reference solution for lab 05.2."""

from __future__ import annotations

import torch

from frontierlab.attention.base import causal_mask, register
from frontierlab.attention.dsa import DSAttention


def index_scores(iq, ik, w):
    dots = torch.einsum("bjtd,bsd->bjts", iq, ik).relu()             # (B, H_I, T, S)
    return torch.einsum("btj,bjts->bts", w, dots)


def select_topk(scores, allowed, k):
    masked = scores.masked_fill(~allowed, float("-inf"))
    idx = masked.topk(min(k, scores.shape[-1]), dim=-1).indices
    return torch.zeros_like(masked, dtype=torch.bool).scatter_(-1, idx, True) & allowed


def indexer_target(probs):
    with torch.no_grad():
        p = probs.sum(1)
        return p / p.sum(-1, keepdim=True)


def indexer_kl(p, scores, support):
    p = p.masked_fill(~support, 0.0)
    p = p / p.sum(-1, keepdim=True)
    logq = torch.log_softmax(scores.masked_fill(~support, float("-inf")), dim=-1)
    terms = torch.where(p > 0, p * (p.clamp_min(1e-30).log() - logq), torch.zeros_like(p))
    return terms.sum(-1).mean()


def trainable(name, stage):
    return stage == "sparse" or ".idx_" in name


@register("dsa-lab-solution")
class LabDSA(DSAttention):
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
