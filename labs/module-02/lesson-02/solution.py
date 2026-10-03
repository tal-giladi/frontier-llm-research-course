"""Reference solution for lab 02.2."""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


def ce_and_grads(h, w, targets, chunk_size):
    N = h.shape[0]
    loss = h.new_zeros(())
    dh = torch.zeros_like(h)
    dw = torch.zeros_like(w)
    for s in range(0, N, chunk_size):
        hc, tc = h[s:s + chunk_size], targets[s:s + chunk_size]
        z = hc @ w.t()                                    # (n, V): only this chunk's logits exist
        lse = torch.logsumexp(z, dim=-1)
        loss = loss + (lse - z.gather(1, tc[:, None]).squeeze(1)).sum()
        g = torch.softmax(z, dim=-1)
        g[torch.arange(len(tc)), tc] -= 1.0               # dL/dz = softmax - onehot (before the 1/N)
        g /= N
        dh[s:s + chunk_size] = g @ w
        dw += g.t() @ hc
    return loss / N, dh, dw


def checkpointed_loss(model, idx):
    B, T = idx.shape
    positions = torch.arange(T, device=idx.device)
    x = model.model.embed_tokens(idx)
    for layer in model.model.layers:
        x = checkpoint(layer, x, positions, None, use_reentrant=False)
    logits = model.lm_head(model.model.norm(x)).float()
    return torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), idx[:, 1:].reshape(-1))


CATEGORIES = {
    "matmul": ("mm", "addmm", "bmm", "matmul", "linear", "gemm"),
    "attention": ("attention", "sdpa", "flash"),
    "loss": ("log_softmax", "nll_loss", "cross_entropy"),
    "optimizer": ("adam", "_foreach", "lerp", "addcdiv", "addcmul"),
}


def categorize(top):
    shares = {k: 0.0 for k in (*CATEGORIES, "other")}
    for name, _ms, _calls, share in top:
        low = name.lower()
        for cat, keys in CATEGORIES.items():
            if any(k in low for k in keys):
                shares[cat] += share
                break
        else:
            shares["other"] += share
    return shares
