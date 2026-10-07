"""Train/inference log-probability mismatch and batch invariance (lesson 14.3).

The rollout engine and the trainer compute the "same" log-probabilities with different programs: cached
decoding vs one teacher-forced forward, other kernels, other precisions, other batch shapes. Their
difference makes nominally on-policy RL off-policy (Yao et al. 2025). :func:`mismatch_stats` summarises it
per batch; :func:`batch_dependence` measures how much one row's result changes with the batch it is computed
in; :func:`fixed_tile_matmul` is the simplest batch-invariant matmul: it always runs the same kernel on the
same tile shape, so a row's reduction order cannot depend on how many other rows there are (the idea of
He and Thinking Machines Lab's batch-invariant kernels, without their performance).
"""

from __future__ import annotations

import torch


def mismatch_stats(trainer_logp: torch.Tensor, sampler_logp: torch.Tensor, mask: torch.Tensor) -> dict:
    """Per-batch summary of the trainer-vs-sampler gap on sampled tokens (all (B, R), mask 1 on response).

    ``mean_abs`` and ``max_abs`` of the log-prob difference d = trainer - sampler; ``k3`` the mean of
    exp(-d) - 1 + d, an estimate of KL(sampler || trainer) per token from sampler draws (>= 0);
    ``max_ratio`` the largest trainer/sampler probability ratio, the weight truncated IS would cap;
    ``seq_logratio_absmax`` the largest |sum_t d| over sequences (what a sequence-level ratio sees).
    """
    m = mask.float()
    n = m.sum().clamp_min(1.0)
    d = (trainer_logp - sampler_logp).double() * m
    k3 = (torch.exp(-d) - 1 + d) * m
    return {"mean_abs": float(d.abs().sum() / n), "max_abs": float(d.abs().max()),
            "k3": float(k3.sum() / n), "max_ratio": float(torch.exp(d).max()),
            "seq_logratio_absmax": float(d.sum(-1).abs().max())}


def fixed_tile_matmul(x: torch.Tensor, w: torch.Tensor, tile: int = 64) -> torch.Tensor:
    """x (M, K) @ w (K, N), computed tile by tile with every tile padded to ``tile`` rows."""
    out = []
    for i in range(0, x.shape[0], tile):
        blk = x[i:i + tile]
        pad = tile - blk.shape[0]
        if pad:
            blk = torch.cat([blk, blk.new_zeros(pad, blk.shape[1])])
        out.append((blk @ w)[: tile - pad])
    return torch.cat(out)


def batch_dependence(fn, x: torch.Tensor, batch_sizes=(1, 3, 16, 64)) -> dict[int, float]:
    """max |fn(x[:b])[0] - fn(x)[0]| for each b: how much row 0's result depends on its batch."""
    ref = fn(x)[0]
    return {b: float((fn(x[:b])[0] - ref).abs().max()) for b in batch_sizes}
