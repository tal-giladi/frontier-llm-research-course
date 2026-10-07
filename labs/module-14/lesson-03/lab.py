"""Lab 14.3 — rollout systems and staleness. Fill in the TODOs; run `pytest labs/module-14/lesson-03` to check.

``async_lab.py`` uses your functions for the schedule model, the mismatch measurements and the batch-invariance
probe; the training runs use the course loop with a bounded-staleness sampler.
"""

from __future__ import annotations

import numpy as np
import torch


def simulate(gen_times, train_time: float, bound: int) -> dict:
    """The bounded-staleness schedule of the lesson, for len(gen_times) updates.

    Batch j (0-based) is generated starting at s_j = max(end of generation j-1, end of update j-bound-1)
    (the second term only when j - bound >= 1; otherwise 0), takes gen_times[j], and is generated with
    version v_j = number of updates that have *ended* at or before s_j. Update j starts at
    max(end of generation j, end of update j-1) and takes train_time. Lag of batch j = j - v_j.

    Return {"total_s": end of the last update, "lags": [lag per batch], "max_lag": int, "mean_lag": float}.
    """
    raise NotImplementedError("TODO 1: the bounded-staleness schedule")


def mismatch_stats(trainer_logp: torch.Tensor, sampler_logp: torch.Tensor, mask: torch.Tensor) -> dict:
    """Over masked tokens, with d = trainer_logp - sampler_logp (computed in float64):
    "mean_abs" = mean |d|; "max_abs" = max |d|; "k3" = mean of exp(-d) - 1 + d; "max_ratio" = max exp(d)
    over masked tokens; "seq_logratio_absmax" = max over sequences of |sum over the sequence's masked tokens of d|."""
    raise NotImplementedError("TODO 2: mismatch statistics")


def fixed_tile_matmul(x: torch.Tensor, w: torch.Tensor, tile: int = 64) -> torch.Tensor:
    """x (M, K) @ w (K, N) computed in row tiles of exactly ``tile`` rows (pad the last tile with zero rows,
    then drop the padded rows), so that every row goes through the same kernel shape whatever M is."""
    raise NotImplementedError("TODO 3: a batch-invariant matmul")
