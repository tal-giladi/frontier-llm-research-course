"""Lab 07.5 — stability forensics. Fill in the TODOs; run `pytest labs/module-07/lesson-05`.

Inputs are plain Python lists read from a run's metrics.jsonl (loss, grad_norm per step) and stability.jsonl
(max_logit, ratio_max per step), as frontierlab.optim.stability writes them.
"""

from __future__ import annotations

import torch


def z_loss(logits: torch.Tensor, coef: float = 1e-4) -> torch.Tensor:
    """PaLM's auxiliary loss: coef * mean over positions of (logsumexp of the logits)^2. logits (B, T, V)."""
    raise NotImplementedError("TODO 1: z-loss")


def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    """Gemma 2's soft-capping: cap * tanh(x / cap)."""
    raise NotImplementedError("TODO 2: soft-capping")


def find_spikes(steps: list[int], losses: list[float], window: int = 20, k: float = 6.0, min_rel: float = 0.05) -> list[int]:
    """Steps whose loss exceeds the median of the previous ``window`` losses by more than
    max(k * 1.4826 * MAD, min_rel * median), MAD = median of |loss - median| over that window.
    Return the flagged steps (not merged)."""
    raise NotImplementedError("TODO 3: rolling-median spike detector")


def first_crossing(steps: list[int], values: list[float], threshold: float, before: int) -> int | None:
    """First step <= ``before`` at which ``values`` reached ``threshold``; None if it never did."""
    raise NotImplementedError("TODO 4: when did a statistic first cross its threshold")


def classify(logit_growth: float, max_logit_before: float, ratio_jump: float, recovered: bool) -> str:
    """The lesson's decision rules, in this order:
    "logit growth"  if logit_growth >= 4 and max_logit_before > 20
    "optimizer"     elif ratio_jump >= 4
    "data"          elif recovered
    "unclear"       otherwise
    """
    raise NotImplementedError("TODO 5: classify a spike from its evidence")
