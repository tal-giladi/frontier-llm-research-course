"""Lab 12.2 — policy-gradient estimators. Fill in the TODOs; run `pytest labs/module-12/lesson-02` to check."""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, baseline: str = "mean", scale: str = "group",
                     eps: float = 1e-6) -> torch.Tensor:
    """(P, G) rewards -> (P, G) advantages. Compute in float64, return in the input dtype.

    baseline: "none" (raw reward), "mean" (subtract the group mean), "loo" (subtract the mean of the
    other G-1 rewards). scale: "none", "group" (divide by the group's std, ddof=1, plus eps), "batch"
    (divide by the std of all P*G rewards, ddof=1, plus eps). The std is of the rewards, not of the
    advantages.
    """
    raise NotImplementedError("TODO 1: baselines and group normalisation")


def zero_variance(rewards: torch.Tensor) -> torch.Tensor:
    """(P,) bool: True for groups in which every response received the same reward."""
    raise NotImplementedError("TODO 2: zero-variance groups")


def kl_estimators(logp: torch.Tensor, ref_logp: torch.Tensor) -> dict:
    """Per-token k1, k2, k3 with delta = logp - ref_logp: k1 = delta, k2 = delta^2 / 2,
    k3 = exp(-delta) - 1 + delta. Gradients must flow through ``logp``."""
    raise NotImplementedError("TODO 3: KL estimators")


def expected_loss_gradient(logits: torch.Tensor, ref_logits: torch.Tensor, kind: str) -> torch.Tensor:
    """Exact E_{y ~ pi}[grad_logits k(y)] for one categorical distribution pi = softmax(logits), float64.

    Enumerate every outcome y: compute k(y) with your ``kl_estimators`` (logp = log_softmax(theta)),
    take its gradient with respect to theta, and weight it by pi(y) (detached: the sample is fixed, as in
    a sampled batch). ``kind`` is "k1", "k2" or "k3".
    """
    raise NotImplementedError("TODO 4: expected gradient by enumeration")


def entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy in nats of softmax(logits) along the last dimension, computed in float32."""
    raise NotImplementedError("TODO 5: entropy")


def decide(diff_mean: float, lo: float, hi: float) -> str:
    """The pre-stated rule: "better" if the interval of (arm - baseline) is above 0, "worse" if below 0,
    otherwise "inconclusive"."""
    raise NotImplementedError("TODO 6: decision rule")
