"""Lab 12.2 — policy-gradient estimators. Reference solution."""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, baseline: str = "mean", scale: str = "group",
                     eps: float = 1e-6) -> torch.Tensor:
    r = rewards.double()
    G = r.shape[1]
    if baseline == "none":
        adv = r.clone()
    elif baseline == "mean":
        adv = r - r.mean(1, keepdim=True)
    elif baseline == "loo":
        adv = r - (r.sum(1, keepdim=True) - r) / (G - 1)
    else:
        raise ValueError(baseline)
    if scale == "group":
        adv = adv / (r.std(1, keepdim=True) + eps)
    elif scale == "batch":
        adv = adv / (r.std() + eps)
    elif scale != "none":
        raise ValueError(scale)
    return adv.to(rewards.dtype)


def zero_variance(rewards: torch.Tensor) -> torch.Tensor:
    return (rewards == rewards[:, :1]).all(1)


def kl_estimators(logp: torch.Tensor, ref_logp: torch.Tensor) -> dict:
    d = logp - ref_logp
    return {"k1": d, "k2": 0.5 * d * d, "k3": torch.exp(-d) - 1.0 + d}


def expected_loss_gradient(logits: torch.Tensor, ref_logits: torch.Tensor, kind: str) -> torch.Tensor:
    theta = logits.detach().double().requires_grad_(True)
    ref = torch.log_softmax(ref_logits.detach().double(), -1)
    logp = torch.log_softmax(theta, -1)
    p = logp.exp().detach()
    k = kl_estimators(logp, ref)[kind]
    g = torch.zeros_like(theta)
    for y in range(theta.shape[-1]):
        gy, = torch.autograd.grad(k[y], theta, retain_graph=True)
        g += p[y] * gy
    return g


def entropy(logits: torch.Tensor) -> torch.Tensor:
    lp = torch.log_softmax(logits.float(), -1)
    return -(lp.exp() * lp).sum(-1)


def decide(diff_mean: float, lo: float, hi: float) -> str:
    if lo > 0:
        return "better"
    if hi < 0:
        return "worse"
    return "inconclusive"
