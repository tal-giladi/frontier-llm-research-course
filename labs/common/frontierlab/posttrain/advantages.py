"""Advantages and baselines for outcome-reward RL on language models (lesson 12.2).

Rewards arrive as a (P, G) tensor: P prompts, G sampled responses per prompt (a *group*). Every token of
response (p, g) gets the same advantage, as in GRPO's outcome supervision (DeepSeekMath section 4.1.2).

Baselines (subtracted from the reward). ``"mean"`` includes the response's own reward, so its expected
gradient is the true gradient times (G-1)/G: same direction, smaller step. Dividing by a std that
includes the response's own reward is not a constant rescaling, and is the "question-level difficulty
bias" of Dr. GRPO.

* ``"none"``  — REINFORCE with the raw reward.
* ``"mean"``  — the group mean (GRPO).
* ``"loo"``   — leave-one-out: the mean of the *other* G-1 rewards (RLOO, Ahmadian et al. 2024, section 2.3).
  ``loo = G/(G-1) * (r - mean)``: the same direction as ``"mean"``, rescaled, and unbiased.

Scale normalisation (divides after subtracting):

* ``"none"``  — keep reward units (Dr. GRPO removes the std, Liu et al. 2025 section 3.2).
* ``"group"`` — divide by the group's std (GRPO; std with ddof=1, as TRL computes it).
* ``"batch"`` — divide by the std of all P*G rewards in the batch (ScaleRL, Khatri et al. 2025 section 3.2).

A *zero-variance group* (every response got the same reward) has advantage 0 under any baseline: it
contributes no policy gradient, only KL and entropy terms and a smaller effective batch.
"""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, baseline: str = "mean", scale: str = "group",
                     eps: float = 1e-6) -> torch.Tensor:
    """(P, G) rewards -> (P, G) advantages."""
    if rewards.dim() != 2:
        raise ValueError("rewards must be (P prompts, G responses)")
    P, G = rewards.shape
    r = rewards.double()
    if baseline == "none":
        adv = r.clone()
    elif baseline == "mean":
        adv = r - r.mean(dim=1, keepdim=True)
    elif baseline == "loo":
        if G < 2:
            raise ValueError("leave-one-out needs G >= 2")
        adv = r - (r.sum(dim=1, keepdim=True) - r) / (G - 1)
    else:
        raise ValueError(baseline)
    if scale == "group":
        adv = adv / (r.std(dim=1, keepdim=True) + eps)
    elif scale == "batch":
        adv = adv / (r.std() + eps)
    elif scale != "none":
        raise ValueError(scale)
    return adv.to(rewards.dtype)


def zero_variance(rewards: torch.Tensor) -> torch.Tensor:
    """(P,) bool: groups in which every response has the same reward."""
    return (rewards == rewards[:, :1]).all(dim=1)


def effective_fraction(rewards: torch.Tensor) -> float:
    """Fraction of groups that carry a policy-gradient signal."""
    return float((~zero_variance(rewards)).float().mean())


def gae(rewards: torch.Tensor, values: torch.Tensor, gamma: float = 1.0, lam: float = 0.95) -> torch.Tensor:
    """Generalised advantage estimation for one trajectory (Schulman et al. 2015, Eq. 16).

    ``rewards`` (T,), ``values`` (T + 1,) with ``values[T]`` = value after the last step (0 if terminal).
    delta_t = r_t + gamma V_{t+1} - V_t;  A_t = sum_l (gamma lam)^l delta_{t+l}.
    PPO-style RLHF uses this per token with a learned value model; GRPO-style methods drop the value
    model and use the group baseline above instead.
    """
    T = rewards.shape[0]
    adv = torch.zeros(T, dtype=torch.float64)
    running = 0.0
    for t in reversed(range(T)):
        delta = float(rewards[t]) + gamma * float(values[t + 1]) - float(values[t])
        running = delta + gamma * lam * running
        adv[t] = running
    return adv.to(rewards.dtype)
