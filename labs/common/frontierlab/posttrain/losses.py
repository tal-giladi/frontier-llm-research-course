"""The policy loss and the details that change RL results (lesson 12.3).

All per-token tensors are (B, R): B = P*G responses (group-major: rows p*G .. p*G+G-1 belong to prompt p),
R = max response tokens, and ``mask`` (B, R) float is 1 on response tokens up to and including EOS.

**Importance ratios.** The update reuses samples drawn from an older policy (several minibatches or epochs
per batch, or stale rollouts from an asynchronous sampler), so each token's gradient is weighted by
rho_t = pi_theta(y_t) / pi_old(y_t) (:func:`token_ratio`). GSPO uses one length-normalised ratio per
sequence, (pi_theta(y)/pi_old(y))^(1/|y|) (:func:`sequence_ratio`). PPO clips the ratio to
[1 - eps_low, 1 + eps_high] inside min(rho A, clip(rho) A) (PPO Eq. 7; DAPO's clip-higher uses
eps_high > eps_low).

**Aggregation** (:func:`aggregate`) — how per-token losses become one number. It decides how much
each token, each response and each prompt weighs:

==============================  =============================================  =======================
mode                            loss                                           used by
==============================  =============================================  =======================
``"seq_mean_token_mean"``       mean over responses of (sum_t l / abs(y_i))    GRPO (DeepSeekMath Eq. 3)
``"token_mean"``                sum over all tokens / number of tokens         DAPO (Eq. 12)
``"seq_mean_token_sum_norm"``   sum over all tokens / (B * L_norm)             Dr. GRPO (constant L_norm)
``"prompt_mean"``               mean over prompts of (that prompt's token      ScaleRL "prompt average"
                                mean)                                          (our reading of sec. 3.2)
==============================  =============================================  =======================

**Truncation.** A response that hits the token limit has no EOS. :func:`soft_overlong_penalty` is DAPO's
length-aware penalty (Eq. 13); :func:`overlong_filter` removes truncated responses from the loss.

**Sampler/trainer mismatch.** :func:`tis_weight` is truncated importance sampling
min(pi_old_trainer / pi_sampler, C) (Yao et al. 2025), applied when the rollout engine's probabilities
differ from the trainer's (other kernels, other precision).
"""

from __future__ import annotations

import torch

AGGREGATIONS = ("seq_mean_token_mean", "token_mean", "seq_mean_token_sum_norm", "prompt_mean")


def token_ratio(logp: torch.Tensor, old_logp: torch.Tensor) -> torch.Tensor:
    return torch.exp(logp - old_logp.detach())


def sequence_ratio(logp: torch.Tensor, old_logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """GSPO: exp(mean_t (logp - old_logp)) per sequence, broadcast to (B, R)."""
    n = mask.sum(-1).clamp_min(1.0)
    s = torch.exp(((logp - old_logp.detach()) * mask).sum(-1) / n)
    return s[:, None].expand_as(logp)


def clipped_surrogate(ratio: torch.Tensor, adv: torch.Tensor, eps_low: float = 0.2,
                      eps_high: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token PPO objective min(rho A, clip(rho, 1-eps_low, 1+eps_high) A) and a clipped flag.

    ``adv`` is (B,) (one advantage per response) or (B, R). Returns (objective, clipped) where
    ``clipped`` is 1 where the clipped branch was the minimum and differs from the unclipped one (no
    gradient flows through those tokens).
    """
    if adv.dim() == 1:
        adv = adv[:, None].expand_as(ratio)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv
    obj = torch.minimum(unclipped, clipped)
    was_clipped = (clipped < unclipped).float()
    return obj, was_clipped


def aggregate(per_token: torch.Tensor, mask: torch.Tensor, mode: str, group_size: int | None = None,
              norm_len: int | None = None) -> torch.Tensor:
    """Reduce (B, R) per-token losses to a scalar. See the module table."""
    m = mask.float()
    x = per_token * m
    if mode == "token_mean":
        return x.sum() / m.sum().clamp_min(1.0)
    if mode == "seq_mean_token_mean":
        return (x.sum(-1) / m.sum(-1).clamp_min(1.0)).mean()
    if mode == "seq_mean_token_sum_norm":
        L = norm_len if norm_len is not None else per_token.shape[1]
        return x.sum() / (per_token.shape[0] * L)
    if mode == "prompt_mean":
        if not group_size:
            raise ValueError("prompt_mean needs group_size")
        B = per_token.shape[0]
        xs = x.view(B // group_size, -1).sum(-1)
        ms = m.view(B // group_size, -1).sum(-1).clamp_min(1.0)
        return (xs / ms).mean()
    raise ValueError(mode)


def token_weights(mask: torch.Tensor, mode: str, group_size: int | None = None,
                  norm_len: int | None = None) -> torch.Tensor:
    """The weight each token gets in the aggregated loss (the gradient of ``aggregate`` w.r.t. per_token).

    Useful for seeing length bias: under ``seq_mean_token_mean`` a token of a long response weighs less.
    """
    pt = torch.zeros_like(mask, dtype=torch.float64, requires_grad=True)
    aggregate(pt, mask.double(), mode, group_size, norm_len).backward()
    return pt.grad


def soft_overlong_penalty(lengths: torch.Tensor, l_max: int, l_cache: int) -> torch.Tensor:
    """DAPO Eq. 13: 0 up to l_max - l_cache, then linear down to -1 at l_max, -1 beyond."""
    L = lengths.double()
    start = l_max - l_cache
    pen = torch.where(L <= start, torch.zeros_like(L), (start - L) / l_cache)
    return torch.where(L > l_max, -torch.ones_like(L), pen).to(lengths.dtype if lengths.is_floating_point()
                                                               else torch.float32)


def overlong_filter(mask: torch.Tensor, finished: torch.Tensor) -> torch.Tensor:
    """Mask with every truncated response (no EOS) removed from the loss (DAPO overlong filtering)."""
    return mask * finished.float()[:, None]


def tis_weight(old_logp_trainer: torch.Tensor, sampler_logp: torch.Tensor, cap: float = 2.0) -> torch.Tensor:
    """Truncated importance weight min(pi_old_trainer / pi_sampler, cap), detached, per token."""
    return torch.clamp(torch.exp(old_logp_trainer.detach() - sampler_logp.detach()), max=cap)


def policy_loss(logp, old_logp, adv, mask, *, aggregation: str = "token_mean", group_size: int | None = None,
                eps_low: float = 0.2, eps_high: float = 0.2, ratio: str = "token", norm_len: int | None = None,
                is_weight: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """Clipped policy-gradient loss (to minimise) and diagnostics.

    ``ratio``: ``"token"`` (PPO/GRPO), ``"sequence"`` (GSPO) or ``"none"`` (plain on-policy REINFORCE:
    rho = 1 with the gradient of log pi, i.e. ratio exp(logp - logp.detach())).
    ``is_weight``: optional (B, R) detached weight (truncated IS for sampler mismatch) multiplied into
    the per-token objective.
    """
    if ratio == "token":
        rho = token_ratio(logp, old_logp)
    elif ratio == "sequence":
        rho = sequence_ratio(logp, old_logp, mask)
    elif ratio == "none":
        rho = torch.exp(logp - logp.detach())
    else:
        raise ValueError(ratio)
    obj, was_clipped = clipped_surrogate(rho, adv, eps_low, eps_high)
    if is_weight is not None:
        obj = obj * is_weight
    loss = -aggregate(obj, mask, aggregation, group_size, norm_len)
    with torch.no_grad():
        m = mask.float()
        diag = {"clip_frac": float((was_clipped * m).sum() / m.sum().clamp_min(1.0)),
                "ratio_mean": float((rho.detach() * m).sum() / m.sum().clamp_min(1.0)),
                "ratio_max": float((rho.detach() * m).max()) if m.sum() > 0 else 1.0}
    return loss, diag
