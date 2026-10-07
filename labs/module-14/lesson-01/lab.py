"""Lab 14.1 — five objectives, derived. Fill in the TODOs; run `pytest labs/module-14/lesson-01` to check.

Shapes (float32 in the loop, float64 in the tests): ``logp``, ``old_logp``, ``mask`` are (B, R) with B = P * G
responses in group-major order and R response positions; ``mask`` is 1.0 on response tokens up to and
including EOS; ``adv`` is (B,), one advantage per response. ``logp`` carries the gradient; ``old_logp`` is
the behaviour policy's and must get none.

Every function returns ``(loss, diag)``: the scalar to *minimise* (minus the objective) and a dict with at
least ``clip_frac`` (fraction of masked tokens whose gradient is zeroed by clipping, or, for CISPO, whose
weight was capped), ``ratio_mean`` and ``ratio_max`` (over masked tokens; for GSPO the sequence ratio
broadcast to its tokens). The tests check the loss and its gradient; :func:`diag` is provided.
"""

from __future__ import annotations

import torch


def diag(ratio: torch.Tensor, flag: torch.Tensor, mask: torch.Tensor) -> dict:
    """Diagnostics over masked tokens (provided)."""
    with torch.no_grad():
        m = mask.float()
        n = m.sum().clamp_min(1.0)
        return {"clip_frac": float((flag.float() * m).sum() / n), "ratio_mean": float((ratio * m).sum() / n),
                "ratio_max": float((ratio * m).max()) if m.sum() > 0 else 1.0}


def grpo_loss(logp, old_logp, adv, mask, eps: float = 0.2):
    """GRPO (DeepSeekMath Eq. 3, no KL term): per token min(r A, clip(r, 1-eps, 1+eps) A), r = exp(logp - old);
    mean over each response's tokens, then mean over responses."""
    raise NotImplementedError("TODO 1: GRPO")


def dapo_loss(logp, old_logp, adv, mask, eps_low: float = 0.2, eps_high: float = 0.28):
    """DAPO (Eq. 8): the same per-token surrogate with clip range [1-eps_low, 1+eps_high], summed over every
    masked token of the batch and divided by the number of masked tokens."""
    raise NotImplementedError("TODO 2: DAPO")


def drgrpo_loss(logp, old_logp, adv, mask, eps: float = 0.2, norm_len: int | None = None):
    """Dr. GRPO (section 3.2): GRPO's surrogate summed over masked tokens, divided by B * norm_len
    (norm_len defaults to R, the generation budget)."""
    raise NotImplementedError("TODO 3: Dr. GRPO")


def gspo_loss(logp, old_logp, adv, mask, eps_low: float = 3e-4, eps_high: float = 4e-4):
    """GSPO (section 4.1, Eqs. 5 and 7): s_i = exp(mean over response i's masked tokens of (logp - old));
    objective min(s_i A_i, clip(s_i, 1-eps_low, 1+eps_high) A_i), mean over responses."""
    raise NotImplementedError("TODO 4: GSPO")


def cispo_loss(logp, old_logp, adv, mask, eps_low: float | None = None, eps_high: float = 0.28):
    """CISPO (MiniMax-M1 section 3.1): per token sg(clip(r, 1-eps_low, 1+eps_high)) * A * logp, summed over
    masked tokens and divided by their number. eps_low=None means no lower bound. The weight must carry no
    gradient: the gradient is the capped weight times A times grad log pi, for *every* token."""
    raise NotImplementedError("TODO 5: CISPO")
