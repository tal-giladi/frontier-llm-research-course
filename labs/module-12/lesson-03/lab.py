"""Lab 12.3 — details that change results. Fill in the TODOs; run `pytest labs/module-12/lesson-03` to check.

Shapes: per-token tensors are (B, R) with B = P * G responses in group-major order (rows p*G .. p*G+G-1 are
prompt p's group) and R response positions; ``mask`` is 1.0 on response tokens up to and including EOS.
"""

from __future__ import annotations

import torch

EOS = 2


def response_mask(response: torch.Tensor):
    """(mask (B, R) float, finished (B,) bool) for sampled responses (B, R) int64.

    mask is 1 on every token up to *and including* the first EOS, 0 after it. A response without EOS is
    truncated: all R tokens are in the mask and finished is False. No Python loops over positions.
    """
    raise NotImplementedError("TODO 1: the response mask")


def aggregate(per_token, mask, mode, group_size=None, norm_len=None):
    """Scalar from per-token values under one of four modes (masked positions contribute nothing):

    "token_mean": sum / number of masked tokens (DAPO);  "seq_mean_token_mean": mean over responses of
    (sum over its tokens / its length) (GRPO);  "seq_mean_token_sum_norm": sum / (B * norm_len) with
    norm_len defaulting to R (Dr. GRPO);  "prompt_mean": for each group of ``group_size`` rows,
    sum / number of masked tokens in the group, then the mean over groups.
    """
    raise NotImplementedError("TODO 2: aggregation")


def token_ratio(logp, old_logp):
    """Per-token importance ratio pi_theta / pi_old; no gradient through old_logp."""
    raise NotImplementedError("TODO 3a: token ratio")


def sequence_ratio(logp, old_logp, mask):
    """GSPO's sequence ratio exp(mean over masked tokens of (logp - old_logp)), broadcast to (B, R)."""
    raise NotImplementedError("TODO 3b: sequence ratio")


def clipped_surrogate(ratio, adv, eps_low=0.2, eps_high=0.2):
    """Per-token min(ratio * A, clip(ratio, 1 - eps_low, 1 + eps_high) * A); adv is (B,) or (B, R)."""
    raise NotImplementedError("TODO 4: clipped surrogate")


def soft_overlong_penalty(lengths, l_max, l_cache):
    """DAPO Eq. 13, float32: 0 if L <= l_max - l_cache; ((l_max - l_cache) - L) / l_cache up to l_max;
    -1 beyond l_max."""
    raise NotImplementedError("TODO 5: soft overlong penalty")


def tis_weight(old_logp_trainer, sampler_logp, cap=2.0):
    """Truncated importance weight min(exp(old_logp_trainer - sampler_logp), cap), detached."""
    raise NotImplementedError("TODO 6: truncated importance sampling")
