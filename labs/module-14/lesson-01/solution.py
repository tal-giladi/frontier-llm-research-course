"""Reference solution for lab 14.1 (the same mathematics as frontierlab.rlscale.objectives, written out)."""

from __future__ import annotations

import torch


def diag(ratio: torch.Tensor, flag: torch.Tensor, mask: torch.Tensor) -> dict:
    with torch.no_grad():
        m = mask.float()
        n = m.sum().clamp_min(1.0)
        return {"clip_frac": float((flag.float() * m).sum() / n), "ratio_mean": float((ratio * m).sum() / n),
                "ratio_max": float((ratio * m).max()) if m.sum() > 0 else 1.0}


def _surrogate(logp, old_logp, adv, lo, hi):
    r = torch.exp(logp - old_logp.detach())                 # (B, R); d r / d logp = r
    a = adv[:, None]
    unclipped, clipped = r * a, torch.clamp(r, lo, hi) * a
    obj = torch.minimum(unclipped, clipped)                 # the clipped branch is a constant: no gradient
    return obj, r, clipped < unclipped


def grpo_loss(logp, old_logp, adv, mask, eps: float = 0.2):
    obj, r, flag = _surrogate(logp, old_logp, adv, 1 - eps, 1 + eps)
    per_seq = (obj * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
    return -per_seq.mean(), diag(r.detach(), flag, mask)


def dapo_loss(logp, old_logp, adv, mask, eps_low: float = 0.2, eps_high: float = 0.28):
    obj, r, flag = _surrogate(logp, old_logp, adv, 1 - eps_low, 1 + eps_high)
    return -(obj * mask).sum() / mask.sum().clamp_min(1.0), diag(r.detach(), flag, mask)


def drgrpo_loss(logp, old_logp, adv, mask, eps: float = 0.2, norm_len: int | None = None):
    obj, r, flag = _surrogate(logp, old_logp, adv, 1 - eps, 1 + eps)
    L = norm_len if norm_len is not None else logp.shape[1]
    return -(obj * mask).sum() / (logp.shape[0] * L), diag(r.detach(), flag, mask)


def gspo_loss(logp, old_logp, adv, mask, eps_low: float = 3e-4, eps_high: float = 4e-4):
    log_s = ((logp - old_logp.detach()) * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
    s = torch.exp(log_s)                                    # (B,), geometric mean of the token ratios
    unclipped, clipped = s * adv, torch.clamp(s, 1 - eps_low, 1 + eps_high) * adv
    obj = torch.minimum(unclipped, clipped)
    flag = (clipped < unclipped)[:, None].expand_as(logp)
    return -obj.mean(), diag(s.detach()[:, None].expand_as(logp), flag, mask)


def cispo_loss(logp, old_logp, adv, mask, eps_low: float | None = None, eps_high: float = 0.28):
    r = torch.exp(logp.detach() - old_logp.detach())
    lo = 0.0 if eps_low is None else 1 - eps_low
    w = torch.clamp(r, min=lo, max=1 + eps_high)            # detached: a constant weight per token
    obj = w * adv[:, None] * logp
    flag = (r > 1 + eps_high) | (r < lo)
    return -(obj * mask).sum() / mask.sum().clamp_min(1.0), diag(r, flag, mask)
