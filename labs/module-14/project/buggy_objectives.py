"""A colleague's objective library (the Module 14 project's debugging task). Do not fix it in place; diagnose it.

Their message:

    "I wrote our own GRPO, GSPO and CISPO so we don't depend on verl's versions. They all train, the
    losses go down, but GSPO barely learns on the long-answer task, CISPO's clip fraction looks normal yet it
    learns slower than the paper says, and GRPO's ratios get huge after a few minibatches. Probably just
    hyperparameters? Can you tune them?"

There are three bugs, one per objective. Each function has the course interface
``(logp, old_logp, adv, mask, **params) -> (loss, diagnostics)``.
"""

from __future__ import annotations

import torch

from frontierlab.rlscale.objectives import _diag


def grpo_loss(logp, old_logp, adv, mask, eps: float = 0.2):
    r = torch.exp(logp - old_logp.detach())
    a = adv[:, None]
    obj = torch.maximum(r * a, torch.clamp(r, 1 - eps, 1 + eps) * a)
    m = mask.float()
    per_seq = (obj * m).sum(-1) / m.sum(-1).clamp_min(1.0)
    flag = (torch.clamp(r, 1 - eps, 1 + eps) * a < r * a).float()
    return -per_seq.mean(), _diag(r, flag, mask)


def gspo_loss(logp, old_logp, adv, mask, eps_low: float = 3e-4, eps_high: float = 4e-4):
    m = mask.float()
    s = torch.exp(((logp - old_logp.detach()) * m).sum(-1))
    unclipped, clipped = s * adv, torch.clamp(s, 1 - eps_low, 1 + eps_high) * adv
    obj = torch.minimum(unclipped, clipped)
    flag = (clipped < unclipped).float()
    return -obj.mean(), _diag(s[:, None].expand_as(logp), flag[:, None].expand_as(logp), mask)


def cispo_loss(logp, old_logp, adv, mask, eps_low=None, eps_high: float = 0.28):
    r = torch.exp(logp - old_logp.detach())
    w = torch.clamp(r, max=1 + eps_high)
    obj = w * adv[:, None]
    m = mask.float()
    capped = (r.detach() > 1 + eps_high).float()
    return -(obj * m).sum() / m.sum().clamp_min(1.0), _diag(r.detach(), capped, mask)


BUGGY = {"grpo": grpo_loss, "gspo": gspo_loss, "cispo": cispo_loss}
