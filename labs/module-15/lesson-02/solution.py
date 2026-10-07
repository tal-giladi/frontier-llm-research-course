"""Reference solution for lab 15.2."""

from __future__ import annotations

import torch


def residual(p, q):
    r = (p - q).clamp_min(0)
    z = r.sum(-1, keepdim=True)
    return torch.where(z > 0, r / z.clamp_min(torch.finfo(r.dtype).tiny), p)


def accept_reject(p, q, drafts, u, generator=None):
    B, g = drafts.shape
    pd = p[:, :g].gather(-1, drafts[..., None]).squeeze(-1)
    qd = q.gather(-1, drafts[..., None]).squeeze(-1)
    ok = u < torch.clamp(pd / qd.clamp_min(torch.finfo(qd.dtype).tiny), max=1.0)
    n = ok.long().cumprod(-1).sum(-1)                         # leading accepts only
    ar = torch.arange(B)
    qpad = torch.cat([q, torch.zeros_like(q[:, :1])], 1)      # q = 0 at the bonus slot, so the residual is p
    nxt = torch.multinomial(residual(p[ar, n], qpad[ar, n]), 1, generator=generator).squeeze(-1)
    return n, nxt


def expected_tokens_per_round(alpha, gamma):
    if alpha >= 1:
        return float(gamma + 1)
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def walltime_improvement(alpha, gamma, c):
    return expected_tokens_per_round(alpha, gamma) / (gamma * c + 1)
