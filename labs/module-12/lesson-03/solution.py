"""Lab 12.3 — details that change results. Reference solution."""

from __future__ import annotations

import torch

EOS = 2


def response_mask(response: torch.Tensor):
    is_eos = response == EOS
    before = torch.cumsum(is_eos.long(), -1) - is_eos.long()
    return (before == 0).float(), is_eos.any(-1)


def aggregate(per_token, mask, mode, group_size=None, norm_len=None):
    m = mask.float()
    x = per_token * m
    if mode == "token_mean":
        return x.sum() / m.sum().clamp_min(1.0)
    if mode == "seq_mean_token_mean":
        return (x.sum(-1) / m.sum(-1).clamp_min(1.0)).mean()
    if mode == "seq_mean_token_sum_norm":
        return x.sum() / (per_token.shape[0] * (norm_len or per_token.shape[1]))
    if mode == "prompt_mean":
        P = per_token.shape[0] // group_size
        return (x.view(P, -1).sum(-1) / m.view(P, -1).sum(-1).clamp_min(1.0)).mean()
    raise ValueError(mode)


def token_ratio(logp, old_logp):
    return torch.exp(logp - old_logp.detach())


def sequence_ratio(logp, old_logp, mask):
    n = mask.sum(-1).clamp_min(1.0)
    return torch.exp(((logp - old_logp.detach()) * mask).sum(-1) / n)[:, None].expand_as(logp)


def clipped_surrogate(ratio, adv, eps_low=0.2, eps_high=0.2):
    if adv.dim() == 1:
        adv = adv[:, None].expand_as(ratio)
    return torch.minimum(ratio * adv, torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * adv)


def soft_overlong_penalty(lengths, l_max, l_cache):
    L = lengths.double()
    start = l_max - l_cache
    pen = torch.where(L <= start, torch.zeros_like(L), (start - L) / l_cache)
    return torch.where(L > l_max, -torch.ones_like(L), pen).float()


def tis_weight(old_logp_trainer, sampler_logp, cap=2.0):
    return torch.clamp(torch.exp(old_logp_trainer.detach() - sampler_logp.detach()), max=cap)
