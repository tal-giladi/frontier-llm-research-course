"""Reference solution for lab 07.5."""

from __future__ import annotations

import statistics

import torch


def z_loss(logits, coef=1e-4):
    return coef * torch.logsumexp(logits.float(), dim=-1).pow(2).mean()


def softcap(x, cap):
    return cap * torch.tanh(x / cap)


def find_spikes(steps, losses, window=20, k=6.0, min_rel=0.05):
    out = []
    for i in range(window, len(losses)):
        prev = losses[i - window:i]
        med = statistics.median(prev)
        mad = statistics.median(abs(v - med) for v in prev)
        if losses[i] - med > max(k * 1.4826 * mad, min_rel * abs(med)):
            out.append(steps[i])
    return out


def first_crossing(steps, values, threshold, before):
    for s, v in zip(steps, values):
        if s <= before and v >= threshold:
            return s
    return None


def classify(logit_growth, max_logit_before, ratio_jump, recovered):
    if logit_growth >= 4 and max_logit_before > 20:
        return "logit growth"
    if ratio_jump >= 4:
        return "optimizer"
    if recovered:
        return "data"
    return "unclear"
