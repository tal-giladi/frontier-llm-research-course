"""Reference solution for lab 04.1."""

from __future__ import annotations

import math

import torch


def oracle_answer(ids, remember, keys, candidates):
    m, link = len(remember), {}
    for i in range(len(ids) - m - 1):
        if ids[i:i + m] == list(remember) and ids[i + m] in keys:
            link.setdefault(ids[i + m], ids[i + m + 1])
    cur, seen = ids[-1], set()
    while cur in link and cur not in seen:
        seen.add(cur)
        cur = link[cur]
        if cur in candidates:
            return cur
    return None


def candidate_score(last_logits, candidates, answer):
    lc = last_logits[torch.tensor(candidates)].double()
    a = candidates.index(answer)
    correct = bool(lc.argmax().item() == a and (lc == lc[a]).sum().item() == 1)
    return correct, float(lc[a] - torch.logsumexp(lc, 0))


@torch.no_grad()
def context_gain(model, windows, W, lo, hi):
    full = windows[:, :hi]
    cut = full[:, lo - W:]
    a = model(full, labels=full).per_token_loss[:, lo - 1:hi - 1]     # targets lo..hi-1, full context
    b = model(cut, labels=cut).per_token_loss[:, W - 1:]               # the same targets, >= W tokens of context
    return b.mean(1) - a.mean(1)


def effective_length(cells, threshold, key="acc"):
    best = None
    for c in sorted(cells, key=lambda c: c["length"]):
        if c[key][1] < threshold:
            break
        best = c["length"]
    return best


def items_needed(p_model, p_chance, z_alpha=1.96, z_power=0.84):
    p0, p1 = p_chance, p_model
    n = ((z_alpha * math.sqrt(p0 * (1 - p0)) + z_power * math.sqrt(p1 * (1 - p1))) / (p1 - p0)) ** 2
    return math.ceil(n)
