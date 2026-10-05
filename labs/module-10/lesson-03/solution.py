"""Reference solution for lab 10.3 — synthetic and rephrased data."""

from __future__ import annotations

import re
from collections import Counter

NUM = re.compile(r"\d+(?:[.,]\d+)*")
WORD = re.compile(r"[a-zA-Z]+")


def generation_flops(n_nonemb, layers, d_attn, prompt, generated, head_flops_per_token=0):
    P, G = prompt, generated
    ctx_sum = G * P + G * (G - 1) / 2
    return (2 * n_nonemb * (P + G) + 2 * layers * d_attn * P * P + 4 * layers * d_attn * ctx_sum
            + head_flops_per_token * (P + G))


def number_check(src, out):
    ns = {n.replace(",", "") for n in NUM.findall(src)}
    no = {n.replace(",", "") for n in NUM.findall(out)}
    return (len(ns & no) / len(ns) if ns else None), len(no - ns)


def novel_ngram_share(src, out, n=4):
    def grams(t):
        w = [x.lower() for x in WORD.findall(t)]
        return [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    g_out, g_src = grams(out), set(grams(src))
    return sum(g not in g_src for g in g_out) / len(g_out) if g_out else None


def distinct_n(outputs, n=2):
    c = Counter()
    for o in outputs:
        w = [x.lower() for x in WORD.findall(o)]
        c.update(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    total = sum(c.values())
    return len(c) / total if total else 0.0


def cost_ratio(gen_flops, train_flops):
    return gen_flops / train_flops


def decide(ci, margin=0.0):
    lo, hi = ci
    if hi < -margin:
        return "adopt"
    if lo > -margin:
        return "reject"
    return "inconclusive"
