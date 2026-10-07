"""Lab 13.4 — Thinking modes and budgets. Reference solution."""

from __future__ import annotations

import numpy as np


def trace(a, b):
    out, carry = [], 0
    for i, c in enumerate("abc"):
        s = (a // 10 ** i) % 10 + (b // 10 ** i) % 10 + carry
        out.append(f"{c}{s:02d}")
        carry = s // 10
    return "".join(out)


def truncated_target(a, b, k):
    return trace(a, b)[:k] + "#" + str(a + b)


def routed_cost(p_easy, threshold, correct_think, correct_nothink, tokens_think, tokens_nothink):
    think = np.asarray(p_easy) < threshold
    acc = float(np.where(think, correct_think, correct_nothink).mean())
    tok = float(np.where(think, tokens_think, tokens_nothink).mean())
    return acc, tok, float(think.mean())


def pareto(points):
    pts = sorted(set(points))
    out = []
    for t, a in pts:
        dominated = any((t2 <= t and a2 >= a) and (t2 < t or a2 > a) for t2, a2 in pts)
        if not dominated:
            out.append((t, a))
    return out
