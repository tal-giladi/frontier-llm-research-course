"""Reference solution for lab 04.3."""

from __future__ import annotations

import numpy as np

from frontierlab.flops import flops_per_token


def same_doc_context(doc_starts, window_starts, T):
    pos = np.asarray(window_starts)[:, None] + np.arange(T)[None, :]
    start = np.asarray(doc_starts)[np.searchsorted(doc_starts, pos, side="right") - 1]
    return np.minimum(pos - start, np.arange(T)[None, :])


def within_doc_start(doc_starts, lengths, T, u):
    ok = np.nonzero(np.asarray(lengths) >= T)[0]
    cum = np.cumsum(np.asarray(lengths)[ok] - T + 1)
    if not 0 <= u < cum[-1]:
        raise ValueError("u out of range")
    j = int(np.searchsorted(cum, u, side="right"))
    before = int(cum[j - 1]) if j else 0
    return int(doc_starts[ok[j]]) + (u - before)


def stage_plan(cfg, stages, peak_flops, mfu):
    rows = []
    for T, tokens in stages:
        f = flops_per_token(cfg, T) * tokens
        rows.append({"seq": T, "tokens": tokens, "flops": f, "gpu_hours": f / (peak_flops * mfu) / 3600})
    rows.append({"seq": "total", "tokens": sum(r["tokens"] for r in rows), "flops": sum(r["flops"] for r in rows),
                 "gpu_hours": sum(r["gpu_hours"] for r in rows)})
    return rows


def decide(short_regression_ci, gain_ci, max_regression, min_gain):
    if short_regression_ci[1] <= max_regression and gain_ci[0] >= min_gain:
        return "adopt"
    if short_regression_ci[0] > max_regression or gain_ci[1] < min_gain:
        return "reject"
    return "inconclusive"
