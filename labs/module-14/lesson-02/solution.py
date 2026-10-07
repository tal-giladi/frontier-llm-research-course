"""Reference solution for lab 14.2."""

from __future__ import annotations

import math

THRESHOLDS = {"collapse_drawdown": 0.10, "grad_spikes": 3, "entropy_drop": 0.25, "ratio_max": 50.0}


def select_lr(tuning):
    ok = {lr: r for lr, r in tuning.items() if not r["collapsed"]}
    if not ok:
        return min(tuning)
    best = max(r["score"] for r in ok.values())
    return min(lr for lr, r in ok.items() if r["score"] >= best - 0.005)


def paired_interval(a, b, alpha=0.05):
    from scipy.stats import t
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    if n < 2:
        return m, float("nan"), float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    h = float(t.ppf(1 - alpha / 2, n - 1)) * sd / math.sqrt(n)
    return m, m - h, m + h


def verdict(vs_baseline, vs_control):
    if not vs_control[1] > 0:
        return "below control"
    if vs_baseline[1] > 0:
        return "better than baseline"
    if vs_baseline[2] < 0:
        return "worse than baseline"
    return "inconclusive"


def stability_flags(st, thresholds=THRESHOLDS):
    checks = [("collapse", "eval_drawdown", "collapse_drawdown"), ("grad_spikes", "grad_spikes", "grad_spikes"),
              ("entropy_collapse", "entropy_drop", "entropy_drop"), ("ratio_blowup", "ratio_max", "ratio_max")]
    out = []
    for name, key, th in checks:
        v = st.get(key, float("nan"))
        if v is not None and not (isinstance(v, float) and math.isnan(v)) and v >= thresholds[th]:
            out.append(name)
    return out
