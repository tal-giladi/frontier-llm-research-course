"""Shared pieces of Eval Suite v2: pass@k, paired comparisons per component, the retention guard.

A suite result is a dict ``{"version", "pins", "components": {name: {"kind", "items": [float, ...]}}}``
where every component lists one score per item in a fixed item order, higher = better, so that two
checkpoints scored with the same pins can be compared item by item.

``kind`` is ``"task"``, ``"retention"`` or ``"instruction"``. :func:`compare` returns, per component,
the paired mean difference (new - base) with a 95% bootstrap interval, and applies the decision rule:

* a ``task`` component *improved* if the interval's lower bound is above 0;
* a ``retention`` or ``instruction`` component *regressed* if the interval's lower bound is below
  ``-guard`` (the largest loss you will accept, stated before the run), *held* if the lower bound is
  at or above ``-guard``.

A stage passes Eval v2 when no retention or instruction component regressed.
"""

from __future__ import annotations

import math

import numpy as np

from frontierlab.stats import paired_bootstrap

VERSION = "eval-v2.0"


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples of which c are correct: 1 - C(n-c, k) / C(n, k) (Chen et al. 2021)."""
    if k > n:
        raise ValueError("k must be <= n")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def compare(base: dict, new: dict, guards: dict | None = None, default_guard: float = 0.02,
            n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired comparison of every component two suite results share. ``guards`` maps component -> guard."""
    if base.get("pins") != new.get("pins"):
        raise ValueError("results were produced with different pins; items are not paired")
    guards = guards or {}
    out = {}
    for name, comp in base["components"].items():
        if name.startswith("_") or name not in new["components"]:
            continue
        a, b = np.asarray(new["components"][name]["items"], float), np.asarray(comp["items"], float)
        res = paired_bootstrap(a, b, n_boot=n_boot, seed=seed)
        lo, hi = res["ci"]
        kind = comp["kind"]
        g = guards.get(name, default_guard)
        if kind == "task":
            verdict = "improved" if lo > 0 else ("worse" if hi < 0 else "no clear change")
        else:
            verdict = "regressed" if lo < -g else "held"
        out[name] = {"kind": kind, "base": float(b.mean()), "new": float(a.mean()), "diff": res["mean_diff"],
                     "ci": (lo, hi), "guard": None if kind == "task" else g, "verdict": verdict, "n": int(a.size)}
    out["_passes"] = all(v["verdict"] != "regressed" for k, v in out.items() if not k.startswith("_"))
    return out


def report(cmp: dict) -> str:
    """A plain-text table of :func:`compare`'s output."""
    lines = [f"{'component':28s} {'kind':12s} {'base':>8s} {'new':>8s} {'diff':>8s}  95% CI              verdict"]
    for name, v in cmp.items():
        if name.startswith("_"):
            continue
        lo, hi = v["ci"]
        lines.append(f"{name:28s} {v['kind']:12s} {v['base']:8.3f} {v['new']:8.3f} {v['diff']:+8.3f}  "
                     f"[{lo:+.3f}, {hi:+.3f}]  {v['verdict']}")
    lines.append(f"Eval v2 retention guard: {'PASS' if cmp['_passes'] else 'FAIL'}")
    return "\n".join(lines)
