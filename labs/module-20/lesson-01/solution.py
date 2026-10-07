"""Lab 20.1 — reference solution. Check it with `LAB_TARGET=solution pytest labs/module-20/lesson-01`."""

from __future__ import annotations

import math

import numpy as np

from frontierlab.record.diff import diff_cards

DIRECTIONS = {"equivalent": ("equivalent", "no-difference-detected"), "a_lower": ("lower",), "a_higher": ("higher",),
              "inconclusive": ("no-difference-detected", "inconclusive")}
LABELS = ("MEASURED", "PUBLICLY DOCUMENTED", "INFERENCE/SPECULATION", "PROJECTED", "ANALYSIS")


def hierarchical_interval(a, b, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0) -> dict:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("need two (seeds, items) arrays of the same shape")
    d = a - b
    S, W = d.shape
    rng = np.random.default_rng(seed)
    s_idx = rng.integers(0, S, size=(n_boot, S))
    w_idx = rng.integers(0, W, size=(n_boot, S, W))
    boots = d[s_idx[:, :, None], w_idx].mean(axis=(1, 2))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"mean_diff": float(d.mean()), "ci": (float(lo), float(hi)),
            "method": f"paired hierarchical bootstrap over {S} seeds and {W} items ({n_boot} resamples)"}


def decide(mean: float, lo: float, hi: float, margin: float) -> str:
    if any(math.isnan(x) for x in (mean, lo, hi)):
        return "inconclusive"
    if lo >= -margin and hi <= margin:
        return "equivalent"
    if hi < 0:
        return "a_lower"
    if lo > 0:
        return "a_higher"
    return "inconclusive"


def claim_problems(claims: list[dict], decisions: dict[str, str]) -> list[tuple[str, str]]:
    out = []
    for i, c in enumerate(claims):
        cid = str(c.get("id", i))
        label = c.get("label")
        if label not in LABELS:
            out.append(("CLAIM_LABEL", cid))
            continue
        if label == "PUBLICLY DOCUMENTED" and not c.get("source"):
            out.append(("CLAIM_SOURCE", cid))
        if label == "MEASURED":
            comp = c.get("comparison")
            if comp not in decisions:
                out.append(("CLAIM_COMPARISON", cid))
            elif c.get("direction") not in DIRECTIONS[decisions[comp]]:
                out.append(("CLAIM_DIRECTION", cid))
            if c.get("scope", "course") != "course":
                out.append(("CLAIM_SCOPE", cid))
    return out


def comparison_problems(claim: dict, comparison: dict, cards: dict[str, dict]) -> list[tuple[str, str]]:
    a, b = comparison["a"], comparison["b"]
    seeds = lambda arm: list(claim["arms"][arm].get("seeds", claim["seeds"]))   # noqa: E731
    out = []
    tuning = claim.get("tuning") or {}
    if tuning.get(a) != tuning.get(b):
        out.append(("TUNING", comparison["name"]))
    if sorted(seeds(a)) != sorted(seeds(b)):
        out.append(("SEEDS", comparison["name"]))
    for s in sorted(set(seeds(a)) & set(seeds(b))):
        ca, cb = cards.get(f"{a}-s{s}"), cards.get(f"{b}-s{s}")
        if ca is None or cb is None:
            continue
        if any(f.severity == "invalidates" for f in diff_cards(cb, ca, changed=comparison.get("changed", []),
                                                               axis=claim.get("axis", "tokens"))):
            out.append(("PARITY", f"seed {s}"))
    return out
