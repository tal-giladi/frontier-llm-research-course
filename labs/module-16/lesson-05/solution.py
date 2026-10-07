"""Reference solution for lab 16.5 — evaluating computer-use agents (extension)."""

from __future__ import annotations

import math

import numpy as np


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def task_reward(feasible: bool, final_action: str, state_ok: bool) -> float:
    if not feasible:
        return float(final_action == "FAIL")
    return float(final_action == "DONE" and state_ok)


def cluster_interval(success: list[float], cluster: list[str], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    s = np.asarray(success, dtype=float)
    keys = sorted(set(cluster))
    idx = {k: np.flatnonzero(np.asarray(cluster) == k) for k in keys}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        rows = np.concatenate([idx[keys[i]] for i in pick])
        boots.append(s[rows].mean())
    return float(s.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def success_at_budget(steps_needed: list[float], budget: int) -> float:
    a = np.asarray(steps_needed, dtype=float)
    return float(np.mean(a <= budget))
