"""Lab 16.5 (extension) — evaluating computer-use agents. Fill in the TODOs; run `pytest labs/module-16/lesson-05`.
``cua_eval_lab.py`` uses these functions on simulated evaluation results.
"""

from __future__ import annotations

import math

import numpy as np


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for k successes in n tasks: centre (p + z^2/2n)/(1 + z^2/n), half-width
    z sqrt(p(1-p)/n + z^2/4n^2)/(1 + z^2/n). (nan, nan) if n == 0."""
    raise NotImplementedError("TODO 1: Wilson interval")


def task_reward(feasible: bool, final_action: str, state_ok: bool) -> float:
    """OSWorld's rule (section 2.1): a feasible task scores 1 if the agent ends with "DONE" and the final state
    passes the task's checker; an infeasible task scores 1 if the agent ends with "FAIL". Otherwise 0."""
    raise NotImplementedError("TODO 2: the task reward with infeasible tasks")


def cluster_interval(success: list[float], cluster: list[str], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(mean, low, high): 95% percentile bootstrap that resamples whole clusters (applications) with replacement,
    ``np.random.default_rng(seed)``, clusters taken in sorted order, ``rng.choice(len(clusters), size=len(clusters))``
    per replicate, and the mean over all tasks of the picked clusters."""
    raise NotImplementedError("TODO 3: app-clustered bootstrap")


def success_at_budget(steps_needed: list[float], budget: int) -> float:
    """Share of tasks solved within ``budget`` steps; ``steps_needed`` is inf for tasks never solved."""
    raise NotImplementedError("TODO 4: success at a step budget")
