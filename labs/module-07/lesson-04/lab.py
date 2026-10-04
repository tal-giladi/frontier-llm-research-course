"""Lab 07.4 — schedules: WSD, decay branches, cooldown. Fill in the TODOs; run `pytest labs/module-07/lesson-04`.

Steps are 0-based, as in the course loop (the loop sets the learning rate for step s before taking it).
"""

from __future__ import annotations


def wsd_lr(step: int, lr: float, warmup: int, decay_start: int, decay_steps: int, shape: str = "linear",
           min_ratio: float = 0.0) -> float:
    """Warmup-stable-decay learning rate.

    warmup  (step < warmup):              lr * (step + 1) / warmup
    stable  (warmup <= step < decay_start): lr
    decay:  u = min(1, (step - decay_start + 1) / decay_steps);
            lr * (min_ratio + (1 - min_ratio) * f(u)), f(u) = 1 - u ("linear") or 1 - sqrt(u) ("1-sqrt")
    """
    raise NotImplementedError("TODO 1: the WSD schedule")


def branch_plan(totals: list[int], decay_frac: float) -> list[dict]:
    """For each total length S: {"total": S, "decay_start": S - D, "decay_steps": D}, D = round(S * decay_frac)."""
    raise NotImplementedError("TODO 2: plan the decay branches")


def branch_cost(plan: list[dict]) -> dict:
    """Steps to get every total in ``plan``: WSD = one stable run to the last decay_start + every decay;
    cosine = one full run per total. Return {"wsd_steps", "cosine_steps", "saving"} (saving = 1 - wsd / cosine)."""
    raise NotImplementedError("TODO 3: compare the cost of branches and separate runs")


def decay_drop(steps: list[int], losses: list[float], decay_start: int, window: int = 10) -> float:
    """Training-loss drop over a decay: mean loss of the ``window`` logged steps just before decay_start minus
    the mean of the last ``window`` logged steps of the run (positive = the decay lowered the loss)."""
    raise NotImplementedError("TODO 4: how much the loss fell during the decay")
