"""Learning-rate schedules for lesson 07.4: cosine, constant, and warmup-stable-decay with an explicit decay branch.

The course loop's ``lr_at`` knows cosine, constant and a WSD whose decay is always the last 20% of
``--steps``, linear, to 0.1 of the peak. Decay *branches* need more control: decay from a chosen step of
one stable (constant-LR) run, for a chosen number of steps, with a chosen shape. :func:`lr_at` here is a
drop-in replacement that adds that (the Module 7 wrapper installs it for one ``loop.main()`` call):

    warmup       s < W                       lr · (s + 1) / W                   (as the loop)
    stable       W <= s < S0                 lr
    decay        S0 <= s < S0 + D            lr · (r + (1 - r) · f(u)),  u = (s - S0 + 1) / D in (0, 1]
                 s >= S0 + D                 lr · r          (the last decay step already reaches r)

with S0 = ``decay_start``, D = ``decay_steps``, r = ``min_ratio`` and shapes

    linear   f(u) = 1 - u
    1-sqrt   f(u) = 1 - sqrt(u)              (Hägele et al., arXiv 2405.18392 section 3.2)
    cosine   f(u) = (1 + cos(pi u)) / 2

Before S0 the WSD value is bit-for-bit the constant schedule's value, so a branch started from a stable
run's checkpoint at step S0 continues that run exactly up to the point where it starts to decay.
MiniCPM (arXiv 2404.06395, sections 4.2-4.3) defines WSD with an exponential decay and reports that a decay
of about 10% of the tokens is enough and that the loss drops sharply during the decay.
"""

from __future__ import annotations

import math

SHAPES = ("linear", "1-sqrt", "cosine")


def decay_factor(u: float, shape: str) -> float:
    if shape == "linear":
        return 1.0 - u
    if shape == "1-sqrt":
        return 1.0 - math.sqrt(u)
    if shape == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * u))
    raise ValueError(f"unknown decay shape {shape!r}; known: {SHAPES}")


def lr_at(step: int, steps: int, lr: float, warmup: int, schedule: str = "cosine", *, decay_start: int | None = None,
          decay_steps: int | None = None, shape: str = "linear", min_ratio: float = 0.0,
          cosine_min_ratio: float = 0.1) -> float:
    """Learning rate at ``step`` (0-based). ``schedule`` in {"cosine", "constant", "wsd"}.

    For "wsd", ``decay_start`` defaults to 80% of ``steps`` and ``decay_steps`` to the rest of the run.
    "cosine" is the loop's cosine (to ``cosine_min_ratio`` · lr at ``steps``), so cosine runs are unchanged.
    """
    if step < warmup:
        return lr * (step + 1) / warmup
    if schedule == "constant":
        return lr
    if schedule == "wsd":
        s0 = int(steps * 0.8) if decay_start is None else int(decay_start)
        d = (steps - s0) if decay_steps is None else int(decay_steps)
        if step < s0:
            return lr
        u = min(1.0, (step - s0 + 1) / max(1, d))
        if u >= 1.0:
            return lr * min_ratio
        return lr * (min_ratio + (1 - min_ratio) * decay_factor(u, shape))
    if schedule == "cosine":
        progress = (step - warmup) / max(1, steps - warmup)
        return lr * (cosine_min_ratio + (1 - cosine_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
    raise ValueError(f"unknown schedule {schedule!r}")


def branch_plan(stable_steps: list[int], decay_frac: float) -> list[dict]:
    """Decay branches off one stable run: for each total length S, decay from S0 = S·(1-f) for D = S·f steps.

    Returns [{"total": S, "decay_start": S0, "decay_steps": D}], the checkpoints the stable run must keep (S0).
    """
    out = []
    for S in stable_steps:
        d = max(1, round(S * decay_frac))
        out.append({"total": S, "decay_start": S - d, "decay_steps": d})
    return out


def branch_cost(plan: list[dict]) -> dict:
    """Training steps for WSD branches (one stable run to the last S0, plus every decay) vs separate cosine runs."""
    stable = max(p["decay_start"] for p in plan)
    wsd = stable + sum(p["decay_steps"] for p in plan)
    cosine = sum(p["total"] for p in plan)
    return {"wsd_steps": wsd, "cosine_steps": cosine, "saving": 1 - wsd / cosine}
