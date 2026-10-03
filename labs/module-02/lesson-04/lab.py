"""Lab 02.4 — validating a performance claim. Fill in the TODOs; run `pytest labs/module-02/lesson-04` to check."""

from __future__ import annotations

import time

import torch  # noqa: F401  (TODO 2)


def naive_time(fn) -> float:
    """How performance claims often get made: one call, no warm-up, no synchronisation. Do not use."""
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def bench(fn, warmup: int = 3, repeats: int = 10, sync=lambda: None) -> list[float]:
    """Seconds per call of ``fn()``: ``warmup`` untimed calls, then ``repeats`` timed calls.

    ``sync()`` must run before the clock is started and before it is read, for every timed call
    (on a GPU it is ``torch.cuda.synchronize``; queued kernels are otherwise not finished).
    Return the list of ``repeats`` durations.
    """
    raise NotImplementedError("TODO 1: a benchmark that warms up, synchronises and repeats")


def grad_agreement(model, idx, loss_a, loss_b) -> dict:
    """Correctness check for an optimisation: same loss and same gradients on the same inputs.

    ``loss_a(model, idx)`` and ``loss_b(model, idx)`` each return a scalar loss. Compute both and
    the gradients of each with respect to every parameter that requires grad
    (``torch.autograd.grad``, so nothing accumulates in ``.grad``). Return
    ``{"loss_diff": |la - lb|, "max_grad_diff": largest |ga - gb| over all parameters}``.
    """
    raise NotImplementedError("TODO 2: loss and gradient agreement")


def decide(correct: bool, speedup_ci: tuple[float, float], mem_ratio: float, max_mem_ratio: float = 0.8,
           min_speedup: float = 0.95) -> str:
    """The decision rule written in the experiment contract *before* the runs.

    - "reject" if the correctness check failed, or if the whole speed-up interval lies below
      ``min_speedup`` (clearly slower than allowed);
    - "adopt" if correct, the speed-up interval's lower end >= ``min_speedup``, and either
      ``mem_ratio`` (new / old memory) <= ``max_mem_ratio`` or the lower end is above 1.0 (clearly
      faster);
    - otherwise "inconclusive" (run more repeats, or report it as not shown).
    """
    raise NotImplementedError("TODO 3: the pre-stated decision rule")
