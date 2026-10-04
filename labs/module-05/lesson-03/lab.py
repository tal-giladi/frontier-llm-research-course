"""Lab 05.3 — turning component timings into a cost model, a crossover and a decision.

Fill in the TODOs; run `pytest labs/module-05/lesson-03`. The scripts `profile_attn.py` and `quality.py` use
these functions.
"""

from __future__ import annotations


def fit_exponent(contexts: list[float], times: list[float]) -> float:
    """Least-squares slope p of log(time) against log(context): time ~ a * context^p.

    p near 2 means the cost grows quadratically, near 1 linearly, near 0 constant (fixed overheads dominate).
    """
    raise NotImplementedError("TODO 1: fit the growth exponent in log-log")


def crossover(contexts: list[float], t_base: list[float], t_new: list[float]) -> float | None:
    """Smallest context at which ``t_new`` < ``t_base``, interpolated in log-log between the two measured
    contexts where the sign of log(t_new / t_base) changes. Contexts are increasing.

    Returns contexts[0] if the new method is already faster at the first point, None if it is never faster.
    """
    raise NotImplementedError("TODO 2: where does the new method become faster?")


def project_time(flops: float, nbytes: float, peak: float, bandwidth: float, mfu: float, bw_eff: float) -> float:
    """Roofline projection (lesson 02.1) in seconds: the larger of compute time at ``mfu`` x peak and
    memory time at ``bw_eff`` x bandwidth."""
    raise NotImplementedError("TODO 3: roofline projection")


def decide(speedup_ci: tuple[float, float], loss_diff_ci: tuple[float, float], min_speedup: float,
           margin: float) -> str:
    """The pre-stated decision rule of the lab's contract.

    speedup_ci: 95% CI of new-over-dense speed-up (> 1 = new is faster); loss_diff_ci: 95% CI of the
    held-out loss difference new - dense (> 0 = new is worse). Returns:
      "adopt"       speed-up lower bound >= min_speedup AND loss upper bound <= margin
      "keep dense"  speed-up upper bound < min_speedup OR loss lower bound > margin
      "inconclusive" otherwise
    """
    raise NotImplementedError("TODO 4: the decision rule")
