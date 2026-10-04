"""Failure rates, checkpoint intervals and goodput (lesson 09.4).

Symbols (all times in the same unit, hours here):

* ``M``     mean time between interruptions of the *job* (any failure of any of its devices stops it);
* ``delta`` time to write one checkpoint (the part that blocks training);
* ``R``     time to restart: detect the failure, replace or exclude the node, relaunch, load the checkpoint;
* ``tau``   compute time between checkpoints;
* ``T_s``   failure-free compute time the run needs.

Failures are modelled as a Poisson process (memoryless), the standard assumption of Young (1974) and
Daly (2006). With ``N`` devices that fail independently with mean time ``m_dev`` each, ``M = m_dev / N``:
doubling the cluster halves the job's mean time between failures.

* Young's first-order optimum (CACM 17(9), 1974): ``tau = sqrt(2 · delta · M)``.
* Daly's estimate (Future Generation Computer Systems 22(3), 2006), in the form most often quoted:
  ``tau = sqrt(2·delta·M) - delta`` for ``delta < M/2``, else ``tau = M``. (The paper also gives a longer
  perturbation series; the course uses the numerical optimum below as the reference instead.)
* Expected wall time under the Poisson model, exact: each segment of ``tau + delta`` is retried until it runs
  without a failure, and every failure costs a restart ``R`` that can itself be interrupted:
  ``E[segment] = M · exp(R/M) · (exp((tau + delta)/M) - 1)``, so ``T_w = E[segment] · T_s / tau``.
  :func:`simulate_wall` checks this by Monte Carlo (``tests/test_dist.py``).
* Goodput = useful compute / wall time = ``T_s / T_w``. First-order waste per unit time, for small
  ``tau``, ``delta`` and ``R`` relative to ``M``: ``delta/tau`` (writing checkpoints) ``+ (tau/2 + R)/M``
  (on average half an interval is recomputed after each failure, plus the restart).

The Llama 3 report (arXiv 2407.21783, section 3.3.4) gives the evidence this lesson uses: 466 job
interruptions in a 54-day snapshot, 47 planned and 419 unexpected, about 78% of the unexpected ones
attributed to confirmed or suspected hardware issues, and higher than 90% effective training time.
"""

from __future__ import annotations

import math

LLAMA3 = {"days": 54, "interruptions": 466, "planned": 47, "unexpected": 419, "hardware_share_of_unexpected": 0.78,
          "gpus": 16384, "effective_training_time": 0.90,
          "source": "arXiv 2407.21783 section 3.3.4 (the GPU count is the largest Table 4 configuration; the report does not "
                    "say the snapshot ran on exactly that many GPUs)"}


def mtbf(hours: float, interruptions: int) -> float:
    """Mean time between interruptions (hours) from an observed count over a period."""
    return hours / interruptions


def job_mtbf(device_mtbf: float, n_devices: int) -> float:
    """Job MTBF when any of ``n_devices`` independent devices stops it."""
    return device_mtbf / n_devices


def device_mtbf(job_mtbf_hours: float, n_devices: int) -> float:
    """The per-device MTBF implied by a job MTBF on ``n_devices`` (the inverse of :func:`job_mtbf`)."""
    return job_mtbf_hours * n_devices


def young(delta: float, M: float) -> float:
    return math.sqrt(2.0 * delta * M)


def daly(delta: float, M: float) -> float:
    if delta >= 0.5 * M:
        return M
    return math.sqrt(2.0 * delta * M) - delta


def expected_wall(T_s: float, tau: float, delta: float, M: float, R: float) -> float:
    """Expected wall time to finish ``T_s`` of compute in segments of ``tau`` (Poisson failures, mean ``M``)."""
    return M * math.exp(R / M) * (math.exp((tau + delta) / M) - 1.0) * T_s / tau


def simulate_wall(T_s: float, tau: float, delta: float, M: float, R: float, runs: int = 2000, seed: int = 0) -> float:
    """Monte Carlo of the same process: mean wall time over ``runs`` simulated runs (a check on the formula)."""
    import random
    rng = random.Random(seed)
    n_seg = max(1, round(T_s / tau))
    total = 0.0
    for _ in range(runs):
        t = 0.0
        for _ in range(n_seg):
            need = tau + delta
            while True:
                fail = rng.expovariate(1.0 / M)
                if fail >= need:
                    t += need
                    break
                t += fail                             # lost work up to the failure
                while True:                           # restart; a failure during it restarts the restart
                    f2 = rng.expovariate(1.0 / M)
                    if f2 >= R:
                        t += R
                        break
                    t += f2
        total += t
    return total / runs


def goodput(tau: float, delta: float, M: float, R: float) -> float:
    """Fraction of wall time that is useful compute, under Daly's complete model."""
    return 1.0 / expected_wall(1.0, tau, delta, M, R)


def first_order_waste(tau: float, delta: float, M: float, R: float) -> float:
    """Approximate fraction of time lost: checkpoint writes + recomputed work + restarts."""
    return delta / tau + (tau / 2.0 + R) / M


def optimal_tau(delta: float, M: float, R: float = 0.0, lo: float | None = None, hi: float | None = None) -> float:
    """Numerical optimum of :func:`goodput` (golden-section search); a check on Young's and Daly's formulas."""
    lo = lo if lo is not None else 1e-6 * M
    hi = hi if hi is not None else 10.0 * M
    g = (math.sqrt(5) - 1) / 2
    a, b = math.log(lo), math.log(hi)                 # search in log(tau): the curve is flat near the optimum
    c, d = b - g * (b - a), a + g * (b - a)
    f = lambda x: -goodput(math.exp(x), delta, M, R)    # noqa: E731
    for _ in range(200):
        if f(c) < f(d):
            b, d = d, c
            c = b - g * (b - a)
        else:
            a, c = c, d
            d = a + g * (b - a)
    return math.exp((a + b) / 2)


def plan_table(delta: float, M: float, R: float) -> dict:
    """Young, Daly and the numerical optimum with the goodput each gives."""
    out = {}
    for name, tau in (("young", young(delta, M)), ("daly", daly(delta, M)), ("numeric", optimal_tau(delta, M, R))):
        out[name] = {"tau": tau, "goodput": goodput(tau, delta, M, R),
                     "first_order_goodput": 1.0 - first_order_waste(tau, delta, M, R)}
    return out


__all__ = ["LLAMA3", "daly", "device_mtbf", "expected_wall", "first_order_waste", "goodput", "job_mtbf", "mtbf",
           "optimal_tau", "plan_table", "simulate_wall", "young"]
