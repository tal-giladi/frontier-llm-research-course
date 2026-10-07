"""A discrete-event model of synchronous vs asynchronous RL with bounded staleness (lesson 14.3).

One rollout engine generates batches; one trainer consumes them. Batch ``j`` is generated with the weights
the trainer had when generation started (version ``v_j`` = number of updates finished by then) and is used
for update ``j``. Its **staleness** (lag) is ``j - v_j``. A bound ``k`` means generation of batch ``j`` may
not start before update ``j - k`` has produced its weights (version ``>= j - k``), so every lag is ``<= k``:

    s_j = max(end of generation j-1, time version j-k exists)        generation start
    e_j = s_j + g_j                                                   generation end (g_j = this batch's time)
    t_j = max(e_j, end of update j-1);  u_j = t_j + T                 update start and end

``k = 0`` is synchronous on-policy RL: generate, then train, then generate with the new weights. With
``k >= 1`` generation of the next batch overlaps the current update (one-step off-policy for ``k = 1``).

Generation time of a batch is set by its *longest* response when the engine decodes the batch together
(``batch_gen_time``): with heavy-tailed lengths a few long responses keep the engine busy while the rest are
finished, which is the cost that asynchronous designs, partial rollouts and in-flight weight updates attack.
"""

from __future__ import annotations

import numpy as np


def batch_gen_time(lengths: np.ndarray, sec_per_token: float, overhead: float = 0.0) -> float:
    """Seconds to decode one batch whose responses have ``lengths`` tokens (bounded by the longest)."""
    return overhead + float(np.max(lengths)) * sec_per_token


def lognormal_lengths(rng: np.random.Generator, n_batches: int, batch: int, median: float, sigma: float,
                      cap: int) -> np.ndarray:
    """(n_batches, batch) response lengths, log-normal around ``median``, truncated at the budget ``cap``."""
    return np.minimum(np.round(median * np.exp(sigma * rng.standard_normal((n_batches, batch)))), cap).clip(1)


def simulate(gen_times, train_time: float, bound: int) -> dict:
    """Run the schedule above for ``len(gen_times)`` updates.

    Returns total time, updates per second, per-update lags, and the fraction of wall-clock each side was busy.
    """
    g = np.asarray(gen_times, dtype=float)
    n = g.size
    gen_end = np.zeros(n)
    upd_end = np.zeros(n)
    lags = np.zeros(n, dtype=int)
    prev_gen_end = 0.0
    for j in range(n):
        need = j - bound                      # generation j needs version >= j - bound, i.e. update need-1 done
        ready = 0.0 if need <= 0 else upd_end[need - 1]
        s = max(prev_gen_end, ready)
        version = int(np.searchsorted(upd_end[:j], s, side="right")) if j > 0 else 0
        lags[j] = j - version
        gen_end[j] = s + g[j]
        prev_gen_end = gen_end[j]
        t = max(gen_end[j], upd_end[j - 1] if j > 0 else 0.0)
        upd_end[j] = t + train_time
    total = float(upd_end[-1])
    return {"bound": bound, "total_s": total, "updates_per_s": n / total, "lags": lags.tolist(),
            "mean_lag": float(lags.mean()), "max_lag": int(lags.max()),
            "gen_busy": float(g.sum() / total), "train_busy": float(n * train_time / total)}


def speedup_table(gen_times, train_time: float, bounds=(0, 1, 2, 4, 8)) -> list[dict]:
    base = simulate(gen_times, train_time, 0)["total_s"]
    rows = []
    for k in bounds:
        r = simulate(gen_times, train_time, k)
        r["speedup"] = base / r["total_s"]
        rows.append(r)
    return rows
