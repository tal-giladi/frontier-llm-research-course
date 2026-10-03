"""Reference solution for lab 02.3."""

from __future__ import annotations

import numpy as np


def bus_bandwidth(nbytes: float, seconds: float, world: int, op: str = "all_reduce") -> float:
    factor = {"all_reduce": 2.0 * (world - 1) / world, "reduce_scatter": (world - 1) / world,
              "all_gather": (world - 1) / world}[op]
    return factor * nbytes / seconds


def exposed_comm(sync_times, nosync_times, n_boot: int = 2000, seed: int = 0) -> dict:
    a, b = np.asarray(sync_times, dtype=np.float64), np.asarray(nosync_times, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = (np.median(a[rng.integers(0, a.size, (n_boot, a.size))], axis=1)
             - np.median(b[rng.integers(0, b.size, (n_boot, b.size))], axis=1))
    exposed = float(np.median(a) - np.median(b))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"exposed_s": exposed, "fraction": exposed / float(np.median(a)), "ci": (float(lo), float(hi))}


def overlap_fraction(comm_s: float, exposed_s: float) -> float:
    if comm_s <= 0:
        return 1.0
    return float(min(1.0, max(0.0, 1.0 - exposed_s / comm_s)))
