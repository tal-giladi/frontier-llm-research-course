"""Lab 02.3 — communication and overlap. Fill in the TODOs; run `pytest labs/module-02/lesson-03` to check."""

from __future__ import annotations

import numpy as np  # noqa: F401  (TODO 2)


def bus_bandwidth(nbytes: float, seconds: float, world: int, op: str = "all_reduce") -> float:
    """Bus bandwidth (bytes/s) of one collective on an ``nbytes`` buffer that took ``seconds``.

    Ring algorithm: an all-reduce moves 2·(n-1)/n · nbytes per rank; a reduce-scatter or an
    all-gather moves (n-1)/n · nbytes. Bus bandwidth = bytes moved per rank / seconds. ``op`` is
    "all_reduce", "reduce_scatter" or "all_gather".
    """
    raise NotImplementedError("TODO 1: bus bandwidth")


def exposed_comm(sync_times, nosync_times, n_boot: int = 2000, seed: int = 0) -> dict:
    """Exposed communication from two sets of step times (seconds).

    ``sync_times``: steps with gradient synchronisation; ``nosync_times``: the same steps with it
    switched off. Return ``{"exposed_s": median(sync) - median(nosync), "fraction": exposed_s /
    median(sync), "ci": (lo, hi)}`` where the 95% interval comes from resampling each list
    independently ``n_boot`` times (``np.random.default_rng(seed)``) and taking the 2.5% and 97.5%
    quantiles of the difference of medians.
    """
    raise NotImplementedError("TODO 2: exposed communication with an interval")


def overlap_fraction(comm_s: float, exposed_s: float) -> float:
    """Share of the communication time that the computation hid: 1 - exposed / comm, clipped to [0, 1].

    ``comm_s`` is how long the gradient communication takes on its own; return 1.0 if it is 0.
    """
    raise NotImplementedError("TODO 3: overlap fraction")
