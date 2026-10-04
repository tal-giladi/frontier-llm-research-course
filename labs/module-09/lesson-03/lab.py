"""Lab 09.3 — a measured multi-GPU investigation. Fill in the TODOs; run `pytest labs/module-09/lesson-03`."""

from __future__ import annotations

import re  # noqa: F401  (TODO 1)

import numpy as np  # noqa: F401  (TODO 4)


def parse_titan_line(line: str) -> dict | None:
    """Parse one torchtitan 0.3.0 metrics line; return None for any other line.

    The trainer prints (torchtitan/components/metrics.py at v0.3.0, colour codes around each field):
        step: 12  loss:  7.12345  grad_norm:  1.2345  memory: 61.23GiB(77.40%)  tps: 6,123  tflops: 345.67  mfu: 34.95%
    Lines may start with a rank prefix such as "[rank3]:". Strip ANSI colour codes (ESC '[' ... 'm') first.
    Return {"rank": int or None, "step": int, "loss": float, "memory_gib": float, "memory_pct": float,
            "tps": float, "tflops": float, "mfu": float (percent)}.
    """
    raise NotImplementedError("TODO 1: parse a torchtitan metrics line")


def exposed_comm(comm: list[tuple[float, float]], compute: list[tuple[float, float]]) -> float:
    """Time during which communication kernels run and no compute kernel runs (from a profiler trace).

    ``comm`` and ``compute`` are lists of (start, end) intervals on one GPU (they may overlap within a list).
    Return the length of union(comm) minus the length of union(comm) ∩ union(compute).
    """
    raise NotImplementedError("TODO 2: exposed communication from intervals")


def state_bytes_per_rank(n_params: int, world: int, mode: str) -> int:
    """Bytes of parameters + gradients + AdamW state (exp_avg, exp_avg_sq) one rank holds, all fp32.

    "ddp": every rank holds everything (4 + 4 + 8 bytes per parameter). "fsdp2": every parameter is split along
    dim 0 over ``world`` ranks, so a rank holds 1/world of it — exactly, when ``n_params`` divides evenly. Return an
    int.
    """
    raise NotImplementedError("TODO 3: state bytes per rank")


def summarize(values: list[float], skip: int, n_boot: int = 2000, seed: int = 0) -> dict:
    """Median of ``values[skip:]`` with a 95% percentile-bootstrap interval of the median.

    Return {"median": ..., "lo": ..., "hi": ..., "n": number of values used}; use np.random.default_rng(seed),
    ``n_boot`` resamples, and the 2.5% / 97.5% quantiles.
    """
    raise NotImplementedError("TODO 4: median and interval")
