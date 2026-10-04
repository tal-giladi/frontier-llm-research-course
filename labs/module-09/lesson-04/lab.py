"""Lab 09.4 — failure and recovery. Fill in the TODOs; run `pytest labs/module-09/lesson-04`."""

from __future__ import annotations

import math  # noqa: F401  (TODOs 1-2)
from pathlib import Path


def young(delta: float, M: float) -> float:
    """Young's checkpoint interval: sqrt(2 · delta · M) (delta = checkpoint write time, M = job MTBF)."""
    raise NotImplementedError("TODO 1: Young's interval")


def goodput(tau: float, delta: float, M: float, R: float) -> float:
    """Useful fraction of wall time with Poisson failures (mean M), checkpoints every tau of compute that take
    delta, and a restart cost R after each failure.

    Each segment of tau + delta is retried until it runs with no failure; its expected wall time is
    M · exp(R/M) · (exp((tau + delta)/M) - 1). Goodput = tau / that.
    """
    raise NotImplementedError("TODO 2: goodput")


def latest_committed(ckpt_root: str | Path) -> Path | None:
    """The newest checkpoint folder ``step_NNNNNN`` under ``ckpt_root`` that contains a ``COMMITTED`` file, or None.

    A folder without ``COMMITTED`` is a save that was interrupted; it must never be loaded."""
    raise NotImplementedError("TODO 3: pick the checkpoint to resume from")


def lost_work(records: list[dict]) -> list[int]:
    """Steps recomputed after each restart, from a run's ``metrics.jsonl`` records (in file order).

    Records have ``event`` "start" (with ``from_step``, the step the launch resumed from) or "step" (with ``step``).
    For every "start" except the first, the lost work is the last step logged before it minus its ``from_step``
    (0 if nothing was logged after the previous start). Return one number per restart.
    """
    raise NotImplementedError("TODO 4: lost work per restart")
