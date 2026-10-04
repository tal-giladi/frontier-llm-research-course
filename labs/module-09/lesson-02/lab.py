"""Lab 09.2 — pipeline schedules and their bubbles. Fill in the TODOs; run `pytest labs/module-09/lesson-02`."""

from __future__ import annotations


def bubble_ratio(schedule: str, p: int, m: int, v: int = 1) -> float:
    """Fraction of device time that is idle, for p stages and m micro-batches, F and B fixed per micro-batch.

    GPipe and 1F1B: every device is idle for (p-1) micro-batch slots out of (m + p - 1), so (p-1)/(m+p-1).
    Interleaved 1F1B with v chunks per device: each chunk is 1/v of the work, so the idle slots shrink by v:
    (p-1)/(v·m + p-1). ``schedule`` is "gpipe", "1f1b" or "interleaved".
    """
    raise NotImplementedError("TODO 1: closed-form bubble ratio")


def one_f_one_b_order(p: int, s: int, m: int) -> list[tuple[str, int]]:
    """The op order of stage ``s`` (0-based) under 1F1B: [("F", 0), ("F", 1), ..., ("B", 0), ...].

    Warm-up: min(p - s - 1, m) forwards. Steady state: alternate one forward (the next micro-batch) and one
    backward (the oldest micro-batch not yet backpropagated). Cool-down: the remaining backwards.
    """
    raise NotImplementedError("TODO 2: 1F1B order")


def peak_in_flight(order: list[tuple[str, int]]) -> int:
    """Largest number of micro-batches whose forward has run but whose backward has not (their activations
    are held in memory), over an op order like the one above."""
    raise NotImplementedError("TODO 3: peak live micro-batches")


def measured_bubble(busy_s: list[float], wall_s: list[float]) -> float:
    """Measured bubble of one rank: 1 - median(busy / wall) over the timed steps (busy = time computing)."""
    raise NotImplementedError("TODO 4: measured bubble")
