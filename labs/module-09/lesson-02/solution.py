"""Reference solution for lab 09.2."""

from __future__ import annotations

import numpy as np


def bubble_ratio(schedule, p, m, v=1):
    if schedule in ("gpipe", "1f1b"):
        return (p - 1) / (m + p - 1)
    if schedule == "interleaved":
        return (p - 1) / (v * m + p - 1)
    raise ValueError(schedule)


def one_f_one_b_order(p, s, m):
    warm = min(p - s - 1, m)
    out = [("F", i) for i in range(warm)]
    for i in range(m - warm):
        out += [("F", warm + i), ("B", i)]
    return out + [("B", i) for i in range(m - warm, m)]


def peak_in_flight(order):
    live = best = 0
    for kind, _ in order:
        live += 1 if kind == "F" else -1
        best = max(best, live)
    return best


def measured_bubble(busy_s, wall_s):
    return float(1.0 - np.median(np.asarray(busy_s) / np.asarray(wall_s)))
