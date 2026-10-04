"""Reference solution for lab 06.6 (entropy patching, extension)."""

from __future__ import annotations

import math


def entropy(probs: list[float]) -> float:
    """H = -sum p log p (nats); zero-probability entries contribute 0."""
    return -sum(p * math.log(p) for p in probs if p > 0)


def patch_starts(H: list[float], threshold: float, mode: str = "global") -> list[int]:
    """BLT section 2.3: byte i starts a patch if H[i] > theta (global) or H[i] - H[i-1] > theta (monotonic).
    Byte 0 always starts a patch."""
    starts = [0]
    for i in range(1, len(H)):
        if (mode == "global" and H[i] > threshold) or (mode == "monotonic" and H[i] - H[i - 1] > threshold):
            starts.append(i)
    return starts


def mean_patch_size(starts: list[int], n: int) -> float:
    return n / len(starts)


def flops_per_byte(global_per_patch: float, local_per_byte: float, avg_patch: float) -> float:
    """The latent (global) model runs once per patch, the local encoder/decoder once per byte."""
    return global_per_patch / avg_patch + local_per_byte
