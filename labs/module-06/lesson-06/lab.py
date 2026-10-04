"""Lab 06.6 — entropy patching (extension). Fill in the TODOs; run `pytest labs/module-06/lesson-06` to check."""

from __future__ import annotations

import math  # noqa: F401


def entropy(probs: list[float]) -> float:
    """Entropy in nats of a next-byte distribution, H = -sum_v p_v log p_v (BLT Eq. 1 — with the minus sign the
    printed equation is missing). Entries with p = 0 contribute 0."""
    raise NotImplementedError("TODO 1: entropy")


def patch_starts(H: list[float], threshold: float, mode: str = "global") -> list[int]:
    """Indices of bytes that start a new patch (BLT section 2.3). Byte 0 always does.
    "global": byte i starts a patch if H[i] > threshold.
    "monotonic": byte i starts a patch if H[i] - H[i-1] > threshold (it breaks an approximately decreasing run)."""
    raise NotImplementedError("TODO 2: the two boundary rules")


def mean_patch_size(starts: list[int], n: int) -> float:
    """Average patch length in bytes for ``n`` bytes split at ``starts``."""
    raise NotImplementedError("TODO 3: average patch size")


def flops_per_byte(global_per_patch: float, local_per_byte: float, avg_patch: float) -> float:
    """BLT's cost model (section 4.5): the large latent transformer runs once per patch, the small local models
    once per byte. Return FLOPs per byte for an average patch of ``avg_patch`` bytes."""
    raise NotImplementedError("TODO 4: FLOPs per byte")
