"""Entropy patching for byte-level models, as in the Byte Latent Transformer (lesson 06.6, extension).

BLT (Pagnoni et al., arXiv 2412.09871, section 2.3) groups bytes into variable-length *patches*; a large
latent transformer runs once per patch, small local models run per byte. A patch boundary goes where the
next byte is hard to predict, measured by a small byte-level LM p_e:

    H(x_i) = - sum_v p_e(x_i = v | x_<i) log p_e(x_i = v | x_<i)
    global constraint:          byte i starts a new patch if H(x_i) > theta_g
    approx. monotonic constraint: byte i starts a new patch if H(x_i) − H(x_{i−1}) > theta_r

(The paper's Eq. 1 is printed without the minus sign; the entropy needs it.) The threshold is chosen to
hit a target average patch size on the training data (section 4.3); since the latent model runs once per
patch, its FLOPs per byte fall roughly as 1 / (average patch size), which is BLT's lever (section 4.5).

The entropy model here is a count-based order-k byte model with add-alpha smoothing — BLT notes that
"when the receptive field of the model is small enough, the trained entropy model can be encoded in an
efficient lookup table" (section 4.2); its default is a 100M-parameter byte transformer.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict


class NgramByteModel:
    """p(x_i | the previous ``order`` bytes) from counts, add-``alpha`` smoothed over 256 byte values."""

    def __init__(self, data: bytes, order: int = 2, alpha: float = 0.1):
        self.order, self.alpha = order, alpha
        self.counts: dict[bytes, Counter] = defaultdict(Counter)
        for i in range(len(data)):
            self.counts[data[max(0, i - order):i]][data[i]] += 1

    def entropy(self, context: bytes) -> float:
        c = self.counts.get(context[-self.order:] if self.order else b"")
        if not c:
            return math.log(256)
        total = sum(c.values()) + self.alpha * 256
        h = 0.0
        for v in range(256):
            p = (c.get(v, 0) + self.alpha) / total
            h -= p * math.log(p)
        return h

    def entropies(self, data: bytes) -> list[float]:
        """H(x_i) for every byte position i of ``data`` (context = the bytes before it)."""
        return [self.entropy(data[max(0, i - self.order):i]) for i in range(len(data))]


def patch_starts(H: list[float], threshold: float, mode: str = "global") -> list[int]:
    """Indices of the bytes that start a patch. Byte 0 always starts one."""
    starts = [0]
    for i in range(1, len(H)):
        if (mode == "global" and H[i] > threshold) or (mode == "monotonic" and H[i] - H[i - 1] > threshold):
            starts.append(i)
    return starts


def patch_lengths(starts: list[int], n: int) -> list[int]:
    return [b - a for a, b in zip(starts, starts[1:] + [n])]


def threshold_for_size(H: list[float], target: float, mode: str = "global", iters: int = 50) -> float:
    """Bisection on theta so the average patch size is about ``target`` bytes (larger theta -> larger patches)."""
    lo, hi = (min(H) - 1.0, max(H) + 1.0) if mode == "global" else (-max(H) - 1.0, max(H) + 1.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        size = len(H) / len(patch_starts(H, mid, mode))
        lo, hi = (mid, hi) if size < target else (lo, mid)
    return (lo + hi) / 2


def latent_flops_per_byte(flops_per_patch: float, avg_patch_size: float) -> float:
    """The global model runs once per patch: its FLOPs per byte are flops_per_patch / average patch size."""
    return flops_per_patch / avg_patch_size
