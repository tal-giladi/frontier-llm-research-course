"""Benchmark harness: warm-up, synchronisation, repeats and an interval (Module 2, lesson 02.4).

Four rules every timing in this course follows:

1. **Warm up.** The first calls pay for one-off work: CUDA context and cuBLAS handle creation,
   kernel autotuning, ``torch.compile`` graph capture and code generation, allocator growth, CPU
   caches. They are run and thrown away.
2. **Synchronise.** CUDA kernels are queued asynchronously; ``time.perf_counter()`` around a GPU
   call measures how long it took to *queue* the work unless ``torch.cuda.synchronize()`` is called
   before reading the clock. :func:`sync` does that (and nothing on CPU).
3. **Repeat and report spread.** One number is not a measurement. Report the median of ``repeats``
   samples with a bootstrap interval (:func:`frontierlab.stats.bootstrap_ci`).
4. **Interleave when comparing.** Run A, B, A, B, ... (:func:`interleaved`) so slow drift (thermal
   throttling, a background process, a noisy neighbour on a shared GPU) hits both arms equally, then
   compare the paired per-round ratios (:func:`speedup`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from frontierlab.stats import bootstrap_ci


def sync(device: str | torch.device = "cpu") -> None:
    """Wait until every queued kernel on ``device`` has finished (no-op on CPU)."""
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


@dataclass
class Timing:
    samples: list[float] = field(default_factory=list)       # seconds, one per repeat
    warmup: list[float] = field(default_factory=list)        # seconds of the discarded warm-up calls

    @property
    def median(self) -> float:
        return float(np.median(self.samples))

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples))

    @property
    def std(self) -> float:
        return float(np.std(self.samples, ddof=1)) if len(self.samples) > 1 else float("nan")

    def ci(self, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
        """Bootstrap interval of the median."""
        _, lo, hi = bootstrap_ci(self.samples, stat=np.median, n_boot=2000, alpha=alpha, seed=seed)
        return lo, hi

    def __str__(self) -> str:
        lo, hi = self.ci()
        return (f"median {self.median * 1e3:.3f} ms  95% CI [{lo * 1e3:.3f}, {hi * 1e3:.3f}]  "
                f"(n={len(self.samples)}, cv={self.std / self.mean:.1%})")


def benchmark(fn, *, warmup: int = 3, repeats: int = 20, device: str | torch.device = "cpu",
              setup=None) -> Timing:
    """Time ``fn()``: ``warmup`` discarded calls, then ``repeats`` synchronised samples.

    ``setup()`` (optional) runs before each call, outside the timed region (for example to zero
    gradients or to make fresh inputs), so it is not counted.
    """
    out = Timing()
    for _ in range(warmup):
        if setup is not None:
            setup()
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        out.warmup.append(time.perf_counter() - t0)
    for _ in range(repeats):
        if setup is not None:
            setup()
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        out.samples.append(time.perf_counter() - t0)
    return out


def interleaved(fns: dict, *, warmup: int = 3, rounds: int = 20, device: str | torch.device = "cpu",
                setup=None) -> dict[str, Timing]:
    """Time several functions in alternating rounds (A, B, A, B, ...). Returns one Timing per name."""
    res = {k: Timing() for k in fns}
    for phase, n in (("warmup", warmup), ("samples", rounds)):
        for _ in range(n):
            for name, fn in fns.items():
                if setup is not None:
                    setup()
                sync(device)
                t0 = time.perf_counter()
                fn()
                sync(device)
                getattr(res[name], phase).append(time.perf_counter() - t0)
    return res


def speedup(base: Timing, new: Timing, alpha: float = 0.05, n_boot: int = 4000, seed: int = 0) -> dict:
    """Speed-up of ``new`` over ``base`` (>1 means new is faster) with a bootstrap interval.

    If both have the same number of samples they are treated as paired rounds (from
    :func:`interleaved`) and the per-round ratios are resampled; otherwise the two sample sets are
    resampled independently and the ratio of medians is used.
    """
    a, b = np.asarray(base.samples), np.asarray(new.samples)
    rng = np.random.default_rng(seed)
    if a.size == b.size:
        r = a / b
        boots = np.median(r[rng.integers(0, r.size, (n_boot, r.size))], axis=1)
        point = float(np.median(r))
    else:
        ba = np.median(a[rng.integers(0, a.size, (n_boot, a.size))], axis=1)
        bb = np.median(b[rng.integers(0, b.size, (n_boot, b.size))], axis=1)
        boots, point = ba / bb, float(np.median(a) / np.median(b))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"speedup": point, "ci": (float(lo), float(hi))}
