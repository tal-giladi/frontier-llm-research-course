"""Lab 17.5 (extension) — interpretable by design, and introspection experiments. Fill in the TODOs; run
`pytest labs/module-17/lesson-05`.

``sparse_lab.py`` trains dense and weight-sparse transformers on the closing-quote task, prunes each to its
smallest circuit and compares their sizes; with ``--hf`` it runs a concept-injection experiment with controls on
Qwen3-0.6B.
"""

from __future__ import annotations

import torch


def magnitude_mask(W: torch.Tensor, frac: float) -> torch.Tensor:
    """Boolean mask of the same shape as W that keeps the round(frac · numel) entries of largest |W|
    (at least 1). Ties at the threshold may keep a few more."""
    raise NotImplementedError("TODO 1: magnitude top-k mask")


def density_at(step: int, steps: int, target: float) -> float:
    """Fraction of weights kept after ``step``: linear from 1.0 at step 0 to ``target`` at steps/2, then
    constant (the anneal of Gao et al. 2025, appendix A.2, in its simplest form)."""
    raise NotImplementedError("TODO 2: the density schedule")


def gated(a: torch.Tensor, gate: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """Mean ablation by gates: a · gate + mean · (1 − gate); gate and mean broadcast over the last dimension."""
    raise NotImplementedError("TODO 3: gated mean ablation")


def rates(grades: list[dict]) -> dict:
    """From per-trial grades {"claims": bool, "names": bool}: the fraction that claim detection, and the
    fraction that claim detection AND name the concept (the paper's success criterion)."""
    raise NotImplementedError("TODO 4: detection rates")
