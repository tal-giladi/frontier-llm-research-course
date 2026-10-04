"""Lab 07.3 — µP relative to a base width, and a transfer test. Fill in the TODOs; run `pytest labs/module-07/lesson-03`.

Conventions (lesson 07.3): m = width / base_width. "hidden" = a 2-D weight inside a block (attention q/k/v/o,
SwiGLU gate/up/down); the embedding is not hidden; with tied embeddings the head is the embedding.
"""

from __future__ import annotations


def width_mult(width: int, base_width: int) -> float:
    """m = width / base_width."""
    raise NotImplementedError("TODO 1: the width multiplier")


def mup_init_std(name: str, ndim: int, m: float, std: float = 0.02) -> float | None:
    """Init std of a parameter under µP: hidden matrices std / sqrt(m), the embedding std; None for vectors."""
    raise NotImplementedError("TODO 2: µP init std")


def mup_lr_scale(name: str, ndim: int, m: float, optimizer: str = "adamw", muon_adjust: str = "match_rms") -> float:
    """Learning-rate factor under µP. AdamW: hidden 1/m, everything else 1.

    Muon hidden matrices (spectral rule): 1/sqrt(m) with "match_rms", 1 with "original". Others 1.
    """
    raise NotImplementedError("TODO 3: µP learning-rate factor")


def readout_mult(m: float) -> float:
    """Multiplier on the logits of a tied output head under µP."""
    raise NotImplementedError("TODO 4: readout multiplier")


def best_lr(losses: dict) -> float:
    """{lr: final loss} -> the lr with the lowest loss (ties: the smaller lr)."""
    raise NotImplementedError("TODO 5: pick the best learning rate")


def transfer_verdict(sweep: dict, grid: list[float]) -> dict:
    """Decision rule of the transfer test.

    sweep = {width: {lr: loss}} for one parametrization; grid = the sorted learning rates swept.
    Return {"best": {width: best lr}, "shift": the largest number of grid steps between the best lr at the
    smallest width and the best lr at any other width, "transfers": shift <= 1}.
    """
    raise NotImplementedError("TODO 6: transfer verdict")
