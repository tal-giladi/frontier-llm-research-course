"""Reference solution for lab 07.3."""

from __future__ import annotations

import math


def _hidden(name, ndim):
    return ndim == 2 and ("self_attn." in name or "mlp." in name) and "norm" not in name


def width_mult(width, base_width):
    return width / base_width


def mup_init_std(name, ndim, m, std=0.02):
    if _hidden(name, ndim):
        return std / math.sqrt(m)
    if "embed_tokens" in name:
        return std
    return None


def mup_lr_scale(name, ndim, m, optimizer="adamw", muon_adjust="match_rms"):
    if not _hidden(name, ndim):
        return 1.0
    if optimizer == "muon":
        return 1 / math.sqrt(m) if muon_adjust == "match_rms" else 1.0
    return 1 / m


def readout_mult(m):
    return 1 / m


def best_lr(losses):
    return min(sorted(losses), key=lambda lr: losses[lr])


def transfer_verdict(sweep, grid):
    grid = sorted(grid)
    best = {w: best_lr(v) for w, v in sweep.items()}
    w0 = min(best)
    shift = max(abs(grid.index(best[w]) - grid.index(best[w0])) for w in best)
    return {"best": best, "shift": shift, "transfers": shift <= 1}
