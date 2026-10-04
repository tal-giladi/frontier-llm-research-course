"""Lab 06.5 — elastic architectures (extension). Fill in the TODOs; run `pytest labs/module-06/lesson-05`."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: F401


def nested_ffn(x, gate_w, up_w, down_w, m: int):
    """The MatFormer FFN at granularity m (paper section 3.1): only the first m hidden neurons.

    gate_w, up_w: (I, C); down_w: (C, I); x (..., C). Return W_down[:, :m] (silu(W_gate[:m] x) * (W_up[:m] x)).
    """
    raise NotImplementedError("TODO 1: the nested SwiGLU")


def mix_n_match(widths: list[int], num_layers: int, budget: float) -> list[int]:
    """Mix'n'Match (section 3.3): choose a width per layer with mean as close as possible to ``budget``, using the
    paper's heuristic — widths never shrink with depth and consecutive layers differ by at most one granularity.
    Here: the first ``split`` layers at granularity i and the rest at i + 1, searched over i and split."""
    raise NotImplementedError("TODO 2: Mix'n'Match")


def consistency(a: torch.Tensor, b: torch.Tensor) -> float:
    """Share of positions where two models' greedy next-token predictions (token ids, same shape) agree."""
    raise NotImplementedError("TODO 3: consistency")


def ple_params(vocab_per_layer: int, num_layers: int, dim: int) -> int:
    """Parameters of a per-layer embedding table (Gemma 3n): each token id has num_layers vectors of size dim."""
    raise NotImplementedError("TODO 4: the PLE table size")
