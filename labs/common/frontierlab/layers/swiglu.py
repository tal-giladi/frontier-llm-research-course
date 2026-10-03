# Copied from the LLM Research Engineer course (llmre/model/swiglu.py, lesson 12.2) via moelab, so frontierlab stands alone.
"""SwiGLU — a gated feed-forward network (Module 12, lesson 02).

The GPT-2 MLP (Module 5, ``llmre.model.block.MLP``) is ``Linear -> GELU ->
Linear`` with a ``4C`` hidden width. LLaMA (``papers/index.md`` #9) replaces it
with **SwiGLU**, a *gated* FFN (Shazeer 2020):

    SwiGLU(x) = ( Swish(x W_gate) ⊙ x W_up ) W_down

where ``Swish(z) = z · sigmoid(z)`` (a smooth ReLU, ``F.silu`` in PyTorch) and
``⊙`` is elementwise multiply. There are **three** weight matrices instead of
two: a ``gate`` and an ``up`` projection (both ``C -> hidden``) whose outputs are
multiplied together, then a ``down`` projection (``hidden -> C``). The elementwise
product is the "gate": the ``up`` signal is modulated by a data-dependent,
Swish-activated mask.

Because SwiGLU uses three matrices, LLaMA shrinks the hidden width to
``hidden ≈ (8/3)·C`` (often rounded to a hardware-friendly multiple) so the total
parameter count matches the ``4C`` two-matrix GELU MLP: ``3 · (8/3) = 8 = 2 · 4``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """Gated SwiGLU feed-forward network, from scratch.

    Computes ``W_down( Swish(W_gate x) ⊙ W_up x )`` position-wise.

    Args:
        dim: model / residual width ``C`` (input and output size).
        hidden_dim: inner width. If ``None``, uses ``(8/3)·dim`` rounded to the
            nearest ``multiple_of`` so that the three-matrix SwiGLU has about the
            same parameter count as a ``4·dim`` two-matrix GELU MLP.
        bias: whether the three ``Linear`` layers carry a bias (LLaMA uses
            ``False``).
        multiple_of: round the derived ``hidden_dim`` up to a multiple of this
            (only used when ``hidden_dim is None``).

    Shapes:
        input / output ``(B, T, C)``; the two gate/up activations are
        ``(B, T, hidden)``.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        bias: bool = False,
        multiple_of: int = 1,
    ):
        super().__init__()
        if hidden_dim is None:
            # (8/3)·dim keeps params ~equal to the 4·dim GELU MLP; round up to
            # a multiple for hardware friendliness (LLaMA rounds to 256).
            hidden_dim = int(8 * dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.hidden_dim = hidden_dim
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)  # C -> hidden
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)    # C -> hidden
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)  # hidden -> C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated FFN to ``x`` (shape ``(B, T, C)``)."""
        # Swish(gate) elementwise-times up, then project back down. F.silu is
        # exactly Swish with beta=1: silu(z) = z * sigmoid(z).
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
