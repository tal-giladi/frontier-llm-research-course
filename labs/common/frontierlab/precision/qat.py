"""Quantisation-aware training (QAT) and post-training quantisation (PTQ) helpers (lesson 08.3).

PTQ rounds trained weights once (:func:`frontierlab.precision.linear.ptq_`). QAT trains with the rounding in the
forward pass so the weights learn to sit where rounding hurts least:

    forward    y = x · q(W)ᵀ                    q = quantise-dequantise ("fake quantisation")
    backward   ∂L/∂x = ∂L/∂y · q(W)             (the function the forward computed)
               ∂L/∂W ≈ ∂L/∂q(W) = (∂L/∂y)ᵀ x     straight-through estimator (STE): treat dq/dW as 1

The master weights stay in high precision; only their rounded copy is used. At the end, :func:`export_` replaces
each fake-quantised layer by a plain ``nn.Linear`` holding q(W), which is exactly what the model saw in training.

Published uses (see lesson 08.3 for the quotes): Kimi K2 Thinking (INT4 weight-only QAT of the MoE components in
post-training; its config.json: symmetric INT4, group size 32), DeepSeek-V4 (MXFP4 QAT of routed-expert weights in
post-training, dequantised losslessly to FP8 for compute). gpt-oss used post-training *with* MXFP4 MoE weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from frontierlab.precision.linear import QLinear, Recipe, swap_linears
from frontierlab.precision.quant import get_spec, qdq


class _FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, spec):
        return qdq(x, spec)

    @staticmethod
    def backward(ctx, g):
        return g, None


def fake_quant(x: torch.Tensor, spec) -> torch.Tensor:
    """qdq(x) in the forward pass, identity in the backward pass (straight-through estimator)."""
    return _FakeQuantSTE.apply(x, get_spec(spec))


def qat_recipe(spec) -> Recipe:
    """Weight-only fake quantisation with ``spec`` (activations and gradients untouched)."""
    spec = get_spec(spec)
    return Recipe(f"qat-{spec.fmt}-{spec.block[0]}x{spec.block[1]}", None, spec, None, dgrad_weight="forward")


def attach_qat(model: nn.Module, spec, keep_blocks=frozenset(), only: str = "all") -> list[str]:
    """Swap the selected linears for weight-only fake-quantised :class:`QLinear` layers (same parameters)."""
    return swap_linears(model, qat_recipe(spec), keep_blocks, only)


@torch.no_grad()
def export_(model: nn.Module) -> int:
    """Replace every :class:`QLinear` by an ``nn.Linear`` whose weight is the recipe's quantised forward weight.
    Returns the number of layers exported. (Only meaningful for weight-only recipes such as QAT.)"""
    n = 0
    for name, m in list(model.named_modules()):
        if isinstance(m, QLinear):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            lin = nn.Linear(m.in_features, m.out_features, bias=False, device=m.weight.device, dtype=m.weight.dtype)
            lin.weight.copy_(qdq(m.weight, m.recipe.weight))
            setattr(parent, child, lin)
            n += 1
    return n
