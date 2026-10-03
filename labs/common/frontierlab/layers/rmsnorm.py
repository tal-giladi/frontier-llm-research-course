# Copied from the LLM Research Engineer course (llmre/model/rmsnorm.py, lesson 12.2) via moelab, so frontierlab stands alone.
"""RMSNorm — Root-Mean-Square LayerNorm (Module 12, lesson 02).

LayerNorm (Module 5, ``llmre.model.block.LayerNorm``) does two things to each
length-``C`` feature vector: it **re-centers** it (subtract the mean) and
**re-scales** it (divide by the standard deviation), then applies a learned gain
and bias. RMSNorm (Zhang & Sennrich 2019; used by LLaMA, ``papers/index.md`` #9)
keeps only the re-scaling:

    y = x / RMS(x) · g,   RMS(x) = sqrt( mean(x²) + eps )

No mean subtraction, no bias. The observation is that the re-centering barely
matters for transformers, so dropping it is cheaper (one fewer reduction, no
subtract) and works as well in practice. The learned per-channel gain ``g``
stays.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the last dimension, from scratch.

    Rescales each length-``dim`` feature vector by its root-mean-square, then
    multiplies by a learned per-channel gain ``g`` (initialised to ones). Unlike
    :class:`~llmre.model.block.LayerNorm` there is no mean subtraction and no
    bias term.

    Args:
        dim: size ``C`` of the normalized last dimension.
        eps: constant added inside the sqrt for numerical stability (LLaMA uses
            ``1e-5``; some models use ``1e-6``).

    Shapes:
        input / output ``(..., C)``, same dtype/device; ``weight`` is ``(C,)``.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # RMS over the last axis, keeping the axis for broadcasting. Computed in
        # float32 for stability even if x is half precision, matching LLaMA.
        # float32 at least (bf16/fp16 are upcast); float64 stays float64 for the correctness suite.
        xf = x.to(torch.promote_types(x.dtype, torch.float32))
        rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * rms).type_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` (shape ``(..., C)``) and apply the learned gain."""
        return self._norm(x) * self.weight
