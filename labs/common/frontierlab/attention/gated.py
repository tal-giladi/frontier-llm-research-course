"""Logit and output control for attention: learned sinks and head-specific output gates (lesson 03.3).

Two kinds, both Baseline-0's GQA (with its QK-norm) plus one mechanism:

* ``"sink"`` — one learned logit ``s_h`` per head joins the softmax denominator and carries no value:

      p_ij = exp(l_ij) / (exp(s_h) + sum_j' exp(l_ij'))        y_i = sum_j p_ij v_j

  so a head can put weight on "nothing" instead of on some token. This is the gpt-oss form ("Each
  attention head has a learned bias in the denominator of the softmax, similar to off-by-one attention
  and attention sinks, which enables the attention mechanism to pay no attention to any tokens",
  gpt-oss model card section 2.2). The parameter is named ``sinks`` as in Hugging Face ``GptOssAttention``.
  Initialised to 0 (the sink starts with the weight of one key with logit 0).

* ``"gated"`` — a sigmoid gate on each head's attention output, computed from the same normalised
  hidden state that produced the query, applied after scaled dot-product attention and before the
  output projection (position "G1" of Qiu et al., Gated Attention for LLMs, arXiv 2505.06708):

      Y' = Y * sigmoid(X W_g)

  ``cfg.extra["gate"]``: ``"elementwise"`` [default] — one gate per head and channel, W_g of shape
  (C, H*hd), the variant the paper reports as best and the one Qwen3-Next fuses into ``q_proj``;
  ``"headwise"`` — one scalar gate per head, W_g of shape (C, H), far fewer parameters.

Either kind combines with ``cfg.extra["window"]`` for a sliding window (gpt-oss uses sinks in both its
banded and dense layers); for interleaved layers use ``local_global`` with ``extra["sinks"] = True``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from frontierlab.attention.base import LayerCache, register
from frontierlab.attention.sliding import _WindowedGQA


def _window(cfg):
    w = cfg.extra.get("window")
    return int(w) if w else None


@register("sink")
class SinkAttention(_WindowedGQA):
    def __init__(self, cfg):
        super().__init__(cfg.with_(extra={**cfg.extra, "sinks": True}), window=_window(cfg))
        self.layer_type = "local" if self.window else "global"


@register("gated")
class GatedAttention(_WindowedGQA):
    def __init__(self, cfg):
        super().__init__(cfg, window=_window(cfg))
        self.layer_type = "local" if self.window else "global"
        self.gate_kind = cfg.extra.get("gate", "elementwise")
        if self.gate_kind not in ("elementwise", "headwise"):
            raise ValueError("extra['gate'] must be 'elementwise' or 'headwise'")
        width = self.H * self.hd if self.gate_kind == "elementwise" else self.H
        self.gate_proj = nn.Linear(cfg.hidden_size, width, bias=False)

    def gate(self, x):
        """sigmoid(x W_g) shaped (B, H, T, hd or 1) to multiply the per-head outputs."""
        B, T, _ = x.shape
        g = torch.sigmoid(self.gate_proj(x)).view(B, T, self.H, -1)
        return g.transpose(1, 2)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, cache: LayerCache | None = None):
        B, T, _ = x.shape
        y = self.heads(x, positions, cache) * self.gate(x)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.hd))
