"""Sliding-window and interleaved local/global attention with a rolling KV cache (lesson 03.2).

Two kinds, both Baseline-0's GQA (same projections, same parameter count) with a different mask:

* ``"sliding"`` — every layer is local: a query at position p sees the ``window`` keys p-window+1 .. p.
* ``"local_global"`` — local layers interleaved with full (global) layers. Layer ``i`` is global when
  ``(i + 1) % global_every == 0``, the rule behind Gemma 3's ``sliding_window_pattern`` (6: five local
  layers, then one global; Gemma 3 report) and gpt-oss's alternating ``layer_types`` (2: banded, dense,
  banded, ...; gpt-oss model card section 2.2). ``cfg.extra["layer_types"]`` (a list of "local" /
  "global", one per layer) overrides the rule.

Settings in ``cfg.extra``: ``window`` [128], ``global_every`` [6], ``layer_types`` [None],
``sinks`` [False] (one learned sink logit per head, gpt-oss style; see :mod:`frontierlab.attention.gated`),
``local_rope_theta`` / ``global_rope_theta`` [cfg.rope_theta] (Gemma 3 uses 10k local, 1M global).

The rolling cache. A local layer never needs keys older than ``window - 1`` positions before the
earliest query of a step, so after each step it keeps only the last ``window`` keys and values
(``LayerCache["k"]`` has at most ``window`` positions, whatever the context length). The keys used *in*
the step are the stored ones plus the new ones; the banded mask removes any that are too old. Trimming
before attending would drop keys that the first queries of a multi-token chunk still need; the
cached-decode test with ``chunk > 1`` catches exactly that.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from frontierlab.attention.base import LayerCache, RotaryEmbedding, apply_rope, register
from frontierlab.attention.gqa import GQAttention
from frontierlab.attention.ops import attend


def layer_types(cfg) -> list[str]:
    """"local" or "global" for each layer of a local_global model."""
    ex = cfg.extra
    if ex.get("layer_types"):
        kinds = list(ex["layer_types"])
        if len(kinds) != cfg.num_hidden_layers or set(kinds) - {"local", "global"}:
            raise ValueError("extra['layer_types'] needs one 'local' or 'global' per layer")
        return kinds
    every = int(ex.get("global_every", 6))
    return ["global" if (i + 1) % every == 0 else "local" for i in range(cfg.num_hidden_layers)]


class _WindowedGQA(GQAttention):
    """GQA with an optional window and optional learned sinks. ``window=None`` is full causal attention."""

    def __init__(self, cfg, window: int | None, rope_theta: float | None = None):
        super().__init__(cfg)
        self.window = window
        self.sinks = nn.Parameter(torch.zeros(self.H)) if cfg.extra.get("sinks") else None
        if rope_theta is not None and rope_theta != cfg.rope_theta:
            self.rope = RotaryEmbedding(self.hd, rope_theta)

    def project(self, x, positions):
        """q (B, H, T, hd), k and v (B, KV, T, hd), with QK-norm and RoPE, exactly as Baseline-0."""
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.H, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.KV, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.KV, self.hd).transpose(1, 2)
        cos, sin = self.rope(positions)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def cached_kv(self, k, v, positions, cache):
        """Keys/values/positions this step attends to; stores the (trimmed, if windowed) cache."""
        k_pos = positions
        if cache is None:
            return k, v, k_pos
        if "k" in cache:
            k = torch.cat((cache["k"], k), dim=2)
            v = torch.cat((cache["v"], v), dim=2)
            k_pos = torch.cat((cache["pos"], positions))
        keep = slice(None) if self.window is None else slice(-self.window, None)
        cache["k"], cache["v"], cache["pos"] = k[:, :, keep], v[:, :, keep], k_pos[keep]
        return k, v, k_pos

    def heads(self, x, positions, cache=None, return_probs=False):
        """Per-head outputs (B, H, T, hd) before the output projection (gated.py multiplies a gate here)."""
        q, k, v = self.project(x, positions)
        k, v, k_pos = self.cached_kv(k, v, positions, cache)
        return attend(q, k, v, positions, k_pos, window=self.window, sink=self.sinks, return_probs=return_probs)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, cache: LayerCache | None = None):
        B, T, _ = x.shape
        y = self.heads(x, positions, cache)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.hd))

    def attention_probs(self, x, positions):
        """Full-sequence probabilities (B, H, T, T) (rows sum to 1 - sink mass) and the masked logits."""
        from frontierlab.attention.ops import attention_logits, sink_softmax
        q, k, _ = self.project(x, positions)
        logits = attention_logits(q, k, positions, positions, self.window)
        return sink_softmax(logits.float(), self.sinks), logits


@register("sliding")
class SlidingAttention(_WindowedGQA):
    def __init__(self, cfg):
        super().__init__(cfg, window=int(cfg.extra.get("window", 128)),
                         rope_theta=cfg.extra.get("local_rope_theta"))
        self.layer_type = "local"


@register("local_global")
class LocalGlobalAttention(_WindowedGQA):
    def __init__(self, cfg, layer_idx: int = 0):
        local = layer_types(cfg)[layer_idx] == "local"
        theta = cfg.extra.get("local_rope_theta" if local else "global_rope_theta")
        super().__init__(cfg, window=int(cfg.extra.get("window", 128)) if local else None, rope_theta=theta)
        self.layer_type = "local" if local else "global"
