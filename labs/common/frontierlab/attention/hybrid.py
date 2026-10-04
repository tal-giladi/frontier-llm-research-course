"""Hybrid layouts: N linear-attention layers per full-attention layer (lesson 05.1).

Kind ``"hybrid"`` takes ``layer_idx`` and builds, for each layer, either a linear kind (``"kda"`` or
``"gdn"``, :mod:`frontierlab.attention.deltanet`) or a full-attention kind (``"gqa"``, ``"gated"``,
``"mla"``, ``"dsa"``, any registered kind). Layer ``i`` is full when ``(i + 1) % full_every == 0``:

* Qwen3-Next-80B-A3B: ``full_attention_interval`` 4 in ``config.json`` (48 layers: 36 Gated DeltaNet,
  12 gated attention), i.e. ``full_every=4``, ``linear_kind="gdn"``, ``full_kind="gated"``.
* Kimi Linear 48B-A3B: "a uniform 3:1 ratio" of KDA to MLA (paper); its ``config.json``
  ``linear_attn_config.full_attn_layers`` is ``[4, 8, 12, 16, 20, 24, 27]`` (1-based) of 27 layers, so the
  last block is 2:1. ``extra["layer_types"]`` (one "linear" / "full" per layer) reproduces such a list.
  Kimi Linear's MLA layers use no positional encoding (``mla_use_nope``); our full layers keep RoPE
  unless you pick a NoPE kind, so say which you used.

Settings in ``cfg.extra``: ``full_every`` [4], ``layer_types`` [None], ``linear_kind`` ["kda"],
``full_kind`` ["gqa"], plus whatever the inner kinds read (``chunk``, ``index_topk``, ...). Every
parameter of the inner module sits under ``self_attn.inner.*``.
"""

from __future__ import annotations

import torch.nn as nn

from frontierlab.attention import deltanet  # noqa: F401  (registers "gdn", "kda")
from frontierlab.attention.base import ATTENTION, LayerCache, register


def hybrid_layer_types(cfg) -> list[str]:
    """"linear" or "full" for every layer of a hybrid model."""
    ex = cfg.extra
    if ex.get("layer_types"):
        kinds = list(ex["layer_types"])
        if len(kinds) != cfg.num_hidden_layers or set(kinds) - {"linear", "full"}:
            raise ValueError("extra['layer_types'] needs one 'linear' or 'full' per layer")
        return kinds
    every = int(ex.get("full_every", 4))
    return ["full" if (i + 1) % every == 0 else "linear" for i in range(cfg.num_hidden_layers)]


def from_full_layer_list(full_layers_1based, num_layers: int) -> list[str]:
    """Kimi Linear's ``full_attn_layers`` (1-based) -> our ``layer_types`` list."""
    full = set(full_layers_1based)
    return ["full" if i + 1 in full else "linear" for i in range(num_layers)]


def _build(name: str, cfg, layer_idx: int):
    if name == "hybrid":
        raise ValueError("a hybrid layer cannot contain another hybrid")
    if name not in ATTENTION:
        raise KeyError(f"unknown attention {name!r}; import the module that registers it")
    cls = ATTENTION[name]
    from frontierlab.model.lm import _takes_layer
    return cls(cfg, layer_idx=layer_idx) if _takes_layer(cls) else cls(cfg)


@register("hybrid")
class HybridAttention(nn.Module):
    def __init__(self, cfg, layer_idx: int = 0):
        super().__init__()
        self.layer_type = hybrid_layer_types(cfg)[layer_idx]
        ex = cfg.extra
        name = ex.get("linear_kind", "kda") if self.layer_type == "linear" else ex.get("full_kind", "gqa")
        self.kind = name
        self.inner = _build(name, cfg, layer_idx)

    def forward(self, x, positions, cache: LayerCache | None = None):
        return self.inner(x, positions, cache)
