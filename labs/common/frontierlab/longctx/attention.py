"""Attention kinds for Module 4. Importing this module registers them in ``frontierlab.attention.ATTENTION``.

* ``"gqa-rope-scaled"`` — Baseline-0's GQA with the RoPE frequencies of ``cfg.extra["rope"]``
  (:class:`frontierlab.longctx.rope.RopeScaling`: default / pi / ntk / yarn / llama3, partial RoPE, YaRN
  attention factor). The parameters and their names are exactly Baseline-0's, so a Baseline-0
  checkpoint loads into it unchanged (``load_state_dict`` with ``strict=True``): only the
  non-persistent ``inv_freq`` buffer differs.
* ``"gqa-irope"`` — RoPE layers interleaved with NoPE layers (no positional rotation at all), the
  layout Meta's Llama 4 blog calls iRoPE. ``cfg.extra["irope"]``::

      {"nope_every": 4,          # layer i is NoPE when (i + 1) % nope_every == 0 (layers 3, 7, 11, ...)
       "temperature": False,     # optional query scaling on NoPE layers, below
       "floor_scale": 8192, "attn_scale": 0.1}

  Meta's blog says only that Llama 4 interleaves attention layers without positional embeddings and
  uses "inference time temperature scaling of attention"; it gives neither the ratio nor a formula.
  The defaults here follow the Hugging Face Transformers 5.18.0 port of Llama 4
  (``models/llama4/configuration_llama4.py``: ``no_rope_layer_interval=4``; ``modeling_llama4.py``:
  queries of NoPE layers multiplied by ``1 + attn_scale * log1p(floor((pos + 1) / floor_scale))``).
  Treat them as one implementation's choices, not as Meta's documented design. The RoPE layers use
  ``cfg.extra["rope"]`` like ``"gqa-rope-scaled"``. (Transformers' port also makes the RoPE layers
  chunked-local and applies QK-norm only to them; this kind keeps Baseline-0's full attention and
  QK-norm on every layer, so it changes one thing: positions in every fourth layer.)

Both follow the interface in :mod:`frontierlab.attention.base` and pass the causal and cached-decode
checks of :mod:`frontierlab.testing` (``labs/common/tests/test_longctx.py``).
"""

from __future__ import annotations


import torch
import torch.nn.functional as F

from frontierlab.attention.base import LayerCache, apply_rope, causal_mask, register
from frontierlab.attention.gqa import GQAttention
from frontierlab.longctx.rope import RopeScaling


@register("gqa-rope-scaled")
class ScaledRopeGQA(GQAttention):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.rope_spec = RopeScaling.from_config(cfg)
        self.rope = self.rope_spec.rotary(cfg.head_dim, cfg.rope_theta)


def nope_temperature(positions: torch.Tensor, floor_scale: float, attn_scale: float) -> torch.Tensor:
    """(T,) query multiplier 1 + attn_scale * ln(1 + floor((pos + 1) / floor_scale)); 1 below floor_scale."""
    return 1.0 + attn_scale * torch.log1p(torch.floor((positions.to(torch.float64) + 1.0) / floor_scale))


@register("gqa-irope")
class IRopeGQA(GQAttention):
    def __init__(self, cfg, layer_idx: int = 0):
        super().__init__(cfg)
        ex = dict(cfg.extra.get("irope", {}))
        self.nope_every = int(ex.get("nope_every", 4))
        self.use_rope = (layer_idx + 1) % self.nope_every != 0
        self.temperature = bool(ex.get("temperature", False))
        self.floor_scale = float(ex.get("floor_scale", 8192))
        self.attn_scale = float(ex.get("attn_scale", 0.1))
        self.rope_spec = RopeScaling.from_config(cfg)
        self.rope = self.rope_spec.rotary(cfg.head_dim, cfg.rope_theta)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, cache: LayerCache | None = None):
        if self.use_rope:
            return super().forward(x, positions, cache)
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.H, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.KV, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.KV, self.hd).transpose(1, 2)
        if self.temperature:                     # depends on the query's own absolute position only
            q = q * nope_temperature(positions, self.floor_scale, self.attn_scale).to(q.dtype)[None, None, :, None]
        k_pos = positions
        if cache is not None:
            if "k" in cache:
                k = torch.cat((cache["k"], k), dim=2)
                v = torch.cat((cache["v"], v), dim=2)
                k_pos = torch.cat((cache["pos"], positions))
            cache["k"], cache["v"], cache["pos"] = k, v, k_pos
        mask = causal_mask(positions, k_pos)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=self.H != self.KV)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.hd))


def convert(model, attention: str = "gqa-rope-scaled", **extra):
    """A copy of ``model`` (a :class:`frontierlab.model.LM`) with another attention kind and ``cfg.extra``.

    The weights are copied with ``strict=True``, so this only works between kinds with the same
    parameters (Baseline-0's ``"gqa"`` and the two kinds above). Used to evaluate a trained model
    with a different RoPE rule without retraining (lesson 04.2) and to start continued training
    from a checkpoint (lesson 04.3).
    """
    from frontierlab.model import LM
    cfg = model.config.with_(attention=attention, extra={**model.config.extra, **extra})
    new = LM(cfg).to(next(model.parameters()).device, next(model.parameters()).dtype)
    new.load_state_dict(model.state_dict(), strict=True)
    return new.train(model.training)


def rope_extra(type: str = "default", factor: float = 1.0, original: int | None = None, **kw) -> dict:
    """``{"rope": {...}}`` for :func:`convert` / ``ModelConfig.extra`` (a small convenience)."""
    d = {"type": type, "factor": float(factor), **kw}
    if original is not None:
        d["original_max_position_embeddings"] = int(original)
    return {"rope": d}


def logit_multiplier(cfg) -> float:
    """Factor by which a scaled-RoPE layer multiplies attention logits of rotated channels (a**2)."""
    return RopeScaling.from_config(cfg).get_attention_factor() ** 2


__all__ = ["IRopeGQA", "ScaledRopeGQA", "convert", "logit_multiplier", "nope_temperature", "rope_extra"]
