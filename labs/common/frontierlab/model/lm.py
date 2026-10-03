"""Baseline-0: a dense decoder-only language model in the Qwen3 layout.

Module names match Hugging Face ``Qwen3ForCausalLM`` (``model.embed_tokens``,
``model.layers.N.self_attn.*``, ``model.layers.N.mlp.{gate,up,down}_proj``,
``model.layers.N.{input,post_attention}_layernorm``, ``model.norm``, ``lm_head``), so a checkpoint
can be handed to Transformers or vLLM by name.

Block (pre-norm):
    h = x + Attention(RMSNorm(x))        attention kind chosen by cfg.attention
    y = h + SwiGLU(RMSNorm(h))

Shapes: idx (B, T) long -> logits (B, T, V) float32. ``loss`` is the mean next-token cross-entropy
over positions 0..T-2 predicting 1..T-1 (labels default to idx, shifted inside).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from frontierlab.attention import ATTENTION, Cache
from frontierlab.layers.rmsnorm import RMSNorm
from frontierlab.model.config import ModelConfig


class MLP(nn.Module):
    """SwiGLU with Qwen3 names: down_proj(silu(gate_proj(x)) * up_proj(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        C, I = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(C, I, bias=False)
        self.up_proj = nn.Linear(C, I, bias=False)
        self.down_proj = nn.Linear(I, C, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        if cfg.attention not in ATTENTION:
            raise KeyError(f"unknown attention {cfg.attention!r}; registered: {sorted(ATTENTION)}")
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        attn_cls = ATTENTION[cfg.attention]
        self.self_attn = attn_cls(cfg, layer_idx=layer_idx) if _takes_layer(attn_cls) else attn_cls(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, positions, cache=None):
        x = x + self.self_attn(self.input_layernorm(x), positions, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


def _takes_layer(cls) -> bool:
    """Attention kinds that differ per layer (e.g. local/global mixes) accept ``layer_idx``."""
    import inspect
    return "layer_idx" in inspect.signature(cls.__init__).parameters


class Backbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Block(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)


@dataclass
class LMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    per_token_loss: torch.Tensor | None = None   # (B, T-1), for bootstrap over tokens/documents


class LM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.model = Backbone(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=self.config.initializer_range)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def new_cache(self) -> Cache:
        return Cache(self.config.num_hidden_layers)

    def forward(self, idx: torch.Tensor, labels: torch.Tensor | None = None, cache: Cache | None = None,
                reduction: str = "mean") -> LMOutput:
        B, T = idx.shape
        start = cache.length if cache is not None else 0
        positions = torch.arange(start, start + T, device=idx.device)
        x = self.model.embed_tokens(idx)
        for i, layer in enumerate(self.model.layers):
            x = layer(x, positions, cache.layers[i] if cache is not None else None)
        if cache is not None:
            cache.length += T
        logits = self.lm_head(self.model.norm(x)).float()
        if labels is None:
            return LMOutput(logits)
        tok = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
                              reduction="none").view(B, T - 1)
        return LMOutput(logits, tok.mean() if reduction == "mean" else tok.sum(), tok)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.0,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """Greedy (temperature 0) or sampled decoding with the cache."""
        cache = self.new_cache()
        logits = self(idx, cache=cache).logits[:, -1]
        out = [idx]
        for _ in range(max_new_tokens):
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=generator)
            out.append(nxt)
            logits = self(nxt, cache=cache).logits[:, -1]
        return torch.cat(out, dim=1)
