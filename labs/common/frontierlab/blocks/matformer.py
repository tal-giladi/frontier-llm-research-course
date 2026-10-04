"""Elastic architectures: MatFormer nested FFNs and Per-Layer Embeddings (lesson 06.5, extension).

**MatFormer** (Devvrit, Kudugunta et al., arXiv 2310.07707, section 3). The FFN's hidden neurons are
ordered: granularity i uses only the first m_i of them, with m_1 < m_2 < ... < m_g = d_ff, so every
smaller FFN is a sub-network of the larger one (T_1 ⊂ T_2 ⊂ ... ⊂ T_g). The paper uses g = 4, m in
{d_ff/8, d_ff/4, d_ff/2, d_ff} (section 3.1), and trains by sampling one granularity per step (uniformly
in most experiments; section 3.2, Eq. 2). After training, "Mix'n'Match" picks a granularity per layer
(section 3.3; the paper recommends sizes that grow slowly with depth).

    NestedMLP(x; m) = W_down[:, :m] ( silu(W_gate[:m] x) * (W_up[:m] x) )

Names and full-size shapes are those of the Qwen3 MLP (``gate_proj``, ``up_proj``, ``down_proj``), so the
full-width model is an ordinary Baseline-0 checkpoint.

**Per-Layer Embeddings (PLE)** as Google documents them for Gemma 3n (developer guide, 2025-06): "a
significant portion of these parameters (the embeddings associated with each layer)" can be "loaded and
computed efficiently on the CPU", so only the core transformer weights sit in accelerator memory; the
Gemma 4 report (arXiv 2607.02770, section 2) says E2B and E4B "use per-layer embeddings as in Gemma 3n".
The mechanism below follows the Hugging Face Transformers implementation (v5.18.0,
``models/gemma3n/modeling_gemma3n.py``: ``embed_tokens_per_layer``, ``per_layer_model_projection``,
``per_layer_input_gate``, ``per_layer_projection``), simplified (no AltUp, no LAuReL):

    p      = (E_ple[x] reshaped to (L, d) + RMSNorm(P(embed(x)) / sqrt(C)) reshaped) / sqrt(2)   per token, (L, d)
    layer l: h <- h + RMSNorm( W_proj^l ( gelu(W_gate^l h) * p_l ) )

E_ple is a (V, L·d) table: large (V·L·d parameters) but read only one row per token, which is why it can
live in host memory and be streamed in. It is a lookup, like Engram's tables, but keyed by the current
token alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from frontierlab.layers.rmsnorm import RMSNorm


def matformer_widths(d_ff: int, g: int = 4) -> list[int]:
    """The paper's exponentially spaced granularities: d_ff/2^(g-1), ..., d_ff/2, d_ff."""
    return [max(1, d_ff // 2 ** (g - 1 - i)) for i in range(g)]


class NestedMLP(nn.Module):
    """SwiGLU whose active width ``m`` can be any prefix of its neurons (``width`` attribute, default full)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.width = intermediate_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.width
        g = F.linear(x, self.gate_proj.weight[:m])
        u = F.linear(x, self.up_proj.weight[:m])
        return F.linear(F.silu(g) * u, self.down_proj.weight[:, :m])


def mix_n_match(widths: list[int], num_layers: int, budget: float) -> list[int]:
    """A per-layer width list whose mean is closest to ``budget`` (a target mean width), with widths that never
    shrink with depth and change by at most one granularity step between consecutive layers — the paper's
    "increasing with minimum slope" heuristic (section 3.3, Appendix D.1), as a simple greedy rule."""
    best, best_err = None, float("inf")
    g = len(widths)
    for lo in range(g):
        hi = min(g - 1, lo + 1)
        for split in range(num_layers + 1):          # first `split` layers at lo, the rest at hi
            cfg = [widths[lo]] * split + [widths[hi]] * (num_layers - split)
            err = abs(sum(cfg) / num_layers - budget)
            if err < best_err:
                best, best_err = cfg, err
    return best


class PerLayerEmbedding(nn.Module):
    """Gemma 3n-style per-layer inputs (module docstring). ``inputs(ids, emb)`` -> (B, T, L, d);
    ``layer(l, h, p_l)`` -> the term added to the residual after layer l."""

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, dim: int, eps: float = 1e-6):
        super().__init__()
        self.L, self.d, self.C = num_layers, dim, hidden_size
        self.embed_tokens_per_layer = nn.Embedding(vocab_size, num_layers * dim)
        self.per_layer_model_projection = nn.Linear(hidden_size, num_layers * dim, bias=False)
        self.per_layer_projection_norm = RMSNorm(dim, eps=eps)
        self.per_layer_input_gate = nn.ModuleList(nn.Linear(hidden_size, dim, bias=False) for _ in range(num_layers))
        self.per_layer_projection = nn.ModuleList(nn.Linear(dim, hidden_size, bias=False) for _ in range(num_layers))
        self.post_per_layer_input_norm = nn.ModuleList(RMSNorm(hidden_size, eps=eps) for _ in range(num_layers))

    def inputs(self, ids: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        table = self.embed_tokens_per_layer(ids).view(B, T, self.L, self.d) * self.d ** 0.5
        proj = (self.per_layer_model_projection(emb) * self.C ** -0.5).view(B, T, self.L, self.d)
        return (self.per_layer_projection_norm(proj) + table) * 2 ** -0.5

    def layer(self, l: int, h: torch.Tensor, p_l: torch.Tensor) -> torch.Tensor:
        """h (..., C), p_l (..., d) = inputs(...)[:, :, l] (broadcast over streams)."""
        gated = F.gelu(self.per_layer_input_gate[l](h), approximate="tanh") * p_l
        return self.post_per_layer_input_norm[l](self.per_layer_projection[l](gated))

    def host_params(self) -> int:
        """The table that Google's guide says can stay off the accelerator: V·L·d."""
        return self.embed_tokens_per_layer.weight.numel()
