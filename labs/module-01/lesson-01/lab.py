"""Lab 01.1 — diagnostic. Fill in the TODOs; run `pytest labs/module-01/lesson-01` to check."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from frontierlab.attention.base import apply_rope, register
from frontierlab.attention.gqa import GQAttention
from frontierlab.model.config import ModelConfig


def count_params(cfg: ModelConfig) -> int:
    """Total parameters of ``frontierlab.model.LM(cfg)``, from the formulas in the lesson.

    Count: per layer the q/k/v/o projections, the two QK-norm gains (only if cfg.qk_norm), the three
    SwiGLU matrices and two RMSNorm gains; then the final norm, the embedding, and the output head
    only if it is NOT tied (cfg.tie_word_embeddings is False). Do not call param_counts().
    """
    raise NotImplementedError("TODO 1: count the parameters by hand")


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pos: torch.Tensor,
           k_pos: torch.Tensor) -> torch.Tensor:
    """Causal attention for q (B, H, Tq, d), k/v (B, KV, Tk, d), absolute positions q_pos (Tq,), k_pos (Tk,).

    Query i may attend key j exactly when k_pos[j] <= q_pos[i]. Tk can be larger than Tq (cached
    decoding), and H may be a multiple of KV (grouped-query attention).

    The line below is the planted bug: is_causal=True aligns the mask to the top-left corner of the
    (Tq, Tk) score matrix, which is right only when Tq == Tk.
    """
    # TODO 2: replace this line with attention under a mask built from q_pos and k_pos.
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=q.shape[1] != k.shape[1])


@register("gqa-lab")
class LabGQAttention(GQAttention):
    """Baseline-0's attention, except that the masking step calls your ``attend``."""

    def forward(self, x, positions, cache=None):
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.H, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.KV, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.KV, self.hd).transpose(1, 2)
        cos, sin = self.rope(positions)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k_pos = positions
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat((cache["k"], k), 2), torch.cat((cache["v"], v), 2)
                k_pos = torch.cat((cache["pos"], positions))
            cache["k"], cache["v"], cache["pos"] = k, v, k_pos
        y = attend(q, k, v, positions, k_pos)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.hd))
