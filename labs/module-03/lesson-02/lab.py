"""Lab 03.2 — local/global attention and KV arithmetic. Fill in the TODOs; run `pytest labs/module-03/lesson-02`."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from frontierlab.attention.base import register
from frontierlab.attention.sliding import LocalGlobalAttention


def window_mask(q_pos: torch.Tensor, k_pos: torch.Tensor, window: int | None) -> torch.Tensor:
    """(T, S) boolean mask: query at q_pos[i] may see key at k_pos[j] iff k_pos[j] <= q_pos[i] and
    k_pos[j] > q_pos[i] - window (``window`` keys including itself). ``window=None``: causal only."""
    raise NotImplementedError("TODO 1: banded causal mask")


def update_window_cache(cache: dict, k, v, positions, window: int | None):
    """Append this step's keys/values (B, KV, T, d) to the cache and return what the step attends to.

    Returns (k_all, v_all, pos_all) = stored + new. Then stores only the last ``window`` positions
    (all of them when ``window`` is None). Trim AFTER building the attended set: the first queries of a
    multi-token chunk still need keys up to window - 1 positions before them.
    """
    raise NotImplementedError("TODO 2: rolling cache update (trim after building the attended set)")


def layer_pattern(num_layers: int, global_every: int) -> list[str]:
    """"local"/"global" per layer: layer i is global when (i + 1) % global_every == 0 (Gemma 3: 6)."""
    raise NotImplementedError("TODO 3: local/global layer pattern")


def kv_bytes(pattern: list[str], S: int, bytes_per_token_layer: float, window: int) -> float:
    """Cache bytes for one sequence after S tokens: global layers keep S tokens, local ones min(S, window)."""
    raise NotImplementedError("TODO 4: cache bytes of a local/global model")


@register("local_global-lab")
class LabLocalGlobal(LocalGlobalAttention):
    """frontierlab's local_global attention with the mask and the cache update replaced by yours."""

    def heads(self, x, positions, cache=None, return_probs=False):
        q, k, v = self.project(x, positions)
        k_pos = positions
        if cache is not None:
            k, v, k_pos = update_window_cache(cache, k, v, positions, self.window)
        mask = window_mask(positions, k_pos, self.window)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=self.H != self.KV)
