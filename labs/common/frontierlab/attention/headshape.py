"""Head shape: partial RoPE for wide heads (extension lesson 03.4).

``"gqa_partial"`` is Baseline-0's GQA in which RoPE rotates only the first ``rope_fraction * head_dim``
channels of every query and key; the remaining channels carry no position signal. Qwen3-Next uses
``head_dim`` 256 with ``partial_rotary_factor`` 0.25 (its config.json), i.e. 64 rotated channels.

Settings in ``cfg.extra``: ``rope_fraction`` [0.25]; the rotated width is rounded down to an even number.
Head count and head width themselves are ordinary config fields (``num_attention_heads``, ``head_dim``).
"""

from __future__ import annotations

from frontierlab.attention.base import RotaryEmbedding, register
from frontierlab.attention.gqa import GQAttention


def rotary_width(head_dim: int, fraction: float) -> int:
    r = int(head_dim * fraction)
    return r - (r % 2)


@register("gqa_partial")
class PartialRopeGQA(GQAttention):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.rot = rotary_width(self.hd, float(cfg.extra.get("rope_fraction", 0.25)))
        if self.rot < 2:
            raise ValueError("rope_fraction too small: fewer than 2 rotated channels")
        self.rope = RotaryEmbedding(self.rot, cfg.rope_theta)
