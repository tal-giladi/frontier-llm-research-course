"""Attention kinds. Importing this package registers every kind in :data:`ATTENTION`.

Stage B modules add files here (``mla.py``, ``sliding.py``, ...) and import them below.
"""

from frontierlab.attention.base import ATTENTION, Cache, LayerCache, RotaryEmbedding, apply_rope, register
from frontierlab.attention import gqa  # noqa: F401  (registers "gqa")

__all__ = ["ATTENTION", "Cache", "LayerCache", "RotaryEmbedding", "apply_rope", "register"]
