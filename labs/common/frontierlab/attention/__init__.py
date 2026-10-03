"""Attention kinds. Importing this package registers every kind in :data:`ATTENTION`.

Stage B modules add files here (``mla.py``, ``sliding.py``, ...) and import them below.
"""

from frontierlab.attention.base import ATTENTION, Cache, LayerCache, RotaryEmbedding, apply_rope, register
from frontierlab.attention import gqa  # noqa: F401  (registers "gqa")
from frontierlab.attention import mla  # noqa: F401  (registers "mla")
from frontierlab.attention import sliding  # noqa: F401  (registers "sliding", "local_global")
from frontierlab.attention import gated  # noqa: F401  (registers "sink", "gated")
from frontierlab.attention import headshape  # noqa: F401  (registers "gqa_partial")
import frontierlab.longctx.attention  # noqa: F401,E402  (registers "gqa-rope-scaled", "gqa-irope")

__all__ = ["ATTENTION", "Cache", "LayerCache", "RotaryEmbedding", "apply_rope", "register"]
