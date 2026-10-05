"""Attention kinds. Importing this package registers every kind in :data:`ATTENTION`.

Stage B modules add files here (``mla.py``, ``sliding.py``, ...) and import them below.
"""

from frontierlab.attention.base import ATTENTION, Cache, LayerCache, RotaryEmbedding, apply_rope, register
from frontierlab.attention import gqa  # noqa: F401  (registers "gqa")
from frontierlab.attention import mla  # noqa: F401  (registers "mla")
from frontierlab.attention import sliding  # noqa: F401  (registers "sliding", "local_global")
from frontierlab.attention import gated  # noqa: F401  (registers "sink", "gated")
from frontierlab.attention import headshape  # noqa: F401  (registers "gqa_partial")
from frontierlab.attention import deltanet  # noqa: F401  (registers "gdn", "kda")
from frontierlab.attention import hybrid  # noqa: F401  (registers "hybrid")
from frontierlab.attention import dsa  # noqa: F401  (registers "dsa")
import frontierlab.longctx.attention  # noqa: F401,E402  (registers "gqa-rope-scaled", "gqa-irope")
import frontierlab.datax.packing  # noqa: F401,E402  (registers "gqa-docmask")

__all__ = ["ATTENTION", "Cache", "LayerCache", "RotaryEmbedding", "apply_rope", "register"]
