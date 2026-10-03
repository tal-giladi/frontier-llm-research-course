"""Module 4 — does the model use its context?

* :mod:`~frontierlab.longctx.rope` — RoPE frequency rules (PI, NTK-aware, YaRN with its attention
  factor, Llama 3.1's rope type, partial RoPE) as functions returning inverse frequencies (lesson 04.2).
* :mod:`~frontierlab.longctx.attention` — attention kinds ``"gqa-rope-scaled"`` and ``"gqa-irope"``
  (registered on import) and :func:`convert` to swap a trained model's RoPE rule (04.2, 04.3).
* :mod:`~frontierlab.longctx.data` — within-document windows from long documents and boundary
  statistics (04.3).
* :mod:`~frontierlab.longctx.extend` — continued training from a checkpoint at a longer length, a
  wrapper around ``frontierlab.train.loop`` (04.3).

Eval Suite v1 (long-context component) is :mod:`frontierlab.evals.suite_v1` (04.1).
"""

from frontierlab.longctx import attention  # noqa: F401  (registers "gqa-rope-scaled", "gqa-irope")
from frontierlab.longctx.attention import convert, rope_extra
from frontierlab.longctx.rope import RopeScaling, ScaledRotaryEmbedding

__all__ = ["RopeScaling", "ScaledRotaryEmbedding", "convert", "rope_extra"]
