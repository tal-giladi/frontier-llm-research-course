"""Module 6 block changes: multi-token prediction, hyper-connections and mHC, a fine-grained MoE FFN (copied
from the MoE course's moelab), Engram-style conditional memory, MatFormer nested FFNs and per-layer
embeddings, entropy patching (BLT), a masked-diffusion objective and latent-reasoning loops.

``BlockLM`` (``frontierlab.blocks.model``) is Baseline-0 with these switched on from ``cfg.extra["blocks"]``;
``python -m frontierlab.blocks.train`` trains it with the unmodified course loop.
"""

from frontierlab.blocks.model import BlockLM, BlockLMOutput, block_settings, build, load_model, with_blocks

__all__ = ["BlockLM", "BlockLMOutput", "block_settings", "build", "load_model", "with_blocks"]
