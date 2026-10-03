"""Lab 01.2 — reading configs as evidence. Fill in the TODOs; run `pytest labs/module-01/lesson-02`.

You get an ``ArchSpec`` (``frontierlab.calc.from_hf``): one vocabulary for every model family. The
tests check your three functions against parameter counts that Hugging Face Transformers produces
from the same configs, and against hand-worked values. Do not call ``frontierlab.calc.arith``.
"""

from __future__ import annotations

from frontierlab.calc.hfconfig import ArchSpec


def attention_params(s: ArchSpec) -> int:
    """Parameters of ONE full-attention block (not linear attention).

    GQA (``s.attention == "gqa"``): q_proj C x (H*d) (twice as wide if ``s.q_gate``), k_proj and
    v_proj C x (KV*d) (v uses ``s.v_head_dim``), o_proj (H*d_v) x C; biases on q/k/v if
    ``s.qkv_bias`` and on o if ``s.o_bias``; QK-norm gains 2*d if ``s.qk_norm == "per_head"`` or
    H*d + KV*d if ``"full"``; one sink per head if ``s.sinks``.

    MLA (``s.attention == "mla"``), with dqk = qk_nope_dim + qk_rope_dim:
      queries   C x q_lora_rank, a norm of q_lora_rank, q_lora_rank x (H*dqk)   (or C x H*dqk if no q_lora_rank)
      keys/vals C x (kv_lora_rank + qk_rope_dim), a norm of kv_lora_rank,
                kv_lora_rank x H*(qk_nope_dim + v_head_dim)
      output    (H*v_head_dim) x C
    """
    raise NotImplementedError("TODO 1: count one attention block for GQA and MLA")


def kv_elements(s: ArchSpec, S: int) -> int:
    """Cached elements for one sequence of S tokens, summed over the non-linear layers.

    Per token and layer: GQA keeps KV*(head_dim + v_head_dim); MLA keeps kv_lora_rank + qk_rope_dim.
    A "full" layer keeps all S tokens; a "sliding" layer keeps at most s.sliding_window of them.
    Skip "linear" layers (their state does not grow with S).
    """
    raise NotImplementedError("TODO 2: KV-cache elements for full, sliding and MLA layers")


def active_params(s: ArchSpec) -> int:
    """Parameters one token uses, counting the output head but NOT the input embedding.

    Use ``frontierlab.calc.arith.ffn_params(s, layer)["active"]`` for the feed-forward part (routing is
    Module 3's MoE link, not this lab), plus every attention block (``attention_params`` for full and
    sliding layers; ``frontierlab.calc.arith.attention_params(s, "linear")`` for linear layers),
    the norms (``s.norms_per_layer`` vectors of width C per layer plus the final norm), and the head
    (V x C; when ``s.tie_embeddings`` the head IS the embedding matrix and is counted once).
    """
    raise NotImplementedError("TODO 3: active parameters per token")
