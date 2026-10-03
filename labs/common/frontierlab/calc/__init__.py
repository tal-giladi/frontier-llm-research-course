"""Reading configs as evidence (Module 1, lesson 01.2).

* :mod:`frontierlab.calc.hfconfig` — Hugging Face ``config.json`` -> :class:`ArchSpec`, with the
  config key behind every field (``spec.source``); dated snapshots of a dozen open models in
  ``snapshots/``.
* :mod:`frontierlab.calc.arith` — parameters, active parameters, FLOPs per token and KV-cache bytes
  for GQA, MLA, sliding-window and linear-attention layers.

    python -m frontierlab.calc deepseek-ai/DeepSeek-V3 --seq 4096 --context 32768
    python -m frontierlab.calc --all
"""

from frontierlab.calc.arith import (attention_params, decode_flops_per_token, flops_per_token, kv_bytes,
                                    kv_bytes_per_token, kv_cache_elements, param_counts, summary_row)
from frontierlab.calc.hfconfig import ArchSpec, fetch_config, from_course, from_hf, load_config, snapshot_names

__all__ = ["ArchSpec", "attention_params", "decode_flops_per_token", "fetch_config", "flops_per_token",
           "from_course", "from_hf", "kv_bytes", "kv_bytes_per_token", "kv_cache_elements", "load_config",
           "param_counts", "snapshot_names", "summary_row"]
