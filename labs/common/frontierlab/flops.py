"""FLOPs, memory and time arithmetic for the course model (Modules 1–2).

Conventions (parent course lessons 07.4 and 10.1): one multiply-add = 2 FLOPs; training costs
forward + backward ≈ 3× forward. Per token, with N non-embedding parameters, L layers, context T
and attention width d_attn = heads × head_dim:

    forward FLOPs/token  ≈ 2·N  +  2·L·T·d_attn  +  2·V·C

The attention term: a token at position t costs 2·d_attn·t for QK^T and 2·d_attn·t for AV, so
4·d_attn·t per layer; averaged over t from 0 to T that is 2·d_attn·T. The last term is the output
head, which N excludes when embeddings are tied. Training FLOPs/token ≈ 3 × forward.
"""

from __future__ import annotations

from frontierlab.model.config import ModelConfig, param_counts

# Dense BF16 tensor-core peak, FLOP/s (vendor datasheets; check the exact SKU you rent).
PEAK_BF16 = {"H100-SXM": 989e12, "H100-PCIe": 756e12, "A100": 312e12, "L4": 121e12, "T4-FP16": 65e12,
             "L40S": 362e12}


def flops_per_token(cfg: ModelConfig, T: int, training: bool = True) -> float:
    pc = param_counts(cfg)
    d_attn = cfg.num_attention_heads * cfg.head_dim
    head = 2 * cfg.vocab_size * cfg.hidden_size if cfg.tie_word_embeddings else 0   # untied head is in N
    fwd = 2 * pc["non_embedding"] + 2 * cfg.num_hidden_layers * T * d_attn + head
    return 3 * fwd if training else fwd


def run_flops(cfg: ModelConfig, tokens: float, T: int) -> float:
    return flops_per_token(cfg, T) * tokens


def mfu(cfg: ModelConfig, T: int, tokens_per_s: float, peak: float) -> float:
    """Model FLOPs utilization: achieved training FLOP/s ÷ hardware peak."""
    return flops_per_token(cfg, T) * tokens_per_s / peak


def projected_gpu_hours(cfg: ModelConfig, tokens: float, T: int, peak: float, mfu_measured: float) -> float:
    """Main-path projection of plan section 12.1: run FLOPs ÷ (peak × MFU measured in a pilot)."""
    return run_flops(cfg, tokens, T) / (peak * mfu_measured) / 3600


def kv_bytes_per_token(cfg: ModelConfig, bytes_per_elem: int = 2) -> int:
    """Decode cache bytes per token for standard GQA: 2 (K and V) × layers × kv_heads × head_dim."""
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * bytes_per_elem
