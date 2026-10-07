"""Reference solution for lab 15.3."""

from __future__ import annotations


def kv_bytes_per_token(layers, kind, *, kv_heads=0, head_dim=0, d_c=0, d_r=0, bytes_per=2):
    per_layer = 2 * kv_heads * head_dim if kind == "gqa" else d_c + d_r
    return layers * per_layer * bytes_per


def local_global_bytes(S, n_local, n_global, window, kv_heads, head_dim, bytes_per=2):
    per_tok_layer = 2 * kv_heads * head_dim * bytes_per
    return (n_global * S + n_local * min(S, window)) * per_tok_layer


def decode_step_time(weight_bytes, kv_bytes_per_seq, batch, flops_per_token, peak_flops, mem_bw):
    return max(batch * flops_per_token / peak_flops, (weight_bytes + batch * kv_bytes_per_seq) / mem_bw)


def capacity(hbm_bytes, weight_bytes, kv_bytes_per_seq, reserve=0.10):
    return max(0, int((hbm_bytes * (1 - reserve) - weight_bytes) // kv_bytes_per_seq))
