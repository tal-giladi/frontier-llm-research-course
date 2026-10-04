"""Reference solution for lab 09.1."""

from __future__ import annotations

import torch


def group_nodes(order, sizes, dims, gpus_per_node):
    stride, st = {}, 1
    for d in order:
        stride[d] = st
        st *= sizes[d]
    ranks = [0]
    for d in dims:
        ranks = [r + i * stride[d] for r in ranks for i in range(sizes[d])]
    return len({r // gpus_per_node for r in ranks})


def state_bytes_per_device(n_params, tp, pp, replicas, zero, w=2, g=4, master=4, optim=8):
    n = n_params / (tp * pp)
    wb = n * w / (replicas if zero >= 3 else 1)
    gb = n * g / (replicas if zero >= 2 else 1)
    ob = n * (master + optim) / (replicas if zero >= 1 else 1)
    return wb + gb + ob


def tp_cp_bytes_per_layer(s, b, h, kv_width, e, tp, cp):
    s_l = s / cp
    return {"tp": 8 * (tp - 1) / tp * s_l * b * h * e if tp > 1 else 0.0,
            "cp": 3 * (cp - 1) * s_l * b * kv_width * e if cp > 1 else 0.0}


def merge(o_a, lse_a, o_b, lse_b):
    lse = torch.logaddexp(lse_a, lse_b)
    wa = torch.nan_to_num(torch.exp(lse_a - lse), nan=0.0)[..., None]
    wb = torch.nan_to_num(torch.exp(lse_b - lse), nan=0.0)[..., None]
    return wa * o_a + wb * o_b, lse
