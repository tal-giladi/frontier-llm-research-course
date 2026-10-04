"""Reference solution for lab 05.4."""

from __future__ import annotations

import torch


def _blocks(x, m):
    B, S, d = x.shape
    n = S // m
    return x[:, : n * m].reshape(B, n, m, d)


def compress_hca(c, z, m):
    return (torch.softmax(_blocks(z, m), dim=2) * _blocks(c, m)).sum(dim=2)


def compress_csa(ca, za, cb, zb, m):
    A_c, A_z, B_c, B_z = _blocks(ca, m), _blocks(za, m), _blocks(cb, m), _blocks(zb, m)
    n = A_c.shape[1]
    prev_z = torch.cat((torch.full_like(B_z[:, :1], float("-inf")), B_z[:, : n - 1]), dim=1)
    prev_c = torch.cat((torch.zeros_like(B_c[:, :1]), B_c[:, : n - 1]), dim=1)
    w = torch.softmax(torch.cat((A_z, prev_z), dim=2), dim=2)
    return (w * torch.cat((A_c, prev_c), dim=2)).sum(dim=2)


def usable_entries(t, m):
    return (t + 1) // m


def layout_kv_bytes(S, layers, *, m, m2, window, entry_bytes):
    per = {"csa": S // m + min(S, window), "hca": S // m2, "dense": S}
    return sum(per[k] for k in layers) * entry_bytes
