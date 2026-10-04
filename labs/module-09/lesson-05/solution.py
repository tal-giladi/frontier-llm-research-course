"""Reference solution for lab 09.5."""

from __future__ import annotations

import math


def torus_hops(a, b, dims, wrap=True):
    h = 0
    for x, y, n in zip(a, b, dims):
        d = abs(x - y)
        h += min(d, n - d) if wrap else d
    return h


def bisection_links(dims, wrap=True):
    n = max(dims)
    links = math.prod(dims) // n
    return 2 * links if wrap else links


def collective_time(kind, nbytes, axis_size, link_bw):
    n = axis_size
    one = nbytes * (n - 1) / n / (2 * link_bw)
    return {"allgather": one, "reducescatter": one, "allreduce": 2 * one}[kind]


def matmul_comm(lhs, rhs):
    (ai, aj), (bj, bk) = lhs, rhs
    if ai and bk and ai == bk:
        return "invalid"
    if aj and bj and aj == bj:
        return "allreduce"
    if aj or bj:
        return "allgather"
    return "none"
