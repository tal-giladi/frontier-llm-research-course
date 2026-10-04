"""Reference solution for lab 09.3."""

from __future__ import annotations

import re

import numpy as np

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LINE = re.compile(r"(?:\[rank(?P<rank>\d+)\]:)?.*?step:\s*(?P<step>\d+)\s+loss:\s*(?P<loss>[-\d.eE+naif]+)\s+"
                   r"grad_norm:\s*(?P<gn>[-\d.eE+naif]+)\s+memory:\s*(?P<mem>[\d.]+)GiB\((?P<pct>[\d.]+)%\)\s+"
                   r"tps:\s*(?P<tps>[\d,]+)\s+tflops:\s*(?P<tflops>[\d,.]+)\s+mfu:\s*(?P<mfu>[\d.]+)%?")


def parse_titan_line(line):
    m = _LINE.search(_ANSI.sub("", line))
    if not m:
        return None
    return {"rank": int(m["rank"]) if m["rank"] is not None else None, "step": int(m["step"]),
            "loss": float(m["loss"]), "memory_gib": float(m["mem"]), "memory_pct": float(m["pct"]),
            "tps": float(m["tps"].replace(",", "")), "tflops": float(m["tflops"].replace(",", "")),
            "mfu": float(m["mfu"])}


def _union(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def exposed_comm(comm, compute):
    c, k = _union(comm), _union(compute)
    total = sum(b - a for a, b in c)
    over, j = 0.0, 0
    for a, b in c:
        for x, y in k:
            lo, hi = max(a, x), min(b, y)
            if hi > lo:
                over += hi - lo
    return total - over


def state_bytes_per_rank(n_params, world, mode):
    per = 4 + 4 + 8
    if mode == "ddp":
        return n_params * per
    if mode == "fsdp2":
        return n_params * per // world
    raise ValueError(mode)


def summarize(values, skip, n_boot=2000, seed=0):
    x = np.asarray(values[skip:], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = np.median(x[rng.integers(0, x.size, (n_boot, x.size))], axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"median": float(np.median(x)), "lo": float(lo), "hi": float(hi), "n": int(x.size)}
