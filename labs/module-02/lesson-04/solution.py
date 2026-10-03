"""Reference solution for lab 02.4."""

from __future__ import annotations

import time

import torch


def bench(fn, warmup: int = 3, repeats: int = 10, sync=lambda: None) -> list[float]:
    for _ in range(warmup):
        fn()
    sync()
    out = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        out.append(time.perf_counter() - t0)
    return out


def grad_agreement(model, idx, loss_a, loss_b) -> dict:
    params = [p for p in model.parameters() if p.requires_grad]
    la = loss_a(model, idx)
    ga = torch.autograd.grad(la, params)
    lb = loss_b(model, idx)
    gb = torch.autograd.grad(lb, params)
    return {"loss_diff": abs(la.item() - lb.item()),
            "max_grad_diff": max((a - b).abs().max().item() for a, b in zip(ga, gb))}


def decide(correct: bool, speedup_ci: tuple[float, float], mem_ratio: float, max_mem_ratio: float = 0.8,
           min_speedup: float = 0.95) -> str:
    if not correct:
        return "reject"
    lo, hi = speedup_ci
    if hi < min_speedup:
        return "reject"
    if lo >= min_speedup and (mem_ratio <= max_mem_ratio or lo > 1.0):
        return "adopt"
    return "inconclusive"
