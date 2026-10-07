"""Reference solution for lab 17.1."""

from __future__ import annotations

import torch


def topk_code(pre: torch.Tensor, k: int) -> torch.Tensor:
    v, i = pre.topk(k, dim=-1)
    return torch.zeros_like(pre).scatter(-1, i, torch.relu(v))


class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pre, theta, eps):
        ctx.save_for_backward(pre, theta)
        ctx.eps = eps
        return pre * (pre > theta).to(pre.dtype)

    @staticmethod
    def backward(ctx, g):
        pre, theta = ctx.saved_tensors
        u = (pre - theta) / ctx.eps
        rect = ((u > -0.5) & (u < 0.5)).to(pre.dtype)
        g_pre = g * (pre > theta).to(pre.dtype)
        g_theta = (g * (-(theta / ctx.eps) * rect)).sum(dim=tuple(range(pre.dim() - 1)))
        return g_pre, g_theta, None


def fvu(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    return float((x - x_hat).pow(2).sum() / (x - x.mean(0)).pow(2).sum())


def splice_edit(sae, keep_error: bool = False):
    def f(x):
        xh = sae(x)
        return xh + (x - xh) if keep_error else xh
    return f
