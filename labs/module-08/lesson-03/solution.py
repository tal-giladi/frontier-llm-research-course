"""Reference solution for lab 08.3."""

from __future__ import annotations

import torch

E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def sr_round_e2m1(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    grid = E2M1_GRID.to(x.dtype)
    a = x.abs().clamp(max=6.0)
    i = torch.searchsorted(grid, a, right=True).clamp(max=len(grid) - 1) - 1     # lower neighbour index
    lo = grid[i]
    hi = grid[(i + 1).clamp(max=len(grid) - 1)]
    gap = torch.where(hi > lo, hi - lo, torch.ones_like(hi))
    up = u < (a - lo) / gap
    return torch.sign(x) * torch.where(up & (hi > lo), hi, lo)


def hadamard16(dtype=torch.float64) -> torch.Tensor:
    H = torch.ones(1, 1, dtype=dtype)
    while H.shape[0] < 16:
        H = torch.cat((torch.cat((H, H), 1), torch.cat((H, -H), 1)), 0)
    return H / 4.0


def rht_blocks(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    Q = hadamard16(x.dtype) * signs.to(x.dtype)[None, :]
    return (x.reshape(*x.shape[:-1], -1, 16) @ Q).reshape(x.shape)


def int4_group_qdq(w: torch.Tensor, group: int = 32) -> torch.Tensor:
    R, C = w.shape
    wb = w.reshape(R, C // group, group)
    s = wb.abs().amax(-1, keepdim=True) / 7.0
    s = torch.where(s > 0, s, torch.ones_like(s))
    return (torch.round(wb / s).clamp(-7, 7) * s).reshape(R, C)


def qat_backward(x: torch.Tensor, w: torch.Tensor, gy: torch.Tensor, group: int = 32):
    wq = int4_group_qdq(w, group)
    return gy @ wq, gy.t() @ x
