"""Reference solution for lab 08.1."""

from __future__ import annotations

import torch


def round_fp(x: torch.Tensor, exp_bits: int, man_bits: int, bias: int, max_normal: float) -> torch.Tensor:
    xw = x if x.dtype == torch.float64 else x.float()
    _, e = torch.frexp(xw)                                   # x = m * 2^e with 0.5 <= |m| < 1
    k = torch.clamp(e - 1, min=1 - bias)                     # binade, floored at the smallest normal exponent
    shift = (k - man_bits).to(torch.int32)                   # spacing of the binade is 2^(k - m)
    y = torch.ldexp(torch.round(torch.ldexp(xw, -shift)), shift)
    return y.clamp(-max_normal, max_normal)


def block_qdq(x: torch.Tensor, exp_bits: int, man_bits: int, bias: int, max_normal: float, block: int) -> torch.Tensor:
    R, C = x.shape
    xb = x.reshape(R, C // block, block)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    s = torch.where(amax > 0, amax / max_normal, torch.ones_like(amax))
    return (round_fp(xb / s, exp_bits, man_bits, bias, max_normal) * s).reshape(R, C)


def mx_shared_scale(amax: torch.Tensor, emax_elem: int) -> torch.Tensor:
    _, e = torch.frexp(amax)
    return torch.exp2((e - 1 - emax_elem).to(amax.dtype))    # largest power of two <= amax, / 2^emax_elem


def nvfp4_qdq(x: torch.Tensor) -> torch.Tensor:
    R, C = x.shape
    t_dec = x.abs().max() / (6.0 * 448.0)                    # tensor decode scale, fp32
    xb = x.reshape(R, C // 16, 16)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    b = round_fp(amax / 6.0 / t_dec, 4, 3, 7, 448.0)        # block scale stored in E4M3
    s = b * t_dec
    safe = torch.where(s > 0, s, torch.ones_like(s))
    q = round_fp(xb / safe, 2, 1, 1, 6.0)
    return torch.where(s > 0, q * s, torch.zeros_like(q)).reshape(R, C)
