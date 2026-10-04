"""Scaled quantisation: per-tensor, per-row, per-tile and per-block scales (lessons 08.1–08.3).

A low-precision tensor is a grid of small numbers ``q`` plus one or more **scales** ``s``: ``x ≈ s · q``. The
scale maps the largest magnitude of a group of elements (its *amax*) onto the format's range, so the format's
few bits are spent where the values are. What differs between recipes is how big the group is and what format
the scale itself is stored in.

Groups are blocks of a 2-D view ``(R, C)`` of the tensor (all leading dims folded into R). ``block=(br, bc)``,
with 0 meaning "the whole extent":

    tensorwise            (0, 0)        one scale per tensor                  torchao "tensorwise"
    rowwise               (1, 0)        one scale per row                     torchao "rowwise"
    1 x 128 tile          (1, 128)      per token per 128 channels            DeepSeek-V3 activations (section 3.3.2)
    128 x 128 block       (128, 128)                                          DeepSeek-V3 weights
    1 x 32                (1, 32)       MX block (OCP MX v1.0, k = 32)        MXFP8, MXFP4, gpt-oss MoE weights
    1 x 16                (1, 16)       NVFP4 activations and gradients       arXiv 2509.25149 section 4
    16 x 16               (16, 16)      NVFP4 weights (2-D)

Blocks run along the **last** dimension, which is the contraction (K) dimension when the tensor is the left
operand of ``x @ w.T``: block-scaled GEMMs need one scale per K-block of each operand. To quantise for a GEMM
that contracts over another dimension, transpose first (``frontierlab.precision.linear`` does).

Scale kinds (``scale=``):

    "fp32"        s = amax / max_elem, stored in fp32 (torchao, DeepSeek-V3)
    "pow2"        s = 2^ceil(log2(amax / max_elem)): a power of two never smaller than needed, so nothing
                  saturates (UE8M0 as DeepSeek-V3.1 / DeepGEMM use it; torchao's power-of-2 rowwise scales)
    "e8m0"        OCP MX v1.0 Algorithm: s = 2^(floor(log2 amax) - emax_elem). Up to one binade of the largest
                  values exceeds max_elem and is clamped (the spec's conversion clamps normal numbers)
    "nvfp4"       two levels (arXiv 2509.25149 appendix B): tensor decode scale amax_t / (6 · 448) in fp32, block
                  decode scale amax_b / 6 divided by it and rounded to E4M3; s = e4m3(...) · tensor scale

Rounding of the elements: "rne" (round to nearest even) or "sr" (stochastic, unbiased). Scales are always
computed from the exact amax of the block ("online" / current scaling, as DeepSeek-V3 section 3.3.2 and the NVFP4
recipe do; delayed scaling from an amax history is not emulated).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from frontierlab.precision.formats import E4M3, FloatFormat, IntFormat, get_format, round_pow2, round_to_format


@dataclass(frozen=True)
class QuantSpec:
    fmt: str = "e4m3"
    block: tuple = (0, 0)
    scale: str = "fp32"          # fp32 | pow2 | e8m0 | nvfp4
    rounding: str = "rne"        # rne | sr

    def with_(self, **kw) -> "QuantSpec":
        return replace(self, **kw)

    @property
    def format(self) -> FloatFormat | IntFormat:
        return get_format(self.fmt)

    def bits_per_value(self, shape: tuple | None = None) -> float:
        """Storage bits per element including scales: element bits + scale bits / block size.

        MXFP4 (1 x 32, 8-bit scale): 4 + 8/32 = 4.25, the figure the gpt-oss model card gives for its MoE weights.
        NVFP4 (16 elements, E4M3 scale, plus one fp32 per tensor): 4 + 8/16 = 4.5 (+ 32/numel).
        """
        elem = self.format.bits
        scale_bits = {"fp32": 32, "pow2": 8, "e8m0": 8, "nvfp4": 8}[self.scale]
        if shape is None:
            br, bc = self.block
            if br == 0 or bc == 0:
                return float(elem)
            return elem + scale_bits / (br * bc)
        R = math.prod(shape[:-1]) if len(shape) > 1 else 1
        C = shape[-1]
        br, bc = (self.block[0] or R), (self.block[1] or C)
        nblocks = math.ceil(R / br) * math.ceil(C / bc)
        extra = 32 if self.scale == "nvfp4" else 0
        return elem + (nblocks * scale_bits + extra) / (R * C)


# Named specs used by the recipes and the lessons.
SPECS = {
    "fp8-tensor-e4m3": QuantSpec("e4m3", (0, 0), "fp32"),
    "fp8-tensor-e5m2": QuantSpec("e5m2", (0, 0), "fp32"),
    "fp8-row-e4m3": QuantSpec("e4m3", (1, 0), "pow2"),
    "fp8-tile128": QuantSpec("e4m3", (1, 128), "fp32"),
    "fp8-block128": QuantSpec("e4m3", (128, 128), "fp32"),
    "fp8-tile128-ue8m0": QuantSpec("e4m3", (1, 128), "pow2"),
    "mxfp8": QuantSpec("e4m3", (1, 32), "e8m0"),
    "mxfp6-e3m2": QuantSpec("e3m2", (1, 32), "e8m0"),
    "mxfp4": QuantSpec("e2m1", (1, 32), "e8m0"),
    "mxfp4-pow2": QuantSpec("e2m1", (1, 32), "pow2"),
    "nvfp4-1d": QuantSpec("e2m1", (1, 16), "nvfp4"),
    "nvfp4-2d": QuantSpec("e2m1", (16, 16), "nvfp4"),
    "int8-row": QuantSpec("int8", (1, 0), "fp32"),
    "int4-g32": QuantSpec("int4", (1, 32), "fp32"),
    "int4-row": QuantSpec("int4", (1, 0), "fp32"),
}


def get_spec(spec) -> QuantSpec:
    return SPECS[spec] if isinstance(spec, str) else spec


def _blocks(x2: torch.Tensor, br: int, bc: int):
    """Pad (R, C) with zeros to multiples of the block and view it as (nR, br, nC, bc). Zeros do not change amax."""
    R, C = x2.shape
    br, bc = br or R, bc or C
    pr, pc = (-R) % br, (-C) % bc
    if pr or pc:
        x2 = torch.nn.functional.pad(x2, (0, pc, 0, pr))
    nR, nC = x2.shape[0] // br, x2.shape[1] // bc
    return x2.view(nR, br, nC, bc), (R, C)


def block_scales(xb: torch.Tensor, spec: QuantSpec, tensor_amax: torch.Tensor | None = None) -> torch.Tensor:
    """Decode scale per block, shape (nR, 1, nC, 1), from the blocked view (nR, br, nC, bc)."""
    fmt = spec.format
    amax = xb.abs().amax(dim=(1, 3), keepdim=True)
    mx = fmt.max_normal
    tiny = torch.finfo(torch.float32).tiny
    if spec.scale == "fp32":
        s = torch.where(amax > 0, amax / mx, torch.ones_like(amax))
    elif spec.scale == "pow2":
        s = torch.where(amax > 0, round_pow2(amax / mx, "ceil"), torch.ones_like(amax))
    elif spec.scale == "e8m0":
        emax = fmt.emax if isinstance(fmt, FloatFormat) else int(math.floor(math.log2(mx)))
        s = torch.where(amax > 0, round_pow2(amax, "floor") * 2.0 ** -emax, torch.ones_like(amax))
    elif spec.scale == "nvfp4":
        t = xb.abs().amax() if tensor_amax is None else tensor_amax
        t_dec = torch.where(t > 0, t / (mx * E4M3.max_normal), torch.ones_like(t))      # amax_t / (6 · 448), fp32
        b = round_to_format(amax / mx / t_dec, E4M3)                                     # block scale, stored in E4M3
        s = b * t_dec
        s = torch.where(s > tiny, s, torch.zeros_like(s))                                 # an all-tiny block underflows to 0
    else:
        raise ValueError(f"unknown scale kind {spec.scale!r}")
    return s


def quantize(x: torch.Tensor, spec, generator: torch.Generator | None = None):
    """Quantise ``x`` with ``spec``. Returns ``(q, scales)``: ``q`` the grid values (same shape as ``x``, float32 or
    float64, every element representable in the format) and ``scales`` of shape (nR, 1, nC, 1).

    ``dequantize(q, scales, spec)`` gives back ``s · q``.
    """
    spec = get_spec(spec)
    xw = x if x.dtype == torch.float64 else x.float()
    shape = xw.shape
    x2 = xw.reshape(-1, shape[-1]) if xw.dim() > 1 else xw.reshape(1, -1)
    xb, (R, C) = _blocks(x2, *spec.block)
    s = block_scales(xb, spec)
    safe = torch.where(s > 0, s, torch.ones_like(s))
    q = round_to_format(xb / safe, spec.format, spec.rounding, generator)
    q = torch.where(s > 0, q, torch.zeros_like(q))
    q = q.reshape(xb.shape[0] * xb.shape[1], -1)[:R, :C].reshape(shape)
    return q, s


def dequantize(q: torch.Tensor, scales: torch.Tensor, spec) -> torch.Tensor:
    spec = get_spec(spec)
    shape = q.shape
    q2 = q.reshape(-1, shape[-1]) if q.dim() > 1 else q.reshape(1, -1)
    qb, (R, C) = _blocks(q2, *spec.block)
    return (qb * scales).reshape(qb.shape[0] * qb.shape[1], -1)[:R, :C].reshape(shape)


def qdq(x: torch.Tensor, spec, generator: torch.Generator | None = None) -> torch.Tensor:
    """Quantise then dequantise ("fake quantisation"): what the low-precision GEMM sees, in ``x``'s dtype."""
    if spec is None:
        return x
    q, s = quantize(x, spec, generator)
    return dequantize(q, s, spec).to(x.dtype)


@torch.no_grad()
def quant_error(x: torch.Tensor, spec, generator: torch.Generator | None = None) -> dict:
    """How much a tensor loses: relative error ||x̂ - x|| / ||x||, the fraction of nonzero elements that became
    zero (underflow), the fraction whose scaled value exceeded the format's maximum and was clamped
    (saturation), and storage bits per value."""
    spec = get_spec(spec)
    xw = x.detach().double()
    if spec.rounding == "sr" and generator is None:      # never draw from the training RNG for a statistic
        generator = torch.Generator(device=xw.device).manual_seed(0)
    q, s = quantize(xw, spec, generator)
    xh = dequantize(q, s, spec)
    x2 = xw.reshape(-1, xw.shape[-1]) if xw.dim() > 1 else xw.reshape(1, -1)
    xb, _ = _blocks(x2, *spec.block)
    sat = ((xb.abs() > spec.format.max_normal * s) & (s > 0)).sum().item()
    nz = xw != 0
    rel = (torch.linalg.vector_norm(xh - xw) / torch.linalg.vector_norm(xw).clamp_min(1e-300)).item()
    return {"rel_err": rel,
            "underflow": ((xh == 0) & nz).sum().item() / max(1, int(nz.sum())),
            "saturated": sat / xw.numel(),
            "bits": spec.bits_per_value(tuple(x.shape))}
