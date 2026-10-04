"""Number formats for low-precision training, emulated exactly in float32/float64 (lesson 08.1).

A binary floating-point format with ``e`` exponent bits, ``m`` mantissa bits and bias ``b`` stores

    normal      (-1)^s · 2^(E - b) · (1 + f / 2^m)      E = 1 .. (max exponent field)
    subnormal   (-1)^s · 2^(1 - b) · (f / 2^m)          E = 0

so inside the binade [2^k, 2^(k+1)) the representable values are spaced ``2^(k - m)`` apart, and below the
smallest normal 2^(1 - b) the spacing stays 2^(1 - b - m) (subnormals). Rounding a real number to the format
therefore needs only its binade: ``k = floor(log2 |x|)`` (from ``torch.frexp``, exact), clamped below at
``1 - b``. Dividing by the spacing, rounding to an integer and multiplying back are exact in float32 for every
format here (all have at most 7 mantissa bits), so :func:`round_to_format` is a bit-exact emulation, not an
approximation.

Formats (sources: OCP 8-bit Floating Point Specification (OFP8) and OCP Microscaling Formats (MX) v1.0;
Micikevicius et al., FP8 Formats for Deep Learning, arXiv 2209.05433):

    name   e  m  bias  max normal   min normal   min subnormal  inf?  NaN encodings     torch dtype
    e4m3   4  3   7    448          2^-6         2^-9           no    S.1111.111        float8_e4m3fn
    e5m2   5  2  15    57344        2^-14        2^-16          yes   S.11111.{01,10,11} float8_e5m2
    e2m1   2  1   1    6            1            0.5            no    none              float4_e2m1fn_x2 (packed)
    e3m2   3  2   3    28           0.25         0.0625         no    none              (MX FP6)
    e2m3   2  3   1    7.5          1            0.125          no    none              (MX FP6)
    bf16   8  7 127    3.39e38      2^-126       2^-133         yes   IEEE              bfloat16

E4M3 gives up infinities to gain one more binade of normal values: the all-ones exponent is a normal binade
except mantissa 111, which is the only NaN, so the largest value is 1.75 · 2^8 = 448 instead of 240.

Scale formats: ``E8M0`` (MX, DeepSeek-V3.1's "UE8M0") is an unsigned exponent with no mantissa: the value is
2^(E - 127), E = 0..254, and E = 255 is NaN. ``round_pow2`` rounds a positive number to a power of two.

Integer formats: :class:`IntFormat` (symmetric, e.g. INT4 in [-7, 7] or [-8, 7]).

Overflow: real hardware casts used by training recipes saturate (clamp to ±max) after scaling; IEEE-style
casts overflow to inf (E5M2) or NaN (E4M3). :func:`round_to_format` saturates by default; pass
``saturate=False`` for the IEEE behaviour. What ``tensor.to(torch.float8_*)`` does on overflow is a property of
your PyTorch build (on torch 2.14.1 CPU, measured in this build: E4M3FN saturates to 448, E5M2 overflows to inf
from 61440 = 57344 + half a spacing); the tests check in-range values bit for bit and the overflow rules separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FloatFormat:
    name: str
    exp_bits: int
    man_bits: int
    bias: int
    max_normal: float
    has_inf: bool
    nan_rule: str            # "ieee" (exp all ones, mantissa != 0), "e4m3fn" (S.1111.111 only), "none"
    torch_dtype: str | None = None

    @property
    def bits(self) -> int:
        return 1 + self.exp_bits + self.man_bits

    @property
    def emin(self) -> int:
        """Exponent of the smallest normal binade (1 - bias)."""
        return 1 - self.bias

    @property
    def emax(self) -> int:
        """Exponent of the largest normal value: max_normal = (1 + f) · 2^emax."""
        return math.floor(math.log2(self.max_normal))

    @property
    def min_normal(self) -> float:
        return 2.0 ** self.emin

    @property
    def min_subnormal(self) -> float:
        return 2.0 ** (self.emin - self.man_bits)

    @property
    def epsilon(self) -> float:
        """Spacing just above 1: 2^-m. Relative rounding error of round-to-nearest is at most half of it."""
        return 2.0 ** -self.man_bits

    @property
    def dynamic_range_binades(self) -> float:
        """log2(max normal / min subnormal): how many factors of two the format spans (with subnormals)."""
        return math.log2(self.max_normal / self.min_subnormal)

    def dtype(self):
        return getattr(torch, self.torch_dtype) if self.torch_dtype and hasattr(torch, self.torch_dtype) else None


@dataclass(frozen=True)
class IntFormat:
    """Symmetric integer grid {-qmax, ..., qmax} (``restricted``) or {-qmax-1, ..., qmax}."""
    name: str
    int_bits: int
    restricted: bool = True

    @property
    def bits(self) -> int:
        return self.int_bits

    @property
    def max_normal(self) -> float:          # the largest grid value, so scaling code treats both kinds alike
        return float(2 ** (self.int_bits - 1) - 1)

    @property
    def min_value(self) -> float:
        return -self.max_normal if self.restricted else -self.max_normal - 1


E4M3 = FloatFormat("e4m3", 4, 3, 7, 448.0, False, "e4m3fn", "float8_e4m3fn")
E5M2 = FloatFormat("e5m2", 5, 2, 15, 57344.0, True, "ieee", "float8_e5m2")
E2M1 = FloatFormat("e2m1", 2, 1, 1, 6.0, False, "none", None)
E3M2 = FloatFormat("e3m2", 3, 2, 3, 28.0, False, "none", None)
E2M3 = FloatFormat("e2m3", 2, 3, 1, 7.5, False, "none", None)
BF16 = FloatFormat("bf16", 8, 7, 127, float(torch.finfo(torch.bfloat16).max), True, "ieee", "bfloat16")
FP16 = FloatFormat("fp16", 5, 10, 15, 65504.0, True, "ieee", "float16")
INT8 = IntFormat("int8", 8)
INT4 = IntFormat("int4", 4)
INT3 = IntFormat("int3", 3)

FORMATS: dict[str, FloatFormat | IntFormat] = {f.name: f for f in (E4M3, E5M2, E2M1, E3M2, E2M3, BF16, FP16, INT8, INT4, INT3)}


def int_format(bits: int, restricted: bool = True) -> IntFormat:
    """INT``bits`` symmetric grid (used by the precision-scaling sweep of lesson 08.4)."""
    return IntFormat(f"int{bits}", bits, restricted)


def get_format(name: str):
    if name in FORMATS:
        return FORMATS[name]
    if name.startswith("int") and name[3:].isdigit():
        return int_format(int(name[3:]))
    raise KeyError(f"unknown format {name!r}; known: {sorted(FORMATS)} or intN")


# ----------------------------------------------------------------------------------------- rounding

def _work(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.float64 else x.float()


def _round_int(q: torch.Tensor, rounding: str, generator: torch.Generator | None) -> torch.Tensor:
    """Round to integers: "rne" = to nearest, ties to even (torch.round); "sr" = stochastic; "trunc" = toward zero."""
    if rounding == "rne":
        return torch.round(q)
    if rounding == "sr":
        u = torch.rand(q.shape, generator=generator, device=q.device, dtype=q.dtype)
        return torch.floor(q + u)            # P(round up) = fractional part: unbiased, E[result] = q
    if rounding == "trunc":
        return torch.trunc(q)
    raise ValueError(f"rounding must be rne, sr or trunc, not {rounding!r}")


def round_to_format(x: torch.Tensor, fmt: FloatFormat | IntFormat | str, rounding: str = "rne",
                    generator: torch.Generator | None = None, saturate: bool = True) -> torch.Tensor:
    """Round every element of ``x`` to the nearest value of ``fmt`` (no scaling). Same dtype as the input
    (float32 work for float16/bfloat16 inputs, returned as float32).

    ``rounding``: "rne" (round to nearest even, what hardware casts do), "sr" (stochastic rounding: up with
    probability equal to the distance from the lower neighbour, in units of the spacing), "trunc".
    ``saturate``: clamp results to ±max_normal (recipes); False gives inf (E5M2) / NaN (E4M3) on overflow.
    NaN inputs stay NaN.
    """
    fmt = get_format(fmt) if isinstance(fmt, str) else fmt
    xw = _work(x)
    if isinstance(fmt, IntFormat):
        q = _round_int(xw, rounding, generator)
        return q.clamp(fmt.min_value, fmt.max_normal)
    if xw.dtype == torch.float32 and fmt.emin - fmt.man_bits >= -126:
        # Fast path, same arithmetic: read the binade straight from the float32 exponent bits (a float32
        # subnormal input reads as -127 and is clamped), build the spacing 2^(k - m) as a float32 bit pattern,
        # divide (exact for a power of two), round, multiply back.
        k = torch.clamp(((xw.view(torch.int32) >> 23) & 0xFF) - 127, min=fmt.emin)
        spacing = ((k - fmt.man_bits + 127) << 23).view(torch.float32)
        y = _round_int(xw / spacing, rounding, generator) * spacing
    else:
        _, e = torch.frexp(xw)                               # x = m · 2^e, 0.5 <= |m| < 1, so floor(log2|x|) = e - 1
        k = torch.clamp(e - 1, min=fmt.emin)                 # binade; subnormals share the smallest normal's spacing
        shift = (k - fmt.man_bits).to(torch.int32)
        q = torch.ldexp(xw, -shift)                          # x / spacing, exact
        y = torch.ldexp(_round_int(q, rounding, generator), shift)
    over = y.abs() > fmt.max_normal
    if saturate:
        y = torch.where(over, torch.sign(y) * fmt.max_normal, y)
    elif fmt.has_inf:
        y = torch.where(over, torch.sign(y) * float("inf"), y)
    else:
        y = torch.where(over, torch.full_like(y, float("nan")), y)
    return torch.where(torch.isnan(xw), xw, y)


def round_pow2(x: torch.Tensor, mode: str = "ceil") -> torch.Tensor:
    """Round positive values to a power of two (an E8M0 / UE8M0 scale). ``mode``: "ceil" (never smaller),
    "floor" (never larger), "nearest" (mantissa >= 1.5 rounds up; what torch's float8_e8m0fnu cast does on
    torch 2.14). The exponent is clamped to E8M0's range [-127, 127]. Zero maps to 2^-127."""
    xw = _work(x)
    m, e = torch.frexp(xw)                                   # x = m · 2^e, m in [0.5, 1)
    k = (e - 1).to(torch.float64)                            # x = (2m) · 2^k, 2m in [1, 2)
    m2 = 2 * m
    if mode == "floor":
        pass
    elif mode == "ceil":
        k = k + (m2 > 1).to(k.dtype)
    elif mode == "nearest":
        k = k + (m2 >= 1.5).to(k.dtype)
    else:
        raise ValueError("mode must be ceil, floor or nearest")
    k = torch.where(xw > 0, k, torch.full_like(k, -127.0)).clamp(-127, 127)
    return torch.exp2(k).to(xw.dtype)


# ------------------------------------------------------------------------------------- bit patterns

def decode(codes: torch.Tensor, fmt: FloatFormat) -> torch.Tensor:
    """Value (float64) of each bit pattern ``codes`` (integer tensor, low ``fmt.bits`` bits used)."""
    c = codes.to(torch.int64)
    m_mask = (1 << fmt.man_bits) - 1
    e_mask = (1 << fmt.exp_bits) - 1
    s = (c >> (fmt.exp_bits + fmt.man_bits)) & 1
    E = (c >> fmt.man_bits) & e_mask
    f = (c & m_mask).to(torch.float64)
    sub = E == 0
    mag = torch.where(sub, f / 2 ** fmt.man_bits * 2.0 ** fmt.emin,
                      (1 + f / 2 ** fmt.man_bits) * torch.exp2((E - fmt.bias).to(torch.float64)))
    if fmt.nan_rule == "ieee":
        top = E == e_mask
        mag = torch.where(top & (f == 0), torch.full_like(mag, float("inf")), mag)
        mag = torch.where(top & (f != 0), torch.full_like(mag, float("nan")), mag)
    elif fmt.nan_rule == "e4m3fn":
        mag = torch.where((E == e_mask) & ((c & m_mask) == m_mask), torch.full_like(mag, float("nan")), mag)
    return torch.where(s == 1, -mag, mag)


def encode(x: torch.Tensor, fmt: FloatFormat) -> torch.Tensor:
    """Bit pattern (int64) of each value of ``x``, which must already be representable in ``fmt`` (round first).

    Zero encodes as +0 or -0 by sign; NaN as the format's canonical NaN; ±inf only where the format has it.
    """
    xw = x.to(torch.float64)
    sign = (torch.signbit(xw)).to(torch.int64)
    a = xw.abs()
    _, e = torch.frexp(a)
    k = torch.clamp(e - 1, min=fmt.emin)
    normal = a >= fmt.min_normal
    E = torch.where(normal, k + fmt.bias, torch.zeros_like(k)).to(torch.int64)
    frac = torch.where(normal, a / torch.exp2(k.to(torch.float64)) - 1, a / 2.0 ** fmt.emin)
    f = torch.round(frac * 2 ** fmt.man_bits).to(torch.int64)
    code = (sign << (fmt.exp_bits + fmt.man_bits)) | (E << fmt.man_bits) | f
    e_mask = (1 << fmt.exp_bits) - 1
    m_mask = (1 << fmt.man_bits) - 1
    if fmt.nan_rule == "e4m3fn":
        nan_code = (e_mask << fmt.man_bits) | m_mask
    elif fmt.nan_rule == "ieee":
        nan_code = (e_mask << fmt.man_bits) | (1 << (fmt.man_bits - 1))
        inf_code = (sign << (fmt.exp_bits + fmt.man_bits)) | (e_mask << fmt.man_bits)
        code = torch.where(torch.isinf(xw), inf_code, code)
    else:
        nan_code = -1
    return torch.where(torch.isnan(xw), torch.full_like(code, nan_code), code)


def representable_values(fmt: FloatFormat | IntFormat, nonnegative: bool = True) -> torch.Tensor:
    """Sorted finite values of a format with at most 8 bits (float64)."""
    if isinstance(fmt, IntFormat):
        v = torch.arange(int(fmt.min_value), int(fmt.max_normal) + 1, dtype=torch.float64)
    else:
        if fmt.bits > 8:
            raise ValueError("enumerate only formats with at most 8 bits")
        v = decode(torch.arange(2 ** fmt.bits), fmt)
        v = v[torch.isfinite(v)]
    v = torch.unique(v.abs() if nonnegative else v)
    return torch.sort(v).values


def describe(fmt: FloatFormat | IntFormat) -> dict:
    """The numbers of the table in the module docstring, computed from the bit layout."""
    if isinstance(fmt, IntFormat):
        return {"name": fmt.name, "bits": fmt.bits, "max": fmt.max_normal, "levels": int(fmt.max_normal - fmt.min_value + 1)}
    return {"name": fmt.name, "bits": fmt.bits, "exp_bits": fmt.exp_bits, "man_bits": fmt.man_bits, "bias": fmt.bias,
            "max_normal": fmt.max_normal, "min_normal": fmt.min_normal, "min_subnormal": fmt.min_subnormal,
            "epsilon": fmt.epsilon, "binades": round(fmt.dynamic_range_binades, 2)}
