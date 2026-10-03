"""RoPE at long range: the frequency-scaling rules of lesson 04.2 as plain functions.

RoPE (parent course lesson 12.1) rotates channel pair ``i`` of a query or key at position ``m`` by
the angle ``m * theta_i`` with ``theta_i = base ** (-2i / D)``, ``i = 0 .. D/2 - 1``, where ``D`` is the
number of rotated channels (the head dimension, or less with partial RoPE). Every rule below returns
a new vector of inverse frequencies ``theta_i`` (float64, shape ``(D/2,)``) and, for YaRN, an attention
factor. Nothing here touches positions: scaling a position by ``1/s`` is the same as scaling every
frequency by ``1/s`` because the angle is their product.

Rules (symbols: ``s`` the scale factor = new length / trained length, ``L`` the trained length):

* ``default``   theta_i = b^(-2i/D)
* ``pi``        Position Interpolation (Chen et al. 2023): theta_i / s for every pair.
* ``ntk``       "NTK-aware" scaling: change the base to b' = b * s^(D/(D-2)); the highest frequency
                (i = 0) is unchanged and the lowest (i = D/2 - 1) is divided by exactly s.
* ``yarn``      YaRN (Peng et al. 2023, arXiv 2309.00071): per pair, count the rotations the pair makes
                over the trained length, r_i = L / lambda_i with wavelength lambda_i = 2*pi / theta_i.
                Pairs with many rotations (r > beta) keep their frequency, pairs with fewer than alpha
                rotations are interpolated (theta_i / s), pairs in between are blended:
                h(theta_i) = (1 - gamma_i) * theta_i / s + gamma_i * theta_i. Plus an attention temperature:
                logits are multiplied by (0.1 ln s + 1)^2, implemented by multiplying cos and sin (hence q
                and k) by 0.1 ln s + 1.
* ``proportional``  p-RoPE (Barbero et al. 2024; Transformers' ``proportional`` type): keep the fastest
                fraction p of the *standard* frequencies b^(-2i/head_dim) and set the rest to zero (those
                pairs never rotate). Compare partial RoPE below, which recomputes the frequencies over the
                rotated channels only.
* ``llama3``    The rope type of the Llama 3.1 checkpoints as implemented in Hugging Face Transformers:
                a wavelength threshold version of the same idea (unchanged below L/high_freq_factor,
                divided by s above L/low_freq_factor, smooth in between).

Two YaRN ramps are provided. ``ramp="paper"`` is gamma linear in r, as in the paper's equations.
``ramp="code"`` (the default) is what the YaRN authors' code and Hugging Face Transformers 5.18.0
(``modeling_rope_utils._compute_yarn_parameters``) compute: the pair indices where r = beta_fast and
r = beta_slow are found, rounded outwards (floor / ceil), and the blend is linear in the *pair index*
between them. Released checkpoints (DeepSeek-V3, Qwen3, gpt-oss) run the code's version, so the
default matches it; ``labs/common/tests/test_longctx.py`` checks it against Transformers numerically.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

TYPES = ("default", "pi", "ntk", "yarn", "llama3", "proportional")


def default_inv_freq(rot_dim: int, base: float) -> torch.Tensor:
    """theta_i = base^(-2i/D), i = 0 .. D/2-1, float64."""
    if rot_dim % 2:
        raise ValueError("the number of rotated channels must be even")
    return 1.0 / (base ** (torch.arange(0, rot_dim, 2, dtype=torch.float64) / rot_dim))


def pi_inv_freq(rot_dim: int, base: float, factor: float) -> torch.Tensor:
    """Position Interpolation: every frequency divided by ``factor`` (positions m -> m / s)."""
    return default_inv_freq(rot_dim, base) / factor


def ntk_base(rot_dim: int, base: float, factor: float) -> float:
    """The NTK-aware base b' = b * s^(D/(D-2))."""
    return base * factor ** (rot_dim / (rot_dim - 2))


def ntk_inv_freq(rot_dim: int, base: float, factor: float) -> torch.Tensor:
    """NTK-aware scaling: the default frequencies computed with the base ``ntk_base``."""
    return default_inv_freq(rot_dim, ntk_base(rot_dim, base, factor))


def wavelengths(inv_freq: torch.Tensor) -> torch.Tensor:
    """lambda_i = 2*pi / theta_i, in tokens: how far apart two positions are when pair i has turned once."""
    return 2 * math.pi / inv_freq


def rotations(inv_freq: torch.Tensor, length: int) -> torch.Tensor:
    """r_i = L / lambda_i: full turns pair i makes over ``length`` tokens."""
    return length / wavelengths(inv_freq)


def yarn_correction_dim(num_rotations: float, rot_dim: int, base: float, length: int) -> float:
    """Pair index (real-valued) at which r_i = ``num_rotations`` over ``length`` tokens.

    From r_i = L * b^(-2i/D) / (2 pi):  i = D * ln(L / (2 pi n)) / (2 ln b).
    """
    return rot_dim * math.log(length / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def yarn_gamma(rot_dim: int, base: float, original_max: int, beta_fast: float = 32.0, beta_slow: float = 1.0,
               ramp: str = "code", truncate: bool = True) -> torch.Tensor:
    """gamma_i in [0, 1] per pair: 1 = keep the frequency (extrapolate), 0 = divide by s (interpolate)."""
    if ramp == "paper":
        r = rotations(default_inv_freq(rot_dim, base), original_max)
        return ((r - beta_slow) / (beta_fast - beta_slow)).clamp(0.0, 1.0)
    if ramp != "code":
        raise ValueError("ramp must be 'code' or 'paper'")
    low = yarn_correction_dim(beta_fast, rot_dim, base, original_max)
    high = yarn_correction_dim(beta_slow, rot_dim, base, original_max)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, rot_dim - 1)
    if low == high:
        high += 0.001                                        # as in the reference code: no division by zero
    idx = torch.arange(rot_dim // 2, dtype=torch.float64)
    return 1.0 - ((idx - low) / (high - low)).clamp(0.0, 1.0)


def yarn_inv_freq(rot_dim: int, base: float, factor: float, original_max: int, beta_fast: float = 32.0,
                  beta_slow: float = 1.0, ramp: str = "code", truncate: bool = True) -> torch.Tensor:
    """YaRN frequencies: h(theta_i) = (1 - gamma_i) * theta_i / s + gamma_i * theta_i."""
    theta = default_inv_freq(rot_dim, base)
    g = yarn_gamma(rot_dim, base, original_max, beta_fast, beta_slow, ramp, truncate)
    return (1 - g) * theta / factor + g * theta


def yarn_attention_factor(factor: float, mscale: float = 1.0) -> float:
    """sqrt(1/t) = 0.1 ln s + 1 (1 for s <= 1). Multiplying q and k by it multiplies the logits by its square."""
    return 1.0 if factor <= 1 else 0.1 * mscale * math.log(factor) + 1.0


def proportional_inv_freq(head_dim: int, base: float, keep: float, factor: float = 1.0) -> torch.Tensor:
    """p-RoPE: the standard frequencies of the first int(keep * head_dim / 2) pairs, zero for the rest, / factor."""
    n = int(keep * head_dim // 2)
    theta = default_inv_freq(head_dim, base)
    theta[n:] = 0.0
    return theta / factor


def llama3_inv_freq(rot_dim: int, base: float, factor: float, original_max: int, low_freq_factor: float = 1.0,
                    high_freq_factor: float = 4.0) -> torch.Tensor:
    """Llama 3.1's rope type, as implemented in Transformers (``_compute_llama3_parameters``)."""
    theta = default_inv_freq(rot_dim, base)
    wl = wavelengths(theta)
    low_wl, high_wl = original_max / low_freq_factor, original_max / high_freq_factor
    out = torch.where(wl > low_wl, theta / factor, theta)
    smooth = (original_max / wl - low_freq_factor) / (high_freq_factor - low_freq_factor)
    mid = (wl >= high_wl) & (wl <= low_wl)
    return torch.where(mid, (1 - smooth) * out / factor + smooth * out, out)


@dataclass
class RopeScaling:
    """One RoPE configuration, read from ``ModelConfig.extra["rope"]`` (all keys optional)::

        {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 256,
         "beta_fast": 32, "beta_slow": 1, "ramp": "code", "attention_factor": None,
         "partial_rotary_factor": 1.0}

    ``attention_factor=None`` means "derive it": YaRN's 0.1 ln s + 1, 1 for every other type.
    ``partial_rotary_factor`` p < 1 rotates only the first int(p * head_dim) channels of each head
    (partial RoPE, the GPT-NeoX / Qwen3-Next convention: frequencies b^(-2i/(p*head_dim)) over the
    rotated channels, so they still span the full range from 1 to about 1/b); the rest carry no
    position signal. With ``type="proportional"`` the same factor means p-RoPE instead (all channels,
    standard frequencies, the slowest (1 - p) set to zero). It is an architecture change, not a
    context-extension trick: a model trained with p = 1 does not work with p < 1 without retraining.
    """

    type: str = "default"
    factor: float = 1.0
    original_max_position_embeddings: int | None = None
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    ramp: str = "code"
    truncate: bool = True
    attention_factor: float | None = None
    mscale: float = 1.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    partial_rotary_factor: float = 1.0

    @classmethod
    def from_config(cls, cfg) -> "RopeScaling":
        d = dict(cfg.extra.get("rope", {}))
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown rope keys {sorted(unknown)}; known: {sorted(known)}")
        spec = cls(**d)
        if spec.type not in TYPES:
            raise ValueError(f"rope type {spec.type!r} not in {TYPES}")
        if spec.original_max_position_embeddings is None:
            spec.original_max_position_embeddings = cfg.max_position_embeddings
        return spec

    def rot_dim(self, head_dim: int) -> int:
        if self.type == "proportional":
            return head_dim
        r = int(head_dim * self.partial_rotary_factor)
        if r % 2 or r <= 0:
            raise ValueError(f"partial_rotary_factor {self.partial_rotary_factor} gives {r} rotated channels; need an even number > 0")
        return r

    def inv_freq(self, head_dim: int, base: float) -> torch.Tensor:
        D, s, L = self.rot_dim(head_dim), self.factor, self.original_max_position_embeddings
        if self.type == "default" or s == 1.0 and self.type not in ("yarn", "proportional"):
            return default_inv_freq(D, base)
        if self.type == "pi":
            return pi_inv_freq(D, base, s)
        if self.type == "ntk":
            return ntk_inv_freq(D, base, s)
        if self.type == "yarn":
            return yarn_inv_freq(D, base, s, L, self.beta_fast, self.beta_slow, self.ramp, self.truncate)
        if self.type == "proportional":
            return proportional_inv_freq(head_dim, base, self.partial_rotary_factor, s)
        return llama3_inv_freq(D, base, s, L, self.low_freq_factor, self.high_freq_factor)

    def get_attention_factor(self) -> float:
        if self.attention_factor is not None:
            return float(self.attention_factor)
        return yarn_attention_factor(self.factor, self.mscale) if self.type == "yarn" else 1.0

    def rotary(self, head_dim: int, base: float) -> "ScaledRotaryEmbedding":
        return ScaledRotaryEmbedding(self.inv_freq(head_dim, base), self.get_attention_factor())

    def to_dict(self) -> dict:
        return asdict(self)


class ScaledRotaryEmbedding(nn.Module):
    """Drop-in for :class:`frontierlab.attention.base.RotaryEmbedding` with given frequencies.

    ``forward(positions)`` returns ``cos * a, sin * a`` of shape ``(T, rot_dim)``, where ``a`` is the
    attention factor. :func:`frontierlab.attention.base.apply_rope` multiplies q and k by them, so the
    rotated channels of both are scaled by ``a`` and their dot product by ``a**2``. As in Transformers,
    channels left unrotated by partial RoPE are *not* scaled.
    """

    def __init__(self, inv_freq: torch.Tensor, attention_factor: float = 1.0):
        super().__init__()
        self.rot_dim = 2 * inv_freq.numel()
        self.attention_factor = float(attention_factor)
        self.register_buffer("inv_freq", inv_freq.float(), persistent=False)

    def forward(self, positions: torch.Tensor):
        inv = self.inv_freq.to(positions.device)
        freqs = torch.outer(positions.to(inv.dtype), inv)          # (T, rot/2): angle m * theta_i
        emb = torch.cat((freqs, freqs), dim=-1)                    # (T, rot), "rotate half" layout
        a = self.attention_factor
        return emb.cos() * a, emb.sin() * a


def frequency_table(head_dim: int, base: float, factor: float, original_max: int, beta_fast: float = 32.0,
                    beta_slow: float = 1.0) -> list[dict]:
    """Per pair: wavelength, rotations over the trained length, and the divisor each rule applies.

    ``divisor = theta_default / theta_rule``: 1 means unchanged, ``factor`` means fully interpolated.
    This is the table of lesson 04.2's worked example.
    """
    base_f = default_inv_freq(head_dim, base)
    rules = {"pi": pi_inv_freq(head_dim, base, factor), "ntk": ntk_inv_freq(head_dim, base, factor),
             "yarn_code": yarn_inv_freq(head_dim, base, factor, original_max, beta_fast, beta_slow, "code"),
             "yarn_paper": yarn_inv_freq(head_dim, base, factor, original_max, beta_fast, beta_slow, "paper")}
    wl, r = wavelengths(base_f), rotations(base_f, original_max)
    rows = []
    for i in range(head_dim // 2):
        row = {"pair": i, "theta": float(base_f[i]), "wavelength": float(wl[i]), "rotations": float(r[i])}
        row.update({k: float(base_f[i] / v[i]) for k, v in rules.items()})
        rows.append(row)
    return rows
