"""Lab 04.2 — RoPE frequency rules for long context. Fill in the TODOs; run ``pytest labs/module-04/lesson-02``.

Conventions (lesson 04.2): D rotated channels, pairs i = 0 .. D/2-1, theta_i = base^(-2i/D),
s = new length / trained length, L = trained length. Return float64 tensors of shape (D/2,).
"""

from __future__ import annotations

import math

import torch


def default_inv_freq(rot_dim: int, base: float) -> torch.Tensor:
    """theta_i = base^(-2i/D) (given)."""
    return 1.0 / (base ** (torch.arange(0, rot_dim, 2, dtype=torch.float64) / rot_dim))


def pi_inv_freq(rot_dim: int, base: float, factor: float) -> torch.Tensor:
    """Position Interpolation (given): every frequency divided by s."""
    return default_inv_freq(rot_dim, base) / factor


def ntk_inv_freq(rot_dim: int, base: float, factor: float) -> torch.Tensor:
    """NTK-aware scaling: the default rule with the base replaced by b' = b * s^(D / (D - 2))."""
    raise NotImplementedError("TODO 1: NTK-aware base change")


def rotations(inv_freq: torch.Tensor, length: int) -> torch.Tensor:
    """r_i = L / lambda_i, with wavelength lambda_i = 2 pi / theta_i: full turns of pair i over ``length`` tokens."""
    raise NotImplementedError("TODO 2: rotations per pair")


def yarn_gamma(rot_dim: int, base: float, original_max: int, beta_fast: float = 32.0, beta_slow: float = 1.0) -> torch.Tensor:
    """YaRN's blend weight per pair, as the reference code computes it (gamma = 1 keep, 0 interpolate).

    1. The real-valued pair index where pair i makes n rotations over L tokens is
       d(n) = D * ln(L / (2 pi n)) / (2 ln base).
    2. low = floor(d(beta_fast)), high = ceil(d(beta_slow)); then low = max(low, 0), high = min(high, D - 1).
    3. ramp_i = clamp((i - low) / (high - low), 0, 1) for i = 0 .. D/2-1 (if low == high, add 0.001 to high).
    4. gamma_i = 1 - ramp_i.
    """
    raise NotImplementedError("TODO 3: the code's ramp, linear in the pair index")


def yarn_inv_freq(rot_dim: int, base: float, factor: float, original_max: int, beta_fast: float = 32.0,
                  beta_slow: float = 1.0) -> torch.Tensor:
    """h(theta_i) = (1 - gamma_i) * theta_i / s + gamma_i * theta_i."""
    raise NotImplementedError("TODO 4: blend interpolated and original frequencies with gamma")


def yarn_attention_factor(factor: float) -> float:
    """sqrt(1/t) = 0.1 ln s + 1 for s > 1, else 1. It multiplies cos and sin, so q.k is scaled by its square."""
    raise NotImplementedError("TODO 5: YaRN's attention factor")


_ = math  # available for TODOs 2, 3 and 5
