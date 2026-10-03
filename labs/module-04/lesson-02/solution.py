"""Reference solution for lab 04.2."""

from __future__ import annotations

import math

import torch


def default_inv_freq(rot_dim, base):
    return 1.0 / (base ** (torch.arange(0, rot_dim, 2, dtype=torch.float64) / rot_dim))


def pi_inv_freq(rot_dim, base, factor):
    return default_inv_freq(rot_dim, base) / factor


def ntk_inv_freq(rot_dim, base, factor):
    return default_inv_freq(rot_dim, base * factor ** (rot_dim / (rot_dim - 2)))


def rotations(inv_freq, length):
    return length / (2 * math.pi / inv_freq)


def yarn_gamma(rot_dim, base, original_max, beta_fast=32.0, beta_slow=1.0):
    def d(n):
        return rot_dim * math.log(original_max / (2 * math.pi * n)) / (2 * math.log(base))
    low, high = max(math.floor(d(beta_fast)), 0), min(math.ceil(d(beta_slow)), rot_dim - 1)
    if low == high:
        high += 0.001
    i = torch.arange(rot_dim // 2, dtype=torch.float64)
    return 1.0 - ((i - low) / (high - low)).clamp(0.0, 1.0)


def yarn_inv_freq(rot_dim, base, factor, original_max, beta_fast=32.0, beta_slow=1.0):
    theta = default_inv_freq(rot_dim, base)
    g = yarn_gamma(rot_dim, base, original_max, beta_fast, beta_slow)
    return (1 - g) * theta / factor + g * theta


def yarn_attention_factor(factor):
    return 0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0
