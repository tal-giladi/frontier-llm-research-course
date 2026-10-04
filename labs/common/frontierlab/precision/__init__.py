"""Module 8 — how low can precision go?

* :mod:`~frontierlab.precision.formats` — FP8 (E4M3, E5M2), FP6, FP4 (E2M1), INT formats and E8M0 scales: bit
  layouts, exact rounding (nearest-even, stochastic) and bit-pattern encode/decode (lesson 08.1).
* :mod:`~frontierlab.precision.quant` — scaled quantisation with per-tensor, per-row, per-tile and per-block
  scales (fp32, power-of-2, OCP E8M0, NVFP4 two-level); error statistics and bits per value (08.1).
* :mod:`~frontierlab.precision.accum` — a model of limited-precision GEMM accumulation and FP32 promotion (08.2).
* :mod:`~frontierlab.precision.hadamard` — random Hadamard transforms (08.3).
* :mod:`~frontierlab.precision.linear` — :class:`QLinear` and the training recipes (FP8 tensorwise / rowwise /
  DeepSeek-V3 fine-grained, MXFP8, MXFP4, NVFP4 and ablations, QAT), linear swapping and PTQ (08.2–08.3).
* :mod:`~frontierlab.precision.qat` — fake quantisation with a straight-through estimator, QAT and export (08.3).
* :mod:`~frontierlab.precision.monitor` — the per-step quantisation-statistics log (08.2–08.3).
* :mod:`~frontierlab.precision.torchao_path` — torchao Float8 training on CUDA (08.2, main path only).
* :mod:`~frontierlab.precision.scaling_law` — the precision scaling law's functional forms and a small fitter (08.4).
* :mod:`~frontierlab.precision.train` — the training wrapper around ``frontierlab.train.loop``.

Everything except ``torchao_path`` is emulation: it reproduces the numerics of a format or recipe in float32 or
float64 and says nothing about speed.
"""

from frontierlab.precision.formats import (BF16, E2M1, E3M2, E2M3, E4M3, E5M2, FORMATS, FloatFormat, IntFormat, decode,
                                           describe, encode, get_format, representable_values, round_pow2,
                                           round_to_format)
from frontierlab.precision.hadamard import apply_rht, hadamard, rht_matrix
from frontierlab.precision.linear import RECIPES, QLinear, Recipe, get_recipe, parse_keep, ptq_, select_linears, swap_linears
from frontierlab.precision.quant import SPECS, QuantSpec, dequantize, get_spec, qdq, quant_error, quantize

__all__ = ["BF16", "E2M1", "E2M3", "E3M2", "E4M3", "E5M2", "FORMATS", "FloatFormat", "IntFormat", "QLinear", "QuantSpec",
           "RECIPES", "Recipe", "SPECS", "apply_rht", "decode", "dequantize", "describe", "encode", "get_format",
           "get_recipe", "get_spec", "hadamard", "parse_keep", "ptq_", "qdq", "quant_error", "quantize",
           "representable_values", "rht_matrix", "round_pow2", "round_to_format", "select_linears", "swap_linears"]
