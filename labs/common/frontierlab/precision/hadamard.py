"""Random Hadamard transforms (RHT) for FP4 weight-gradient GEMMs (lesson 08.3).

A Hadamard matrix H of size d (a power of two) has entries ±1/√d and orthonormal rows: H Hᵀ = I. Multiplying by
a random sign diagonal S keeps that: (H S)(H S)ᵀ = I. So for the weight-gradient GEMM, which contracts over the
token dimension M,

    dW = Gᵀ X = (Gᵀ Q)(Qᵀ X)          for any orthogonal Q acting on M,

and applying Q = blockdiag(H S, H S, ...) in blocks of d tokens changes nothing in exact arithmetic. It changes
what the *quantiser* sees: each block of d values becomes d mixtures of all of them, so one outlier is spread
over the block and the block's amax-based scale stops being set by a single value. Gaussian-like blocks
quantise with less error than blocks with outliers.

NVIDIA's NVFP4 recipe (arXiv 2509.25149 section 4.2): "Apply Random Hadamard transforms of size 16 × 16 to inputs
of weight gradient GEMMs", restricted to Wgrad because Fprop and Dgrad showed no measurable benefit at smaller
scales; "a single random sign vector that is shared across all linear layers throughout training".
"""

from __future__ import annotations

import torch


def hadamard(d: int, dtype=torch.float64, device="cpu") -> torch.Tensor:
    """Normalised Sylvester Hadamard matrix of size d (power of two): H_{2n} = [[H, H], [H, -H]] / √2."""
    if d < 1 or d & (d - 1):
        raise ValueError("d must be a power of two")
    H = torch.ones(1, 1, dtype=dtype, device=device)
    while H.shape[0] < d:
        H = torch.cat((torch.cat((H, H), 1), torch.cat((H, -H), 1)), 0)
    return H / H.shape[0] ** 0.5


def random_signs(d: int, seed: int = 0, dtype=torch.float64, device="cpu") -> torch.Tensor:
    """A fixed ±1 vector from its own generator, so it never touches (or depends on) the training RNG."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (d,), generator=g) * 2 - 1).to(dtype=dtype, device=device)


def rht_matrix(d: int = 16, seed: int = 0, dtype=torch.float64, device="cpu") -> torch.Tensor:
    """Q = H · diag(signs): orthogonal, entries ±1/√d."""
    return hadamard(d, dtype, device) * random_signs(d, seed, dtype, device)[None, :]


def apply_rht(x: torch.Tensor, d: int = 16, seed: int = 0) -> torch.Tensor:
    """Transform the LAST dimension of ``x`` in blocks of d: each block v becomes v · Q (zero-padded, then cut back
    only if no padding was needed — callers pass a last dimension that is a multiple of d, or accept the padding).

    Returns a tensor whose last dimension is rounded up to a multiple of d (padding with zeros, which keeps
    Gᵀ X unchanged when both operands are padded the same way).
    """
    n = x.shape[-1]
    pad = (-n) % d
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    Q = rht_matrix(d, seed, dtype=x.dtype if x.dtype == torch.float64 else torch.float32, device=x.device)
    xb = x.reshape(*x.shape[:-1], -1, d)
    return (xb.to(Q.dtype) @ Q).reshape(*x.shape[:-1], -1).to(x.dtype)
