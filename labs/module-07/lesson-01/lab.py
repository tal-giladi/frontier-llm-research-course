"""Lab 07.1 — Muon from scratch. Fill in the TODOs; run `pytest labs/module-07/lesson-01` to check.

Shapes: a weight matrix W is (A, B) = (fan_out, fan_in), as nn.Linear stores it; its gradient has the same
shape. Work in float64 here (the tests compare with float64 references); the shared implementation in
frontierlab.optim.muon runs the same arithmetic in bfloat16.
"""

from __future__ import annotations

import torch

QUINTIC = (3.4445, -4.7750, 2.0315)


def ns_step(X: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    """One Newton-Schulz step on a wide matrix X (r, n), r <= n: A = X X^T, B = b A + c A A, return a X + B X."""
    raise NotImplementedError("TODO 1: one Newton-Schulz step")


def newton_schulz(G: torch.Tensor, coeffs=QUINTIC, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalise G (A, B): same singular vectors, singular values pushed towards 1.

    1. If G is tall (A > B), work on G^T so that X X^T is the small (min(A,B))^2 Gram matrix.
    2. Divide by the Frobenius norm (+ eps): every singular value is then <= 1.
    3. Apply ``steps`` calls of ns_step with ``coeffs``.
    4. Transpose back if you transposed. Return the same shape as G.
    """
    raise NotImplementedError("TODO 2: the Newton-Schulz iteration")


def muon_direction(grad: torch.Tensor, buf: torch.Tensor, momentum: float = 0.95, nesterov: bool = True):
    """Muon's direction for one matrix. Updates ``buf`` in place (buf <- momentum * buf + grad).

    Returns NS(u) with u = grad + momentum * buf (Nesterov) or u = buf (plain momentum), using newton_schulz.
    """
    raise NotImplementedError("TODO 3: momentum, then orthogonalise")


def muon_apply(W: torch.Tensor, O: torch.Tensor, lr: float, weight_decay: float, adjust: str = "match_rms") -> None:
    """In place: W <- W - lr * weight_decay * W, then W <- W - lr * s(A, B) * O.

    s = 0.2 * sqrt(max(A, B)) for "match_rms" (Moonlight Eq. 4), sqrt(max(1, A / B)) for "original".
    """
    raise NotImplementedError("TODO 4: decoupled weight decay and the scaled update")


def muon_param_names(model) -> list[str]:
    """Names of the parameters of a frontierlab.model.LM that Muon should update (everything else: AdamW)."""
    raise NotImplementedError("TODO 5: which parameters go to Muon")


def ns_flops(shape, steps: int = 5) -> int:
    """FLOPs of ``steps`` Newton-Schulz steps on a matrix of ``shape`` (1 multiply-add = 2 FLOPs).

    Count only the three matrix products per step, on the wide orientation (r = min, n = max).
    """
    raise NotImplementedError("TODO 6: count the Newton-Schulz FLOPs")
