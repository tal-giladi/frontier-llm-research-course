"""Lab 17.1 — features and sparse autoencoders. Fill in the TODOs; run `pytest labs/module-17/lesson-01`.

``sae_lab.py`` checks your functions against the course's (``frontierlab.interp.sae``) and then trains ReLU,
TopK and JumpReLU SAEs on a toy language model's residual stream and splices them back in.
"""

from __future__ import annotations

import torch


def topk_code(pre: torch.Tensor, k: int) -> torch.Tensor:
    """TopK activation (Gao et al. 2024, Eq. 2, with a ReLU after): along the last dimension keep the k largest
    entries of ``pre``, pass them through ReLU, and set every other entry to 0. Same shape as ``pre``."""
    raise NotImplementedError("TODO 1: TopK activation")


class JumpReLU(torch.autograd.Function):
    """z = pre · H(pre − θ) with θ (d_sae,) > 0, applied over the last dimension of pre (..., d_sae).

    backward must return:
      * w.r.t. pre: g · H(pre − θ)
      * w.r.t. θ:   Σ over all leading dimensions of g · (−θ/ε) · rect((pre − θ)/ε), where rect(u) = 1 if
                    −1/2 < u < 1/2 else 0 (the straight-through estimator of Rajamanoharan et al. 2024, Eq. 11)
      * w.r.t. ε:   None
    """

    @staticmethod
    def forward(ctx, pre, theta, eps):
        raise NotImplementedError("TODO 2: JumpReLU forward (save what backward needs)")

    @staticmethod
    def backward(ctx, g):
        raise NotImplementedError("TODO 2: JumpReLU backward with the straight-through estimator")


def fvu(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Fraction of variance unexplained: Σ‖x − x̂‖² / Σ‖x − mean(x)‖², rows are tokens (N, d)."""
    raise NotImplementedError("TODO 3: fraction of variance unexplained")


def splice_edit(sae, keep_error: bool = False):
    """An edit function for ``frontierlab.interp.hooks.run_with``: given the activation x (B, T, d) return the
    SAE reconstruction ``sae(x)``; with ``keep_error=True`` return ``sae(x) + (x − sae(x))``, which must give
    back exactly the model's own loss (the splice check)."""
    raise NotImplementedError("TODO 4: the splice edit")
