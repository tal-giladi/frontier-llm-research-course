"""Lab 02.2 — profiling a training step. Fill in the TODOs; run `pytest labs/module-02/lesson-02` to check."""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint  # noqa: F401  (TODO 2)


def ce_and_grads(h: torch.Tensor, w: torch.Tensor, targets: torch.Tensor, chunk_size: int):
    """Mean cross-entropy of ``h @ w.T`` and its gradients, ``chunk_size`` rows at a time.

    h (N, C), w (V, C), targets (N,) long. Return ``(loss, dL/dh, dL/dw)`` with the same shapes as
    ``()``, h and w. Never build the full (N, V) logits: for each chunk compute its logits, add its
    summed loss, form dL/dz = (softmax(z) - onehot(targets)) / N for those rows, and accumulate
    dL/dh for the chunk and dL/dw. Do not call ``.backward()`` or ``torch.autograd``.
    """
    raise NotImplementedError("TODO 1: chunked cross-entropy with hand-derived gradients")


def checkpointed_loss(model, idx: torch.Tensor) -> torch.Tensor:
    """The same loss as ``model(idx, labels=idx).loss``, with every transformer block checkpointed.

    Mirror ``LM.forward`` (embed, the blocks with positions 0..T-1 and no cache, final norm, output
    head, fp32 logits, cross-entropy of positions 0..T-2 against tokens 1..T-1), but call each block
    through ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`` so its internal
    activations are recomputed in backward instead of being saved.
    """
    raise NotImplementedError("TODO 2: activation checkpointing per block")


CATEGORIES = {
    "matmul": ("mm", "addmm", "bmm", "matmul", "linear", "gemm"),
    "attention": ("attention", "sdpa", "flash"),
    "loss": ("log_softmax", "nll_loss", "cross_entropy"),
    "optimizer": ("adam", "_foreach", "lerp", "addcdiv", "addcmul"),
}


def categorize(top: list[tuple[str, float, int, float]]) -> dict[str, float]:
    """Sum the time shares of profiler rows ``(name, ms, calls, share)`` per category.

    A row belongs to the first category in ``CATEGORIES`` (in dict order) whose keys appear in its
    lower-cased name; rows matching none go to ``"other"``. Return a dict with every category in
    ``CATEGORIES`` plus ``"other"`` (0.0 where nothing matched).
    """
    raise NotImplementedError("TODO 3: where did the time go, by category")
