"""Lab 06.7 — non-autoregressive and latent reasoning (extension). Fill in the TODOs; run `pytest labs/module-06/lesson-07`."""

from __future__ import annotations

import math  # noqa: F401

import torch
import torch.nn.functional as F  # noqa: F401


def masked_terms(logits: torch.Tensor, x0: torch.Tensor, masked: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """LLaDA's training loss (Eq. 3) per position, divided by the sequence length L so it reads in nats per token:

        term[b, i] = (1 / t[b]) * 1[masked[b, i]] * CE(logits[b, i], x0[b, i]) / L

    logits (B, L, V), x0 (B, L) long, masked (B, L) bool, t (B,) mask rates. Return (B, L)."""
    raise NotImplementedError("TODO 1: the masked-diffusion loss terms")


def eq6_estimate(ce: torch.Tensor, masked: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    """LLaDA's lower-variance likelihood estimate (Eq. 6) for one draw per sequence: l[b] positions out of L are
    masked uniformly at random; the estimate is (L / l) * (sum of CE over the masked positions), divided by L.
    ce (B, L) per-position cross-entropy, masked (B, L) bool, l (B,) long. Return (B,)."""
    raise NotImplementedError("TODO 2: the Eq. 6 estimator")


def commit_count(remaining: int, steps_left: int) -> int:
    """Low-confidence remasking (LLaDA section 2.4) unmasks the most confident predictions each step. How many
    should it commit now so that no mask is left after ``steps_left`` more steps, spreading them evenly?"""
    raise NotImplementedError("TODO 3: tokens to commit per step")


def effective_depth(prelude: int, core: int, coda: int, r: int) -> int:
    """Layers a token passes through in a recurrent-depth model (Geiping et al. section 3) with r core iterations."""
    raise NotImplementedError("TODO 4: effective depth")
