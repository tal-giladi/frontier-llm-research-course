"""Lab 15.2 — Speculative decoding. Fill in the TODOs; run `pytest labs/module-15/lesson-02` to check.

``spec_lab.py`` plugs your ``accept_reject`` into the course's generation loop (``frontierlab.ttc.speculative``)
and checks that greedy speculative decoding reproduces plain greedy decoding token for token before it
measures anything.
"""

from __future__ import annotations

import torch


def residual(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """norm(max(0, p - q)) along the last axis. Where p - q has no positive part (p == q), return p."""
    raise NotImplementedError("TODO 1: the residual distribution")


def accept_reject(p: torch.Tensor, q: torch.Tensor, drafts: torch.Tensor, u: torch.Tensor,
                  generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """One verification round for a batch.

    p (B, g+1, V) target distributions at the g draft positions and one bonus position; q (B, g, V) draft
    distributions; drafts (B, g) int64; u (B, g) uniforms. Draft i is accepted when u[:, i] < min(1, p/q) at
    its token AND every earlier draft was accepted. Return (n (B,) = number of accepted drafts, next (B,)):
    ``next`` is sampled (torch.multinomial with ``generator``) from residual(p[:, n], q[:, n]) if n < g, or from
    p[:, g] if every draft was accepted."""
    raise NotImplementedError("TODO 2: accept / reject")


def expected_tokens_per_round(alpha: float, gamma: int) -> float:
    """Tokens emitted per round when each draft is accepted independently with probability alpha:
    1 + alpha + ... + alpha^gamma = (1 - alpha^(gamma+1)) / (1 - alpha), and gamma + 1 at alpha = 1."""
    raise NotImplementedError("TODO 3: expected tokens per round")


def walltime_improvement(alpha: float, gamma: int, c: float) -> float:
    """Expected speed-up over plain decoding when one draft step costs c target steps (Leviathan et al.,
    Theorem 3.8): expected tokens per round divided by (gamma * c + 1)."""
    raise NotImplementedError("TODO 4: walltime improvement")
