"""Lab 17.2 — causal interventions. Fill in the TODOs; run `pytest labs/module-17/lesson-02`.

``patch_lab.py`` checks your functions against the course's (``frontierlab.interp.patching``), recovers the
known induction circuit with them, then runs the same pipeline with controls on Qwen3-0.6B (``--hf``).
"""

from __future__ import annotations

import torch


def normalised_effect(m_patched: torch.Tensor, m_clean: torch.Tensor, m_corrupt: torch.Tensor, mode: str) -> torch.Tensor:
    """Per-prompt normalised effect, dividing by the gap between the *mean* clean and the *mean* corrupt metric:
    denoising ("denoise"): (m_patched − m_corrupt) / gap;  noising ("noise"): (m_clean − m_patched) / gap."""
    raise NotImplementedError("TODO 1: normalised patching effect")


def head_patch_edit(source_z: torch.Tensor, head: int, head_dim: int):
    """An edit for a ``z.L`` site (B, T, H·head_dim): copy head ``head``'s slice [head·d, (head+1)·d) from
    ``source_z`` at every position; leave every other head untouched. Must not modify its input in place."""
    raise NotImplementedError("TODO 2: patch one head")


def head_delta(z_from: torch.Tensor, z_to: torch.Tensor, W_O: torch.Tensor, heads: list[int], head_dim: int) -> torch.Tensor:
    """What layer L's attention output would change by if only ``heads`` switched from ``z_from`` to ``z_to``:
    Σ_h (z_to − z_from)[..., h-slice] @ W_O[:, h-slice]ᵀ. z (B, T, H·d), W_O (C, H·d) -> (B, T, C).
    This is the sender's contribution that path patching adds to one receiver's input only."""
    raise NotImplementedError("TODO 3: the sender's direct contribution")


def p_random(effect: float, random_effects) -> float:
    """Rank p-value of a candidate among random controls: (1 + #{r ≥ effect}) / (1 + n)."""
    raise NotImplementedError("TODO 4: the control p-value")
