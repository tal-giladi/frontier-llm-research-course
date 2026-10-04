"""Lab 07.2 — Muon at scale: update-RMS matching and QK-Clip. Fill in the TODOs; run `pytest labs/module-07/lesson-02`.

Shapes: q (B, H, T, d) and k (B, H, T, d) after projection, norm and RoPE (keys already repeated to H heads
for GQA); weights as nn.Linear stores them, (out_features, in_features).
"""

from __future__ import annotations

import torch


def orthogonal_update_rms(shape) -> float:
    """RMS of the entries of a full-rank matrix of ``shape`` whose singular values are all 1 (Moonlight Lemma 1)."""
    raise NotImplementedError("TODO 1: RMS of an orthogonal update")


def rms_matched_scale(shape, target: float = 0.2) -> float:
    """Factor that brings that RMS to ``target`` (Moonlight Eq. 4 uses 0.2 · sqrt(max(A, B)))."""
    raise NotImplementedError("TODO 2: the update-RMS matching factor")


def head_max_logits(q: torch.Tensor, k: torch.Tensor, scale: float, mask: torch.Tensor) -> torch.Tensor:
    """S_max per head: max over batch and allowed (i, j) of scale · q_i · k_j. mask (T, T) bool, True = allowed. -> (H,)"""
    raise NotImplementedError("TODO 3: per-head maximum logit")


def qk_clip_gamma(smax: torch.Tensor, tau: float) -> torch.Tensor:
    """gamma_h = min(1, tau / S_max^h) (Kimi K2 section 2.1)."""
    raise NotImplementedError("TODO 4: the per-head clip factor")


def clip_gqa_weights(wq: torch.Tensor, wk: torch.Tensor, gamma: torch.Tensor, H: int, KV: int, d: int) -> None:
    """In place. Rows h*d:(h+1)*d of wq belong to query head h; rows g*d:(g+1)*d of wk to key head g.

    MHA (KV == H): scale head h's query rows and key rows by sqrt(gamma_h) each.
    GQA (KV < H): scale head h's query rows by gamma_h and leave the shared key rows alone.
    Skip heads with gamma_h >= 1.
    """
    raise NotImplementedError("TODO 5: QK-Clip for GQA/MHA weights")


def clip_mla_weights(wq: torch.Tensor, wkv: torch.Tensor, gamma: torch.Tensor, H: int, d_n: int, d_r: int, d_v: int) -> None:
    """In place, for MLA. wq (H*(d_n+d_r), ·): per head [q^C (d_n rows); q^R (d_r rows)].
    wkv (H*(d_n+d_v), d_c): per head [W_UK (d_n rows, makes k^C); W_UV (d_v rows)].

    Kimi K2: q^C and k^C by sqrt(gamma_h), q^R by gamma_h, shared k^R untouched, W_UV untouched.
    """
    raise NotImplementedError("TODO 6: QK-Clip for MLA weights")
