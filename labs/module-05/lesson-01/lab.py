"""Lab 05.1 — the gated delta rule, recurrent and chunked, and a hybrid layout.

Fill in the TODOs; run `pytest labs/module-05/lesson-01`. Shapes follow frontierlab.attention.deltanet:
the state S of one head is (d_k, d_v); a decode step is o = S^T q after the update.
"""

from __future__ import annotations

import torch

from frontierlab.attention.base import register
from frontierlab.attention.deltanet import KimiDeltaAttention


def recurrent_step(S, q, k, v, log_alpha, beta):
    """One token of S_t = (I - beta k k^T) Diag(alpha) S_{t-1} + beta k v^T, then o = S_t^T q.

    S (B, H, d_k, d_v); q, k (B, H, d_k); v (B, H, d_v); log_alpha (B, H, d_k), alpha = exp(log_alpha);
    beta (B, H). Returns (o (B, H, d_v), new S). Do not modify S in place.
    """
    raise NotImplementedError("TODO 1: one step of the gated delta rule")


def chunk_forward(q, k, v, g, beta, S0):
    """One chunk of C tokens in parallel. q, k (B, H, C, d_k); v (B, H, C, d_v); g (B, H, C, d_k) log-decay;
    beta (B, H, C); S0 (B, H, d_k, d_v). Returns (O (B, H, C, d_v), S_C).

    With b_r = cumsum(g)_r (per channel) and Gamma_r = exp(b_r):
      D[r, i]  = exp(b_r - b_i) for i <= r, 0 above the diagonal        (B, H, C, C, d_k)
      A[r, i]  = beta_r * sum_c k_r,c D[r,i,c] k_i,c   for i < r, else 0
      M[r, i]  = sum_c q_r,c D[r,i,c] k_i,c            for i <= r, else 0
      U = (I + A)^-1 (beta * V),   W = (I + A)^-1 (beta * (Gamma * K)),   E = U - W S0
      O = (Gamma * Q) S0 + M E
      S_C = Gamma_C * S0 + Khat^T E,   Khat rows = exp(b_C - b_i) * k_i
    Use torch.linalg.solve_triangular(L, B, upper=False, unitriangular=True) for (I + A)^-1 B, and build D by
    masking the exponent with -inf above the diagonal BEFORE calling exp (exp of a large positive number
    overflows).
    """
    raise NotImplementedError("TODO 2: the chunk in parallel (WY form)")


def delta_chunked(q, k, v, g, beta, state=None, chunk: int = 16):
    """Whole sequence: chunk_forward on consecutive chunks, carrying the state. (Provided.)"""
    B, H, T, dk = k.shape
    S = torch.zeros(B, H, dk, v.shape[-1], dtype=v.dtype, device=v.device) if state is None else state.to(v.dtype)
    outs = []
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        O, S = chunk_forward(q[:, :, s:e], k[:, :, s:e], v[:, :, s:e], g[:, :, s:e], beta[:, :, s:e], S)
        outs.append(O)
    return torch.cat(outs, dim=2), S


def delta_recurrent(q, k, v, g, beta, state=None):
    """Whole sequence with recurrent_step. (Provided.)"""
    B, H, T, dk = k.shape
    S = torch.zeros(B, H, dk, v.shape[-1], dtype=v.dtype, device=v.device) if state is None else state.to(v.dtype)
    outs = []
    for t in range(T):
        o, S = recurrent_step(S, q[:, :, t], k[:, :, t], v[:, :, t], g[:, :, t], beta[:, :, t])
        outs.append(o)
    return torch.stack(outs, dim=2), S


def hybrid_pattern(num_layers: int, full_every: int) -> list[str]:
    """"linear" / "full" per layer: layer i is full when (i + 1) % full_every == 0 (Qwen3-Next: 4)."""
    raise NotImplementedError("TODO 3: hybrid layer pattern")


def cache_bytes(pattern: list[str], S: int, kv_bytes_per_token_layer: float, state_bytes_per_layer: float) -> float:
    """Decode memory of one sequence after S tokens: full layers grow with S, linear layers keep a fixed state."""
    raise NotImplementedError("TODO 4: cache bytes of a hybrid model")


@register("kda-lab")
class LabKDA(KimiDeltaAttention):
    """frontierlab's KDA layer with the delta-rule core replaced by yours."""

    def core(self, q, k, v, g, beta, state, mode=None):
        wd = torch.promote_types(v.dtype, torch.float32)
        args = [t.to(wd) for t in (q, k, v, g, beta)]
        if (mode or self.mode) == "recurrent":
            return delta_recurrent(*args, state)
        return delta_chunked(*args, state, self.chunk)
