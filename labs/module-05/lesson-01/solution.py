"""Reference solution for lab 05.1."""

from __future__ import annotations

import torch

from frontierlab.attention.base import register
from frontierlab.attention.deltanet import KimiDeltaAttention


def recurrent_step(S, q, k, v, log_alpha, beta):
    S = S * log_alpha.exp().unsqueeze(-1)                              # Diag(alpha) S: decay each key channel
    pred = torch.einsum("bhk,bhkv->bhv", k, S)                         # what the memory returns for k
    S = S + torch.einsum("bhk,bhv->bhkv", k, beta.unsqueeze(-1) * (v - pred))
    return torch.einsum("bhk,bhkv->bhv", q, S), S


def chunk_forward(q, k, v, g, beta, S0):
    C = k.shape[2]
    b = g.cumsum(dim=2)
    incl = torch.ones(C, C, dtype=torch.bool, device=v.device).tril()
    strict = incl.clone().fill_diagonal_(False)
    D = (b[:, :, :, None, :] - b[:, :, None, :, :]).masked_fill(~incl[:, :, None], float("-inf")).exp()
    A = (torch.einsum("bhrc,bhric,bhic->bhri", k, D, k) * beta[..., None]).masked_fill(~strict, 0.0)
    M = torch.einsum("bhrc,bhric,bhic->bhri", q, D, k)
    Gam = b.exp()
    L = torch.eye(C, dtype=v.dtype, device=v.device) + A
    U = torch.linalg.solve_triangular(L, beta[..., None] * v, upper=False, unitriangular=True)
    W = torch.linalg.solve_triangular(L, beta[..., None] * (Gam * k), upper=False, unitriangular=True)
    E = U - W @ S0
    O = (Gam * q) @ S0 + M @ E
    bC = b[:, :, -1]
    Khat = k * (bC[:, :, None, :] - b).exp()
    return O, S0 * bC.exp().unsqueeze(-1) + Khat.transpose(-1, -2) @ E


def delta_chunked(q, k, v, g, beta, state=None, chunk: int = 16):
    B, H, T, dk = k.shape
    S = torch.zeros(B, H, dk, v.shape[-1], dtype=v.dtype, device=v.device) if state is None else state.to(v.dtype)
    outs = []
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        O, S = chunk_forward(q[:, :, s:e], k[:, :, s:e], v[:, :, s:e], g[:, :, s:e], beta[:, :, s:e], S)
        outs.append(O)
    return torch.cat(outs, dim=2), S


def delta_recurrent(q, k, v, g, beta, state=None):
    B, H, T, dk = k.shape
    S = torch.zeros(B, H, dk, v.shape[-1], dtype=v.dtype, device=v.device) if state is None else state.to(v.dtype)
    outs = []
    for t in range(T):
        o, S = recurrent_step(S, q[:, :, t], k[:, :, t], v[:, :, t], g[:, :, t], beta[:, :, t])
        outs.append(o)
    return torch.stack(outs, dim=2), S


def hybrid_pattern(num_layers: int, full_every: int) -> list[str]:
    return ["full" if (i + 1) % full_every == 0 else "linear" for i in range(num_layers)]


def cache_bytes(pattern, S, kv_bytes_per_token_layer, state_bytes_per_layer):
    return sum(S * kv_bytes_per_token_layer if p == "full" else state_bytes_per_layer for p in pattern)


@register("kda-lab-solution")
class LabKDA(KimiDeltaAttention):
    def core(self, q, k, v, g, beta, state, mode=None):
        wd = torch.promote_types(v.dtype, torch.float32)
        args = [t.to(wd) for t in (q, k, v, g, beta)]
        if (mode or self.mode) == "recurrent":
            return delta_recurrent(*args, state)
        return delta_chunked(*args, state, self.chunk)
