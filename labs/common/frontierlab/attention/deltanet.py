"""Gated delta-rule linear attention: Gated DeltaNet (scalar gate) and Kimi Delta Attention (channel-wise gate).

Lesson 05.1. Two kinds, registered on import:

* ``"gdn"`` — Gated DeltaNet (Yang, Kautz, Hatamizadeh, arXiv 2412.06464): one data-dependent decay
  ``alpha_t`` in (0, 1) per head and token. The linear layers of Qwen3-Next are of this family.
* ``"kda"`` — Kimi Delta Attention (Kimi Linear, arXiv 2510.26692): one decay per head, token *and
  key channel*, ``alpha_t`` in [0, 1]^{d_k} ("each feature dimension maintains an independent
  forgetting rate").

The recurrence, per head, with the state S_t of shape (d_k, d_v) (the convention of the Kimi Linear
paper and of flash-linear-attention's ``initial_state`` [N, H, K, V]):

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T          o_t = S_t^T q_t

Read it as one step of online regression: Diag(alpha_t) forgets, ``S^T k_t`` is what the memory
currently predicts for key k_t, and the update moves that prediction a fraction beta_t of the way to
v_t (the delta rule). With alpha = 1 and beta = 0 nothing changes; with alpha = 1, no delta term and
no normalisation it is plain linear attention (parent course lesson 12.5). For Gated DeltaNet
Diag(alpha_t) = alpha_t I, so the same code serves both kinds.

Two exact implementations of the same function:

* :func:`delta_rule_recurrent` — the recurrence above, one token at a time. O(T d_k d_v) work, O(1)
  memory in T. This is what decoding does.
* :func:`delta_rule_chunked` — chunked-parallel form. Inside a chunk of C tokens the C updates are
  solved together with one unit-lower-triangular system (the WY / UT-transform idea of DeltaNet,
  Gated DeltaNet and KDA); across chunks the state is carried recurrently. Derivation in the module
  docstring of lesson 05.1 and in comments below. Every step is a matrix product or a triangular
  solve, which is what makes training parallel. Our version forms the pairwise decay tensor
  exp(b_r - b_i) of shape (C, C, d_k) exactly (stable for any decay, memory C^2 d_k per head and
  chunk); flash-linear-attention's ``chunk_kda`` uses C = 64 with a factorised, safeguarded form.

The two must agree to float64 rounding (``labs/common/tests/test_attention_m05.py``). That test is the
"recurrent vs chunked-parallel equivalence" check of the course correctness suite.

Layer parameterisation (both kinds; PUBLICLY DOCUMENTED in outline by both papers, details are this
implementation's choices where noted):

    q, k, v = W_q x, W_k x, W_v x  -> causal depthwise short convolution (kernel 4) -> SiLU
    q, k    -> L2 normalised per head; q also scaled by d_k^-1/2
    beta    = sigmoid(W_b x)                                   (B, H, T)
    log alpha = -exp(A_log) * softplus(W_a x + dt_bias)        (B, H, T)       gdn
    log alpha = -exp(A_log) * softplus(W_up W_down x + dt_bias)  (B, H, T, d_k)  kda (low-rank)
    o       = W_o [ RMSNorm(o_heads) * act(W_g x) ]   act = SiLU for gdn, sigmoid for kda (our choice)

The ``-exp(A_log) * softplus(. + dt_bias)`` decay is the form flash-linear-attention 0.5.2 computes
when ``use_gate_in_kernel=True`` (docstring of ``fla.ops.kda.chunk_kda``); A_log and dt_bias are
initialised as in Mamba-2 (A in [1, 16], softplus(dt_bias) in [0.001, 0.1]). There is no RoPE: order
comes from the decay and the short convolution, so ``positions`` is ignored.

The LayerCache holds the recurrent state, not keys and values:

    cache["state"]  (B, H, d_k, d_v)  in at least float32 (kept in fp32 under bf16 autocast)
    cache["conv"]   (B, K - 1, H (2 d_k + d_v))  the last K - 1 pre-convolution projections

Its size does not grow with the context. ``cfg.extra`` settings: ``linear_heads`` [H],
``linear_head_dim`` [head_dim], ``conv_size`` [4; 0 or 1 disables it], ``chunk`` [16],
``gate_rank`` [max(8, d_k // 2)] (kda only), ``linear_mode`` ["chunked" | "recurrent" | "fla"],
``out_gate`` ["silu" for gdn, "sigmoid" for kda] (the output-gate activation, so the two kinds can be
compared with only the decay granularity changed).

``linear_mode="fla"`` calls flash-linear-attention 0.5.2 (``chunk_gated_delta_rule`` /
``chunk_kda``, layout [B, T, H, D]) on CUDA. NOT RUN IN THIS BUILD (the package is not installed
here; it needs Triton and a GPU); part of the Module 5 pilot, which must first pass
``test_fla_matches_reference`` in the pilot notebook.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from frontierlab.attention.base import LayerCache, register
from frontierlab.layers.rmsnorm import RMSNorm


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Divide each vector (last axis) by its Euclidean norm."""
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def _work_dtype(t: torch.Tensor) -> torch.dtype:
    return torch.promote_types(t.dtype, torch.float32)


def delta_rule_recurrent(q, k, v, g, beta, state=None):
    """The gated delta rule, token by token.

    q, k   (B, H, T, d_k)   q already scaled; k L2-normalised (not required for correctness)
    v      (B, H, T, d_v)
    g      (B, H, T, d_k)   log alpha (<= 0), one per key channel (expand a scalar gate to d_k)
    beta   (B, H, T)
    state  (B, H, d_k, d_v) or None (zeros)
    returns o (B, H, T, d_v) in v's dtype, final state in at least float32
    """
    wd = _work_dtype(v)
    q, k, v, g, beta = (t.to(wd) for t in (q, k, v, g, beta))
    B, H, T, dk = k.shape
    S = torch.zeros(B, H, dk, v.shape[-1], dtype=wd, device=v.device) if state is None else state.to(wd)
    outs = []
    for t in range(T):
        S = S * g[:, :, t].exp().unsqueeze(-1)                         # Diag(alpha_t) S
        kt = k[:, :, t]
        pred = torch.einsum("bhk,bhkv->bhv", kt, S)                    # what the memory predicts for k_t
        S = S + torch.einsum("bhk,bhv->bhkv", kt, beta[:, :, t, None] * (v[:, :, t] - pred))
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], S))      # o_t = S_t^T q_t
    return torch.stack(outs, dim=2), S


def delta_rule_chunked(q, k, v, g, beta, state=None, chunk: int = 16):
    """The same function in chunked-parallel form (same arguments; exact, not an approximation).

    Inside a chunk with positions r = 1..C and state S_0 at its start, let b_r = sum_{i<=r} g_i (per
    channel) and Gamma_r = exp(b_r). Writing e_r = beta_r (v_r - (Diag(alpha_r) S_{r-1})^T k_r), the
    recurrence unrolls to

        S_r = Diag(Gamma_r) S_0 + sum_{i<=r} Diag(Gamma_r / Gamma_i) k_i e_i^T

    and substituting it into e_r gives a triangular system for all C rows E = [e_1..e_C] at once:

        (I + A) E = Diag(beta) (V - K~ S_0),   A_ri = beta_r k_r^T Diag(Gamma_r/Gamma_i) k_i  (i < r)

    with K~ the rows Gamma_r * k_r. So E = U - W S_0, U = (I+A)^-1 Diag(beta) V, W = (I+A)^-1 Diag(beta) K~
    (the "WY" form). Then

        O   = Q~ S_0 + (M o E),   M_ri = q_r^T Diag(Gamma_r/Gamma_i) k_i  (i <= r),  Q~ rows Gamma_r * q_r
        S_C = Diag(Gamma_C) S_0 + K^T E,   K^ rows (Gamma_C / Gamma_i) * k_i
    """
    wd = _work_dtype(v)
    q, k, v, g, beta = (t.to(wd) for t in (q, k, v, g, beta))
    B, H, T, dk = k.shape
    dv = v.shape[-1]
    S = torch.zeros(B, H, dk, dv, dtype=wd, device=v.device) if state is None else state.to(wd)
    outs = []
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        C = e - s
        qc, kc, vc, gc, bc = q[:, :, s:e], k[:, :, s:e], v[:, :, s:e], g[:, :, s:e], beta[:, :, s:e]
        b = gc.cumsum(dim=2)                                            # (B, H, C, d_k), log Gamma_r
        incl = torch.ones(C, C, dtype=torch.bool, device=v.device).tril()
        strict = incl.clone().fill_diagonal_(False)
        diff = b[:, :, :, None, :] - b[:, :, None, :, :]                # (B, H, C, C, d_k): b_r - b_i
        D = diff.masked_fill(~incl[:, :, None], float("-inf")).exp()   # exp(b_r - b_i) for i <= r, else 0
        A = torch.einsum("bhrc,bhric,bhic->bhri", kc, D, kc) * bc[..., None]
        A = A.masked_fill(~strict, 0.0)
        M = torch.einsum("bhrc,bhric,bhic->bhri", qc, D, kc)           # diagonal: q_r . k_r (D = 1)
        Gam = b.exp()
        L = torch.eye(C, dtype=wd, device=v.device) + A
        U = torch.linalg.solve_triangular(L, bc[..., None] * vc, upper=False, unitriangular=True)
        W = torch.linalg.solve_triangular(L, bc[..., None] * (Gam * kc), upper=False, unitriangular=True)
        E = U - W @ S                                                   # (B, H, C, d_v)
        outs.append((Gam * qc) @ S + M @ E)
        bC = b[:, :, -1]                                                # (B, H, d_k)
        Khat = kc * (bC[:, :, None, :] - b).exp()
        S = S * bC.exp().unsqueeze(-1) + Khat.transpose(-1, -2) @ E
    return torch.cat(outs, dim=2), S


def inv_softplus(y: torch.Tensor) -> torch.Tensor:
    return y + torch.log(-torch.expm1(-y))


class GatedDeltaAttention(nn.Module):
    """Shared layer for ``gdn`` (gate="scalar") and ``kda`` (gate="channel"). See the module docstring."""

    gate_kind = "channel"

    def __init__(self, cfg):
        super().__init__()
        ex = cfg.extra
        C = cfg.hidden_size
        self.H = int(ex.get("linear_heads", cfg.num_attention_heads))
        self.dk = self.dv = int(ex.get("linear_head_dim", cfg.head_dim))
        self.K = int(ex.get("conv_size", 4))
        self.chunk = int(ex.get("chunk", 16))
        self.mode = ex.get("linear_mode", "chunked")
        self.out_gate = ex.get("out_gate", "silu" if self.gate_kind == "scalar" else "sigmoid")
        H, dk, dv = self.H, self.dk, self.dv
        self.q_proj = nn.Linear(C, H * dk, bias=False)
        self.k_proj = nn.Linear(C, H * dk, bias=False)
        self.v_proj = nn.Linear(C, H * dv, bias=False)
        width = H * (2 * dk + dv)
        if self.K > 1:
            bound = 1.0 / math.sqrt(self.K)
            self.conv_weight = nn.Parameter(torch.empty(width, self.K).uniform_(-bound, bound))
        else:
            self.conv_weight = None
        self.b_proj = nn.Linear(C, H, bias=False)
        if self.gate_kind == "scalar":
            self.a_proj = nn.Linear(C, H, bias=False)
            n_dt = H
        else:
            r = int(ex.get("gate_rank", max(8, dk // 2)))
            self.f_down = nn.Linear(C, r, bias=False)
            self.f_up = nn.Linear(r, H * dk, bias=False)
            n_dt = H * dk
        self.A_log = nn.Parameter(torch.empty(H).uniform_(1, 16).log())
        dt = torch.exp(torch.empty(n_dt).uniform_(math.log(1e-3), math.log(1e-1)))
        self.dt_bias = nn.Parameter(inv_softplus(dt))
        self.g_proj = nn.Linear(C, H * dv, bias=False)
        self.o_norm = RMSNorm(dv, eps=cfg.rms_norm_eps)
        self.o_proj = nn.Linear(H * dv, C, bias=False)

    # --- pieces --------------------------------------------------------------------------------
    def short_conv(self, z: torch.Tensor, cache: LayerCache | None):
        """Causal depthwise convolution over time, z (B, T, W). Keeps the last K-1 inputs in the cache."""
        if self.conv_weight is None:
            return z
        B, T, Wd = z.shape
        prev = cache.get("conv") if cache is not None else None
        if prev is None:
            prev = z.new_zeros(B, self.K - 1, Wd)
        ext = torch.cat((prev.to(z.dtype), z), dim=1)                  # (B, K-1+T, W)
        w = self.conv_weight.to(z.dtype)
        out = sum(ext[:, j:j + T] * w[:, j] for j in range(self.K))
        if cache is not None:
            cache["conv"] = ext[:, -(self.K - 1):]
        return out

    def decay(self, x: torch.Tensor) -> torch.Tensor:
        """log alpha, (B, H, T, d_k) (a scalar gate is expanded over d_k)."""
        B, T, _ = x.shape
        A = self.A_log.float().exp() if x.dtype in (torch.float16, torch.bfloat16) else self.A_log.exp()
        if self.gate_kind == "scalar":
            g = -A * F.softplus(self.a_proj(x) + self.dt_bias)            # (B, T, H)
            return g.transpose(1, 2).unsqueeze(-1).expand(B, self.H, T, self.dk)
        raw = self.f_up(self.f_down(x)) + self.dt_bias                  # (B, T, H*d_k)
        g = -A.repeat_interleave(self.dk) * F.softplus(raw)
        return g.view(B, T, self.H, self.dk).transpose(1, 2)

    def mix(self, x: torch.Tensor, cache: LayerCache | None = None):
        """q, k, v (B, H, T, d), log-decay g (B, H, T, d_k) and beta (B, H, T)."""
        B, T, _ = x.shape
        H, dk, dv = self.H, self.dk, self.dv
        z = torch.cat((self.q_proj(x), self.k_proj(x), self.v_proj(x)), dim=-1)
        z = F.silu(self.short_conv(z, cache))
        q, k, v = z.split((H * dk, H * dk, H * dv), dim=-1)
        q = l2norm(q.view(B, T, H, dk)).transpose(1, 2) * dk ** -0.5
        k = l2norm(k.view(B, T, H, dk)).transpose(1, 2)
        v = v.view(B, T, H, dv).transpose(1, 2)
        beta = torch.sigmoid(self.b_proj(x)).transpose(1, 2)
        return q, k, v, self.decay(x), beta

    def core(self, q, k, v, g, beta, state, mode=None):
        mode = mode or self.mode
        if mode == "recurrent":
            return delta_rule_recurrent(q, k, v, g, beta, state)
        if mode == "chunked":
            return delta_rule_chunked(q, k, v, g, beta, state, self.chunk)
        if mode == "fla":
            return self._fla(q, k, v, g, beta, state)
        raise ValueError(f"unknown linear_mode {mode!r}")

    def _fla(self, q, k, v, g, beta, state):
        """flash-linear-attention 0.5.2 path, CUDA only. NOT RUN IN THIS BUILD (Module 5 pilot)."""
        if not q.is_cuda:
            raise RuntimeError("linear_mode='fla' needs CUDA tensors (Triton kernels)")
        tr = (lambda t: t.transpose(1, 2).contiguous())                 # (B, H, T, .) -> (B, T, H, .)
        init = None if state is None else state.float().contiguous()
        if self.gate_kind == "scalar":
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            o, S = chunk_gated_delta_rule(tr(q), tr(k), tr(v), tr(g[..., 0].float()), tr(beta), scale=1.0,
                                          initial_state=init, output_final_state=True)
        else:
            from fla.ops.kda import chunk_kda
            o, S = chunk_kda(tr(q), tr(k), tr(v), tr(g.float()), tr(beta), scale=1.0,
                             initial_state=init, output_final_state=True)
        return o.transpose(1, 2), S

    def forward(self, x: torch.Tensor, positions: torch.Tensor, cache: LayerCache | None = None):
        B, T, _ = x.shape
        q, k, v, g, beta = self.mix(x, cache)
        state = cache.get("state") if cache is not None else None
        o, S = self.core(q, k, v, g, beta, state)
        if cache is not None:
            cache["state"] = S
        o = self.o_norm(o.to(x.dtype).transpose(1, 2))                  # (B, T, H, d_v)
        gate = self.g_proj(x).view(B, T, self.H, self.dv)
        gate = F.silu(gate) if self.out_gate == "silu" else torch.sigmoid(gate)
        return self.o_proj((o * gate).reshape(B, T, self.H * self.dv))


@register("gdn")
class GatedDeltaNet(GatedDeltaAttention):
    gate_kind = "scalar"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.layer_type = "linear"


@register("kda")
class KimiDeltaAttention(GatedDeltaAttention):
    gate_kind = "channel"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.layer_type = "linear"


def set_linear_mode(model, mode: str) -> int:
    """Switch every gated-delta layer of ``model`` to "chunked", "recurrent" or "fla"; returns how many."""
    n = 0
    for m in model.modules():
        if isinstance(m, GatedDeltaAttention):
            m.mode = mode
            n += 1
    return n
