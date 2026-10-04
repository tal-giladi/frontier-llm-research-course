"""Residual-stream designs: hyper-connections (HC) and manifold-constrained hyper-connections (mHC), lesson 06.2.

A pre-norm block computes ``x <- x + F(x)`` for each sub-layer F (attention, then the FFN). The residual
stream is one C-wide vector per token and the identity path (the bare ``x``) carries the signal from any
layer to any deeper one unchanged.

**Hyper-connections** (Zhu et al., arXiv 2409.19606, section 2) widen the stream to n vectors per token,
X in R^{n x C}, and give every sub-layer three small learnable mappings (mHC paper notation, Eq. 3):

    x_{l+1} = H_res x_l + H_post^T F(H_pre x_l)        H_pre (1 x n) reads, H_post (1 x n) writes,
                                                       H_res (n x n) mixes the streams

Each mapping is static (a learned bias) plus dynamic (a function of the current stream). The embedding is
copied into all n streams at the bottom; the streams are summed at the top before the final norm
(HC section 2.1). Parameters per sub-layer are O(n·C + n²); FLOPs are small; what HC adds is *memory
traffic*: every residual read and write is n times wider (mHC Table 2: (5n+1)C + n² + 2n reads per token
per sub-layer instead of 2C). That is why the cost axis of lesson 06.2 is wall-clock, not FLOPs.

Two implementations:

* :class:`HyperConnection` — Zhu et al.'s dynamic HC (DHC), Eqs. 10–13, with the paper's initialisation
  (Eq. 14): dynamic weights 0, B = 1 (write to every stream), A_m = e_{k mod n} (sub-layer k reads stream
  k mod n), A_r = I. With n = 1 at initialisation it *is* the plain residual (tested exactly). Nothing
  constrains A_r: products of A_r over depth can grow or shrink without bound.
* :class:`ManifoldHC` — mHC (Xie et al., arXiv 2512.24880, section 4.2, Eqs. 7–9): coefficients from the
  RMS-normalised, flattened stream (nC values); H_pre = sigmoid(.), H_post = 2·sigmoid(.), and H_res =
  Sinkhorn-Knopp(.) — a doubly-stochastic matrix (non-negative, rows and columns sum to 1), so H_res x is a
  convex combination of the streams, its spectral norm is at most 1, and a product of such matrices is
  again doubly stochastic. ``t_max = 20`` iterations as in the paper (section 4.2, Table 5).

Course choices (the paper does not give these values; stated in the lesson): mHC static biases at
initialisation b_res = 4·I (H_res ≈ 0.95 on the diagonal for n = 4), b_pre = +4 on stream k mod n and −4
elsewhere (reads mostly one stream, like HC's e_{k mod n}), b_post = 0 (H_post = 1); gating factors
alpha = 0.01 (mHC Table 5) for mHC and for HC's s_alpha, s_beta; the dynamic projections are N(0, 0.02²)
for mHC and zero for HC (HC section 2.3).

Shapes: the stream is (B, T, n, C); a sub-layer's input and output are (B, T, C). ``last_res`` keeps the
(B, T, n, n) residual matrix of the latest forward (in the ``x_{l+1} = H_res x_l`` orientation) for the
Amax-gain diagnostic of :func:`amax_gain`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from frontierlab.layers.rmsnorm import RMSNorm


def sinkhorn_knopp(logits: torch.Tensor, t_max: int = 20) -> torch.Tensor:
    """Project ``logits`` (..., n, n) onto (approximately) doubly-stochastic matrices (mHC Eq. 9).

    M0 = exp(logits) (the max is subtracted first; scaling every entry by one constant changes nothing after
    normalisation), then ``t_max`` times: normalise columns, then rows. After the last step rows sum to 1
    exactly (up to rounding); columns to within a tolerance that shrinks with t_max. Differentiable: autograd
    traces every iteration (mHC writes a fused kernel with a recomputing backward, section 4.3.1).
    """
    m = torch.exp(logits - logits.amax(dim=(-2, -1), keepdim=True))
    for _ in range(t_max):
        m = m / m.sum(dim=-2, keepdim=True)      # T_c: columns sum to 1
        m = m / m.sum(dim=-1, keepdim=True)      # T_r: rows sum to 1
    return m


def doubly_stochastic_error(m: torch.Tensor) -> float:
    """max |row sum − 1|, |column sum − 1| and max(0, −entry): 0 for an exact doubly-stochastic matrix."""
    return max((m.sum(-1) - 1).abs().max().item(), (m.sum(-2) - 1).abs().max().item(),
               (-m).clamp_min(0).max().item())


class HyperConnection(nn.Module):
    """Zhu et al.'s dynamic hyper-connection around one sub-layer (unconstrained H_res).

    ``forward(X, F)``: X (B, T, n, C), F a callable (B, T, C) -> (B, T, C). Returns the new stream.
    Notation of the HC paper (Eqs. 2, 10–13), per token with H = X (n x C):

        H_bar = RMSNorm(H) per stream
        B     = s_beta  · tanh(H_bar W_beta)^T + B_static      (1 x n)   write weights
        A_m   = s_alpha · tanh(H_bar W_m)      + A_m_static    (n x 1)   read weights
        A_r   = s_alpha · tanh(H_bar W_r)      + A_r_static    (n x n)   stream mixing
        h0    = A_m^T H                        (1 x C)          the sub-layer's input
        H_out = B^T F(h0) + A_r^T H            (n x C)

    In mHC orientation H_res = A_r^T, H_pre = A_m^T, H_post = B.
    """

    def __init__(self, n: int, C: int, layer_index: int, eps: float = 1e-6, scale_init: float = 0.01):
        super().__init__()
        self.n = n
        self.norm = RMSNorm(C, eps=eps)
        self.w_beta = nn.Parameter(torch.zeros(C, 1))
        self.w_m = nn.Parameter(torch.zeros(C, 1))
        self.w_r = nn.Parameter(torch.zeros(C, n))
        self.s_alpha = nn.Parameter(torch.full((), scale_init))
        self.s_beta = nn.Parameter(torch.full((), scale_init))
        self.b_static = nn.Parameter(torch.ones(n))
        am = torch.zeros(n)
        am[layer_index % n] = 1.0
        self.am_static = nn.Parameter(am)
        self.ar_static = nn.Parameter(torch.eye(n))
        self.last_res: torch.Tensor | None = None
        self.track = False

    def coefficients(self, X: torch.Tensor):
        hb = self.norm(X)                                                   # (B, T, n, C)
        b = self.s_beta * torch.tanh(hb @ self.w_beta).squeeze(-1) + self.b_static          # (B, T, n)
        am = self.s_alpha * torch.tanh(hb @ self.w_m).squeeze(-1) + self.am_static          # (B, T, n)
        ar = self.s_alpha * torch.tanh(hb @ self.w_r) + self.ar_static                      # (B, T, n, n) rows j
        return b, am, ar

    def forward(self, X: torch.Tensor, F_sub) -> torch.Tensor:
        b, am, ar = self.coefficients(X)
        h0 = torch.einsum("btj,btjc->btc", am, X)                          # A_m^T H
        y = F_sub(h0)                                                       # (B, T, C)
        mixed = torch.einsum("btji,btjc->btic", ar, X)                      # A_r^T H: out_i = sum_j A_r[j,i] H_j
        if self.track:
            self.last_res = ar.transpose(-1, -2).detach()
        return mixed + b.unsqueeze(-1) * y.unsqueeze(-2)


class ManifoldHC(nn.Module):
    """mHC around one sub-layer: H_res projected onto doubly-stochastic matrices by Sinkhorn-Knopp.

    Per token, x_vec = vec(X) in R^{nC} (mHC Eqs. 7–8):

        x'          = RMSNorm(x_vec)
        H~_pre      = alpha_pre  · (x' phi_pre)  + b_pre           (n)
        H~_post     = alpha_post · (x' phi_post) + b_post          (n)
        H~_res      = alpha_res  · mat(x' phi_res) + b_res         (n x n)
        H_pre = sigmoid(H~_pre),  H_post = 2·sigmoid(H~_post),  H_res = SinkhornKnopp(H~_res, t_max)
        X_out       = H_res X + H_post^T F(H_pre X)
    """

    def __init__(self, n: int, C: int, layer_index: int, t_max: int = 20, eps: float = 1e-6,
                 alpha_init: float = 0.01, res_diag: float = 4.0, pre_logit: float = 4.0, init_std: float = 0.02):
        super().__init__()
        self.n, self.t_max = n, t_max
        self.norm = RMSNorm(n * C, eps=eps)
        self.phi = nn.Parameter(torch.randn(n * C, n * n + 2 * n) * init_std)   # [pre | post | res], fused
        self.alpha_pre = nn.Parameter(torch.full((), alpha_init))
        self.alpha_post = nn.Parameter(torch.full((), alpha_init))
        self.alpha_res = nn.Parameter(torch.full((), alpha_init))
        bp = torch.full((n,), -pre_logit)
        bp[layer_index % n] = pre_logit
        self.b_pre = nn.Parameter(bp if n > 1 else torch.zeros(1))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res = nn.Parameter(torch.eye(n) * res_diag)
        self.last_res: torch.Tensor | None = None
        self.track = False

    def coefficients(self, X: torch.Tensor):
        Bsz, T, n, C = X.shape
        z = self.norm(X.reshape(Bsz, T, n * C)) @ self.phi                 # (B, T, n² + 2n)
        pre = torch.sigmoid(self.alpha_pre * z[..., :n] + self.b_pre)
        post = 2 * torch.sigmoid(self.alpha_post * z[..., n:2 * n] + self.b_post)
        res = sinkhorn_knopp(self.alpha_res * z[..., 2 * n:].reshape(Bsz, T, n, n) + self.b_res, self.t_max)
        return pre, post, res

    def forward(self, X: torch.Tensor, F_sub) -> torch.Tensor:
        pre, post, res = self.coefficients(X)
        y = F_sub(torch.einsum("btj,btjc->btc", pre, X))
        if self.track:
            self.last_res = res.detach()
        return torch.einsum("btij,btjc->btic", res, X) + post.unsqueeze(-1) * y.unsqueeze(-2)


RESIDUALS = {"hc": HyperConnection, "mhc": ManifoldHC}


def expand_streams(x: torch.Tensor, n: int) -> torch.Tensor:
    """(B, T, C) -> (B, T, n, C): the embedding copied into every stream (HC section 2.1)."""
    return x.unsqueeze(-2).expand(*x.shape[:-1], n, x.shape[-1]).contiguous()


def collapse_streams(X: torch.Tensor) -> torch.Tensor:
    """(B, T, n, C) -> (B, T, C): sum of the streams (HC section 2.1), then the model's final norm."""
    return X.sum(dim=-2)


@torch.no_grad()
def amax_gain(res_mats: list[torch.Tensor]) -> dict:
    """mHC section 3.1 diagnostic. ``res_mats``: the H_res of every sub-layer, bottom to top, each (B, T, n, n).

    For the single-layer maps and for the composite product H_res^(L) ... H_res^(1) (per token), the
    forward gain is the maximum absolute row sum and the backward gain the maximum absolute column sum,
    averaged over tokens. Both are exactly 1 for doubly-stochastic matrices; mHC reports peaks near 3,000
    for HC in its 27B model (Fig. 3b) and about 1.6 for mHC (Fig. 7b)."""
    single_f, single_b = [], []
    comp = None
    comp_f, comp_b = [], []
    for m in res_mats:
        m = m.double()
        single_f.append(m.abs().sum(-1).amax(-1).mean().item())
        single_b.append(m.abs().sum(-2).amax(-1).mean().item())
        comp = m if comp is None else m @ comp
        comp_f.append(comp.abs().sum(-1).amax(-1).mean().item())
        comp_b.append(comp.abs().sum(-2).amax(-1).mean().item())
    return {"single_fwd_max": max(single_f), "single_bwd_max": max(single_b),
            "composite_fwd": comp_f[-1], "composite_bwd": comp_b[-1],
            "composite_fwd_max": max(comp_f), "composite_bwd_max": max(comp_b)}


def residual_io_elements(n: int, C: int, kind: str = "hc") -> dict:
    """Per token and sub-layer, elements read and written for the residual itself (mHC Table 2; forward pass,
    excluding the sub-layer F). Plain residual: 2C read, C written. HC (and unfused mHC): (5n+1)C + n² + 2n
    read, (3n+1)C + n² + 2n written."""
    if kind == "plain":
        return {"read": 2 * C, "write": C}
    return {"read": (5 * n + 1) * C + n * n + 2 * n, "write": (3 * n + 1) * C + n * n + 2 * n}


def hc_flops_per_token_sublayer(n: int, C: int, kind: str = "mhc", t_max: int = 20) -> float:
    """Forward FLOPs per token for one wrapped sub-layer, beyond F itself (course count, multiply-add = 2).

    mHC: coefficient projection 2·nC·(n² + 2n); RMSNorm over nC ≈ 3nC; read H_pre X 2nC; mix H_res X 2n²C;
    write H_post^T y 2nC; Sinkhorn ≈ t_max · 4n² (two divisions and two sums per entry). HC: per-stream
    projections 2·nC·(n + 2) plus the same read/mix/write terms."""
    proj = 2 * n * C * (n * n + 2 * n) if kind == "mhc" else 2 * n * C * (n + 2)
    sk = t_max * 4 * n * n if kind == "mhc" else 0
    return proj + 3 * n * C + 2 * n * C + 2 * n * n * C + 2 * n * C + sk


def plain_residual(x: torch.Tensor, F_sub) -> torch.Tensor:
    """The reference the n = 1 tests compare against: x + F(x)."""
    return x + F_sub(x)


__all__ = ["HyperConnection", "ManifoldHC", "RESIDUALS", "amax_gain", "collapse_streams", "doubly_stochastic_error",
           "expand_streams", "hc_flops_per_token_sublayer", "plain_residual", "residual_io_elements", "sinkhorn_knopp"]
