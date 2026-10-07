"""Sparse autoencoders, from scratch: ReLU (L1), TopK and JumpReLU (L0) (lesson 17.1).

An SAE rewrites an activation ``x`` (d) as a sparse, non-negative combination of ``d_sae ≫ d`` learned
directions (the decoder rows), plus an error:

    pre = (x − b_dec) W_enc + b_enc                  W_enc (d, d_sae)
    z   = σ(pre)                                     sparse code (d_sae,), σ below
    x̂   = z W_dec + b_dec                            W_dec (d_sae, d), rows kept at unit norm
    x   = x̂ + e                                      e = reconstruction error

=========  ===========================================  ==============================================
kind       σ                                            sparsity term in the loss
=========  ===========================================  ==============================================
relu       ReLU(pre)                                    λ · Σ_i z_i (L1; Bricken et al. 2023)
topk       keep the k largest pre, ReLU, zero the rest  none: L0 = k by construction (Gao et al. 2024);
                                                        optional AuxK loss on dead latents
jumprelu   pre · H(pre − θ_i), θ_i > 0 learned          λ · Σ_i H(pre_i − θ_i) (L0; Rajamanoharan et al.
                                                        2024), trained with straight-through estimators
=========  ===========================================  ==============================================

Inputs are divided by one scalar so that E[x_j²] = 1 per dimension (the JumpReLU paper's convention; it is
what makes its bandwidth ε = 0.001 meaningful); ``forward`` and ``encode`` accept activations in the
model's own units and the scale is undone on the way out.

Quality is never one number. :func:`evaluate` reports the fraction of variance unexplained (FVU), L0 and
dead latents; :func:`spliced_losses` puts the reconstruction back into the model and measures the change
in next-token loss — the measure Gemma Scope calls delta LM loss and Gao et al. call downstream loss. A
low FVU with a large loss increase means the SAE keeps the variance and loses what the model uses.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from frontierlab.interp import hooks as HK


# --------------------------------------------------------------------------------------------- JumpReLU STE

def _rect(u: torch.Tensor) -> torch.Tensor:
    return ((u > -0.5) & (u < 0.5)).to(u.dtype)


class _JumpReLU(torch.autograd.Function):
    """z = pre · H(pre − θ). Gradient w.r.t. pre: H(pre − θ). Pseudo-gradient w.r.t. θ (paper Eq. 11):
    −(θ/ε) · rect((pre − θ)/ε)."""

    @staticmethod
    def forward(ctx, pre, theta, eps):
        ctx.save_for_backward(pre, theta)
        ctx.eps = eps
        return pre * (pre > theta).to(pre.dtype)

    @staticmethod
    def backward(ctx, g):
        pre, theta = ctx.saved_tensors
        eps = ctx.eps
        g_pre = g * (pre > theta).to(pre.dtype)
        g_theta = (g * (-(theta / eps) * _rect((pre - theta) / eps))).sum(dim=tuple(range(pre.dim() - 1)))
        return g_pre, g_theta, None


class _Step(torch.autograd.Function):
    """H(pre − θ) with pseudo-gradient w.r.t. θ (paper Eq. 12): −(1/ε) · rect((pre − θ)/ε); none w.r.t. pre."""

    @staticmethod
    def forward(ctx, pre, theta, eps):
        ctx.save_for_backward(pre, theta)
        ctx.eps = eps
        return (pre > theta).to(pre.dtype)

    @staticmethod
    def backward(ctx, g):
        pre, theta = ctx.saved_tensors
        eps = ctx.eps
        g_theta = (g * (-(1.0 / eps) * _rect((pre - theta) / eps))).sum(dim=tuple(range(pre.dim() - 1)))
        return torch.zeros_like(pre), g_theta, None


def jumprelu(pre, theta, eps: float = 1e-3):
    return _JumpReLU.apply(pre, theta, eps)


def step(pre, theta, eps: float = 1e-3):
    return _Step.apply(pre, theta, eps)


# --------------------------------------------------------------------------------------------- the SAE

class SAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, kind: str = "topk", k: int = 32, eps: float = 1e-3,
                 theta_init: float = 1e-3, seed: int = 0):
        super().__init__()
        if kind not in ("relu", "topk", "jumprelu"):
            raise ValueError("kind is 'relu', 'topk' or 'jumprelu'")
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(d_sae, d_in, generator=g)
        W = W / W.norm(dim=1, keepdim=True)
        self.kind, self.k, self.eps = kind, k, eps
        self.W_dec = nn.Parameter(W.clone())                       # (d_sae, d_in), unit-norm rows
        self.W_enc = nn.Parameter(W.T.clone())                     # (d_in, d_sae): transpose init (Gao et al.)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.threshold = nn.Parameter(torch.full((d_sae,), float(theta_init)), requires_grad=kind == "jumprelu")
        self.register_buffer("scale", torch.ones(()))

    @property
    def d_sae(self) -> int:
        return self.W_dec.shape[0]

    @torch.no_grad()
    def init_from(self, acts: torch.Tensor):
        """Set the input scale (E[x_j²] = 1 after scaling) and b_dec to the data mean."""
        x = acts.float()
        self.scale.copy_((x.pow(2).mean()).sqrt())
        self.b_dec.copy_((x / self.scale).mean(0))

    def pre(self, xn: torch.Tensor) -> torch.Tensor:
        return (xn - self.b_dec) @ self.W_enc + self.b_enc

    def code(self, pre: torch.Tensor) -> torch.Tensor:
        if self.kind == "relu":
            return torch.relu(pre)
        if self.kind == "topk":
            v, i = pre.topk(self.k, dim=-1)
            return torch.zeros_like(pre).scatter(-1, i, torch.relu(v))
        return jumprelu(pre, self.threshold.clamp_min(1e-6), self.eps)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse code z (…, d_sae) of activations x (…, d) in model units."""
        return self.code(self.pre(x.to(self.W_enc.dtype) / self.scale))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruction in model units."""
        return (z @ self.W_dec + self.b_dec) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x)).to(x.dtype)

    def loss(self, x: torch.Tensor, coeff: float = 0.0, aux_alpha: float = 0.0, dead: torch.Tensor | None = None,
             k_aux: int = 256) -> dict:
        """Training loss on a batch x (B, d) in model units. MSE is per-token sum of squares in scaled units."""
        xn = x.to(self.W_enc.dtype) / self.scale
        pre = self.pre(xn)
        z = self.code(pre)
        xh = z @ self.W_dec + self.b_dec
        mse = (xh - xn).pow(2).sum(-1).mean()
        out = {"mse": mse, "z": z}
        if self.kind == "relu":
            sp = z.sum(-1).mean()
        elif self.kind == "jumprelu":
            sp = step(pre, self.threshold.clamp_min(1e-6), self.eps).sum(-1).mean()
        else:
            sp = torch.zeros(())
        out["sparsity"] = sp
        total = mse + coeff * sp
        if self.kind == "topk" and aux_alpha > 0 and dead is not None and dead.any():
            # AuxK (Gao et al. A.2): reconstruct the residual error with the top-k_aux *dead* latents.
            e = (xn - xh).detach()
            pd = pre.masked_fill(~dead, float("-inf"))
            ka = min(k_aux, int(dead.sum()))
            v, i = pd.topk(ka, dim=-1)
            za = torch.zeros_like(pre).scatter(-1, i, torch.relu(v))
            aux = (za @ self.W_dec - e).pow(2).sum(-1).mean()
            out["aux"] = aux
            total = total + aux_alpha * aux
        out["loss"] = total
        return out

    @torch.no_grad()
    def renorm(self):
        """Keep decoder rows at unit norm (Towards Monosemanticity); removes the gradient component that would
        only change the norms, so the L1 penalty cannot be dodged by shrinking z and growing W_dec."""
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))


def train_sae(acts: torch.Tensor, kind: str = "topk", d_sae: int = 1024, k: int = 32, coeff: float | None = None,
              steps: int = 3000, batch: int = 512, lr: float = 1e-3, aux_alpha: float = 1 / 32, seed: int = 0,
              dead_after: int = 200, log_every: int = 0) -> tuple[SAE, list[dict]]:
    """Train one SAE on activations ``acts`` (N, d). ``coeff`` is λ (L1 for relu, L0 for jumprelu; warmed up
    linearly over the first 5% of steps); ignored for topk. A latent counts as dead in training after
    ``dead_after`` steps without firing (it then receives the AuxK loss)."""
    torch.manual_seed(seed)
    acts = acts.float()
    sae = SAE(acts.shape[1], d_sae, kind, k=k, seed=seed).to(acts.device)
    sae.init_from(acts)
    if coeff is None:
        coeff = {"relu": 1.0, "jumprelu": 0.5, "topk": 0.0}[kind]
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    g = torch.Generator().manual_seed(seed)
    since = torch.zeros(d_sae, dtype=torch.long, device=acts.device)
    log = []
    for s in range(steps):
        x = acts[torch.randint(0, len(acts), (batch,), generator=g).to(acts.device)]
        lam = coeff * min(1.0, (s + 1) / max(1, steps // 20))
        out = sae.loss(x, lam, aux_alpha if kind == "topk" else 0.0, since >= dead_after)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        opt.step()
        sae.renorm()
        fired = (out["z"] > 0).any(0)
        since = torch.where(fired, torch.zeros_like(since), since + 1)
        if log_every and (s % log_every == 0 or s == steps - 1):
            row = {"step": s, "mse": float(out["mse"]), "l0": float((out["z"] > 0).float().sum(-1).mean()),
                   "dead_train": float((since >= dead_after).float().mean())}
            log.append(row)
    return sae.eval(), log


# --------------------------------------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate(sae: SAE, acts: torch.Tensor, batch: int = 4096) -> dict:
    """FVU = Σ‖x − x̂‖² / Σ‖x − mean(x)‖², mean L0, and the fraction of latents that never fire on ``acts``."""
    acts = acts.float()
    mu = acts.mean(0)
    se = tot = l0 = 0.0
    fired = torch.zeros(sae.d_sae, dtype=torch.bool, device=acts.device)
    for i in range(0, len(acts), batch):
        x = acts[i:i + batch]
        z = sae.encode(x)
        xh = sae.decode(z)
        se += float((x - xh).pow(2).sum())
        tot += float((x - mu).pow(2).sum())
        l0 += float((z > 0).float().sum())
        fired |= (z > 0).any(0)
    return {"fvu": se / tot, "l0": l0 / len(acts), "dead": float((~fired).float().mean())}


@torch.no_grad()
def spliced_losses(model, windows: torch.Tensor, site: str, sae: SAE | None = None, mode: str = "sae",
                   batch: int = 16, mean_act: torch.Tensor | None = None) -> np.ndarray:
    """Per-window next-token loss with ``site`` replaced. ``mode``: ``"clean"`` (no change), ``"sae"`` (x̂),
    ``"sae+error"`` (x̂ + (x − x̂): must equal clean — the splice check), ``"zero"``, ``"mean"``
    (``mean_act``, (d,), e.g. the mean of the training activations)."""
    def edit(x):
        if mode == "sae":
            return sae(x)
        if mode == "sae+error":
            return sae(x) + (x - sae(x))
        if mode == "zero":
            return torch.zeros_like(x)
        if mode == "mean":
            return mean_act.to(x.dtype).expand_as(x).clone()
        return x
    out = []
    for i in range(0, len(windows), batch):
        x = windows[i:i + batch]
        logits = HK.run_with(model, x, {} if mode == "clean" else {site: edit})
        lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, x[:, 1:, None])[..., 0]
        out.extend((-lp.mean(1)).tolist())
    return np.array(out)


def loss_recovered(clean: float, spliced: float, ablated: float) -> float:
    """(L_ablated − L_spliced) / (L_ablated − L_clean): 1 = as good as the model, 0 = as bad as ablation."""
    return (ablated - spliced) / (ablated - clean)


@torch.no_grad()
def collect(model, windows: torch.Tensor, site: str, batch: int = 16, skip_first: int = 1) -> torch.Tensor:
    """Activations at ``site`` for every token of ``windows`` (N·T', d), dropping the first ``skip_first``
    positions of each window (position 0 is an outlier in most models: it attends only to itself)."""
    rows = []
    for i in range(0, len(windows), batch):
        _, a = HK.capture(model, windows[i:i + batch], [site])
        rows.append(a[site][:, skip_first:].reshape(-1, a[site].shape[-1]).float())
    return torch.cat(rows)


@torch.no_grad()
def top_contexts(sae: SAE, model, windows: torch.Tensor, site: str, latent: int, k: int = 8, width: int = 6,
                 batch: int = 16):
    """The ``k`` (window, position, activation) triples where ``latent`` fires most, with ``width`` tokens of
    left context: [(activation, [ids before], id at position)]."""
    best = []
    for i in range(0, len(windows), batch):
        x = windows[i:i + batch]
        _, a = HK.capture(model, x, [site])
        z = sae.encode(a[site])[..., latent]                   # (b, T)
        for b in range(z.shape[0]):
            for t in range(1, z.shape[1]):
                if z[b, t] > 0:
                    best.append((float(z[b, t]), i + b, t))
    best.sort(reverse=True)
    out = []
    for v, w, t in best[:k]:
        out.append((v, windows[w, max(0, t - width):t].tolist(), int(windows[w, t])))
    return out
