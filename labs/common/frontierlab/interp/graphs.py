"""Transcoders, the local replacement model and attribution graphs, from scratch (lesson 17.3).

A **transcoder** (Dunefsky et al. 2024) is a sparse stand-in for one MLP: it reads the MLP's input and
predicts its output, ``y ≈ W_dec · σ(W_enc x + b_enc) + b_dec``. Replacing every MLP with its transcoder
gives a **replacement model** whose MLP computation is a sum of sparse, readable features.

Circuit Tracing (Ameisen et al. 2025) turns one prompt into a graph. For that prompt it builds the
**local replacement model**:

* attention patterns and normalisation denominators are frozen at their values on the prompt;
* each MLP output is the transcoder's reconstruction plus an **error node** holding the exact difference,
  so the local replacement model reproduces the original logits exactly;
* with patterns, norms and errors fixed, every feature's pre-activation is a *linear* function of the
  token embeddings, the error nodes and the activations of earlier features.

The **attribution graph** has a node for every token embedding, every active feature at every position,
every error node and the top output logits. The edge weight from source ``s`` to target ``u`` is the
direct effect ``a_s · ∂(input_u)/∂a_s`` through the residual stream and the frozen attention, holding
every other feature fixed (each feature activation is a leaf in the computation). For a logit node the
input is the logit minus the mean logit. Edges into a target sum exactly to its input minus its encoder bias (a test checks it).

**Pruning** keeps the nodes that carry most of the influence on the logits: normalise the absolute edge
weights into each node to sum to 1, compute the total influence ``B = Â + Â² + … = (I − Â)⁻¹ − I``,
weight it by the logit probabilities and keep nodes up to a cumulative 0.8 of the influence (circuit-tracer
defaults: node threshold 0.8, edge threshold 0.98).

A graph is a **hypothesis**. :func:`intervene_feature` tests it in the *original* model: it removes one
feature's write from one MLP output (the rest of that MLP output, including the error, unchanged) and
lets attention, norms and every other MLP recompute. The graph's prediction comes from the same edit in
the local replacement model (:func:`replacement_logits`). Random active features of similar activation
are the control.

This module implements Baseline-0's GQA layout (``frontierlab.model.LM`` with ``attention="gqa"``); the main
path uses circuit-tracer 0.5.0 on Qwen3 (:mod:`frontierlab.interp.hf`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from frontierlab.attention.base import apply_rope
from frontierlab.interp import hooks as HK
from frontierlab.interp.sae import SAE


# --------------------------------------------------------------------------------------------- transcoders

class Transcoder(SAE):
    """An SAE whose decoder predicts a *different* tensor (the MLP output) from its encoder input (the MLP
    input). Separate input and output scales; ``b_dec`` is the output bias and is not subtracted from the
    input (Dunefsky et al. Eq. 3-4)."""

    def __init__(self, d_in: int, d_sae: int, kind: str = "topk", k: int = 16, seed: int = 0, **kw):
        super().__init__(d_in, d_sae, kind, k=k, seed=seed, **kw)
        self.register_buffer("out_scale", torch.ones(()))

    @torch.no_grad()
    def init_from_pair(self, x_in: torch.Tensor, y_out: torch.Tensor):
        self.scale.copy_(x_in.float().pow(2).mean().sqrt())
        self.out_scale.copy_(y_out.float().pow(2).mean().sqrt())
        self.b_dec.copy_((y_out.float() / self.out_scale).mean(0))

    def pre(self, xn):
        return xn @ self.W_enc + self.b_enc

    def decode(self, z):
        return (z @ self.W_dec + self.b_dec) * self.out_scale

    def loss_pair(self, x_in, y_out, coeff: float = 0.0) -> dict:
        pre = self.pre(x_in.float() / self.scale)
        z = self.code(pre)
        yh = z @ self.W_dec + self.b_dec
        mse = (yh - y_out.float() / self.out_scale).pow(2).sum(-1).mean()
        sp = z.sum(-1).mean() if self.kind == "relu" else torch.zeros(())
        return {"mse": mse, "z": z, "loss": mse + coeff * sp}


def train_transcoder(x_in: torch.Tensor, y_out: torch.Tensor, d_sae: int = 1024, kind: str = "topk", k: int = 16,
                     coeff: float = 5e-3, steps: int = 2000, batch: int = 512, lr: float = 1e-3, seed: int = 0) -> Transcoder:
    torch.manual_seed(seed)
    tc = Transcoder(x_in.shape[1], d_sae, kind, k=k, seed=seed)
    tc.init_from_pair(x_in, y_out)
    opt = torch.optim.Adam(tc.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for s in range(steps):
        i = torch.randint(0, len(x_in), (batch,), generator=g)
        out = tc.loss_pair(x_in[i], y_out[i], coeff * min(1.0, (s + 1) / max(1, steps // 20)))
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        opt.step()
        tc.renorm()
    return tc.eval()


@torch.no_grad()
def transcoder_fvu(tc: Transcoder, x_in, y_out) -> dict:
    yh = tc.decode(tc.encode(x_in))
    fvu = float((y_out - yh).pow(2).sum() / (y_out - y_out.mean(0)).pow(2).sum())
    return {"fvu": fvu, "l0": float((tc.encode(x_in) > 0).float().sum(-1).mean())}


@torch.no_grad()
def collect_mlp(model, windows: torch.Tensor, layer: int, batch: int = 16, skip_first: int = 1):
    """(mlp_in, mlp_out) pairs of one layer for every token of ``windows``."""
    xs, ys = [], []
    for i in range(0, len(windows), batch):
        _, a = HK.capture(model, windows[i:i + batch], [f"mlp_in.{layer}", f"mlp_out.{layer}"])
        xs.append(a[f"mlp_in.{layer}"][:, skip_first:].reshape(-1, a[f"mlp_in.{layer}"].shape[-1]))
        ys.append(a[f"mlp_out.{layer}"][:, skip_first:].reshape(-1, a[f"mlp_out.{layer}"].shape[-1]))
    return torch.cat(xs).float(), torch.cat(ys).float()


# --------------------------------------------------------------------------------------------- glass-box forward

def _rms(x, eps):
    return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


@torch.no_grad()
def trace(model, idx: torch.Tensor) -> dict:
    """An explicit forward of Baseline-0 (GQA) that records what the local replacement model freezes:
    per layer the input-norm and post-attention-norm inverse RMS (B, T, 1), the attention pattern
    (B, H, T, T), ``mlp_in`` and ``mlp_out`` (B, T, C); the final norm's inverse RMS; the logits."""
    mm = model.model
    x = mm.embed_tokens(idx)
    T = idx.shape[1]
    pos = torch.arange(T)
    rec = {"embed": x.clone(), "layers": []}
    mask = pos[None, :] <= pos[:, None]
    for layer in mm.layers:
        at = layer.self_attn
        r_in = _rms(x, layer.input_layernorm.eps)
        n = x * r_in * layer.input_layernorm.weight
        B = x.shape[0]
        q = at.q_norm(at.q_proj(n).view(B, T, at.H, at.hd)).transpose(1, 2)
        k = at.k_norm(at.k_proj(n).view(B, T, at.KV, at.hd)).transpose(1, 2)
        v = at.v_proj(n).view(B, T, at.KV, at.hd).transpose(1, 2)
        cos, sin = at.rope(pos)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = at.H // at.KV
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        s = (q @ k.transpose(-1, -2)) / at.hd ** 0.5
        pat = torch.softmax(s.masked_fill(~mask, float("-inf")), -1)
        y = (pat @ v).transpose(1, 2).reshape(B, T, at.H * at.hd)
        x = x + at.o_proj(y)
        r_post = _rms(x, layer.post_attention_layernorm.eps)
        m_in = x * r_post * layer.post_attention_layernorm.weight
        m_out = layer.mlp(m_in)
        x = x + m_out
        rec["layers"].append({"r_in": r_in, "pattern": pat, "r_post": r_post, "mlp_in": m_in, "mlp_out": m_out})
    rec["r_final"] = _rms(x, mm.norm.eps)
    rec["logits"] = model.lm_head(x * rec["r_final"] * mm.norm.weight)
    return rec


def _frozen_attn(layer, x, r_in, pat):
    """Attention output with the pattern and the input norm's 1/RMS frozen: linear in ``x``."""
    at = layer.self_attn
    B, T, _ = x.shape
    n = x * r_in * layer.input_layernorm.weight
    v = at.v_proj(n).view(B, T, at.KV, at.hd).transpose(1, 2).repeat_interleave(at.H // at.KV, 1)
    return at.o_proj((pat @ v).transpose(1, 2).reshape(B, T, at.H * at.hd))


def replacement_logits(model, rec: dict, tcs: list[Transcoder], acts: list[torch.Tensor] | None = None,
                       errors: list[torch.Tensor] | None = None, embed: torch.Tensor | None = None,
                       recompute_features: bool = False, edits: dict | None = None):
    """Logits of the local replacement model for the prompt recorded in ``rec``.

    With ``recompute_features=False`` (attribution mode) the feature activations ``acts[l]`` (B, T, F)
    are inputs, and the function also returns every layer's feature pre-activations (the targets).
    With ``recompute_features=True`` each layer's features are recomputed from its own MLP input through
    the transcoder's nonlinearity (with ``edits = {(layer, pos, feature): value}`` clamped), the errors
    stay fixed: this is the graph's *prediction* for an intervention."""
    x = rec["embed"] if embed is None else embed
    pres = []
    for l, layer in enumerate(model.model.layers):
        L = rec["layers"][l]
        x = x + _frozen_attn(layer, x, L["r_in"], L["pattern"])
        m_in = x * L["r_post"] * layer.post_attention_layernorm.weight
        tc = tcs[l]
        pre = tc.pre(m_in / tc.scale)
        pres.append(pre)
        if recompute_features:
            z = tc.code(pre)
            if edits:
                z = z.clone()
                for (el, ep, ef), val in edits.items():
                    if el == l:
                        z[:, ep, ef] = val
        else:
            z = acts[l]
        x = x + tc.decode(z) + errors[l]
    logits = model.lm_head(x * rec["r_final"] * model.model.norm.weight)
    return logits, pres


@torch.no_grad()
def clean_codes(rec: dict, tcs: list[Transcoder]):
    """Feature activations of each layer on the prompt and the error nodes ``mlp_out − decode(z)``."""
    acts, errs = [], []
    for L, tc in zip(rec["layers"], tcs):
        z = tc.encode(L["mlp_in"])
        acts.append(z)
        errs.append(L["mlp_out"] - tc.decode(z))
    return acts, errs


# --------------------------------------------------------------------------------------------- the graph

@dataclass
class Graph:
    nodes: list            # ("emb", pos, token) | ("feat", layer, pos, f) | ("err", layer, pos) | ("logit", token)
    A: np.ndarray          # (n, n): A[u, s] = direct effect of source s on target u's input
    activations: np.ndarray  # activation of each node (embedding/error: 1; feature: a; logit: its input)
    logit_probs: np.ndarray  # probability of each logit node (weights for pruning)
    meta: dict = field(default_factory=dict)

    def index(self, node) -> int:
        return self.nodes.index(node)


def attribute(model, idx: torch.Tensor, tcs: list[Transcoder], n_logits: int = 3, pos: int = -1,
              max_feature_nodes: int | None = None) -> Graph:
    """Attribution graph of one prompt ``idx`` (1, T) for the top ``n_logits`` next tokens at ``pos``."""
    assert idx.shape[0] == 1, "one prompt at a time"
    rec = trace(model, idx)
    acts, errs = clean_codes(rec, tcs)
    T = idx.shape[1]
    pos = pos % T
    # leaves
    emb = rec["embed"].clone().requires_grad_(True)
    a_leaf = [a.clone().requires_grad_(True) for a in acts]
    e_leaf = [e.clone().requires_grad_(True) for e in errs]
    with torch.enable_grad():
        logits, pres = replacement_logits(model, rec, tcs, a_leaf, e_leaf, emb)
    probs = torch.softmax(logits[0, pos].detach(), -1)
    top = probs.topk(n_logits).indices.tolist()
    # nodes
    feats = [(l, p, f) for l, a in enumerate(acts) for p, f in (a[0] > 0).nonzero().tolist()]
    if max_feature_nodes is not None and len(feats) > max_feature_nodes:
        feats.sort(key=lambda t: -float(acts[t[0]][0, t[1], t[2]]))
        feats = feats[:max_feature_nodes]
    nodes = ([("emb", p, int(idx[0, p])) for p in range(T)] + [("feat", l, p, f) for l, p, f in feats]
             + [("err", l, p) for l in range(len(tcs)) for p in range(T)] + [("logit", t) for t in top])
    where = {nd: i for i, nd in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n))
    act = np.ones(n)
    for (l, p, f) in feats:
        act[where[("feat", l, p, f)]] = float(acts[l][0, p, f])
    leaves = [emb] + a_leaf + e_leaf

    emb_idx = np.array([where[("emb", p, int(idx[0, p]))] for p in range(T)])
    err_idx = [np.array([where[("err", l, p)] for p in range(T)]) for l in range(len(tcs))]
    per_layer = []
    for l in range(len(tcs)):
        fl = [(p, f) for (ll, p, f) in feats if ll == l]
        per_layer.append((torch.tensor([p for p, _ in fl], dtype=torch.long), torch.tensor([f for _, f in fl], dtype=torch.long),
                          np.array([where[("feat", l, p, f)] for p, f in fl], dtype=np.int64)))

    def fill(u: int, target: torch.Tensor):
        g = torch.autograd.grad(target, leaves, retain_graph=True, allow_unused=True)
        g_emb, g_a, g_e = g[0], g[1:1 + len(a_leaf)], g[1 + len(a_leaf):]
        A[u, emb_idx] = (g_emb * rec["embed"])[0].sum(-1).detach().double().numpy()
        for l in range(len(tcs)):
            ps, fs, ni = per_layer[l]
            if g_a[l] is not None and len(ni):
                A[u, ni] = (g_a[l] * acts[l])[0][ps, fs].detach().double().numpy()
            if g_e[l] is not None:
                A[u, err_idx[l]] = (g_e[l] * errs[l])[0].sum(-1).detach().double().numpy()

    with torch.enable_grad():
        for (l, p, f) in feats:
            fill(where[("feat", l, p, f)], pres[l][0, p, f])
        for t in top:
            target = logits[0, pos, t] - logits[0, pos].mean()
            fill(where[("logit", t)], target)
            act[where[("logit", t)]] = float(target.detach())
    lp = np.zeros(n)
    for t in top:
        lp[where[("logit", t)]] = float(probs[t])
    return Graph(nodes, A, act, lp, {"pos": pos, "tokens": idx[0].tolist(), "top": top,
                                     "top_probs": [float(probs[t]) for t in top]})


def influence(g: Graph) -> np.ndarray:
    """Total (direct + indirect) influence of every node on the logits, from the normalised graph:
    Â = |A| / row sums, B = (I − Â)⁻¹ − I, influence = wᵀ B with w the logit probabilities."""
    Ah = np.abs(g.A)
    rs = Ah.sum(1, keepdims=True)
    Ah = np.divide(Ah, rs, out=np.zeros_like(Ah), where=rs > 0)
    n = len(g.nodes)
    B = np.linalg.inv(np.eye(n) - Ah) - np.eye(n)
    w = g.logit_probs / g.logit_probs.sum()
    return w @ B


def prune(g: Graph, node_threshold: float = 0.8) -> list[int]:
    """Indices of the nodes kept: logit nodes always, plus the most influential nodes until their cumulative
    share of the total influence reaches ``node_threshold``."""
    inf = influence(g)
    order = np.argsort(-inf)
    total = inf.sum()
    keep, run = [], 0.0
    for i in order:
        if g.nodes[i][0] == "logit":
            continue
        if run / max(total, 1e-12) >= node_threshold:
            break
        keep.append(int(i))
        run += inf[i]
    return sorted(keep + [i for i, nd in enumerate(g.nodes) if nd[0] == "logit"])


def error_share(g: Graph) -> float:
    """Share of the logit influence that flows from error nodes: 1 − this is a completeness-style score
    (how much of the explanation goes through interpretable nodes)."""
    inf = influence(g)
    err = [i for i, nd in enumerate(g.nodes) if nd[0] == "err"]
    # embeddings and errors have no inputs (roots); every path to a logit starts at one of them
    roots = [i for i, nd in enumerate(g.nodes) if nd[0] in ("emb", "err")]
    tot = inf[roots].sum()
    return float(inf[err].sum() / tot) if tot > 0 else float("nan")


# --------------------------------------------------------------------------------------------- interventions

@torch.no_grad()
def intervene_feature(model, idx: torch.Tensor, tcs: list[Transcoder], layer: int, pos: int, feature: int,
                      value: float = 0.0) -> torch.Tensor:
    """Logits of the *original* model with one feature's write removed from (or set in) one MLP output:
    ``mlp_out ← mlp_out + (value − a) · W_dec[feature] · out_scale`` at ``pos``. Attention, norms and all
    other MLPs recompute."""
    tc = tcs[layer]
    _, a = HK.capture(model, idx, [f"mlp_in.{layer}"])
    z = tc.encode(a[f"mlp_in.{layer}"])
    d = (value - z[:, pos, feature])[:, None] * tc.W_dec[feature] * tc.out_scale

    def g(y):
        y = y.clone()
        y[:, pos] = y[:, pos] + d.to(y.dtype)
        return y
    return HK.run_with(model, idx, {f"mlp_out.{layer}": g})


@torch.no_grad()
def predicted_feature_effect(model, idx, tcs, layer: int, pos: int, feature: int, value: float = 0.0):
    """The local replacement model's logits for the same intervention (frozen attention and norms, errors
    fixed, downstream features recomputed through their nonlinearity)."""
    rec = trace(model, idx)
    _, errs = clean_codes(rec, tcs)
    logits, _ = replacement_logits(model, rec, tcs, errors=errs, recompute_features=True,
                                   edits={(layer, pos, feature): value})
    return logits


# --------------------------------------------------------------------------------------------- the replacement model

@torch.no_grad()
def replaced_logits(model, idx: torch.Tensor, tcs: list[Transcoder], layers=None) -> torch.Tensor:
    """Logits with the MLPs of ``layers`` (all by default) replaced by their transcoders, *without* error terms,
    attention and norms recomputed: the (global) replacement model of Circuit Tracing."""
    layers = range(len(tcs)) if layers is None else layers
    handles = []
    try:
        for l in layers:
            tc = tcs[l]
            handles.append(HK.decoder_layers(model)[l].mlp.register_forward_hook(
                lambda mod, args, out, tc=tc: tc(args[0]).to(out.dtype)))
        return HK.logits_of(model, idx)
    finally:
        for h in handles:
            h.remove()


def target_logit(logits: torch.Tensor, pos: int, token: int) -> float:
    """The quantity a logit node explains: the token's logit minus the mean logit at ``pos`` (prompt 0)."""
    row = logits[0, pos]
    return float(row[token] - row.mean())
