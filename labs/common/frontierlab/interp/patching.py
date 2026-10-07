"""Causal interventions on activations, from scratch (lesson 17.2).

Vocabulary (Heimersheim & Nanda 2024; Zhang & Nanda 2023):

* **Denoising** (restoring): run the *corrupt* prompt, copy one activation in from the *clean* run, and
  measure how much of the clean behaviour comes back. A large effect says the component is *sufficient*
  (together with everything downstream) to carry the information that differs between the prompts.
* **Noising**: run the *clean* prompt, copy one activation in from the *corrupt* run, and measure how much
  behaviour is lost. A large effect says the component is *necessary* on this distribution.
* **Normalised effect** of a patch: ``(m_patched - m_corrupt) / (m_clean - m_corrupt)`` for denoising —
  0 = nothing restored, 1 = fully restored; ``(m_clean - m_patched) / (m_clean - m_corrupt)`` for noising.
* **Ablation** replaces an activation with zero, its mean over a reference set, or its value on a
  different prompt (resample). Zero ablation pushes the model off its training distribution; mean and
  resample ablation stay closer to it.
* **Path patching** (Wang et al. 2022): the effect of a sender only along the direct path into one
  receiver. The residual stream is a sum, so the sender's change ``Δ`` can be added to the receiver's
  input alone while every other component reads the clean residual (:func:`path_patch`).
* **Attribution patching** (Nanda 2023; Syed et al. 2023): the first-order estimate
  ``Δm ≈ (a_clean − a_corrupt) · ∂m/∂a`` from one forward and one backward pass. Cheap, approximate,
  and only a way to choose what to patch for real.

Controls (what makes an effect a claim rather than a number): the same intervention on random directions
of matched norm (:func:`direction_control`) and on unrelated components (:func:`component_control`),
the effect on held-out prompts (other templates, other tokens), and the effect on off-target behaviour
(next-token loss on ordinary text, :func:`offtarget_loss`). Every effect is reported per prompt so that a
bootstrap interval over prompts can be computed (``frontierlab.stats``).
"""

from __future__ import annotations

import numpy as np
import torch

from frontierlab.interp import hooks as HK
from frontierlab.interp.tasks import Pairs
from frontierlab.stats import bootstrap_ci


# --------------------------------------------------------------------------------------------- metrics

def logit_diff(logits: torch.Tensor, pos: int, good: torch.Tensor, bad: torch.Tensor) -> torch.Tensor:
    """Per-prompt ``logit[good] - logit[bad]`` at position ``pos``; logits (B, T, V) -> (B,)."""
    row = logits[:, pos]
    ar = torch.arange(row.shape[0])
    return row[ar, good] - row[ar, bad]


def metric_fn(p: Pairs):
    return lambda logits: logit_diff(logits, p.pos, p.good, p.bad)


@torch.no_grad()
def baselines(model, p: Pairs):
    """Clean and corrupt metric per prompt, plus their means."""
    mc = metric_fn(p)(HK.logits_of(model, p.clean))
    mk = metric_fn(p)(HK.logits_of(model, p.corrupt))
    return {"clean": mc, "corrupt": mk, "clean_mean": float(mc.mean()), "corrupt_mean": float(mk.mean())}


def normalised(m_patched: torch.Tensor, m_clean: torch.Tensor, m_corrupt: torch.Tensor, mode: str) -> torch.Tensor:
    """Normalised effect of the *mean* metrics, broadcast per prompt for intervals (see module docstring).
    Normalising by the batch-mean gap (not per prompt) avoids dividing by near-zero per-prompt gaps."""
    gap = (m_clean.mean() - m_corrupt.mean())
    if mode == "denoise":
        return (m_patched - m_corrupt) / gap
    return (m_clean - m_patched) / gap


# --------------------------------------------------------------------------------------------- patching

@torch.no_grad()
def patch(model, p: Pairs, site: str, positions=None, head: int | None = None, mode: str = "denoise",
          base=None) -> torch.Tensor:
    """Per-prompt normalised effect of patching ``site`` (all positions, or ``positions``; one head of a ``z``
    site if ``head`` is given). ``mode`` is ``"denoise"`` (clean -> corrupt run) or ``"noise"``."""
    base = base or baselines(model, p)
    src, dst = (p.clean, p.corrupt) if mode == "denoise" else (p.corrupt, p.clean)
    _, acts = HK.capture(model, src, [site])
    hd = HK.head_dim(model)[1] if head is not None else None
    logits = HK.run_with(model, dst, {site: HK.replace_at(acts[site], positions, head, hd)})
    return normalised(metric_fn(p)(logits), base["clean"], base["corrupt"], mode)


@torch.no_grad()
def layer_position_sweep(model, p: Pairs, kind: str = "resid_pre", positions=None, mode: str = "denoise"):
    """Mean normalised effect of patching ``kind.L`` at one position at a time: array (layers, positions)."""
    base = baselines(model, p)
    T = p.clean.shape[1]
    positions = list(range(T)) if positions is None else positions
    out = np.zeros((HK.n_layers(model), len(positions)))
    src = p.clean if mode == "denoise" else p.corrupt
    _, acts = HK.capture(model, src, [f"{kind}.{L}" for L in range(HK.n_layers(model))])
    dst = p.corrupt if mode == "denoise" else p.clean
    for L in range(HK.n_layers(model)):
        site = f"{kind}.{L}"
        for j, t in enumerate(positions):
            logits = HK.run_with(model, dst, {site: HK.replace_at(acts[site], [t])})
            out[L, j] = float(normalised(metric_fn(p)(logits), base["clean"], base["corrupt"], mode).mean())
    return out


@torch.no_grad()
def head_sweep(model, p: Pairs, mode: str = "denoise", positions=None) -> np.ndarray:
    """Mean normalised effect of patching each attention head's output (the ``z`` slice): (layers, heads)."""
    base = baselines(model, p)
    H, hd = HK.head_dim(model)
    src, dst = (p.clean, p.corrupt) if mode == "denoise" else (p.corrupt, p.clean)
    L = HK.n_layers(model)
    _, acts = HK.capture(model, src, [f"z.{l}" for l in range(L)])
    out = np.zeros((L, H))
    for l in range(L):
        for h in range(H):
            logits = HK.run_with(model, dst, {f"z.{l}": HK.replace_at(acts[f"z.{l}"], positions, h, hd)})
            out[l, h] = float(normalised(metric_fn(p)(logits), base["clean"], base["corrupt"], mode).mean())
    return out


def attribution_heads(model, p: Pairs) -> np.ndarray:
    """Attribution-patching estimate of every head's denoising effect, normalised like :func:`head_sweep`:
    ``Σ_{b,t,i∈head} (z_clean − z_corrupt) · ∂m/∂z`` at the corrupt run, divided by the clean-corrupt gap."""
    H, hd = HK.head_dim(model)
    L = HK.n_layers(model)
    sites = [f"z.{l}" for l in range(L)]
    with torch.no_grad():
        _, clean = HK.capture(model, p.clean, sites)
        base = baselines(model, p)
    with torch.enable_grad():
        logits, acts = HK.capture(model, p.corrupt, sites, detach=False)
        m = metric_fn(p)(logits).mean()
        grads = torch.autograd.grad(m, [acts[s] for s in sites])
    gap = base["clean_mean"] - base["corrupt_mean"]
    out = np.zeros((L, H))
    for l, (s, g) in enumerate(zip(sites, grads)):
        d = (clean[s] - acts[s].detach()) * g                 # (B, T, H·hd); the metric is a batch mean
        out[l] = (d.sum(dim=(0, 1)).view(H, hd).sum(-1) / gap).float().numpy()
    return out


# --------------------------------------------------------------------------------------------- path patching

def _head_delta(model, layer: int, z_from: torch.Tensor, z_to: torch.Tensor, heads) -> torch.Tensor:
    """Change in layer ``layer``'s attention output if only ``heads`` switch from ``z_from`` to ``z_to``:
    ``Σ_h (z_to − z_from)[h] W_O[:, h]ᵀ``, shape (B, T, C)."""
    H, hd = HK.head_dim(model)
    W = HK.decoder_layers(model)[layer].self_attn.o_proj.weight            # (C, H·hd)
    mask = torch.zeros(H * hd, dtype=z_to.dtype)
    for h in heads:
        mask[h * hd:(h + 1) * hd] = 1
    return ((z_to - z_from) * mask) @ W.T.to(z_to.dtype)


@torch.no_grad()
def path_patch(model, p: Pairs, sender_layer: int, sender_heads, receiver: tuple, mode: str = "noise",
               base=None) -> torch.Tensor:
    """Per-prompt normalised effect of the sender heads' change along the direct path into ``receiver``.

    ``receiver`` is ``("q"|"k"|"v", layer)``, ``("q"|"k"|"v", layer, head)`` (a query head, or a key/value
    head with grouped-query attention), ``("mlp_in", layer)`` or ``("logits",)``. ``mode="noise"``: the
    run is clean and the sender takes its corrupt value; ``"denoise"`` the opposite. Only the receiver's
    input sees the change: it reads ``norm(resid_pre_clean + Δ)``; every other component reads the
    unchanged residual. With GQA, a key/value head serves several query heads, so a ``k`` receiver is
    the shared projection."""
    base = base or baselines(model, p)
    run, other = (p.clean, p.corrupt) if mode == "noise" else (p.corrupt, p.clean)
    zs = f"z.{sender_layer}"
    _, a_run = HK.capture(model, run, [zs])
    _, a_oth = HK.capture(model, other, [zs])
    delta = _head_delta(model, sender_layer, a_run[zs], a_oth[zs], sender_heads)
    kind = receiver[0]
    layers = HK.decoder_layers(model)
    if kind == "logits":
        edits = {"final": HK.add_at(delta)}
    elif kind == "mlp_in":
        L = receiver[1]
        assert L > sender_layer, "the receiver must come after the sender"
        norm = layers[L].post_attention_layernorm
        stash = {}
        edits = {f"attn_out.{L}": lambda x: x}       # placeholder so the dict stays uniform

        def mlp_pre(x):                              # x = post_attention_layernorm(h); rebuild from h + Δ
            return norm(stash["h"] + delta.to(x.dtype))
        # h (the residual after attention) = resid_pre + attn_out
        _, cap = HK.capture(model, run, [f"resid_pre.{L}", f"attn_out.{L}"])
        stash["h"] = cap[f"resid_pre.{L}"] + cap[f"attn_out.{L}"]
        edits = {f"mlp_in.{L}": mlp_pre}
    elif kind in ("q", "k", "v"):
        L = receiver[1]
        assert L > sender_layer, "the receiver must come after the sender"
        head = receiver[2] if len(receiver) > 2 else None
        _, cap = HK.capture(model, run, [f"resid_pre.{L}"])
        layer = layers[L]
        proj = getattr(layer.self_attn, f"{kind}_proj")
        new_in = layer.input_layernorm(cap[f"resid_pre.{L}"] + delta)
        new_out = proj(new_in)
        hd = HK.head_dim(model)[1]

        def post(mod, args, out):
            if head is None:
                return new_out.to(out.dtype)
            y = out.clone()
            y[..., head * hd:(head + 1) * hd] = new_out[..., head * hd:(head + 1) * hd].to(y.dtype)
            return y
        h = proj.register_forward_hook(post)
        try:
            logits = HK.logits_of(model, run)
        finally:
            h.remove()
        return normalised(metric_fn(p)(logits), base["clean"], base["corrupt"], mode)
    else:
        raise ValueError(f"unknown receiver {receiver!r}")
    logits = HK.run_with(model, run, edits)
    return normalised(metric_fn(p)(logits), base["clean"], base["corrupt"], mode)


# --------------------------------------------------------------------------------------------- ablations

@torch.no_grad()
def ablate_heads(model, idx: torch.Tensor, heads: dict[int, list[int]], how: str = "mean",
                 reference: torch.Tensor | None = None, gen: torch.Generator | None = None) -> torch.Tensor:
    """Logits with the listed heads ablated: ``heads = {layer: [h, ...]}``. ``how``: ``"zero"``; ``"mean"``
    (each head's mean output over ``reference`` prompts and positions); ``"resample"`` (each prompt gets
    the head output of a random other prompt of ``reference``, same positions)."""
    H, hd = HK.head_dim(model)
    ref = idx if reference is None else reference
    sites = [f"z.{l}" for l in heads]
    _, acts = HK.capture(model, ref, sites) if how != "zero" else (None, {})
    edits = {}
    for l, hs in heads.items():
        s = f"z.{l}"

        def f(x, s=s, hs=hs):
            y = x.clone()
            for h in hs:
                sl = slice(h * hd, (h + 1) * hd)
                if how == "zero":
                    y[..., sl] = 0
                elif how == "mean":
                    y[..., sl] = acts[s][..., sl].mean(dim=(0, 1)).to(y.dtype)
                elif how == "resample":
                    perm = torch.randint(0, acts[s].shape[0], (y.shape[0],), generator=gen)
                    y[..., sl] = acts[s][perm][:, : y.shape[1], sl].to(y.dtype)
                else:
                    raise ValueError(how)
            return y
        edits[s] = f
    return HK.run_with(model, idx, edits)


def ablation_effect(model, p: Pairs, heads: dict[int, list[int]], how: str = "mean",
                    reference: torch.Tensor | None = None, seed: int = 0) -> torch.Tensor:
    """Per-prompt fraction of the clean metric lost when ``heads`` are ablated on the clean prompts:
    ``(m_clean - m_ablated) / mean(m_clean - m_corrupt)``."""
    base = baselines(model, p)
    gen = torch.Generator().manual_seed(seed)
    logits = ablate_heads(model, p.clean, heads, how, reference if reference is not None else p.corrupt, gen)
    return normalised(metric_fn(p)(logits), base["clean"], base["corrupt"], "noise")


# --------------------------------------------------------------------------------------------- controls

def random_unit(d: int, n: int, seed: int = 0, dtype=torch.float32) -> torch.Tensor:
    """(n, d) random unit vectors (uniform on the sphere)."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, d, generator=g, dtype=torch.float64)
    return (v / v.norm(dim=1, keepdim=True)).to(dtype)


@torch.no_grad()
def direction_effect(model, p: Pairs, site: str, direction: torch.Tensor, positions=None) -> torch.Tensor:
    """Per-prompt fraction of the clean metric lost when ``direction`` is projected out of ``site`` on the
    clean prompts (normalised by the clean-corrupt gap)."""
    base = baselines(model, p)
    logits = HK.run_with(model, p.clean, {site: HK.project_out(direction, positions)})
    return normalised(metric_fn(p)(logits), base["clean"], base["corrupt"], "noise")


def direction_control(model, p: Pairs, site: str, direction: torch.Tensor, n_random: int = 32,
                      positions=None, seed: int = 0) -> dict:
    """The candidate direction's effect against ``n_random`` random directions in the same site.

    Returns the candidate's mean effect with a bootstrap 95% interval over prompts, the random effects'
    mean and 95th percentile, and the candidate's rank among them (``p_random`` = fraction of random
    directions with an effect at least as large; with 32 random directions the smallest possible
    value is 1/33)."""
    eff = direction_effect(model, p, site, direction, positions)
    m, lo, hi = bootstrap_ci(eff.numpy())
    rnd = random_unit(direction.numel(), n_random, seed, direction.dtype)
    r = np.array([float(direction_effect(model, p, site, u, positions).mean()) for u in rnd])
    return {"effect": m, "ci": (lo, hi), "random_mean": float(r.mean()), "random_p95": float(np.quantile(r, 0.95)),
            "random": r.tolist(), "p_random": float((1 + (r >= m).sum()) / (1 + len(r)))}


def component_control(model, p: Pairs, candidate: dict[int, list[int]], how: str = "mean", n_random: int = 20,
                      seed: int = 0, exclude: dict[int, list[int]] | None = None, layer_matched: bool = False) -> dict:
    """Ablate the candidate heads, then ``n_random`` random sets of the same size drawn from the other heads;
    with ``layer_matched`` the same number of heads in each of the candidate's layers. A candidate whose effect
    is inside the random distribution is not specific."""
    H, _ = HK.head_dim(model)
    L = HK.n_layers(model)
    eff = ablation_effect(model, p, candidate, how)
    m, lo, hi = bootstrap_ci(eff.numpy())
    taken = {(l, h) for l, hs in candidate.items() for h in hs} | {(l, h) for l, hs in (exclude or {}).items() for h in hs}
    pool = [(l, h) for l in range(L) for h in range(H) if (l, h) not in taken]
    rng = np.random.default_rng(seed)
    k = len({(l, h) for l, hs in candidate.items() for h in hs})
    r = []
    for _ in range(n_random):
        hs: dict[int, list[int]] = {}
        if layer_matched:
            for l, ch in candidate.items():
                lp = [h for (ll, h) in pool if ll == l]
                hs[l] = [int(h) for h in rng.choice(lp, size=min(len(ch), len(lp)), replace=False)]
        else:
            pick = [pool[i] for i in rng.choice(len(pool), size=min(k, len(pool)), replace=False)]
            for l, h in pick:
                hs.setdefault(l, []).append(h)
        r.append(float(ablation_effect(model, p, hs, how).mean()))
    r = np.array(r)
    return {"effect": m, "ci": (lo, hi), "random_mean": float(r.mean()), "random_p95": float(np.quantile(r, 0.95)),
            "random": r.tolist(), "p_random": float((1 + (r >= m).sum()) / (1 + len(r)))}


@torch.no_grad()
def offtarget_loss(model, windows: torch.Tensor, edits: dict | None = None, batch: int = 16) -> np.ndarray:
    """Per-window next-token loss on ordinary text with ``edits`` applied (the off-target check: an
    intervention that is 'specific' should not change this much)."""
    out = []
    for i in range(0, len(windows), batch):
        x = windows[i:i + batch]
        logits = HK.run_with(model, x, edits or {})
        lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, x[:, 1:, None])[..., 0]
        out.extend((-lp.mean(1)).tolist())
    return np.array(out)


def summarise(effects: torch.Tensor) -> dict:
    m, lo, hi = bootstrap_ci(np.asarray(effects, dtype=np.float64))
    return {"mean": m, "ci": (lo, hi), "n": int(len(effects))}
