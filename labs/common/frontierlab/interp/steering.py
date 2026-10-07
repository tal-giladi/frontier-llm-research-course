"""Steering and persona vectors, from scratch (lesson 17.4). Harmless traits only: style, topic, sycophancy.

A **steering vector** for a trait is a direction ``v`` in the residual stream at one layer. The course
extracts it the way contrastive activation addition (CAA, Panickssery et al. 2023) and persona vectors
(Chen et al. 2025) do, as a *difference of means*:

    v = mean_{x ∈ with trait} h_L(x) − mean_{x ∈ without trait} h_L(x)

at a fixed read position (the answer letter of an A/B item, or the mean over response tokens). Steering
adds ``α · v`` to the residual stream at layer L at every position from ``start`` on; **monitoring**
projects activations onto ``v̂``.

Every steering result in the course reports four things, each on prompts *not* used to extract ``v``:

1. **effect** — the trait score as a function of α (here: the log-odds of the trait answer of an A/B item);
2. **control** — the same α·‖v‖ along random directions, and a vector extracted with shuffled labels;
3. **side effects** — next-token loss and KL to the unsteered model on neutral text, and the score on an
   unrelated task;
4. **generalisation** — the effect on a held-out template family.

Out of scope, by the course's publishing rule: vectors that reduce refusals or any safety behaviour. The
functions are generic; the course's datasets (:mod:`frontierlab.interp.persona`) contain only harmless traits.
"""

from __future__ import annotations

import numpy as np
import torch

from frontierlab.interp import hooks as HK
from frontierlab.stats import bootstrap_ci


def _batches(model, seqs, batch):
    """Yield (ids, mask, rows) for left-padded batches (HF models), or single sequences for the course's LM."""
    hf = hasattr(model, "generate") and not hasattr(model, "new_cache")
    if not hf:
        for i, s in enumerate(seqs):
            yield s[None], None, [i]
        return
    for i in range(0, len(seqs), batch):
        ids, mask = HK.left_pad(seqs[i:i + batch])
        yield ids, mask, list(range(i, min(i + batch, len(seqs))))


@torch.no_grad()
def read(model, seqs: list[torch.Tensor], layers, pos: int | str = -1, batch: int = 8) -> dict:
    """Residual activations after each layer in ``layers`` for a list of 1-D id tensors (any lengths):
    {layer: (N, C)}. ``pos`` is ``-1`` (last token) or ``"mean"`` (mean over the real tokens)."""
    layers = [layers] if isinstance(layers, int) else list(layers)
    out = {L: [None] * len(seqs) for L in layers}
    for ids, mask, rows in _batches(model, seqs, batch):
        _, a = HK.capture(model, ids, [f"resid_post.{L}" for L in layers], attention_mask=mask)
        for L in layers:
            h = a[f"resid_post.{L}"].float()
            for j, r in enumerate(rows):
                if pos == "mean":
                    m = mask[j].bool() if mask is not None else torch.ones(h.shape[1], dtype=torch.bool)
                    out[L][r] = h[j][m].mean(0)
                else:
                    out[L][r] = h[j, pos]
    return {L: torch.stack(v) for L, v in out.items()}


def mean_diff(pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> torch.Tensor:
    """Difference of means (C,)."""
    return pos_acts.float().mean(0) - neg_acts.float().mean(0)


def shuffled_control(pos_acts: torch.Tensor, neg_acts: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """The same estimator with the trait labels permuted: what the mean difference looks like by chance."""
    allx = torch.cat([pos_acts, neg_acts]).float()
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(allx), generator=g)
    a, b = allx[perm[:len(pos_acts)]], allx[perm[len(pos_acts):]]
    return a.mean(0) - b.mean(0)


def steer_edit(vec: torch.Tensor, alpha: float, start: int = 0):
    """Edit for ``resid_post.L``: add ``alpha · vec`` at positions ``start`` onward."""
    def f(x):
        y = x.clone()
        y[:, start:] = y[:, start:] + (alpha * vec).to(y.dtype).to(y.device)
        return y
    return f


@torch.no_grad()
def steered_logits(model, ids: torch.Tensor, layer: int, vec: torch.Tensor | None, alpha: float, start: int = 0,
                   attention_mask: torch.Tensor | None = None):
    edits = {} if vec is None or alpha == 0 else {f"resid_post.{layer}": steer_edit(vec, alpha, start)}
    return HK.run_with(model, ids, edits, attention_mask)


def projection(acts: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Scalar projection of each row on the unit vector along ``vec``."""
    return acts.float() @ (vec.float() / vec.float().norm())


@torch.no_grad()
def ab_logodds(model, items, layer: int | None = None, vec: torch.Tensor | None = None, alpha: float = 0.0,
               batch: int = 8):
    """Per-item log-odds of the trait answer: ``log p(trait letter) − log p(other letter)`` at the last
    position. ``items`` are dicts with ``ids`` (1-D), ``trait_id`` and ``other_id``. Steering, if given,
    is applied at every position (the whole prompt is the context the answer reads; with left padding the
    padded positions are masked as keys, so adding to them changes nothing)."""
    out = np.zeros(len(items))
    for ids, mask, rows in _batches(model, [it["ids"] for it in items], batch):
        lg = steered_logits(model, ids, layer, vec, alpha, attention_mask=mask)[:, -1]
        lp = torch.log_softmax(lg.float(), -1)
        for j, r in enumerate(rows):
            out[r] = float(lp[j, items[r]["trait_id"]] - lp[j, items[r]["other_id"]])
    return out


@torch.no_grad()
def side_effects(model, windows: torch.Tensor, layer: int, vec: torch.Tensor | None, alpha: float,
                 batch: int = 8) -> dict:
    """Mean next-token loss with steering, its increase over no steering, and the mean per-token KL from
    the unsteered next-token distribution, on neutral text ``windows`` (N, T)."""
    nll, nll0, kl = [], [], []
    for i in range(0, len(windows), batch):
        x = windows[i:i + batch]
        l0 = HK.run_with(model, x)
        l1 = steered_logits(model, x, layer, vec, alpha)
        lp0, lp1 = torch.log_softmax(l0[:, :-1], -1), torch.log_softmax(l1[:, :-1], -1)
        tgt = x[:, 1:, None]
        nll0.extend((-lp0.gather(-1, tgt)[..., 0].mean(1)).tolist())
        nll.extend((-lp1.gather(-1, tgt)[..., 0].mean(1)).tolist())
        kl.extend((lp0.exp() * (lp0 - lp1)).sum(-1).mean(1).tolist())
    nll, nll0, kl = map(np.array, (nll, nll0, kl))
    d = nll - nll0
    m, lo, hi = bootstrap_ci(d)
    return {"loss": float(nll.mean()), "loss_increase": m, "loss_increase_ci": (lo, hi), "kl": float(kl.mean())}


def dose_response(model, items, layer: int, vec: torch.Tensor, alphas, base=None) -> list[dict]:
    """Mean change in trait log-odds (with a bootstrap interval over items) for each α."""
    base = ab_logodds(model, items) if base is None else base
    rows = []
    for a in alphas:
        s = ab_logodds(model, items, layer, vec, a)
        d = s - base
        m, lo, hi = bootstrap_ci(d)
        rows.append({"alpha": float(a), "delta_logodds": m, "ci": (lo, hi), "trait_rate": float((s > 0).mean()),
                     "flip_rate": float(((s > 0) != (base > 0)).mean())})
    return rows


def random_like(vec: torch.Tensor, n: int, seed: int = 0) -> torch.Tensor:
    """(n, C) random directions with the same norm as ``vec``."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(n, vec.numel(), generator=g, dtype=torch.float64)
    return (r / r.norm(dim=1, keepdim=True) * float(vec.norm())).to(vec.dtype)


@torch.no_grad()
def generate(model, tok, prompt_ids: torch.Tensor, layer: int | None = None, vec=None, alpha: float = 0.0,
             max_new_tokens: int = 40) -> str:
    """Greedy generation (Hugging Face ``generate`` with the cache) with ``alpha · vec`` added at every
    position, prompt and generated tokens alike. For qualitative inspection only: the measured effect is
    always the A/B log-odds on held-out items."""
    edits = {} if vec is None or alpha == 0 else {f"resid_post.{layer}": steer_edit(vec, alpha, 0)}
    with HK.hooks(model, edits):
        out = model.generate(prompt_ids[None], max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", 0))
    return tok.decode(out[0, prompt_ids.numel():].tolist(), skip_special_tokens=True)
