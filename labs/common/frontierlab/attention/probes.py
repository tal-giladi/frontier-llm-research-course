"""Probes for lesson 03.3: where attention mass goes, how large logits get, and massive activations.

All probes run a full-sequence forward on fixed inputs (no cache, eval mode, no gradients) and capture
each layer's attention input with a forward pre-hook, then recompute that layer's logits and
probabilities from its own projections. Supported: every GQA-family kind (Baseline-0's ``gqa`` and the
Module 3 kinds, which expose ``attention_probs``) and ``mla``.

Definitions used in the course:

* **first-token mass** — the average, over layers, heads and query positions t >= ``skip``, of the
  probability a query puts on the key at position 0 of the window. StreamingLLM calls tokens that
  collect such mass, regardless of their meaning, attention sinks (arXiv 2309.17453).
* **learned-sink mass** — for the ``sink`` kind, 1 minus the row sum of the token probabilities: the
  weight the head gave to "nothing".
* **max logit** — the largest pre-softmax score q·k·scale in a layer, over the batch, heads and allowed
  pairs. QK-norm bounds its growth (lesson 03.3).
* **massive-activation ratio** — per layer output (the residual stream after the block), the largest
  absolute entry divided by the median absolute entry. Sun et al. (arXiv 2402.17762) describe a few
  activations "100,000 times larger" than the rest in large models; at course scale expect far smaller
  ratios, and compare arms, not absolute values.
"""

from __future__ import annotations

import torch

from frontierlab.attention.base import apply_rope


def _gqa_probs(mod, x, positions):
    """Probabilities and logits of Baseline-0's GQAttention (no window, no sink)."""
    from frontierlab.attention.ops import attention_logits
    B, T, _ = x.shape
    q = mod.q_norm(mod.q_proj(x).view(B, T, mod.H, mod.hd)).transpose(1, 2)
    k = mod.k_norm(mod.k_proj(x).view(B, T, mod.KV, mod.hd)).transpose(1, 2)
    cos, sin = mod.rope(positions)
    logits = attention_logits(apply_rope(q, cos, sin), apply_rope(k, cos, sin), positions, positions)
    return torch.softmax(logits.float(), -1), logits


def layer_probs(mod, x, positions):
    if hasattr(mod, "attention_probs"):
        return mod.attention_probs(x, positions)
    return _gqa_probs(mod, x, positions)


@torch.no_grad()
def capture(model, idx: torch.Tensor) -> dict:
    """Run ``model`` on ``idx`` (B, T) and return per-layer probs, logits and block outputs.

    Returns {"probs": [(B, H, T, T) float32], "logits": [(B, H, T, T)], "hidden": [(B, T, C)]}. Memory is
    O(layers · B · H · T²): keep B·T small (the lab uses B = 8, T = 128).
    """
    was = model.training
    model.eval()
    inputs, hidden, hooks = {}, {}, []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_pre_hook(
            lambda m, args, i=i: inputs.__setitem__(i, (args[0], args[1]))))
        hooks.append(layer.register_forward_hook(lambda m, args, out, i=i: hidden.__setitem__(i, out)))
    try:
        model(idx)
    finally:
        for h in hooks:
            h.remove()
    model.train(was)
    out = {"probs": [], "logits": [], "hidden": []}
    for i, layer in enumerate(model.model.layers):
        x, pos = inputs[i]
        p, lg = layer_probs(layer.self_attn, x, pos)
        out["probs"].append(p)
        out["logits"].append(lg)
        out["hidden"].append(hidden[i])
    return out


def first_token_mass(probs: torch.Tensor, skip: int = 8) -> float:
    """Mean probability on key 0 over batch, heads and queries t >= skip. probs (B, H, T, T)."""
    return probs[:, :, skip:, 0].float().mean().item()


def learned_sink_mass(probs: torch.Tensor, skip: int = 8) -> float:
    """1 - row sum, averaged like first_token_mass (zero for kinds without a learned sink)."""
    return (1 - probs[:, :, skip:].float().sum(-1)).mean().item()


def max_logit(logits: torch.Tensor) -> float:
    """Largest finite pre-softmax score (masked entries are -inf)."""
    return logits.float().masked_fill(~torch.isfinite(logits), float("-inf")).max().item()


def massive_ratio(h: torch.Tensor) -> float:
    """max |h| / median |h| over every entry of one layer's output (B, T, C)."""
    a = h.float().abs().flatten()
    return (a.max() / a.median().clamp_min(1e-12)).item()


def report(model, idx: torch.Tensor, skip: int = 8) -> dict:
    """All probes, per layer and summarised (means over layers; max over layers for logits and ratios)."""
    cap = capture(model, idx)
    per = [{"layer": i, "first_token_mass": first_token_mass(p, skip), "sink_mass": learned_sink_mass(p, skip),
            "max_logit": max_logit(lg), "massive_ratio": massive_ratio(h),
            "argmax_pos": int(h.float().abs().amax(-1).argmax(-1).mode().values.item())}
           for i, (p, lg, h) in enumerate(zip(cap["probs"], cap["logits"], cap["hidden"]))]
    n = len(per)
    return {"layers": per,
            "first_token_mass": sum(r["first_token_mass"] for r in per) / n,
            "sink_mass": sum(r["sink_mass"] for r in per) / n,
            "max_logit": max(r["max_logit"] for r in per),
            "massive_ratio": max(r["massive_ratio"] for r in per),
            "uniform_first_token_mass": _uniform_reference(idx.shape[1], skip)}


def _uniform_reference(T: int, skip: int) -> float:
    """First-token mass if every query spread its attention uniformly over its causal prefix."""
    return sum(1 / (t + 1) for t in range(skip, T)) / (T - skip)
