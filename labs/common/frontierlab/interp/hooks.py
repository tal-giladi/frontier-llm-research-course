"""Reading and editing activations by name, from scratch, with PyTorch forward hooks (Module 17).

One addressing scheme serves the course's own models (``frontierlab.model.LM``, Baseline-0 and its toy
sizes) and Hugging Face Qwen3/Llama-style models (``Qwen3ForCausalLM``): both keep their decoder layers at
``model.model.layers`` with ``input_layernorm``, ``self_attn.{q,k,v,o}_proj``, ``post_attention_layernorm``
and ``mlp``, and a final ``model.model.norm``. A *site* is a string:

=================  ==========================================  =========================
site               what it is                                  shape (B, T, ·)
=================  ==========================================  =========================
``resid_pre.L``    residual stream entering layer L            (B, T, C)
``resid_post.L``   residual stream leaving layer L             (B, T, C)
``attn_out.L``     output of layer L's attention (after o_proj) (B, T, C)
``z.L``            input of o_proj: every head's output        (B, T, H·d) — head h is [h·d, (h+1)·d)
``mlp_in.L``       input of layer L's MLP (after its norm)     (B, T, C)
``mlp_out.L``      output of layer L's MLP                     (B, T, C)
``final``          input of the final norm (= resid_post.last) (B, T, C)
=================  ==========================================  =========================

``capture`` returns copies of the activations; ``run_with`` runs the model with edit functions
``f(activation) -> activation`` applied at sites. Nothing is monkey-patched: hooks are registered for one
call and always removed (also on an exception), so a forgotten hook cannot leak into the next experiment.
That leak is the most common silent bug in interpretability code, so the tests check it.

Hooks run on whatever device and dtype the model runs on; the edit functions must return a tensor of the
same shape. Under bf16 autocast, activations are bf16: compare effects in float32.
"""

from __future__ import annotations

import contextlib
from typing import Callable

import torch
import torch.nn as nn

Edit = Callable[[torch.Tensor], torch.Tensor]
KINDS = ("resid_pre", "resid_post", "attn_out", "z", "mlp_in", "mlp_out", "final")


def decoder_layers(model: nn.Module):
    """The list of decoder layers (``model.model.layers`` for both model families)."""
    return model.model.layers


def n_layers(model: nn.Module) -> int:
    return len(decoder_layers(model))


def head_dim(model: nn.Module) -> tuple[int, int]:
    """(number of query heads, head dimension), read from the attention module or the config."""
    attn = decoder_layers(model)[0].self_attn
    if hasattr(attn, "H"):
        return attn.H, attn.hd
    cfg = model.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return cfg.num_attention_heads, hd


def parse(site: str) -> tuple[str, int | None]:
    if site == "final":
        return "final", None
    kind, _, layer = site.partition(".")
    if kind not in KINDS or not layer.lstrip("-").isdigit():
        raise ValueError(f"unknown site {site!r}; use one of {KINDS} with a layer, e.g. 'resid_post.3'")
    return kind, int(layer)


def _module_and_mode(model, site: str):
    """The module to hook and whether the activation is its input ('pre') or output ('post')."""
    kind, L = parse(site)
    layers = decoder_layers(model)
    if L is not None and L < 0:
        L += len(layers)
    if kind == "final":
        return model.model.norm, "pre"
    layer = layers[L]
    return {"resid_pre": (layer, "pre"), "resid_post": (layer, "post"), "attn_out": (layer.self_attn, "post"),
            "z": (layer.self_attn.o_proj, "pre"), "mlp_in": (layer.mlp, "pre"),
            "mlp_out": (layer.mlp, "post")}[kind]


def _first(x):
    return x[0] if isinstance(x, tuple) else x


def _replace_first(x, new):
    return (new,) + tuple(x[1:]) if isinstance(x, tuple) else new


def _hook(module, mode: str, fn: Edit):
    """Register a hook that replaces the activation with ``fn(activation)``; returns the handle."""
    if mode == "pre":
        def pre(mod, args, kwargs):
            if args:
                return (fn(args[0]),) + tuple(args[1:]), kwargs
            key = "hidden_states" if "hidden_states" in kwargs else next(iter(kwargs))
            kwargs = dict(kwargs)
            kwargs[key] = fn(kwargs[key])
            return args, kwargs
        return module.register_forward_pre_hook(pre, with_kwargs=True)

    def post(mod, args, out):
        return _replace_first(out, fn(_first(out)))
    return module.register_forward_hook(post)


@contextlib.contextmanager
def hooks(model: nn.Module, edits: dict[str, Edit]):
    """Context manager: every forward inside runs with ``edits`` applied. Hooks are removed on exit."""
    handles = []
    try:
        for site, fn in edits.items():
            mod, mode = _module_and_mode(model, site)
            handles.append(_hook(mod, mode, fn))
        yield
    finally:
        for h in handles:
            h.remove()


def logits_of(model: nn.Module, idx: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Logits (B, T, V) for either model family: half precision is upcast to float32, float64 stays float64.
    ``attention_mask`` (Hugging Face models only) allows left-padded batches."""
    out = model(idx) if attention_mask is None else model(idx, attention_mask=attention_mask)
    lg = out.logits if hasattr(out, "logits") else out[0]
    return lg.float() if lg.dtype in (torch.float16, torch.bfloat16) else lg


def run_with(model: nn.Module, idx: torch.Tensor, edits: dict[str, Edit] | None = None,
             attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Logits of ``model(idx)`` with ``edits`` applied (no gradient tracking unless the caller enables it)."""
    with hooks(model, edits or {}):
        return logits_of(model, idx, attention_mask)


def capture(model: nn.Module, idx: torch.Tensor, sites, edits: dict[str, Edit] | None = None,
            detach: bool = True, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run once and return (logits, {site: activation}). With ``detach=False`` the activations stay in the
    graph (attribution patching differentiates the metric with respect to them)."""
    acts: dict[str, torch.Tensor] = {}

    def saver(name):
        def f(x):
            acts[name] = x.detach().clone() if detach else x
            return x
        return f

    all_edits = dict(edits or {})
    for s in sites:
        if s in all_edits:                      # capture *after* the edit at the same site
            inner = all_edits[s]
            all_edits[s] = (lambda inner, sv: (lambda x: sv(inner(x))))(inner, saver(s))
        else:
            all_edits[s] = saver(s)
    with hooks(model, all_edits):
        logits = logits_of(model, idx, attention_mask)
    return logits, acts


def n_hooks(model: nn.Module) -> int:
    """Total forward hooks registered on the model (0 when nothing has leaked)."""
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


# ------------------------------------------------------------------------------------- edit builders

def replace_at(new: torch.Tensor, positions=None, head: int | None = None, hd: int | None = None) -> Edit:
    """An edit that copies ``new`` (same shape as the activation) in at ``positions`` (all if None), and only
    into head ``head``'s slice ``[head·hd, (head+1)·hd)`` of the last dimension when given (``z`` sites)."""
    def f(x):
        y = x.clone()
        sl = slice(None) if head is None else slice(head * hd, (head + 1) * hd)
        if positions is None:
            y[..., sl] = new[..., sl].to(y.dtype)
        else:
            y[:, positions, sl] = new[:, positions, sl].to(y.dtype)
        return y
    return f


def add_at(vec: torch.Tensor, positions=None) -> Edit:
    """Add ``vec`` (C,) or (B, T, C) to the activation at ``positions`` (all if None)."""
    def f(x):
        y = x.clone()
        v = vec.to(y.dtype).to(y.device)
        if positions is None:
            y = y + v
        else:
            y[:, positions] = y[:, positions] + (v if v.dim() == 1 else v[:, positions])
        return y
    return f


def project_out(direction: torch.Tensor, positions=None) -> Edit:
    """Remove the component along ``direction`` (C,): x - (x·u)u with u = direction/|direction|."""
    u = direction / direction.norm()

    def f(x):
        uu = u.to(x.dtype).to(x.device)
        y = x.clone()
        sel = y if positions is None else y[:, positions]
        sel = sel - (sel @ uu)[..., None] * uu
        if positions is None:
            return sel
        y[:, positions] = sel
        return y
    return f


def compose(*fns: Edit) -> Edit:
    def f(x):
        for g in fns:
            x = g(x)
        return x
    return f


def left_pad(seqs: list[torch.Tensor], pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad 1-D id tensors into (B, T) ids and a (B, T) attention mask, so that position -1 is every
    sequence's last real token. Hugging Face Qwen3/Llama models give the same logits for the real tokens as
    unpadded runs (RoPE depends only on relative positions; padded keys are masked)."""
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, T - len(s):] = s
        mask[i, T - len(s):] = 1
    return ids, mask
