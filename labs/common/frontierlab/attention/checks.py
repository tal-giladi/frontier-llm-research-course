"""The architecture correctness suite applied to one attention kind (Module 3 onward).

:func:`attention_suite` runs, for a config whose ``attention`` is any registered kind:

1. **op-level float64 gradient check** of one attention block, with respect to its input *and* every
   parameter (``torch.func.functional_call`` turns the parameters into gradcheck inputs);
2. **causal check** on a small model (changing future tokens leaves earlier logits unchanged);
3. **cached-decode agreement**, one token per step and in chunks (chunks crossing window edges);
4. for MLA, **naive vs weight-absorbed equivalence** of the full forward and of cached decoding.

``frontierlab.layers.RMSNorm`` normalises in float32 even for float64 inputs (``x.float()``). That
makes a float64 gradient check fail by about 1e-7 for reasons unrelated to the attention code, and it
silently rounds every model-level float64 comparison to float32 resolution at each norm: two paths that
differ by 1e-10 can produce bit-identical logits. :func:`exact_rmsnorm` makes RMSNorm compute in the
input dtype for the duration of the checks (proposed as a permanent fix in
``curriculum/inbox/module-03-shared-changes.md``); every check in :func:`attention_suite` runs inside it.
"""

from __future__ import annotations

import contextlib

import torch
from torch.func import functional_call

from frontierlab.testing import cache_agreement, causal_check, equivalence, grad_check


@contextlib.contextmanager
def exact_rmsnorm():
    from frontierlab.layers import rmsnorm
    orig = rmsnorm.RMSNorm._norm

    def _norm(self, x):
        dt = torch.promote_types(x.dtype, torch.float32)
        xd = x.to(dt)
        return (xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)

    rmsnorm.RMSNorm._norm = _norm
    try:
        yield
    finally:
        rmsnorm.RMSNorm._norm = orig


@torch.no_grad()
def sharpen(model, std: float = 0.2, seed: int = 0):
    """Re-initialise every Linear weight with N(0, std²) (and sinks uniformly in [-1, 1]).

    At the course's initialisation (std 0.02) attention is almost uniform and each attention output is
    tiny next to the residual stream, so a wrong mask or a wrong absorbed path can change the logits by
    less than float64 rounding. Checks run on a sharpened copy, where scores are O(1) and peaked.
    """
    g = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        if p.dim() == 2 and "embed" not in name:
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * std)
        elif name.endswith("sinks"):
            p.copy_(torch.rand(p.shape, generator=g, dtype=p.dtype) * 2 - 1)
    return model


def op_gradcheck(cfg, layer_idx: int = 0, T: int = 6, seed: int = 0) -> bool:
    """float64 gradcheck of one attention block w.r.t. its input and all its parameters (use a tiny cfg)."""
    from frontierlab.attention.accounting import attention_module
    torch.manual_seed(seed)
    mod = attention_module(cfg, layer_idx).double()
    for p in mod.parameters():                    # move zero-initialised sinks off their initial value
        if p.dim() == 1 and p.numel() == cfg.num_attention_heads:
            with torch.no_grad():
                p.uniform_(-1, 1)
    names = [n for n, _ in mod.named_parameters()]
    params = [p.detach().clone() for _, p in mod.named_parameters()]
    x = torch.randn(1, T, cfg.hidden_size, dtype=torch.float64)
    pos = torch.arange(T)

    def fn(x, *ps):
        return functional_call(mod, dict(zip(names, ps)), (x, pos))

    with exact_rmsnorm():
        return grad_check(fn, x, *params)


def attention_suite(model, tiny_cfg, vocab_size: int, log=print) -> dict:
    """All checks; raises AssertionError on the first failure. ``model`` is a small LM with the kind;
    ``tiny_cfg`` a very small config of the same kind for the gradient check (it is slow per parameter).
    The causal, cache and equivalence checks run on a :func:`sharpened <sharpen>` copy of ``model``."""
    import copy

    from frontierlab.attention.mla import set_mla_mode
    with exact_rmsnorm():
        return _suite(sharpen(copy.deepcopy(model)), tiny_cfg, vocab_size, log, set_mla_mode)


def _suite(model, tiny_cfg, vocab_size, log, set_mla_mode):
    import copy
    res = {"gradcheck": op_gradcheck(tiny_cfg)}
    log(f"PASS  op-level float64 gradcheck (input and {sum(1 for _ in model.model.layers[0].self_attn.parameters())}"
        " parameter tensors)")
    res["causal"] = causal_check(model, vocab_size, T=20, split=11)
    log(f"PASS  causal check: max |diff| = {res['causal']:.2e}")
    for chunk in (1, 5):
        res[f"cache_chunk{chunk}"] = cache_agreement(model, vocab_size, T=30, prefix=6, chunk=chunk)
        log(f"PASS  cached decode, {chunk} token(s) per step: max |diff| = {res[f'cache_chunk{chunk}']:.2e}")
    if model.config.attention == "mla":
        m = copy.deepcopy(model).double().eval()
        idx = torch.randint(0, vocab_size, (2, 18), generator=torch.Generator().manual_seed(1))
        set_mla_mode(m, "naive")
        a = m(idx).logits
        set_mla_mode(m, "absorbed")
        b = m(idx).logits
        res["naive_vs_absorbed"] = equivalence(a, b, atol=1e-10, what="naive vs absorbed MLA")
        log(f"PASS  MLA naive vs absorbed (float64): max |diff| = {res['naive_vs_absorbed']:.2e}")
        res["cache_absorbed"] = cache_agreement(m, vocab_size, T=30, prefix=6, chunk=1)
        log(f"PASS  cached decode in absorbed mode: max |diff| = {res['cache_absorbed']:.2e}")
    return res
