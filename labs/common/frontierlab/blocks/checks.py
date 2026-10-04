"""The architecture correctness suite applied to Module 6 block changes (plan section 5, Stage B).

Every BlockLM variant must pass, before any comparison counts:

1. **op-level float64 gradcheck** of each new component it contains, with respect to its inputs and every
   parameter (:func:`module_gradcheck`, the pattern of ``frontierlab.attention.checks.op_gradcheck``);
2. **causal check** on a small model (``frontierlab.testing.causal_check``);
3. **cached-decode agreement**, one token per step and in chunks of 5 (for MTP models this checks the main
   head's decode path, which is the only path decoding uses);
4. component checks: Sinkhorn output doubly stochastic to tolerance, n = 1 HC equals the plain residual,
   the MoE layer equals a dense loop reference, nested MatFormer widths equal a small dense SwiGLU.

All model-level checks run on a sharpened float64 copy with RMSNorm computed in the input dtype
(``frontierlab.attention.checks.exact_rmsnorm`` / ``sharpen``), for the reasons given there.
"""

from __future__ import annotations

import copy

import torch
from torch.func import functional_call

from frontierlab.attention.checks import exact_rmsnorm, sharpen
from frontierlab.testing import cache_agreement, causal_check, grad_check


def module_gradcheck(mod: torch.nn.Module, *inputs, extra_args: tuple = ()) -> bool:
    """float64 gradcheck of ``mod(*inputs, *extra_args)`` w.r.t. the tensor ``inputs`` and every parameter.

    ``extra_args`` are passed through without gradients (a callable sub-layer, token ids, positions)."""
    mod = copy.deepcopy(mod).double()
    names = [n for n, _ in mod.named_parameters()]
    params = [p.detach().clone() for _, p in mod.named_parameters()]
    n_in = len(inputs)

    def fn(*args):
        return functional_call(mod, dict(zip(names, args[n_in:])), (*args[:n_in], *extra_args))

    with exact_rmsnorm():
        return grad_check(fn, *inputs, *params)


def model_checks(model, vocab_size: int, log=print, cache: bool = True) -> dict:
    """Causal and cached-decode checks on a sharpened float64 copy (``cache=False`` for bidirectional models)."""
    res = {}
    with exact_rmsnorm():
        m = sharpen(copy.deepcopy(model)).double().eval()
        res["causal"] = causal_check(m, vocab_size, T=20, split=11)
        log(f"PASS  causal check: max |diff| = {res['causal']:.2e}")
        if cache:
            for chunk in (1, 5):
                res[f"cache_chunk{chunk}"] = cache_agreement(m, vocab_size, T=30, prefix=6, chunk=chunk)
                log(f"PASS  cached decode, {chunk} token(s) per step: max |diff| = {res[f'cache_chunk{chunk}']:.2e}")
    return res
