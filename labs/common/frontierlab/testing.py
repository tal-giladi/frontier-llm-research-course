"""The architecture correctness suite (plan section 5, Stage B).

Every new attention kind or block variant must pass these before any comparison counts:

* :func:`grad_check` — analytic vs numerical gradients in float64 at toy size.
* :func:`causal_check` — changing future tokens must not change earlier outputs.
* :func:`cache_agreement` — full-sequence logits equal token-by-token (or chunked) cached decoding.
* :func:`equivalence` — two implementations of the same function agree (recurrent vs parallel,
  naive vs absorbed, own code vs a reference) within a stated tolerance.

Each returns the measured error so lessons can print it, and raises ``AssertionError`` with a
readable message when the check fails.
"""

from __future__ import annotations

import copy

import torch


def _model_in_double(model):
    m = copy.deepcopy(model).double().eval()
    return m


def grad_check(fn, *inputs, eps: float = 1e-6, atol: float = 1e-5, rtol: float = 1e-3) -> bool:
    """``torch.autograd.gradcheck`` on ``fn`` with float64 copies of ``inputs`` (requires_grad set)."""
    ins = tuple(x.detach().double().requires_grad_(x.is_floating_point()) if torch.is_tensor(x) else x
                for x in inputs)
    return torch.autograd.gradcheck(fn, ins, eps=eps, atol=atol, rtol=rtol)


@torch.no_grad()
def causal_check(model, vocab_size: int, T: int = 16, split: int = 9, B: int = 2, seed: int = 0,
                 atol: float = 1e-9) -> float:
    """Replace tokens at positions >= ``split``; logits at positions < ``split`` must not change."""
    m = _model_in_double(model)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, vocab_size, (B, T), generator=g)
    idx2 = idx.clone()
    idx2[:, split:] = torch.randint(0, vocab_size, (B, T - split), generator=g)
    a, b = m(idx).logits[:, :split], m(idx2).logits[:, :split]
    err = (a - b).abs().max().item()
    assert err <= atol, f"future tokens changed past outputs: max |diff| = {err:.3e} (> {atol})"
    return err


@torch.no_grad()
def cache_agreement(model, vocab_size: int, T: int = 24, prefix: int = 8, chunk: int = 1, B: int = 2,
                    seed: int = 0, atol: float = 1e-9) -> float:
    """Logits of one full forward vs a prefix forward followed by cached decoding in ``chunk``-token steps."""
    m = _model_in_double(model)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, vocab_size, (B, T), generator=g)
    full = m(idx).logits
    cache = m.new_cache()
    parts = [m(idx[:, :prefix], cache=cache).logits]
    for s in range(prefix, T, chunk):
        parts.append(m(idx[:, s:s + chunk], cache=cache).logits)
    inc = torch.cat(parts, dim=1)
    err = (full - inc).abs().max().item()
    assert err <= atol, f"cached decode disagrees with full forward: max |diff| = {err:.3e} (> {atol})"
    return err


def equivalence(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float = 0.0, what: str = "outputs") -> float:
    """Max absolute difference between two implementations' outputs; asserts it is within tolerance."""
    err = (a - b).abs().max().item()
    tol = atol + rtol * b.abs().max().item()
    assert err <= tol, f"{what} differ: max |diff| = {err:.3e} (> {tol:.3e})"
    return err


def run_suite(model, vocab_size: int) -> dict:
    """Causal and cache checks on a model (gradient checks are per-module, at op level)."""
    return {"causal_max_abs_diff": causal_check(model, vocab_size),
            "cache_max_abs_diff_step1": cache_agreement(model, vocab_size, chunk=1),
            "cache_max_abs_diff_chunk5": cache_agreement(model, vocab_size, chunk=5)}
