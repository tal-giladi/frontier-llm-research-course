"""Derivation tests for an objective library (the project's correctness suite).

    pytest labs/module-14/project                       # the course objectives: every test passes
    OBJ_HOOKS=buggy pytest labs/module-14/project       # the colleague's library: which tests fail?

Point ``library()`` at your own implementation to test it the same way.
"""

import os
from pathlib import Path

import pytest
import torch

from frontierlab.rlscale import objectives as O


def library():
    lib = {n: O.OBJECTIVES[n].loss for n in ("grpo", "gspo", "cispo")}
    if os.environ.get("OBJ_HOOKS") == "buggy":
        from frontierlab.labkit import load_path
        lib.update(load_path(str(Path(__file__).resolve().parent / "buggy_objectives.py")).BUGGY)
    return lib


def batch(seed, spread=0.4, B=6, R=6):
    g = torch.Generator().manual_seed(seed)
    old = torch.log(torch.rand(B, R, generator=g, dtype=torch.float64) * 0.8 + 0.1)
    logp = old + spread * torch.randn(B, R, generator=g, dtype=torch.float64)
    lengths = torch.tensor([6, 2, 3, 1, 4, 5])[:B]
    mask = (torch.arange(R)[None] < lengths[:, None]).double()
    return logp, old, torch.randn(B, generator=g, dtype=torch.float64), mask


def weights(f, logp, old, adv, mask):
    lp = logp.clone().requires_grad_(True)
    f(lp, old, adv, mask)[0].backward()
    return -lp.grad


def test_grpo_clipping_is_pessimistic():
    """Where the clipped branch is smaller, the token gets no gradient (PPO's min, not max)."""
    logp, old, adv, mask = batch(0, spread=0.6)
    r = torch.exp(logp - old)
    a = adv[:, None]
    should_be_zero = (((a > 0) & (r > 1.2)) | ((a < 0) & (r < 0.8))) & mask.bool()
    assert should_be_zero.any()
    w = weights(library()["grpo"], logp, old, adv, mask)
    assert (w[should_be_zero] == 0).all()
    assert torch.allclose(w, O.gradient_weights("grpo", logp, old, adv, mask), atol=1e-12)


def test_gspo_ratio_is_length_normalised():
    """Appending tokens with the same ratio must not change the sequence ratio (geometric mean, not product)."""
    old = torch.zeros(2, 6, dtype=torch.float64)
    lp = torch.full((2, 6), 2e-4, dtype=torch.float64)            # every token ratio 1.0002
    mask = torch.tensor([[1.0, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.float64)
    adv = torch.tensor([1.0, 1.0], dtype=torch.float64)
    w = weights(library()["gspo"], lp, old, adv, mask)
    s = torch.exp(torch.tensor(2e-4))
    want = torch.tensor([[s / 2 / 2] * 2 + [0] * 4, [s / 2 / 6] * 6], dtype=torch.float64)
    assert torch.allclose(w, want, atol=1e-9), "s_i must be inside the clip range for both lengths"


def test_cispo_keeps_gradient_past_the_cap():
    logp, old, adv, mask = batch(1, spread=0.8)
    r = torch.exp(logp - old)
    capped = (r > 1.28) & mask.bool()
    assert capped.any()
    w = weights(library()["cispo"], logp, old, adv, mask)
    assert (w[capped] != 0).all(), "a capped token must keep its gradient at weight 1 + eps_high"
    assert torch.allclose(w, O.gradient_weights("cispo", logp, old, adv, mask), atol=1e-12)


@pytest.mark.parametrize("name", ["grpo", "gspo", "cispo"])
def test_on_policy_gradient_is_reinforce(name):
    """At ratio 1 every objective is a policy gradient with its paper's token weights."""
    logp, _, adv, mask = batch(2)
    w = weights(library()[name], logp, logp.clone(), adv, mask)
    if name == "cispo":
        want = adv[:, None] / mask.sum() * mask
    else:
        want = (adv / mask.sum(-1) / len(adv))[:, None] * mask
    assert torch.allclose(w, want, atol=1e-12)


@pytest.mark.parametrize("name", ["grpo", "gspo", "cispo"])
def test_padding_and_old_logp(name):
    logp, old, adv, mask = batch(3)
    a = weights(library()[name], logp, old, adv, mask)
    b = weights(library()[name], logp + (1 - mask) * 4.0, old, adv, mask)
    assert torch.allclose(a, b, atol=1e-12)
    o = old.clone().requires_grad_(True)
    library()[name](logp.clone().requires_grad_(True), o, adv, mask)[0].backward()
    assert o.grad is None or torch.count_nonzero(o.grad) == 0
