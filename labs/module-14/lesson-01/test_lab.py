import pytest
import torch

from frontierlab.labkit import load_target
from frontierlab.rlscale import objectives as O

lab = load_target(__file__)
NAMES = ["grpo", "dapo", "drgrpo", "gspo", "cispo"]


def fn(name):
    return getattr(lab, f"{name}_loss")


def batch(seed, spread=0.4, B=6, R=5):
    g = torch.Generator().manual_seed(seed)
    old = torch.log(torch.rand(B, R, generator=g, dtype=torch.float64) * 0.8 + 0.1)
    logp = old + spread * torch.randn(B, R, generator=g, dtype=torch.float64)
    lengths = torch.tensor([5, 2, 3, 1, 4, 5])[:B]
    mask = (torch.arange(R)[None] < lengths[:, None]).double()
    adv = torch.randn(B, generator=g, dtype=torch.float64)
    return logp, old, adv, mask


def weights(f, logp, old, adv, mask, **kw):
    lp = logp.clone().requires_grad_(True)
    loss, d = f(lp, old, adv, mask, **kw)
    loss.backward()
    return loss.detach(), -lp.grad, d


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("spread", [0.0, 1e-4, 0.4])
def test_value_and_gradient_match_reference(name, spread):
    for seed in range(3):
        logp, old, adv, mask = batch(seed, spread)
        kw = {"norm_len": 7} if name == "drgrpo" else {}
        ref = O.OBJECTIVES[name].loss(logp, old, adv, mask, **{**O.OBJECTIVES[name].params, **kw})[0]
        got, w, _ = weights(fn(name), logp, old, adv, mask, **kw)
        assert torch.allclose(got, ref, atol=1e-12), (name, spread, seed)
        assert torch.allclose(w, O.gradient_weights(name, logp, old, adv, mask, **kw), atol=1e-12), (name, spread)


def test_worked_example_by_hand():
    old = torch.zeros(2, 3, dtype=torch.float64)
    logp = torch.log(torch.tensor([[1.5, 0.9, 1.0], [0.7, 1.1, 1.0]], dtype=torch.float64))
    mask = torch.tensor([[1.0, 1, 0], [1, 1, 1]], dtype=torch.float64)
    adv = torch.tensor([1.0, -1.0], dtype=torch.float64)
    want = {"grpo": [[0, 0.225, 0], [0, -1.1 / 6, -1 / 6]], "dapo": [[0, 0.18, 0], [0, -0.22, -0.2]],
            "drgrpo": [[0, 0.1125, 0], [0, -0.1375, -0.125]], "gspo": [[0, 0, 0], [0, 0, 0]],
            "cispo": [[0.256, 0.18, 0], [-0.14, -0.22, -0.2]]}
    for name, w in want.items():
        kw = {"norm_len": 4} if name == "drgrpo" else {}
        _, got, _ = weights(fn(name), logp, old, adv, mask, **kw)
        assert torch.allclose(got, torch.tensor(w, dtype=torch.float64), atol=1e-4), name


def test_no_gradient_reaches_old_logp_and_padding_is_ignored():
    logp, old, adv, mask = batch(9)
    for name in NAMES:
        o = old.clone().requires_grad_(True)
        lp = logp.clone().requires_grad_(True)
        fn(name)(lp, o, adv, mask)[0].backward()
        assert o.grad is None or torch.count_nonzero(o.grad) == 0, name
        noisy = logp + (1 - mask) * 3.0
        a = weights(fn(name), logp, old, adv, mask)[1]
        b = weights(fn(name), noisy, old, adv, mask)[1]
        assert torch.allclose(a, b, atol=1e-12) and (b[mask == 0] == 0).all(), name


def test_diagnostics():
    logp, old, adv, mask = batch(1, spread=0.8)
    for name in NAMES:
        d = fn(name)(logp, old, adv, mask)[1]
        assert {"clip_frac", "ratio_mean", "ratio_max"} <= set(d) and 0.0 <= d["clip_frac"] <= 1.0, name
    assert fn("gspo")(logp, old, adv, mask)[1]["clip_frac"] > 0.5      # tiny ranges: most sequences clipped
