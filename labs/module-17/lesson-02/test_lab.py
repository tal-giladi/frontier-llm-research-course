import torch

from frontierlab.interp import hooks as HK
from frontierlab.interp import patching as P
from frontierlab.interp import tasks as T
from frontierlab.labkit import load_target
from frontierlab.model import LM

lab = load_target(__file__)


def small_model():
    torch.manual_seed(0)
    return LM(T.induction_config()).double().eval()


def test_normalised_effect_by_hand():
    clean, corrupt = torch.tensor([4.0, 6.0]), torch.tensor([0.0, -2.0])     # gap = 5 - (-1) = 6
    patched = torch.tensor([2.0, 1.0])
    assert torch.allclose(lab.normalised_effect(patched, clean, corrupt, "denoise"), torch.tensor([2 / 6, 3 / 6]))
    assert torch.allclose(lab.normalised_effect(patched, clean, corrupt, "noise"), torch.tensor([2 / 6, 5 / 6]))
    for mode in ("denoise", "noise"):
        assert torch.allclose(lab.normalised_effect(patched, clean, corrupt, mode),
                              P.normalised(patched, clean, corrupt, mode))


def test_head_patch_matches_course():
    m = small_model()
    p = T.induction_pairs(6, kind="key", seed=3)
    _, a = HK.capture(m, p.clean, ["z.1"])
    z_before = a["z.1"].clone()
    edit = lab.head_patch_edit(a["z.1"], 2, 16)
    ours = HK.run_with(m, p.corrupt, {"z.1": edit})
    course = HK.run_with(m, p.corrupt, {"z.1": HK.replace_at(a["z.1"], None, 2, 16)})
    assert torch.allclose(ours, course, atol=1e-12)
    assert torch.equal(a["z.1"], z_before)
    x = torch.zeros_like(a["z.1"])
    y = edit(x)
    assert torch.equal(x, torch.zeros_like(x)) and torch.equal(y[..., 32:48], a["z.1"][..., 32:48])
    assert torch.equal(y[..., :32], x[..., :32])


def test_head_delta_matches_course_and_linearity():
    m = small_model()
    p = T.induction_pairs(5, kind="key", seed=4)
    _, a = HK.capture(m, p.clean, ["z.0"])
    _, b = HK.capture(m, p.corrupt, ["z.0"])
    W = m.model.layers[0].self_attn.o_proj.weight
    d = lab.head_delta(a["z.0"], b["z.0"], W, [1, 3], 16)
    assert torch.allclose(d, P._head_delta(m, 0, a["z.0"], b["z.0"], [1, 3]), atol=1e-12)
    # all heads together = the change of the whole attention output
    full = lab.head_delta(a["z.0"], b["z.0"], W, [0, 1, 2, 3], 16)
    assert torch.allclose(full, b["z.0"] @ W.T - a["z.0"] @ W.T, atol=1e-12)


def test_p_random():
    assert lab.p_random(0.9, [0.1] * 19) == 1 / 20
    assert lab.p_random(0.1, [0.1, 0.2, 0.0]) == 3 / 4
