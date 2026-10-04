import math

import torch

import frontierlab.attention  # noqa: F401  (registers mla)
from frontierlab.attention.ops import band_mask
from frontierlab.labkit import load_target
from frontierlab.model import LM, toy
from frontierlab.optim.muon import orthogonal_polar
from frontierlab.optim.qkclip import clip_module, head_max_logits as ref_hml

lab = load_target(__file__)


def test_update_rms_theory_and_matching():
    torch.manual_seed(0)
    for shape in ((64, 64), (32, 128), (256, 64)):
        assert math.isclose(lab.orthogonal_update_rms(shape), 1 / math.sqrt(max(shape)))
        O = orthogonal_polar(torch.randn(*shape, dtype=torch.float64))          # exact U V^T
        measured = O.pow(2).mean().sqrt().item()
        assert abs(measured - lab.orthogonal_update_rms(shape)) < 1e-6
        assert math.isclose(lab.rms_matched_scale(shape), 0.2 * math.sqrt(max(shape)))


def _inputs(model, x):
    mods = [l.self_attn for l in model.model.layers]
    got = {}
    hooks = [m.register_forward_pre_hook(lambda m, a, i=i: got.__setitem__(i, (a[0].detach(), a[1])))
             for i, m in enumerate(mods)]
    model(x)
    for h in hooks:
        h.remove()
    return mods, got


def test_head_max_logits_and_gamma():
    torch.manual_seed(0)
    q, k = torch.randn(2, 3, 6, 4, dtype=torch.float64), torch.randn(2, 3, 6, 4, dtype=torch.float64)
    pos = torch.arange(6)
    mask = band_mask(pos, pos)
    s = lab.head_max_logits(q, k, 0.5, mask)
    ref = torch.tensor([max((0.5 * q[b, h, i] @ k[b, h, j]).item() for b in range(2) for i in range(6)
                           for j in range(i + 1)) for h in range(3)])
    assert torch.allclose(s, ref.to(s.dtype), atol=1e-12)
    g = lab.qk_clip_gamma(torch.tensor([50.0, 100.0, 400.0]), 100.0)
    assert torch.equal(g, torch.tensor([1.0, 1.0, 0.25]))


def test_clip_matches_shared_for_gqa_mha_and_mla():
    x = torch.randint(0, 61, (2, 12), generator=torch.Generator().manual_seed(1))
    for kw in ({"qk_norm": False}, {"qk_norm": False, "num_key_value_heads": 4},
               {"attention": "mla", "extra": {"kv_lora_rank": 32, "qk_rope_head_dim": 16}}):
        torch.manual_seed(0)
        a = LM(toy(vocab_size=61).with_(**kw)).double()
        b = LM(toy(vocab_size=61).with_(**kw)).double()
        b.load_state_dict(a.state_dict())
        mods_a, inp = _inputs(a, x)
        mods_b = [l.self_attn for l in b.model.layers]
        for i, (ma, mb) in enumerate(zip(mods_a, mods_b)):
            smax = ref_hml(ma, *inp[i])
            gamma = torch.clamp(0.7 * smax.min() / smax, max=1.0)
            clip_module(ma, gamma)
            with torch.no_grad():
                if kw.get("attention") == "mla":
                    lab.clip_mla_weights(mb.q_proj.weight, mb.kv_b_proj.weight, gamma, mb.H, mb.d_n, mb.d_r, mb.d_v)
                else:
                    lab.clip_gqa_weights(mb.q_proj.weight, mb.k_proj.weight, gamma, mb.H, mb.KV, mb.hd)
        for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            assert (p - q).abs().max() < 1e-12, (kw, n)
