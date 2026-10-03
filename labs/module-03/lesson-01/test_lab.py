import torch

from frontierlab.attention import mla  # noqa: F401  (registers "mla")
from frontierlab.labkit import load_target
from frontierlab.model import LM, baseline0, toy

lab = load_target(__file__)


def mla_layer(seed=0):
    torch.manual_seed(seed)
    cfg = toy(vocab_size=50).with_(attention="mla", extra={"kv_lora_rank": 24, "qk_rope_head_dim": 8})
    return LM(cfg).double().model.layers[0].self_attn


def pieces(mod, T=7, S=None, seed=1):
    """Queries for T new tokens at the end of an S-token context, and the S cached latents."""
    S = S or T
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, S, 128, dtype=torch.float64, generator=g)
    pos = torch.arange(S)
    cos, sin = mod.rope(pos)
    q_nope, q_rope = mod.queries(x, cos, sin)
    c, k_rope = mod.latents(x, cos, sin)
    return q_nope[:, :, -T:], q_rope[:, :, -T:], c, k_rope, pos[-T:], pos


def test_expand_latent_matches_kv_b_proj():
    mod = mla_layer()
    _, _, c, _, _, _ = pieces(mod)
    k, v = lab.expand_latent(c, mod.kv_b_proj.weight, mod.H, mod.d_n, mod.d_v)
    kv = mod.kv_b_proj(c).view(2, c.shape[1], mod.H, mod.d_n + mod.d_v).transpose(1, 2)
    assert k.shape == (2, mod.H, c.shape[1], mod.d_n) and v.shape == (2, mod.H, c.shape[1], mod.d_v)
    assert torch.allclose(k, kv[..., :mod.d_n], atol=1e-13) and torch.allclose(v, kv[..., mod.d_n:], atol=1e-13)


def test_absorbed_query_gives_the_same_scores():
    mod = mla_layer()
    q_nope, _, c, _, _, _ = pieces(mod)
    k_nope, _ = mod.kv_b_proj(c).view(2, c.shape[1], mod.H, -1).transpose(1, 2).split((mod.d_n, mod.d_v), -1)
    q_lat = lab.absorb_query(q_nope, mod.kv_b_proj.weight, mod.H, mod.d_n, mod.d_v)
    assert q_lat.shape == (2, mod.H, q_nope.shape[2], mod.d_c)
    naive = q_nope @ k_nope.transpose(-1, -2)                       # (B, H, T, S)
    absorbed = q_lat @ c.unsqueeze(1).transpose(-1, -2)
    assert (naive - absorbed).abs().max() < 1e-12


def test_absorbed_attention_equals_naive_full_and_decode():
    mod = mla_layer()
    for T, S in ((7, 7), (1, 9), (3, 11)):                          # full sequence, 1-token and chunked decode
        q_nope, q_rope, c, k_rope, q_pos, k_pos = pieces(mod, T, S)
        ref = mod.naive(q_nope, q_rope, c, k_rope, q_pos, k_pos)
        out = lab.absorbed_attention(q_nope, q_rope, c, k_rope, mod.kv_b_proj.weight, mod.H, mod.d_n, mod.d_v,
                                     q_pos, k_pos, mod.scale)
        assert out.shape == ref.shape and (out - ref).abs().max() < 1e-12, (T, S)


def test_kv_elements_per_token():
    b0 = baseline0()
    L, K, d = b0.num_hidden_layers, b0.num_key_value_heads, b0.head_dim
    assert lab.kv_elements_per_token("gqa", L, K=K, d=d) * 2 == 12_288          # lesson 01.1, BF16
    assert lab.kv_elements_per_token("mha", L, K=12, d=d) == 12 * 2 * 12 * 64
    assert lab.kv_elements_per_token("mqa", L, K=12, d=d) == 12 * 2 * 64
    assert lab.kv_elements_per_token("mla", L, d_c=256, d_r=32) == 12 * 288
    assert lab.kv_elements_per_token("mla", 61, d_c=512, d_r=64) == 35_136       # DeepSeek-V3, lesson 01.2
