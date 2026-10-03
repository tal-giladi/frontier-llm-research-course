import torch

from frontierlab.attention.base import RotaryEmbedding, apply_rope
from frontierlab.labkit import load_target

lab = load_target(__file__)


def test_attention_flops_by_hand():
    assert lab.attention_flops_per_token(2, 3, 3, 10) == 2 * 2 * 6 * 10
    # Kimi K2 MLA (naive dims): 64 heads, d_qk = 128 + 64, d_v = 128, at 128K keys
    assert lab.attention_flops_per_token(64, 192, 128, 131072) == 2 * 64 * 320 * 131072


def test_decode_attention_share():
    assert lab.decode_attention_share(1, 1, 1, 1, 2) == 4 / 8
    assert lab.decode_attention_share(12, 64, 64, 0, 1e6) == 0


def test_partial_rope_matches_apply_rope():
    x = torch.randn(2, 3, 5, 16, dtype=torch.float64)
    cos, sin = RotaryEmbedding(4, 10000.0)(torch.arange(5))
    out = lab.partial_rope(x, cos.double(), sin.double())
    assert torch.allclose(out, apply_rope(x, cos.double(), sin.double()), atol=1e-14)
    assert torch.equal(out[..., 4:], x[..., 4:])                    # unrotated channels untouched
