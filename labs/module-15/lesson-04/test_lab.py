import torch

from frontierlab.labkit import load_target
from frontierlab.ttc import kvquant as KQ

lab = load_target(__file__)


def test_asym_qdq_by_hand():
    x = torch.tensor([[0.0, 1.0, 2.0, 3.0, 10.0, 10.0, 10.0, 10.0]], dtype=torch.float64)
    # group 1: min 0, max 3, 2 bits -> scale 1, exact; group 2 is constant -> unchanged
    assert torch.equal(lab.asym_qdq(x, 2, 1, 4), x)
    y = lab.asym_qdq(torch.tensor([[0.0, 0.4, 0.6, 3.0]], dtype=torch.float64), 1, 1, 4)
    assert torch.allclose(y, torch.tensor([[0.0, 0.0, 0.0, 3.0]], dtype=torch.float64))
    g = torch.Generator().manual_seed(0)
    z = torch.randn(3, 64, 5, generator=g, dtype=torch.float64)
    for bits, dim, grp in ((4, 1, 16), (2, 1, 32), (3, 2, 5)):
        assert torch.allclose(lab.asym_qdq(z, bits, dim, grp), KQ.asym_qdq(z, bits, dim, grp))


def test_kivi_layout_matches_course():
    g = torch.Generator().manual_seed(1)
    k = torch.randn(2, 2, 150, 16, generator=g, dtype=torch.float64)
    v = torch.randn(2, 2, 150, 16, generator=g, dtype=torch.float64)
    for bits, grp, res in ((2, 32, 128), (4, 32, 0), (2, 16, 40), (4, 32, 200)):
        a = lab.kivi_qdq(k, v, bits, grp, res)
        b = KQ.kivi_qdq(k, v, bits, grp, res)
        assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])


def test_cache_bytes():
    # Qwen3-1.7B layer: 2 x 8 x 128 = 2048 elements per token; 4096 tokens, KIVI-2 with G = 32, R = 128
    n_q = (4096 - 128) // 32 * 32
    expect = (n_q * 2048 * (2 + 1) + (4096 - n_q) * 2048 * 16) / 8
    assert lab.kv_cache_bytes(4096, 2048, 2, 32, 128) == expect
    assert lab.kv_cache_bytes(100, 10, 4, 32, 128) == 100 * 10 * 2              # all in the residual window
    assert lab.kv_cache_bytes(4096, 2048, 2, 32, 128) == KQ.kv_cache_bits(4096, 2048, 2, 32, 128) / 8
