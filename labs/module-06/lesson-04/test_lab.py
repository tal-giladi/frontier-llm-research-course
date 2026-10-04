import torch

from frontierlab.blocks.engram import EngramModule, collision_rate, hash_ngrams
from frontierlab.labkit import load_target

lab = load_target(__file__)


def test_ngram_hash_matches_module():
    torch.manual_seed(0)
    ids = torch.randint(0, 8192, (3, 40))
    for n, mult, M in ((2, 1_000_003, 4093), (3, 1_015_841, 4091)):
        assert torch.equal(lab.ngram_hash(ids, n, mult, M, 8192), hash_ngrams(ids, n, mult, M, 8192))


def test_gate_matches_module():
    torch.manual_seed(0)
    mod = EngramModule(16, 50, max_n=3, heads=1, table_size=31, head_dim=4).double()
    ids = torch.randint(0, 50, (1, 6))
    h = torch.randn(1, 6, 16, dtype=torch.float64)
    mod(h, ids)
    e = mod.lookup(ids, 6)
    k = mod.k_proj(e)
    assert torch.allclose(lab.engram_gate(h, k), mod.last_gate[..., 0], atol=1e-12)


def test_experts_at_matches_paper():
    # Engram section 3.1: 106 and 99 experts at rho = 1; "46 experts for the 5.7B model and 43 experts for the
    # 9.9B model" at rho ~ 40%; section 4: 72 -> 55 experts at rho = 74.3%. All consistent with top-6 routing.
    assert lab.experts_at(0.4, 106, 6) == 46
    assert lab.experts_at(0.4, 99, 6) == 43
    assert lab.experts_at(0.743, 72, 6) == 55
    assert lab.experts_at(1.0, 72, 6) == 72


def test_collision_free_share():
    assert abs(lab.collision_free_share(1, 100) - 1.0) < 1e-12
    one = lab.collision_free_share(5000, 4093, 1)
    two = lab.collision_free_share(5000, 4093, 2)
    assert 0.25 < one < 0.32 and abs(two - (1 - (1 - one) ** 2)) < 1e-12
    st = collision_rate(torch.randint(0, 500, (20000,)), 2, 4093)
    assert abs((1 - st["share_colliding"]) - lab.collision_free_share(st["distinct_ngrams"], st["table"])) < 0.05
