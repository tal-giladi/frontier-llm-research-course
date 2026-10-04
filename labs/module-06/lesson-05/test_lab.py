import torch

from frontierlab.blocks.matformer import NestedMLP, PerLayerEmbedding
from frontierlab.labkit import load_target

lab = load_target(__file__)


def test_nested_ffn_matches_module():
    torch.manual_seed(0)
    mlp = NestedMLP(16, 64).double()
    x = torch.randn(3, 5, 16, dtype=torch.float64)
    for m in (8, 16, 32, 64):
        mlp.width = m
        y = lab.nested_ffn(x, mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight, m)
        assert torch.allclose(y, mlp(x), atol=1e-12)


def test_mix_n_match_properties():
    widths = [48, 96, 192, 384]
    for budget in (60, 120, 150, 250, 384):
        cfg = lab.mix_n_match(widths, 4, budget)
        assert len(cfg) == 4 and cfg == sorted(cfg)
        idx = [widths.index(w) for w in cfg]
        assert max(idx) - min(idx) <= 1
        assert abs(sum(cfg) / 4 - budget) <= max(b - a for a, b in zip(widths, widths[1:])) / 4 + 1e-9 or budget > 384


def test_consistency():
    a = torch.tensor([[1, 2, 3, 4]])
    assert lab.consistency(a, torch.tensor([[1, 2, 0, 4]])) == 0.75


def test_ple_params():
    ple = PerLayerEmbedding(100, 32, 4, 8)
    assert lab.ple_params(100, 4, 8) == ple.host_params()
    # Transformers' Gemma3nTextConfig defaults: vocab_size_per_layer_input 262,144, 35 layers, 256 per layer
    assert lab.ple_params(262_144, 35, 256) == 2_348_810_240
