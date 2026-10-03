import math

import pytest
import torch

from frontierlab.labkit import load_target
from frontierlab.longctx import rope as ref

lab = load_target(__file__)

SHAPES = [(32, 1e4, 4.0, 256), (64, 1e4, 32.0, 1024), (128, 1e6, 4.0, 32768), (64, 1e4, 2.0, 2048)]


def rel(a, b):
    return ((a - b) / b).abs().max().item()


@pytest.mark.parametrize("D,base,s,L", SHAPES)
def test_rules_match_reference(D, base, s, L):
    assert rel(lab.ntk_inv_freq(D, base, s), ref.ntk_inv_freq(D, base, s)) < 1e-12
    theta = ref.default_inv_freq(D, base)
    assert rel(lab.rotations(theta, L), ref.rotations(theta, L)) < 1e-12
    assert torch.allclose(lab.yarn_gamma(D, base, L), ref.yarn_gamma(D, base, L), atol=1e-12)
    assert rel(lab.yarn_inv_freq(D, base, s, L), ref.yarn_inv_freq(D, base, s, L)) < 1e-12
    assert abs(lab.yarn_attention_factor(s) - ref.yarn_attention_factor(s)) < 1e-12


def test_by_hand():
    # NTK keeps the fastest pair and divides the slowest by exactly s
    n = lab.ntk_inv_freq(8, 1e4, 4.0)
    assert n[0] == 1.0 and abs(n[-1] / (1e4 ** (-6 / 8) / 4.0) - 1) < 1e-12
    # pair 0 turns L / (2 pi) times over L tokens
    assert abs(lab.rotations(torch.tensor([1.0], dtype=torch.float64), 628)[0] - 628 / (2 * math.pi)) < 1e-12
    assert lab.yarn_attention_factor(1.0) == 1.0 and abs(lab.yarn_attention_factor(math.e) - 1.1) < 1e-12


def test_yarn_matches_transformers():
    transformers = pytest.importorskip("transformers")
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    cfg = transformers.Qwen3Config(hidden_size=128, num_attention_heads=4, head_dim=32, max_position_embeddings=1024,
                                   rope_parameters={"rope_type": "yarn", "rope_theta": 1e4, "factor": 4.0,
                                                    "original_max_position_embeddings": 256})
    inv, af = ROPE_INIT_FUNCTIONS["yarn"](cfg, "cpu")
    assert rel(lab.yarn_inv_freq(32, 1e4, 4.0, 256), inv.double()) < 1e-6
    assert abs(lab.yarn_attention_factor(4.0) - af) < 1e-12
