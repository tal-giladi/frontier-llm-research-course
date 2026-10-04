import torch

from frontierlab.blocks.diffusion import diffusion_loss_terms, mask_tokens
from frontierlab.labkit import load_target

lab = load_target(__file__)


def test_masked_terms_match_module():
    torch.manual_seed(0)
    logits = torch.randn(3, 7, 11, dtype=torch.float64)
    x0 = torch.randint(0, 10, (3, 7))
    t = torch.tensor([0.2, 0.6, 1.0], dtype=torch.float64)
    _, masked = mask_tokens(x0, t, 10, torch.Generator().manual_seed(1))
    assert torch.allclose(lab.masked_terms(logits, x0, masked, t), diffusion_loss_terms(logits, x0, masked, t), atol=1e-12)


def test_eq6_estimate():
    ce = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    masked = torch.tensor([[True, False, True, False]])
    # L = 4, l = 2: (4 / 2) * (1 + 3) / 4 = 2.0
    assert torch.allclose(lab.eq6_estimate(ce, masked, torch.tensor([2])), torch.tensor([2.0], dtype=torch.float64))


def test_eq6_is_unbiased_for_constant_ce():
    """With the same CE c at every position, every draw of Eq. 6 equals c: the L/l weight undoes the masking rate."""
    ce = torch.full((1, 6), 1.7, dtype=torch.float64)
    for l in range(1, 7):
        masked = torch.zeros(1, 6, dtype=torch.bool)
        masked[0, :l] = True
        assert abs(lab.eq6_estimate(ce, masked, torch.tensor([l])).item() - 1.7) < 1e-12


def test_commit_count():
    assert [lab.commit_count(r, s) for r, s in ((32, 8), (30, 8), (5, 1), (1, 4))] == [4, 4, 5, 1]


def test_effective_depth():
    assert lab.effective_depth(2, 4, 2, 32) == 132           # Geiping et al. section 3: (2, 4, 2) at r = 32
    assert lab.effective_depth(1, 2, 1, 1) == 4
