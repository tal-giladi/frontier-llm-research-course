import torch

from frontierlab.blocks import BlockLM, accounting, with_blocks
from frontierlab.blocks.mtp import deepseek_losses, lambda_at
from frontierlab.labkit import load_target
from frontierlab.model import baseline0, toy

lab = load_target(__file__)
V = 97


def model(depth=2):
    torch.manual_seed(0)
    return BlockLM(with_blocks(toy(vocab_size=V), mtp="deepseek", mtp_depth=depth)).double()


def test_mtp_chain_matches_reference():
    m = model()
    idx = torch.randint(0, V, (2, 14))
    h0 = m(idx, return_hidden=True).hidden["h"]
    mine = lab.mtp_chain(m, h0, idx)
    ref = m.mtp(h0, idx, m.model.embed_tokens, m.lm_head)
    assert len(mine) == len(ref) == 2
    for a, b in zip(mine, ref):
        assert a.shape == b.shape and (a - b).abs().max().item() < 1e-12


def test_mtp_chain_is_causal():
    """Changing tokens from position 9 on must not change depth-k logits at positions i with i + k < 9."""
    m = model()
    idx = torch.randint(0, V, (1, 14))
    idx2 = idx.clone()
    idx2[:, 9:] = (idx2[:, 9:] + 1) % V
    a = lab.mtp_chain(m, m(idx, return_hidden=True).hidden["h"], idx)
    b = lab.mtp_chain(m, m(idx2, return_hidden=True).hidden["h"], idx2)
    for k, (x, y) in enumerate(zip(a, b), start=1):
        assert (x[:, :9 - k] - y[:, :9 - k]).abs().max().item() < 1e-12


def test_mtp_loss_matches_reference():
    m = model()
    idx = torch.randint(0, V, (2, 14))
    logits = m.mtp(m(idx, return_hidden=True).hidden["h"], idx, m.model.embed_tokens, m.lm_head)
    ref = torch.stack(deepseek_losses(logits, idx)).mean()
    assert abs(lab.mtp_loss(logits, idx).item() - ref.item()) < 1e-12


def test_lambda_schedule():
    for steps in (148, 1000, 9500):
        for s in range(0, steps, max(1, steps // 37)):
            assert lab.lambda_schedule(s, steps) == lambda_at(s, steps)


def test_accepted_prefix():
    assert lab.accepted_prefix([5, 7, 9], [5, 7, 9]) == 3
    assert lab.accepted_prefix([5, 7, 9], [5, 8, 9]) == 1
    assert lab.accepted_prefix([5], [6]) == 0
    assert lab.accepted_prefix([], []) == 0


def test_mtp_extra_flops():
    for cfg, T, D in ((toy(vocab_size=8192), 128, 1), (toy(vocab_size=8192), 128, 2), (baseline0(), 1024, 1)):
        ref = accounting.flops_per_token(with_blocks(cfg, mtp="deepseek", mtp_depth=D), T) \
            - accounting.flops_per_token(with_blocks(cfg), T)
        assert abs(lab.mtp_extra_flops(cfg, T, D) - ref) / ref < 1e-9
