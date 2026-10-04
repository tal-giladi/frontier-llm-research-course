import torch

from frontierlab.labkit import load_target
from frontierlab.optim.stability import detect_spikes

lab = load_target(__file__)


def test_z_loss_and_softcap():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11) * 3
    want = 1e-4 * torch.logsumexp(logits, -1).pow(2).mean()
    assert torch.allclose(lab.z_loss(logits), want, atol=1e-9)
    x = torch.tensor([-1000.0, -1.0, 0.0, 2.0, 1000.0])
    y = lab.softcap(x, 30.0)
    assert y.abs().max() <= 30.0 and abs(y[3].item() - 30 * torch.tanh(torch.tensor(2 / 30)).item()) < 1e-6
    assert abs(y[1].item() + 1.0) < 1e-3                       # small values pass almost unchanged


def test_find_spikes_matches_shared_detector():
    steps = list(range(1, 151))
    losses = [6.0 - 0.004 * s + 0.01 * ((s * 7) % 5) for s in steps]
    losses[79] += 1.5
    losses[80] += 0.8
    flagged = lab.find_spikes(steps, losses)
    merged = detect_spikes(steps, losses)
    assert flagged == [80, 81]
    assert [m["start"] for m in merged] == [80] and merged[0]["end"] == 81


def test_first_crossing_and_classify():
    steps = [10, 20, 30, 40]
    vals = [1.0, 5.0, 30.0, 90.0]
    assert lab.first_crossing(steps, vals, 20.0, before=40) == 30
    assert lab.first_crossing(steps, vals, 100.0, before=40) is None
    assert lab.first_crossing(steps, vals, 20.0, before=25) is None
    assert lab.classify(6.0, 80.0, 1.0, True) == "logit growth"
    assert lab.classify(6.0, 10.0, 5.0, True) == "optimizer"
    assert lab.classify(1.2, 8.0, 1.1, True) == "data"
    assert lab.classify(1.2, 8.0, 1.1, False) == "unclear"
