import numpy as np
import torch

from frontierlab.interp import hooks as HK
from frontierlab.interp import steering as ST
from frontierlab.labkit import load_target
from frontierlab.model import LM, toy

lab = load_target(__file__)


def test_mean_diff():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    b = torch.tensor([[0.0, 0.0], [0.0, 2.0], [0.0, 1.0]], dtype=torch.float64)
    v = lab.mean_diff(a, b)
    assert v.dtype == torch.float32 and torch.allclose(v, torch.tensor([2.0, 2.0]))
    assert torch.allclose(v, ST.mean_diff(a, b))


def test_steer_edit_positions_and_no_in_place():
    x = torch.zeros(2, 5, 3)
    v = torch.tensor([1.0, -1.0, 0.5])
    y = lab.steer_edit(v, 2.0, start=2)(x)
    assert torch.equal(x, torch.zeros(2, 5, 3))
    assert torch.equal(y[:, :2], torch.zeros(2, 2, 3)) and torch.allclose(y[:, 2:], (2 * v).expand(2, 3, 3))
    torch.manual_seed(0)
    m = LM(toy(vocab_size=40).with_(num_hidden_layers=2)).eval()
    ids = torch.randint(0, 40, (1, 8))
    ours = HK.run_with(m, ids, {"resid_post.0": lab.steer_edit(v.repeat(128 // 3 + 1)[:128], 1.5, 3)})
    course = HK.run_with(m, ids, {"resid_post.0": ST.steer_edit(v.repeat(128 // 3 + 1)[:128], 1.5, 3)})
    assert torch.allclose(ours, course)


def test_token_kl():
    p = torch.log(torch.tensor([[0.5, 0.5], [0.9, 0.1]]))
    q = torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.5]]))
    kl = lab.token_kl(p, q)
    assert abs(float(kl[0])) < 1e-7
    assert abs(float(kl[1]) - (0.9 * np.log(0.9 / 0.5) + 0.1 * np.log(0.1 / 0.5))) < 1e-6
    assert (lab.token_kl(q, p) >= 0).all()


def test_flip_rate():
    assert lab.flip_rate([1.0, -2.0, 0.5, -0.1], [2.0, 1.0, -0.5, -3.0]) == 0.5
    assert lab.flip_rate(np.zeros(3) - 1, np.zeros(3) - 2) == 0.0
