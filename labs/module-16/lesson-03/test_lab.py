import pytest
import torch

from frontierlab.agents import agentrl as R
from frontierlab.agents import turns as T
from frontierlab.labkit import load_target

lab = load_target(__file__)


def test_loss_mask():
    a = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 0.0]])
    o = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
    assert lab.loss_mask(a, o).tolist() == a.tolist()
    assert lab.loss_mask(a, o, "all").tolist() == [[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]]
    with pytest.raises(ValueError):
        lab.loss_mask(a, o, "observations")


def test_turn_returns_match_library():
    for rs, f, g in (([0.1, 0.2], 1.0, 0.5), ([], 1.0, 1.0), ([0.3, 0.0, -0.1], 0.0, 0.9)):
        assert lab.turn_returns(rs, f, g) == pytest.approx(R.turn_returns([rs], [f], g)[0])
    assert lab.turn_returns([0.1, 0.2], 1.0, 0.5) == pytest.approx([0.45, 0.7, 1.0])


def test_spread_over_tokens():
    ti = torch.tensor([0, 0, 0, 1, 1, 2, -1, -1])
    assert lab.spread_over_tokens([1.0, -0.5, 2.0], ti).tolist() == [1, 1, 1, -0.5, -0.5, 2, 0, 0]
    assert lab.spread_over_tokens([1.0], ti).tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    want = R.token_advantages([[1.0, -0.5, 2.0]], ti[None])[0]
    assert torch.equal(lab.spread_over_tokens([1.0, -0.5, 2.0], ti), want)


def test_context_cost():
    full = lab.context_cost(3, [2, 2, 3], [3, 3, 0])
    assert full == {"prefill_tokens": 3 + 8 + 13, "peak_context": 16}
    win = lab.context_cost(3, [2, 2, 3], [3, 3, 0], "window", 1)
    assert win == {"prefill_tokens": 3 + 8 + 8, "peak_context": 11}
    assert [c for c in T.context_tokens(3, [2, 2, 3], [3, 3, 0], "window", 1)] == [3, 8, 8]
    assert lab.context_cost(5, [], []) == {"prefill_tokens": 0, "peak_context": 5}
