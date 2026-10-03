import pytest

from frontierlab.flops import flops_per_token
from frontierlab.labkit import load_target
from frontierlab.model import toy

lab = load_target(__file__)
SMALL = toy()
WIDE = toy().with_(hidden_size=256, num_hidden_layers=6, head_dim=64, intermediate_size=768)


def test_matched_steps():
    assert lab.matched_steps("tokens", WIDE, SMALL, 400, 128) == 400
    ratio = flops_per_token(WIDE, 128) / flops_per_token(SMALL, 128)
    assert lab.matched_steps("flops", WIDE, SMALL, 400, 128) == round(400 * ratio)
    assert lab.matched_steps("wallclock", WIDE, SMALL, 400, 128, ref_tok_per_s=1000, other_tok_per_s=3000) == 1200
    with pytest.raises(ValueError):
        lab.matched_steps("params", WIDE, SMALL, 400, 128)


class CountingDict(dict):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.reads.append(key)
        return super().get(key, default)

    def values(self):
        self.reads.append("*")
        return super().values()

    def items(self):
        self.reads.append("*")
        return super().items()


def test_select_on_val_report_test():
    val = {"lr1e-3": 3.10, "lr3e-3": 3.02, "lr1e-2": 3.05}
    test = CountingDict({"lr1e-3": 3.00, "lr3e-3": 3.08, "lr1e-2": 2.95})   # test would pick lr1e-2
    assert lab.select_then_report(val, test) == ("lr3e-3", 3.08)
    assert test.reads == ["lr3e-3"], "the test split must be read once, for the chosen config only"


@pytest.mark.parametrize("ci,expected", [((-0.05, -0.03), "adopt"), ((-0.015, 0.01), "reject"),
                                         ((-0.03, -0.01), "inconclusive"), ((0.01, 0.02), "reject"),
                                         ((-0.04, 0.02), "inconclusive")])
def test_decide(ci, expected):
    assert lab.decide(ci, 0.02) == expected
