from pathlib import Path

import pytest
import torch
import yaml

from frontierlab.labkit import load_target

lab = load_target(__file__)
CARDS = Path(__file__).parent / "cards"


def test_flatten():
    assert lab.flatten({"a": {"b": 1, "c": {"d": [1, 2]}}, "e": {}, "f": None}) == \
        {"a.b": 1, "a.c.d": [1, 2], "e": {}, "f": None}
    card = lab.flatten(yaml.safe_load((CARDS / "b0-s0.yaml").read_text()))
    assert card["data_files.train.bin_sha256"].startswith("8ac79b") and card["args.lr"] == 0.003


@pytest.mark.parametrize("key,kw,expected", [
    ("args.log_every", {}, "ignore"),
    ("run", {}, "ignore"),
    ("config.attention", {"changed": ("config.attention",)}, "changed"),
    ("config.attention", {}, "invalidates"),
    ("params.total", {"changed": ("config.attention",)}, "changed"),
    ("params.total", {}, "invalidates"),
    ("params.non_embedding", {"changed": ("config",)}, "changed"),
    ("args.seed", {}, "invalidates"),
    ("args.seed", {"seeds_are_replicates": True}, "replicate"),
    ("data_files.val.bin_sha256", {"changed": ("config.attention",)}, "invalidates"),
    ("data.tokenizer_sha256", {}, "invalidates"),
    ("args.eval_windows", {}, "invalidates"),
    ("hardware.torch", {}, "warn"),
    ("hardware.gpu", {"axis": "wallclock"}, "invalidates"),
    ("budget.tokens", {"axis": "tokens"}, "invalidates"),
    ("args.steps", {"axis": "flops", "changed": ("config.hidden_size",)}, "changed"),
    ("args.lr", {"changed": ("config.attention",)}, "invalidates"),
    ("args.lr", {"changed": ("args.lr",)}, "changed"),
    ("measured.cost_usd", {}, "warn"),
])
def test_classify(key, kw, expected):
    assert lab.classify(key, **kw) == expected


def test_planted_cards_are_not_comparable():
    """The two provided cards differ in the declared variable AND in things that break the comparison."""
    a = lab.flatten(yaml.safe_load((CARDS / "b0-s0.yaml").read_text()))
    b = lab.flatten(yaml.safe_load((CARDS / "mla-s0.yaml").read_text()))
    diffs = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    sev = {k: lab.classify(k, changed=("config.attention", "config.extra", "args.attention")) for k in diffs}
    bad = sorted(k for k, s in sev.items() if s == "invalidates")
    assert bad == ["args.batch", "args.lr", "args.seq", "data_files.val.bin_sha256"]


def test_same_bits():
    x = torch.tensor([1.0, 0.0, float("nan")])
    assert lab.same_bits(x, x.clone())                                     # NaN has the same bits
    assert not lab.same_bits(x, torch.tensor([1.0, -0.0, float("nan")]))   # -0.0 has different bits
    assert not lab.same_bits(x, x.double())
    assert not lab.same_bits(torch.ones(4), torch.ones(4) + 1e-7)
    assert lab.same_bits(torch.ones(2, 3).t(), torch.ones(3, 2))          # non-contiguous input
