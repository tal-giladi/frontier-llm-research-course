import copy

import torch

from frontierlab.record import comparable, diff_cards, flatten, state_diff
from frontierlab.record.__main__ import main as cli

CARD = {
    "run": "b0-s0", "question": "noise floor", "parent_run": None,
    "git": {"commit": "abc123", "dirty": False},
    "hardware": {"python": "3.12.10", "torch": "2.14.1", "platform": "Linux", "cpu_threads": 16,
                 "gpu": "NVIDIA H100 80GB HBM3"},
    "config": {"hidden_size": 768, "attention": "gqa", "num_key_value_heads": 4},
    "args": {"run": "runs/b0-s0", "lr": 3e-3, "steps": 9500, "seq": 1024, "batch": 32, "seed": 0,
             "data_seed": None, "eval_windows": 64, "log_every": 20, "stop_after": None},
    "data": {"name": "Data-v0", "revision": "87f0", "tokenizer_sha256": "c725"},
    "data_files": {"train": {"tokens": 2_490_000_000, "bin_sha256": "8ac7"}},
    "budget": {"steps": 9500, "tokens": 2.49e9, "train_flops": 1.96e18},
    "params": {"total": 121_917_696},
}


def variant(**changes):
    c = copy.deepcopy(CARD)
    for dotted, v in changes.items():
        node = c
        *path, last = dotted.split("__")
        for p in path:
            node = node[p]
        node[last] = v
    return c


def sev(findings):
    return {f.key: f.severity for f in findings}


def test_flatten():
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": {}}) == {"a.b": 1, "a.c.d": 2, "e": {}}


def test_identical_and_bookkeeping_only():
    b = variant(run="b0-s0-resumed", args__run="runs/x", args__log_every=50, args__stop_after=150,
                hardware__cpu_threads=8)
    f = diff_cards(CARD, b)
    assert comparable(f) and all(x.severity == "ignore" for x in f)


def test_declared_change_and_its_consequences():
    b = variant(config__attention="mla", params__total=118_000_000)
    f = diff_cards(CARD, b, changed=["config.attention"])
    assert sev(f) == {"config.attention": "changed", "params.total": "changed"} and comparable(f)


def test_undeclared_second_variable_invalidates():
    b = variant(config__attention="mla", args__lr=4e-3)
    f = diff_cards(CARD, b, changed=["config.attention"])
    assert sev(f)["args.lr"] == "invalidates" and not comparable(f)


def test_data_and_eval_invalidate():
    assert sev(diff_cards(CARD, variant(data_files__train__bin_sha256="ffff")))["data_files.train.bin_sha256"] == "invalidates"
    assert sev(diff_cards(CARD, variant(args__eval_windows=128)))["args.eval_windows"] == "invalidates"


def test_seeds_depend_on_design():
    b = variant(args__seed=1)
    assert sev(diff_cards(CARD, b))["args.seed"] == "invalidates"
    assert sev(diff_cards(CARD, b, seeds_are_replicates=True))["args.seed"] == "replicate"


def test_budget_against_axis():
    longer = variant(args__steps=12000, budget__steps=12000, budget__tokens=3.1e9, budget__train_flops=1.96e18,
                     config__hidden_size=640, params__total=90e6)
    changed = ["config.hidden_size"]
    assert not comparable(diff_cards(CARD, longer, changed=changed, axis="tokens"))
    assert comparable(diff_cards(CARD, longer, changed=changed, axis="flops"))
    off = variant(budget__train_flops=2.2e18)
    assert sev(diff_cards(CARD, off, axis="flops"))["budget.train_flops"] == "invalidates"


def test_hardware_and_software_matter_only_for_wallclock():
    b = variant(hardware__gpu="NVIDIA A100-SXM4-80GB", hardware__torch="2.13.0", git__commit="def456")
    assert comparable(diff_cards(CARD, b)) and set(sev(diff_cards(CARD, b)).values()) == {"warn"}
    assert not comparable(diff_cards(CARD, b, axis="wallclock"))


def test_dirty_tree_warns_even_when_both_dirty():
    a, b = variant(git__dirty=True), variant(git__dirty=True)
    assert sev(diff_cards(a, b)) == {"git.dirty": "warn"}


def test_cli_exit_code(tmp_path):
    import yaml
    for name, card in (("a", CARD), ("b", variant(args__lr=1e-3))):
        (tmp_path / name).mkdir()
        (tmp_path / name / "run_card.yaml").write_text(yaml.safe_dump(card))
    assert cli([str(tmp_path / "a"), str(tmp_path / "a")]) == 0
    assert cli([str(tmp_path / "a"), str(tmp_path / "b")]) == 1


def test_state_diff_is_bitwise():
    a = {"w": torch.ones(3), "opt": {"state": {0: {"exp_avg": torch.zeros(2)}}}, "step": 5}
    b = copy.deepcopy(a)
    assert state_diff(a, b) == []
    b["w"] = torch.ones(3) + torch.tensor([0.0, 0.0, 1e-7])
    b["opt"]["state"][0]["exp_avg"] = torch.tensor([0.0, -0.0])        # equal values, different bits
    assert state_diff(a, b) == ["opt.state.0.exp_avg", "w"]
    nan = {"x": torch.tensor([float("nan")])}
    assert state_diff(nan, copy.deepcopy(nan)) == []                     # same bits, although nan != nan


def test_whole_config_declared():
    b = variant(config__hidden_size=640, params__total=90e6)
    assert comparable(diff_cards(CARD, b, changed=["config"]))
