import numpy as np

from frontierlab.datax import groups
from frontierlab.labkit import load_target

lab = load_target(__file__)


def test_tokens_per_byte():
    enc = lambda s: list(s.encode("utf-8"))           # noqa: E731  (a byte tokenizer: exactly 1)
    assert lab.tokens_per_byte(enc, ["shalom", "שלום"]) == 1.0
    words = lambda s: s.split()                        # noqa: E731
    assert abs(lab.tokens_per_byte(words, ["ab cd"]) - 2 / 5) < 1e-12


def test_quantile_threshold():
    ref, tgt = np.arange(100.0), np.random.default_rng(0).normal(5, 2, 5000)
    thr = lab.quantile_threshold(ref, 25.0, tgt)
    assert abs(thr - groups.quantile_threshold(ref, 25.0, tgt)) < 1e-12
    assert abs((tgt < thr).mean() - 0.25) < 0.002


def test_rehydration_weight():
    assert lab.rehydration_weight(0.1, 0.4, 0.1) == 10.0
    assert lab.rehydration_weight(0.5, 0.4, 0.1) == 1.0
    assert abs(lab.rehydration_weight(0.25, 0.4, 0.1) - 5.5) < 1e-12
    cs = np.array([1] * 10 + [2] * 10 + [3] * 10)
    rm = np.array([1] * 6 + [0] * 4 + [1] * 1 + [0] * 9 + [1] * 5 + [0] * 5, dtype=bool)
    ref = groups.rehydration_weights(cs, rm)
    for s, r in ref["removal_rate"].items():
        assert abs(lab.rehydration_weight(r, ref["global_removal_rate"], min(ref["removal_rate"].values()))
                   - ref["weights"][s]) < 1e-12


def test_group_split_and_keys():
    keys = [f"repo{i % 7}" for i in range(70)]
    assert [lab.group_split(k) for k in keys] == groups.group_split(keys)
    assert lab.swesmith_keys("oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc") == ("oauthlib__oauthlib", "combine_file")
    assert lab.swesmith_keys("pydicom__pydicom.7d361b3d.func_pm_op_change__x1") == ("pydicom__pydicom", "func_pm_op_change")
