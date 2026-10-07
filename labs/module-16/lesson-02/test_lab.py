import math

from frontierlab.labkit import load_target

lab = load_target(__file__)


def test_split_groups_keeps_groups_together():
    keys = ["a", "a", "b", "c", "c", "c", "d", "e", "f", "g", "h", "h"]
    sp = lab.split_groups(keys, 0.25, seed=3)
    for k in set(keys):
        assert len({s for kk, s in zip(keys, sp) if kk == k}) == 1
    assert len({k for k, s in zip(keys, sp) if s == "held"}) == 2          # round(0.25 * 8)
    assert lab.split_groups(keys, 0.25, seed=3) == sp
    assert lab.split_groups(["x"], 0.01) == ["held"]


def test_sibling_leak():
    keys = ["f", "f", "g", "g", "h"]
    assert lab.sibling_leak(["train", "held", "train", "train", "held"], keys) == 0.5
    assert lab.sibling_leak(["train", "train", "held", "held", "train"], keys) == 0.0
    assert math.isnan(lab.sibling_leak(["train"] * 5, keys))


def test_patch_files():
    p = ("diff --git a/src/x.py b/src/x.py\nindex 1..2 100644\n--- a/src/x.py\n+++ b/src/x.py\n@@ -1 +1 @@\n-a\n+b\n"
         "diff --git a/tests/t.py b/tests/t.py\n")
    assert lab.patch_files(p) == {"src/x.py", "tests/t.py"}
    assert lab.patch_files("") == set()


def test_instance_keys():
    k = lab.instance_keys("oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc")
    assert k == {"instance": "oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc", "repository": "oauthlib__oauthlib",
                 "family": "combine_file"}
    assert lab.instance_keys("cantools__cantools.0c6a7871.func_basic__9co3eysz")["family"] == "func_basic"
