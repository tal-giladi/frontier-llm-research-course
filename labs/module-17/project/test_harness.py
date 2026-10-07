"""Derivation tests for the claim harness. ``pytest labs/module-17/project`` checks harness.py;
``HARNESS=buggy pytest labs/module-17/project`` checks the colleague's version (three tests fail)."""

import importlib.util
import os
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def harness():
    name = os.environ.get("HARNESS", "harness")
    name = "buggy_harness" if name == "buggy" else name
    spec = importlib.util.spec_from_file_location(f"m17_test_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = harness()


def test_effect_divides_by_the_mean_gap():
    clean = torch.tensor([5.0, 0.01])
    corrupt = torch.tensor([0.0, 0.0])
    ablated = torch.tensor([0.0, 0.0])
    e = H.effect(ablated, clean, corrupt, "noise")
    gap = 2.505
    assert torch.allclose(e, torch.tensor([5.0 / gap, 0.01 / gap]))       # a near-zero-gap prompt stays small


def test_heldout_is_a_different_distribution():
    sel, held = H.selection("induction", 8), H.heldout("induction", 8)
    differs = (sel.meta["gap"] != held.meta["gap"]) or (sel.meta["query"] != held.meta["query"])
    assert differs, "held-out prompts must differ in distribution (gap, query position), not only in seed"
    from frontierlab.interp import hf
    _, tok = hf.load("", smoke=True)
    s, h = H.selection("ioi", 12, tok=tok), H.heldout("ioi", 12, tok=tok)
    assert s.meta["family"] != h.meta["family"]
    assert set(s.good.tolist()).isdisjoint(set(h.good.tolist()))         # other names


def test_random_sets_match_size_and_exclude_the_candidate():
    cand = {3: [1, 4], 5: [0, 2, 7]}
    sets = H.random_sets(cand, 8, 8, 20, seed=0)
    assert len(sets) == 20
    for s in sets:
        heads = [(l, h) for l, hs in s.items() for h in hs]
        assert len(heads) == 5 and len(set(heads)) == 5
        assert not any(h in cand.get(l, []) for l, h in heads)
        assert {l: len(hs) for l, hs in s.items()} == {3: 2, 5: 3}            # layer-matched by default
    loose = H.random_sets(cand, 8, 8, 20, seed=0, layer_matched=False)
    assert all(sum(len(hs) for hs in s.values()) == 5 for s in loose)
