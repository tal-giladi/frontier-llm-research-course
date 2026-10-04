import numpy as np

from frontierlab.labkit import load_target

lab = load_target(__file__)


def test_interaction():
    cells = {(0, 0): 6.0, (1, 0): 5.8, (0, 1): 5.9, (1, 1): 5.75}
    # A alone helps 0.20; with B on, A helps 5.90 - 5.75 = 0.15: the effects overlap by 0.05
    assert abs(lab.interaction(cells) - 0.05) < 1e-12
    additive = {(0, 0): 1.0, (1, 0): 0.7, (0, 1): 0.9, (1, 1): 0.6}
    assert abs(lab.interaction(additive)) < 1e-12


def test_interaction_ci_is_paired():
    rng = np.random.default_rng(0)
    difficulty = rng.normal(6.0, 1.0, 400)                 # large per-window variation shared by all cells
    noise = lambda: rng.normal(0, 0.01, 400)  # noqa: E731
    pw = {(0, 0): difficulty + noise(), (1, 0): difficulty - 0.2 + noise(),
          (0, 1): difficulty - 0.1 + noise(), (1, 1): difficulty - 0.25 + noise()}
    point, lo, hi = lab.interaction_ci(pw)
    assert abs(point - 0.05) < 0.01 and lo > 0.03 and hi < 0.07       # paired: tight despite std-1 windows


def test_additive_prediction():
    assert abs(lab.additive_prediction(6.0, {"a": 5.8, "b": 5.9, "c": 6.05}) - 5.75) < 1e-12


def test_design_cells():
    full = lab.design_cells(["a", "b", "c", "d"], "full")
    lite = lab.design_cells(["a", "b", "c", "d"], "lite")
    assert len(full) == 16 and len(lite) == 10
    assert frozenset() in lite and frozenset("abcd") in lite and frozenset("abc") in lite
    assert len(lab.design_cells(["a", "b"], "lite")) == 4
    assert len(lab.design_cells(["a", "b", "c"], "lite")) == 8
