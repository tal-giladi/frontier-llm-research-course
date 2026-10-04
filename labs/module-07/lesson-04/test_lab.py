import math

from frontierlab.labkit import load_target
from frontierlab.optim import schedules

lab = load_target(__file__)


def test_wsd_matches_shared_schedule():
    for shape in ("linear", "1-sqrt"):
        for r in (0.0, 0.1):
            for s in range(0, 140):
                want = schedules.lr_at(s, 120, 3e-3, 10, "wsd", decay_start=100, decay_steps=20, shape=shape, min_ratio=r)
                assert math.isclose(lab.wsd_lr(s, 3e-3, 10, 100, 20, shape, r), want, rel_tol=1e-12), (shape, r, s)


def test_stable_phase_is_constant():
    assert all(lab.wsd_lr(s, 1.0, 5, 50, 10) == 1.0 for s in range(5, 50))


def test_branch_plan_and_cost():
    plan = lab.branch_plan([300, 600], 0.1)
    assert plan == schedules.branch_plan([300, 600], 0.1)
    c = lab.branch_cost(plan)
    assert c == schedules.branch_cost(plan)
    assert c["wsd_steps"] == 540 + 30 + 60 and c["cosine_steps"] == 900


def test_decay_drop():
    steps = list(range(1, 101))
    losses = [5.0] * 80 + [5.0 - 0.05 * i for i in range(1, 21)]
    assert math.isclose(lab.decay_drop(steps, losses, 80, window=5), 5.0 - sum(losses[-5:]) / 5)
