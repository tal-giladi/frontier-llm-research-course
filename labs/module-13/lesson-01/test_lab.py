import math

import numpy as np
import torch

from frontierlab.labkit import load_target
from frontierlab.pipeline import dpo as D

lab = load_target(__file__)


def _logps(seed=0, B=6):
    g = torch.Generator().manual_seed(seed)
    return [-(torch.rand(B, generator=g, dtype=torch.float64) * 5 + 0.5) for _ in range(4)]


def test_dpo_loss_by_hand():
    one = lambda v: torch.tensor([v], dtype=torch.float64)
    # beta 0.1, log-ratios +2.0 (chosen) and -1.0 (rejected): h = 0.3, loss = -log sigmoid(0.3) = 0.5544
    loss = lab.dpo_loss(one(-3.0), one(-6.0), one(-5.0), one(-5.0), 0.1)
    assert abs(float(loss) - 0.554355244468527) < 1e-9
    # policy == reference: loss is log 2 whatever the responses
    pc, pr, _, _ = _logps()
    assert abs(float(lab.dpo_loss(pc, pr, pc, pr, 0.5)) - math.log(2)) < 1e-12


def test_dpo_variants_match_reference():
    pc, pr, rc, rr = _logps(1)
    lc, lr = torch.tensor([3.0, 5, 2, 7, 4, 1], dtype=torch.float64), torch.tensor([6.0, 2, 2, 3, 8, 1], dtype=torch.float64)
    for kw in (dict(), dict(normalise=True), dict(nll_coef=0.2), dict(normalise=True, nll_coef=0.2)):
        ref, _ = D.dpo_loss(pc, pr, rc, rr, 0.3, lc, lr, **kw)
        mine = lab.dpo_loss(pc, pr, rc, rr, 0.3, lc, lr, **kw)
        assert abs(float(mine) - float(ref)) < 1e-12, kw


def test_dpo_gradient_is_weighted_by_misordering():
    # dL/d pol_c = -beta * sigmoid(-h) / B and dL/d pol_r = +beta * sigmoid(-h) / B (Rafailov et al. section 4)
    pc, pr, rc, rr = _logps(2)
    pc, pr = pc.clone().requires_grad_(True), pr.clone().requires_grad_(True)
    beta = 0.7
    lab.dpo_loss(pc, pr, rc, rr, beta).backward()
    h = beta * ((pc - rc) - (pr - rr)).detach()
    w = beta * torch.sigmoid(-h) / len(h)
    assert torch.allclose(pc.grad, -w, atol=1e-12) and torch.allclose(pr.grad, w, atol=1e-12)


def test_dpo_large_margins_are_finite():
    big = torch.tensor([-1.0, -900.0], dtype=torch.float64)
    loss = lab.dpo_loss(big.flip(0), big, torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64), 1.0)
    assert torch.isfinite(loss)


def test_binarise_rule():
    for seed in range(20):
        scores = list(np.random.default_rng(seed).integers(0, 3, size=5) / 2)
        mine = lab.binarise(scores, np.random.default_rng(100 + seed))
        s = np.asarray(scores)
        if s.max() == s.min():
            assert mine is None
            continue
        rng = np.random.default_rng(100 + seed)
        c = int(rng.choice(np.flatnonzero(s == s.max())))
        r = int(rng.choice(np.flatnonzero(s < s.max())))
        assert mine == (c, r)
        assert s[mine[0]] > s[mine[1]]
    assert lab.binarise([1.0, 1.0, 1.0], np.random.default_rng(0)) is None


def test_noise_numbers():
    assert abs(lab.binomial_se(0.8, 541) - 0.017197) < 1e-5            # IFEval: 541 prompts
    assert abs(lab.mde_unpaired(0.8, 541) - 0.047666) < 1e-5
    assert abs(lab.mde_unpaired(0.5, 30) - 0.253035) < 1e-5            # one AIME year: 30 problems


def test_audit():
    rows = lab.audit([("SFT", 76.2), ("DPO", 84.3), ("RLVR", 87.6)], n_items=1319)
    assert [r["to"] for r in rows] == ["DPO", "RLVR"]
    assert abs(rows[0]["delta"] - 8.1) < 1e-9 and rows[0]["verdict"] == "beyond item noise"
    assert abs(rows[0]["mde"] - 100 * 1.96 * math.sqrt(2 * 0.762 * 0.238 / 1319)) < 1e-9
    assert rows[1]["verdict"] == "beyond item noise" and abs(rows[1]["mde"] - 2.78) < 0.01   # +3.3 vs an MDE of 2.78
    avg = lab.audit([("DPO", 64.7), ("RLVR", 65.1)], n_items=None, seed_spread=0.3)
    assert avg[0]["mde"] is None and avg[0]["verdict"] == "beyond seed spread"
    assert lab.audit([("a", 1.0), ("b", 1.1)], None)[0]["verdict"] == "no noise estimate"
