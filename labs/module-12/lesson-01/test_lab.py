import itertools
import math

import numpy as np
import torch

from frontierlab.labkit import load_target
from frontierlab.posttrain import reward as RW

lab = load_target(__file__)


def test_bt_loss():
    rc, rr = torch.tensor([2.0, 0.0, -1.0]), torch.tensor([0.0, 0.0, 1.0])
    want = np.mean([math.log1p(math.exp(-2.0)), math.log(2.0), math.log1p(math.exp(2.0))])
    assert abs(lab.bt_loss(rc, rr).item() - want) < 1e-6
    big = lab.bt_loss(torch.tensor([0.0]), torch.tensor([200.0]))          # stable for large margins
    assert torch.isfinite(big) and abs(big.item() - 200.0) < 1e-3


def test_bon_kl():
    for n in (1, 2, 4, 64, 4096):
        assert abs(lab.bon_kl(n) - float(RW.bon_kl(n))) < 1e-12
    assert abs(lab.bon_kl(4) - (math.log(4) - 0.75)) < 1e-12


def test_bon_expected_against_enumeration():
    rng = np.random.default_rng(1)
    proxy, val = rng.normal(size=8), rng.normal(size=8)
    for n in (1, 3, 8):
        brute = np.mean([val[list(c)][np.argmax(proxy[list(c)])] for c in itertools.combinations(range(8), n)])
        assert abs(lab.bon_expected(proxy, val, n) - brute) < 1e-10
    big_p, big_v = rng.normal(size=5000), rng.normal(size=5000)
    assert abs(lab.bon_expected(big_p, big_v, 1000) - RW.bon_expected(big_p, big_v, 1000)) < 1e-9


def test_ece():
    p = np.array([0.95] * 10 + [0.55] * 10 + [1.0])
    y = np.array([1] * 9 + [0] + [1] * 5 + [0] * 5 + [1], dtype=float)
    assert abs(lab.ece(p, y) - RW.ece(p, y)) < 1e-12
    perfect = np.array([0.25] * 4)
    assert abs(lab.ece(perfect, np.array([1.0, 0, 0, 0]))) < 1e-12


def test_verifier_errors():
    acc = [1, 1, 0, 1, 0, 0]
    tru = [1, 0, 0, 1, 1, 0]
    e = lab.verifier_errors(acc, tru)
    assert abs(e["false_positive_rate"] - 1 / 3) < 1e-12
    assert abs(e["false_negative_rate"] - 1 / 3) < 1e-12
    assert abs(e["precision"] - 2 / 3) < 1e-12
    assert lab.verifier_errors([0, 0], [0, 0])["false_negative_rate"] == 0.0


def test_peak():
    out = lab.peak([1, 4, 16, 64], [0.0, 1.0, 1.2, 1.1], tol=0.05)
    assert out == {"n_peak": 16, "gold_peak": 1.2, "gold_last": 1.1, "declined": True}
    assert lab.peak([1, 4], [0.0, 1.0])["declined"] is False
