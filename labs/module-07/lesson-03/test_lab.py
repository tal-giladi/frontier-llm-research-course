import math

import torch

from frontierlab.labkit import load_target
from frontierlab.model import toy
from frontierlab.optim import mup
from frontierlab.optim.stabilizers import OptLM

lab = load_target(__file__)


def test_width_mult_and_readout():
    assert lab.width_mult(512, 128) == 4
    assert lab.readout_mult(4.0) == 0.25


def test_init_std_matches_shared_apply_mup():
    torch.manual_seed(0)
    cfg = mup.width_config(toy(vocab_size=97), 512)
    model = OptLM(cfg)
    mup.apply_mup(model, base_width=128)
    m = lab.width_mult(512, 128)
    for name, p in model.named_parameters():
        want = lab.mup_init_std(name, p.ndim, m)
        if p.ndim == 1:
            assert want is None, name
        else:
            assert want is not None and abs(p.std().item() - want) < 0.1 * want, name


def test_lr_scales_match_shared():
    cfg = mup.width_config(toy(vocab_size=97), 256)
    model = OptLM(cfg)
    for opt, adjust in (("adamw", "match_rms"), ("muon", "match_rms"), ("muon", "original")):
        ref = mup.lr_scales(model, 64, opt, "spectral", adjust)
        for name, p in model.named_parameters():
            assert math.isclose(lab.mup_lr_scale(name, p.ndim, 4.0, opt, adjust), ref[name]), (opt, adjust, name)


def test_best_lr_and_transfer_verdict():
    grid = [1e-3, 3e-3, 1e-2, 3e-2]
    assert lab.best_lr({1e-3: 5.0, 3e-3: 4.0, 1e-2: 4.0}) == 3e-3
    sp = {64: {1e-3: 5.1, 3e-3: 4.9, 1e-2: 4.7, 3e-2: 4.8}, 256: {1e-3: 4.6, 3e-3: 4.4, 1e-2: 4.9, 3e-2: 5.5}}
    v = lab.transfer_verdict(sp, grid)
    assert v["best"] == {64: 1e-2, 256: 3e-3} and v["shift"] == 1 and v["transfers"]
    sp[256] = {1e-3: 4.3, 3e-3: 4.4, 1e-2: 4.9, 3e-2: 5.5}
    v = lab.transfer_verdict(sp, grid)
    assert v["shift"] == 2 and not v["transfers"]
