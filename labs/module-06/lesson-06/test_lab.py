import math

from frontierlab.blocks import patching
from frontierlab.labkit import load_target

lab = load_target(__file__)


def test_entropy():
    assert abs(lab.entropy([0.25] * 4) - math.log(4)) < 1e-12
    assert lab.entropy([1.0, 0.0, 0.0]) == 0.0
    m = patching.NgramByteModel(b"the cat sat on the mat. the cat sat.", order=2)
    c = m.counts[b"th"]
    total = sum(c.values()) + m.alpha * 256
    probs = [(c.get(v, 0) + m.alpha) / total for v in range(256)]
    assert abs(lab.entropy(probs) - m.entropy(b"th")) < 1e-12


def test_patch_starts_match_module():
    H = [3.1, 0.2, 0.4, 2.2, 1.9, 0.3, 2.8, 0.1, 0.05, 1.4]
    for mode in ("global", "monotonic"):
        for th in (0.5, 1.0, 1.5, 2.5):
            assert lab.patch_starts(H, th, mode) == patching.patch_starts(H, th, mode)


def test_mean_patch_size():
    assert lab.mean_patch_size([0, 3, 7], 8) == 8 / 3


def test_flops_per_byte():
    assert lab.flops_per_byte(8e8, 1e7, 4.0) == 2e8 + 1e7
    assert lab.flops_per_byte(8e8, 0.0, 8.0) == patching.latent_flops_per_byte(8e8, 8.0)
