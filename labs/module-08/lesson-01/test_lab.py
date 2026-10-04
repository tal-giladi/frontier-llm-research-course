import torch

from frontierlab.labkit import load_target
from frontierlab.precision import E2M1, E4M3, E5M2, QuantSpec, qdq, round_pow2, round_to_format

lab = load_target(__file__)

LAYOUT = {"e4m3": (4, 3, 7, 448.0), "e5m2": (5, 2, 15, 57344.0), "e2m1": (2, 1, 1, 6.0)}


def values(n=50_000, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g) * torch.exp2(torch.randint(-18, 14, (n,), generator=g).float())


def test_round_fp_matches_torch_fp8_casts():
    for (name, dt) in (("e4m3", torch.float8_e4m3fn), ("e5m2", torch.float8_e5m2)):
        mx = LAYOUT[name][3]
        x = values().clamp(-mx, mx)
        assert torch.equal(lab.round_fp(x, *LAYOUT[name]), x.to(dt).float()), name


def test_round_fp_e2m1_and_saturation():
    x = torch.tensor([0.2, 0.26, 0.74, 0.76, 1.25, 1.3, 2.5, 3.5, 5.0, 5.1, 7.0, -100.0])
    assert torch.equal(lab.round_fp(x, *LAYOUT["e2m1"]), round_to_format(x, E2M1))
    assert lab.round_fp(torch.tensor([1e5]), *LAYOUT["e4m3"]).item() == 448.0


def test_block_qdq_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(8, 256) * torch.rand(8, 1) * 10
    for block in (32, 128, 256):
        ref = qdq(x, QuantSpec("e4m3", (1, block), "fp32"))
        assert torch.allclose(lab.block_qdq(x, *LAYOUT["e4m3"], block), ref, rtol=0, atol=1e-6), block
    z = torch.zeros(2, 32)
    assert torch.equal(lab.block_qdq(z, *LAYOUT["e4m3"], 32), z)


def test_mx_shared_scale():
    amax = torch.tensor([0.7, 1.0, 5.9, 6.0, 448.0, 500.0, 3e-3])
    for emax in (8, 2):
        assert torch.equal(lab.mx_shared_scale(amax, emax), round_pow2(amax, "floor") * 2.0 ** -emax)


def test_nvfp4_matches_reference():
    torch.manual_seed(1)
    x = torch.randn(16, 64) * torch.exp2(torch.randint(-6, 6, (16, 1)).float())
    ref = qdq(x, QuantSpec("e2m1", (1, 16), "nvfp4"))
    assert torch.allclose(lab.nvfp4_qdq(x), ref, rtol=1e-6, atol=1e-7)
