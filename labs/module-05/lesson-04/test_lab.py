import torch

from frontierlab.attention import compressed
from frontierlab.labkit import load_target

lab = load_target(__file__)


def rand(*s, seed=0):
    return torch.randn(*s, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def test_hca_by_hand_and_against_reference():
    c = torch.tensor([[[1.0], [3.0], [10.0], [20.0], [99.0]]])   # S = 5, m = 2: two entries, token 4 dropped
    z = torch.zeros_like(c)                                       # equal weights: mean pooling
    assert torch.allclose(lab.compress_hca(c, z, 2), torch.tensor([[[2.0], [15.0]]]))
    z2 = torch.tensor([[[0.0], [torch.log(torch.tensor(3.0))], [0.0], [0.0], [0.0]]])
    assert torch.allclose(lab.compress_hca(c, z2, 2)[0, 0], torch.tensor([0.25 * 1 + 0.75 * 3]))
    c, z = rand(2, 23, 4), rand(2, 23, 4, seed=1)
    assert torch.allclose(lab.compress_hca(c, z, 4), compressed.compress_hca(c, z, 4))


def test_csa_matches_reference_and_overlaps():
    ca, za, cb, zb = (rand(2, 20, 3, seed=s) for s in range(4))
    out = lab.compress_csa(ca, za, cb, zb, 4)
    assert out.shape == (2, 5, 3)
    assert torch.allclose(out, compressed.compress_csa(ca, za, cb, zb, 4))
    cb2 = cb.clone()
    cb2[:, 4:8] += 1.0                                            # block 1 through path b ...
    out2 = lab.compress_csa(ca, za, cb2, zb, 4)
    changed = (out2 - out).abs().amax(dim=(0, 2)) > 1e-12
    assert changed.tolist() == [False, False, True, False, False]   # ... only changes entry 2


def test_usable_entries_is_causal():
    assert [lab.usable_entries(t, 4) for t in range(9)] == [0, 0, 0, 1, 1, 1, 1, 2, 2]
    c, z = rand(1, 16, 2), rand(1, 16, 2, seed=1)
    for t in range(16):
        c2 = c.clone()
        c2[:, t + 1:] += 5.0                                      # change the future of position t
        n = lab.usable_entries(t, 4)
        assert torch.equal(compressed.compress_hca(c, z, 4)[:, :n], compressed.compress_hca(c2, z, 4)[:, :n])


def test_layout_kv_bytes():
    layers = ["csa", "hca"] * 3
    for S in (100, 4096, 1 << 20):
        expect = 3 * (compressed.kv_entries(S, m=4, window=128) + compressed.kv_entries(S, m=128)) * 600
        assert lab.layout_kv_bytes(S, layers, m=4, m2=128, window=128, entry_bytes=600) == expect
    assert lab.layout_kv_bytes(10, ["dense"], m=4, m2=128, window=0, entry_bytes=2) == 20
