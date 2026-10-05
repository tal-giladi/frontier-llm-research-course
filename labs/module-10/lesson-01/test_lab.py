import numpy as np
import torch

from frontierlab.datax import leakage, neardup, packing
from frontierlab.datax.mixture import _rng
from frontierlab.labkit import load_target

lab = load_target(__file__)


def test_segment_ids():
    assert lab.segment_ids(np.array([0, 5, 7]), 3, 6).tolist() == [0, 0, 1, 1, 2, 2]
    rng = np.random.default_rng(0)
    starts = np.concatenate([[0], np.cumsum(rng.integers(1, 30, 200))])
    for s in rng.integers(0, int(starts[-1]) - 64, 20):
        assert np.array_equal(lab.segment_ids(starts, int(s), 64), packing.segment_ids(starts, int(s), 64))


def test_docmask_matches_reference_and_cache_layout():
    g = torch.Generator().manual_seed(0)
    seg = torch.sort(torch.randint(0, 3, (2, 12), generator=g), dim=1).values
    for q, k in ((torch.arange(12), torch.arange(12)), (torch.arange(8, 12), torch.arange(12))):
        assert torch.equal(lab.docmask(q, k, seg), packing.docmask(q, k, seg))
    m = lab.docmask(torch.arange(4), torch.arange(4), torch.tensor([[0, 0, 1, 1]]))
    assert m[0, 0].tolist() == [[True, False, False, False], [True, True, False, False],
                                [False, False, True, False], [False, False, True, True]]


def test_window_counts():
    assert lab.window_counts([0.5, 0.3, 0.2], 7).tolist() == [4, 2, 1]
    rng = np.random.default_rng(1)
    from frontierlab.datax.mixture import window_counts
    for _ in range(50):
        w = rng.dirichlet(np.ones(5))
        assert np.array_equal(lab.window_counts(w, 100), window_counts(w, 100))


def test_locate_matches_sampler_schedule():
    counts = np.array([5, 3, 2])

    def order(b):
        slots = np.repeat(np.arange(3), counts)
        return slots[_rng(0, 7, b).permutation(slots.size)]
    seen = np.zeros(3, dtype=int)
    for k in range(73):
        s, w = lab.locate(k, 10, counts, order)
        assert w == seen[s], "each source's windows must be consumed in order 0, 1, 2, ..."
        seen[s] += 1


def test_minhash_and_bands():
    rng = np.random.default_rng(2)
    h = neardup.MinHasher(64, 0)
    sh = np.unique(rng.integers(0, 2**63, 300, dtype=np.int64).astype(np.uint64))
    assert np.array_equal(lab.minhash_signature(sh, h.salts), h.signature(sh))
    assert (lab.minhash_signature(np.zeros(0, np.uint64), h.salts) == np.iinfo(np.uint64).max).all()
    sigs = rng.integers(0, 3, size=(40, 16)).astype(np.uint64)
    groups = ["train"] * 30 + ["val"] * 10
    assert lab.band_candidates(sigs, groups, 8) == neardup.cross_split_pairs(groups, sigs, 8)


def test_ngram_overlap():
    idx = leakage.NgramIndex(3).add_texts(["a b c d e f", "x y z"]).finish()
    item = leakage.ngram_hashes(neardup.doc_words("b c d q r"), 3)       # b c d is in, c d q and d q r are not
    assert abs(lab.ngram_overlap(item, idx.hashes) - 1 / 3) < 1e-12
    assert lab.ngram_overlap(np.zeros(0, np.uint64), idx.hashes) == 0.0


def test_decide():
    assert lab.decide_noninferior((-0.02, 0.005), 0.01) == "adopt"
    assert lab.decide_noninferior((0.012, 0.03), 0.01) == "reject"
    assert lab.decide_noninferior((-0.01, 0.02), 0.01) == "inconclusive"
