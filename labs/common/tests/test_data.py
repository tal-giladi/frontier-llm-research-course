from frontierlab.data.loader import TokenData
from frontierlab.data.prepare import doc_hash, split_of


def test_split_follows_normalized_hash():
    a, b = "Hello   world.\n", "Hello world."
    assert doc_hash(a) == doc_hash(b)                   # whitespace-normalized duplicates collide
    assert split_of(doc_hash(a)) == split_of(doc_hash(b))
    counts = {"train": 0, "val": 0, "test": 0}
    for i in range(20000):
        counts[split_of(doc_hash(f"document {i}"))] += 1
    assert 150 < counts["val"] < 260 and 150 < counts["test"] < 260


def test_eval_windows_fixed_and_disjoint(tiny_data):
    d = TokenData("val", root=tiny_data)
    w1, w2 = d.eval_windows(20, 64), d.eval_windows(20, 64)
    assert w1 == w2
    assert all(b - a >= 64 for a, b in zip(w1, w1[1:]))
