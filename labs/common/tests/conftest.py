import json

import numpy as np
import pytest


@pytest.fixture
def tiny_data(tmp_path):
    """Synthetic Data-v0-shaped files (no download): a repeating pattern the toy model can learn."""
    rng = np.random.default_rng(0)
    meta = {"name": "synthetic", "vocab_size": 64, "eot_id": 0}
    for split, n in (("train", 60000), ("val", 8000), ("test", 8000)):
        base = np.tile(np.arange(1, 33, dtype=np.uint16), n // 32 + 1)[:n]
        noise = rng.integers(1, 64, size=n).astype(np.uint16)
        toks = np.where(rng.random(n) < 0.1, noise, base).astype(np.uint16)
        toks.tofile(tmp_path / f"{split}.bin")
        np.save(tmp_path / f"{split}_docs.npy", np.arange(0, n, 500, dtype=np.int64))
        meta[split] = {"tokens": n}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return tmp_path
