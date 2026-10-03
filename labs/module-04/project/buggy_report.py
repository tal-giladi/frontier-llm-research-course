"""A colleague's evaluation of an extended model (the Module 4 project's debugging task). It has two bugs.

    python labs/module-04/project/buggy_report.py --base runs/m04/base-cpu --extended runs/m04/project/cpu-stage1 \\
        --original 256 --new 1024

Their summary from this script: "after extension the far-position loss at 1,024 is only 0.03 nats
better than the base model with YaRN switched on, so the training did little at long range, and it
cost 0.085 nats of short-context loss, so extension is too expensive for us". Find both bugs from
the symptoms; do not diff this file against the course scripts.
"""

import argparse

import numpy as np
import torch

from frontierlab.data.loader import TokenData
from frontierlab.data.prepare import DEFAULT_OUT
from frontierlab.evals import suite_v1 as v1
from frontierlab.evals.heldout import window_losses
from frontierlab.longctx.attention import convert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--extended", required=True)
    ap.add_argument("--original", type=int, default=256)
    ap.add_argument("--new", type=int, default=1024)
    a = ap.parse_args()
    L0, L1 = a.original, a.new
    long_root = DEFAULT_OUT.parent / "v0-long"
    data = TokenData("val", long_root if (long_root / "meta.json").exists() else DEFAULT_OUT)
    val = TokenData("val")

    base = v1.load_model(a.base)
    ext = v1.load_model(a.extended)
    # "make sure both use the same YaRN setting at the new length"
    yarn = {"type": "yarn", "factor": L1 / L0, "original_max_position_embeddings": L1}
    base_yarn = convert(base, "gqa-rope-scaled", rope=yarn)
    ext = convert(ext, "gqa-rope-scaled", rope=yarn)

    far = []
    for m in (base_yarn, ext):
        losses = v1.doc_position_losses(m, data, L1, max_docs=100)
        far.append(losses[:, L1 // 2:].mean(1))
    d = far[1] - far[0]
    print(f"far-position loss, extended - base+yarn: {d.mean():+.4f} (paired over {d.size} documents)")

    # short-context check: each model at the length it was trained at
    hb = np.mean(window_losses(base, val, 128, L0))
    he = np.mean(window_losses(ext, val, 128, L1))
    print(f"short-context held-out loss: base {hb:.4f}, extended {he:.4f}, change {he - hb:+.4f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
