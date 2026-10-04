"""Module 5 project, debugging task: a sparse-attention result that is too good to be true.

    python labs/module-05/project/buggy_dsa.py            # reproduce the report's numbers, then diagnose

A teammate's report says their DSA variant ("dsa-fast", registered below) reaches a *lower* held-out loss
than dense attention after only 60 training steps from the Module 4 base model, and asks to adopt it. This
script retrains both arms for 60 steps from ``runs/m04/base-cpu`` (about 2 minutes on a laptop) and prints
the held-out losses. Your task, in the project write-up:

1. Before reading the code below the marker line, run the correctness suite on the variant
   (``frontierlab.testing.causal_check`` and ``cache_agreement`` on a toy model of kind "dsa-fast", as in
   ``labs/common/tests/test_attention_m05.py``) and report which check fails and by how much.
2. Find the bug, state in one sentence why it lowers training *and* held-out loss, and why ordinary
   training curves could not have revealed it.
3. Name the check in the Module 5 lab sequence that would have caught it before any training was run.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import m05  # noqa: E402,F401
from frontierlab.attention.base import register  # noqa: E402
from frontierlab.attention.dsa import DSAttention, index_scores  # noqa: E402

# ------------------------------------------------------------- do not read below this line before step 1


@register("dsa-fast")
class FastDSA(DSAttention):
    """'Faster' selection: one top-k over the whole score matrix row, then a causal mask on the result."""

    def forward(self, x, positions, cache=None):
        B, T, _ = x.shape
        q, k, v = self.project(x, positions)
        iq, ik, w = self.indexer(x, positions)
        k_pos = positions
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat((cache["k"], k), 2), torch.cat((cache["v"], v), 2)
                ik = torch.cat((cache["idx_k"], ik), 1)
                k_pos = torch.cat((cache["pos"], positions))
            cache["k"], cache["v"], cache["idx_k"], cache["pos"] = k, v, ik, k_pos
        scores = index_scores(iq, ik, w)
        idx = scores.topk(min(self.topk, scores.shape[-1]), dim=-1).indices
        sel = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, idx, True)
        y, p = self.attend_masked(q, k, v, sel)
        if self.collect:
            from frontierlab.attention.dsa import indexer_kl
            self.indexer_loss = indexer_kl(p.sum(1).detach().to(scores.dtype), scores, sel)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.hd))


def main():
    parent = Path("runs/m04/base-cpu/checkpoint.pt")
    if not parent.exists():
        sys.exit("needs runs/m04/base-cpu (lesson 04.1: python labs/module-04/lesson-01/train_base.py)")
    out = Path(tempfile.mkdtemp(prefix="m05-bug-"))
    common = ["--preset", "toy", "--steps", "60", "--batch", "16", "--seq", "256", "--lr", "3e-4", "--warmup", "5",
              "--eval-every", "60", "--eval-windows", "64", "--log-every", "20", "--ckpt-every", "60"]
    code = (f"import sys; sys.path.insert(0, {str(HERE)!r}); sys.path.insert(0, {str(HERE.parent)!r}); "
            "import buggy_dsa, train_arm; train_arm.main(sys.argv[1:])")
    for arm in ("b0", "dsa"):
        args = ["--arm", arm, "--init-from", str(parent), *(["--dsa-stage", "sparse"] if arm == "dsa" else []), "--",
                "--run", str(out / arm), *common, *(["--attention", "dsa-fast"] if arm == "dsa" else [])]
        subprocess.check_call([sys.executable, "-c", code, *args])
    import json
    for arm in ("b0", "dsa"):
        rows = [json.loads(line) for line in (out / arm / "metrics.jsonl").read_text().splitlines()]
        val = [r["loss"] for r in rows if r["split"] == "val"][-1]
        print(f"{arm:<4s} held-out loss after 60 steps: {val:.4f}")
    print(f"\nruns in {out}. The report claims the second line beats the first. Now do step 1.")


if __name__ == "__main__":
    main()
