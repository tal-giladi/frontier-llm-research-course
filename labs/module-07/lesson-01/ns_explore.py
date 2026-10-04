"""Lab 07.1, step 2: what Newton-Schulz does to the singular values of a real gradient.

    python labs/module-07/lesson-01/ns_explore.py                 # uses your lab.py where implemented
    LAB_TARGET=solution python labs/module-07/lesson-01/ns_explore.py

Takes the gradient of two weight matrices of the toy model after one backward pass on Data-v0, prints its
condition number, then the smallest and largest singular value after every step of three schedules
(Jordan's quintic x5, DeepSeek-V4's 8 quintic + 2 (2, -1.5, 0.5), classic cubic x5), in float64 and bf16,
and the relative distance to the exact orthogonal factor U V^T.
"""

import os
import sys
from pathlib import Path

import torch

from frontierlab.data.loader import TokenData
from frontierlab.labkit import load_target
from frontierlab.model import LM, toy
from frontierlab.optim.muon import NS_SCHEDULES, newton_schulz, orthogonal_polar

lab = load_target(str(Path(__file__).parent / "test_lab.py"))


def lab_ns(G):
    try:
        return lab.newton_schulz(G)
    except NotImplementedError:
        return None


def main():
    torch.manual_seed(0)
    data = TokenData("train")
    model = LM(toy(vocab_size=data.meta["vocab_size"]))
    x = data.batch(16, 128, torch.Generator().manual_seed(0))
    model(x, labels=x).loss.backward()
    mats = {"layers.1.self_attn.q_proj (128x128)": model.model.layers[1].self_attn.q_proj.weight.grad,
            "layers.1.mlp.up_proj (384x128)": model.model.layers[1].mlp.up_proj.weight.grad}
    print(f"checking {os.environ.get('LAB_TARGET', 'lab')}.py")
    for name, G in mats.items():
        G = G.double()
        s = torch.linalg.svdvals(G)
        print(f"\n{name}: condition number {s.max() / s.min():.1f}, "
              f"after Frobenius normalisation s in [{s.min() / G.norm():.4f}, {s.max() / G.norm():.4f}]")
        print(f"{'schedule':10s} {'dtype':9s} " + " ".join(f"step{i + 1:>2d} min/max" for i in range(10)))
        for sched in ("quintic5", "v4-hybrid", "cubic5"):
            for dt in (torch.float64, torch.bfloat16):
                cells = []
                for k in range(1, len(NS_SCHEDULES[sched]) + 1):
                    O = newton_schulz(G, NS_SCHEDULES[sched][:k], dtype=dt).double()
                    sv = torch.linalg.svdvals(O)
                    cells.append(f"{sv.min():.2f}/{sv.max():.2f}")
                err = (O - orthogonal_polar(G)).norm() / orthogonal_polar(G).norm()
                print(f"{sched:10s} {str(dt)[6:]:9s} " + " ".join(f"{c:>12s}" for c in cells) + f"   rel. error {err:.3f}")
        mine = lab_ns(G)
        if mine is not None:
            diff = (mine - newton_schulz(G, "quintic5", dtype=torch.float64)).abs().max().item()
            print(f"your newton_schulz vs reference (float64, quintic x5): max |diff| = {diff:.2e}")
        else:
            print("your newton_schulz: not implemented yet (TODO 2)")


if __name__ == "__main__":
    sys.exit(main())
