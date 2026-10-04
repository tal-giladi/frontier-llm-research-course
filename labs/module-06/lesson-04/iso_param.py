"""Lab 06.4, step 3: a small iso-parameter test of the allocation idea - MoE with 8 experts vs MoE with 6 + Engram.

    python labs/module-06/lesson-04/iso_param.py                  # free CPU, two toy runs, about 8 minutes
    python labs/module-06/lesson-04/iso_param.py --variant main --device cuda     # not run in this build

Arms (toy preset, 200 steps, seed 0, the Module 6 CPU settings):
  moe              fine-grained MoE FFN in layers 1-3: 8 routed experts (top-2, width 96) + 1 shared
  moe6+engram1465  6 routed experts; the 2 x 3 removed experts' 221,184 parameters moved into an Engram module at
                   layer 1 (2- and 3-grams, 2 hash heads, 4 tables of ~1,455 rows x 32): 2,390,080 vs 2,392,448
                   total parameters (-0.1%), allocation ratio rho = 4/6 = 0.67 of the inactive budget
Reports held-out loss (paired over 256 windows), parameters, training FLOPs, and the mean Engram gate value.
This is one point of the paper's U-curve, at one tiny scale, one seed: an illustration, not a test of it.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m06  # noqa: E402
from frontierlab.blocks import accounting  # noqa: E402
from frontierlab.blocks import train as blocks_train  # noqa: E402
from frontierlab.data.loader import TokenData  # noqa: E402

ARMS = ("moe", "moe6+engram1465")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(m06.VARIANTS), default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    v = m06.VARIANTS[a.variant]
    vocab = m06.vocab_size()
    for arm in ARMS:
        m06.train_arm(a.variant, arm, a.seed, question=f"lesson 06.4: MoE vs MoE + Engram at equal parameters, {arm}",
                      device=a.device)
    kw = dict(n=v["eval_n"], T=v["eval_T"], device=a.device)
    losses = {arm: m06.eval_losses(m06.run_dir(a.variant, arm, a.seed), **kw) for arm in ARMS}
    for arm in ARMS:
        pc = accounting.param_counts(m06.arm_config(arm, a.variant, vocab))
        print(f"{arm:16s} held-out {np.mean(losses[arm]):.4f}  total params {pc['total']:,}  active non-emb "
              f"{pc['active_non_embedding']:,}  Engram tables {pc['engram_tables']:,}  train "
              f"{m06.flops_per_token(arm, a.variant, vocab) / 1e6:.3f} MFLOP/token  wall {m06.wallclock(m06.run_dir(a.variant, arm, a.seed)):.0f} s")
    print(f"\nmoe6+engram - moe (paired by window): {m06.fmt(m06.compare(losses[ARMS[1]], losses[ARMS[0]]))}")
    model = blocks_train.load_model(m06.run_dir(a.variant, ARMS[1], a.seed) / "checkpoint.pt").eval()
    val = TokenData("val")
    x = torch.stack([val.window(s, v["eval_T"]) for s in val.eval_windows(16, v["eval_T"])])
    with torch.no_grad():
        model(x)
    g = model.model.engram["1"].last_gate
    print(f"Engram gate alpha at layer 1 on 16 held-out windows: mean {g.mean():.3f}, 10th-90th percentile "
          f"{g.quantile(0.1):.3f}-{g.quantile(0.9):.3f}")


if __name__ == "__main__":
    main()
