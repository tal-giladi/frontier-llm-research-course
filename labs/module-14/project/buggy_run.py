"""The debugging task: the colleague's objectives (buggy_objectives.py) in the course loop, next to the course's.

    python labs/module-14/project/buggy_run.py          # about 6 minutes on a laptop CPU
    OBJ_HOOKS=buggy pytest labs/module-14/project       # the derivation tests against their library

Same settings as lesson 14.1's loop part (SFT start, seed 0, 80 steps, 2 epochs x 4 minibatches), so the
course runs of lesson 14.1 (runs/m14/l141) are reused as the reference when they exist.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from frontierlab.labkit import load_path
from frontierlab.metrics.jsonl import read_jsonl
from frontierlab.posttrain import rl
from frontierlab.posttrain.sft import ensure_sft
from frontierlab.rlscale import runner
from frontierlab.rlscale.stability import run_stability

HERE = Path(__file__).resolve().parent


def summary(run):
    rows = read_jsonl(Path(run) / "metrics.jsonl")
    ev = [r for r in rows if r["split"] == "eval"]
    st = run_stability(run)
    return ev[-1]["sampled_acc"], st


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    bug = load_path(str(HERE / "buggy_objectives.py"))
    sft = ensure_sft("runs/m12/sft")
    base = rl.RLConfig(init=str(sft), steps=80, epochs=2, minibatches=4, eval_every=40, eval_n=300)
    print(f"{'objective':16s} {'eval acc':>9s} {'clip_frac':>10s} {'ratio_max':>10s} {'entropy min':>12s} {'KL':>7s}")
    for name in ("grpo", "gspo", "cispo"):
        for label, root, fn in (("course", Path("runs/m14/l141"), None), ("colleague", Path("runs/m14/project-debug"),
                                                                           bug.BUGGY[name])):
            cfg = runner.objective_config(base, name, seed=0, run=str(root / f"{name}-s0"))
            if not (Path(cfg.run) / "policy.pt").exists():
                runner.train_objective(cfg, name, loss_fn=fn)
            acc, st = summary(cfg.run)
            print(f"{name + ' ' + label:16s} {acc:9.3f} {st['clip_frac']:10.4f} {st['ratio_max']:10.2f} "
                  f"{st['entropy_min']:12.3f} {st['kl_final']:7.3f}")


if __name__ == "__main__":
    main()
