"""Lab 14.1: the five objectives on one hand-sized batch, then inside the course loop.

    python labs/module-14/lesson-01/objectives_lab.py                 # free CPU, about 4 minutes
    python labs/module-14/lesson-01/objectives_lab.py --part table    # the hand-sized batch only (seconds)
    python labs/module-14/lesson-01/objectives_lab.py --variant main --print

Part ``table``: the per-token gradient weight (minus d loss / d log pi) of each of *your* losses on the
lesson's worked example, next to the values derived by hand, and on a near-on-policy version of it.

Part ``loop``: each of your losses drives Module 12's loop (``frontierlab.posttrain.rl``) through its
``policy_loss`` hook for 80 steps from the SFT start, seed 0, with 2 epochs x 4 minibatches per batch so
that most minibatches are off-policy (the ratio is not 1). This is a smoke test with diagnostics, not a
comparison: one seed, untuned. Lesson 14.2 runs the controlled comparison. Watch ``clip_frac`` (what fraction
of tokens each objective silences or caps) and ``ratio_max``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from frontierlab.labkit import load_path
from frontierlab.posttrain import rl
from frontierlab.posttrain.sft import ensure_sft
from frontierlab.rlscale import runner
from frontierlab.rlscale.objectives import OBJECTIVES
from frontierlab.rlscale.stability import run_stability

HERE = Path(__file__).resolve().parent
ROOT = Path("runs/m14/l141")
NAMES = ["grpo", "dapo", "drgrpo", "gspo", "cispo"]
HAND = {"grpo": [[0, 0.225, 0], [0, -0.1833, -0.1667]], "dapo": [[0, 0.18, 0], [0, -0.22, -0.2]],
        "drgrpo": [[0, 0.1125, 0], [0, -0.1375, -0.125]], "gspo": [[0, 0, 0], [0, 0, 0]],
        "cispo": [[0.256, 0.18, 0], [-0.14, -0.22, -0.2]]}


def weights(f, logp, old, adv, mask, **kw):
    lp = logp.clone().requires_grad_(True)
    f(lp, old, adv, mask, **kw)[0].backward()
    return -lp.grad


def table(lab):
    old = torch.zeros(2, 3, dtype=torch.float64)
    mask = torch.tensor([[1.0, 1, 0], [1, 1, 1]], dtype=torch.float64)
    adv = torch.tensor([1.0, -1.0], dtype=torch.float64)
    for title, ratios in (("worked example: ratios (1.5, 0.9 | 0.7, 1.1, 1.0)", [[1.5, 0.9, 1.0], [0.7, 1.1, 1.0]]),
                          ("near on-policy: ratios (1.0002, 1.0004 | 0.9999, 1.0001, 1.0)",
                           [[1.0002, 1.0004, 1.0], [0.9999, 1.0001, 1.0]])):
        logp = torch.log(torch.tensor(ratios, dtype=torch.float64))
        print(f"\n{title}; A = (+1, -1); masked tokens shown")
        print(f"  {'objective':8s} {'resp 1 (2 tokens)':>20s} {'resp 2 (3 tokens)':>28s}")
        for n in NAMES:
            kw = {"norm_len": 4} if n == "drgrpo" else {}
            w = weights(getattr(lab, f"{n}_loss"), logp, old, adv, mask, **kw)
            line = f"  {n:8s} " + " ".join(f"{float(w[0, t]):+.4f}" for t in range(2)) + "   " + \
                   " ".join(f"{float(w[1, t]):+.4f}" for t in range(3))
            if title.startswith("worked"):
                ok = torch.allclose(w, torch.tensor(HAND[n], dtype=torch.float64), atol=1e-4)
                line += "   matches hand values" if ok else "   DIFFERS from the hand values"
            print(line)


def loop(lab):
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    sft = ensure_sft("runs/m12/sft")
    base = rl.RLConfig(init=str(sft), steps=80, epochs=2, minibatches=4, eval_every=40, eval_n=300)
    print(f"\n{'objective':8s} {'eval acc':>9s} {'clip_frac':>10s} {'ratio_max':>10s} {'entropy':>8s} {'KL':>7s}  seconds")
    for n in NAMES:
        cfg = runner.objective_config(base, n, seed=0, run=str(ROOT / f"{n}-s0"))
        if not (Path(cfg.run) / "policy.pt").exists():
            runner.train_objective(cfg, n, loss_fn=getattr(lab, f"{n}_loss"))
        from frontierlab.metrics.jsonl import read_jsonl
        rows = read_jsonl(Path(cfg.run) / "metrics.jsonl")
        tr = [r for r in rows if r["split"] == "train"]
        ev = [r for r in rows if r["split"] == "eval"]
        st = run_stability(cfg.run)
        print(f"{n:8s} {ev[-1]['sampled_acc']:9.3f} {st['clip_frac']:10.4f} {st['ratio_max']:10.3f} "
              f"{tr[-1]['entropy']:8.3f} {st['kl_final']:7.3f}  {tr[-1]['seconds']:.0f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["all", "table", "loop"], default="all")
    ap.add_argument("--variant", choices=["cpu", "main", "t4"], default="cpu")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    if a.variant != "cpu":
        extra = "" if a.variant == "main" else (" --model Qwen/Qwen3-0.6B-Base --revision "
                                                "da87bfb608c14b7cf20ba1ce41287e8de496c0cd --prompts 8 --max-new 256")
        for n in NAMES:
            print(f"python -m frontierlab.rlscale.hf_rl --objective {n} --steps 50 --eval-every 50 "
                  f"--run runs/m14/l141-{a.variant}/{n}-s0 --seed 0{extra}")
        return
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    if a.part in ("all", "table"):
        table(lab)
    if a.part in ("all", "loop"):
        loop(lab)


if __name__ == "__main__":
    main()
