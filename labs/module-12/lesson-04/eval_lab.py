"""Lab 12.4: run Eval Suite v2 after RL and decide whether the stage passes.

    python labs/module-12/lesson-04/eval_lab.py                 # free CPU, about 8 minutes
    python labs/module-12/lesson-04/eval_lab.py --variant main --print

Three RL arms from the same SFT checkpoint, 2 seeds each, 200 steps (GRPO, plain addition only):
``rl`` (lr 3e-4, no KL), ``rl-kl`` (the same plus k2 KL in the loss, beta 1.0) and ``rl-hot`` (lr 3e-3:
an aggressive run). Eval v2 scores the SFT checkpoint and every RL checkpoint on the same held-out items
with the same sampling seed; each RL checkpoint is compared with the SFT checkpoint component by
component (paired by item), with guards stated before the run: 0.02 for accuracies and for the SFT
loss (nats). Then pass@k for k = 1..16 from 16 samples per problem shows where RL's gain lives.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from frontierlab.evals.suite_v2 import core
from frontierlab.evals.suite_v2.toy import run_suite
from frontierlab.labkit import load_path
from frontierlab.posttrain import rl
from frontierlab.posttrain.policy import sample
from frontierlab.posttrain.sft import ensure_sft, load_policy
from frontierlab.posttrain.tasks import encode_prompts, problems_from, score, split_problems

HERE = Path(__file__).resolve().parent
ROOT = Path("runs/m12/l124")
ARMS = {"rl": dict(), "rl-kl": dict(kl_place="loss", kl_kind="k2", kl_beta=1.0), "rl-hot": dict(lr=3e-3)}
GUARDS = {"sub_greedy": 0.02, "sft_nll": 0.02, "if_strict": 0.02, "if_correct": 0.02}


@torch.no_grad()
def pass_curve(lab, policy, n: int = 16, n_items: int = 200) -> list[float]:
    _, held = split_problems(2)
    probs = problems_from(held, ".", 2, "+")[:n_items]
    rep = [p for p in probs for _ in range(n)]
    ro = sample(policy, encode_prompts(rep), 8, 1.0, torch.Generator().manual_seed(99))
    c = score(ro.response, rep).view(len(probs), n).sum(1).tolist()
    return [float(np.mean([lab.pass_at_k(n, int(ci), k) for ci in c])) for k in (1, 2, 4, 8, 16)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["cpu", "main"], default="cpu")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    if a.variant == "main":
        print("python -m frontierlab.posttrain.hf --run runs/m12/l124-main/rl-s0 --steps 200")
        print("python -m frontierlab.posttrain.hf --run runs/m12/l124-main/rl-kl-s0 --steps 200 --kl-place loss "
              "--kl-kind k2 --kl-beta 0.05")
        print("python labs/module-12/lesson-04/eval_main.py --model runs/m12/l124-main/rl-s0/policy "
              "--base Qwen/Qwen3-1.7B-Base --out runs/m12/l124-main/eval")
        return
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    sft = ensure_sft("runs/m12/sft")
    ROOT.mkdir(parents=True, exist_ok=True)
    base_res = run_suite(load_policy(sft))
    out = {"sft": {"summary": base_res["summary"], "pass_curve": pass_curve(lab, load_policy(sft))}}
    print("SFT  " + "  ".join(f"{k} {v:.3f}" for k, v in base_res["summary"].items()))
    for arm, over in ARMS.items():
        for s in (0, 1):
            cfg = replace(rl.RLConfig(init=str(sft), steps=200, eval_every=50), **over, seed=s,
                          run=str(ROOT / f"{arm}-s{s}"))
            if not (Path(cfg.run) / "policy.pt").exists():
                rl.train(cfg)
            pol = load_policy(Path(cfg.run) / "policy.pt")
            res = run_suite(pol)
            cmp = core.compare(base_res, res, guards=GUARDS)
            for name, v in cmp.items():                       # your rule must agree with the suite's
                if not name.startswith("_"):
                    v["verdict_lab"] = lab.verdict(v["kind"], *v["ci"], v["guard"] or 0.0)
            out[f"{arm}-s{s}"] = {"summary": res["summary"], "compare": cmp, "pass_curve": pass_curve(lab, pol)}
            print(f"\n== {arm} seed {s} vs SFT")
            print(core.report(cmp))
    print("\npass@k on held-out addition (16 samples per problem), k = 1, 2, 4, 8, 16")
    for name, v in out.items():
        print(f"  {name:10s} " + " ".join(f"{x:.3f}" for x in v["pass_curve"]))
    (ROOT / "results.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
