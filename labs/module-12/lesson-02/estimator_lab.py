"""Lab 12.2: compare advantage estimators, KL placements and an entropy bonus in the course RL loop.

    python labs/module-12/lesson-02/estimator_lab.py                   # free CPU, about 20 minutes
    python labs/module-12/lesson-02/estimator_lab.py --part estimators # only the first comparison
    python labs/module-12/lesson-02/estimator_lab.py --variant main --print   # main-path commands

Part 1, estimators (3 seeds, 120 steps each, everything else fixed): ``reinforce`` (no baseline),
``grpo`` (group mean and std), ``rloo`` (leave-one-out), ``drgrpo`` (group mean, no std). The advantages
come from *your* ``group_advantages`` (passed to the loop as a hook).

Part 2, KL (2 seeds, GRPO, beta = 0.5): none; k1, k2, k3 in the loss; k1 in the reward. The loop logs the
exact per-token reverse KL to the SFT policy (``kl_exact``, summed over the 35-token vocabulary), so the
estimators can be judged against the quantity they are meant to control.

Part 3, entropy (3 seeds): GRPO with an entropy bonus of 0.02 against Part 1's GRPO.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import torch

from frontierlab.labkit import load_path
from frontierlab.posttrain import arms as AR
from frontierlab.posttrain import rl
from frontierlab.posttrain.sft import ensure_sft

HERE = Path(__file__).resolve().parent
ROOT = Path("runs/m12/l122")

ESTIMATORS = {"reinforce": dict(baseline="none", scale="none"), "grpo": dict(baseline="mean", scale="group"),
              "rloo": dict(baseline="loo", scale="none"), "drgrpo": dict(baseline="mean", scale="none")}
KL = {"kl-none": dict(), "kl-loss-k1": dict(kl_place="loss", kl_kind="k1", kl_beta=0.5),
      "kl-loss-k2": dict(kl_place="loss", kl_kind="k2", kl_beta=0.5),
      "kl-loss-k3": dict(kl_place="loss", kl_kind="k3", kl_beta=0.5),
      "kl-reward-k1": dict(kl_place="reward", kl_beta=0.5)}
ENTROPY = {"grpo": dict(), "grpo-ent": dict(entropy_coef=0.02)}

MAIN = ("python -m frontierlab.posttrain.hf --run runs/m12/l122-main/{arm}-s{seed} --seed {seed} --steps 100 "
        "--baseline {baseline} --scale {scale}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["all", "estimators", "kl", "entropy"], default="all")
    ap.add_argument("--variant", choices=["cpu", "main"], default="cpu")
    ap.add_argument("--print", action="store_true", help="print the main-path commands instead of running")
    a = ap.parse_args(argv)
    if a.variant == "main":
        for arm, kw in ESTIMATORS.items():
            for seed in (0, 1):
                print(MAIN.format(arm=arm, seed=seed, **kw))
        print("# KL arms: add --kl-place loss --kl-kind k3 --kl-beta 0.05 (and k2; --kl-place reward) to the grpo line")
        return
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    sft = ensure_sft("runs/m12/sft")
    base = rl.RLConfig(init=str(sft), steps=120, eval_every=40, eval_n=300)
    hooks = {"advantages": lab.group_advantages}
    original = rl.train
    rl_train = lambda cfg, hooks=hooks: original(cfg, hooks)        # noqa: E731
    AR.train = rl_train                                             # the arms runner uses your advantages
    out = {}
    if a.part in ("all", "estimators", "entropy"):
        out["estimators"] = AR.run_arms(base, ESTIMATORS, [0, 1, 2], ROOT / "est")
        print(AR.table(out["estimators"], "final_sampled", "grpo"))
        print(AR.table(out["estimators"], "train_pass_auc", "grpo"))
        print(AR.table(out["estimators"], "zero_var_frac"))
        print(AR.table(out["estimators"], "entropy"))
        for arm in ("reinforce", "rloo", "drgrpo"):
            d = AR.paired_diff(out["estimators"], arm, "grpo", "final_sampled")
            print(f"  decision {arm} vs grpo on held-out sampled accuracy: {lab.decide(d['mean'], *d['ci'])}")
    if a.part in ("all", "kl"):
        out["kl"] = AR.run_arms(replace(base), KL, [0, 1], ROOT / "kl")
        print(AR.table(out["kl"], "kl_exact", "kl-none"))
        print(AR.table(out["kl"], "kl_k3"))
        print(AR.table(out["kl"], "final_sampled", "kl-none"))
    if a.part in ("all", "entropy"):
        ent = AR.run_arms(base, {"grpo-ent": ENTROPY["grpo-ent"]}, [0, 1, 2], ROOT / "ent")
        ent["grpo"] = out["estimators"]["grpo"]
        out["entropy"] = ent
        print(AR.table(ent, "entropy", "grpo"))
        print(AR.table(ent, "final_sampled", "grpo"))
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / f"results-{a.part}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
