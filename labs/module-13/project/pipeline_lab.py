"""Module 13 project: SFT -> DPO -> RLVR -> distillation, with Eval v2 after every stage, then the short form on Recipe-R.

    python labs/module-13/project/pipeline_lab.py toy               # free CPU, about 10 minutes (2 seeds)
    python labs/module-13/project/pipeline_lab.py recipe-r          # free CPU, about 12 minutes (needs Module 11's checkpoint)
    python labs/module-13/project/pipeline_lab.py main --print      # the main-path commands (Qwen3-1.7B-Base)

**Toy pipeline** (seeds 0 and 1; each stage starts from the previous stage's checkpoint):

====  =====================  ===============================================================================
S0    SFT                    Module 12's warm start (``runs/m12/sft``; shared by both seeds)
S1    preference (DPO)       DPO + NLL (lesson 13.1's winning arm) on 8,000 prompts x 4 rated samples
S2    RLVR                   Module 12's GRPO loop, 200 steps, k2 KL in the loss (beta 1.0, lesson 12.4's passing arm)
S2c   RLVR control           the same from S1 with Bernoulli(0.5) rewards (Shao et al.'s random reward)
S3    distillation           SFT on teacher outputs (lesson 13.2's winning arm), all operations and tags
====  =====================  ===============================================================================

Every stage is scored by Eval Suite v2 and compared with the stage before it and with S0 (paired by item,
guards 0.02); every stage's FLOPs go into one ledger per seed, the teacher's sampling included.

**Recipe-R short form** (:mod:`frontierlab.pipeline.recipe_r`, seed 0): SFT with 25% pretraining-text replay ->
DPO + NLL -> 60 GRPO steps, and the random-reward control; no distillation (no teacher shares Recipe-R's
tokenizer). The retention component is Data-v0 validation loss.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from frontierlab.evals.suite_v2 import core
from frontierlab.pipeline import distill as DI
from frontierlab.pipeline import dpo as D
from frontierlab.pipeline import toy
from frontierlab.pipeline.compute import Ledger, n_params
from frontierlab.pipeline.seqs import Example, train_sft
from frontierlab.posttrain import advantages as A
from frontierlab.posttrain import rl
from frontierlab.posttrain.sft import ensure_sft, load_policy, save_policy

ROOT = Path("runs/m13/project")
RR_RL_LR = 3e-5          # Recipe-R's GRPO learning rate (3e-4, the toy loop's, collapsed the pretrained model)


def random_reward_hook(seed: int):
    g = torch.Generator().manual_seed(seed)

    def adv(R, baseline, scale):
        Rr = (torch.rand(R.shape, generator=g) < 0.5).float()
        return A.group_advantages(Rr, baseline, scale)
    return adv


def score_stage(name, model, prev_res, s0_res, out):
    res = toy.eval_v2(model)
    out[name] = {"summary": res["summary"],
                 "vs_prev": core.compare(prev_res, res, guards=toy.GUARDS) if prev_res else None,
                 "vs_s0": core.compare(s0_res, res, guards=toy.GUARDS) if s0_res else None}
    return res


def toy_pipeline(seeds=(0, 1)):
    s0_path = ensure_sft("runs/m12/sft")
    teacher = load_policy(DI.ensure_teacher("runs/m13/teacher"))
    all_out = {}
    for s in seeds:
        root = ROOT / f"toy-s{s}"
        f = root / "pipeline.json"
        if f.exists():
            all_out[s] = json.loads(f.read_text())
            continue
        root.mkdir(parents=True, exist_ok=True)
        out, led, t0 = {}, Ledger(), time.perf_counter()
        s0 = load_policy(s0_path)
        r0 = score_stage("S0-sft", s0, None, None, out)
        # S1: DPO + NLL on on-policy pairs rated on two aspects
        probs = toy.train_problems(8000, seed=11 + s)
        cands = toy.sample_texts(s0, probs, 4, 1.0, seed=3 + s, ledger=led)
        pairs, _ = toy.build_pairs(probs, cands, toy.aspect_score, seed=s)
        s1 = copy.deepcopy(s0)
        D.train_dpo(s1, s0, pairs, 150, beta=0.1, nll_coef=0.2, lr=5e-5, seed=s, ledger=led)
        save_policy(s1, root / "s1-dpo" / "policy.pt")
        r1 = score_stage("S1-dpo", s1, r0, r0, out)
        # S2: RLVR from S1, and the random-reward control from S1
        base_cfg = rl.RLConfig(init=str(root / "s1-dpo" / "policy.pt"), steps=200, seed=s, eval_every=200,
                               kl_place="loss", kl_kind="k2", kl_beta=1.0)
        cfg = copy.copy(base_cfg); cfg.run = str(root / "s2-rlvr")
        rl.train(cfg)
        led.merge(DI.rl_ledger(cfg.run, n_params(s1)))
        s2 = load_policy(Path(cfg.run) / "policy.pt")
        r2 = score_stage("S2-rlvr", s2, r1, r0, out)
        cc = copy.copy(base_cfg); cc.run = str(root / "s2c-random")
        rl.train(cc, hooks={"advantages": random_reward_hook(100 + s)})
        score_stage("S2c-random-reward", load_policy(Path(cc.run) / "policy.pt"), r1, r0, out)
        # S3: distillation from the teacher, every operation and tag
        s3 = copy.deepcopy(s2)
        dprobs = toy.train_problems(200 * 16, seed=600 + s)
        ex = DI.teacher_examples(teacher, dprobs, n=8, seed=s, ledger=led)
        train_sft(s3, ex, steps=200, batch=128, lr=3e-4, seed=s, ledger=led)
        save_policy(s3, root / "s3-distill" / "policy.pt")
        score_stage("S3-distill", s3, r2, r0, out)
        out["ledger"] = led.to_dict()
        out["seconds"] = round(time.perf_counter() - t0, 1)
        f.write_text(json.dumps(out, indent=1, default=str))
        all_out[s] = out
    for s, out in all_out.items():
        print(f"\n===== toy pipeline, seed {s} ({out['seconds']} s; {Ledger.from_dict(out['ledger']).line()})")
        for st in ("S0-sft", "S1-dpo", "S2-rlvr", "S2c-random-reward", "S3-distill"):
            sm = out[st]["summary"]
            line = f"{st:18s} " + "  ".join(f"{k} {sm[k]:.3f}" for k in ("add_pass1", "add_greedy", "sub_greedy",
                                                                         "sft_nll", "if_correct"))
            v = out[st]["vs_prev"]
            if v:
                line += f"  | vs previous: guard {'PASS' if v['_passes'] else 'FAIL'}, add_pass1 {v['add_pass1']['diff']:+.3f} " \
                        f"[{v['add_pass1']['ci'][0]:+.3f}, {v['add_pass1']['ci'][1]:+.3f}]"
            print(line)
    return all_out


def recipe_r_pipeline(seed: int = 0, replay: float = 0.25):
    from frontierlab.pipeline import recipe_r as R
    if not R.DEFAULT_CKPT.exists():
        raise SystemExit(f"Recipe-R checkpoint not found at {R.DEFAULT_CKPT}: run the Module 11 project first "
                         "(or pass your own checkpoint path in recipe_r.DEFAULT_CKPT).")
    f = ROOT / "recipe-r" / "pipeline.json"
    if f.exists():
        out = json.loads(f.read_text())
    else:
        f.parent.mkdir(parents=True, exist_ok=True)
        tok, led, out, t0 = R.RTok(), Ledger(), {}, time.perf_counter()
        m = R.load_recipe_r()
        prev = R.suite(m, tok)
        base = prev
        out["R0-recipe-r"] = {"summary": prev["summary"]}

        def stage(name, model):
            nonlocal prev
            res = R.suite(model, tok)
            out[name] = {"summary": res["summary"], "vs_prev": core.compare(prev, res, guards={"val_nll": 0.05}),
                         "vs_r0": core.compare(base, res, guards={"val_nll": 0.05})}
            prev = res
            return res
        # SFT with replay of pretraining text (the retention lever)
        data = np.memmap(R.DATA / "train.bin", dtype=np.uint16, mode="r")
        rng = np.random.default_rng(seed)
        ex = [R.example(tok, p) for p in R.train_problems(20000, seed + 300)]
        k = int(len(ex) * replay / (1 - replay))
        ex += [Example.of([], np.asarray(data[st:st + 64], dtype=np.int64).tolist())
               for st in rng.integers(0, len(data) - 65, size=k)]
        train_sft(m, ex, 1500, batch=64, lr=3e-4, seed=seed, ledger=led)
        stage("R1-sft", m)
        dp = R.stage_dpo(m, tok, steps=150, seed=seed, ledger=led)
        out["dpo_pairs"] = dp["pairs"]
        r2 = stage("R2-dpo", m)
        r2_state = copy.deepcopy(m.state_dict())
        torch.save(r2_state, f.parent / "r2-dpo.pt")
        R.stage_rlvr(m, tok, steps=60, lr=RR_RL_LR, seed=seed, ledger=led)
        stage("R3-rlvr", m)
        mc = R.load_recipe_r()
        mc.load_state_dict(r2_state)
        R.stage_rlvr(mc, tok, steps=60, lr=RR_RL_LR, seed=seed, control="random")
        res_c = R.suite(mc, tok)
        out["R3c-random-reward"] = {"summary": res_c["summary"],
                                    "vs_prev": core.compare(r2, res_c, guards={"val_nll": 0.05})}
        out["ledger"] = led.to_dict()
        out["seconds"] = round(time.perf_counter() - t0, 1)
        f.write_text(json.dumps(out, indent=1, default=str))
    print(f"\n===== Recipe-R short pipeline ({out['seconds']} s; {Ledger.from_dict(out['ledger']).line()}; "
          f"{out['dpo_pairs']} DPO pairs)")
    for st in ("R0-recipe-r", "R1-sft", "R2-dpo", "R3-rlvr", "R3c-random-reward"):
        sm = out[st]["summary"]
        line = f"{st:18s} add_greedy {sm['add_greedy']:.3f}  sub_greedy {sm['sub_greedy']:.3f}  val_nll {sm['val_nll']:.3f}"
        v = out[st].get("vs_prev")
        if v:
            line += f"  | vs previous: val_nll {v['val_nll']['diff']:+.3f} ({v['val_nll']['verdict']})"
        print(line)
    return out


MAIN = [
    "# Main path (not run in this build; part of the Module 13 pilot). Stage D base: Qwen/Qwen3-1.7B-Base ea980cb0.",
    "python -m frontierlab.pipeline.hf_eval score --model Qwen/Qwen3-1.7B-Base --revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 --out runs/m13/main/s0-plain.json",
    "python -m frontierlab.pipeline.hf_stages sft --model Qwen/Qwen3-1.7B-Base --steps 1000 --batch 32 --out runs/m13/main/sft-s0",
    "python -m frontierlab.pipeline.hf_eval score --model runs/m13/main/sft-s0/policy --out runs/m13/main/s1-plain.json",
    "python -m frontierlab.pipeline.hf_eval score --model runs/m13/main/sft-s0/policy --chat --out runs/m13/main/s1-chat.json",
    "python -m frontierlab.pipeline.hf_stages dpo --model runs/m13/main/sft-s0/policy --steps 300 --nll 0.2 --out runs/m13/main/dpo-s0",
    "python -m frontierlab.pipeline.hf_eval score --model runs/m13/main/dpo-s0/policy --chat --out runs/m13/main/s2-chat.json",
    "python -m frontierlab.pipeline.hf_stages rlvr --model runs/m13/main/dpo-s0/policy --steps 200 --kl-beta 0.05 --out runs/m13/main/rlvr-s0",
    "python -m frontierlab.pipeline.hf_stages rlvr --model runs/m13/main/dpo-s0/policy --steps 200 --kl-beta 0.05 --control random --out runs/m13/main/rlvr-random-s0",
    "python -m frontierlab.pipeline.hf_eval score --model runs/m13/main/rlvr-s0/policy --chat --out runs/m13/main/s3-chat.json",
    "python -m frontierlab.pipeline.hf_eval score --model runs/m13/main/rlvr-random-s0/policy --chat --out runs/m13/main/s3c-chat.json",
    "python -m frontierlab.pipeline.hf_stages distill --mode onpolicy --student runs/m13/main/rlvr-s0/policy --teacher Qwen/Qwen3-8B --steps 100 --out runs/m13/main/distill-s0",
    "python -m frontierlab.pipeline.hf_eval score --model runs/m13/main/distill-s0/policy --chat --out runs/m13/main/s4-chat.json",
    "# compare consecutive stages in the same format, e.g.:",
    "python -m frontierlab.pipeline.hf_eval compare runs/m13/main/s0-plain.json runs/m13/main/s1-plain.json",
    "python -m frontierlab.pipeline.hf_eval compare runs/m13/main/s1-chat.json runs/m13/main/s2-chat.json",
    "python -m frontierlab.pipeline.hf_eval compare runs/m13/main/s2-chat.json runs/m13/main/s3-chat.json",
    "python -m frontierlab.pipeline.hf_eval compare runs/m13/main/s2-chat.json runs/m13/main/s3c-chat.json",
    "python -m frontierlab.pipeline.hf_eval compare runs/m13/main/s3-chat.json runs/m13/main/s4-chat.json",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["toy", "recipe-r", "main"])
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(8)
    if a.part == "main":
        print("\n".join(MAIN))
    elif a.part == "toy":
        toy_pipeline()
    else:
        recipe_r_pipeline()


if __name__ == "__main__":
    main()
