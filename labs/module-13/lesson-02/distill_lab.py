"""Lab 13.2: SFT on teacher outputs vs on-policy distillation vs RL, with the teacher's compute counted.

    python labs/module-13/lesson-02/distill_lab.py                   # free CPU, about 20 minutes (teacher included)
    python labs/module-13/lesson-02/distill_lab.py --variant main --print

Student: Module 12's SFT checkpoint (308,400 parameters, about 30% greedy accuracy on plain addition).
Teacher: a 1,174,880-parameter model of the same layout and tokenizer, trained here by SFT on gold targets
to about 97-99% on every operation and tag (``runs/m13/teacher``; its training compute is reported
separately, as a one-off cost).

Every arm sees the same 200 steps x 16 prompts of plain addition (training triples only), 2 seeds:

* ``sft-teacher``  off-policy distillation: the teacher writes 8 responses per prompt (temperature 1), the
                   student is fine-tuned on them (200 steps of 128, lr 3e-4)
* ``opd``          on-policy distillation, sampled per-token reverse KL (your ``sampled_rkl_loss``), 8 student
                   samples per prompt, one update per batch, lr 3e-4
* ``opd-exact``    on-policy distillation, exact per-position reverse KL (your ``reverse_kl``)
* ``rl``           GRPO with the strict verifier (Module 12's loop, its defaults: lr 3e-4, group 8)
* ``opd-self``     control: on-policy distillation from a copy of the *student* (a teacher with nothing to teach)

Each final student is scored by Eval Suite v2 against the SFT start; every arm's FLOPs are counted.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import torch

from frontierlab.evals.suite_v2 import core
from frontierlab.labkit import load_path
from frontierlab.pipeline import distill as DI
from frontierlab.pipeline import toy
from frontierlab.pipeline.compute import Ledger, n_params
from frontierlab.pipeline.seqs import train_sft
from frontierlab.posttrain import rl
from frontierlab.posttrain.sft import ensure_sft, load_policy, save_policy
from frontierlab.posttrain.tasks import make_problems, split_problems

HERE = Path(__file__).resolve().parent
ROOT = Path("runs/m13/l132")
STEPS, PROMPTS, GROUP, LR = 200, 16, 8, 3e-4
ARMS = ("sft-teacher", "opd", "opd-exact", "rl", "opd-self")


def problem_fn(seed):
    _, held = split_problems(2)
    return lambda step: make_problems(PROMPTS, 2, "+", ".", seed=10_000 * (seed + 1) + step, exclude=held)


def check(lab):
    """Your functions must agree with the course's on real logits before any result counts."""
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(4, 6, 38, generator=g), torch.randn(4, 6, 38, generator=g)
    assert torch.allclose(lab.reverse_kl(a, b), DI.exact_reverse_kl(a, b), atol=1e-5), "reverse_kl disagrees"
    lp, tl, m = -torch.rand(4, 6, generator=g), -torch.rand(4, 6, generator=g), torch.ones(4, 6)
    ref, _ = DI.opd_loss(lp, lp, tl, m)
    assert abs(float(lab.sampled_rkl_loss(lp, tl, m)) - float(ref)) < 1e-5, "sampled_rkl_loss disagrees"


def run_arm(lab, arm, seed, base_path, teacher, res0):
    run = ROOT / f"{arm}-s{seed}"
    if (run / "result.json").exists():
        return json.loads((run / "result.json").read_text())
    run.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    student = load_policy(base_path)
    Ns, Nt = n_params(student), n_params(teacher)
    led = Ledger()
    pf = problem_fn(seed)
    if arm == "sft-teacher":
        probs = [p for s in range(1, STEPS + 1) for p in pf(s)]
        ex = DI.teacher_examples(teacher, probs, n=GROUP, seed=seed, ledger=led)
        train_sft(student, ex, steps=STEPS, batch=PROMPTS * GROUP, lr=LR, seed=seed, ledger=led)
        hist = {}
    elif arm in ("opd", "opd-exact", "opd-self"):
        t = copy.deepcopy(student) if arm == "opd-self" else teacher
        if arm == "opd-exact":
            fn = lambda logp, tl, mask, sl, tlog: (lab.reverse_kl(sl, tlog) * mask).sum() / mask.sum()
        else:
            fn = lambda logp, tl, mask, sl, tlog: lab.sampled_rkl_loss(logp, tl, mask)
        out = DI.train_opd(student, t, pf, STEPS, prompts=PROMPTS, group=GROUP, lr=LR,
                           kind="exact" if arm == "opd-exact" else "sampled", seed=seed, ledger=led, loss_fn=fn)
        hist = out["history"]
    elif arm == "rl":
        cfg = rl.RLConfig(init=str(base_path), run=str(run / "rl"), steps=STEPS, prompts=PROMPTS, group=GROUP,
                          lr=LR, seed=seed, eval_every=STEPS)
        rl.train(cfg)
        student = load_policy(run / "rl" / "policy.pt")
        led = DI.rl_ledger(run / "rl", Ns)
        hist = {}
    res = toy.eval_v2(student)
    cmp = core.compare(res0, res, guards=toy.GUARDS)
    save_policy(student, run / "policy.pt", {"arm": arm, "seed": seed})
    r = {"arm": arm, "seed": seed, "summary": res["summary"], "compare": cmp, "ledger": led.to_dict(),
         "history": hist, "seconds": round(time.perf_counter() - t0, 1),
         "flops_lab": lab.arm_flops(Ns, Ns if arm == "opd-self" else Nt, student_sampled=led.tokens.get("student_sample", 0.0),
                                    student_trained=led.tokens.get("student_train", 0.0),
                                    teacher_generated=led.tokens.get("teacher_sample", 0.0),
                                    teacher_scored=led.tokens.get("teacher_score", 0.0))}
    (run / "result.json").write_text(json.dumps(r, default=str))
    return r


MAIN = [
    "# Main path (not run in this build; part of the Module 13 pilot). Student: the project's RLVR or SFT checkpoint",
    "# of Qwen3-1.7B-Base (or Qwen3-1.7B-Base itself); teacher: Qwen/Qwen3-8B b968826d (non-thinking mode), same tokenizer.",
    "python -m frontierlab.pipeline.hf_stages distill --mode offline --student Qwen/Qwen3-1.7B-Base "
    "--teacher Qwen/Qwen3-8B --steps 100 --out runs/m13/l132-main/sft-teacher-s0",
    "python -m frontierlab.pipeline.hf_stages distill --mode onpolicy --student Qwen/Qwen3-1.7B-Base "
    "--teacher Qwen/Qwen3-8B --steps 100 --out runs/m13/l132-main/opd-s0",
    "python -m frontierlab.posttrain.hf --run runs/m13/l132-main/rl-s0 --steps 100",
    "python -m frontierlab.pipeline.hf_stages distill --mode onpolicy --student Qwen/Qwen3-1.7B-Base "
    "--teacher Qwen/Qwen3-1.7B-Base --steps 100 --out runs/m13/l132-main/opd-self-s0     # control",
    "# then Eval v2 for each: python -m frontierlab.pipeline.hf_eval score --model <run>/policy --out <run>/eval.json",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["cpu", "main"], default="cpu")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args(argv)
    if a.variant == "main":
        print("\n".join(MAIN))
        return
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    check(lab)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    t0 = time.perf_counter()
    base_path = ensure_sft("runs/m12/sft")
    teacher = load_policy(DI.ensure_teacher("runs/m13/teacher"))
    tmeta = json.loads(Path("runs/m13/teacher/teacher.json").read_text())
    print(f"teacher: {tmeta['params']:,} parameters, held-out greedy accuracy "
          + " ".join(f"{k} {v:.2f}" for k, v in tmeta["heldout_greedy_acc"].items())
          + f"; training {tmeta['train_flops']:.2e} FLOPs (one-off)")
    ROOT.mkdir(parents=True, exist_ok=True)
    f0 = ROOT / "sft-eval.json"
    if not f0.exists():
        f0.write_text(json.dumps(toy.eval_v2(load_policy(base_path))))
    res0 = json.loads(f0.read_text())
    out = {}
    for s in range(a.seeds):
        for arm in ARMS:
            r = run_arm(lab, arm, s, base_path, teacher, res0)
            out[f"{arm}-s{s}"] = r
            c = r["compare"]
            led = Ledger.from_dict(r["ledger"])
            print(f"{arm:12s} s{s}  add_pass1 {c['add_pass1']['diff']:+.3f} [{c['add_pass1']['ci'][0]:+.3f}, "
                  f"{c['add_pass1']['ci'][1]:+.3f}]  add_greedy {c['add_greedy']['diff']:+.3f}  "
                  f"sft_nll {c['sft_nll']['diff']:+.3f}  sub {c['sub_greedy']['diff']:+.3f}  "
                  f"guard {'PASS' if c['_passes'] else 'FAIL'}  | {led.line()}  ({r['seconds']} s)")
    (ROOT / "results.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"\ntotal {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
