"""Lab 13.3: train a mini-spec with an AI judge, count the judge, measure adherence and retention.

    python labs/module-13/lesson-03/spec_lab.py                      # free CPU, about 12 minutes
    python labs/module-13/lesson-03/spec_lab.py --variant main --print

1. **AI feedback and the judge.** 3,000 Spec-T prompts (25% restricted, 25% borderline 80-89, 50% other),
   2 samples each from Module 12's SFT policy plus the refusal ``nil``; a simulated AI labeller labels them
   (5% random flips, and it misreads the boundary on 85-89 80% of the time); a 308k-parameter judge is
   trained on those labels. Its errors are measured against Spec-T itself on held-out prompts.
2. **Training against the judge.** 6,000 prompts, 4 samples each plus ``nil``. Arms (2 seeds each, from the
   SFT start, DPO with beta 0.1 and NLL 0.2, lr 3e-4, 300 steps of 64 pairs, unless stated):
   * ``judge-dpo``          pairs ranked by the judge (RLAIF, DPO form)
   * ``judge+verifier-dpo`` pairs ranked by the judge plus 0.5 if a verifier says the number is right
   * ``judge-sft``          SFT on the judge's best candidate per prompt (Constitutional AI's SL stage, in miniature)
   * ``oracle-dpo``         pairs ranked by Spec-T itself (the "human labels" upper bound)
   * ``random-dpo``         control: pairs ranked at random
3. **Measure.** Adherence on 600 held-out prompts (greedy, scored by Spec-T), Eval v2 against the SFT start,
   and the FLOPs of each arm with the judge's training and scoring counted.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from frontierlab.evals.suite_v2 import core
from frontierlab.labkit import load_path
from frontierlab.pipeline import dpo as D
from frontierlab.pipeline import judge as J
from frontierlab.pipeline import toy
from frontierlab.pipeline.compute import Ledger
from frontierlab.pipeline.seqs import train_sft
from frontierlab.posttrain.sft import ensure_sft, load_policy, save_policy

HERE = Path(__file__).resolve().parent
ROOT = Path("runs/m13/l133")
ARMS = ("judge-dpo", "judge+verifier-dpo", "judge-sft", "oracle-dpo", "random-dpo")
STEPS, LR, BATCH = 300, 3e-4, 64


def flat(problems, cands):
    P, T, F = [], [], []
    for p, cs in zip(problems, cands):
        for t, f in list(dict.fromkeys(list(cs) + [(J.REFUSAL, True)])):
            P.append(p); T.append(t); F.append(f)
    return P, T, F


def check(lab, P, T, F):
    for p, t, f in zip(P[:3000], T, F):
        if lab.spec_check(p.a, p.tag, t, f) != J.spec_check(p, t, f):
            raise SystemExit(f"your spec_check disagrees with Spec-T on {p.prompt!r} -> {t!r} ({f})")


def build_judge(lab, base, led):
    jp = J.spec_problems(3000, seed=21)
    P, T, F = flat(jp, toy.sample_texts(base, jp, 2, 1.0, seed=5, ledger=led))
    y = J.ai_feedback(P, T, F, seed=0)
    judge, st = J.train_judge(P, T, F, y, steps=400, ledger=led)
    hp = J.heldout_spec_problems(200)
    HP, HT, HF = flat(hp, toy.sample_texts(base, hp, 2, 1.0, seed=9))
    check(lab, HP, HT, HF)
    probs = J.judge_probs(judge, HP, HT, HF)
    truth = np.array([J.spec_check(p, t, f)[0] for p, t, f in zip(HP, HT, HF)], float)
    rep = {**J.judge_report(probs, HP, HT, HF), **{f"lab_{k}": v for k, v in lab.judge_errors(probs, truth).items()},
           "label_agreement_with_spec": float((y == np.array([J.spec_check(p, t, f)[0] for p, t, f in zip(P, T, F)])).mean()),
           "train": st, "n_labels": len(y)}
    return judge, rep


def pairs_by(lab, problems, cands, score):
    pairs = []
    for p, cs in zip(problems, cands):
        pool = list(dict.fromkeys(list(cs) + [(J.REFUSAL, True)]))
        pick = lab.pair_from_scores([score(p, t, f) for t, f in pool])
        if pick is not None:
            (ct, cf), (rt, rf) = pool[pick[0]], pool[pick[1]]
            pairs.append((toy.example(p, ct, cf), toy.example(p, rt, rf)))
    return pairs


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
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    t0 = time.perf_counter()
    base = load_policy(ensure_sft("runs/m12/sft"))
    ROOT.mkdir(parents=True, exist_ok=True)
    judge_led = Ledger()
    judge, rep = build_judge(lab, base, judge_led)
    (ROOT / "judge.json").write_text(json.dumps(rep, indent=1))
    print("judge vs Spec-T on held-out candidates: " + ", ".join(
        f"{k} {rep[k]:.3f}" for k in ("accuracy", "false_accept", "false_reject", "accuracy_restricted",
                                       "accuracy_borderline", "accuracy_normal", "accepts_refusal_85_89")))
    print(f"labeller agreement with Spec-T {rep['label_agreement_with_spec']:.3f} on {rep['n_labels']} labels; "
          f"judge cost {judge_led.line()}")
    res0 = toy.eval_v2(base)
    adh0 = J.adherence(base)
    print("SFT start adherence: " + ", ".join(f"{k} {v:.3f}" for k, v in adh0.items() if k != "items"))
    out = {"judge": rep, "sft": {"adherence": adh0, "summary": res0["summary"]}}
    for s in range(a.seeds):
        tp = J.spec_problems(6000, seed=31 + s)
        cand_led = Ledger()
        tc = toy.sample_texts(base, tp, 4, 1.0, seed=7 + s, ledger=cand_led)
        P, T, F = flat(tp, tc)
        score_led = Ledger()
        jp_ = J.judge_probs(judge, P, T, F, ledger=score_led)
        look = {(id(p), t, f): v for p, t, f, v in zip(P, T, F, jp_)}
        rng = np.random.default_rng(100 + s)
        scorers = {
            "judge-dpo": lambda p, t, f: look[(id(p), t, f)],
            "judge+verifier-dpo": lambda p, t, f: lab.combined_score(look[(id(p), t, f)], f and toy.number_right(p, t)),
            "oracle-dpo": lambda p, t, f: float(lab.spec_check(p.a, p.tag, t, f)[0]),
            "random-dpo": lambda p, t, f: float(rng.random()),
        }
        for arm in ARMS:
            run = ROOT / f"{arm}-s{s}"
            if not (run / "result.json").exists():
                t1 = time.perf_counter()
                pol, led = copy.deepcopy(base), Ledger().merge(cand_led)
                if arm.startswith("judge"):
                    led.merge(judge_led).merge(score_led)
                if arm == "judge-sft":
                    ex = []
                    for p, cs in zip(tp, tc):
                        pool = list(dict.fromkeys(list(cs) + [(J.REFUSAL, True)]))
                        sc = [look[(id(p), t, f)] for t, f in pool]
                        k = int(np.argmax(sc))
                        if sc[k] > 0.5:
                            ex.append(toy.example(p, *pool[k]))
                    train_sft(pol, ex, steps=STEPS, batch=BATCH, lr=LR, seed=s, ledger=led)
                    n_data = len(ex)
                else:
                    pairs = pairs_by(lab, tp, tc, scorers[arm])
                    D.train_dpo(pol, base, pairs, STEPS, beta=0.1, nll_coef=0.2, batch=BATCH, lr=LR, seed=s, ledger=led)
                    n_data = len(pairs)
                adh = J.adherence(pol)
                res = toy.eval_v2(pol)
                cmp = core.compare(res0, res, guards=toy.GUARDS)
                run.mkdir(parents=True, exist_ok=True)
                save_policy(pol, run / "policy.pt", {"arm": arm, "seed": s})
                (run / "result.json").write_text(json.dumps({"adherence": adh, "summary": res["summary"], "compare": cmp,
                                                             "ledger": led.to_dict(), "n_data": n_data,
                                                             "seconds": round(time.perf_counter() - t1, 1)}, default=str))
            r = json.loads((run / "result.json").read_text())
            out[f"{arm}-s{s}"] = r
            ad, c = r["adherence"], r["compare"]
            print(f"{arm:19s} s{s}  compliance {ad['compliance']:.3f}  refuse-restricted {ad['refusal_restricted']:.3f}  "
                  f"over-refuse 85-89 {ad['overrefusal_85_89']:.3f} 80-84 {ad['overrefusal_80_84']:.3f} <80 "
                  f"{ad['overrefusal_normal']:.3f}  | add_pass1 {c['add_pass1']['diff']:+.3f}  sft_nll "
                  f"{c['sft_nll']['diff']:+.3f}  sub {c['sub_greedy']['diff']:+.3f}  guard "
                  f"{'PASS' if c['_passes'] else 'FAIL'} | {Ledger.from_dict(r['ledger']).line()}")
    (ROOT / "results.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"\ntotal {time.perf_counter() - t0:.0f} s")


MAIN = [
    "# Main path (not run in this build; part of the Module 13 pilot). Policy: the project's SFT checkpoint of",
    "# Qwen3-1.7B-Base. Training judge: Qwen/Qwen3-8B b968826d (non-thinking) given SPEC_MAIN (frontierlab/pipeline/hf_stages.py).",
    "# Evaluation judge: Qwen/Qwen3-4B 1cfa9a72 asked only 'did it refuse?', plus a human audit of 50 responses.",
    "python -m frontierlab.pipeline.hf_stages spec --model runs/m13/main/sft-s0/policy --judge Qwen/Qwen3-8B "
    "--n 4000 --k 4 --out runs/m13/l133-main/judge-dpo-s0",
    "python -m frontierlab.pipeline.hf_stages spec-eval --model runs/m13/main/sft-s0/policy --judge Qwen/Qwen3-4B "
    "--out runs/m13/l133-main/sft-eval",
    "python -m frontierlab.pipeline.hf_stages spec-eval --model runs/m13/l133-main/judge-dpo-s0/policy --judge Qwen/Qwen3-4B "
    "--out runs/m13/l133-main/judge-dpo-s0/eval",
    "python -m frontierlab.pipeline.hf_eval score --model runs/m13/main/sft-s0/policy --chat --out runs/m13/l133-main/sft-evalv2.json",
    "python -m frontierlab.pipeline.hf_eval score --model runs/m13/l133-main/judge-dpo-s0/policy --chat "
    "--out runs/m13/l133-main/judge-dpo-s0/evalv2.json",
    "python -m frontierlab.pipeline.hf_eval compare runs/m13/l133-main/sft-evalv2.json runs/m13/l133-main/judge-dpo-s0/evalv2.json",
]


if __name__ == "__main__":
    main()
