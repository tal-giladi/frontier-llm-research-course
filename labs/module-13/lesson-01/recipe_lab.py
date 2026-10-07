"""Lab 13.1: the preference stage of an open recipe, measured with Eval v2, and an audit of published stage tables.

    python labs/module-13/lesson-01/recipe_lab.py                    # free CPU, about 6 minutes
    python labs/module-13/lesson-01/recipe_lab.py --part audit       # Part B only (seconds)
    python labs/module-13/lesson-01/recipe_lab.py --variant main --print

Part A. From Module 12's SFT checkpoint, 8,000 training prompts (all operations and tags) get 4 sampled
responses each (Tülu 3 samples four responses per prompt); every response is rated on two programmatic
aspects (number right, format right; mean 0 / 0.5 / 1) and the ratings are binarised with your
``binarise``. Four DPO arms, 2 seeds each, 150 steps of 64 pairs, lr 5e-5:

* ``dpo``        plain DPO, beta 0.1                              (Rafailov et al.)
* ``dpo-norm``   length-normalised DPO, beta 0.5                  (Tülu 3's form)
* ``dpo-nll``    DPO + 0.2 x NLL on the chosen response, beta 0.1  (Llama 3's form)
* ``random-nll`` control: the dpo-nll arm with chosen/rejected swapped at random (same pairs, no signal)

Each final policy is scored by Eval Suite v2 and compared with the SFT start (paired by item, guards 0.02).

Part B. Published per-stage tables (Tülu 3 Table 6 and 14, OLMo 3 Table 22, Qwen3 Table 21) audited with
your ``audit``: which reported stage gains are larger than the item noise of the benchmark they are on.
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
from frontierlab.pipeline import toy
from frontierlab.pipeline.compute import Ledger
from frontierlab.pipeline.seqs import sequence_logps
from frontierlab.posttrain.sft import ensure_sft, load_policy, save_policy

HERE = Path(__file__).resolve().parent
ROOT = Path("runs/m13/l131")
ARMS = {"dpo": dict(beta=0.1), "dpo-norm": dict(beta=0.5, normalise=True), "dpo-nll": dict(beta=0.1, nll_coef=0.2),
        "random-nll": dict(beta=0.1, nll_coef=0.2, shuffle=True)}
STEPS, BATCH, LR, N_PROMPTS, N_SAMPLES = 150, 64, 5e-5, 8000, 4

# Published per-stage numbers (percent). Item counts: GSM8K test 1,319; MATH-500 500; IFEval 541; one AIME year 30.
PUBLISHED = {
    "Tülu 3 8B, average over the development suite (Table 6)": ([("SFT", 60.6), ("DPO", 64.7), ("RLVR", 65.1)], None),
    "Tülu 3 8B, GSM8K 8-shot CoT (Table 6)": ([("SFT", 76.2), ("DPO", 84.3), ("RLVR", 87.6)], 1319),
    "Tülu 3 8B, IFEval prompt-loose (Table 6)": ([("SFT", 72.8), ("DPO", 81.1), ("RLVR", 82.4)], 541),
    "Tülu 3 8B, MMLU 0-shot CoT (Table 6)": ([("SFT", 65.9), ("DPO", 68.7), ("RLVR", 68.2)], 14042),
    "Olmo 3 Think 7B, average of a subset, one run (Table 22)": ([("SFT", 70.1), ("SFT+DPO", 72.7), ("SFT+DPO+RLVR", 74.1)], None),
    "Olmo 3 Think 7B, IFEval (Table 22)": ([("SFT", 77.9), ("SFT+DPO", 75.9), ("SFT+DPO+RLVR", 82.3)], 541),
    "Olmo 3 Think 7B, AIME 2025 (Table 22)": ([("SFT", 57.6), ("SFT+DPO", 62.7), ("SFT+DPO+RLVR", 64.2)], 30),
    "Qwen3-8B, AIME'24 from the off-policy-distilled checkpoint: RL (Table 21)": ([("off-policy distill", 55.0), ("+ RL", 67.6)], 30),
    "Qwen3-8B, AIME'24 from the same checkpoint: on-policy distillation (Table 21)": ([("off-policy distill", 55.0), ("+ on-policy distill", 74.4)], 30),
}
TULU_SFT_SEED_SPREAD = 60.1 - 59.8          # Tülu 3 Table 14: five 8B SFT seeds averaged 59.8-60.1


def pairs_with(lab, problems, cands, seed):
    rng = np.random.default_rng(seed)
    pairs, n_skip = [], 0
    for p, cs in zip(problems, cands):
        uniq = list(dict.fromkeys(cs))
        scores = [toy.aspect_score(p, t, f) for t, f in uniq]
        pick = lab.binarise(scores, rng)
        if pick is None:
            n_skip += 1
            continue
        (ct, cf), (rt, rf) = uniq[pick[0]], uniq[pick[1]]
        pairs.append((toy.example(p, ct, cf), toy.example(p, rt, rf)))
    return pairs, n_skip


def check_loss(lab, policy, ref, pairs):
    """Your dpo_loss must agree with the course's on a real batch before any result counts."""
    chunk = pairs[:32]
    ids, mask = D.pair_batch(chunk)
    with torch.no_grad():
        lp, n = sequence_logps(policy, ids, mask)
        rp, _ = sequence_logps(ref, ids, mask)
    B = len(chunk)
    for kw in (dict(), dict(normalise=True), dict(nll_coef=0.2)):
        ref_loss, _ = D.dpo_loss(lp[:B], lp[B:], rp[:B], rp[B:], 0.3, n[:B], n[B:], **kw)
        mine = lab.dpo_loss(lp[:B], lp[B:], rp[:B], rp[B:], 0.3, n[:B], n[B:], **kw)
        if abs(float(mine) - float(ref_loss)) > 1e-5:
            raise SystemExit(f"your dpo_loss disagrees with the reference ({kw}): {float(mine)} vs {float(ref_loss)}")


def part_a(lab) -> dict:
    sft = ensure_sft("runs/m12/sft")
    base = load_policy(sft)
    ROOT.mkdir(parents=True, exist_ok=True)
    base_eval = ROOT / "sft-eval.json"
    if not base_eval.exists():
        base_eval.write_text(json.dumps(toy.eval_v2(base)))
    res0 = json.loads(base_eval.read_text())
    out = {"sft": res0["summary"]}
    for s in (0, 1):
        led_data = Ledger()
        probs = toy.train_problems(N_PROMPTS, seed=11 + s)
        cands = toy.sample_texts(base, probs, N_SAMPLES, 1.0, seed=3 + s, ledger=led_data)
        pairs, n_skip = pairs_with(lab, probs, cands, seed=s)
        check_loss(lab, base, base, pairs)
        print(f"seed {s}: {len(pairs)} pairs from {len(probs)} prompts ({n_skip} prompts had equal ratings)")
        for arm, kw in ARMS.items():
            run = ROOT / f"{arm}-s{s}"
            if not (run / "result.json").exists():
                kw = dict(kw)
                P = toy.shuffle_labels(pairs, seed=100 + s) if kw.pop("shuffle", False) else pairs
                pol, led = copy.deepcopy(base), Ledger().merge(led_data)
                t0 = time.perf_counter()
                tr = D.train_dpo(pol, base, P, STEPS, batch=BATCH, lr=LR, seed=s, ledger=led, **kw)
                res = toy.eval_v2(pol)
                cmp = core.compare(res0, res, guards=toy.GUARDS)
                run.mkdir(parents=True, exist_ok=True)
                save_policy(pol, run / "policy.pt", {"arm": arm, "seed": s})
                (run / "result.json").write_text(json.dumps({"train": tr["final"], "summary": res["summary"],
                                                             "compare": cmp, "ledger": led.to_dict(),
                                                             "seconds": round(time.perf_counter() - t0, 1),
                                                             "pairs": len(pairs)}, default=str))
            r = json.loads((run / "result.json").read_text())
            out[f"{arm}-s{s}"] = r
            f = r["train"]
            print(f"\n== {arm} seed {s}: reward_acc {f['reward_acc']:.2f}  chosen logp change {f['chosen_logp_change']:+.2f}  "
                  f"rejected {f['rejected_logp_change']:+.2f}  ({r['seconds']} s; {Ledger.from_dict(r['ledger']).line()})")
            print(core.report(r["compare"]))
    (ROOT / "results.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def part_b(lab):
    print("\nPart B: are the published stage gains larger than the benchmark's item noise? (95%, unpaired)")
    rows = []
    for name, (stages, n) in PUBLISHED.items():
        spread = TULU_SFT_SEED_SPREAD if n is None and name.startswith("Tülu") else None
        for r in lab.audit(stages, n, spread):
            mde = f"{r['mde']:.1f}" if r["mde"] is not None else "  - "
            print(f"  {name[:62]:62s} {r['from']:>18s} -> {r['to']:<20s} {r['delta']:+5.1f}  MDE {mde:>5s}  {r['verdict']}")
            rows.append({"table": name, **r})
    (ROOT / "audit.json").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "audit.json").write_text(json.dumps(rows, indent=1))


MAIN = [
    "# Main path (not run in this build; part of the Module 13 pilot): what each released OLMo 2 1B stage bought.",
    "# Plain format for all four (pins equal, so base -> SFT is comparable), then chat format for SFT, DPO, Instruct.",
    *[f"python -m frontierlab.pipeline.hf_eval score --model allenai/OLMo-2-0425-1B{suf} --revision {rev} "
      f"--out runs/m13/l131-main/{name}-plain.json" for name, suf, rev in (
          ("base", "", "a1847dff35000b4271fa70afc5db10fd29fedbdf"),
          ("sft", "-SFT", "0d85a3d037876ce6ac7d4311d994400fc66ac27f"),
          ("dpo", "-DPO", "c4b0485961ab24c2433b090f3b922f0913a9290f"),
          ("instruct", "-Instruct", "48d788eca847d4d7548f375ad03d3c9312f6139e"))],
    *[f"python -m frontierlab.pipeline.hf_eval score --model allenai/OLMo-2-0425-1B{suf} --revision {rev} --chat "
      f"--out runs/m13/l131-main/{name}-chat.json" for name, suf, rev in (
          ("sft", "-SFT", "0d85a3d037876ce6ac7d4311d994400fc66ac27f"),
          ("dpo", "-DPO", "c4b0485961ab24c2433b090f3b922f0913a9290f"),
          ("instruct", "-Instruct", "48d788eca847d4d7548f375ad03d3c9312f6139e"))],
    "python -m frontierlab.pipeline.hf_eval compare runs/m13/l131-main/base-plain.json runs/m13/l131-main/sft-plain.json",
    "python -m frontierlab.pipeline.hf_eval compare runs/m13/l131-main/sft-chat.json runs/m13/l131-main/dpo-chat.json",
    "python -m frontierlab.pipeline.hf_eval compare runs/m13/l131-main/dpo-chat.json runs/m13/l131-main/instruct-chat.json",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["cpu", "main"], default="cpu")
    ap.add_argument("--part", choices=["all", "dpo", "audit"], default="all")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    if a.variant == "main":
        print("\n".join(MAIN))
        return
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    t0 = time.perf_counter()
    if a.part in ("all", "dpo"):
        part_a(lab)
    if a.part in ("all", "audit"):
        part_b(lab)
    print(f"\ntotal {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
