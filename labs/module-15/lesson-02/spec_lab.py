"""Lab 15.2: speculative decoding with three drafts, measured acceptance and latency on the CPU.

    python labs/module-15/lesson-02/spec_lab.py                     # free CPU, about 17 minutes the first time
    python labs/module-15/lesson-02/spec_lab.py --variant main --print

1. Train (once, cached under runs/m15/models) the target — the toy preset with one DeepSeek-style MTP module —
   an independent 1-layer draft LM, and an EAGLE-style head on the frozen target (``frontierlab.ttc.spec_models``).
2. Correctness first: with your ``accept_reject``, greedy speculative decoding must reproduce plain greedy
   decoding token for token for every draft and gamma; the script stops if it does not.
3. Acceptance at temperature 0 and 1 for gamma 1, 2, 4: acceptance rate, acceptance by draft position, tokens
   per round, and alpha estimated as the mean overlap sum_x min(p, q).
4. Cost ratio c (one draft step / one target decode step) measured with ``frontierlab.perf.benchmark``; the
   predicted speed-up from Leviathan et al.'s formula; the measured speed-up of whole generations, interleaved
   with plain decoding (``frontierlab.perf.interleaved``), with a paired interval (``perf.speedup``).
"""

from __future__ import annotations

import argparse
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from frontierlab.labkit import load_path
from frontierlab.perf import interleaved, speedup
from frontierlab.ttc import spec_models as SM
from frontierlab.ttc import speculative as SP

HERE = Path(__file__).resolve().parent
GAMMAS = (1, 2, 4)
MAX_NEW = 64


@contextmanager
def use_lab(lab):
    saved = SP.accept_reject
    SP.accept_reject = lab.accept_reject
    try:
        yield
    finally:
        SP.accept_reject = saved


def drafts(target, draft_lm, head):
    emb, out = target.model.embed_tokens, target.lm_head
    return {"independent LM (1 layer)": lambda: SP.LMDraft(draft_lm),
            "EAGLE-style head": lambda: SP.HiddenDraft(head, emb, out),
            "MTP module (joint)": lambda: SP.HiddenDraft(target.mtp.layers[0], emb, out)}


def draft_step_fn(name, target, draft_lm, head):
    """One draft step at batch 1 (what c is made of)."""
    if name.startswith("independent"):
        c = draft_lm.new_cache()
        with torch.no_grad():
            draft_lm(torch.ones((1, 64), dtype=torch.long), cache=c)
        tok = torch.ones((1, 1), dtype=torch.long)

        def step():
            with torch.no_grad():
                draft_lm(tok, cache=c)
                SP.truncate_cache(c, 64)
        return step
    mod = head if name.startswith("EAGLE") else target.mtp.layers[0]
    from frontierlab.attention.base import LayerCache
    lc = LayerCache()
    h = torch.zeros((1, 64, target.config.hidden_size))
    with torch.no_grad():
        mod.block(mod.combine(h, target.model.embed_tokens(torch.ones((1, 64), dtype=torch.long))),
                  torch.arange(64), lc)
    h1 = torch.zeros((1, 1, target.config.hidden_size))
    e1 = target.model.embed_tokens(torch.ones((1, 1), dtype=torch.long)).detach()

    def step():
        with torch.no_grad():
            g = mod.block(mod.combine(h1, e1), torch.tensor([64]), lc)
            target.lm_head(mod.norm(g))
            SP._truncate_layer(lc, 64)
    return step


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["cpu", "main", "t4"], default="cpu")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--prompts", type=int, default=16)
    a = ap.parse_args(argv)
    if a.variant != "cpu":
        extra = "" if a.variant == "main" else "  # T4: fp16; expect lower speed-ups (the T4's bandwidth/FLOP ratio differs)"
        print("# not run in this build; part of the Module 15 pilot" + extra)
        print("python -m frontierlab.ttc.hf_spec own  --out runs/m15/l152-main --gammas 1,2,4,6 --prompts 64 --max-new 128")
        print("python -m frontierlab.ttc.hf_spec vllm --out runs/m15/l152-main --method draft_model --k 4")
        print("python -m frontierlab.ttc.hf_spec vllm --out runs/m15/l152-main --method eagle3 --k 3")
        print("python -m frontierlab.ttc.hf_spec vllm --out runs/m15/l152-main --method ngram --k 4")
        print("# MTP as a draft on GPU: train the Module 6 MTP model at pilot-30m and rerun this script with --device cuda")
        return
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    t0 = time.perf_counter()
    target = SM.ensure_target()
    draft_lm = SM.ensure_draft()
    head, hist = SM.ensure_eagle(target)
    from frontierlab.pipeline.compute import n_params
    print(f"target {n_params(target):,} parameters (MTP module {n_params(target.mtp):,}); draft LM {n_params(draft_lm):,}; "
          f"EAGLE head {n_params(head):,}; head training: final soft CE {hist[-1]['ce']:.3f}, top-1 agreement "
          f"{hist[-1]['top1']:.3f}")
    prompts = SM.val_prompts(a.prompts)
    D = drafts(target, draft_lm, head)

    with use_lab(lab):
        # 1. correctness
        for name, mk in D.items():
            for g in (1, 4):
                for p in prompts[:4]:
                    ref = SP.autoregressive_generate(target, p, 32, 0.0)
                    out, _ = SP.speculative_generate(target, mk(), p, 32, g, 0.0)
                    if not torch.equal(out, ref):
                        raise SystemExit(f"greedy speculative decoding with {name}, gamma {g} differs from plain greedy: "
                                         "fix accept_reject before measuring anything")
        print("correctness: greedy speculative == plain greedy for every draft and gamma (4 prompts x 32 tokens)")

        # 2. acceptance
        acc = {}
        print(f"\nAcceptance on {len(prompts)} validation prompts x {MAX_NEW} new tokens")
        print(f"  {'draft':26s} {'T':>3s} {'gamma':>5s} {'accept':>7s} {'alpha':>6s} {'tok/round':>9s}  by position")
        for T in (0.0, 1.0):
            for name, mk in D.items():
                for g in GAMMAS:
                    tot = {"accepted": 0, "proposed": 0, "rounds": 0, "pos_a": np.zeros(g), "pos_p": np.zeros(g), "beta": []}
                    for i, p in enumerate(prompts):
                        _, st = SP.speculative_generate(target, mk(), p, MAX_NEW, g, T, torch.Generator().manual_seed(i))
                        tot["accepted"] += st["accepted"]; tot["proposed"] += st["proposed"]; tot["rounds"] += st["rounds"]
                        tot["pos_a"] += st["accepted_per_position"]; tot["pos_p"] += st["proposed_per_position"]
                        tot["beta"].append(st["mean_overlap"])
                    rate = tot["accepted"] / tot["proposed"]
                    alpha = float(np.mean(tot["beta"]))
                    tpr = len(prompts) * MAX_NEW / tot["rounds"]
                    acc[(name, T, g)] = {"rate": rate, "alpha": alpha, "tpr": tpr}
                    pos = " ".join(f"{x:.2f}" for x in tot["pos_a"] / np.maximum(1, tot["pos_p"]))
                    print(f"  {name:26s} {T:3.0f} {g:5d} {rate:7.3f} {alpha:6.3f} {tpr:9.2f}  [{pos}]")

        # 3. cost ratio and latency
        print("\nCost ratio and speed-up at temperature 0, batch 1 (CPU; median of interleaved rounds, 95% CI)")
        print(f"  {'draft':26s} {'c':>5s} {'gamma':>5s} {'predicted':>9s} {'measured':>9s}  CI")
        tgt_step = None
        for name, mk in D.items():
            cr = SP.step_cost_ratio(target, draft_step_fn(name, target, draft_lm, head))
            tgt_step = cr["target_step_s"]
            for g in GAMMAS:
                ps = prompts[:4]
                fns = {"plain": lambda: [SP.autoregressive_generate(target, p, MAX_NEW, 0.0) for p in ps],
                       "spec": lambda g=g: [SP.speculative_generate(target, mk(), p, MAX_NEW, g, 0.0, track_overlap=False)
                                            for p in ps]}
                tm = interleaved(fns, warmup=1, rounds=5)
                sp = speedup(tm["plain"], tm["spec"])
                al = acc[(name, 0.0, g)]["alpha"]
                pred = lab.walltime_improvement(al, g, cr["c"])
                print(f"  {name:26s} {cr['c']:5.2f} {g:5d} {pred:9.2f} {sp['speedup']:9.2f}  "
                      f"[{sp['ci'][0]:.2f}, {sp['ci'][1]:.2f}]")
        print(f"  (target decode step at batch 1: {tgt_step * 1e3:.2f} ms)")
    print(f"\ntotal {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
