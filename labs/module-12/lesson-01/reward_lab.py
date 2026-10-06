"""Lab 12.1: audit three verifiers, then over-optimise learned reward models against a hidden gold reward.

    python labs/module-12/lesson-01/reward_lab.py                 # free CPU, about 10 minutes
    python labs/module-12/lesson-01/reward_lab.py --rl-lenient    # + two short RL runs (about 3 more minutes)

Part A, verifiable rewards. The SFT policy (trained first if ``runs/m12/sft`` is missing) answers 100
held-out addition problems 64 times each. Three verifiers score every response: ``strict`` (the
specification: finished, exactly the answer), ``last_number`` and ``lenient``. A probe set of hand-made
responses (answer repeated, answer inside a longer number, unfinished, padded...) shows what each one
accepts. With ``--rl-lenient`` the policy is trained for 200 steps against ``lenient`` and against
``strict`` (temperature 1.5) to see whether RL finds the lenient verifier's loophole.

Part B, learned rewards. ``LetterWorld`` has a hidden gold reward; a noisy annotator labels pairs drawn
from the base sampler; reward models are trained on 250, 1,000 and 4,000 pairs (2 seeds each). For a pool
of 20,000 base samples, the unbiased best-of-n estimator gives the proxy and gold reward of BoN for
n = 1..4096 (KL = log n - (n-1)/n), next to the gold-oracle BoN (selecting by the gold reward itself).
Results go to ``runs/m12/l121/results.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from frontierlab.labkit import load_path
from frontierlab.posttrain import reward as RW
from frontierlab.posttrain.policy import sample
from frontierlab.posttrain.sft import ensure_sft, load_policy
from frontierlab.posttrain.tasks import (LetterWorld, encode_prompts, problems_from, response_text, split_problems,
                                         VERIFIERS)

HERE = Path(__file__).resolve().parent
OUT = Path("runs/m12/l121")
NS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def probe_set(problems):
    """(text, finished, truly_correct) probes per problem."""
    rows = []
    for p in problems:
        r = str(p.result)
        wrong = str(p.result + 1)
        for text, fin, ok in [(r, True, True), (r, False, False), (r + r, True, False), ("1" + r + "7", True, False),
                              (wrong + r, True, False), ("0" + r, True, False), (wrong, True, False),
                              (r + "!", True, False), ("#" + r + "#", True, False), ("9" * 7, False, False)]:
            rows.append((p, text, fin, ok))
    return rows


def part_a(lab, sft_path, rl_lenient: bool) -> dict:
    _, held = split_problems(2)
    probs = problems_from(held, ".", 2, "+")[:100]
    res = {"probes": {}, "samples": {}}
    probes = probe_set(probs[:20])
    truth = [ok for *_, ok in probes]
    for name, fn in VERIFIERS.items():
        acc = [fn(t, f, p) for p, t, f, _ in probes]
        res["probes"][name] = lab.verifier_errors(acc, truth)
    pol = load_policy(sft_path)
    for T in (1.0, 2.0):
        rep = [p for p in probs for _ in range(64)]
        ro = sample(pol, encode_prompts(rep), 8, T, torch.Generator().manual_seed(0))
        texts = [response_text(r) for r in ro.response]
        strict = [VERIFIERS["strict"](t, f, p) for (t, f), p in zip(texts, rep)]
        row = {"strict_pass": float(np.mean(strict)), "mean_len": float(ro.lengths.mean()),
               "truncated": float((~ro.finished).float().mean())}
        for name in ("last_number", "lenient"):
            acc = [VERIFIERS[name](t, f, p) for (t, f), p in zip(texts, rep)]
            row[name] = {"pass": float(np.mean(acc)), **lab.verifier_errors(acc, strict)}
        res["samples"][f"T={T}"] = row
    if rl_lenient:
        from frontierlab.metrics.jsonl import read_jsonl
        from frontierlab.posttrain.rl import RLConfig, evaluate, train
        res["rl"] = {}
        for v in ("lenient", "strict"):
            run = OUT / f"rl-{v}"
            train(RLConfig(init=str(sft_path), run=str(run), steps=200, verifier=v, temperature=1.5, lr=1e-3,
                           eval_every=50))
            final = load_policy(run / "policy.pt")
            ev = {name: evaluate(final, probs, 8, 1.0, torch.Generator().manual_seed(1), name)["sampled_acc"]
                  for name in ("strict", "lenient")}
            rows = [r for r in read_jsonl(run / "metrics.jsonl") if r["split"] == "train"]
            res["rl"][v] = {"final_sampled": ev, "train_len_last20": float(np.mean([r["len"] for r in rows[-20:]]))}
    return res


def part_b(lab, pairs_list, seeds, pool_n=20000, steps=400) -> dict:
    world = LetterWorld()
    pool = world.sample(pool_n, np.random.default_rng(5))
    gold = np.array([world.gold(s) for s in pool])
    L = np.array([len(s) for s in pool], dtype=np.float64)
    kl = np.array([lab.bon_kl(n) for n in NS])
    d = np.sqrt(kl)
    oracle = [lab.bon_expected(gold, gold, n) - gold.mean() for n in NS]
    out = {"n": NS, "kl": kl.tolist(), "oracle_gold": oracle, "rms": []}
    for n_pairs in pairs_list:
        for seed in seeds:
            t = time.perf_counter()
            rm, _ = RW.train_rm(world, n_pairs, steps=steps, seed=seed)
            pm = RW.preference_metrics(rm, world, 2000)
            px = RW.score_strings(rm, pool, world.max_len)
            g = [lab.bon_expected(px, gold, n) - gold.mean() for n in NS]
            pr = [lab.bon_expected(px, px, n) - px.mean() for n in NS]
            ln = [lab.bon_expected(px, L, n) for n in NS]
            fit = RW.fit_gao(d[1:], np.array(g[1:]), "bon")
            row = {"n_pairs": n_pairs, "seed": seed, "seconds": round(time.perf_counter() - t, 1),
                   "acc_vs_gold": pm["acc_vs_gold"], "acc_vs_labels": pm["acc_vs_labels"], "ece": pm["ece"],
                   "gold": g, "proxy": pr, "length": ln, "peak": lab.peak(NS, g, tol=0.05), "fit": fit}
            out["rms"].append(row)
            print(f"pairs {n_pairs:5d} seed {seed}  acc(gold) {pm['acc_vs_gold']:.3f}  ECE {pm['ece']:.3f}  "
                  f"gold@n: " + " ".join(f"{x:.2f}" for x in g[::2]) + f"  peak n={row['peak']['n_peak']}"
                  f"  ({row['seconds']:.0f}s)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rl-lenient", action="store_true")
    ap.add_argument("--pairs", default="250,1000,4000")
    ap.add_argument("--seeds", default="0,1")
    a = ap.parse_args(argv)
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    sft = ensure_sft("runs/m12/sft")
    res = {"part_a": part_a(lab, sft, a.rl_lenient)}
    pa = res["part_a"]
    print("\nPart A: verifier errors on the probe set (truth = the strict specification)")
    for k, v in pa["probes"].items():
        print(f"  {k:12s} false positives {v['false_positive_rate']:.2f}  false negatives {v['false_negative_rate']:.2f}")
    for k, v in pa["samples"].items():
        print(f"  policy samples {k}: strict pass {v['strict_pass']:.3f}, mean length {v['mean_len']:.2f}; "
              f"last_number FP {v['last_number']['false_positive_rate']:.4f}, "
              f"lenient FP {v['lenient']['false_positive_rate']:.4f}")
    if "rl" in pa:
        for v, r in pa["rl"].items():
            print(f"  RL against {v:8s}: final sampled strict {r['final_sampled']['strict']:.3f}, "
                  f"lenient {r['final_sampled']['lenient']:.3f}, train length {r['train_len_last20']:.2f}")
    print("\nPart B: best-of-n against learned reward models (gold relative to the base mean)")
    res["part_b"] = part_b(lab, [int(x) for x in a.pairs.split(",")], [int(x) for x in a.seeds.split(",")])
    pb = res["part_b"]
    print("  n      KL     oracle  " + "  ".join(f"{r['n_pairs']}/s{r['seed']}" for r in pb["rms"]))
    for i, n in enumerate(NS):
        print(f"  {n:5d} {pb['kl'][i]:6.2f} {pb['oracle_gold'][i]:7.2f}  "
              + "  ".join(f"{r['gold'][i]:7.2f}" for r in pb["rms"]))
    res["seconds"] = round(time.perf_counter() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT / 'results.json'} ({res['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
