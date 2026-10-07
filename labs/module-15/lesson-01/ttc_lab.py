"""Lab 15.1: a budget-matched test-time-compute experiment on the free-CPU world.

    python labs/module-15/lesson-01/ttc_lab.py                      # free CPU, about 10 minutes the first time
    python labs/module-15/lesson-01/ttc_lab.py --variant main --print
    python labs/module-15/lesson-01/ttc_lab.py --variant t4 --print

Steps (each result is cached under runs/m15/l151, so a rerun only redoes what is missing):

1. Train the world's policy (400 SFT steps, about 1–2 minutes), an outcome verifier and a process verifier
   (400 steps each on 16,000 policy samples of training problems, about 1.5–2.5 minutes each).
2. On 300 held-out problems: greedy single chains in each reasoning mode (none, short, long) and with budget
   forcing; pools of 32 short-mode and 16 long-mode samples at temperature 0.7, each scored by the ORM; PRM beam
   search at three settings.
3. Measure the policy's decode-step time and the verifier's forward time at batch sizes 1–256
   (``frontierlab.perf``), build the latency model, and check it against end-to-end wall-clock of three strategies.
4. Print the verifier's quality, the selection-gap table (oracle pass@N vs what each procedure selects; your
   ``lab.py``), the full table, the best strategy of each family at five budgets, and the recommendation under
   the contract's latency target.
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
from frontierlab.pipeline.compute import n_params
from frontierlab.posttrain.sft import load_policy
from frontierlab.ttc import report as R
from frontierlab.ttc import select as SE
from frontierlab.ttc import world as W
from frontierlab.ttc.budget import LatencyModel

HERE = Path(__file__).resolve().parent
ROOT = Path("runs/m15/l151")
N_EVAL, TEMP = 300, 0.7
BUDGETS = (20, 40, 80, 160, 320, 640)


def setup(root: Path = ROOT):
    pol = load_policy(W.ensure_policy(root / "policy.pt", steps=400))
    orm, oi = W.ensure_verifier(pol, "orm", root / "orm.pt", n_problems=4000, k=4, steps=400)
    prm, pi = W.ensure_verifier(pol, "prm", root / "prm.pt", n_problems=4000, k=4, steps=400)
    return pol, orm, prm, oi, pi


def pools(pol, orm, prm, probs, root: Path = ROOT, seed: int = 0):
    path = root / f"pools-s{seed}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    out = {"single": {}, "pool": {}}
    for mode in "nmh":
        ps = [p.with_mode(mode) for p in probs]
        out["single"][mode] = [r[0] for r in W.sample(pol, ps, 1, None, 0.0)]
    hp = [p.with_mode("h") for p in probs]
    for B in (9, 18, 27):
        out["single"][f"h@{B}"] = [r[0] for r in W.sample(pol, hp, 1, B, 0.0)]
    g = torch.Generator().manual_seed(seed)
    for mode, n in (("m", 32), ("h", 16)):
        ps = [p.with_mode(mode) for p in probs]
        rows = W.sample(pol, ps, n, None, TEMP, g)
        flat_p = [p for p in ps for _ in range(n)]
        flat = [c for r in rows for c in r]
        sc = W.score(orm, flat_p, [c["text"] for c in flat], [c["finished"] for c in flat])
        for c, s, p in zip(flat, sc, flat_p):
            c["score"] = float(s)
            c["scored_tokens"] = W.scored_tokens(p, c["text"], c["finished"])
        out["pool"][mode] = rows
    out["search"] = {}
    for w, e in ((2, 2), (2, 4), (4, 4)):
        t0 = time.perf_counter()
        res = W.beam_search(pol, prm, orm, probs, w, e, TEMP, g)
        out["search"][(w, e)] = {"res": res, "seconds": time.perf_counter() - t0}
    torch.save(out, path)
    return out


def latency_model(pol, orm, root: Path = ROOT) -> LatencyModel:
    path = root / "step_times.json"
    if not path.exists():
        t = W.measure_step_times(pol, orm)
        path.write_text(json.dumps(t))
    t = json.loads(path.read_text())
    lm = LatencyModel({int(k): v for k, v in t["decode_step"].items()}, t["prefill_per_token"],
                      {int(k): v for k, v in t["verifier"].items()})
    return lm


@torch.no_grad()
def wallclock_checks(pol, orm, probs, lm: LatencyModel) -> list[tuple[str, float, float]]:
    """End-to-end seconds per problem (one problem at a time, 20 problems, after a warm-up) vs the latency model."""
    out = []
    g = torch.Generator().manual_seed(123)
    for mode, n in (("m", 1), ("m", 16), ("h", 1)):               # warm-up: allocator, caches, first-call costs
        W.sample(pol, [probs[0].with_mode(mode)] * 2, n, None, TEMP, g)
    for label, mode, n, verify in (("single chain, short mode", "m", 1, False), ("16 samples + ORM", "m", 16, True),
                                   ("single chain, long mode", "h", 1, False)):
        ts, lens = [], []
        for p in probs[:20]:
            q = p.with_mode(mode)
            t0 = time.perf_counter()
            rows = W.sample(pol, [q], n, None, TEMP if n > 1 else 0.0, g)[0]
            if verify:
                W.score(orm, [q] * n, [r["text"] for r in rows], [r["finished"] for r in rows])
            ts.append(time.perf_counter() - t0)
            lens.append(max(r["gen_tokens"] for r in rows))          # a batch runs until its longest row ends
        chain = float(np.mean(lens))
        out.append((label, float(np.median(ts)), lm.parallel(W.prompt_tokens(probs[0]), chain, n, verify)))
    return out


def run(lab, seed: int = 0, latency_x: float = 2.0, root: Path = ROOT, quiet: bool = False) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    pol, orm, prm, oi, pi = setup(root)
    probs = W.make_problems(N_EVAL, seed=1, held=True)
    gold = [p.answer for p in probs]
    data = pools(pol, orm, prm, probs, root, seed)
    lm = latency_model(pol, orm, root)
    Np, Nv = n_params(pol), n_params(orm)
    P = float(W.prompt_tokens(probs[0]))

    # verifier quality on the held-out pool
    pm = data["pool"]["m"]
    sc = np.array([[c["score"] for c in r] for r in pm])
    cor = np.array([[c["correct"] for c in r] for r in pm])
    fp = float(((sc > 0.5) & ~cor).sum() / max(1, (~cor).sum()))
    fn = float(((sc <= 0.5) & cor).sum() / max(1, cor.sum()))
    print(f"policy {Np:,} parameters; ORM and PRM {Nv:,} each (label samples: {oi['label_samples']:,} each)")
    print(f"ORM on held-out short-mode samples: AUC {W.auc(sc.ravel(), cor.ravel()):.3f}, "
          f"false-positive rate {fp:.3f}, false-negative rate {fn:.3f} at 0.5")

    # selection gap with the learner's functions
    answers = [[c["answer"] for c in r] for r in pm]
    print("\nSelection gap (short mode, T = 0.7): oracle pass@N is coverage; the rest are what a procedure picks")
    print(f"  {'N':>3s} {'oracle':>7s} {'majority':>9s} {'best-of-N':>10s} {'weighted':>9s}")
    for N in (1, 2, 4, 8, 16, 32):
        orc = SE.oracle_pass_at_k(cor, N).mean()
        vals = [lab.subset_success(answers, gold, N, m, sc.tolist(), 100, seed).mean()
                for m in ("majority", "best_of_n", "weighted")]
        print(f"  {N:3d} {orc:7.3f} {vals[0]:9.3f} {vals[1]:10.3f} {vals[2]:9.3f}")

    # the full table
    rows = []
    for mode, d in data["single"].items():
        gen = float(np.mean([r["gen_tokens"] for r in d]))
        rows.append(R.single_row(f"greedy, mode {mode}", [r["correct"] for r in d], gen, policy_params=Np,
                                 prompt_tokens=P, latency=lm))
    for mode, Ns in (("m", (1, 2, 4, 8, 16, 32)), ("h", (1, 2, 4, 8, 16))):
        rows += R.pool_rows(f"mode {mode}", data["pool"][mode], gold, Ns, policy_params=Np, verifier_params=Nv,
                            prompt_tokens=P, latency=lm, resamples=100, seed=seed)
    rule_pool = [[{**c, "score": float(W.rule_verify(p, c["text"], c["finished"])), "scored_tokens": 0.0} for c in r]
                 for p, r in zip(probs, pm)]
    rows += R.pool_rows("mode m", rule_pool, gold, (2, 4, 8, 16, 32), policy_params=Np, verifier_params=0,
                        prompt_tokens=P, latency=lm, methods=("best_of_n",), resamples=100, seed=seed, oracle=False,
                        verifier="rule checker")
    for r in rows[-5:]:
        r["latency_s"] = lm.parallel(int(P), float(np.mean([c["gen_tokens"] for x in pm for c in x])), r["N"])
    for (w, e), s in data["search"].items():
        res = s["res"]
        D, step = probs[0].digits, probs[0].step_len
        lat = lm.search(int(P), D, step, w, e, D + 2)
        rows.append(R.search_row(f"width {w}, expand {e}", res, policy_params=Np, verifier_params=Nv,
                                 prompt_tokens=P, latency_s=lat))
    rows.sort(key=lambda r: r["pte"])
    if not quiet:
        print("\nEvery strategy (pte = policy-token equivalents per problem; latency from the measured model)")
        print(R.format_rows(rows))

    single_m = next(r for r in rows if r["label"] == "single chain (greedy, mode m)")
    target = latency_x * single_m["latency_s"]
    print(f"\nLatency target: {latency_x:g} x one greedy short chain = {target * 1e3:.1f} ms per problem")
    recs = {}
    for B in BUDGETS:
        best = R.best_per_family(rows, B)
        line = ", ".join(f"{r['family']} {r['success']:.3f} (N={r['N']})" for r in best[:4])
        rec = R.recommend(rows, B, target, seed)
        recs[B] = rec
        ch = rec["choice"]
        print(f"  budget {B:4d} pte: best per family [{line}]")
        desc = f"{ch['label']} (success {ch['success']:.3f}, {ch['pte']:.0f} pte)" if ch else "none"
        print(f"      recommended within the latency target: {desc}; {rec['why']}")

    checks = wallclock_checks(pol, orm, probs, lm)
    print("\nLatency model vs measured end-to-end (median of 20 problems, one at a time, after a warm-up)")
    for label, meas, pred in checks:
        print(f"  {label:28s} measured {meas * 1e3:7.2f} ms   model {pred * 1e3:7.2f} ms")
    print(f"\ntotal {time.perf_counter() - t0:.0f} s")
    return {"rows": rows, "recommendations": recs, "latency_target": target, "checks": checks}


MAIN = [
    "# Main path (1x H100; not run in this build; part of the Module 15 pilot). Policy Qwen/Qwen3-1.7B 70d244cc,",
    "# GSM8K test (first 500), ORM Skywork-Reward-V2-Qwen3-1.7B e51ea3e, PRM Qwen2.5-Math-PRM-7B 0610740.",
    "python -m frontierlab.ttc.hf_ttc sample --out runs/m15/l151-main --n-questions 500 --n 32 --sample-budget 512",
    "python -m frontierlab.ttc.hf_ttc single --out runs/m15/l151-main --n-questions 500 --budgets 256,512,1024,2048,4096",
    "python -m frontierlab.ttc.hf_ttc score  --out runs/m15/l151-main",
    "python -m frontierlab.ttc.hf_ttc search --out runs/m15/l151-main --n-questions 200 --width 4 --expand 4",
    "python -m frontierlab.ttc.hf_ttc latency --out runs/m15/l151-main --ns 1,4,16,32 --budgets 512,2048,4096",
    "python -m frontierlab.ttc.hf_ttc report --out runs/m15/l151-main --budget 8192 --latency 20",
]
T4 = [
    "# Free GPU (T4, fp16 vLLM or Transformers): Qwen/Qwen3-0.6B (thinking model) at fewer questions and samples.",
    "python -m frontierlab.ttc.hf_ttc sample --out runs/m15/l151-t4 --model Qwen/Qwen3-0.6B --revision <pin at the pilot> "
    "--n-questions 100 --n 8 --sample-budget 256 --max-tokens 512",
    "python -m frontierlab.ttc.hf_ttc single --out runs/m15/l151-t4 --model Qwen/Qwen3-0.6B --revision <pin> "
    "--n-questions 100 --budgets 256,512,1024",
    "python -m frontierlab.ttc.hf_ttc score --out runs/m15/l151-t4",
    "python -m frontierlab.ttc.hf_ttc report --out runs/m15/l151-t4 --policy-params 751632384 --budget 2048",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["cpu", "main", "t4"], default="cpu")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latency-x", type=float, default=2.0)
    a = ap.parse_args(argv)
    if a.variant != "cpu":
        print("\n".join(MAIN if a.variant == "main" else T4))
        return
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    run(lab, a.seed, a.latency_x)


if __name__ == "__main__":
    main()
