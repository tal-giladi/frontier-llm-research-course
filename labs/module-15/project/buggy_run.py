"""The colleague's harness on lesson 15.1's cached pools, next to the course harness (the debugging task).

    python labs/module-15/project/buggy_run.py          # seconds, after ttc_lab.py has run once
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from frontierlab.labkit import load_path
from frontierlab.pipeline.compute import n_params
from frontierlab.posttrain.sft import load_policy
from frontierlab.ttc import world as W

HERE = Path(__file__).resolve().parent
ROOT = Path("runs/m15/l151")


def table(H, pool, gold, lm, Np, P, label):
    rng = np.random.default_rng(0)
    print(f"\n{label}")
    print(f"  {'N':>3s} {'majority':>9s} {'best-of-N':>10s} {'BoN pte':>8s} {'parallel latency':>17s}")
    gen = float(np.mean([c["gen_tokens"] for r in pool for c in r]))
    scored = float(np.mean([c["scored_tokens"] for r in pool for c in r]))
    for N in (1, 4, 16):
        maj = bon = 0.0
        for cands, g in zip(pool, gold):
            for _ in range(50):
                idx = rng.choice(len(cands), N, replace=False)
                a = [cands[j]["answer"] for j in idx]
                s = [round(cands[j]["score"], 1) for j in idx]
                c = [cands[j]["correct"] for j in idx]
                maj += H.majority(a, c) == g
                bon += H.best_of_n(a, s, c) == g
        k = len(pool) * 50
        pte = H.spend_pte(P, N * gen, N * scored, Np, Np)
        print(f"  {N:3d} {maj / k:9.3f} {bon / k:10.3f} {pte:8.0f} {H.latency_parallel(lm, int(P), gen, N, True) * 1e3:14.1f} ms")


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    path = ROOT / "pools-s0.pt"
    if not path.exists():
        raise SystemExit("run labs/module-15/lesson-01/ttc_lab.py first (it caches the sample pools)")
    data = torch.load(path, weights_only=False)
    ttc_lab = load_path(str(HERE.parent / "lesson-01" / "ttc_lab.py"))
    pol = load_policy(ROOT / "policy.pt")
    orm, _ = W.ensure_verifier(pol, "orm", ROOT / "orm.pt")
    lm = ttc_lab.latency_model(pol, orm)
    probs = W.make_problems(ttc_lab.N_EVAL, seed=1, held=True)
    gold = [p.answer for p in probs]
    Np, P = n_params(pol), float(W.prompt_tokens(probs[0]))
    for name in ("buggy_harness.py", "harness.py"):
        H = load_path(str(HERE / name))
        table(H, data["pool"]["m"], gold, lm, Np, P, f"{name} (scores rounded to 1 decimal, as the colleague logs them)")


if __name__ == "__main__":
    main()
