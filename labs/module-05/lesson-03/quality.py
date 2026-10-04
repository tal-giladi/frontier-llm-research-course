"""Lab 05.3: Eval Suite v1 on the Module 5 arms, paired against their dense baselines, with the MiniMax test.

    python labs/module-05/lesson-03/quality.py                         # free CPU: the 05.1 and 05.2 CPU runs
    python labs/module-05/lesson-03/quality.py --variant main --device cuda --bf16 --lengths 1024 4096 16384

Pairs (each arm against the dense model it is a controlled comparison with):

    05.1  b0-s0      vs  hybrid-kda-s0, hybrid-gdn-s0      (from scratch, equal parameters and tokens)
    05.2  control    vs  sparse                            (from the same parent, equal tokens)

For each model it runs Eval v1 (``frontierlab.evals.suite_v1.run_suite``: position loss, context gain,
retrieval at five depths and two-hop chains, every item with its evidence-ablated twin) once and caches the
JSON next to the run. Then it prints ``suite_v1.compare`` for every pair and the summary the MiniMax
counter-hypothesis needs: the evidence effect pooled over retrieval cells (hops = 1) and over two-hop cells
(hops = 2), and the paired difference arm - dense for each. MiniMax's claim, restated as a testable
prediction: a hybrid can match dense on retrieval and natural-text loss while losing on multi-hop use of
the context. At course scale this may not be detectable; the lab reports whatever the intervals allow.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m05  # noqa: E402,F401  (registers every kind)
from frontierlab.data.loader import TokenData  # noqa: E402
from frontierlab.data.prepare import DEFAULT_OUT  # noqa: E402
from frontierlab.evals import suite_v1 as v1  # noqa: E402
from frontierlab.stats import paired_bootstrap  # noqa: E402

PAIRS = {
    "cpu": [("runs/m05/l51/cpu/b0-s0", "runs/m05/l51/cpu/hybrid-kda-s0"),
            ("runs/m05/l51/cpu/b0-s0", "runs/m05/l51/cpu/hybrid-gdn-s0"),
            ("runs/m05/l51/cpu/b0-s0", "runs/m05/l51/cpu/hybrid-kda-silu-s0"),
            ("runs/m05/l52/cpu/control", "runs/m05/l52/cpu/sparse")],
    "t4": [("runs/m05/l51/t4/b0-s0", "runs/m05/l51/t4/hybrid-kda-s0"),
           ("runs/m05/l52/t4/control", "runs/m05/l52/t4/sparse")],
    "main": [("runs/m05/l51/main/b0-s0", "runs/m05/l51/main/hybrid-kda-s0"),
             ("runs/m05/l51/main/b0-s0", "runs/m05/l51/main/hybrid-gdn-s0"),
             ("runs/m05/l52/main/control", "runs/m05/l52/main/sparse")],
}
TRAIN_LEN = {"cpu": 256, "t4": 512, "main": 1024}


def evaluate(run: str, lengths, train_len, n, device, bf16, data_root):
    out = Path(run) / f"eval_v1_{'_'.join(map(str, lengths))}.json"
    if out.exists():
        return json.loads(out.read_text())
    data = TokenData("val", data_root)
    vocab = v1.vocab_from_tokenizer(DEFAULT_OUT / "tokenizer.json", TokenData("val", DEFAULT_OUT))
    model, _ = m05.load_run(run, device)
    res = v1.run_suite(model, vocab, data, list(lengths), train_len, n=n, device=device,
                       autocast_dtype=torch.bfloat16 if bf16 else None)
    res["model"] = {"run": run, "attention": model.config.attention}
    out.write_text(json.dumps(res))
    return res


def pooled_effect(res, hops):
    a, b = [], []
    for c in res["synthetic"]:
        if c["hops"] == hops:
            a += [x["logp_cand"] for x in c["scores"]]
            b += [x["logp_cand"] for x in c["scores_ablated"]]
    return a, b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(PAIRS), default="cpu")
    ap.add_argument("--lengths", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--full", action="store_true", help="also print suite_v1.compare for every pair")
    a = ap.parse_args()
    long_root = DEFAULT_OUT.parent / "v0-long"
    root = long_root if (long_root / "meta.json").exists() else DEFAULT_OUT
    print(f"Eval v1 documents from {root.name}; lengths {a.lengths}; {a.n} items per synthetic cell\n")
    results = {}
    for pair in PAIRS[a.variant]:
        for run in pair:
            if run not in results and Path(run, "checkpoint.pt").exists():
                results[run] = evaluate(run, a.lengths, TRAIN_LEN[a.variant], a.n, a.device, a.bf16, root)
    print("MiniMax test: evidence effect (logp_cand with - without the evidence), pooled; then arm - dense, paired")
    print(f"{'pair':<44s} {'hops':>4s} {'dense':>16s} {'arm':>16s} {'arm - dense [95% CI]':>28s}")
    for base, arm in PAIRS[a.variant]:
        if base not in results or arm not in results:
            print(f"  missing run for {base} / {arm}; train it first")
            continue
        for hops in (1, 2):
            da, db = pooled_effect(results[base], hops)
            ea, eb = pooled_effect(results[arm], hops)
            eff_d = [x - y for x, y in zip(da, db)]
            eff_a = [x - y for x, y in zip(ea, eb)]
            pd, pa = paired_bootstrap(da, db), paired_bootstrap(ea, eb)
            diff = paired_bootstrap(eff_a, eff_d)
            name = f"{Path(base).name} vs {Path(arm).name}"
            print(f"{name:<44s} {hops:>4d} {pd['mean_diff']:+7.3f}{'':>9s} {pa['mean_diff']:+7.3f}{'':>9s} "
                  f"{diff['mean_diff']:+8.3f} [{diff['ci'][0]:+.3f}, {diff['ci'][1]:+.3f}]")
        for L in a.lengths:
            nb = {n["length"]: n for n in results[base]["natural"]}.get(L)
            na = {n["length"]: n for n in results[arm]["natural"]}.get(L)
            if nb and na and nb.get("docs") and na.get("docs"):
                lb = [sum(x) / len(x) for x in zip(*nb["doc_mean_by_bucket"])]
                la = [sum(x) / len(x) for x in zip(*na["doc_mean_by_bucket"])]
                r = paired_bootstrap(la, lb)
                print(f"{'':<44s} natural-text loss L={L} ({len(lb)} docs), arm - dense: {r['mean_diff']:+.4f} "
                      f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
        if a.full:
            print(v1.compare(results[base], results[arm]))
        print()


if __name__ == "__main__":
    main()
