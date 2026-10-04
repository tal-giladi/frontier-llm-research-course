"""Module 5 project: put measurements, projections and the decision rule for a stated constraint in one table.

    python labs/module-05/project/memo_inputs.py --context 32768 --batch 8 --gpu-gb 80 \\
        --decode runs/m05/l53/cpu-decode.json --prefill runs/m05/l53/cpu-prefill.json \\
        --heldout runs/m05/project/heldout-l51.json runs/m05/project/heldout-l52.json --margin 0.03 --min-speedup 1.3

For every arm (dense = Baseline-0, the 3:1 KDA hybrid, DSA) it prints:

1. **Memory at the stated context** from ``frontierlab.attention.accounting.m05_cache_bytes`` at Baseline-0 shape
   (BF16 K/V, fp32 state): bytes per sequence, and how many sequences fit next to the weights in the GPU
   (``--gpu-gb``, 10% kept free). PROJECTED from the formula, which the lab tests showed equals the measured
   ``Cache.nbytes()`` exactly.
2. **Speed at the stated context:** the measured paired speed-up against dense (with CI) at the profiled
   context nearest to ``--context``, labelled with the machine it was measured on, and the roofline
   projection to an H100 from the same JSON, labelled PROJECTED.
3. **Quality:** the paired held-out loss difference against the arm's dense control (from
   ``heldout_compare.py --out``), with its CI.
4. **The decision** by the rule you stated in your contract (``decide`` from your lesson 05.3 lab).

The table is an input to your memo, not the memo: the memo also says what each number cannot show.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import m05  # noqa: E402,F401
from frontierlab.attention import accounting  # noqa: E402
from frontierlab.attention.accounting import human_bytes  # noqa: E402
from frontierlab.labkit import load_target  # noqa: E402
from frontierlab.model import baseline0  # noqa: E402

lab53 = load_target(str(HERE.parent / "lesson-03" / "test_lab.py"))

ARM_CFG = {
    "dense": lambda b: b,
    "linear": lambda b: m05.arm_config("hybrid-kda", b, 1024, match_params=True, chunk=64),
    "dsa": lambda b: m05.arm_config("dsa", b, 1024, topk=2048),
}
HELDOUT_NAME = {"linear": "hybrid-kda-silu", "dsa": "sparse"}   # run-name prefixes; --linear-run overrides


def nearest(rows, L):
    return min(rows, key=lambda r: abs(r["L"] - L))


def heldout_diffs(paths):
    out = {}
    for p in paths or []:
        for r in json.loads(Path(p).read_text()):
            if "paired" in r:
                out[Path(r["run"]).name] = r["paired"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--gpu-gb", type=float, default=80)
    ap.add_argument("--decode", default=None, help="profile_attn.py --mode decode JSON")
    ap.add_argument("--prefill", default=None, help="profile_attn.py JSON (prefill)")
    ap.add_argument("--heldout", nargs="*", default=None, help="heldout_compare.py --out JSON files")
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--min-speedup", type=float, default=1.3)
    ap.add_argument("--linear-run", default=None, help="run-name prefix of the hybrid to use (default hybrid-kda-silu)")
    a = ap.parse_args()
    if a.linear_run:
        HELDOUT_NAME["linear"] = a.linear_run
    b0 = baseline0()
    weights = accounting.param_counts(b0)["total"] * 2
    budget = a.gpu_gb * 1e9 * 0.9 - weights
    print(f"Constraint: context {a.context:,} tokens, decode batch {a.batch}, {a.gpu_gb:.0f} GB GPU, "
          f"Baseline-0 shape (weights {human_bytes(weights)} BF16), margin {a.margin} nats, "
          f"required speed-up {a.min_speedup}x\n")
    print("1. Memory per sequence at the stated context (PROJECTED from the tested formula)")
    for arm, f in ARM_CFG.items():
        cfg = f(b0)
        per = accounting.m05_cache_bytes(cfg, a.context)
        print(f"   {arm:<7s} {human_bytes(per):>11s}   sequences that fit: {int(budget // per):>6d}")
    prof = {}
    for phase, path in (("decode", a.decode), ("prefill", a.prefill)):
        if path:
            prof[phase] = json.loads(Path(path).read_text())
    print("\n2. Speed at the profiled context nearest to the stated one")
    for phase, js in prof.items():
        row = nearest(js["rows"], a.context)
        proj = nearest(js["projected"], a.context)
        dev = js["args"]["device"]
        print(f"   {phase}: measured at L = {row['L']} on {dev} ({js['args']['dtype']}), layer shape "
              f"{js['args']['shape']}; projected to {js['args']['project_to']} for the same shape (PROJECTED)")
        lin_key = "linear.step" if phase == "decode" else "linear.core"
        for arm in ("linear", "dsa"):
            sp = row["arms"][arm]["speedup_vs_dense"]
            pj = proj["dense.sdpa"] / (proj[lin_key] if arm == "linear" else proj["dsa.total"])
            print(f"      {arm:<7s} measured speed-up {sp['speedup']:.2f} [{sp['ci'][0]:.2f}, {sp['ci'][1]:.2f}]   "
                  f"projected {pj:.1f}x (roofline, no launch overheads)")
    diffs = heldout_diffs(a.heldout)
    print("\n3. Quality: held-out loss difference against the arm's dense control (paired, 95% CI)")
    for arm, name in HELDOUT_NAME.items():
        key = next((k for k in diffs if k.startswith(name + "-s") or k == name), None)
        if key:
            d = diffs[key]
            print(f"      {arm:<7s} {d['mean_diff']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]  ({key})")
        else:
            print(f"      {arm:<7s} no held-out result passed for {name}")
    if "decode" in prof:
        print(f"\n4. Decision by your rule (decode speed-up measured on {prof['decode']['args']['device']}; "
              "re-run with the GPU profile before the memo is final)")
        row = nearest(prof["decode"]["rows"], a.context)
        for arm, name in HELDOUT_NAME.items():
            key = next((k for k in diffs if k.startswith(name + "-s") or k == name), None)
            if key is None:
                continue
            sp = row["arms"][arm]["speedup_vs_dense"]["ci"]
            print(f"      {arm:<7s} {lab53.decide(tuple(sp), tuple(diffs[key]['ci']), a.min_speedup, a.margin)}")


if __name__ == "__main__":
    main()
