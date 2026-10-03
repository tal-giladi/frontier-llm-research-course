"""Module 3 project: quality of each arm against Baseline-0 on Eval v0, per comparison axis, with the
pre-stated decision rule; then the cache arithmetic for the memo's serving constraint.

    python labs/module-03/project/compare.py runs/m03/cpu --margin 0.05
    python labs/module-03/project/compare.py runs/m03/main --margin 0.02 --context 32768 --concurrent 256 \\
        --gpu-gb 80 --decode runs/m03/main-decode.json

For each part (A: equal parameters, B: equal training FLOPs), arm and metric (held-out loss; LAMBADA
target log-probability per passage):

* per seed: the paired mean difference over the same items (arm - b0, same seed) with its bootstrap
  interval — evaluation noise only;
* across seeds: the mean of the per-seed differences with a t interval over seeds — the interval the
  decision uses (with 2 seeds it is very wide; that is the honest answer);
* Baseline-0's seed standard deviation, for the noise floor.

Decision rule (stated in the project's contract): an arm is NON-INFERIOR on held-out loss if the upper
end of the seed-level 95% interval of (arm - b0) is below ``--margin``; INFERIOR if its lower end is
above the margin; otherwise INCONCLUSIVE.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m03  # noqa: E402,F401  (registers the attention kinds)
from frontierlab.attention import accounting  # noqa: E402
from frontierlab.model import ModelConfig  # noqa: E402
from frontierlab.stats import paired_bootstrap  # noqa: E402

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def load(root: Path) -> dict:
    out = {}
    for f in sorted(root.glob("*/eval_v0_val.json")):
        part, rest = f.parent.name.split("-", 1)                 # <part>-<arm>-s<seed>
        arm, seed = rest.rsplit("-s", 1)
        r = json.loads(f.read_text())
        out[(part, arm, int(seed))] = {"loss": np.array(r["heldout"]["losses"]),
                                       "logprob": np.array([it["logprob"] for it in r["lambada"]["items"]]),
                                       "run": f.parent}
    return out


def seed_interval(d) -> tuple[float, tuple[float, float]]:
    d = np.asarray(d, dtype=float)
    if d.size < 2:
        return float(d.mean()), (float("nan"), float("nan"))
    half = T975[d.size - 1] * d.std(ddof=1) / math.sqrt(d.size)
    return float(d.mean()), (float(d.mean() - half), float(d.mean() + half))


def verdict(ci, margin) -> str:
    if math.isnan(ci[0]):
        return "INCONCLUSIVE (one seed)"
    if ci[1] < margin:
        return "NON-INFERIOR"
    if ci[0] > margin:
        return "INFERIOR"
    return "INCONCLUSIVE"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--margin", type=float, default=0.02, help="non-inferiority margin on held-out loss (nats)")
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--concurrent", type=int, default=256)
    ap.add_argument("--gpu-gb", type=float, default=80.0)
    ap.add_argument("--overhead", type=float, default=0.10, help="fraction of GPU memory kept for activations etc.")
    ap.add_argument("--decode", type=Path, default=None, help="decode_compare.py JSON with measured latency")
    a = ap.parse_args()
    res = load(a.root)
    if not res:
        raise SystemExit(f"no eval_v0_val.json under {a.root}; run evaluate.py first")
    seeds = sorted(s for (p, arm, s) in res if p == "A" and arm == "b0")
    b0 = [res[("A", "b0", s)]["loss"].mean() for s in seeds]
    print(f"Baseline-0: seeds {seeds}, held-out loss {np.mean(b0):.4f}"
          + (f", seed std {np.std(b0, ddof=1):.4f}" if len(seeds) > 1 else ""))
    verdicts = {}
    for part, title in (("A", "equal parameters, equal tokens"), ("B", "equal training FLOPs")):
        arms = sorted({arm for (p, arm, _) in res if p == part and arm != "b0"})
        if arms:
            print(f"\nPart {part} ({title}); differences are arm - b0, same seed, same items")
        for arm in arms:
            for metric in ("loss", "logprob"):
                per, line = [], []
                for s in seeds:
                    if (part, arm, s) in res:
                        pb = paired_bootstrap(res[(part, arm, s)][metric], res[("A", "b0", s)][metric])
                        per.append(pb["mean_diff"])
                        line.append(f"s{s} {pb['mean_diff']:+.4f} [{pb['ci'][0]:+.4f}, {pb['ci'][1]:+.4f}]")
                m, ci = seed_interval(per)
                label = "held-out loss  " if metric == "loss" else "LAMBADA logprob"
                print(f"  {arm:<13s} {label} mean {m:+.4f}  95% CI over seeds [{ci[0]:+.4f}, {ci[1]:+.4f}]"
                      f"   per seed (window/passage bootstrap): " + "; ".join(line))
                if metric == "loss":
                    verdicts[(part, arm)] = verdict(ci, a.margin)
                    print(f"  {'':<13s} -> {verdicts[(part, arm)]} at margin {a.margin} nats (higher loss is worse)")
    budget = a.gpu_gb * 1e9 * (1 - a.overhead)
    print(f"\nServing constraint: {a.concurrent} sequences x {a.context:,} tokens on one {a.gpu_gb:g} GB GPU "
          f"({a.overhead:.0%} kept free), BF16 weights and cache")
    measured = {}
    if a.decode and a.decode.exists():
        rows = json.loads(a.decode.read_text())
        top = max(r["S"] for r in rows)
        measured = {r["arm"]: r for r in rows if r["S"] == top}
    for (part, arm, s), r in sorted(res.items()):
        if s != seeds[0]:
            continue
        cfg = ModelConfig(**yaml.safe_load((r["run"] / "run_card.yaml").read_text())["config"])
        weights = accounting.param_counts(cfg)["total"] * 2
        kv = accounting.kv_bytes(cfg, a.context)
        need = weights + a.concurrent * kv
        lat = measured.get(arm) or measured.get(f"{arm}-absorbed") or {}
        print(f"  {part}-{arm:<13s} cache {kv / 2**20:7.1f} MiB/sequence, total {need / 1e9:6.1f} GB of "
              f"{budget / 1e9:.1f} -> {'fits' if need <= budget else 'does NOT fit'} (PROJECTED, formula)"
              f"   quality: {verdicts.get((part, arm), 'baseline')}"
              + (f"   measured decode {lat['step_ms']:.3f} ms at S={lat['S']}" if lat else ""))


if __name__ == "__main__":
    main()
