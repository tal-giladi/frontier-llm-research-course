"""Lab 06.1, step 4: compare the arms on Eval v0's held-out windows and apply the contract's decision rule.

    python labs/module-06/lesson-01/compare.py                     # runs/m06/cpu
    python labs/module-06/lesson-01/compare.py --variant main --device cuda

Held-out next-token loss of the main head on 256 fixed validation windows (CPU: 128 tokens each; GPU variants:
the training length), per seed; differences are paired by window after averaging the seeds (lesson 01.4).
Equal tokens: mtp-ds and mtp-meta vs b0 at the same steps. Equal training FLOPs: mtp-ds vs Baseline-0
trained for the steps whose FLOPs match (the decision axis). Also: the noise floor from b0's seeds, the MDE,
the final MTP loss, measured wall-clock per arm.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m06  # noqa: E402

MARGIN = 0.0     # the decision rule's threshold in nats (stated in the lesson's contract)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(m06.VARIANTS), default="cpu")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    v = m06.VARIANTS[a.variant]
    vocab = m06.vocab_size()
    eq = m06.equal_flops_steps("mtp-ds", "b0", a.variant, vocab)
    arms = {"b0": dict(arm="b0"), "mtp-ds": dict(arm="mtp-ds"), "mtp-meta": dict(arm="mtp-meta"),
            f"b0@{eq}": dict(arm="b0", steps=eq)}
    kw = dict(n=v["eval_n"], T=v["eval_T"], device=a.device)
    per_seed, per_window, wc = {}, {}, {}
    for name, spec in arms.items():
        runs = [m06.run_dir(a.variant, spec["arm"], s, steps=spec.get("steps")) for s in a.seeds]
        runs = [r for r in runs if (r / "checkpoint.pt").exists()]
        if not runs:
            continue
        losses = [m06.eval_losses(r, **kw) for r in runs]
        per_seed[name] = [float(np.mean(x)) for x in losses]
        per_window[name] = np.mean(losses, axis=0)
        wc[name] = float(np.mean([m06.wallclock(r) for r in runs]))
    print(f"{'arm':12s} {'held-out (per seed)':>26s} {'mean':>8s} {'train MFLOP/tok':>16s} {'wall s':>8s} {'final MTP loss':>15s}")
    for name, spec in arms.items():
        if name not in per_seed:
            continue
        fl = m06.flops_per_token(spec["arm"], a.variant, vocab) / 1e6
        run0 = m06.run_dir(a.variant, spec["arm"], a.seeds[0], steps=spec.get("steps"))
        mtp = [r.get("mtp_loss") for r in m06.blocks_log(run0) if r.get("mtp_loss") is not None]
        print(f"{name:12s} {' '.join(f'{x:.4f}' for x in per_seed[name]):>26s} {np.mean(per_seed[name]):8.4f} {fl:16.3f}"
              f" {wc[name]:8.0f} {(f'{mtp[-1]:.4f}' if mtp else '-'):>15s}")
    if "b0" in per_seed and len(per_seed["b0"]) > 1:
        nf = m06.noise_floor(per_seed["b0"], len(per_seed["b0"]))
        print(f"\nnoise floor: b0 seed std {nf['seed_std']:.4f} nats; MDE with {len(per_seed['b0'])} seeds per arm "
              f"{nf['mde']:.4f} nats (two seeds give a very rough estimate)")
    print("\nPaired by window (seed-averaged), a - b, 95% bootstrap CI:")
    pairs = [("mtp-ds", "b0", "equal tokens"), ("mtp-meta", "b0", "equal tokens"), ("mtp-ds", f"b0@{eq}", "equal FLOPs")]
    res = {}
    for x, y, axis in pairs:
        if x in per_window and y in per_window:
            r = res[(x, y)] = m06.compare(per_window[x], per_window[y])
            d = [p - q for p, q in zip(per_seed[x], per_seed[y])]
            print(f"  {x:9s} - {y:9s} ({axis:12s}): {m06.fmt(r)}   per seed {' '.join(f'{z:+.4f}' for z in d)}")
    key = ("mtp-ds", f"b0@{eq}")
    if key in res:
        r = res[key]
        d = [p - q for p, q in zip(per_seed["mtp-ds"], per_seed[f"b0@{eq}"])]
        if r["ci"][1] < -MARGIN and all(z < 0 for z in d):
            verdict = "ADOPT MTP as a training objective for Lineage-F (better at equal training FLOPs)"
        elif r["ci"][0] > MARGIN:
            verdict = "REJECT at this scale: worse at equal training FLOPs"
        else:
            verdict = "INCONCLUSIVE at this scale: keep next-token training as the default, carry MTP as a draft-head option"
        print(f"\nDecision rule (equal FLOPs): {verdict}")


if __name__ == "__main__":
    main()
