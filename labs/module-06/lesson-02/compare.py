"""Lab 06.2, step 4: stability and quality of b0 / HC / mHC at the standard and the raised learning rates.

    python labs/module-06/lesson-02/compare.py                     # runs/m06/cpu
    python labs/module-06/lesson-02/compare.py --variant main --device cuda

Per arm and learning rate: final held-out loss (256 fixed windows), max and final gradient norm, loss spikes
(``frontierlab.optim.stability.detect_spikes`` on the logged training loss), and for HC/mHC the composite
Amax gains of the residual maps (forward = max row sum, backward = max column sum; 1.0 for doubly-stochastic
maps) at the first and last logged step and their maximum over the run. Then the contract's rules.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m06  # noqa: E402
from frontierlab.optim.stability import detect_spikes  # noqa: E402

LRS = (None, 1e-2, 3e-2)           # None = the variant's learning rate (1.5e-3)


def gains(run):
    rows = [r["hyper"] for r in m06.blocks_log(run) if r.get("hyper")]
    if not rows:
        return None
    return {"first": rows[0]["composite_fwd"], "last": rows[-1]["composite_fwd"],
            "max_fwd": max(r["composite_fwd_max"] for r in rows), "max_bwd": max(r["composite_bwd_max"] for r in rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(m06.VARIANTS), default="cpu")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    v = m06.VARIANTS[a.variant]
    kw = dict(n=v["eval_n"], T=v["eval_T"], device=a.device)
    table = {}
    print(f"{'lr':>7s} {'arm':4s} {'held-out':>9s} {'gnorm max':>9s} {'gnorm end':>9s} {'spikes':>6s} "
          f"{'gain first':>10s} {'gain last':>9s} {'max fwd':>9s} {'max bwd':>9s}")
    for lr in LRS:
        for arm in ("b0", "hc", "mhc"):
            run = m06.run_dir(a.variant, arm, 0, lr)
            if not (run / "checkpoint.pt").exists():
                continue
            rows = m06.train_rows(run)
            steps, losses = [r["step"] for r in rows], [r["loss"] for r in rows]
            gn = [r["grad_norm"] for r in rows]
            sp = detect_spikes(steps, losses, window=5, k=6.0, min_rel=0.05)
            ho = float(np.mean(m06.eval_losses(run, **kw)))
            g = gains(run)
            table[(lr, arm)] = {"heldout": ho, "spikes": len(sp), "g": g, "losses": m06.eval_losses(run, **kw),
                                "finite": bool(np.isfinite(losses).all())}
            gs = (f"{g['first']:10.3f} {g['last']:9.3f} {g['max_fwd']:9.3f} {g['max_bwd']:9.3f}" if g else
                  f"{'-':>10s} {'-':>9s} {'-':>9s} {'-':>9s}")
            print(f"{lr or v['lr']:7.1e} {arm:4s} {ho:9.4f} {max(gn):9.2f} {gn[-1]:9.2f} {len(sp):6d} {gs}")
    # seed noise at the standard learning rate
    print("\nStandard learning rate, seeds 0 and 1:")
    per_seed = {}
    for arm in ("b0", "hc", "mhc"):
        runs = [m06.run_dir(a.variant, arm, s) for s in (0, 1)]
        if all((r / "checkpoint.pt").exists() for r in runs):
            per_seed[arm] = [float(np.mean(m06.eval_losses(r, **kw))) for r in runs]
            print(f"  {arm:4s} {' '.join(f'{x:.4f}' for x in per_seed[arm])}")
    if all(k in per_seed for k in ("b0", "hc", "mhc")):
        win = {arm: m06.seed_mean_losses([m06.run_dir(a.variant, arm, s) for s in (0, 1)], **kw) for arm in per_seed}
        nf = m06.noise_floor(per_seed["b0"], 2)
        print(f"  b0 seed std {nf['seed_std']:.4f}, MDE (2 seeds per arm) {nf['mde']:.4f} nats")
        for x in ("hc", "mhc"):
            print(f"  {x} - b0 (equal tokens, paired, seed-averaged): {m06.fmt(m06.compare(win[x], win['b0']))}")
        print(f"  mhc - hc: {m06.fmt(m06.compare(win['mhc'], win['hc']))}")
    print("\nRules (stated in the lesson's contract):")
    for lr in LRS[1:]:
        hc, mhc, b0 = table.get((lr, "hc")), table.get((lr, "mhc")), table.get((lr, "b0"))
        if not (hc and mhc and b0):
            continue
        hc_unstable = (not hc["finite"]) or hc["spikes"] > b0["spikes"] or (hc["g"] and hc["g"]["max_fwd"] > 2.0)
        mhc_bounded = mhc["g"] is not None and mhc["g"]["max_fwd"] < 1.1 and mhc["g"]["max_bwd"] < 1.6
        d = m06.compare(mhc["losses"], hc["losses"])
        print(f"  lr {lr:g}: H1 (HC shows instability: non-finite loss, more spikes than b0, or composite gain > 2): "
              f"{'OBSERVED' if hc_unstable else 'NOT OBSERVED'}; H2 (mHC gains stay near 1): "
              f"{'HOLDS' if mhc_bounded else 'FAILS'}; mHC - HC held-out {m06.fmt(d)}")


if __name__ == "__main__":
    main()
