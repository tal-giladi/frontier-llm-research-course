"""Module 7 project: Muon vs tuned AdamW at equal tokens and equal wall-clock, with a µP transfer test.

    python labs/module-07/project/run_project.py --variant cpu                # every stage, free CPU
    python labs/module-07/project/run_project.py --variant cpu --stage tune   # one stage
    python labs/module-07/project/run_project.py --variant main --print       # main-path commands (not run in this build)
    python labs/module-07/project/run_project.py --variant cpu --stage report

Stages (each skips finished runs, resumes interrupted ones):
  tune      learning-rate sweep for each optimizer at the small width (µP relative to it): the same grid and the
            same number of runs for both — equal tuning budgets
  transfer  at the large width: each optimizer at its transferred learning rate and one grid step either side
  seeds     two more seeds of each optimizer at the large width, at its transferred learning rate
  wallclock AdamW at the large width for N_wc steps, N_wc = steps x (measured Muon step time / AdamW step time),
            three seeds — the equal-wall-clock arm (Muon keeps its steps)
  report    held-out loss per run (256 fixed windows), the transfer verdict, the seed noise floor, both comparisons
            with intervals, and the decision rule of the contract
Both optimizers run through frontierlab.optim.MuonAdamW (AdamW groups or Muon + AdamW groups), µP on, the loop's
cosine schedule, QK-norm on, the same data order per seed.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m07  # noqa: E402
from frontierlab.stats import min_detectable_effect, paired_bootstrap  # noqa: E402

VARIANTS = {
    # preset, small width, large width, steps, batch, seq, lr grid, extra loop args
    "cpu": ("toy", 64, 128, 200, 16, 128, (1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
            ["--eval-every", "100000", "--eval-windows", "64", "--ckpt-every", "100", "--warmup", "20", "--log-every", "10"]),
    "main": ("baseline0", 384, 768, 9500, 32, 1024, (1e-3, 2e-3, 4e-3, 8e-3),
             ["--grad-accum", "8", "--eval-every", "100000", "--ckpt-every", "500", "--warmup", "200", "--log-every", "20",
              "--dtype", "bf16", "--peak", "H100-SXM", "--loss", "plain"]),
}
OPTS = ("adamw", "muon")
SEEDS = (0, 1, 2)


def name(opt, width, lr, seed=0, steps=None):
    s = f"{opt}-w{width}-lr{lr:g}-s{seed}"
    return s + (f"-n{steps}" if steps else "")


def argv_for(v, opt, width, lr, seed, small):
    preset, _, _, _, batch, seq, _, extra = v
    return ["--optimizer", opt, "--preset", preset, "--width", str(width), "--mup-base-width", str(small),
            "--batch", str(batch), "--seq", str(seq), "--lr", f"{lr:g}", "--seed", str(seed), *extra,
            "--question", f"module 7 project: {opt} width {width} lr {lr:g} seed {seed}"]


def heldout(run, device):
    return m07.eval_losses(run, device=device)


def best_lr(root, opt, width, grid, device):
    losses = {lr: m07.mean(heldout(root / name(opt, width, lr), device)) for lr in grid
              if (root / name(opt, width, lr) / "checkpoint.pt").exists()}
    return min(sorted(losses), key=losses.get), losses


def step_seconds(run):
    return m07.wallclock(run) / max(1, m07.train_rows(run)[-1]["step"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cpu")
    ap.add_argument("--stage", choices=["tune", "transfer", "seeds", "wallclock", "report", "all"], default="all")
    ap.add_argument("--print", action="store_true", help="print the training commands instead of running them")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--wallclock-ratio", type=float, default=None,
                    help="Muon/AdamW step-time ratio to use (default: measured from the seed runs)")
    ap.add_argument("--out", type=Path, default=Path("runs/m07/project"))
    a = ap.parse_args()
    v = VARIANTS[a.variant]
    preset, small, large, steps, batch, seq, grid, extra = v
    root = a.out / a.variant
    stages = ["tune", "transfer", "seeds", "wallclock", "report"] if a.stage == "all" else [a.stage]

    def train(opt, width, lr, seed=0, n=steps, tag=None):
        run = root / name(opt, width, lr, seed, tag)
        argv = argv_for(v, opt, width, lr, seed, small)
        if a.print:
            print("python -m frontierlab.optim.train --run", run, "--steps", n, " ".join(argv))
            return
        m07.run_arm(run, argv, n)

    best = {}
    if "tune" in stages:
        for opt in OPTS:
            for lr in grid:
                train(opt, small, lr)
    if a.print and "tune" not in stages:
        pass
    for opt in OPTS:
        try:
            best[opt] = best_lr(root, opt, small, grid, a.device)[0]
        except ValueError:
            best[opt] = grid[1]                                 # --print before tuning: placeholder
    if "transfer" in stages:
        for opt in OPTS:
            i = grid.index(best[opt])
            for j in (i - 1, i, i + 1):
                if 0 <= j < len(grid):
                    train(opt, large, grid[j])
    if "seeds" in stages:
        for opt in OPTS:
            for seed in SEEDS[1:]:
                train(opt, large, best[opt], seed)
    ratio = a.wallclock_ratio
    if ratio is None and not a.print:
        try:
            t = {opt: statistics.mean(step_seconds(root / name(opt, large, best[opt], s)) for s in SEEDS) for opt in OPTS}
            ratio = t["muon"] / t["adamw"]
        except Exception:  # noqa: BLE001 - seed runs not there yet
            ratio = None
    n_wc = int(steps * ratio) if ratio else None
    if "wallclock" in stages:
        if n_wc is None and a.print:
            print("# wallclock stage: steps = steps x measured Muon/AdamW step-time ratio (known after the seeds stage)")
        elif n_wc is None:
            raise SystemExit("run the seeds stage first (or pass --wallclock-ratio)")
        else:
            for seed in SEEDS:
                train("adamw", large, best["adamw"], seed, n_wc, tag=n_wc)
    if "report" in stages and not a.print:
        report(root, a.device, v, best, ratio, n_wc)


def report(root, device, v, best, ratio, n_wc):
    preset, small, large, steps, batch, seq, grid, extra = v
    out = {"best_small": best}
    print(f"tuning at width {small} (held-out loss, 256 windows x 128 tokens), {len(grid)} runs per optimizer:")
    for opt in OPTS:
        b, losses = best_lr(root, opt, small, grid, device)
        print(f"  {opt:5s} " + "  ".join(f"lr {lr:g}: {l:.4f}" for lr, l in sorted(losses.items())) + f"   best {b:g}")
    print(f"\ntransfer to width {large} (muP, lr from width {small}):")
    for opt in OPTS:
        sub = [lr for lr in grid if (root / name(opt, large, lr) / "checkpoint.pt").exists()]
        losses = {lr: m07.mean(heldout(root / name(opt, large, lr), device)) for lr in sub}
        b = min(sorted(losses), key=losses.get)
        shift = abs(grid.index(b) - grid.index(best[opt]))
        out[f"transfer_{opt}"] = {"losses": {str(k): x for k, x in losses.items()}, "best_large": b, "shift": shift}
        print(f"  {opt:5s} " + "  ".join(f"lr {lr:g}: {l:.4f}" for lr, l in sorted(losses.items()))
              + f"   best {b:g}; transferred {best[opt]:g}; shift {shift} grid step(s) -> {'transfers' if shift == 0 else 'off by ' + str(shift)}")
    per_seed = {}
    for opt in OPTS:
        per_seed[opt] = [heldout(root / name(opt, large, best[opt], s), device) for s in SEEDS
                         if (root / name(opt, large, best[opt], s) / "checkpoint.pt").exists()]
    if all(len(per_seed[o]) == len(SEEDS) for o in OPTS):
        means = {o: [float(np.mean(x)) for x in per_seed[o]] for o in OPTS}
        sd = {o: statistics.stdev(means[o]) for o in OPTS}
        mde = min_detectable_effect(max(sd.values()), len(SEEDS))
        print(f"\nseed noise floor at width {large}: AdamW per-seed {['%.4f' % m for m in means['adamw']]} (std {sd['adamw']:.4f}); "
              f"Muon {['%.4f' % m for m in means['muon']]} (std {sd['muon']:.4f}); MDE with {len(SEEDS)} seeds/arm = {mde:.4f}")
        d_tok = paired_bootstrap(np.mean(per_seed["muon"], 0), np.mean(per_seed["adamw"], 0))
        seed_diff = [m - a for m, a in zip(means["muon"], means["adamw"])]
        print(f"equal tokens ({steps} steps each): Muon - AdamW = {d_tok['mean_diff']:+.4f}, paired-window CI "
              f"[{d_tok['ci'][0]:+.4f}, {d_tok['ci'][1]:+.4f}] (seed-averaged); per-seed differences "
              f"{['%+.4f' % x for x in seed_diff]}")
        out.update(seed_means=means, mde=mde, equal_tokens=d_tok, seed_diff_tokens=seed_diff)
        if n_wc:
            wc = [heldout(root / name("adamw", large, best["adamw"], s, n_wc), device) for s in SEEDS
                  if (root / name("adamw", large, best["adamw"], s, n_wc) / "checkpoint.pt").exists()]
            if len(wc) == len(SEEDS):
                d_wc = paired_bootstrap(np.mean(per_seed["muon"], 0), np.mean(wc, 0))
                wmeans = [float(np.mean(x)) for x in wc]
                sdiff = [m - w for m, w in zip(means["muon"], wmeans)]
                print(f"equal wall-clock (AdamW {n_wc} steps = {steps} x measured ratio {ratio:.3f}): Muon - AdamW = "
                      f"{d_wc['mean_diff']:+.4f} [{d_wc['ci'][0]:+.4f}, {d_wc['ci'][1]:+.4f}]; per-seed {['%+.4f' % x for x in sdiff]}")
                out.update(equal_wallclock=d_wc, wallclock_ratio=ratio, n_wc=n_wc, seed_diff_wallclock=sdiff)
                if d_wc["ci"][1] < -0.02 and all(x < 0 for x in sdiff):
                    verdict = "adopt Muon"
                elif d_wc["ci"][0] > -0.02:
                    verdict = "keep AdamW"
                else:
                    verdict = "inconclusive (keep AdamW, the simpler choice)"
                out["verdict"] = verdict
                print(f"verdict by the rule: {verdict}")
        print("\ndecision rule (contract): adopt Muon for Recipe-R only if, at equal wall-clock, every per-seed difference "
              "and the upper CI bound are below -0.02 nats; reject if the lower bound is above -0.02; otherwise inconclusive.")
    (root / "report.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
