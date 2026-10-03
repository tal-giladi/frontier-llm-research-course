"""Module 3 project: train Baseline-0 and the attention arms on both comparison axes.

    python labs/module-03/project/run.py --variant cpu --part A            # equal parameters (+ equal tokens)
    python labs/module-03/project/run.py --variant cpu --part B            # equal training FLOPs
    python labs/module-03/project/run.py --variant main --part A --seeds 0 1 2
    python labs/module-03/project/run.py --variant cpu --part A --dry-run  # print the plan only

Part A, equal parameters: every arm's SwiGLU width is changed so its non-embedding parameters equal
Baseline-0's (``--match-params``); all arms train the same number of tokens. Part B, equal training
FLOPs: the arms as designed (no width change), each trained for the number of steps whose training
FLOPs equal Baseline-0's (``--equal-flops-to``). Baseline-0 is the same run in both parts.

Runs go to runs/m03/<variant>/<part>-<arm>-s<seed>. Each is a normal frontierlab.train.loop run
(run card, JSONL metrics, exact resume): rerun the same command after a disconnect; finished runs
are skipped. On Colab add --max-minutes so the session checkpoints before it is cut off.

Main path (PROJECTED, pending the Module 3 pilot): baseline0, 9,500 steps x 32 x 8 x 1,024 tokens =
2.49B tokens, 1.96e18 training FLOPs per run, 1.84 GPU-hours per run at an assumed 30% MFU on one
H100 SXM; 7 arm-runs per seed (b0, 3 in part A, 3 in part B) = about 13 GPU-hours per seed, about
39 GPU-hours for 3 seeds, before evaluation.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_variant  # noqa: E402

VARIANTS = {   # preset, steps, batch, accum, seq, lr, extra loop args
    "cpu": ("toy", 200, 16, 1, 128, 1.5e-3, ["--eval-every", "200", "--ckpt-every", "100", "--eval-windows", "64"]),
    "t4": ("pilot-10m", 2000, 32, 1, 512, 3e-3, ["--dtype", "fp32", "--eval-every", "500", "--ckpt-every", "250"]),
    "main": ("baseline0", 9500, 32, 8, 1024, 3e-3, ["--dtype", "bf16", "--compile", "--peak", "H100-SXM",
                                                   "--eval-every", "500", "--ckpt-every", "500"]),
}
ARMS = ("gqa-kv", "mla", "local-global")


def plan(variant: str, part: str, seeds, lr=None):
    preset, steps, batch, accum, seq, lr0, extra = VARIANTS[variant]
    lr = lr or lr0
    jobs = []
    for seed in seeds:
        arms = [("A", "b0")] + [(part, arm) for arm in ARMS]
        for p, arm in arms:
            run = Path("runs/m03") / variant / f"{p}-{arm}-s{seed}"
            argv = ["--run", str(run), "--preset", preset, "--steps", str(steps), "--batch", str(batch),
                    "--grad-accum", str(accum), "--seq", str(seq), "--lr", f"{lr:g}", "--seed", str(seed),
                    "--question", f"Module 3 project, part {p}: {arm} vs Baseline-0", *extra]
            if arm != "b0":
                argv += ["--parent", str(Path("runs/m03") / variant / f"A-b0-s{seed}")]
            jobs.append((run, arm, p == "A" and arm != "b0", steps if p == "B" else None, argv))
    return jobs


def finished(run: Path) -> bool:
    log = run / "metrics.jsonl"
    if not log.exists():
        return False
    import json
    rows = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    card = run / "run_card.yaml"
    if not card.exists():
        return False
    import yaml
    steps = yaml.safe_load(card.read_text())["args"]["steps"]
    return any(r["split"] == "val" and r["step"] == steps for r in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--part", choices=["A", "B"], required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--lr", type=float, default=None, help="Baseline-0's tuned learning rate (Module 1 project)")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for run, arm, match, flops_to, argv in plan(a.variant, a.part, a.seeds, a.lr):
        if a.max_minutes:
            argv = argv + ["--max-minutes", str(a.max_minutes)]
        if finished(run):
            print(f"{run}: finished, skipping")
            continue
        cfg, argv = train_variant.plan(arm, argv, match_params=match, equal_flops_to=flops_to)
        steps = argv[argv.index("--steps") + 1]
        print(f"\n=== {run}  ({arm}, steps {steps}): {train_variant.m03.describe(cfg, VARIANTS[a.variant][4])}")
        if not a.dry_run:
            train_variant.train(cfg, argv)


if __name__ == "__main__":
    main()
