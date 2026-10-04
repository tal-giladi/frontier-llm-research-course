"""Lab 07.3, steps 3-4: the transfer test — a learning-rate sweep at three widths, in SP and in µP.

    python labs/module-07/lesson-03/sweep.py                    # free CPU: 24 short runs (see the lesson for the time)
    python labs/module-07/lesson-03/sweep.py --variant main     # main path, not run in this build
    python labs/module-07/lesson-03/sweep.py --analyze          # read the runs, apply the decision rule

Runs: widths (toy preset widened: head_dim 32, heads and SwiGLU scaled) x learning rates x {sp, mup}, AdamW
(``MuonAdamW`` in its AdamW configuration), the loop's cosine schedule, seed 0, same data order. µP is relative to
the middle width (``--mup-base-width``), so at that width SP and µP are the same parametrization.
Add ``--optimizer muon`` to repeat the test for Muon (spectral rule) — an extension, not part of the pass check.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m07  # noqa: E402

VARIANTS = {  # preset, widths, base width, steps, batch, seq, lrs, extra loop args
    "cpu": ("toy", (64, 128, 256), 128, 150, 16, 128, (1e-3, 3e-3, 1e-2, 3e-2),
            ["--eval-every", "150", "--eval-windows", "64", "--ckpt-every", "150", "--warmup", "15"]),
    "main": ("pilot-30m", (256, 512, 1024), 512, 2000, 32, 1024, (1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2),
             ["--eval-every", "2000", "--ckpt-every", "500", "--warmup", "100", "--dtype", "bf16", "--peak", "H100-SXM"]),
}


def run_name(param, width, lr):
    return f"{param}-w{width}-lr{lr:g}"


def analyze(root: Path, widths, lrs):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from frontierlab.labkit import load_target
    lab = load_target(str(Path(__file__).parent / "test_lab.py"))
    out = {}
    for param in ("sp", "mup"):
        sweep = {}
        for w in widths:
            sweep[w] = {}
            for lr in lrs:
                run = root / run_name(param, w, lr)
                if (run / "checkpoint.pt").exists():
                    sweep[w][lr] = m07.mean(m07.eval_losses(run))
        out[param] = sweep
        print(f"\n{param.upper()}: held-out loss (256 windows x 128 tokens) by width and learning rate")
        print(f"{'width':>6s} " + " ".join(f"{lr:>9g}" for lr in lrs))
        for w in widths:
            print(f"{w:6d} " + " ".join(f"{sweep[w].get(lr, float('nan')):9.4f}" for lr in lrs))
        try:
            v = lab.transfer_verdict({w: s for w, s in sweep.items() if s}, list(lrs))
            print(f"best lr per width: {v['best']}; largest shift {v['shift']} grid step(s); transfers: {v['transfers']}")
            out[param + "_verdict"] = {"best": {str(k): b for k, b in v["best"].items()}, "shift": v["shift"]}
        except NotImplementedError:
            print("transfer_verdict not implemented yet (TODO 6)")
    (root / "summary.json").write_text(json.dumps({k: {str(w): {str(lr): l for lr, l in d.items()} for w, d in s.items()}
                                                   if not k.endswith("verdict") else s for k, s in out.items()}, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cpu")
    ap.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--only", choices=["sp", "mup"], default=None)
    ap.add_argument("--out", type=Path, default=Path("runs/m07/l73"))
    a = ap.parse_args()
    preset, widths, base, steps, batch, seq, lrs, extra = VARIANTS[a.variant]
    root = a.out / a.variant / a.optimizer
    if a.analyze:
        analyze(root, widths, lrs)
        return
    for param in ("sp", "mup"):
        if a.only and param != a.only:
            continue
        for w in widths:
            for lr in lrs:
                argv = ["--optimizer", a.optimizer, "--preset", preset, "--width", str(w), "--batch", str(batch),
                        "--seq", str(seq), "--lr", f"{lr:g}", "--seed", "0", *extra,
                        "--question", f"lesson 07.3 transfer test: {param} width {w} lr {lr:g}"]
                if param == "mup":
                    argv += ["--mup-base-width", str(base)]
                dt = m07.run_arm(root / run_name(param, w, lr), argv, steps)
                if dt:
                    print(f"{param} w{w} lr{lr:g}: {dt:.0f} s")
    analyze(root, widths, lrs)


if __name__ == "__main__":
    main()
