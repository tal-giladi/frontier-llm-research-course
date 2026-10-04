"""Lab 07.4, step 3: compare the decay branches with the cosine runs, and count what each cost.

    python labs/module-07/lesson-04/compare_schedules.py               # runs/m07/l74/cpu
    python labs/module-07/lesson-04/compare_schedules.py runs/m07/l74/main --device cuda

Held-out loss on 256 fixed validation windows of 128 tokens for every arm and for the stable run's kept
checkpoints (before any decay); paired differences (lesson 01.4) of each WSD arm against the cosine run of the
same total length; the training-loss drop during each decay; and the steps each way of getting all lengths costs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m07  # noqa: E402
from frontierlab.labkit import load_target  # noqa: E402

lab = load_target(str(Path(__file__).parent / "test_lab.py"))

PAIRS = [("wsd-T300-d10", "cos-T300", 270), ("wsd-T600-d10", "cos-T600", 540), ("wsd-T600-d20", "cos-T600", 480),
         ("wsd-T600-d10-sqrt", "cos-T600", 540), ("wsd-T900-d10", "cos-T600+300", 810)]


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    root = Path(args[0]) if args else Path("runs/m07/l74/cpu")
    device = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"
    scale = 10 if root.name == "main" else 1
    print("held-out loss, 256 windows x 128 tokens")
    for ck in sorted((root / "stable").glob("step*.pt"), key=lambda p: int(p.stem[4:])):
        print(f"  stable run at step {int(ck.stem[4:]):5d} (no decay)   {m07.mean(m07.eval_losses(ck, device=device)):.4f}")
    for name in ("cos-T300", "cos-T600", "cos-T600+300"):
        if (root / name / "checkpoint.pt").exists():
            print(f"  {name:28s}            {m07.mean(m07.eval_losses(root / name, device=device)):.4f}")
    print(f"\n{'WSD arm':20s} {'held-out':>9s} {'vs cosine (paired)':>28s} {'decay drop (train)':>19s}")
    for arm, ref, s0 in PAIRS:
        a, b = root / arm, root / ref
        if not ((a / "checkpoint.pt").exists() and (b / "checkpoint.pt").exists()):
            continue
        la, lb = m07.eval_losses(a, device=device), m07.eval_losses(b, device=device)
        stable_rows = [r for r in m07.train_rows(root / "stable") if r["step"] <= s0 * scale]
        rows = stable_rows + m07.train_rows(a)            # the branch continues the stable run
        try:
            drop = lab.decay_drop([r["step"] for r in rows], [r["loss"] for r in rows], s0 * scale, window=10)
            drop_s = f"{drop:+.4f}"
        except NotImplementedError:
            drop_s = "TODO 4"
        print(f"{arm:20s} {m07.mean(la):9.4f}   {ref:>12s} {m07.compare(la, lb):>15s} {drop_s:>19s}")
    try:
        plan = lab.branch_plan([300 * scale, 600 * scale], 0.1)
        c = lab.branch_cost(plan)
        print(f"\ncost of lengths 300 and 600 (x{scale}): WSD branches {c['wsd_steps']} steps, separate cosine runs "
              f"{c['cosine_steps']} steps, saving {c['saving']:.0%}")
    except NotImplementedError:
        print("\nbranch_plan / branch_cost not implemented yet (TODO 2-3)")


if __name__ == "__main__":
    main()
