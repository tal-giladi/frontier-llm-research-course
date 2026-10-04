"""Module 7 project, debugging task: a colleague's report stage. It contains two bugs; find and fix them.

    python labs/module-07/project/buggy_report.py runs/m07/project/cpu

It reads the same runs as ``run_project.py --stage report`` and prints a confident conclusion.
Do not trust the conclusion. Trust the run cards.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m07  # noqa: E402
from frontierlab.stats import paired_bootstrap  # noqa: E402


def pick(root: Path, opt: str, width: int, seed: int = 0):
    """'The' run of an optimizer at a width."""
    runs = sorted(p for p in root.iterdir() if p.name.startswith(f"{opt}-w{width}-") and p.name.endswith(f"-s{seed}")
                  and (p / "checkpoint.pt").exists())
    return runs[0]


def best(root: Path, opt: str, width: int):
    runs = [p for p in root.iterdir() if p.name.startswith(f"{opt}-w{width}-") and p.name.endswith("-s0")
            and (p / "checkpoint.pt").exists()]
    return min(runs, key=lambda p: m07.mean(m07.eval_losses(p)))


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/m07/project/cpu")
    widths = sorted({int(p.name.split("-w")[1].split("-")[0]) for p in root.iterdir() if "-w" in p.name})
    large = widths[-1]
    muon = best(root, "muon", large)
    adamw = pick(root, "adamw", large)
    d = paired_bootstrap(m07.eval_losses(muon), m07.eval_losses(adamw))
    print(f"equal tokens:     Muon - AdamW = {d['mean_diff']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]")
    wc = pick(root, "adamw", large)                      # the wall-clock arm
    t_m, t_a = m07.wallclock(muon), m07.wallclock(wc)
    d2 = paired_bootstrap(m07.eval_losses(muon), m07.eval_losses(wc))
    print(f"equal wall-clock: Muon - AdamW = {d2['mean_diff']:+.4f} [{d2['ci'][0]:+.4f}, {d2['ci'][1]:+.4f}] "
          f"(Muon {t_m:.0f} s, AdamW {t_a:.0f} s)")
    if d["ci"][1] < -0.02 and d2["ci"][1] < -0.02:
        print("conclusion: Muon wins by a wide margin at equal tokens and at equal wall-clock; the overhead does not matter.")
    else:
        print("conclusion: no clear winner.")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
