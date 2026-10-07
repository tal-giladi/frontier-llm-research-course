"""Lab 19.3: lint a write-up against its run cards, review it, and draw a figure that answers one question.

    python labs/module-19/lesson-03/review_lab.py                  # seconds
    python labs/module-19/lesson-03/review_lab.py --part figure    # the figure from your lesson 19.2 runs

Part review: the claims register of flawed/writeup.md and fixed/writeup.md checked against their EXAMPLE run cards,
first with your functions (seed_issues, checkpoint_issues, scope_issue, figure_issues), then with the full linter
``frontierlab.research.writeup``; then your REVIEW against the problems the linter can see. Anything the linter
cannot see (a baseline that was not re-tuned, a wrong unit in the prose) is for your review alone.

Part figure: final held-out loss against learning rate for the two arms of lesson 19.2 (mean over seeds, min-max band,
log x-axis), one claim in the title, the seeds in the caption; written to runs/m19/l193/figure.png with its spec
figure.yaml, and the spec linted. Needs runs/m19/l192/<variant>/results.json from repro_lab.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from frontierlab.labkit import load_path
from frontierlab.research import writeup as W

HERE = Path(__file__).resolve().parent
LAB = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
OUT = Path("runs/m19/l193")


def review() -> None:
    for which in ("flawed", "fixed"):
        md = (HERE / which / "writeup.md").read_text(encoding="utf-8")
        cards = W.load_cards(HERE / which / "cards")
        print(f"\n=== {which}/writeup.md")
        for claim in W.parse_claims(md):
            arms = {a: [cards[r] for r in rs if r in cards] for a, rs in claim["arms"].items()}
            print(f"[{claim['id']}] {claim['text']}")
            print(f"  your seed_issues:       {LAB.seed_issues(arms)}")
            print(f"  your checkpoint_issues: {LAB.checkpoint_issues(claim)}")
            print(f"  your scope_issue:       {LAB.scope_issue(claim['text'] + ' ' + str(claim.get('scope', '')))}")
        issues = W.lint_writeup(HERE / which / "writeup.md", HERE / which / "cards")
        print(f"  full linter: {len(issues)} issue(s)")
        for i in issues:
            print(f"    {i}")
        spec = yaml.safe_load((HERE / which / "figure.yaml").read_text(encoding="utf-8"))
        print(f"  figure, your codes: {sorted(LAB.figure_issues(spec))}; linter: "
              f"{sorted({i.code for i in W.lint_figure(spec)})}")
    print("\n=== your review of flawed/writeup.md")
    for r in LAB.REVIEW:
        print(f"- {r.get('problem')}: {r.get('where')} | {r.get('evidence')} | {r.get('request')}")
    planted = ("missing seeds", "unmatched budget", "cherry-picked checkpoint", "claim beyond evidence")
    missing = [p for p in planted if p not in {r.get("problem") for r in LAB.REVIEW}]
    print("planted problems not yet in your review: " + (", ".join(missing) if missing else "none"))


def figure(variant: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    res = json.loads((Path("runs/m19/l192") / variant / "results.json").read_text())
    losses = res["losses"]
    fig, ax = plt.subplots(figsize=(5.5, 3.6), dpi=150)
    spec_arms = {}
    styles = {"qknorm": ("QK-norm", "#2a6fbb", "o"), "noqk": ("no QK-norm", "#c4502b", "s")}
    for arm, (label, color, marker) in styles.items():
        seeds = sorted(losses[arm])
        lrs = sorted({float(lr) for s in seeds for lr in losses[arm][s]})
        m = np.array([[losses[arm][s][f"{lr:g}"]["loss"] for lr in lrs] for s in seeds])
        ax.fill_between(lrs, m.min(0), m.max(0), color=color, alpha=0.2, linewidth=0)
        ax.plot(lrs, m.mean(0), color=color, marker=marker, label=f"{label} (n = {len(seeds)} seeds)")
        spec_arms[arm] = {"n_seeds": len(seeds), "tokens_per_step": 2048, "x_end": max(lrs)}
    ax.set_xscale("log")
    ax.set_xticks(lrs, [f"{x:g}" for x in lrs])
    ax.minorticks_off()
    ax.set_xlabel("peak learning rate (equal tokens per run)")
    ax.set_ylabel("final held-out loss (nats)")
    claim = "Without QK-norm, held-out loss degrades faster as the learning rate rises (toy preset, 1.8M parameters)"
    ax.set_title("QK-norm reduces learning-rate sensitivity at toy scale", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figure.png")
    caption = ("Final held-out loss (128 validation windows) after 614,400 tokens per run, mean over n = 3 seeds per "
               "arm; the band is the min-max range over seeds.")
    spec = {"claim": claim, "x": "lr", "contract_axis": "tokens", "y": "val_loss",
            "uncertainty": "min-max range over seeds", "arms": spec_arms, "caption": caption}
    (OUT / "figure.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    issues = W.lint_figure(spec)
    print(f"wrote {OUT / 'figure.png'} and figure.yaml; figure linter: " + ("clean" if not issues else ""))
    for i in issues:
        print(f"  {i}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["review", "figure", "all"], default="review")
    ap.add_argument("--variant", default="cpu", help="which lesson 19.2 sweep to plot")
    a = ap.parse_args()
    if a.part in ("review", "all"):
        review()
    if a.part in ("figure", "all"):
        figure(a.variant)


if __name__ == "__main__":
    main()
