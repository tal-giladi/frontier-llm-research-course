"""Lab 03.3: probe the trained arms — first-token mass, learned-sink mass, max logit, massive
activations — and compare held-out loss against b0 with a paired interval over fixed windows.

    python labs/module-03/lesson-03/probe.py runs/m03/l33/cpu
    python labs/module-03/lesson-03/probe.py runs/m03/l33/main --device cuda

Probes run on 8 fixed validation windows of 128 tokens; loss on 256 fixed windows of the run's
sequence length (window seed 1234, as Eval v0). Your lab.py functions compute the statistics; the
reference versions in frontierlab.attention.probes are used if a TODO is not done yet.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m03  # noqa: E402
from frontierlab.attention import probes  # noqa: E402
from frontierlab.data.loader import TokenData  # noqa: E402
from frontierlab.evals.heldout import window_losses  # noqa: E402
from frontierlab.labkit import load_target  # noqa: E402
from frontierlab.stats import paired_bootstrap  # noqa: E402

lab = load_target(str(Path(__file__).parent / "test_lab.py"))


def stat(fn_name, *args, **kw):
    try:
        return getattr(lab, fn_name)(*args, **kw)
    except NotImplementedError:
        return getattr(probes, fn_name)(*args, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--skip", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    runs = sorted(p for p in a.root.iterdir() if (p / "checkpoint.pt").exists())
    val = TokenData("val")
    starts = val.eval_windows(8, 128, seed=1234)
    idx = torch.stack([val.window(s, 128) for s in starts]).to(a.device)
    rows, losses = [], {}
    for run in runs:
        model, card = m03.load_run(run, a.device)
        T = int(card["args"]["seq"])
        losses[run.name] = window_losses(model, val, a.windows, T, device=a.device)
        cap = probes.capture(model, idx)
        per = []
        for p, lg, h in zip(cap["probs"], cap["logits"], cap["hidden"]):
            per.append({"first": stat("first_token_mass", p, skip=a.skip),
                        "sink": probes.learned_sink_mass(p, skip=a.skip),
                        "max_logit": stat("max_logit", lg), "massive": stat("massive_ratio", h)})
        rows.append({"run": run.name, "lr": card["args"]["lr"], "val_loss": sum(losses[run.name]) / len(losses[run.name]),
                     "first_token_mass": sum(r["first"] for r in per) / len(per),
                     "first_token_mass_by_layer": [round(r["first"], 4) for r in per],
                     "sink_mass": sum(r["sink"] for r in per) / len(per),
                     "max_logit": max(r["max_logit"] for r in per),
                     "massive_ratio": max(r["massive"] for r in per)})
    uni = probes._uniform_reference(128, a.skip)
    print(f"{len(runs)} runs; probes on 8 windows x 128 tokens (queries t >= {a.skip}); "
          f"uniform attention would give first-token mass {uni:.3f}\n")
    print(f"{'run':<18s} {'lr':>8s} {'val loss':>9s} {'1st-token':>9s} {'sink':>6s} {'max logit':>9s} {'massive':>8s}  first-token mass by layer")
    for r in rows:
        print(f"{r['run']:<18s} {r['lr']:>8.4g} {r['val_loss']:9.4f} {r['first_token_mass']:9.3f} {r['sink_mass']:6.3f} "
              f"{r['max_logit']:9.2f} {r['massive_ratio']:8.1f}  {r['first_token_mass_by_layer']}")
    print("\nheld-out loss vs b0 at the same learning rate: paired bootstrap over "
          f"{a.windows} windows (one seed: evaluation noise only)")
    for n, ls in losses.items():
        arm, seed = n.rsplit("-s", 1)
        if arm.split("@")[0] == "b0":
            continue
        ref = ("b0@hi" if "@hi" in arm else "b0") + f"-s{seed}"
        if ref in losses:
            pb = paired_bootstrap(ls, losses[ref])
            print(f"  {n:<18s} vs {ref:<10s} {pb['mean_diff']:+.4f}  95% CI [{pb['ci'][0]:+.4f}, {pb['ci'][1]:+.4f}]")
    if a.out:
        a.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
