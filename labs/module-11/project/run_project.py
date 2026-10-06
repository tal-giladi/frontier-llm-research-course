"""Module 11 project: the Recipe-R run. Ladder -> pre-registered prediction -> the run -> the check.

    python labs/module-11/project/run_project.py ladder  --recipe my_recipe_r.json            # fixed-ratio ladder
    python labs/module-11/project/run_project.py predict --recipe my_recipe_r.json            # writes prereg.json once
    python labs/module-11/project/run_project.py run     --recipe my_recipe_r.json            # the target run
    python labs/module-11/project/run_project.py check   --recipe my_recipe_r.json            # prediction vs result
    python labs/module-11/project/run_project.py ladder  --recipe my_recipe_r.json --variant main --print

Every run uses the recipe's wrapper and arguments (``"via"`` and ``"args"`` in the recipe file), so the ladder and the
target differ only in size and tokens. The ladder is OLMo-ladder style (arXiv 2412.04403): every rung at several
token-per-parameter ratios. The prediction has two parts, both pre-registered: the target's final validation loss
(parametric fit, bootstrap interval, the predicted curve at 10% steps) and its accuracy on the lesson 11.2 cloze task
(two-step: task NLL from L(N, D), accuracy from a sigmoid of task NLL). Runs live in ``runs/m11/project/<variant>/``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

from frontierlab.data.loader import TokenData  # noqa: E402
from frontierlab.scaling import derisk, ladder  # noqa: E402
from frontierlab.scaling import downstream as ds  # noqa: E402
from frontierlab.scaling import fit as sfit  # noqa: E402

DATA_CPU = str(common.REPO / "labs" / "common" / "data" / "m11-v1024")
VARIANTS = {
    "cpu": dict(data=DATA_CPU, rungs=["m11-r1", "m11-r2", "m11-r3", "m11-r4"], ratios=[5, 20], target="m11-r5", target_ratio=10,
                seq=128, batch=16, windows=64, device="cpu", vocab=1024, items=500),
    "t4": dict(data=DATA_CPU, rungs=["m11-r3", "m11-r4", "m11-r5", "m11-r6"], ratios=[5, 20], target="m11-r7", target_ratio=10,
               seq=256, batch=32, windows=128, device="cuda", vocab=1024, items=500),
    "main": dict(data="labs/common/data/data-v1-main", rungs=["pilot-10m", "pilot-30m", "pilot-70m", "baseline0"], ratios=[5, 20, 60],
                 target="m11-350m", target_ratio=60, seq=1024, batch=32, windows=256, device="cuda", dtype="bf16", vocab=32768, items=1000),
}
FRACTIONS = [round(0.1 * i, 1) for i in range(1, 11)]


def out(variant):
    return common.RUNS / "project" / variant


def load_recipe(path):
    r = json.loads(Path(path).read_text())
    for k in ("via", "args", "lr"):
        if k not in r:
            raise SystemExit(f"recipe {path} needs '{k}'")
    return r


def plan(v):
    rows = ladder.plan_fixed_ratio(v["rungs"], v["ratios"], seq=v["seq"], batch=v["batch"], vocab_size=v["vocab"])
    return [dict(p, name=f"{p['preset']}-x{p['ratio']}") for p in rows]


def target(v):
    p = ladder.plan_fixed_ratio([v["target"]], [v["target_ratio"]], seq=v["seq"], batch=v["batch"], vocab_size=v["vocab"])[0]
    return dict(p, name=f"target-{v['target']}-x{v['target_ratio']}")


def args_for(p, v, recipe, run):
    via = ["--via", recipe["via"]] + list(recipe["args"])
    return via, common.run_args(p, v, run, lr=recipe["lr"], extra=["--data", v["data"], "--question", f"Recipe-R:{recipe['name']}"])


def train(p, v, recipe, a):
    run = out(a.variant) / p["name"]
    via, args = args_for(p, v, recipe, run)
    return common.ensure_run(run, args, via=via, print_only=a.print)


def cmd_ladder(a, v, recipe):
    rows = []
    for p in plan(v):
        r = train(p, v, recipe, a)
        if r is not None:
            rows.append({**r, "ratio": p["ratio"], "preset": p["preset"]})
    if a.print:
        t = target(v)
        print(f"# target: {t['preset']} at {t['ratio']} tokens/param, {t['tokens']:.3g} tokens, {t['C']:.3g} FLOPs, "
              f"PROJECTED {ladder.projected_gpu_hours(t['C']):.1f} H100-hours at 30% MFU; ladder "
              f"{sum(p['C'] for p in plan(v)):.3g} FLOPs, {ladder.projected_gpu_hours(sum(p['C'] for p in plan(v))):.1f} H100-hours")
        return
    print(f"\n{'run':>16} {'N_total':>9} {'tokens':>10} {'C':>9} {'loss':>7}")
    for r in rows:
        print(f"{r['run']:>16} {r['N_total']:9d} {r['D']:10.3g} {r['C']:9.3g} {r['loss']:7.4f}")
    ladder.save_json(rows, out(a.variant) / "ladder.json")


def score(run_dir, items, device):
    import torch
    from frontierlab.model import ModelConfig
    from frontierlab.optim.train import load_model
    m = load_model(run_dir / "checkpoint.pt").to(device).eval()
    with torch.no_grad():
        return ds.score_cloze(m, items, device=device)


def cmd_predict(a, v, recipe):
    rows = json.loads((out(a.variant) / "ladder.json").read_text())
    N = np.array([r["N_total"] for r in rows], float)
    D = np.array([r["D"] for r in rows], float)
    L = np.array([r["loss"] for r in rows])
    full = sfit.fit_parametric(N, D, L)
    boots = sfit.bootstrap(N, D, L, n_boot=300, seed=0)
    t = target(v)
    mid, lo, hi = sfit.prediction_interval(boots, t["N_total"], t["tokens"], 0.9)
    point = float(sfit.predict(full, t["N_total"], t["tokens"]))
    band = derisk.curve_band(rows, t["N_total"], t["tokens"], FRACTIONS, full["alpha"], full["beta"], n_boot=300)
    # leave-one-rung-out: how well does the ladder predict its own largest rung?
    big = N == N.max()
    held = sfit.holdout(N, D, L, big)
    print(f"Ladder fit ({len(rows)} runs): E {full['E']:.3f}, alpha {full['alpha']:.3f}, beta {full['beta']:.3f}; "
          f"largest rung held out: mean |error| {held['mae']:.4f}")
    print(f"Target {t['preset']}: N = {t['N_total']:.3g}, D = {t['tokens']:.3g} ({t['ratio']} tokens/param), C = {t['C']:.3g}")
    print(f"  predicted final validation loss {point:.4f}, 90% interval [{lo:.4f}, {hi:.4f}]")
    items = ds.build_cloze(TokenData("test", v["data"]), n_items=v["items"], ctx=48, cont=8, seed=0)
    sc = []
    for r in rows:
        s = score(out(a.variant) / r["run"], items, v["device"])
        sc.append({"run": r["run"], "acc": s["acc"], "nll": s["nll_correct"]})
    nll_fit = sfit.fit_parametric(N, D, [s["nll"] for s in sc])
    sig = ds.fit_sigmoid([s["nll"] for s in sc], [s["acc"] for s in sc], lo=0.25, hi=1.0)
    nll_t = float(sfit.predict(nll_fit, t["N_total"], t["tokens"]))
    nb = sfit.bootstrap(N, D, [s["nll"] for s in sc], n_boot=200, seed=1)
    accs = [float(ds.predict_sigmoid(sig, sfit.predict(f, t["N_total"], t["tokens"]))) for f in nb]
    acc_lo, acc_hi = np.quantile(accs, [0.05, 0.95])
    acc_t = float(ds.predict_sigmoid(sig, nll_t))
    print(f"  predicted cloze accuracy {acc_t:.3f} [{acc_lo:.3f}, {acc_hi:.3f}] (task NLL {nll_t:.4f}; step 2 uncertainty not included)")
    rec = {"recipe": recipe, "target": {k: t[k] for k in ("preset", "ratio", "steps", "tokens", "C", "N_total")},
           "prediction": {"loss": point, "lo": lo, "hi": hi, "level": 0.9},
           "downstream": {"task": f"cloze, {v['items']} test items, 4 options", "acc": acc_t, "lo": float(acc_lo), "hi": float(acc_hi),
                          "nll_correct": nll_t, "sigmoid": sig},
           "band": band, "fit": {k: full[k] for k in ("E", "A", "B", "alpha", "beta")}, "heldout_mae": held["mae"],
           "ladder_digest": derisk.ladder_digest(rows),
           "rules": {"success": "final loss inside the 90% interval", "off_band_tol": 0.02, "spike_factor": 4.0}}
    path = out(a.variant) / "prereg.json"
    try:
        derisk.preregister(path, rec)
        print(f"pre-registered: {path}")
    except FileExistsError:
        print(f"{path} exists (pre-registered earlier); not overwritten")


def cmd_run(a, v, recipe):
    train(target(v), v, recipe, a)


def cmd_check(a, v, recipe):
    from frontierlab.metrics.jsonl import read_jsonl
    t = target(v)
    run = out(a.variant) / t["name"]
    body = derisk.load_prereg(out(a.variant) / "prereg.json")
    r = ladder.read_run(run)
    c = derisk.check_after(out(a.variant) / "prereg.json", r, run)
    print(f"Loss: predicted {c['predicted']:.4f} [{c['interval'][0]:.4f}, {c['interval'][1]:.4f}], measured {c['measured']:.4f} "
          f"({c['error']:+.4f}, {c['where']}); prediction older than the run: {c['timing_ok']}")
    items = ds.build_cloze(TokenData("test", v["data"]), n_items=v["items"], ctx=48, cont=8, seed=0)
    s = score(run, items, v["device"])
    d = body["downstream"]
    inside = d["lo"] <= s["acc"] <= d["hi"]
    print(f"Cloze accuracy: predicted {d['acc']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}], measured {s['acc']:.3f} "
          f"({'inside' if inside else 'outside'}); task NLL predicted {d['nll_correct']:.4f}, measured {s['nll_correct']:.4f}")
    rows = read_jsonl(run / "metrics.jsonl")
    curve = sorted({int(x["step"]): float(x["loss"]) for x in rows if x.get("split") == "val"}.items())
    gn = [(int(x["step"]), float(x["grad_norm"])) for x in rows if x.get("split") == "train"]
    mon = derisk.monitor(curve, t["steps"], body["band"], tol=0.02, grad_norms=gn)
    print(f"Monitor over the whole run: {mon['decision']}" + (f" (would have stopped at step {mon['stop_step']})" if mon["stop_step"] else ""))
    for b in body["band"]:
        m = derisk.loss_at_fraction(curve, t["steps"], b["fraction"])
        print(f"  {b['fraction']:.1f}: measured {m:.4f}  band [{b['lo']:.4f}, {b['hi']:.4f}]")
    ladder.save_json({**c, "downstream_measured": s["acc"], "downstream_inside": inside, "monitor": mon}, out(a.variant) / "check.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["ladder", "predict", "run", "check"])
    ap.add_argument("--recipe", type=Path, default=HERE / "recipe_r_example.json")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cpu")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    v, recipe = VARIANTS[a.variant], load_recipe(a.recipe)
    {"ladder": cmd_ladder, "predict": cmd_predict, "run": cmd_run, "check": cmd_check}[a.cmd](a, v, recipe)


if __name__ == "__main__":
    main()
