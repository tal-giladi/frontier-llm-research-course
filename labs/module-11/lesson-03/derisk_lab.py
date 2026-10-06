"""Lab 11.3: de-risk a run 3x beyond the ladder: transfer check, pre-registered prediction, monitored run, go/no-go.

    python labs/module-11/lesson-03/derisk_lab.py transfer     # learning-rate sweep at two ladder sizes (4 runs, ~8 min CPU)
    python labs/module-11/lesson-03/derisk_lab.py predict      # fit the 11.1 ladder, choose the target, pre-register
    python labs/module-11/lesson-03/derisk_lab.py run          # train the target (~15 min CPU), then:
    python labs/module-11/lesson-03/derisk_lab.py check        # prediction vs measurement, monitor, launch checklist
    python labs/module-11/lesson-03/derisk_lab.py bad          # the same target with a planted data-loader bug, monitored

Needs the 11.1 iso-FLOP ladder (``ladder_lab.py isoflop``). ``predict`` writes ``runs/m11/l113/<variant>/prereg.json``
once (it refuses to overwrite: delete the folder to start over, and say so in your write-up).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from importlib import util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

from frontierlab.scaling import derisk, ladder  # noqa: E402
from frontierlab.scaling import fit as sfit  # noqa: E402

spec = util.spec_from_file_location("ladder_lab", HERE.parent / "lesson-01" / "ladder_lab.py")
ll = util.module_from_spec(spec)
spec.loader.exec_module(ll)

TARGETS = {"cpu": dict(budget=3e13, candidates=["m11-r4", "m11-r5", "m11-r6", "m11-r7"], sweep_budget=3e12,
                       sweep_rungs=["m11-r2", "m11-r4"], sweep_lrs=[1e-3, 3e-3, 1e-2], bad_unique=65536),
           "t4": dict(budget=3e14, candidates=["m11-r6", "m11-r7"], sweep_budget=3e13, sweep_rungs=["m11-r4", "m11-r6"],
                      sweep_lrs=[1e-3, 3e-3, 1e-2], bad_unique=262144),
           "main": dict(budget=3e18, candidates=["pilot-70m", "baseline0", "m11-200m", "m11-350m"], sweep_budget=3e17,
                        sweep_rungs=["pilot-30m", "baseline0"], sweep_lrs=[1e-3, 3e-3, 1e-2], bad_unique=50_000_000)}
FRACTIONS = [round(0.1 * i, 1) for i in range(1, 11)]
TOL = 0.02          # nats above the band's upper edge before the off-band rule fires
MAX_WIDTH = 0.15    # widest acceptable 90% interval at the target, nats
MAX_HELDOUT = 0.05  # largest acceptable held-out error of the ladder fit, nats


def out(variant):
    return common.RUNS / "l113" / variant


def ladder_runs(variant):
    p = ll.out_dir(variant) / "results.json"
    if not p.exists():
        raise SystemExit(f"{p} missing: run `ladder_lab.py isoflop --variant {variant}` first")
    return json.loads(p.read_text())


def target_plan(v, t, preset):
    return ladder.plan_isoflop([t["budget"]], [preset], seq=v["seq"], batch=v["batch"], vocab_size=v["vocab"], min_steps=1)[0]


def cmd_transfer(lab, a, v, t):
    rows = {}
    for p in ladder.plan_isoflop([t["sweep_budget"]], t["sweep_rungs"], seq=v["seq"], batch=v["batch"], vocab_size=v["vocab"]):
        for lr in t["sweep_lrs"]:
            if abs(lr - v["lr"]) < 1e-12:
                run = ll.out_dir(a.variant) / f"{p['preset']}-c{p['budget']:.0e}"      # the ladder's own run
            else:
                run = out(a.variant) / f"lr-{p['preset']}-c{p['budget']:.0e}-lr{lr:g}"
            args = common.run_args(p, v, run, lr=lr, extra=["--data", v["data"], "--question", "LR transfer (11.3)"])
            r = common.ensure_run(run, args, print_only=a.print)
            if r is not None:
                rows.setdefault(p["preset"], {})[lr] = r["loss"]
    if a.print:
        return
    init = math.log(v["vocab"])                       # loss of a uniform prediction, the l0 of Wortsman et al.
    print(f"\nLearning-rate sweep at C = {t['sweep_budget']:.0e} (validation loss):")
    print(f"{'rung':>8} " + " ".join(f"{lr:>8g}" for lr in t["sweep_lrs"]) + f" {'best':>8} {'sensitivity':>11}")
    res = {}
    for k, d in rows.items():
        best = min(d, key=d.get)
        sens = lab.lr_sensitivity(d, init)
        res[k] = {"losses": {str(x): y for x, y in d.items()}, "best": best, "sensitivity": sens}
        print(f"{k:>8} " + " ".join(f"{d[lr]:8.4f}" for lr in t["sweep_lrs"]) + f" {best:8g} {sens:11.4f}")
    same = len({r["best"] for r in res.values()}) == 1
    res["same_best"] = same
    print(f"same best learning rate at both sizes: {same}")
    ladder.save_json(res, out(a.variant) / "transfer.json")


def cmd_predict(lab, a, v, t):
    L = ladder_runs(a.variant)
    runs = [r for r in L["runs"] if r["D"] / r["N_total"] >= L["min_tpp"]]     # 11.1's rule for approach 3
    N = np.array([r["N_total"] for r in runs], float)
    D = np.array([r["D"] for r in runs], float)
    y = np.array([r["loss"] for r in runs])
    full = sfit.fit_parametric(N, D, y)
    boots = sfit.bootstrap(N, D, y, n_boot=300, seed=0)
    print(f"Ladder: {len(runs)} of {len(L['runs'])} runs (>= {L['min_tpp']:g} tokens/param); fit E = {full['E']:.3f}, alpha = {full['alpha']:.3f}, beta = {full['beta']:.3f}; "
          f"{len(boots)} bootstrap refits")
    print(f"\nCandidates at C = {t['budget']:.0e} (tokens from the exact FLOPs per token):")
    cands = []
    for preset in t["candidates"]:
        p = target_plan(v, t, preset)
        pts = [float(sfit.predict(f, p["N_total"], p["tokens"])) for f in boots]
        lo, hi = lab.interval(pts, 0.9)
        cands.append({**p, "pred": float(sfit.predict(full, p["N_total"], p["tokens"])), "lo": lo, "hi": hi})
        print(f"  {preset:>9}: N = {p['N_total']:.3g}, D = {p['tokens']:.3g} ({p['tokens'] / p['N_total']:.0f} tok/param), "
              f"predicted {cands[-1]['pred']:.4f} [{lo:.4f}, {hi:.4f}]")
    tgt = min(cands, key=lambda c: c["pred"])
    print(f"chosen: {tgt['preset']} (lowest predicted loss)")
    band = derisk.curve_band(runs, tgt["N_total"], tgt["tokens"], FRACTIONS, full["alpha"], full["beta"], n_boot=300)
    held = L["holdout"]
    minima_ok = all(not m["edge"] for m in L[f"approach2_N_total_w{L['window']}"]["minima"])
    tr = out(a.variant) / "transfer.json"
    transfer = json.loads(tr.read_text()) if tr.exists() else None
    ladder_C = sum(r["C"] for r in L["runs"])
    checks = {
        "held-out error of the ladder fit": (held["mae"] <= MAX_HELDOUT, f"{held['mae']:.4f} nats (max {MAX_HELDOUT})"),
        "interior iso-FLOP minima": (minima_ok, "every budget's vertex inside its sizes" if minima_ok else "an edge minimum"),
        "learning rate transfers": (bool(transfer and transfer["same_best"]),
                                    "same best rate at both sizes" if transfer and transfer["same_best"] else
                                    ("best rate moves with size" if transfer else "transfer not run")),
        "interval width": (tgt["hi"] - tgt["lo"] <= MAX_WIDTH, f"{tgt['hi'] - tgt['lo']:.4f} nats (max {MAX_WIDTH})"),
        "ladder compute share": (ladder_C >= 0.01 * t["budget"], f"ladder {ladder_C:.3g} FLOPs = {ladder_C / t['budget']:.1%} of the target"),
        "extrapolation distance": (t["budget"] <= 10 * max(r["budget"] for r in L["runs"]), f"{t['budget'] / max(r['budget'] for r in L['runs']):.0f}x the largest ladder budget"),
    }
    decision = lab.go_no_go(checks)
    print("\nLaunch checklist:")
    for k, (ok, d) in checks.items():
        print(f"  [{'x' if ok else ' '}] {k}: {d}")
    print(f"decision: {decision}")
    rec = {"target": {k: tgt[k] for k in ("preset", "budget", "steps", "tokens", "C", "N_total", "N_nonemb")},
           "prediction": {"loss": tgt["pred"], "lo": tgt["lo"], "hi": tgt["hi"], "level": 0.9},
           "band": band, "fit": {k: full[k] for k in ("E", "A", "B", "alpha", "beta")},
           "ladder_digest": derisk.ladder_digest(runs), "checks": {k: [ok, d] for k, (ok, d) in checks.items()},
           "decision": decision,
           "rules": {"off_band_tol": TOL, "min_fraction": 0.1, "spike_factor": 4.0,
                     "success": "measured final loss inside the 90% interval",
                     "stop": "validation loss above the band's upper edge by more than off_band_tol at >= 10% of the run, "
                             "or a gradient norm above 4x the running median after 10% of the run"}}
    path = out(a.variant) / "prereg.json"
    try:
        derisk.preregister(path, rec)
        print(f"\npre-registered: {path}")
    except FileExistsError:
        print(f"\n{path} exists (pre-registered earlier); not overwritten")


def target_run(a, v, t, bad=False):
    body = derisk.load_prereg(out(a.variant) / "prereg.json")
    p = {"preset": body["target"]["preset"], "steps": body["target"]["steps"]}
    name = f"target-{p['preset']}" + ("-bad" if bad else "")
    run = out(a.variant) / name
    args = common.run_args(p, v, run, extra=["--data", v["data"], "--question", "pre-registered target (11.3)"])
    if bad:
        args = ["--unique-tokens", str(t["bad_unique"])] + args + ["--stop-after", str(int(0.5 * p["steps"]))]
    return body, run, args


def cmd_run(lab, a, v, t):
    body, run, args = target_run(a, v, t)
    common.ensure_run(run, args, print_only=a.print)


def report(lab, body, run_dir, steps, partial=False):
    from frontierlab.metrics.jsonl import read_jsonl
    rows = read_jsonl(run_dir / "metrics.jsonl")
    last = {}
    for r in rows:
        if r.get("split") == "val":
            last[int(r["step"])] = float(r["loss"])
    curve = sorted(last.items())
    gn = [(int(r["step"]), float(r["grad_norm"])) for r in rows if r.get("split") == "train"]
    print(f"{'fraction':>8} {'step':>6} {'measured':>9} {'band lo':>8} {'mid':>8} {'hi':>8}")
    for b in body["band"]:
        m = lab.loss_at_fraction(curve, steps, b["fraction"])
        if not math.isnan(m):
            flag = "  <- above band" if m > b["hi"] + TOL else ""
            print(f"{b['fraction']:8.1f} {int(b['fraction'] * steps):6d} {m:9.4f} {b['lo']:8.4f} {b['mid']:8.4f} {b['hi']:8.4f}{flag}")
    mine = lab.first_off_band(curve, steps, body["band"], TOL)
    mon = derisk.monitor(curve, steps, body["band"], tol=TOL, grad_norms=gn)
    print(f"off-band rule (yours): first step {mine}; monitor: {mon['decision']}"
          + (f" at step {mon['stop_step']}, saving {mon['compute_saved']:.0%} of the run's compute" if mon["stop_step"] else ""))
    if mon["grad_spike"]:
        print(f"gradient-spike rule fired: {mon['grad_spike']}")
    return curve, mon


def cmd_check(lab, a, v, t):
    body, run, _ = target_run(a, v, t)
    r = ladder.read_run(run)
    c = derisk.check_after(out(a.variant) / "prereg.json", r, run)
    print(f"Target {body['target']['preset']} at C = {body['target']['C']:.3g}: predicted {c['predicted']:.4f} "
          f"[{c['interval'][0]:.4f}, {c['interval'][1]:.4f}] (90%), measured {c['measured']:.4f}, error {c['error']:+.4f}: {c['where']}; "
          f"prediction written before the run: {c['timing_ok']}")
    print(f"pre-launch decision recorded in the pre-registration: {body['decision']}\n")
    report(lab, body, run, body["target"]["steps"])
    ladder.save_json(c, out(a.variant) / "check.json")


def cmd_bad(lab, a, v, t):
    body, run, args = target_run(a, v, t, bad=True)
    if a.print:
        common.ensure_run(run, args, print_only=True)
        return
    if not (run / "metrics.jsonl").exists() or not (run / "checkpoint.pt").exists():
        common._quiet(lambda: common.strain.main(args))
    print(f"Planted bug: the data loader reads only the first {t['bad_unique']:,} training tokens "
          f"(stopped at 50% of the run):")
    report(lab, body, run, body["target"]["steps"], partial=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["transfer", "predict", "run", "check", "bad"])
    ap.add_argument("--variant", choices=sorted(TARGETS), default="cpu")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--lab", default=None)
    a = ap.parse_args(argv)
    lab = common.lab_module(a.lab, HERE)
    v, t = ll.VARIANTS[a.variant], TARGETS[a.variant]
    {"transfer": cmd_transfer, "predict": cmd_predict, "run": cmd_run, "check": cmd_check, "bad": cmd_bad}[a.cmd](lab, a, v, t)


if __name__ == "__main__":
    main()
