"""Module 10 project: build Data-v1 from a recipe file, write its provenance manifest and integrity report, and
compare it with Data-v0 at equal tokens.

    python labs/module-10/project/run_project.py --recipe labs/module-10/project/data_v1_example.json      # free CPU, ~45 min
    python labs/module-10/project/run_project.py --recipe my_data_v1.json --variant main --print           # main-path commands

A recipe is JSON:

    {"name": "data-v1",
     "sources": [{"name": "edu", "root": "labs/common/data/v0", "weight": 0.5, "evidence": "10.4 regmix ..."}, ...],
     "doc_mask": false, "doc_mask_evidence": "10.1 ...",
     "anneal": {"candidates": [...], "share": 0.5, "evidence": "10.4 micro-anneals ..."}   (optional, reported only)
    }

Every choice carries an ``evidence`` string pointing at the ablation (lesson, run folder, interval) behind it.
The script:

1. checks the recipe: weights positive, every source has evidence, every root exists;
2. writes ``<out>/manifest.json`` (``frontierlab.datax.provenance.build_manifest``) and stops if any source is
   BLOCKED;
3. writes ``<out>/integrity.json`` (``frontierlab.datax.integrity``: exact and near-duplicates between every
   training source and Data-v0's held-out splits, n-gram leakage of Eval v0 windows, Eval v1 haystacks and
   LAMBADA, the planted control);
4. trains Data-v0 and Data-v1 with 3 seeds at equal tokens (``frontierlab.datax.arms``) and scores both on each
   source's validation split and LAMBADA;
5. prints the seed-level comparison and the decision: adopt Data-v1 if the 95% interval of (v1 − v0) on the mean
   of the held-out losses is below 0 and its interval on Eval v0's held-out set (Data-v0 validation) has an upper
   bound of at most +0.02 nats; reject if the first interval is above 0 or the second's lower bound exceeds 0.02;
   else inconclusive.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from frontierlab.data.prepare import DEFAULT_OUT
from frontierlab.datax import arms, integrity, provenance
from frontierlab.datax import evaluate as ev
from frontierlab.datax.mixture import MixtureSpec, SourceRef

VARIANTS = {
    "cpu": dict(preset="toy", steps=600, batch=16, seq=128, lr=3e-3, warmup=50, seeds=[0, 1, 2], device="cpu",
                windows=256, lambada=1000),
    "t4": dict(preset="pilot-10m", steps=4000, batch=32, seq=512, lr=3e-3, warmup=200, seeds=[0, 1, 2], device="cuda",
               windows=256, lambada=5153),
    "main": dict(preset="baseline0", steps=9500, batch=32, seq=1024, lr=None, warmup=200, seeds=[0, 1, 2], device="cuda",
                 windows=256, lambada=5153),
}


def load_recipe(path: Path) -> dict:
    r = json.loads(path.read_text())
    problems = []
    for s in r["sources"]:
        if s["weight"] <= 0:
            problems.append(f"{s['name']}: weight must be positive")
        if not s.get("evidence"):
            problems.append(f"{s['name']}: no evidence for this source and weight")
        if not (Path(s["root"]) / "meta.json").exists():
            problems.append(f"{s['name']}: {s['root']} is not a prepared source")
    if "doc_mask" in r and not r.get("doc_mask_evidence"):
        problems.append("doc_mask: no evidence")
    if problems:
        raise SystemExit("recipe problems:\n  " + "\n  ".join(problems))
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", type=Path, default=Path(__file__).resolve().parent / "data_v1_example.json")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cpu")
    ap.add_argument("--out", type=Path, default=Path("runs/m10/project"))
    ap.add_argument("--lr", type=float, default=None, help="main path: your Module 1 tuned learning rate")
    ap.add_argument("--print", action="store_true", help="print the training commands and stop")
    ap.add_argument("--skip-integrity", action="store_true")
    a = ap.parse_args(argv)
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    v = dict(VARIANTS[a.variant])
    if a.lr:
        v["lr"] = a.lr
    if v["lr"] is None:
        raise SystemExit("main path: pass --lr (the learning rate your Module 1 sweep chose)")
    recipe = load_recipe(a.recipe)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    v1 = MixtureSpec([SourceRef(s["name"], s["root"], float(s["weight"]), s.get("split", "train")) for s in recipe["sources"]],
                     name=recipe.get("name", "data-v1"))
    v0 = MixtureSpec([SourceRef("edu", str(DEFAULT_OUT), 1.0)], name="data-v0")
    extra = ["--doc-mask"] if recipe.get("doc_mask") else []
    if a.print:
        for name, spec, ex in (("data-v0", v0, []), ("data-v1", v1, extra)):
            for s in v["seeds"]:
                print(f"python -m frontierlab.datax.train --mixture {a.out}/{name}-s{s}/mixture.json {' '.join(ex)} "
                      f"--run {a.out}/{name}-s{s} --preset {v['preset']} --steps {v['steps']} --batch {v['batch']} "
                      f"--seq {v['seq']} --lr {v['lr']} --warmup {v['warmup']} --seed {s} --dtype bf16")
        return
    man = provenance.build_manifest(v1)
    (a.out / "manifest.json").write_text(json.dumps(man, indent=2))
    print(f"manifest: {len(man['sources'])} sources, blocked {len(man['blocked'])}, warnings {len(man['warnings'])}, "
          f"share-alike {man['share_alike_sources']}")
    for w in man["warnings"] + man["blocked"]:
        print("  ", w)
    if man["blocked"]:
        raise SystemExit("a source is BLOCKED; fix the record before training")
    ip = a.out / "integrity.json"
    if not a.skip_integrity and not ip.exists():
        rep = integrity.report({s.name: s.root for s in v1.sources}, DEFAULT_OUT,
                               eval_v1=DEFAULT_OUT.parent / "v0-long", splits={s.name: s.split for s in v1.sources})
        ip.write_text(json.dumps(rep, indent=2))
    if ip.exists():
        print(integrity.summary(json.loads(ip.read_text())))
    sets = {s.name: s.root for s in v1.sources if (Path(s.root) / "val.bin").exists()
            and Path(s.root, "val.bin").stat().st_size > 2 * v["seq"] * 8}
    sets.setdefault("edu", str(DEFAULT_OUT))
    results = {}
    for name, spec, ex in (("data-v0", v0, []), ("data-v1", v1, extra)):
        results[name] = {}
        for s in v["seeds"]:
            run = arms.train_arm(a.out / f"{name}-s{s}", spec, preset=v["preset"], steps=v["steps"], batch=v["batch"],
                                 seq=v["seq"], lr=v["lr"], warmup=v["warmup"], seed=s, device=v["device"], extra=ex,
                                 question="Data-v1 vs Data-v0 at equal tokens (Module 10 project)")
            results[name][s] = arms.score(run, sets, windows=v["windows"], seq=v["seq"], n_lambada=v["lambada"],
                                          device=v["device"])
    rows = arms.table(results, "data-v0", ["avg", *sets, "lambada"], v["seeds"])
    arms.print_table(rows)
    seeds = v["seeds"]
    avg = ev.seed_level([ev.mean_of(results["data-v0"][s], "avg") for s in seeds],
                        [ev.mean_of(results["data-v1"][s], "avg") for s in seeds])
    guard = ev.seed_level([ev.mean_of(results["data-v0"][s], "edu") for s in seeds],
                          [ev.mean_of(results["data-v1"][s], "edu") for s in seeds])
    if avg["ci"][1] < 0 and guard["ci"][1] <= 0.02:
        decision = "adopt Data-v1"
    elif avg["ci"][0] > 0 or guard["ci"][0] > 0.02:
        decision = "reject Data-v1"
    else:
        decision = "inconclusive"
    sd = arms.seed_std(results, "data-v0", "avg", seeds)
    from frontierlab.stats import min_detectable_effect
    print(f"mean held-out loss (v1 − v0): {avg['mean_diff']:+.4f} [{avg['ci'][0]:+.4f}, {avg['ci'][1]:+.4f}]; "
          f"Eval v0 guard: {guard['mean_diff']:+.4f} [{guard['ci'][0]:+.4f}, {guard['ci'][1]:+.4f}] -> {decision}")
    print(f"seed std of Data-v0 on the mean held-out loss {sd:.4f}; unpaired MDE with 3 seeds "
          f"{min_detectable_effect(sd, 3):.4f}; sd of the paired differences {avg['sd_diff']:.4f}")
    acc = json.loads((a.out / f"data-v1-s{seeds[0]}" / "mixture_accounting.json").read_text())
    print("Data-v1 accounting (seed 0):", {k: (x["tokens"], round(x["epochs"], 3)) for k, x in acc["sources"].items()})
    (a.out / "report.json").write_text(json.dumps({"recipe": recipe, "rows": rows, "decision": decision,
                                                   "avg": avg, "guard": guard}, indent=2))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
