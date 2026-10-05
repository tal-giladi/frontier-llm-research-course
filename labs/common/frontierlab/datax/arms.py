"""Train, score and compare the arms of a Module 10 data ablation (used by the lesson 10.2–10.5 labs and the project).

An *arm* is a mixture spec; it is trained once per seed with ``frontierlab.datax.train`` (the mixture's own
seed set to the run seed, so seeds differ in data order and initialisation together, and two arms with the
same seed share the initial weights). Runs are idempotent: a finished run (its final validation row is in
``metrics.jsonl``) is not retrained, an interrupted one resumes exactly.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np

from frontierlab.datax import evaluate as ev
from frontierlab.datax.mixture import MixtureSpec
from frontierlab.metrics.jsonl import read_jsonl


def finished(run: Path, steps: int) -> bool:
    m = run / "metrics.jsonl"
    return (run / "checkpoint.pt").exists() and m.exists() and any(
        r.get("split") == "val" and r.get("step") == steps for r in read_jsonl(m))


def train_arm(run: Path, spec: MixtureSpec, *, preset: str, steps: int, batch: int, seq: int, lr: float,
              warmup: int, seed: int, device: str = "cpu", extra: list[str] | None = None, question: str = "",
              eval_windows: int = 32, eval_every: int | None = None) -> Path:
    run = Path(run)
    if finished(run, steps):
        return run
    run.mkdir(parents=True, exist_ok=True)
    s = copy.deepcopy(spec)
    s.seed = seed
    s.save(run / "mixture.json")
    args = ["--mixture", str(run / "mixture.json"), *(extra or []), "--run", str(run), "--preset", preset,
            "--steps", str(steps), "--batch", str(batch), "--seq", str(seq), "--lr", str(lr), "--warmup", str(warmup),
            "--seed", str(seed), "--eval-every", str(eval_every or steps), "--eval-windows", str(eval_windows), "--log-every", "50",
            "--ckpt-every", str(max(50, steps // 4)), "--device", device, "--question", question]
    if device == "cuda":
        args += ["--dtype", "bf16"]
    from frontierlab.datax import train as dtrain
    t0 = time.time()
    dtrain.main(args)
    with open(run / "wall_seconds.txt", "a") as f:
        f.write(f"{time.time() - t0:.1f}\n")
    return run


def score(run: Path, sets: dict, *, windows: int, seq: int, n_lambada: int, device: str = "cpu",
          name: str = "eval_m10.json") -> dict:
    path = Path(run) / name
    if path.exists():
        res = json.loads(path.read_text())
        if set(res["sets"]) == set(sets) and (("lambada" in res) == bool(n_lambada)):
            return res
    res = ev.score_run(run, sets, windows, seq, n_lambada, device)
    path.write_text(json.dumps(res))
    return res


def table(results: dict, baseline: str, metrics: list[str], seeds: list[int], decide=None) -> list[dict]:
    """``results[arm][seed]`` are score() dicts. For each arm vs ``baseline`` and each metric: per-seed means,
    the seed-level paired difference (arm − baseline) with a 95% t-interval, and the decision if given."""
    rows = []
    for arm in results:
        for m in metrics:
            a = [ev.mean_of(results[baseline][s], m) for s in seeds]
            b = [ev.mean_of(results[arm][s], m) for s in seeds]
            row = {"arm": arm, "metric": m, "mean": float(np.mean(b)), "per_seed": b}
            if arm != baseline:
                sl = ev.seed_level(a, b)
                row.update(diff=sl["mean_diff"], ci=sl["ci"])
                if decide is not None:
                    row["decision"] = decide(sl["ci"]) if m != "lambada" else decide((-sl["ci"][1], -sl["ci"][0]))
            rows.append(row)
    return rows


def print_table(rows: list[dict]) -> str:
    lines = [f"{'arm':14s} {'metric':8s} {'mean':>9s} {'diff vs baseline':>17s} {'95% CI (seeds)':>22s}  decision"]
    for r in rows:
        if "diff" in r:
            lines.append(f"{r['arm']:14s} {r['metric']:8s} {r['mean']:9.4f} {r['diff']:+17.4f} "
                         f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]  {r.get('decision', '')}")
        else:
            lines.append(f"{r['arm']:14s} {r['metric']:8s} {r['mean']:9.4f} {'(baseline)':>17s}")
    out = "\n".join(lines)
    print(out)
    return out


def seed_std(results: dict, arm: str, metric: str, seeds: list[int]) -> float:
    return float(np.std([ev.mean_of(results[arm][s], metric) for s in seeds], ddof=1))
