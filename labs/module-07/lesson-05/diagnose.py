"""Lab 07.5, steps 2, 4 and 5: read runs' logs and say what went wrong.

    python labs/module-07/lesson-05/diagnose.py labs/module-07/lesson-05/traces     # the three blind traces
    python labs/module-07/lesson-05/diagnose.py runs/m07/l75/cpu                     # your induced runs
    LAB_TARGET=solution python labs/module-07/lesson-05/diagnose.py runs/m07/l75/cpu

For every folder below the given one that has metrics.jsonl and stability.jsonl: the detected loss spikes (your
``find_spikes`` on the running median of the per-step losses, width 5), and for the first spike the evidence — max attention logit before it and its growth, the
update/weight ratio jump at the spike, whether the loss recovered — and the verdict of your ``classify``. Then,
for folders named f1*, a table of the fixes: max attention logit over the run, the step it first passed 20 (your
``first_crossing``), and, if checkpoints exist, held-out loss paired against f1-logits.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m07  # noqa: E402
from frontierlab.labkit import load_target  # noqa: E402
from frontierlab.metrics import read_jsonl  # noqa: E402
from frontierlab.optim import stability as st  # noqa: E402

lab = load_target(str(Path(__file__).parent / "test_lab.py"))


def call(fn, *args, fallback=None):
    try:
        return fn(*args)
    except NotImplementedError:
        return fallback(*args) if fallback else "TODO"


def runs_under(root: Path):
    return [p for p in sorted(root.iterdir()) if (p / "metrics.jsonl").exists() and (p / "stability.jsonl").exists()]


def load_rows(run: Path):
    """metrics and stability rows; a run started with --branch-from gets its parent's rows before the branch point."""
    metrics, stab = st.read_rows(run / "metrics.jsonl"), st.read_rows(run / "stability.jsonl")
    card = run / "run_card.yaml"
    if card.exists():
        import yaml
        src = (yaml.safe_load(card.read_text()).get("optim") or {}).get("branch_from")
        parent = Path(src).parent if src else None
        if parent is not None and (parent / "metrics.jsonl").exists():
            first = min(r["step"] for r in metrics if r.get("split") == "train")
            metrics = [r for r in st.read_rows(parent / "metrics.jsonl") if r["step"] < first] + metrics
            if (parent / "stability.jsonl").exists():
                first_s = min(r["step"] for r in stab)
                stab = [r for r in st.read_rows(parent / "stability.jsonl") if r["step"] < first_s] + stab
    return metrics, stab


def describe(run: Path):
    metrics, stab = load_rows(run)
    steps, loss = st.series(metrics, "loss", "train")
    flagged = call(lab.find_spikes, steps, st.smooth(loss, 5))     # on the running median of the losses
    ev = st.diagnose(metrics, stab)
    verdict = call(lab.classify, ev.get("logit_growth", 0.0), ev.get("max_logit_before", 0.0), ev.get("ratio_jump", 0.0),
                   ev.get("recovered", False)) if ev["spikes"] else ev["verdict"]
    print(f"\n{run.name}")
    print(f"  your find_spikes: {flagged if isinstance(flagged, str) else flagged[:12]}")
    if ev["spikes"]:
        s = ev["spikes"][0]
        print(f"  first spike: steps {s['start']}-{s['end']}, peak {s['peak_loss']:.3f} vs median {s['baseline']:.3f} "
              f"(threshold {s['threshold']:.3f}); {len(ev['spikes'])} spike(s) in all")
        print(f"  evidence: max logit before {ev['max_logit_before']:.1f} (growth x{ev['logit_growth']:.1f}); "
              f"update/weight ratio jump x{ev['ratio_jump']:.1f}; recovered within 20 steps: {ev['recovered']}")
    else:
        print(f"  no spike; max attention logit over the run {ev['max_logit_run']:.1f} (growth x{ev['logit_growth']:.1f})")
    print(f"  verdict (your classify): {verdict}    (reference rules: {ev['verdict']})")
    return metrics, stab


def fixes(root: Path):
    arms = [p for p in runs_under(root) if p.name.startswith("f1")]
    if not arms:
        return
    ref = root / "f1-logits"
    ref_l = m07.eval_losses(ref) if (ref / "checkpoint.pt").exists() else None
    print(f"\n{'fix arm':12s} {'max logit':>10s} {'step >= 20':>11s} {'end logit':>10s} {'spikes':>7s} {'held-out':>9s}  vs f1-logits")
    for run in arms:
        stab, metrics = read_jsonl(run / "stability.jsonl"), read_jsonl(run / "metrics.jsonl")
        ls, lg = st.series(stab, "max_logit")
        cross = call(lab.first_crossing, ls, lg, 20.0, ls[-1])
        sp = st.detect_spikes(*st.series(metrics, "loss", "train"))
        if (run / "checkpoint.pt").exists():
            l = m07.eval_losses(run)
            h = f"{m07.mean(l):9.4f}  " + (m07.compare(l, ref_l) if ref_l is not None and run != ref else "")
        else:
            h = "-"
        print(f"{run.name:12s} {max(lg):10.2f} {str(cross):>11s} {lg[-1]:10.2f} {len(sp):7d} {h}")


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/m07/l75/cpu")
    print(st.RULES)
    for run in runs_under(root):
        describe(run)
    fixes(root)


if __name__ == "__main__":
    main()
