"""Lab 05.2: convert a dense model to DSA-style sparse attention with the two documented stages.

    python labs/module-05/lesson-02/dsa_stages.py                      # free CPU, from runs/m04/base-cpu
    python labs/module-05/lesson-02/dsa_stages.py --variant t4         # free GPU
    python labs/module-05/lesson-02/dsa_stages.py --variant main       # main path (1x H100 / A100)
    python labs/module-05/lesson-02/dsa_stages.py --report-only        # only the tables, from finished runs

Runs (all from the same dense parent checkpoint, same data order and seed):

    warmup      dense attention, only the indexer trains (KL to the main attention)    --dsa-stage warmup
    sparse      top-k on, from the warm-up; indexer by KL only, main model by LM only  --dsa-stage sparse
    control     the dense parent trained on, same steps, tokens and schedule as sparse (the comparison arm)
    no-warmup   (with --ablation) sparse stage straight from a random indexer

Then the report: (1) attention-mass recall of the indexer's top-k against the oracle top-k and the k most
recent keys, before the warm-up, after it, and after sparse training; (2) held-out loss of sparse and
no-warmup against control, paired by window; (3) the sparse model's held-out loss as k varies at
evaluation time.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

VARIANTS = {   # parent, preset, seq, topk, warm steps, warm lr, sparse steps, sparse lr, batch, extra loop args
    "cpu": ("runs/m04/base-cpu/checkpoint.pt", "toy", 256, 32, 150, 1e-3, 400, 3e-4, 16,
            ["--eval-every", "1000", "--ckpt-every", "100", "--eval-windows", "64", "--log-every", "50"]),
    "t4": ("runs/m04/base-t4/checkpoint.pt", "pilot-10m", 512, 64, 300, 1e-3, 1000, 3e-4, 32,
           ["--dtype", "fp32", "--eval-every", "500", "--ckpt-every", "250", "--max-minutes", "150"]),
    "main": ("runs/m01/b0-s0/checkpoint.pt", "baseline0", 1024, 128, 500, 1e-3, 2000, 3e-4, 32,
             ["--dtype", "bf16", "--grad-accum", "2", "--eval-every", "500", "--ckpt-every", "250",
              "--peak", "H100-SXM"]),
}


def run(cmd, dry):
    print(" ".join(str(c) for c in cmd))
    if not dry and subprocess.call([str(c) for c in cmd]) != 0:
        sys.exit("run failed")


def finished(run_dir: Path, steps: int) -> bool:
    log = run_dir / "metrics.jsonl"
    return log.exists() and any(f'"step": {steps}' in line for line in log.read_text().splitlines())


def train(arm, run_dir, steps, lr, init, stage, a, v, dry):
    parent, preset, seq, topk, *_, batch, extra = v
    if finished(run_dir, steps) and not dry:
        print(f"  {run_dir} finished, skipping")
        return
    cmd = [sys.executable, HERE.parent / "train_arm.py", "--arm", arm, "--init-from", init,
           *(["--topk", topk] if arm == "dsa" else []), *(["--dsa-stage", stage] if stage else []), "--",
           "--run", run_dir, "--preset", preset, "--steps", steps, "--batch", batch, "--seq", seq, "--lr", lr,
           "--warmup", max(10, steps // 20), "--seed", a.seed,
           "--question", f"05.2 {run_dir.name}: does DSA-style selection keep held-out loss at k = {topk}?", *extra]
    run(cmd, dry)


@torch.no_grad()
def report(out: Path, v, device="cpu", windows=64):
    import m05
    from frontierlab.attention import dsa
    from frontierlab.data.loader import TokenData
    from frontierlab.evals.heldout import window_losses
    from frontierlab.model import LM
    from frontierlab.stats import paired_bootstrap
    parent, preset, seq, topk = v[:4]
    val = TokenData("val")
    starts = val.eval_windows(16, seq, 99)
    batch = torch.stack([val.window(s, seq) for s in starts]).to(device)

    def recall(model):
        rows = dsa.selection_recall(model, batch, topk)
        return {k: sum(r[k] for r in rows) / len(rows) for k in ("indexer", "oracle", "window")}

    print(f"\n1. Attention-mass recall at k = {topk} of {seq} keys (16 val windows, mean over layers and queries)")
    print(f"   {'model':<34s} {'indexer':>8s} {'oracle':>8s} {'recent-k':>8s}")
    dense, _ = m05.load_run(Path(parent).parent if Path(parent).name == "checkpoint.pt" else parent, device)
    conv = LM(dense.config.with_(attention="dsa", extra={"index_topk": topk, "index_heads": 4, "index_head_dim": 32}))
    torch.manual_seed(0)
    conv.load_state_dict(dense.state_dict(), strict=False)
    res = {"random indexer (converted parent)": recall(conv.to(device).eval())}
    for name in ("warmup", "sparse", "no-warmup"):
        if (out / name / "checkpoint.pt").exists():
            m, _ = m05.load_run(out / name, device)
            res[f"after {name}"] = recall(m)
    for k, r in res.items():
        print(f"   {k:<34s} {r['indexer']:8.3f} {r['oracle']:8.3f} {r['window']:8.3f}")

    print(f"\n2. Held-out loss, {windows * 4} val windows of {seq} tokens, paired against control")
    ref = window_losses(m05.load_run(out / "control", device)[0], val, windows * 4, seq, device=device)
    print(f"   control                  {sum(ref) / len(ref):.4f}")
    table = {"recall": res}
    for name in ("sparse", "no-warmup"):
        if (out / name / "checkpoint.pt").exists():
            m, _ = m05.load_run(out / name, device)
            ls = window_losses(m, val, windows * 4, seq, device=device)
            pb = paired_bootstrap(ls, ref)
            table[name] = pb
            print(f"   {name:<24s} {sum(ls) / len(ls):.4f}  diff {pb['mean_diff']:+.4f} "
                  f"[{pb['ci'][0]:+.4f}, {pb['ci'][1]:+.4f}]")

    print("\n3. The sparse model at other k (evaluation only; trained at k = %d)" % topk)
    m, _ = m05.load_run(out / "sparse", device)
    for k in sorted({max(1, topk // 4), topk // 2, topk, 2 * topk, seq}):
        dsa.set_dsa(m, topk=k)
        ls = window_losses(m, val, windows * 4, seq, device=device)
        pb = paired_bootstrap(ls, ref)
        print(f"   k = {k:>5d}  loss {sum(ls) / len(ls):.4f}  vs control {pb['mean_diff']:+.4f} "
              f"[{pb['ci'][0]:+.4f}, {pb['ci'][1]:+.4f}]")
    (out / "report.json").write_text(json.dumps(table, default=float))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cpu")
    ap.add_argument("--parent", default=None, help="dense checkpoint to start from (default: the variant's)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablation", action="store_true", help="also run the no-warmup arm")
    ap.add_argument("--out", type=Path, default=Path("runs/m05/l52"))
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    v = list(VARIANTS[a.variant])
    if a.parent:
        v[0] = a.parent
    parent, preset, seq, topk, wsteps, wlr, ssteps, slr, batch, extra = v
    out = a.out / a.variant
    if not a.report_only:
        if not Path(parent).exists() and not a.print:
            sys.exit(f"{parent} not found: train it first (lesson 04.1: python labs/module-04/lesson-01/train_base.py)")
        train("dsa", out / "warmup", wsteps, wlr, parent, "warmup", a, v, a.print)
        train("dsa", out / "sparse", ssteps, slr, out / "warmup" / "checkpoint.pt", "sparse", a, v, a.print)
        train("b0", out / "control", ssteps, slr, parent, None, a, v, a.print)
        if a.ablation:
            train("dsa", out / "no-warmup", ssteps, slr, parent, "sparse", a, v, a.print)
    if not a.print:
        report(out, v, a.device)


if __name__ == "__main__":
    main()
