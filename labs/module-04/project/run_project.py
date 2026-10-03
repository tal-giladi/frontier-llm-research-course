"""Module 4 project: a two-stage context extension of the base model, an equal-token control, and the evaluation.

    python labs/module-04/project/run_project.py --variant cpu            # 256 -> 1,024 -> 2,048
    python labs/module-04/project/run_project.py --variant main --print   # Baseline-0 1K -> 8K -> 32K (commands only)
    python labs/module-04/project/run_project.py --variant cpu --stage eval

Stages (each resumes exactly if interrupted; rerun the same command):

    stage1   from the base: YaRN with s = L1 / L0, within-document windows of L1 tokens
    stage2   from stage1:   YaRN with s = L2 / L0 (original length kept at L0, as DeepSeek-V3 keeps one
             YaRN setting for both phases), within-document windows of L2 tokens
    control  from the base: the original length L0, RoPE unchanged, as many tokens as stage1 + stage2
    eval     Eval v1 at L0, L1, L2 (W = L0) and Eval v0 at L0 for base, base+yarn, stage1, stage2, control

The main path needs long training documents: run ``prepare_long`` with ``--splits train val test``
first (see the project brief) and pass ``--long-data`` to this script.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

VARIANTS = {
    #        base run,               preset,      L0,   L1,    L2,  (batch, accum, steps) per stage,  extra
    "cpu": ("runs/m04/base-cpu", "toy", 256, 1024, 2048, {"stage1": (4, 1, 400), "stage2": (2, 1, 200),
                                                            "control": (24, 1, 400)}, []),
    "t4": ("runs/m04/base-t4", "pilot-10m", 512, 2048, 4096, {"stage1": (8, 1, 600), "stage2": (2, 2, 300),
                                                                 "control": (32, 1, 900)}, ["--device", "cuda"]),
    "main": ("runs/m01/main/seeds-lr3e-3-s0", "baseline0", 1024, 8192, 32768,
             {"stage1": (4, 8, 1526), "stage2": (1, 8, 763), "control": (32, 8, 2289)},
             ["--device", "cuda", "--dtype", "bf16", "--loss", "chunked", "--peak", "H100-SXM", "--max-minutes", "230"]),
}


def commands(v: str, out: str, base: str | None, long_data: str | None):
    base0, preset, L0, L1, L2, plan, extra = VARIANTS[v]
    base = base or base0
    data = ["--data", long_data, "--short-data", "labs/common/data/v0"] if long_data else []
    common = ["--preset", preset, "--lr", "1e-3", "--warmup", "20", "--seed", "0", "--log-every", "20", *extra]

    def run(name, init, seq, rope, long_fraction, data_args):
        B, acc, steps = plan[name]
        args = ["--run", f"{out}/{v}-{name}", "--init-from", f"{init}/checkpoint.pt", "--seq", str(seq),
                "--batch", str(B), "--grad-accum", str(acc), "--steps", str(steps), "--eval-every", str(steps),
                *rope, *data_args, *common, "--question", f"Module 4 project {name}"]
        if long_fraction is not None:
            args += ["--long-fraction", str(long_fraction)]
        return [sys.executable, "-m", "frontierlab.longctx.extend", *args]

    yarn = lambda L: ["--rope", "yarn", "--factor", f"{L / L0:g}", "--original", str(L0)]  # noqa: E731
    return {"stage1": run("stage1", base, L1, yarn(L1), 1.0, data),
            "stage2": run("stage2", f"{out}/{v}-stage1", L2, yarn(L2), 1.0, data),
            "control": run("control", base, L0, ["--rope", "default"], None, [])}, (base, L0, L1, L2)


def evaluate(v, out, base, L0, L1, L2, device):
    """Eval v1 and Eval v0 for every model, cached as JSON next to each checkpoint, then a summary."""
    from frontierlab.data.loader import TokenData
    from frontierlab.data.prepare import DEFAULT_OUT
    from frontierlab.evals import suite_v0, suite_v1 as v1
    from frontierlab.stats import bootstrap_ci
    long_root = DEFAULT_OUT.parent / "v0-long"
    data = TokenData("val", long_root if (long_root / "meta.json").exists() else DEFAULT_OUT)
    val = TokenData("val")
    vocab = v1.vocab_from_tokenizer(DEFAULT_OUT / "tokenizer.json", val)
    suite_v0.download_lambada()
    enc = suite_v0.data_v0_encoder()
    yarn2 = {"type": "yarn", "factor": L2 / L0, "original_max_position_embeddings": L0}
    models = {"base": (base, None), "base+yarn": (base, yarn2), "stage1": (f"{out}/{v}-stage1", None),
              "stage2": (f"{out}/{v}-stage2", None), "control": (f"{out}/{v}-control", None)}
    res = {}
    for name, (run, rope) in models.items():
        tag = "_yarn" if rope else ""
        p1, p0 = Path(run) / f"project_eval_v1{tag}.json", Path(run) / f"project_eval_v0{tag}.json"
        if not p1.exists() or not p0.exists():
            m = v1.load_model(run, device, rope)
            if not p1.exists():
                p1.write_text(json.dumps(v1.run_suite(m, vocab, data, [L0, L1, L2], L0, device=device)))
            if not p0.exists():
                p0.write_text(json.dumps(suite_v0.run_suite(m, val, enc, n_windows=256, T=L0, device=device)))
        res[name] = (json.loads(p1.read_text()), json.loads(p0.read_text()))
    print(f"Context gain (W = {L0}) at each length, and Eval v0 at {L0} against the base (paired):")
    for name, (r1, r0) in res.items():
        cells = []
        for nat in r1["natural"]:
            g = nat.get("context_gain", {}).get(str(L0))
            if g:
                m, lo, hi = bootstrap_ci(g, n_boot=4000)
                cells.append(f"L={nat['length']}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        line = f"  {name:<10} " + "   ".join(cells)
        if name != "base":
            reg = v1.short_context_regression(res["base"][1], r0, n_boot=4000)
            h, lp = reg["heldout_loss_diff"], reg["lambada_logprob_diff"]
            line += (f"\n  {'':<10} held-out at {L0} {h['mean_diff']:+.4f} [{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}]"
                     f"   LAMBADA logp {lp['mean_diff']:+.3f} [{lp['ci'][0]:+.3f}, {lp['ci'][1]:+.3f}]  (vs base)")
        if name in ("stage1", "stage2"):
            reg = v1.short_context_regression(res["control"][1], r0, n_boot=4000)
            h, lp = reg["heldout_loss_diff"], reg["lambada_logprob_diff"]
            line += (f"\n  {'':<10} held-out at {L0} {h['mean_diff']:+.4f} [{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}]"
                     f"   LAMBADA logp {lp['mean_diff']:+.3f} [{lp['ci'][0]:+.3f}, {lp['ci'][1]:+.3f}]  (vs control)")
        print(line)
    print("\nSynthetic tasks per length: retrieval accuracy and evidence effect (mean over depths), two-hop accuracy")
    for name, (r1, _) in res.items():
        parts = []
        for L in (L0, L1, L2):
            one = [c for c in r1["synthetic"] if c["length"] == L and c["hops"] == 1]
            two = [c for c in r1["synthetic"] if c["length"] == L and c["hops"] == 2]
            acc = sum(v1.summarize_scores(c["scores"])["acc"][0] for c in one) / len(one)
            ev = sum(v1.evidence_effect(c)["mean_diff"] for c in one) / len(one)
            acc2 = v1.summarize_scores(two[0]["scores"])["acc"][0] if two else float("nan")
            parts.append(f"L={L}: {acc:.2f} / {ev:+.3f} / {acc2:.2f}")
        print(f"  {name:<10} " + "   ".join(parts))
    print("\nFull reports: python -m frontierlab.evals.suite_v1 compare <a.json> <b.json>  (files listed below)")
    for name, (run, rope) in models.items():
        print(f"  {name:<10} {run}/project_eval_v1{'_yarn' if rope else ''}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cpu")
    ap.add_argument("--stage", choices=["stage1", "stage2", "control", "eval", "all"], default="all")
    ap.add_argument("--base", default=None, help="base run (main path: your Module 1 Baseline-0 seed-0 run)")
    ap.add_argument("--long-data", default=None, help="prepare_long folder with a train split (main path)")
    ap.add_argument("--out", default="runs/m04/project")
    ap.add_argument("--device", default=None)
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    cmds, (base, L0, L1, L2) = commands(a.variant, a.out, a.base, a.long_data)
    for name in ("stage1", "stage2", "control"):
        if a.stage in (name, "all"):
            print(" ".join(cmds[name]), flush=True)
            if not a.print and subprocess.call(cmds[name]):
                sys.exit(f"{name} failed")
    if a.stage in ("eval", "all") and not a.print:
        import torch
        evaluate(a.variant, a.out, base, L0, L1, L2, a.device or ("cuda" if torch.cuda.is_available() else "cpu"))


if __name__ == "__main__":
    main()
