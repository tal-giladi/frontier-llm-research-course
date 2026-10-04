"""Lab 08.3, step 4: 4-bit weights after training — post-training quantisation (PTQ) vs quantisation-aware training (QAT).

    python labs/module-08/lesson-03/qat_ptq.py                   # free CPU, about 10 minutes

1. base       train the toy model 200 steps in BF16 (fp32 on CPU).
2. PTQ        round the trained weights once and evaluate:
                 int4-g32 on every block linear      (Kimi-K2-Thinking's scheme: symmetric INT4, groups of 32)
                 mxfp4 on the SwiGLU matrices only   (gpt-oss quantised its MoE weights; the MLP is the dense analogue)
                 mxfp4 on every block linear
3. continue   from the base weights, 60 more steps three ways (fresh AdamW, warm-up 10, constant lr 1e-3; same
              data order): plain BF16 (the control), INT4 QAT (``--recipe int4-qat``), MXFP4 QAT on the MLP only
              (``--recipe mxfp4-qat --only mlp``, the DeepSeek-V4 analogue: FP4 QAT of expert weights in post-training).
4. compare    each QAT model exported to real quantised weights vs the control quantised by PTQ with the same
              scheme: equal tokens, equal starting point; paired by window over 256 held-out windows.
Runs: runs/m08/l83/qat/<name>. Emulation: speed is not measured.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m08  # noqa: E402

from frontierlab.precision import ptq_  # noqa: E402
from frontierlab.precision import train as prec_train  # noqa: E402
from frontierlab.precision.qat import export_  # noqa: E402

ROOT = Path("runs/m08/l83/qat")
COMMON = ["--preset", "toy", "--batch", "16", "--seq", "128", "--device", "cpu", "--eval-every", "1000",
          "--eval-windows", "64", "--log-every", "10"]
CONT = ["--schedule", "constant", "--warmup", "10", "--lr", "1e-3", "--data-seed", "7", "--ckpt-every", "60"]


def ptq_losses(run, spec, only, tag):
    cache = Path(run) / f"eval_256x128{tag}.json"
    if cache.exists():
        import json
        return json.loads(cache.read_text())
    m = prec_train.load_model(Path(run) / "checkpoint.pt", recipe="bf16")
    info = ptq_(m, spec, only=only)
    losses = m08.eval_model(m)
    import json
    cache.write_text(json.dumps(losses))
    print(f"  PTQ {spec} ({only}): {info['matrices']} matrices, {info['bits_per_weight']:.2f} bits/weight, "
          f"mean weight rel err {info['mean_rel_err']:.4f}")
    return losses


def qat_losses(run):
    cache = Path(run) / "eval_256x128_exported.json"
    if cache.exists():
        import json
        return json.loads(cache.read_text())
    m = prec_train.load_model(Path(run) / "checkpoint.pt")         # rebuilt with its QAT layers (run card)
    export_(m)                                                       # real quantised weights, plain nn.Linear
    losses = m08.eval_model(m)
    import json
    cache.write_text(json.dumps(losses))
    return losses


def main():
    base = ROOT / "base"
    m08.run_arm(base, [*COMMON, "--ckpt-every", "200"], 200)
    ck = str(base / "checkpoint.pt")
    arms = {"control": [], "int4-qat": ["--recipe", "int4-qat"], "mxfp4-qat-mlp": ["--recipe", "mxfp4-qat", "--only", "mlp"]}
    for name, extra in arms.items():
        m08.run_arm(ROOT / name, [*COMMON, *CONT, "--init-from", ck, *extra], 60)
    L = {"base": m08.eval_losses(base, recipe="bf16"), "control": m08.eval_losses(ROOT / "control", recipe="bf16")}
    print("\nheld-out loss, 256 windows x 128 tokens (mean):")
    print(f"  base (200 steps, BF16)            {m08.mean(L['base']):.4f}")
    for spec, only in (("int4-g32", "all"), ("mxfp4", "mlp"), ("mxfp4", "all")):
        l = ptq_losses(base, spec, only, f"_ptq_{spec}_{only}")
        print(f"  base + PTQ {spec:9s} {only:4s}     {m08.mean(l):.4f}   vs base {m08.fmt(m08.compare(l, L['base']))}")
    print(f"  control (+60 steps, BF16)         {m08.mean(L['control']):.4f}")
    for qat, spec, only in (("int4-qat", "int4-g32", "all"), ("mxfp4-qat-mlp", "mxfp4", "mlp")):
        ptq = ptq_losses(ROOT / "control", spec, only, f"_ptq_{spec}_{only}")
        q = qat_losses(ROOT / qat)
        print(f"  control + PTQ {spec:9s} {only:4s}  {m08.mean(ptq):.4f}   vs control {m08.fmt(m08.compare(ptq, L['control']))}")
        print(f"  {qat:14s} exported          {m08.mean(q):.4f}   vs control {m08.fmt(m08.compare(q, L['control']))}   "
              f"QAT - PTQ {m08.fmt(m08.compare(q, ptq))}")


if __name__ == "__main__":
    main()
