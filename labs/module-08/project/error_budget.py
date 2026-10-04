"""Module 8 project: the error budget of a precision plan, one component at a time, on a trained BF16 model.

    python labs/module-08/project/error_budget.py                                    # CPU: the 08.2 bf16-s0 toy model
    python labs/module-08/project/error_budget.py --ckpt runs/b0-s0/checkpoint.pt --device cuda --hw H100-SXM \\
        --preset baseline0 --batch 16 --seq 1024 --T 1024                            # main path: Baseline-0 (not run in this build)

For each component of the plan, the trained weights are evaluated with only that component quantised (emulation,
forward pass only: what quantisation does to the function the model computes; the training-time effect of the same
choice comes from the 08.2 / 08.3 runs). Every row is paired with the BF16 evaluation on the same 256 windows.
Also: storage bits per weight of the quantised matrices and the PROJECTED step-time ratio on --hw
(``frontierlab.precision.cost``, casts fused). Last, an additivity check: the gaps of "attention linears" and
"MLP linears" against the gap of "all linears" for the same recipe.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m08  # noqa: E402

from frontierlab.model import PRESETS  # noqa: E402
from frontierlab.precision import get_recipe, ptq_, select_linears  # noqa: E402
from frontierlab.precision import train as prec_train  # noqa: E402
from frontierlab.precision.cost import projected_speedup  # noqa: E402

COMPONENTS = [  # name, recipe or PTQ spec, only, lowp class for the projection
    ("fp8-deepseek attn", "fp8-deepseek", "attn", "fp8"),
    ("fp8-deepseek mlp", "fp8-deepseek", "mlp", "fp8"),
    ("fp8-deepseek all", "fp8-deepseek", "all", "fp8"),
    ("fp8-tensorwise all", "fp8-tensorwise", "all", "fp8"),
    ("nvfp4 attn", "nvfp4", "attn", "fp4"),
    ("nvfp4 mlp", "nvfp4", "mlp", "fp4"),
    ("nvfp4 all", "nvfp4", "all", "fp4"),
    ("PTQ int4-g32 mlp weights", "ptq:int4-g32", "mlp", None),
    ("PTQ mxfp4 mlp weights", "ptq:mxfp4", "mlp", None),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("runs/m08/l82/cpu/bf16-s0/checkpoint.pt"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hw", default="H100-SXM")
    ap.add_argument("--preset", default="baseline0", help="model for the throughput projection")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--T", type=int, default=128, help="evaluation window length")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if not a.ckpt.exists():
        raise SystemExit(f"{a.ckpt} not found: run labs/module-08/lesson-02/train_fp8.py first (or pass --ckpt)")
    out_dir = a.out or a.ckpt.parent / "error_budget"
    out_dir.mkdir(parents=True, exist_ok=True)

    def losses(name, build):
        cache = out_dir / f"{name.replace(' ', '_')}.json"
        if cache.exists():
            return json.loads(cache.read_text())
        m = build().to(a.device)
        l = m08.eval_model(m, T=a.T, device=a.device)
        cache.write_text(json.dumps(l))
        return l

    base = losses("bf16", lambda: prec_train.load_model(a.ckpt, recipe="bf16", map_location=a.device))
    cfg = PRESETS[a.preset](vocab_size=32768)
    print(f"checkpoint {a.ckpt}; BF16 held-out loss {m08.mean(base):.4f} on {len(base)} windows of {a.T} tokens")
    print(f"budget reference: 0.25% of the BF16 loss = {0.0025 * m08.mean(base):.4f} nats\n")
    print(f"{'component':28s} {'gap vs BF16 [95% CI]':>30s} {'bits/weight':>11s} {'proj. speed-up':>14s}")
    gaps = {}
    for name, rec, only, lowp in COMPONENTS:
        if rec.startswith("ptq:"):
            spec = rec[4:]

            def build(spec=spec, only=only):
                m = prec_train.load_model(a.ckpt, recipe="bf16", map_location=a.device)
                ptq_(m, spec, only=only)
                return m
            from frontierlab.precision import get_spec
            bits = get_spec(spec).bits_per_value()
        else:
            def build(rec=rec, only=only):
                return prec_train.load_model(a.ckpt, recipe=rec, only=only, map_location=a.device)
            bits = get_recipe(rec).weight.bits_per_value()
        l = losses(name, build)
        c = m08.compare(l, base)
        gaps[name] = c
        sp = "" if lowp is None else f"{projected_speedup(cfg, a.batch, a.seq, a.hw if lowp == 'fp8' else 'B200', lowp, True)['speedup']:.2f}x*"
        print(f"{name:28s} {m08.fmt(c):>30s} {bits:11.2f} {sp:>14s}")
    print(f"\n* PROJECTED whole-step ratio with every block linear in that format ({a.preset}, {a.batch} x {a.seq}, "
          f"{a.hw} for FP8, B200 for FP4; casts fused). Not a measurement: bench_fp8.py on real kernels is.")
    for rec in ("fp8-deepseek", "nvfp4"):
        s = gaps[f"{rec} attn"]["diff"] + gaps[f"{rec} mlp"]["diff"]
        print(f"additivity {rec}: attn + mlp = {s:+.4f}, all = {gaps[f'{rec} all']['diff']:+.4f}")
    n_lin = len(select_linears(prec_train.load_model(a.ckpt, recipe="bf16")))
    print(f"({n_lin} block linears; embedding and output head never quantised)")


if __name__ == "__main__":
    main()
