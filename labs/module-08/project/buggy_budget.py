"""Module 8 project, debugging task: a colleague's version of the error-budget report. Find what is wrong.

    python labs/module-08/project/buggy_budget.py                 # uses runs/m08/l82/cpu (lesson 08.2's runs)

It prints a gap for NVFP4 MLP weights, a memory figure and a speed verdict, and concludes that FP8 is too slow
to use and that 4-bit MLP weights cost almost nothing. Do not trust any of the three numbers until you have
checked how each was produced.
"""

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m08  # noqa: E402

from frontierlab.data.loader import TokenData  # noqa: E402
from frontierlab.evals.heldout import window_losses  # noqa: E402
from frontierlab.precision import get_spec, ptq_  # noqa: E402
from frontierlab.precision import train as prec_train  # noqa: E402

ROOT = Path("runs/m08/l82/cpu")


def main():
    ck = ROOT / "bf16-s0" / "checkpoint.pt"
    val = TokenData("val")
    base_model = prec_train.load_model(ck, recipe="bf16")
    base = window_losses(base_model, val, 256, 128)
    q_model = prec_train.load_model(ck, recipe="nvfp4", only="mlp")
    quant = window_losses(q_model, val, 256, 128, seed=4321)
    c = m08.compare(quant, base)
    print(f"NVFP4 MLP linears: gap {m08.fmt(c)}")
    w = prec_train.load_model(ck, recipe="bf16")
    info = ptq_(w, "nvfp4-2d", only="mlp")
    bits = get_spec("nvfp4-2d").format.bits
    print(f"MLP weights: {info['params']:,} parameters at {bits} bits = {info['params'] * bits / 8 / 1e6:.3f} MB")
    t_bf16 = statistics.median(r["tok_per_s"] for r in m08.train_rows(ROOT / "bf16-s0") if r["step"] > 20)
    t_fp8 = statistics.median(r["tok_per_s"] for r in m08.train_rows(ROOT / "fp8-deepseek-s0") if r["step"] > 20)
    print(f"FP8 throughput: {t_fp8 / t_bf16:.2f}x of BF16 -> FP8 is too slow to adopt")
    print("Conclusion: 4-bit MLP weights cost almost nothing; FP8 should not be used.")


if __name__ == "__main__":
    main()
