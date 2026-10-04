"""Lab 08.3, step 2: what the random Hadamard transform and stochastic rounding do to an FP4 weight gradient.

    python labs/module-08/lesson-03/wgrad_error.py            # free CPU, about 2 minutes (reuses lesson 08.1's checkpoint)

On the real x and dy of every linear of the toy model (one batch, the 60-step checkpoint of lesson 08.1, float64):
  dW = dyᵀ x exactly, versus the NVFP4 Wgrad (1 x 16 blocks along the token dim) with and without a 16-point RHT,
  with round-to-nearest or stochastic rounding of dy. Reported per variant, mean over layers:
    rel err      ||dW_q - dW|| / ||dW|| for one draw
    bias         ||mean over 64 draws of dW_q - dW|| / ||dW||   (what is left after averaging: RNE keeps its bias,
                 SR's shrinks like 1/sqrt(draws))
    underflow    fraction of nonzero dyᵀ values flushed to zero
Also with one token in every 64 scaled x 30 (an outlier token), the case the RHT is meant for.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lesson-01"))

import m08  # noqa: E402
from error_tour import capture  # noqa: E402

from frontierlab.data.loader import TokenData  # noqa: E402
from frontierlab.precision import QuantSpec, apply_rht, qdq, quant_error  # noqa: E402
from frontierlab.precision import train as prec_train  # noqa: E402

G_RNE = QuantSpec("e2m1", (1, 16), "nvfp4", "rne")
G_SR = QuantSpec("e2m1", (1, 16), "nvfp4", "sr")
X = QuantSpec("e2m1", (1, 16), "nvfp4", "rne")


def wgrad(dy, x, gspec, rht, gen):
    gt, xt = dy.t(), x.t()
    if rht:
        gt, xt = apply_rht(gt, 16, 0), apply_rht(xt, 16, 0)
    from frontierlab.precision.quant import dequantize, quantize
    q, s = quantize(gt, gspec, gen)
    return dequantize(q, s, gspec) @ qdq(xt, X).t(), gt


def report(tensors, label, draws=64):
    print(f"\n{label}")
    print(f"{'variant':22s} {'rel err':>8s} {'bias':>8s} {'dy underflow':>13s}")
    gen = torch.Generator().manual_seed(0)
    for name, gspec, rht in (("RNE", G_RNE, False), ("SR", G_SR, False), ("RHT + RNE", G_RNE, True),
                             ("RHT + SR (NVFP4)", G_SR, True)):
        rel, bias, uf = [], [], []
        for t in tensors.values():
            exact = t["dy"].t() @ t["x"]
            one, gt = wgrad(t["dy"], t["x"], gspec, rht, gen)
            n = draws if gspec.rounding == "sr" else 1
            avg = sum(wgrad(t["dy"], t["x"], gspec, rht, gen)[0] for _ in range(n)) / n
            rel.append(((one - exact).norm() / exact.norm()).item())
            bias.append(((avg - exact).norm() / exact.norm()).item())
            uf.append(quant_error(gt, G_RNE)["underflow"])
        k = len(rel)
        print(f"{name:22s} {sum(rel) / k:8.4f} {sum(bias) / k:8.4f} {sum(uf) / k:13.4f}")


def main():
    run = Path("runs/m08/l81/base-toy")
    m08.run_arm(run, ["--preset", "toy", "--batch", "16", "--seq", "128", "--device", "cpu", "--eval-every", "1000",
                      "--ckpt-every", "60", "--log-every", "20"], 60)
    model = prec_train.load_model(run / "checkpoint.pt").double()
    x = TokenData("train").batch(16, 128, torch.Generator().manual_seed(123))
    tensors = capture(model, x)
    report(tensors, "real tensors (2,048 tokens, 28 linears)")
    for t in tensors.values():
        t["x"] = t["x"].clone()
        t["x"][::64] *= 30
        t["dy"] = t["dy"].clone()
        t["dy"][::64] *= 30
    report(tensors, "one token in every 64 scaled x 30 in both x and dy")


if __name__ == "__main__":
    main()
