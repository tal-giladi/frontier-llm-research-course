"""Lab 08.1, step 3: how much do real training tensors lose in each format and scaling granularity?

    python labs/module-08/lesson-01/error_tour.py                     # free CPU (toy, ~2 minutes)
    python labs/module-08/lesson-01/error_tour.py --device cuda --preset pilot-10m --steps 300   # GPU variant

1. Trains the toy model for --steps steps (bf16 recipe = no quantisation; fp32 on CPU) so the tensors are those
   of a model that has started learning, not of a random initialisation.
2. Runs one training batch with hooks on every block linear and records its input activations x, its weight W
   and its output gradient dy (the three tensors the recipes quantise).
3. For every spec, prints the mean over layers of the relative error ||q(t) - t|| / ||t||, the fraction of nonzero
   values flushed to zero, the fraction clamped at the format maximum, and storage bits per value.
4. Repeats the activation row with one injected outlier channel (x 100) to show what per-tensor scaling does.

Everything here is emulation in float64; it measures numerics, not speed.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m08  # noqa: E402
from frontierlab.data.loader import TokenData  # noqa: E402
from frontierlab.precision import QuantSpec, quant_error, select_linears  # noqa: E402
from frontierlab.precision import train as prec_train  # noqa: E402

SPECS = {
    "e4m3 tensor": QuantSpec("e4m3", (0, 0), "fp32"),
    "e5m2 tensor": QuantSpec("e5m2", (0, 0), "fp32"),
    "e4m3 row pow2": QuantSpec("e4m3", (1, 0), "pow2"),
    "e4m3 1x128": QuantSpec("e4m3", (1, 128), "fp32"),
    "e4m3 128x128": QuantSpec("e4m3", (128, 128), "fp32"),
    "mxfp8 1x32": QuantSpec("e4m3", (1, 32), "e8m0"),
    "mxfp4 1x32": QuantSpec("e2m1", (1, 32), "e8m0"),
    "mxfp4 1x32 ceil": QuantSpec("e2m1", (1, 32), "pow2"),
    "nvfp4 1x16": QuantSpec("e2m1", (1, 16), "nvfp4"),
    "nvfp4 16x16": QuantSpec("e2m1", (16, 16), "nvfp4"),
    "e2m1 tensor": QuantSpec("e2m1", (0, 0), "fp32"),
    "int4 g32": QuantSpec("int4", (1, 32), "fp32"),
}


def capture(model, x):
    """One forward/backward; returns {layer: {"x": input (M, K), "w": weight (N, K), "dy": output grad (M, N)}}."""
    names = select_linears(model)
    out, hooks = {n: {} for n in names}, []
    for n in names:
        m = model.get_submodule(n)
        hooks.append(m.register_forward_hook(lambda mod, inp, o, n=n: out[n].update(x=inp[0].detach().reshape(-1, inp[0].shape[-1]))))
        hooks.append(m.register_full_backward_hook(lambda mod, gi, go, n=n: out[n].update(dy=go[0].detach().reshape(-1, go[0].shape[-1]))))
        out[n]["w"] = m.weight.detach()
    model.train()
    model(x, labels=x).loss.backward()
    for h in hooks:
        h.remove()
    return out


def table(tensors, roles=("x", "w", "dy")):
    print(f"{'spec':18s} {'bits':>5s} | " + " | ".join(f"{r + ' rel err':>10s} {'uflow':>6s} {'sat':>6s}" for r in roles))
    rows = {}
    for name, spec in SPECS.items():
        cells = []
        for role in roles:
            es = [quant_error(t[role], spec) for t in tensors.values()]
            agg = {k: sum(e[k] for e in es) / len(es) for k in ("rel_err", "underflow", "saturated", "bits")}
            rows[(name, role)] = agg
            cells.append(f"{agg['rel_err']:10.4f} {agg['underflow']:6.3f} {agg['saturated']:6.4f}")
        print(f"{name:18s} {agg['bits']:5.2f} | " + " | ".join(cells))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="toy")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/m08/l81")
    a = ap.parse_args(argv)
    run = Path(a.out) / f"base-{a.preset}"
    m08.run_arm(run, ["--preset", a.preset, "--batch", "16", "--seq", "128", "--device", a.device,
                      "--eval-every", "1000", "--ckpt-every", str(a.steps), "--log-every", "20"], a.steps)
    model = prec_train.load_model(run / "checkpoint.pt", map_location=a.device).to(a.device).double()
    g = torch.Generator().manual_seed(123)
    x = TokenData("train").batch(16, 128, g, a.device)
    tensors = {n: {k: v.cpu() for k, v in d.items()} for n, d in capture(model, x).items()}
    print(f"\n{len(tensors)} linears, one batch of 16 x 128 tokens; mean over layers (float64 emulation)\n")
    table(tensors)
    print("\nActivations with one outlier channel (channel 7 of every linear input x 100):\n")
    for t in tensors.values():
        t["x"] = t["x"].clone()
        t["x"][:, 7] *= 100
    table(tensors, roles=("x",))
    amax = max(t["dy"].abs().max().item() for t in tensors.values())
    tiny = min(t["dy"][t["dy"] != 0].abs().min().item() for t in tensors.values())
    print(f"\ngradient dy: largest |value| {amax:.3e}, smallest nonzero {tiny:.3e} "
          f"(ratio 2^{torch.log2(torch.tensor(amax / tiny)).item():.1f}; E4M3 spans 2^17.8, E5M2 2^31.8)")


if __name__ == "__main__":
    main()
