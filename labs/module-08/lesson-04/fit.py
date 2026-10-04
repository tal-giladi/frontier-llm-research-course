"""Lab 08.4, step 3: fit the precision law's functional form to your sweep, and read what it does and does not say.

    python labs/module-08/lesson-04/fit.py                    # reads runs/m08/l84/cpu/results.json
    LAB_TARGET=solution python labs/module-08/lesson-04/fit.py

Fits, with your lab.py functions:
  Part A  L = A · (N (1 - e^(-P/γ)))^(-α) + E  at fixed D, over all widths and precisions (P = inf for unquantised)
          -> γ, α; the P* that Eq. 5 would imply if activations and KV cache shared this γ (k = 3; an assumption,
             since the sweep quantises weights only), and the same with the paper's γ_w
          -> leave-one-width-out: refit without the widest model and predict its losses (a check of the fit, not of the law)
  Part B  δ_PTQ(D) ∝ D^p at fixed N for INT4 and INT3 -> p (the paper's γ_D is 0.5068)
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

from frontierlab.labkit import load_target
from frontierlab.precision.scaling_law import PAPER_CONSTANTS

lab = load_target(str(Path(__file__).parent / "test_lab.py"))
ALPHAS = np.linspace(0.05, 1.5, 59)
GAMMAS = np.linspace(0.2, 6.0, 117)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/m08/l84/cpu/results.json")
    res = json.loads(path.read_text())
    tr = res["train"]
    N = np.array([r["N"] for r in tr], dtype=np.float64)
    P = np.array([np.inf if r["bits"] is None else r["bits"] for r in tr], dtype=np.float64)
    L = np.array([r["loss"] for r in tr])
    f = lab.fit(N, P, L, ALPHAS, GAMMAS)
    rms = math.sqrt(f["sse"] / len(L))
    print(f"Part A: {len(L)} runs. Fit: A = {f['A']:.4g}, alpha = {f['alpha']:.3f}, gamma = {f['gamma']:.3f}, "
          f"E = {f['E']:.4f}; residual RMS {rms:.4f} nats")
    for name, v, grid in (("alpha", f["alpha"], ALPHAS), ("gamma", f["gamma"], GAMMAS)):
        if v in (grid[0], grid[-1]):
            print(f"   WARNING: {name} = {v:.3f} is at the edge of its grid: the data do not pin it down (degenerate fit)")
    for r, l, pred in zip(tr, L, f["A"] * lab.n_eff(N, P, f["gamma"]) ** -f["alpha"] + f["E"]):
        print(f"   {r['run']:12s} measured {l:.4f}  fitted {pred:.4f}  ({l - pred:+.4f})")
    print(f"   P* if activations and KV cache shared this gamma (Eq. 5, k = 3; NOT measured here): {lab.p_star(f['gamma']):.2f} bits; "
          f"with the paper's gamma_w = {PAPER_CONSTANTS['gamma_w']}: {lab.p_star(PAPER_CONSTANTS['gamma_w']):.2f} bits; "
          f"the paper reports 7-8 bits from its own fits")
    widest = N.max()
    keep = N < widest
    f2 = lab.fit(N[keep], P[keep], L[keep], ALPHAS, GAMMAS)
    pred = f2["A"] * lab.n_eff(N[~keep], P[~keep], f2["gamma"]) ** -f2["alpha"] + f2["E"]
    err = L[~keep] - pred
    print(f"   leave-widest-out: gamma {f2['gamma']:.3f}, alpha {f2['alpha']:.3f}; prediction error on the widest model "
          f"mean {err.mean():+.4f}, max |err| {np.abs(err).max():.4f} nats")
    pq = res["ptq"]
    D = np.array([r["D"] for r in pq], dtype=np.float64)
    for b in (4, 3):
        d = np.array([r[f"delta_int{b}"] for r in pq])
        line = ", ".join(f"D={int(x):,}: {y:+.4f} [{r[f'delta_int{b}_ci'][0]:+.4f}, {r[f'delta_int{b}_ci'][1]:+.4f}]"
                         for x, y, r in zip(D, d, pq))
        if (d > 0).all():
            print(f"Part B INT{b}: {line}\n   delta ~ D^p with p = {lab.ptq_exponent(D, d):.3f} (paper: gamma_D = 0.5068)")
        else:
            print(f"Part B INT{b}: {line}\n   some deltas are <= 0: no power law can be fitted")


if __name__ == "__main__":
    main()
