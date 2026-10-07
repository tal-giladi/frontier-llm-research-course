"""Reference solution for lab 19.2."""

from __future__ import annotations

import math

import numpy as np

from frontierlab.research.reproduce import Deviation, t_interval

# Stated before the runs (2026-10-07). The noise floor is lesson 01.4's measured seed std of the toy recipe's
# held-out loss (5 seeds, lr 3e-3, 300 steps of 16 x 128 tokens), so it exists before any run of this lab.
NOISE_FLOOR = 0.0286
TOLERANCE = 2 * NOISE_FLOOR
STATED_ON = "2026-10-07"


def lr_sensitivity(losses: dict, l0: float) -> float:
    vals = np.array([l0 if not math.isfinite(float(v)) else min(float(v), l0) for v in losses.values()])
    return float(vals.mean() - vals.min())


def seed_effects(sens_without: dict, sens_with: dict) -> list[float]:
    return [sens_without[s] - sens_with[s] for s in sorted(sens_without)]


def decide(effects, tolerance, published=None, magnitude_tol=None) -> dict:
    mean, lo, hi = t_interval(effects)
    if lo > 0 and mean >= tolerance:
        direction = "reproduced"
    elif hi < 0:
        direction = "contradicted"
    elif hi < tolerance:
        direction = "not reproduced"
    else:
        direction = "inconclusive"
    if published is None or magnitude_tol is None:
        magnitude = "not comparable"
    else:
        magnitude = "matches" if abs(mean - published) <= magnitude_tol else "differs"
    return {"direction": direction, "magnitude": magnitude, "mean": mean, "ci": (lo, hi), "n": len(effects),
            "tolerance": tolerance, "published": published, "magnitude_tol": magnitude_tol}


def deviations() -> list[Deviation]:
    return [
        Deviation("scale", "a range of sizes up to billions of parameters (Figure 1; 1.2B in section 3.1.1)",
                  "toy preset, 1.8M parameters, one size", "free CPU budget",
                  "unknown", note=""),
        Deviation("data", "C4", "Data-v0 CPU size (FineWeb-Edu sample, 8,192-token BPE)",
                  "the course data; C4 is not prepared", "unknown"),
        Deviation("tokens", "1e5 steps x 256 x 512 tokens", "300 steps x 16 x 128 tokens (0.6M)",
                  "free CPU budget", "weakens",
                  note=""),
        Deviation("optimizer", "AdamW b1 0.9, b2 0.95, eps 1e-8, clip 1.0, independent decay 1e-4, z-loss 1e-4, "
                  "warm-up 5% then cosine to 1e-5",
                  "same, with decay passed as 1e-4/lr (torch multiplies decay by the lr), cosine to 0.1 x peak, not to 1e-5",
                  "matched what is cheap to match", "none expected"),
        Deviation("learning rates", "3e-4 to 3e-1, 7 values", "3e-3, 1e-2, 3e-2",
                  "three CPU runs per arm and seed", "weakens"),
        Deviation("evaluation", "final validation loss; l0 the loss at initialisation",
                  "final loss on 128 fixed validation windows; l0 = ln(vocab)",
                  "the course's held-out windows; ln V is the loss of a uniform prediction", "none expected"),
        Deviation("seeds", "not stated per point in Figure 1", "3 seeds per arm and learning rate",
                  "the decision needs a seed interval", "none expected"),
        Deviation("code", "the authors' JAX code (not used)", "frontierlab.optim.train with --qk-norm on/off",
                  "the course loop; QK-norm is RMSNorm here, LayerNorm in the paper", "unknown"),
    ]
