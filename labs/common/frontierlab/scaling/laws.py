"""Scaling-law formulas with published constants (lesson 11.1).

Everything is plain NumPy in float64 so the numbers can be checked by hand. Symbols:

* ``N``: parameters (which parameters is a choice: non-embedding, as Kaplan et al. count, or all of them,
  as Hoffmann et al. count; lesson 11.1 shows why the choice matters at small scale);
* ``D``: training tokens; ``C``: training FLOPs, ``C ≈ k·N·D`` with ``k = 6`` (forward 2 + backward 4
  FLOPs per parameter per token, parent course lesson 07.4);
* the Chinchilla form (Hoffmann et al., arXiv 2203.15556, Eq. 2 / section 3.3):

      L(N, D) = E + A / N^α + B / D^β

  ``E`` is the loss of a perfect model on this data ("entropy of natural text"), the two power laws are what a
  finite model and a finite dataset add.

Minimising ``L`` at fixed ``C = k·N·D`` gives (Hoffmann et al., Eq. 4, written out):

      N_opt(C) = G · (C/k)^(β/(α+β)),   D_opt(C) = G^-1 · (C/k)^(α/(α+β)),   G = (α·A / (β·B))^(1/(α+β))

Data-constrained training (Muennighoff et al., arXiv 2305.16264, section 3): with ``U`` unique tokens seen
``1 + R`` times, the repeated tokens are worth less; the effective data is

      D' = U + U · R* · (1 - exp(-R / R*))

so the first few repeats count almost fully and returns vanish after about ``R*`` repeats.

Inference-aware sizing (Sardana et al., arXiv 2401.00448): to reach a target loss and then serve ``D_inf``
tokens, minimise total FLOPs ``6·N·D + 2·N·D_inf`` over ``N`` with ``D`` the tokens needed at that ``N``.
"""

from __future__ import annotations

import math

import numpy as np

# Hoffmann et al. 2022, approach 3 (section 3.3, Eq. 10): fitted on their runs; Chinchilla's N counts all parameters.
CHINCHILLA = {"E": 1.69, "A": 406.4, "B": 410.7, "alpha": 0.34, "beta": 0.28}
# Besiroglu et al. 2024 (arXiv 2404.10102), refit of approach 3 on data reconstructed from Hoffmann et al.'s Figure 4.
BESIROGLU = {"E": 1.8172, "A": 482.01, "B": 2085.43, "alpha": 0.3478, "beta": 0.3658}
# Muennighoff et al. 2023 (arXiv 2305.16264), fitted decay constants of repeated data and excess parameters.
MUENNIGHOFF = {"R_D_star": 15.387756, "R_N_star": 5.309743}   # appendix A fit


def loss(N, D, c=CHINCHILLA) -> np.ndarray:
    """L(N, D) = E + A·N^-α + B·D^-β."""
    N, D = np.asarray(N, dtype=np.float64), np.asarray(D, dtype=np.float64)
    return c["E"] + c["A"] * N ** -c["alpha"] + c["B"] * D ** -c["beta"]


def compute_optimal(C, c=CHINCHILLA, k: float = 6.0) -> dict:
    """Closed-form N_opt, D_opt, tokens per parameter and the loss at that point, for C FLOPs (C = k·N·D)."""
    C = np.asarray(C, dtype=np.float64)
    a, b = c["alpha"], c["beta"]
    G = (a * c["A"] / (b * c["B"])) ** (1.0 / (a + b))
    N = G * (C / k) ** (b / (a + b))
    D = (C / k) / N
    return {"N": N, "D": D, "tokens_per_param": D / N, "loss": loss(N, D, c),
            "exp_N": b / (a + b), "exp_D": a / (a + b), "G": G}


def optimal_loss(C, c=CHINCHILLA, k: float = 6.0) -> np.ndarray:
    return compute_optimal(C, c, k)["loss"]


def compute_for_loss(L_target: float, c=CHINCHILLA, k: float = 6.0) -> float:
    """The smallest C whose compute-optimal model reaches ``L_target`` (bisection in log C); inf if L ≤ E."""
    if L_target <= c["E"]:
        return math.inf
    lo, hi = 0.0, 40.0                               # log10 C
    if optimal_loss(10.0 ** lo, c, k) < L_target:
        return 10.0 ** lo
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if optimal_loss(10.0 ** mid, c, k) > L_target else (lo, mid)
    return 10.0 ** ((lo + hi) / 2)


def overtraining(N: float, D: float, c=CHINCHILLA, k: float = 6.0) -> dict:
    """What a model of N parameters trained on D tokens gives up against the compute-optimal model.

    Returns the spent compute ``C = k·N·D``, the loss reached, the compute-optimal loss at the same C (lower),
    the compute a compute-optimal run would need to reach the same loss, and ``overhead`` = C / that − 1
    (0.5 = this run spent 50% more training compute than necessary for its loss; it buys a smaller model).
    """
    C = k * N * D
    L = float(loss(N, D, c))
    opt = compute_optimal(C, c, k)
    C_same = compute_for_loss(L, c, k)
    return {"C": C, "loss": L, "tokens_per_param": D / N, "loss_opt_same_C": float(opt["loss"]),
            "N_opt_same_C": float(opt["N"]), "C_opt_same_loss": C_same, "overhead": C / C_same - 1.0}


def tokens_for_loss(N, L_target: float, c=CHINCHILLA) -> np.ndarray:
    """D needed by a model of N parameters to reach ``L_target``; inf where that model can never reach it."""
    N = np.asarray(N, dtype=np.float64)
    gap = L_target - c["E"] - c["A"] * N ** -c["alpha"]
    with np.errstate(divide="ignore", invalid="ignore"):
        D = np.where(gap > 0, (c["B"] / np.where(gap > 0, gap, 1.0)) ** (1.0 / c["beta"]), np.inf)
    return D


def inference_aware(L_target: float, D_inference: float, c=CHINCHILLA, logN=None) -> dict:
    """Sardana et al.: minimise 6·N·D(N) + 2·N·D_inference over N at a fixed target loss (grid in log10 N)."""
    logN = np.linspace(6, 13, 7001) if logN is None else np.asarray(logN, dtype=np.float64)
    N = 10.0 ** logN
    D = tokens_for_loss(N, L_target, c)
    train = 6 * N * D
    total = train + 2 * N * D_inference
    i = int(np.argmin(total))
    j = int(np.argmin(train))
    return {"N": float(N[i]), "D": float(D[i]), "tokens_per_param": float(D[i] / N[i]), "total_flops": float(total[i]),
            "train_flops": float(train[i]), "N_train_only": float(N[j]), "D_train_only": float(D[j]),
            "total_flops_if_train_only_optimal": float(total[j])}


def effective_data(U, D_total, R_star: float = MUENNIGHOFF["R_D_star"]) -> np.ndarray:
    """Muennighoff et al.: tokens' worth of D_total tokens drawn from U unique tokens (R = D_total/U − 1 repeats)."""
    U, D_total = np.asarray(U, dtype=np.float64), np.asarray(D_total, dtype=np.float64)
    R = np.maximum(D_total / U - 1.0, 0.0)
    return np.where(D_total <= U, D_total, U + U * R_star * (1.0 - np.exp(-R / R_star)))


def six_nd(N: float, D: float) -> float:
    """The 6·N·D approximation of training FLOPs."""
    return 6.0 * N * D
