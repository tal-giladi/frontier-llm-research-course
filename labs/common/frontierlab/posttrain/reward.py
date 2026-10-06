"""Learned rewards (lesson 12.1): Bradley-Terry reward models, calibration, best-of-n and over-optimisation.

**Reward model.** The policy architecture with the LM head replaced by a scalar head that reads the final
hidden state at the EOS position: ``r(y)`` (B,) float32. Trained on pairs (chosen, rejected) with the
Bradley-Terry loss  -log sigmoid(r(chosen) - r(rejected))  (InstructGPT Eq. 1 with K = 2).

**Best-of-n.** Draw n responses from the base policy and keep the one the reward model scores highest.
Its KL from the base policy is exactly  log n - (n - 1)/n  (Gao et al. 2022 section 2, after Stiennon et
al. 2020), so BoN gives an optimisation curve with a known x-axis and no RL noise.

:func:`bon_expected` is the unbiased estimate of E[score of the BoN pick] from one pool of N >= n samples
(every n-subset of the pool, weighted exactly): sort by the proxy; the i-th smallest (1-based) is the
pick of a random n-subset with probability C(i-1, n-1)/C(N, n).

**Over-optimisation fits** (Gao et al. 2022 section 1): with d = sqrt(KL),
gold_bon(d) = d (alpha - beta d) and gold_rl(d) = d (alpha - beta log d), rewards measured relative to
the base policy's mean.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from frontierlab.model import LM
from frontierlab.posttrain.sft import policy_config


class RewardModel(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.lm = LM(cfg or policy_config())
        self.lm.lm_head = None                       # the backbone only; tied head weights stay in embed_tokens
        self.score = nn.Linear(self.lm.config.hidden_size, 1)
        nn.init.normal_(self.score.weight, std=0.02)
        nn.init.zeros_(self.score.bias)

    def forward(self, ids: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        """ids (B, T) int64, last (B,) index of the EOS token -> scores (B,) float32."""
        m = self.lm.model
        T = ids.shape[1]
        x = m.embed_tokens(ids)
        positions = torch.arange(T, device=ids.device)
        for layer in m.layers:
            x = layer(x, positions)
        h = m.norm(x)[torch.arange(ids.shape[0], device=ids.device), last]
        return self.score(h).squeeze(-1).float()


def bt_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry negative log-likelihood, mean over pairs."""
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def make_pairs(world, n: int, rng: np.random.Generator, temperature: float = 0.5):
    """``n`` preference pairs from the base sampler, labelled by the noisy annotator: (chosen, rejected)."""
    a, b = world.sample(n, rng), world.sample(n, rng)
    chosen, rejected = [], []
    for x, y in zip(a, b):
        if world.preference(x, y, rng, temperature):
            chosen.append(x); rejected.append(y)
        else:
            chosen.append(y); rejected.append(x)
    return chosen, rejected


@torch.no_grad()
def score_strings(rm: RewardModel, strings: list[str], max_len: int, batch: int = 2048) -> np.ndarray:
    from frontierlab.posttrain.tasks import encode_strings
    rm.eval()
    out = []
    for i in range(0, len(strings), batch):
        ids, last = encode_strings(strings[i:i + batch], max_len)
        out.append(rm(ids, last).numpy())
    return np.concatenate(out).astype(np.float64)


def train_rm(world, n_pairs: int, steps: int = 400, batch: int = 64, lr: float = 2e-3, seed: int = 0,
             temperature: float = 0.5, cfg=None, log_every: int = 0) -> tuple[RewardModel, dict]:
    """Train a reward model on ``n_pairs`` labelled pairs (several epochs if steps*batch > n_pairs)."""
    from frontierlab.posttrain.tasks import encode_strings
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    chosen, rejected = make_pairs(world, n_pairs, rng, temperature)
    ci, cl = encode_strings(chosen, world.max_len)
    ri, rl = encode_strings(rejected, world.max_len)
    rm = RewardModel(cfg)
    opt = torch.optim.AdamW(rm.parameters(), lr=lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    hist = []
    for step in range(1, steps + 1):
        idx = torch.randint(0, n_pairs, (batch,), generator=g)
        loss = bt_loss(rm(ci[idx], cl[idx]), rm(ri[idx], rl[idx]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log_every and step % log_every == 0:
            hist.append({"step": step, "loss": float(loss)})
            print(f"rm step {step} loss {loss.item():.4f}")
    rm.eval()
    return rm, {"n_pairs": n_pairs, "steps": steps, "history": hist}


def preference_metrics(rm: RewardModel, world, n: int = 4000, seed: int = 99, temperature: float = 0.5,
                       bins: int = 10) -> dict:
    """Held-out pair accuracy (against the noisy labels and against the noise-free gold order), mean BT
    loss, and calibration: expected calibration error of P(chosen) = sigmoid(r_c - r_r)."""
    rng = np.random.default_rng(seed)
    a, b = world.sample(n, rng), world.sample(n, rng)
    labels = np.array([world.preference(x, y, rng, temperature) for x, y in zip(a, b)], dtype=np.float64)
    ra, rb = score_strings(rm, a, world.max_len), score_strings(rm, b, world.max_len)
    p = 1.0 / (1.0 + np.exp(-(ra - rb)))
    ga = np.array([world.gold(x) for x in a]); gb = np.array([world.gold(y) for y in b])
    decided = ga != gb
    acc_label = float(((p > 0.5) == (labels == 1)).mean())
    acc_gold = float(((ra > rb) == (ga > gb))[decided].mean())
    nll = float(-(labels * np.log(p + 1e-12) + (1 - labels) * np.log(1 - p + 1e-12)).mean())
    return {"acc_vs_labels": acc_label, "acc_vs_gold": acc_gold, "nll": nll, "ece": ece(p, labels, bins),
            "reliability": reliability(p, labels, bins)}


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    """Per bin of predicted probability: mean prediction, observed frequency, count."""
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if sel.any():
            out.append({"lo": float(lo), "hi": float(hi), "mean_p": float(p[sel].mean()),
                        "freq": float(y[sel].mean()), "n": int(sel.sum())})
    return out


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: sum over bins of (n_bin / n) * |mean prediction - observed frequency|."""
    rel = reliability(np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64), bins)
    n = sum(r["n"] for r in rel)
    return float(sum(r["n"] / n * abs(r["mean_p"] - r["freq"]) for r in rel))


def bon_kl(n) -> np.ndarray:
    """KL(best-of-n || base) = log n - (n - 1)/n, in nats."""
    n = np.asarray(n, dtype=np.float64)
    return np.log(n) - (n - 1) / n


def bon_expected(proxy: np.ndarray, value: np.ndarray, n: int) -> float:
    """Unbiased E[value of the best-of-n pick by ``proxy``] from a pool of N samples (N >= n)."""
    proxy, value = np.asarray(proxy, dtype=np.float64), np.asarray(value, dtype=np.float64)
    N = proxy.shape[0]
    if n > N:
        raise ValueError("n larger than the pool")
    order = np.argsort(proxy, kind="stable")
    i = np.arange(1, N + 1)                      # 1-based rank, ascending in proxy
    with np.errstate(invalid="ignore"):
        logw = np.where(i >= n, _lgamma_comb(i - 1, n - 1) - _lgamma_comb(np.full(N, N), n), -np.inf)
    w = np.exp(logw)
    return float((w * value[order]).sum())


def _lgamma_comb(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64) * np.ones_like(a)
    from scipy.special import gammaln
    return gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)


def fit_gao(d: np.ndarray, r: np.ndarray, kind: str = "bon") -> dict:
    """Least-squares fit of Gao et al.'s forms (no intercept: r(0) = 0). Returns alpha, beta, the argmax d."""
    d, r = np.asarray(d, dtype=np.float64), np.asarray(r, dtype=np.float64)
    if kind == "bon":
        X = np.stack([d, -d * d], 1)
    elif kind == "rl":
        X = np.stack([d, -d * np.log(np.maximum(d, 1e-12))], 1)
    else:
        raise ValueError(kind)
    (alpha, beta), *_ = np.linalg.lstsq(X, r, rcond=None)
    if kind == "bon":
        d_star = alpha / (2 * beta) if beta > 0 else math.inf
    else:
        d_star = math.exp(alpha / beta - 1) if beta > 0 else math.inf
    pred = X @ np.array([alpha, beta])
    ss = float(((r - r.mean()) ** 2).sum())
    return {"alpha": float(alpha), "beta": float(beta), "d_star": float(d_star),
            "r2": float(1 - ((r - pred) ** 2).sum() / ss) if ss > 0 else float("nan")}
