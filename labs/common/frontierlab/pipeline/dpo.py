"""Direct Preference Optimization from scratch (lesson 13.1; used again in 13.3 and the project).

For a prompt x with a chosen response y_c and a rejected one y_r, a policy pi_theta and a frozen
reference pi_ref (the checkpoint DPO starts from), define the *implicit reward*

    r_hat(x, y) = beta * log( pi_theta(y | x) / pi_ref(y | x) )

and the DPO loss (Rafailov et al. 2023, section 4, Eq. 7), a Bradley-Terry likelihood on r_hat:

    L = - log sigmoid( r_hat(x, y_c) - r_hat(x, y_r) )

Its gradient (their section 4, "What does the DPO update do?") is

    dL/dtheta = - beta * sigmoid( r_hat(x, y_r) - r_hat(x, y_c) ) * [ grad log pi(y_c|x) - grad log pi(y_r|x) ]

so each pair is weighted by how wrongly the implicit reward currently orders it.

Variants used by the open recipes (lesson 13.1):

* ``normalise=True`` — length-normalised DPO (Tülu 3, section 5.1.2, Eq. 6): each log-ratio is divided by
  the length of its response, so long responses do not get a larger implicit reward for being long.
  Tülu 3's final 8B used beta = 5 with this form (Table 20), because per-token log-ratios are small.
* ``nll_coef > 0`` — Llama 3 (section 4.1.4) adds a negative log-likelihood term on the chosen response with
  coefficient 0.2, to stop the probability of the chosen response from falling.
* ``mask`` — Llama 3 also masks formatting tokens (headers, termination) out of both responses; pass a
  mask that is 0 on them.

All per-pair tensors are (B,) float32: B pairs.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from frontierlab.pipeline.compute import Ledger, n_params
from frontierlab.pipeline.seqs import Example, batches, pad, sequence_logps


def dpo_loss(pol_c: torch.Tensor, pol_r: torch.Tensor, ref_c: torch.Tensor, ref_r: torch.Tensor, beta: float,
             len_c: torch.Tensor | None = None, len_r: torch.Tensor | None = None, normalise: bool = False,
             nll_coef: float = 0.0, label: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """DPO loss (mean over pairs) from sequence log-probabilities (B,) and its diagnostics.

    ``label`` (B,) in [0, 1] (optional): soft preference targets, P(y_c is preferred). 1 everywhere is
    ordinary DPO; 0.5 is "no preference"; a judge's probability gives soft-label DPO (as with the soft
    AI-feedback labels of Constitutional AI, section 4.3).
    """
    lc = pol_c - ref_c
    lr = pol_r - ref_r
    if normalise:
        if len_c is None or len_r is None:
            raise ValueError("length-normalised DPO needs the response lengths")
        lc, lr = lc / len_c.clamp_min(1.0), lr / len_r.clamp_min(1.0)
    rc, rr = beta * lc, beta * lr
    h = rc - rr
    if label is None:
        loss = -F.logsigmoid(h)
    else:
        loss = -(label * F.logsigmoid(h) + (1 - label) * F.logsigmoid(-h))
    loss = loss.mean()
    if nll_coef > 0:
        n = len_c.clamp_min(1.0) if len_c is not None else torch.ones_like(pol_c)
        loss = loss + nll_coef * (-(pol_c / n)).mean()
    with torch.no_grad():
        diag = {"reward_acc": float((h > 0).float().mean()), "margin": float(h.mean()),
                "chosen_reward": float(rc.mean()), "rejected_reward": float(rr.mean()),
                "chosen_logp_change": float((pol_c - ref_c).mean()),
                "rejected_logp_change": float((pol_r - ref_r).mean()),
                "grad_weight": float(torch.sigmoid(-h).mean())}
    return loss, diag


def pair_batch(pairs: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Chosen rows first, then rejected rows, in one padded batch (2B, T)."""
    return pad([c for c, _ in pairs] + [r for _, r in pairs])


@torch.no_grad()
def reference_logps(ref, pairs: list, batch: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference log-probabilities of every chosen and rejected response, computed once before training."""
    ref.eval()
    out_c, out_r = [], []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        ids, mask = pair_batch(chunk)
        lp, _ = sequence_logps(ref, ids, mask)
        out_c.append(lp[:len(chunk)])
        out_r.append(lp[len(chunk):])
    return torch.cat(out_c), torch.cat(out_r)


def train_dpo(policy, ref, pairs: list, steps: int, beta: float = 0.1, batch: int = 64, lr: float = 1e-4,
              normalise: bool = False, nll_coef: float = 0.0, labels: torch.Tensor | None = None, seed: int = 0,
              warmup: int = 10, grad_clip: float = 1.0, ledger: Ledger | None = None, log=None,
              log_every: int = 25) -> dict:
    """Offline DPO on fixed ``pairs`` [(chosen Example, rejected Example), ...] starting from ``policy``,
    with ``ref`` frozen. Charges the ledger 2 N per reference token (once) and 6 N per policy token."""
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    ref_c, ref_r = reference_logps(ref, pairs)
    N = n_params(policy)
    if ledger is not None:
        toks = float(sum(len(c) + len(r) for c, r in pairs))
        ledger.forward("reference", n_params(ref), toks)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    policy.train()
    t0, hist = time.perf_counter(), []
    for step, idx in enumerate(batches(len(pairs), batch, steps, gen), start=1):
        chunk = [pairs[i] for i in idx.tolist()]
        ids, mask = pair_batch(chunk)
        lp, n = sequence_logps(policy, ids, mask)
        B = len(chunk)
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, step / max(1, warmup))
        loss, d = dpo_loss(lp[:B], lp[B:], ref_c[idx], ref_r[idx], beta, n[:B], n[B:], normalise, nll_coef,
                           labels[idx] if labels is not None else None)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        opt.step()
        if ledger is not None:
            ledger.train("student_train", N, float((ids != 0).sum()))
        if step % log_every == 0 or step == steps:
            row = {"stage": "dpo", "step": step, "loss": float(loss.detach()), **d,
                   "seconds": round(time.perf_counter() - t0, 2)}
            hist.append(row)
            if log is not None:
                log.log(**row)
    policy.eval()
    return {"steps": steps, "history": hist, "final": hist[-1] if hist else {},
            "seconds": round(time.perf_counter() - t0, 2)}


def implied_policy(ref_probs: torch.Tensor, reward: torch.Tensor, beta: float) -> torch.Tensor:
    """The KL-regularised optimum DPO targets (Rafailov et al. Eq. 4): pi*(y) proportional to
    pi_ref(y) exp(r(y) / beta), over a finite set of responses (a check used in the tests and lesson 13.1)."""
    w = ref_probs * torch.exp(reward / beta)
    return w / w.sum()
