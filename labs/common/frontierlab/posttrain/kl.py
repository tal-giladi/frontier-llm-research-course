"""KL regularisation toward a reference policy: estimators, where they go, and what gradient they give
(lesson 12.2).

Per token, with the response sampled from the current policy pi and delta = log pi(y_t) - log pi_ref(y_t):

* k1 = delta                          unbiased for KL(pi || ref), high variance, negative on some samples
* k2 = delta^2 / 2                    biased, low variance, >= 0
* k3 = exp(-delta) - 1 + delta        unbiased, low variance, >= 0  (GRPO's estimator, DeepSeekMath Eq. 4)

(Schulman, "Approximating KL Divergence", 2020, with r = ref/pi.)

**Two placements.**

* *In the reward* (InstructGPT Eq. 2, PPO-style RLHF): the value of the estimator is detached and
  subtracted from the reward, ``r - beta * sum_t k1_t``, and goes through the advantage like any other
  reward. The policy-gradient machinery then produces the gradient of -beta * KL(pi || ref).
* *In the loss* (GRPO Eq. 3): the estimator is a differentiable term added to the loss and
  differentiated directly. Then what matters is not whether the *value* is unbiased but what its
  *gradient* is in expectation. For on-policy samples (Tang & Munos 2025, section 3):

  - grad k1 has expectation E_pi[grad log pi] = 0: zero-mean noise, no regularisation at all;
  - grad k2 has expectation grad KL(pi || ref): the intended reverse-KL gradient;
  - grad k3 has expectation grad KL(ref || pi): the *forward* KL, a different regulariser.

:func:`exact_categorical` checks all of this exactly on a categorical distribution, by enumeration.
"""

from __future__ import annotations

import torch


def estimators(logp: torch.Tensor, ref_logp: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-token k1, k2, k3 (same shape as the inputs); gradients flow through ``logp``."""
    d = logp - ref_logp
    return {"k1": d, "k2": 0.5 * d * d, "k3": torch.exp(-d) - 1.0 + d}


def kl_penalty_reward(logp: torch.Tensor, ref_logp: torch.Tensor, mask: torch.Tensor, beta: float,
                      kind: str = "k1") -> torch.Tensor:
    """(B,) detached per-sequence penalty ``beta * sum_t k_t`` to subtract from the outcome reward."""
    k = estimators(logp.detach(), ref_logp.detach())[kind]
    return beta * (k * mask).sum(-1)


def kl_loss_terms(logp: torch.Tensor, ref_logp: torch.Tensor, kind: str = "k3") -> torch.Tensor:
    """Per-token differentiable KL term for the loss (aggregate it like the policy loss)."""
    return estimators(logp, ref_logp.detach())[kind]


def exact_categorical(logits: torch.Tensor, ref_logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Exact expected gradients (w.r.t. ``logits``) of each estimator used as a loss, plus the true
    gradients of the reverse and forward KL, for one categorical distribution. Float64.

    E_{y~pi}[grad_theta k(y)] is computed by enumerating every y, with the sample treated as fixed
    (no gradient through the sampling), which is what autograd does with a sampled batch.
    """
    theta = logits.detach().double().requires_grad_(True)
    ref = torch.log_softmax(ref_logits.detach().double(), -1)
    out = {}
    logp = torch.log_softmax(theta, -1)
    p = logp.exp().detach()
    for name in ("k1", "k2", "k3"):
        k = estimators(logp, ref)[name]
        g = torch.zeros_like(theta)
        for y in range(theta.shape[-1]):
            gy, = torch.autograd.grad(k[y], theta, retain_graph=True)
            g = g + p[y] * gy
        out[name] = g
    rev = (logp.exp() * (logp - ref)).sum()
    out["grad_reverse_kl"], = torch.autograd.grad(rev, theta, retain_graph=True)
    fwd = (ref.exp() * (ref - logp)).sum()
    out["grad_forward_kl"], = torch.autograd.grad(fwd, theta)
    out["reverse_kl"] = rev.detach()
    out["forward_kl"] = fwd.detach()
    return out
