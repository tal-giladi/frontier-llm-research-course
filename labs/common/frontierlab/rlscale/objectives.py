"""Five RL objectives for language models, each derived and implemented on its own (lesson 14.1).

Shared conventions (the same as :mod:`frontierlab.posttrain.losses`): per-token tensors are (B, R) with
B = P * G responses in group-major order and R response positions; ``logp`` is the current policy's
log-probability of each sampled token (gradients flow through it), ``old_logp`` the behaviour policy's
(no gradient), ``adv`` one advantage per response (B,) or per token (B, R), ``mask`` (B, R) float, 1 on
response tokens up to and including EOS. Every function returns ``(loss_to_minimise, diagnostics)``.

Notation: $r_{i,t} = \\pi_\\theta(y_{i,t}) / \\pi_{old}(y_{i,t})$ is the token ratio, $|y_i|$ the response length,
$A_i$ the advantage.

================  ==========================================================================  ==================
objective         per-token gradient weight on $\\nabla \\log \\pi_\\theta(y_{i,t})$ (times $-1$ in the loss)    source
================  ==========================================================================  ==================
``grpo``          $\\frac{1}{B}\\frac{1}{|y_i|}\\, r_{i,t} A_i$ unless the token is clipped            DeepSeekMath Eq. 3
``dapo``          $\\frac{1}{\\sum_j |y_j|}\\, r_{i,t} A_i$ unless clipped (asymmetric range)          DAPO Eq. 8
``drgrpo``        $\\frac{1}{B\\,L}\\, r_{i,t} A_i$ unless clipped, $L$ a constant                    Dr. GRPO sec. 3.2
``gspo``          $\\frac{1}{B}\\frac{1}{|y_i|}\\, s_i A_i$ unless the *sequence* is clipped           GSPO sec. 4
``cispo``         $\\frac{1}{\\sum_j |y_j|}\\, \\mathrm{clip}(r_{i,t}) A_i$ for every token              MiniMax-M1 sec. 3.1
================  ==========================================================================  ==================

with $s_i = \\exp\\big(\\frac{1}{|y_i|}\\sum_t \\log r_{i,t}\\big)$ GSPO's length-normalised sequence ratio. On the
first, on-policy minibatch every ratio is 1, so ``grpo`` and ``gspo`` give the same gradient (per-sequence
mean of the REINFORCE terms) and ``dapo`` and ``cispo`` give the same gradient (token mean): the objectives
differ only in how they treat samples once the policy has moved, and in how they weigh tokens.

Each loss also has the keyword interface of the course loop's ``policy_loss`` hook
(:func:`frontierlab.posttrain.rl.train`), through :func:`as_hook`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch


def _adv2(adv: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return adv[:, None].expand_as(like) if adv.dim() == 1 else adv


def _token_ppo(logp, old_logp, adv, mask, eps_low, eps_high):
    """Per-token PPO surrogate and the 'gradient is zero here' flag."""
    ratio = torch.exp(logp - old_logp.detach())
    a = _adv2(adv, ratio)
    unclipped = ratio * a
    clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * a
    obj = torch.minimum(unclipped, clipped)
    zero_grad = (clipped < unclipped).float()
    return obj, ratio, zero_grad


def _diag(ratio, flag, mask, extra=None):
    with torch.no_grad():
        m = mask.float()
        n = m.sum().clamp_min(1.0)
        d = {"clip_frac": float((flag * m).sum() / n), "ratio_mean": float((ratio.detach() * m).sum() / n),
             "ratio_max": float((ratio.detach() * m).max()) if m.sum() > 0 else 1.0}
        d.update(extra or {})
    return d


# --------------------------------------------------------------------------- GRPO

def grpo_loss(logp, old_logp, adv, mask, eps: float = 0.2):
    """GRPO (Shao et al. 2024, DeepSeekMath Eq. 3) without its KL term:

    $J = \\frac{1}{G}\\sum_i \\frac{1}{|o_i|}\\sum_t \\min(r_{i,t} A_i, \\mathrm{clip}(r_{i,t}, 1-\\epsilon, 1+\\epsilon) A_i)$.

    The KL to the reference is a separate, optional term (the comparison of lesson 14.2 holds it fixed).
    """
    obj, ratio, flag = _token_ppo(logp, old_logp, adv, mask, eps, eps)
    m = mask.float()
    per_seq = (obj * m).sum(-1) / m.sum(-1).clamp_min(1.0)
    return -per_seq.mean(), _diag(ratio, flag, mask)


# --------------------------------------------------------------------------- DAPO

def dapo_loss(logp, old_logp, adv, mask, eps_low: float = 0.2, eps_high: float = 0.28):
    """DAPO's policy loss (Yu et al. 2025, Eq. 8): token-level mean over the whole batch, clip-higher.

    $J = \\frac{1}{\\sum_i |o_i|}\\sum_i\\sum_t \\min(r_{i,t} A_i, \\mathrm{clip}(r_{i,t}, 1-\\epsilon_{low}, 1+\\epsilon_{high}) A_i)$.

    Dynamic sampling (keep only groups with 0 < correct < G) and overlong shaping are batch and reward
    choices, set in the loop (``zero_var="filter"``, ``overlong="soft"``), not in the loss.
    """
    obj, ratio, flag = _token_ppo(logp, old_logp, adv, mask, eps_low, eps_high)
    m = mask.float()
    return -(obj * m).sum() / m.sum().clamp_min(1.0), _diag(ratio, flag, mask)


# --------------------------------------------------------------------------- Dr. GRPO

def drgrpo_loss(logp, old_logp, adv, mask, eps: float = 0.2, norm_len: int | None = None):
    """Dr. GRPO (Liu et al. 2025, section 3.2): GRPO with $1/|o_i|$ replaced by a constant.

    $J = \\frac{1}{G\\,L}\\sum_i\\sum_t \\min(r_{i,t} A_i, \\mathrm{clip}(r_{i,t}) A_i)$, $L$ = the generation budget
    (``norm_len``, default R). The other half of Dr. GRPO, no division by the group's reward std, is in the
    advantage (``scale="none"``).
    """
    obj, ratio, flag = _token_ppo(logp, old_logp, adv, mask, eps, eps)
    m = mask.float()
    L = norm_len if norm_len is not None else logp.shape[1]
    return -(obj * m).sum() / (logp.shape[0] * L), _diag(ratio, flag, mask)


# --------------------------------------------------------------------------- GSPO

def sequence_log_ratio(logp, old_logp, mask):
    """(B,) $\\log s_i = \\frac{1}{|y_i|}\\sum_t (\\log\\pi_\\theta - \\log\\pi_{old})$; gradient flows through logp."""
    m = mask.float()
    return ((logp - old_logp.detach()) * m).sum(-1) / m.sum(-1).clamp_min(1.0)


def gspo_loss(logp, old_logp, adv, mask, eps_low: float = 3e-4, eps_high: float = 4e-4):
    """GSPO (Zheng et al. 2025, section 4): one length-normalised ratio per sequence, clipped per sequence.

    $s_i = (\\pi_\\theta(y_i)/\\pi_{old}(y_i))^{1/|y_i|}$ (the geometric mean of the token ratios),
    $J = \\frac{1}{G}\\sum_i \\min(s_i A_i, \\mathrm{clip}(s_i, 1-\\epsilon_{low}, 1+\\epsilon_{high}) A_i)$.
    $\\nabla s_i = s_i \\frac{1}{|y_i|}\\sum_t \\nabla\\log\\pi_\\theta(y_{i,t})$: every token of a sequence gets the same
    weight $s_i A_i / |y_i|$, and a clipped sequence contributes no gradient at all.
    ``adv`` must be (B,): GSPO is sequence-level by construction (use :func:`gspo_token_loss` otherwise).
    """
    if adv.dim() != 1:
        raise ValueError("gspo_loss needs one advantage per sequence; use gspo_token_loss for (B, R)")
    s = torch.exp(sequence_log_ratio(logp, old_logp, mask))
    unclipped = s * adv
    clipped = torch.clamp(s, 1.0 - eps_low, 1.0 + eps_high) * adv
    obj = torch.minimum(unclipped, clipped)
    flag = (clipped < unclipped).float()
    return -obj.mean(), _diag(s[:, None].expand_as(logp), flag[:, None].expand_as(logp), mask)


def gspo_token_loss(logp, old_logp, adv, mask, eps_low: float = 3e-4, eps_high: float = 4e-4):
    """GSPO-token (GSPO section 4.3): $s_{i,t} = \\mathrm{sg}[s_i]\\,\\pi_\\theta(y_{i,t}) / \\mathrm{sg}[\\pi_\\theta(y_{i,t})]$.

    Its value equals $s_i$ for every token and its gradient is $s_i \\nabla\\log\\pi_\\theta(y_{i,t})$, so with one
    advantage per sequence it is GSPO exactly; it exists so that token-level advantages can be used.
    """
    s = torch.exp(sequence_log_ratio(logp, old_logp, mask)).detach()
    st = s[:, None] * torch.exp(logp - logp.detach())
    a = _adv2(adv, st)
    unclipped = st * a
    clipped = torch.clamp(st, 1.0 - eps_low, 1.0 + eps_high) * a
    obj = torch.minimum(unclipped, clipped)
    flag = (clipped < unclipped).float()
    m = mask.float()
    per_seq = (obj * m).sum(-1) / m.sum(-1).clamp_min(1.0)
    return -per_seq.mean(), _diag(st, flag, mask)


# --------------------------------------------------------------------------- CISPO

def cispo_loss(logp, old_logp, adv, mask, eps_low: float | None = None, eps_high: float = 0.28):
    """CISPO (MiniMax-M1, section 3.1): clip the importance-sampling *weight*, never drop a token.

    $J = \\frac{1}{\\sum_i |o_i|}\\sum_i\\sum_t \\mathrm{sg}(\\hat r_{i,t})\\, A_i \\log\\pi_\\theta(y_{i,t})$ with
    $\\hat r_{i,t} = \\mathrm{clip}(r_{i,t}, 1-\\epsilon^{IS}_{low}, 1+\\epsilon^{IS}_{high})$.
    The weight is a constant for autograd (stop-gradient), so the gradient is $\\hat r_{i,t} A_i \\nabla\\log\\pi_\\theta$:
    a token whose ratio passed the bound keeps its gradient at the capped weight instead of losing it.
    ``eps_low=None`` means no lower bound (MiniMax-M1 sets $\\epsilon^{IS}_{low}$ large and tunes only the upper one).
    ``eps_high=0.28`` is this course's default (DAPO's upper range), not a value from the paper.
    """
    ratio = torch.exp(logp.detach() - old_logp.detach())
    lo = 0.0 if eps_low is None else 1.0 - eps_low
    w = torch.clamp(ratio, min=lo, max=1.0 + eps_high)
    a = _adv2(adv, logp)
    obj = w * a * logp
    m = mask.float()
    capped = ((ratio > 1.0 + eps_high) | (ratio < lo)).float()
    return -(obj * m).sum() / m.sum().clamp_min(1.0), _diag(ratio, capped, mask)


# --------------------------------------------------------------------------- registry

@dataclass(frozen=True)
class Objective:
    """An objective = its loss, its default parameters, and the loop settings the paper pairs with it."""
    name: str
    loss: Callable
    params: dict = field(default_factory=dict)
    loop: dict = field(default_factory=dict)     # RLConfig overrides (advantage scale, zero-variance handling)
    source: str = ""


OBJECTIVES: dict[str, Objective] = {
    "grpo": Objective("grpo", grpo_loss, {"eps": 0.2}, {"baseline": "mean", "scale": "group",
                                                         "aggregation": "seq_mean_token_mean"},
                      "DeepSeekMath Eq. 3"),
    "dapo": Objective("dapo", dapo_loss, {"eps_low": 0.2, "eps_high": 0.28},
                      {"baseline": "mean", "scale": "group", "aggregation": "token_mean", "zero_var": "filter"},
                      "DAPO Eq. 8, section 3.2 (dynamic sampling as filtering)"),
    "drgrpo": Objective("drgrpo", drgrpo_loss, {"eps": 0.2}, {"baseline": "mean", "scale": "none",
                                                               "aggregation": "seq_mean_token_sum_norm"},
                        "Dr. GRPO section 3.2"),
    "gspo": Objective("gspo", gspo_loss, {"eps_low": 3e-4, "eps_high": 4e-4},
                      {"baseline": "mean", "scale": "group", "aggregation": "seq_mean_token_mean"},
                      "GSPO section 4, ranges from section 5.1"),
    "cispo": Objective("cispo", cispo_loss, {"eps_low": None, "eps_high": 0.28},
                       {"baseline": "mean", "scale": "group", "aggregation": "token_mean"},
                       "MiniMax-M1 section 3.1"),
}


def as_hook(obj: Objective | str, loss_fn: Callable | None = None, **params) -> Callable:
    """A ``policy_loss`` hook for :func:`frontierlab.posttrain.rl.train` that ignores the loop's ratio,
    clip and aggregation flags and uses the objective's own. ``loss_fn`` replaces the loss (a lab's version);
    ``params`` override the objective's defaults (for example a tuned ``eps_high``)."""
    o = OBJECTIVES[obj] if isinstance(obj, str) else obj
    fn = loss_fn or o.loss
    p = {**o.params, **params}
    if "norm_len" in fn.__code__.co_varnames:
        def hook(logp, old_logp, adv, mask, *, norm_len=None, is_weight=None, **_):
            return _with_is(fn, logp, old_logp, adv, mask, is_weight, norm_len=norm_len, **p)
    else:
        def hook(logp, old_logp, adv, mask, *, is_weight=None, **_):
            return _with_is(fn, logp, old_logp, adv, mask, is_weight, **p)
    return hook


def _with_is(fn, logp, old_logp, adv, mask, is_weight, **p):
    if is_weight is None:
        return fn(logp, old_logp, adv, mask, **p)
    # A truncated-IS weight w > 0 for sampler/trainer mismatch multiplies each token's contribution. Because
    # w is positive and detached, min(r A w, clip(r) A w) = w min(r A, clip(r) A): scaling the per-token
    # advantage by w is exactly "multiply the per-token objective by w".
    a = _adv2(adv, logp) * is_weight.detach()
    if fn in (gspo_loss,):
        raise ValueError("truncated IS with sequence-level GSPO is not defined here; use gspo_token")
    return fn(logp, old_logp, a, mask, **p)


def gradient_weights(name: str, logp: torch.Tensor, old_logp: torch.Tensor, adv: torch.Tensor,
                     mask: torch.Tensor, loss_fn: Callable | None = None, **params) -> torch.Tensor:
    """(B, R) float64: minus the gradient of the loss with respect to each token's log-probability, i.e. the
    weight each token's $\\nabla\\log\\pi$ gets in the update. The quantity the lesson's table derives by hand."""
    o = OBJECTIVES[name]
    lp = logp.detach().double().clone().requires_grad_(True)
    loss, _ = (loss_fn or o.loss)(lp, old_logp.double(), adv.double(), mask.double(), **{**o.params, **params})
    loss.backward()
    return -lp.grad
