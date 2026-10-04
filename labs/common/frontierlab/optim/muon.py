"""Muon from scratch, and one optimizer that runs Muon and AdamW parameter groups side by side (lesson 07.1).

Muon ("MomentUm Orthogonalized by Newton-Schulz", K. Jordan, https://kellerjordan.github.io/posts/muon/)
updates each 2-D weight matrix W (shape A x B = fan_out x fan_in, as ``nn.Linear`` stores it) by:

    M_t = mu * M_{t-1} + G_t                       momentum buffer (SGD momentum, mu = 0.95)
    U_t = G_t + mu * M_t      (Nesterov)           or U_t = M_t without Nesterov
    O_t = NS(U_t)                                  Newton-Schulz: an approximate orthogonalisation,
                                                   U = P S Q^T  ->  O ~ P S' Q^T with S' ~ 1
    W_t = W_{t-1} - lr * wd * W_{t-1}              decoupled weight decay (Moonlight section 2.2, Eq. 3)
    W_t = W_t - lr * s(A, B) * O_t                 s = sqrt(max(1, A/B))      "original" (Jordan)
                                                   s = 0.2 * sqrt(max(A, B))  "match_rms" (Moonlight Eq. 4)

The Newton-Schulz iteration (Jordan's ``newtonschulz5``) works on X = U / ||U||_F (so every singular value
is at most 1), on the wide orientation (rows <= columns), and repeats

    A = X X^T,   B = b A + c A A,   X = a X + B X        i.e. each singular value s -> a s + b s^3 + c s^5

with the quintic coefficients (a, b, c) = (3.4445, -4.7750, 2.0315) for 5 steps, in bfloat16. Those
coefficients maximise the slope at zero (small singular values grow fast); the price is that the result
is not exactly orthogonal: singular values end up spread around 1 instead of equal to 1. DeepSeek-V4
(arXiv 2606.19348, section 2.4) runs 10 steps: 8 with the same quintic coefficients, then 2 with
(2, -1.5, 0.5), a polynomial whose fixed point at 1 is stable, to "stabilize the singular values precisely
at 1". ``NS_SCHEDULES`` holds both, plus the classic cubic iteration (1.5, -0.5, 0) for comparison.

Which parameters use Muon: only 2-D matrices inside the blocks. The token embedding, the output head
(tied to the embedding in Baseline-0), every norm gain, learned sinks and other vectors stay on AdamW
(Jordan's post: "Scalar and vector parameters of the network, as well as the input and output layers,
should be optimized by a standard method such as AdamW"; Moonlight, DeepSeek-V4 section 2.4 the same).

:class:`MuonAdamW` is one ``torch.optim.Optimizer`` with two kinds of parameter groups (``kind="muon"`` or
``kind="adamw"``). Being a single optimizer, it drops into the course loop unchanged: the loop sets
``group["lr"]`` from its schedule for every group, ``state_dict()`` carries the momentum buffers, Adam
moments and step counters, and exact resume holds (tests/test_optim.py). Each group may also carry
``lr_scale`` (µP, lesson 07.3): the step uses ``lr * lr_scale``. With ``adjust="match_rms"`` Muon reuses
AdamW's learning rate and weight decay, which is why one schedule can drive both kinds of group.
``torch.optim.Muon`` (PyTorch 2.14) implements the same update for Muon-only groups; the tests compare.
"""

from __future__ import annotations

import math

import torch

QUINTIC = (3.4445, -4.7750, 2.0315)     # Jordan's coefficients (maximal slope at zero)
CUBIC = (1.5, -0.5, 0.0)                # classic Newton-Schulz: X <- 1.5 X - 0.5 X X^T X
V4_FINAL = (2.0, -1.5, 0.5)             # DeepSeek-V4's last two steps (stable fixed point at 1)

NS_SCHEDULES = {
    "quintic5": [QUINTIC] * 5,                       # Jordan / Moonlight / Kimi K2
    "v4-hybrid": [QUINTIC] * 8 + [V4_FINAL] * 2,     # DeepSeek-V4 section 2.4
    "cubic5": [CUBIC] * 5,
    "cubic20": [CUBIC] * 20,
}


def ns_polynomial(s, coeffs=QUINTIC):
    """What one Newton-Schulz step does to a singular value s: a s + b s^3 + c s^5 (works on floats or tensors)."""
    a, b, c = coeffs
    return a * s + b * s ** 3 + c * s ** 5


def newton_schulz(G: torch.Tensor, schedule="quintic5", eps: float = 1e-7,
                  dtype: torch.dtype | None = torch.bfloat16) -> torch.Tensor:
    """Approximate orthogonalisation of a matrix G (A, B) -> O (A, B), same singular vectors, singular values ~ 1.

    ``schedule`` is a name from :data:`NS_SCHEDULES` or a list of (a, b, c) tuples, one per step.
    ``dtype`` is the working precision (bfloat16 as in Jordan's code; float32/float64 for tests; None = G's).
    The result is returned in the working dtype; the caller casts it.
    """
    if G.ndim != 2:
        raise ValueError(f"Newton-Schulz needs a matrix, got shape {tuple(G.shape)}")
    steps = NS_SCHEDULES[schedule] if isinstance(schedule, str) else list(schedule)
    X = G.to(dtype) if dtype is not None else G.clone()
    tall = X.size(0) > X.size(1)
    if tall:
        X = X.T                                          # work on the wide orientation: X X^T is the small Gram
    X = X / (X.norm() + eps)                             # Frobenius norm >= spectral norm, so all s <= 1
    for a, b, c in steps:
        A = X @ X.T                                      # (r, r), r = min(A, B)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if tall else X


def orthogonal_polar(G: torch.Tensor) -> torch.Tensor:
    """The exact target U V^T of G = U S V^T (float64 SVD), for tests and for measuring NS error."""
    U, _, Vh = torch.linalg.svd(G.double(), full_matrices=False)
    return U @ Vh


def update_scale(shape, adjust: str = "match_rms") -> float:
    """Multiplier s(A, B) of the orthogonalised update for a weight of shape (A, B) = (fan_out, fan_in).

    ``"original"``: sqrt(max(1, A/B)) (Jordan's code); ``"match_rms"``: 0.2 * sqrt(max(A, B)), which gives an
    update RMS of about 0.2, AdamW's typical range (Moonlight section 2.2, Eq. 4); ``"none"``: 1.
    """
    A, B = shape[0], shape[1]
    if adjust == "original":
        return math.sqrt(max(1.0, A / B))
    if adjust == "match_rms":
        return 0.2 * math.sqrt(max(A, B))
    if adjust == "none":
        return 1.0
    raise ValueError(f"unknown adjust {adjust!r}")


def rms(t: torch.Tensor) -> float:
    return t.float().pow(2).mean().sqrt().item()


# ------------------------------------------------------------------------------------- parameter groups

EXCLUDE_FROM_MUON = ("embed_tokens", "lm_head")


def muon_eligible(name: str, p: torch.Tensor) -> bool:
    """True for 2-D weight matrices inside the blocks; False for embedding, head, norms, sinks and vectors."""
    return p.ndim == 2 and not any(x in name for x in EXCLUDE_FROM_MUON) and "norm" not in name


def split_params(model) -> dict:
    """{"muon": [(name, p)], "adamw_decay": [...], "adamw_no_decay": [...]} for a ``frontierlab.model.LM``.

    Tied weights appear once (``named_parameters`` de-duplicates): the tied head is the embedding.
    AdamW decay follows the course loop: matrices decay, vectors and norm gains do not.
    """
    out = {"muon": [], "adamw_decay": [], "adamw_no_decay": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if muon_eligible(n, p):
            out["muon"].append((n, p))
        elif p.ndim >= 2 and "norm" not in n:
            out["adamw_decay"].append((n, p))
        else:
            out["adamw_no_decay"].append((n, p))
    return out


def param_groups(model, *, optimizer: str = "muon", lr: float = 3e-3, weight_decay: float = 0.1,
                 adjust: str = "match_rms", momentum: float = 0.95, nesterov: bool = True,
                 schedule="quintic5", ns_dtype: str = "bfloat16", betas=(0.9, 0.95), eps: float = 1e-8,
                 lr_scales: dict | None = None) -> list[dict]:
    """Parameter groups for :class:`MuonAdamW`.

    ``optimizer="adamw"`` puts every parameter on AdamW (the baseline arm, same code path as the Muon arm);
    ``"muon"`` puts :func:`muon_eligible` matrices on Muon. ``lr_scales`` ({name: factor}, from
    :func:`frontierlab.optim.mup.lr_scales`) splits groups further so each group has one ``lr_scale``.
    """
    if optimizer not in ("muon", "adamw"):
        raise ValueError(optimizer)
    sp = split_params(model)
    if optimizer == "adamw":
        sp["adamw_decay"] = sp["muon"] + sp["adamw_decay"]
        sp["muon"] = []
    lr_scales = lr_scales or {}
    groups = []
    for key, base in (("muon", {"kind": "muon", "weight_decay": weight_decay, "momentum": momentum,
                                "nesterov": nesterov, "adjust": adjust, "ns_schedule": schedule,
                                "ns_dtype": ns_dtype}),
                      ("adamw_decay", {"kind": "adamw", "weight_decay": weight_decay, "betas": betas, "eps": eps}),
                      ("adamw_no_decay", {"kind": "adamw", "weight_decay": 0.0, "betas": betas, "eps": eps})):
        by_scale: dict[float, list] = {}
        for n, p in sp[key]:
            by_scale.setdefault(float(lr_scales.get(n, 1.0)), []).append((n, p))
        for scale, items in sorted(by_scale.items(), reverse=True):
            groups.append({**base, "params": [p for _, p in items], "names": [n for n, _ in items],
                           "lr": lr, "lr_scale": scale})
    return groups


# ------------------------------------------------------------------------------------------- optimizer

_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float64": torch.float64, "none": None}


class MuonAdamW(torch.optim.Optimizer):
    """Muon for ``kind="muon"`` groups, AdamW for ``kind="adamw"`` groups, in one optimizer (module docstring).

    AdamW here is written out (the same arithmetic as ``torch.optim.AdamW`` with ``foreach=False``; the tests
    check it to float64 precision) so that both arms of a comparison run the same code.

    ``collect_stats = True`` makes the next ``step()`` record, per parameter name, the RMS of the update
    and of the weight (``last_stats``), for the stability logger (lesson 07.5). ``post_step_hooks`` are
    called with the optimizer after every step (QK-Clip, lesson 07.2).
    """

    def __init__(self, groups: list[dict], lr: float = 3e-3):
        defaults = {"lr": lr, "lr_scale": 1.0, "weight_decay": 0.0, "kind": "adamw", "names": None,
                    "momentum": 0.95, "nesterov": True, "adjust": "match_rms", "ns_schedule": "quintic5",
                    "ns_dtype": "bfloat16", "betas": (0.9, 0.95), "eps": 1e-8}
        super().__init__(groups, defaults)
        for g in self.param_groups:
            if g["kind"] == "muon" and any(p.ndim != 2 for p in g["params"]):
                raise ValueError("Muon groups take 2-D matrices only")
        self.steps_taken = 0
        self.collect_stats = False
        self.last_stats: dict = {}
        self.post_step_hooks: list = []

    # exact resume: the global step counter travels with the state
    def state_dict(self):
        sd = super().state_dict()
        sd["muon_adamw"] = {"steps_taken": self.steps_taken}
        return sd

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        extra = state_dict.pop("muon_adamw", {"steps_taken": 0})
        super().load_state_dict(state_dict)
        self.steps_taken = int(extra["steps_taken"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        stats = {} if self.collect_stats else None
        for g in self.param_groups:
            lr = g["lr"] * g["lr_scale"]
            names = g["names"] or [None] * len(g["params"])
            for name, p in zip(names, g["params"]):
                if p.grad is None:
                    continue
                if g["kind"] == "muon":
                    upd = self._muon(p, g, lr)
                else:
                    upd = self._adamw(p, g, lr)
                if stats is not None and name is not None:
                    stats[name] = {"kind": g["kind"], "update_rms": rms(upd), "weight_rms": rms(p)}
        self.steps_taken += 1
        if stats is not None:
            self.last_stats = stats
        for hook in self.post_step_hooks:
            hook(self)
        return loss

    def _muon(self, p, g, lr):
        st = self.state[p]
        if "momentum_buffer" not in st:
            st["momentum_buffer"] = torch.zeros_like(p)
        buf = st["momentum_buffer"]
        grad = p.grad
        buf.mul_(g["momentum"]).add_(grad)                          # M = mu M + G
        u = grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
        O = newton_schulz(u, g["ns_schedule"], dtype=_DTYPES[g["ns_dtype"]]).to(p.dtype)
        step = O * (lr * update_scale(p.shape, g["adjust"]))
        if g["weight_decay"]:
            step = step + p * (lr * g["weight_decay"])
        p.sub_(step)
        return step

    def _adamw(self, p, g, lr):
        st = self.state[p]
        if "step" not in st:
            st["step"] = 0
            st["exp_avg"] = torch.zeros_like(p)
            st["exp_avg_sq"] = torch.zeros_like(p)
        st["step"] += 1
        b1, b2 = g["betas"]
        m, v, t = st["exp_avg"], st["exp_avg_sq"], st["step"]
        before = p.detach().clone() if self.collect_stats else None
        p.mul_(1 - lr * g["weight_decay"])                         # decoupled weight decay, as torch.optim.AdamW
        m.lerp_(p.grad, 1 - b1)
        v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
        bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
        denom = (v.sqrt() / math.sqrt(bc2)).add_(g["eps"])
        p.addcdiv_(m, denom, value=-lr / bc1)
        return (before - p) if before is not None else p.new_zeros(())


def make_optimizer(model, *, optimizer: str = "muon", lr: float = 3e-3, weight_decay: float = 0.1, **kw) -> MuonAdamW:
    """The optimizer of a Module 7 run (see :func:`param_groups` for the keyword arguments)."""
    return MuonAdamW(param_groups(model, optimizer=optimizer, lr=lr, weight_decay=weight_decay, **kw), lr=lr)
