"""What Muon costs: optimizer FLOPs, the share of a training step, and steps for an equal-wall-clock budget (07.1).

Newton-Schulz on one matrix of shape (A, B), working on the wide orientation (r = min(A, B) rows,
n = max(A, B) columns), costs per step (one multiply-add = 2 FLOPs, the course convention):

    A = X X^T          2 r^2 n
    A A                2 r^3
    B X                2 r^2 n
    total per step     4 r^2 n + 2 r^3          (the elementwise terms a X, b A + c AA are O(r n), ignored)

so ``k`` steps cost ``k (4 r^2 n + 2 r^3)``. A training step on ``tokens`` tokens costs about
6 · N · tokens FLOPs for N weights. For a square m x m matrix the ratio is
k · 6 m^3 / (6 m^2 · tokens) = k m / tokens — Jordan's bound "at most T*m/B" (T NS steps, m model width,
B batch size in tokens). It is small for real batches, but FLOPs are not time: NS runs on small matrices,
one after another, and in a sharded setting each matrix must be gathered whole first. That is why an
equal-wall-clock comparison must *measure* the optimizer step (:func:`time_optimizer_step`) instead of
trusting the FLOP ratio.
"""

from __future__ import annotations

import torch

from frontierlab.optim.muon import NS_SCHEDULES, split_params


def ns_flops(shape, steps: int = 5) -> int:
    """FLOPs of ``steps`` Newton-Schulz steps on a matrix of ``shape`` (formula in the module docstring)."""
    r, n = min(shape[0], shape[1]), max(shape[0], shape[1])
    return steps * (4 * r * r * n + 2 * r ** 3)


def optimizer_flops(model, schedule="quintic5") -> dict:
    """Newton-Schulz FLOPs per optimizer step for every Muon matrix of ``model``, and their sum."""
    steps = len(NS_SCHEDULES[schedule]) if isinstance(schedule, str) else len(schedule)
    per = {n: ns_flops(p.shape, steps) for n, p in split_params(model)["muon"]}
    return {"per_matrix": per, "total": sum(per.values()), "ns_steps": steps}


def shape_flops(shapes, steps: int = 5) -> int:
    """Same sum from a list of shapes, without building the model (main-path projections)."""
    return sum(ns_flops(s, steps) for s in shapes)


def muon_shapes(cfg) -> list[tuple[int, int]]:
    """Shapes of the Muon matrices of a GQA ``ModelConfig`` (Baseline-0 layout), for projections at any size."""
    C, H, KV, d, I, L = (cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
                         cfg.intermediate_size, cfg.num_hidden_layers)
    per_layer = [(H * d, C), (KV * d, C), (KV * d, C), (C, H * d), (I, C), (I, C), (C, I)]
    return per_layer * L


def overhead(model_or_cfg, tokens_per_step: int, train_flops_per_token: float, schedule="quintic5") -> dict:
    """Optimizer FLOPs as a fraction of the training FLOPs of one step."""
    steps = len(NS_SCHEDULES[schedule]) if isinstance(schedule, str) else len(schedule)
    if hasattr(model_or_cfg, "named_parameters"):
        opt = optimizer_flops(model_or_cfg, schedule)["total"]
    else:
        opt = shape_flops(muon_shapes(model_or_cfg), steps)
    train = train_flops_per_token * tokens_per_step
    return {"optimizer_flops": opt, "train_flops": train, "fraction": opt / train}


def jordan_bound(width: int, tokens_per_step: int, steps: int = 5) -> float:
    """T · m / B, Jordan's upper bound on the overhead fraction."""
    return steps * width / tokens_per_step


def equal_wallclock_steps(base_steps: int, base_step_s: float, new_step_s: float) -> int:
    """Steps the new arm gets so that steps x measured step time equals the baseline's budget."""
    return max(1, int(base_steps * base_step_s / new_step_s))


def time_optimizer_step(model, make_opt, *, batch: torch.Tensor, rounds: int = 10, warmup: int = 3,
                        device: str = "cpu"):
    """Measure forward + backward + optimizer step, and the optimizer step alone, with frontierlab.perf.

    ``make_opt(model)`` builds the optimizer. Returns ``{"step": Timing, "opt_only": Timing}``. The model is
    trained during the measurement (its weights change); pass a copy if that matters.
    """
    from frontierlab.perf.timing import benchmark
    opt = make_opt(model)

    def fwd_bwd():
        opt.zero_grad(set_to_none=True)
        model(batch, labels=batch).loss.backward()

    def full():
        fwd_bwd()
        opt.step()

    step = benchmark(full, warmup=warmup, repeats=rounds, device=device)
    fwd_bwd()
    grads = {p: p.grad.clone() for p in model.parameters() if p.grad is not None}

    def restore():
        for p, g in grads.items():
            p.grad = g.clone()

    opt_only = benchmark(opt.step, warmup=warmup, repeats=rounds, device=device, setup=restore)
    return {"step": step, "opt_only": opt_only}
