"""Interpretable by design: weight-sparse transformers and circuit pruning, at toy scale (lesson 17.5).

Gao et al. (2025, "Weight-sparse transformers have interpretable circuits", section 2.1) train transformers
in which, after every AdamW step, all but the largest-magnitude entries of each weight matrix are set to
zero; the number of kept entries is annealed from dense to the target over the first half of training.
They then find the smallest circuit for a task by learning a mask over nodes, with deleted nodes
*mean-ablated*, until the task loss reaches a target (section 2.2), and report that at matched loss the
sparse models' circuits are roughly 16 times smaller than the dense models' (Fig. 2).

The course version:

* task — **closing quote type**, one of the paper's tasks in miniature: a string opens with ``'`` or ``"``,
  some word tokens follow, and at the end the model must output the matching quote;
* models — Baseline-0's architecture at toy size, trained dense or with per-matrix magnitude top-k
  sparsity (:func:`train_task_model`);
* nodes — every MLP neuron and every attention-output channel (the ``z`` inputs of o_proj);
* pruning — a sigmoid gate per node, trained to minimise task loss + λ · (expected kept nodes), with
  mean ablation of gated-off nodes (:func:`prune_circuit`); the circuit is the nodes whose gate stays open
  and the edges are the nonzero weights between kept nodes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from frontierlab.interp import hooks as HK
from frontierlab.model import LM, ModelConfig

Q1, Q2, OPEN_PAD = 1, 2, 3          # token ids: single quote, double quote, padding before the string


def task_config(vocab: int = 40, hidden_size: int = 64, num_hidden_layers: int = 2) -> ModelConfig:
    return ModelConfig(vocab_size=vocab, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers,
                       num_attention_heads=4, num_key_value_heads=4, head_dim=hidden_size // 4,
                       intermediate_size=2 * hidden_size, max_position_embeddings=64)


def quote_batch(B: int, gen: torch.Generator, vocab: int = 40, T: int = 16):
    """(B, T) sequences ``pad… Q word word … word`` and the answer (the same quote) at the last position.
    The opening quote sits at a random position, so the model must find it by content."""
    x = torch.randint(4, vocab, (B, T), generator=gen)
    start = torch.randint(0, T - 4, (B,), generator=gen)
    q = torch.where(torch.rand(B, generator=gen) < 0.5, torch.tensor(Q1), torch.tensor(Q2))
    ar = torch.arange(T)[None]
    x = torch.where(ar < start[:, None], torch.tensor(OPEN_PAD), x)
    x = torch.where(ar == start[:, None], q[:, None], x)
    return x, q


def task_loss(model, x, q):
    logits = HK.logits_of(model, x)[:, -1]
    return nn.functional.cross_entropy(logits, q), logits


def _sparse_params(model):
    return [p for n, p in model.named_parameters() if p.dim() == 2 and "layers" in n]


@torch.no_grad()
def apply_topk_(model, frac: float):
    """Keep the largest-magnitude ``frac`` of the entries of every block weight matrix; zero the rest."""
    for p in _sparse_params(model):
        k = max(1, int(round(frac * p.numel())))
        thr = p.abs().flatten().kthvalue(p.numel() - k + 1).values
        p.mul_((p.abs() >= thr).to(p.dtype))


def train_task_model(density: float = 1.0, steps: int = 1500, batch: int = 128, lr: float = 3e-3, seed: int = 0,
                     log_every: int = 0, **cfg) -> LM:
    """Train on the quote task. ``density`` < 1: after every step keep only that fraction of each block weight
    matrix, annealed linearly from 1 to ``density`` over the first half of training."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    m = LM(task_config(**cfg))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    for s in range(steps):
        x, q = quote_batch(batch, gen)
        loss, _ = task_loss(m, x, q)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if density < 1:
            frac = 1 - (1 - density) * min(1.0, s / (steps / 2))
            apply_topk_(m, frac)
        if log_every and s % log_every == 0:
            print(f"density {density}: step {s} loss {loss.item():.4f}")
    return m.eval()


def nonzero_fraction(model) -> float:
    ps = _sparse_params(model)
    return float(sum((p != 0).sum() for p in ps) / sum(p.numel() for p in ps))


# --------------------------------------------------------------------------------------------- pruning

def _node_sites(model):
    H, hd = HK.head_dim(model)
    out = []
    for l, layer in enumerate(HK.decoder_layers(model)):
        out.append((f"z.{l}", H * hd))
        out.append((f"mlpact.{l}", layer.mlp.down_proj.in_features))
    return out


class _Gates(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.logits = nn.ParameterList([nn.Parameter(torch.full((n,), 3.0)) for n in sizes])


def _gated_run(model, x, gates, means, hard: bool = False):
    """Logits with every node multiplied by its gate and the rest of its value replaced by its mean."""
    handles = []
    try:
        for (site, _), g, mu in zip(_node_sites(model), gates.logits, means):
            kind, L = site.split(".")
            L = int(L)
            gate = (g > 0).float() if hard else torch.sigmoid(g)
            fn = (lambda gate, mu: (lambda a: a * gate + mu * (1 - gate)))(gate, mu)
            mod = (HK.decoder_layers(model)[L].self_attn.o_proj if kind == "z"
                   else HK.decoder_layers(model)[L].mlp.down_proj)
            handles.append(mod.register_forward_pre_hook(lambda m, args, fn=fn: (fn(args[0]),)))
        return HK.logits_of(model, x)
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def node_means(model, n: int = 512, seed: int = 123):
    """Mean of every node over task inputs (the value a pruned node is replaced by)."""
    gen = torch.Generator().manual_seed(seed)
    x, _ = quote_batch(n, gen)
    means, handles = [], []
    store = {}
    for site, _ in _node_sites(model):
        kind, L = site.split(".")
        mod = (HK.decoder_layers(model)[int(L)].self_attn.o_proj if kind == "z"
               else HK.decoder_layers(model)[int(L)].mlp.down_proj)
        handles.append(mod.register_forward_pre_hook(lambda m, args, s=site: store.__setitem__(s, args[0].mean((0, 1)))))
    try:
        HK.logits_of(model, x)
    finally:
        for h in handles:
            h.remove()
    return [store[s] for s, _ in _node_sites(model)]


def prune_circuit(model, target_loss: float = 0.15, lam: float = 1e-3, steps: int = 600, lr: float = 0.05,
                  batch: int = 128, seed: int = 0) -> dict:
    """Learn node gates (model frozen). Returns kept nodes per site, their count, the hard-gated task loss
    on fresh inputs, and the number of nonzero weights among kept nodes (edges)."""
    for p in model.parameters():
        p.requires_grad_(False)
    means = node_means(model)
    gates = _Gates([n for _, n in _node_sites(model)])
    opt = torch.optim.Adam(gates.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    for s in range(steps):
        x, q = quote_batch(batch, gen)
        logits = _gated_run(model, x, gates, means)[:, -1]
        task = nn.functional.cross_entropy(logits, q)
        size = sum(torch.sigmoid(g).sum() for g in gates.logits)
        # only pay for size while the task loss is below target: the target is a constraint
        loss = task + (lam * size if task.item() < target_loss else 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        x, q = quote_batch(1024, torch.Generator().manual_seed(seed + 99))
        logits = _gated_run(model, x, gates, means, hard=True)[:, -1]
        hard_loss = float(nn.functional.cross_entropy(logits, q))
        acc = float((logits[:, [Q1, Q2]].argmax(-1) == (q == Q2).long()).float().mean())
    kept = {site: (g > 0).nonzero().flatten().tolist() for (site, _), g in zip(_node_sites(model), gates.logits)}
    for p in model.parameters():
        p.requires_grad_(True)
    return {"kept": kept, "n_nodes": sum(len(v) for v in kept.values()),
            "total_nodes": sum(n for _, n in _node_sites(model)), "loss": hard_loss, "acc": acc,
            "edges": circuit_edges(model, kept)}


@torch.no_grad()
def circuit_edges(model, kept: dict) -> int:
    """Nonzero weights that read from or write to a kept node (o_proj columns of kept z channels, and the
    gate/up rows and down_proj columns of kept MLP neurons)."""
    n = 0
    for l, layer in enumerate(HK.decoder_layers(model)):
        z = kept.get(f"z.{l}", [])
        mlp = kept.get(f"mlpact.{l}", [])
        if z:
            n += int((layer.self_attn.o_proj.weight[:, z] != 0).sum())
        if mlp:
            n += int((layer.mlp.down_proj.weight[:, mlp] != 0).sum())
            n += int((layer.mlp.gate_proj.weight[mlp] != 0).sum() + (layer.mlp.up_proj.weight[mlp] != 0).sum())
    return n


@torch.no_grad()
def accuracy(model, n: int = 1024, seed: int = 7) -> float:
    x, q = quote_batch(n, torch.Generator().manual_seed(seed))
    logits = HK.logits_of(model, x)[:, -1]
    return float((logits[:, [Q1, Q2]].argmax(-1) == (q == Q2).long()).float().mean())
