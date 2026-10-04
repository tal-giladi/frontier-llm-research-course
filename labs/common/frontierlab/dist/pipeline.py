"""A small real pipeline over processes: GPipe or 1F1B on the course model (lesson 09.2).

Rank ``s`` of ``p`` holds a contiguous slice of the model's layers (:func:`frontierlab.dist.layout.stage_layers`):
rank 0 also holds the token embedding, the last rank the final norm and the output head. One optimizer step
runs ``m`` micro-batches through the stages in the order the schedule gives (:mod:`frontierlab.dist.schedule`
produces the same order the simulator uses), sending activations forward and their gradients back with
point-to-point ``isend`` / ``recv``.

Tied embeddings: Baseline-0 shares the embedding matrix with the output head, but here they live on different
ranks. Each keeps a copy, and after the backward pass the two copies' gradients are all-reduced between the
first and the last rank (the same fix Megatron-LM uses), so both take the same update.

Measured per rank and step: wall time, *busy* time (forward and backward compute only) and time spent blocked
in ``recv``. The measured bubble is ``1 - busy / wall`` of each rank. On CPU processes (gloo) the numbers show
the schedule's structure; they are not GPU or interconnect timings, and ranks share the machine's cores, so
one rank's compute can slow another's (each process gets ``cores / p`` threads).

``check=True`` also runs the whole model in one process on the same micro-batches and returns the largest
difference of the loss and of every parameter gradient (float64: about 1e-15 when the pipeline is correct).
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from frontierlab.dist.layout import stage_layers
from frontierlab.dist.schedule import one_f_one_b_order


def schedule_order(name: str, p: int, s: int, m: int) -> list[tuple[str, int]]:
    """Op order of stage ``s``: GPipe (all F then all B) or 1F1B."""
    if name == "gpipe":
        return [("F", i) for i in range(m)] + [("B", i) for i in range(m)]
    if name == "1f1b":
        return one_f_one_b_order(p, s, m)
    raise ValueError(f"unknown schedule {name!r}")


def _build(cfg, seed, dtype):
    from frontierlab.model import LM
    torch.manual_seed(seed)
    return LM(cfg).to(dtype)


def pipeline_worker(rank, world, schedule="1f1b", m=8, layers=None, hidden=128, vocab=512, micro_batch=2, seq=64,
                    steps=3, seed=0, dtype="float32", check=False, lr=1e-3):
    from frontierlab.model import toy
    dt = getattr(torch, dtype)
    L = layers or 2 * world
    cfg = toy(vocab_size=vocab).with_(num_hidden_layers=L, hidden_size=hidden, intermediate_size=3 * hidden)
    full = _build(cfg, seed, dt)
    mine = stage_layers(L, world)[rank]
    first, last = rank == 0, rank == world - 1
    blocks = torch.nn.ModuleList(full.model.layers[i] for i in mine)
    emb = full.model.embed_tokens if first else None
    norm = full.model.norm if last else None
    head = None
    if last:
        head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False).to(dt)
        head.weight.data.copy_(full.lm_head.weight.data)      # own copy of the tied matrix
    params = [p for mod in (emb, blocks, norm, head) if mod is not None for p in mod.parameters()]
    opt = torch.optim.SGD(params, lr=lr)
    tied = dist.new_group([0, world - 1]) if world > 1 else None   # every rank must call new_group
    positions = torch.arange(seq)
    g = torch.Generator().manual_seed(seed + 1)
    data = torch.randint(0, vocab, (steps, m, micro_batch, seq), generator=g)
    shape = (micro_batch, seq, cfg.hidden_size)
    order = schedule_order(schedule, world, rank, m)
    out = {"wall": [], "busy": [], "recv_wait": [], "f": [], "b": [], "losses": []}

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        acts_in, acts_out, losses, sends = {}, {}, {}, []
        busy = wait = 0.0
        fts, bts = [], []
        dist.barrier()
        t_step = time.perf_counter()
        for kind, i in order:
            if kind == "F":
                if first:
                    x_in = None
                else:
                    buf = torch.empty(shape, dtype=dt)
                    t0 = time.perf_counter()
                    dist.recv(buf, rank - 1)
                    wait += time.perf_counter() - t0
                    x_in = buf.requires_grad_(True)
                t0 = time.perf_counter()
                x = emb(data[step, i]) if first else x_in
                for blk in blocks:
                    x = blk(x, positions)
                if last:
                    logits = head(norm(x)).float() if dt != torch.float64 else head(norm(x))
                    ids = data[step, i]
                    losses[i] = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), ids[:, 1:].reshape(-1)) / m
                dt_f = time.perf_counter() - t0
                busy += dt_f
                fts.append(dt_f)
                if not last:
                    sends.append(dist.isend(x.detach().contiguous(), rank + 1))
                acts_in[i], acts_out[i] = x_in, x
            else:
                if last:
                    t0 = time.perf_counter()
                    losses[i].backward()
                else:
                    gbuf = torch.empty(shape, dtype=dt)
                    t0 = time.perf_counter()
                    dist.recv(gbuf, rank + 1)
                    wait += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    acts_out[i].backward(gbuf)
                dt_b = time.perf_counter() - t0
                busy += dt_b
                bts.append(dt_b)
                if not first:
                    sends.append(dist.isend(acts_in[i].grad.contiguous(), rank - 1))
                acts_in.pop(i, None)
                acts_out.pop(i, None)                 # free this micro-batch's activations
        for r in sends:
            r.wait()
        if world > 1 and (first or last):             # tied embedding: sum the two copies' gradients
            gw = emb.weight.grad if first else head.weight.grad
            dist.all_reduce(gw, group=tied)
        wall = time.perf_counter() - t_step
        if check and step == 0:
            out.update(_check(full_ref=_build(cfg, seed, dt), data=data[0], m=m, rank=rank, world=world, mine=mine,
                              emb=emb, blocks=blocks, norm=norm, head=head, losses=losses))
        opt.step()
        out["wall"].append(wall)
        out["busy"].append(busy)
        out["recv_wait"].append(wait)
        out["f"].append(sum(fts) / max(1, len(fts)))
        out["b"].append(sum(bts) / max(1, len(bts)))
        out["losses"].append(float(sum(v.detach() for v in losses.values())) if last else None)
    out["peak_live_microbatches"] = _peak_live(order)
    return out


def _peak_live(order) -> int:
    live = best = 0
    for kind, _ in order:
        live += 1 if kind == "F" else -1
        best = max(best, live)
    return best


def _check(full_ref, data, m, rank, world, mine, emb, blocks, norm, head, losses):
    """Single-process reference on the same micro-batches: loss and gradients of this rank's parameters."""
    total = 0.0
    for i in range(m):
        total = total + full_ref(data[i], labels=data[i]).loss / m
    total.backward()
    err = 0.0
    if rank == 0:
        err = max(err, float((emb.weight.grad - full_ref.model.embed_tokens.weight.grad).abs().max()))
    if rank == world - 1:
        err = max(err, float((head.weight.grad - full_ref.lm_head.weight.grad).abs().max()))
        err = max(err, float((norm.weight.grad - full_ref.model.norm.weight.grad).abs().max()))
    for j, blk in zip(mine, blocks):
        for (n, p), (_, q) in zip(blk.named_parameters(), full_ref.model.layers[j].named_parameters()):
            err = max(err, float((p.grad - q.grad).abs().max()))
    res = {"grad_max_abs_err": err}
    if rank == world - 1:
        res["loss_abs_err"] = abs(float(sum(v.detach() for v in losses.values())) - float(total.detach()))
    return res


__all__ = ["pipeline_worker", "schedule_order"]
