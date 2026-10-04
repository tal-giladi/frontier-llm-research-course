"""Measure one layout on processes: step time, exposed communication, bytes held per rank (lesson 09.3, CPU path).

``layout_worker`` trains the course model for a few steps under one layout and returns, per rank:

* step times with gradient synchronisation and without it (DDP ``no_sync``; FSDP2
  ``set_requires_gradient_sync(False)``), so exposed communication = median(sync) − median(no-sync), as in 02.3;
* the bytes of tensors the rank holds after a step — parameters (local shards), gradients and optimizer state —
  counted from the tensors themselves (``numel × element size`` of local shards). On a GPU the allocator's peak
  (``torch.cuda.max_memory_allocated``) is also returned; on CPU there is no allocator statistic, so the count is
  of persistent state only, not of activations.

Layouts: ``ddp``, ``fsdp2`` (one mesh dimension), ``fsdp2_tp`` (2-D mesh: FSDP2 over ``world/tp`` ranks × TP over
``tp`` ranks, with the per-rank head counts and the QK-norm gradient all-reduce found by the capability probes).
The model is the course model (``frontierlab.model``); every rank's tokens are different, the same seed per rank.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch
import torch.distributed as dist


def _local_bytes(t) -> int:
    from torch.distributed.tensor import DTensor
    if t is None:
        return 0
    t = t.to_local() if isinstance(t, DTensor) else t
    return t.numel() * t.element_size()


def held_bytes(model, opt) -> dict:
    seen = set()
    params = grads = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        params += _local_bytes(p)
        grads += _local_bytes(p.grad)
    optim = sum(_local_bytes(v) for st in opt.state.values() for v in st.values() if torch.is_tensor(v) and v.dim() > 0)
    return {"params": params, "grads": grads, "optimizer": optim, "total": params + grads + optim}


def layout_worker(rank, world, mode="ddp", tp=2, preset="toy", vocab=8192, batch=8, seq=128, steps=8, warmup=2,
                  device="cpu", seed=0):
    from frontierlab.model import LM, PRESETS
    dev = torch.device("cpu" if device == "cpu" else f"cuda:{rank}")
    torch.manual_seed(seed)
    cfg = PRESETS[preset](vocab_size=vocab)
    model = LM(cfg).to(dev)
    tp_group = None
    if mode == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP
        wrapped = DDP(model, device_ids=[rank] if dev.type == "cuda" else None)
        nosync = wrapped.no_sync
        dp_rank, dp_size = rank, world
    elif mode in ("fsdp2", "fsdp2_tp"):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        if mode == "fsdp2_tp":
            from torch.distributed.tensor.parallel import parallelize_module

            from frontierlab.dist.capability import tp_plan_for
            mesh = init_device_mesh(dev.type, (world // tp, tp), mesh_dim_names=("dp", "tp"))
            parallelize_module(model, mesh["tp"], tp_plan_for(cfg.num_hidden_layers))
            for layer in model.model.layers:
                layer.self_attn.H //= tp
                layer.self_attn.KV //= tp
            dp_mesh, tp_group = mesh["dp"], mesh["tp"].get_group()
            dp_rank, dp_size = mesh["dp"].get_local_rank(), mesh["dp"].size()
        else:
            dp_mesh = init_device_mesh(dev.type, (world,))
            dp_rank, dp_size = rank, world
        for layer in model.model.layers:
            fully_shard(layer, mesh=dp_mesh)
        fully_shard(model, mesh=dp_mesh)
        wrapped = model

        @contextmanager
        def nosync():
            model.set_requires_gradient_sync(False)
            try:
                yield
            finally:
                model.set_requires_gradient_sync(True)
    else:
        raise ValueError(mode)
    opt = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed + dp_rank)
    x = torch.randint(0, cfg.vocab_size, (batch, seq), generator=g).to(dev)

    def step(sync: bool):
        opt.zero_grad(set_to_none=True)
        if sync:
            wrapped(x, labels=x).loss.backward()
        else:
            with nosync():
                wrapped(x, labels=x).loss.backward()
        if tp_group is not None:
            from frontierlab.dist.capability import _allreduce_qk_norm_grads
            _allreduce_qk_norm_grads(model, group=tp_group)
        opt.step()

    res = {"mode": mode, "dp": dp_size, "tp": tp if mode == "fsdp2_tp" else 1}
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    for arm, flag in (("sync", True), ("nosync", False)):
        for _ in range(warmup):
            step(flag)
        ts = []
        for _ in range(steps):
            dist.barrier()
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            step(flag)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            ts.append(time.perf_counter() - t0)
        res[arm] = ts
        if arm == "sync":
            res["held"] = held_bytes(model, opt)
    if dev.type == "cuda":
        res["max_allocated"] = torch.cuda.max_memory_allocated(dev)
    res["tokens_per_step"] = batch * seq * dp_size
    return res


__all__ = ["held_bytes", "layout_worker"]
