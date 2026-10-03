"""Multi-process helpers and communication arithmetic (lesson 02.3).

:func:`spawn` starts ``world_size`` processes, initialises a process group in each (``gloo`` on CPU,
``nccl`` on GPUs) through a file in a temporary directory, runs a worker function and collects the
dict each rank returns. Works on Linux, macOS and Windows (``spawn`` start method), so the worker
must be an importable top-level function — the workers below are.

The CPU (gloo) variant runs **real collectives** with the same semantics as NCCL on GPUs, so it shows
bucketing, the order of communication and how overlap is measured. Its *times* are those of
processes on one machine copying through shared memory and loopback sockets; they say nothing about
NVLink or InfiniBand performance.

Communication arithmetic (ring algorithm, n ranks, a buffer of S bytes):

    all-reduce:      each rank sends and receives 2·(n-1)/n · S bytes
    reduce-scatter:  (n-1)/n · S       all-gather: (n-1)/n · S
    bus bandwidth  = bytes moved per rank / time          (the number nccl-tests reports as "busbw")
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def allreduce_bytes_per_rank(nbytes: float, world: int) -> float:
    """Bytes each rank sends (and receives) in a ring all-reduce of an ``nbytes`` buffer."""
    return 2.0 * (world - 1) / world * nbytes


def bus_bandwidth(nbytes: float, seconds: float, world: int, op: str = "all_reduce") -> float:
    """Bus bandwidth in bytes/s, comparable across world sizes (nccl-tests convention)."""
    factor = {"all_reduce": 2.0 * (world - 1) / world, "reduce_scatter": (world - 1) / world,
              "all_gather": (world - 1) / world}[op]
    return factor * nbytes / seconds


def ddp_buckets(param_bytes: list[int], cap_bytes: int, first_cap_bytes: int | None = None) -> list[int]:
    """Approximate DDP's gradient buckets: parameters in *reverse* registration order (the order
    gradients become ready in backward), packed greedily until a bucket reaches ``cap_bytes``.

    DDP's default is ``bucket_cap_mb=25`` with a smaller 1 MiB first bucket so the first all-reduce can
    start early (``torch.nn.parallel.distributed``, torch 2.14.1). Returns bucket sizes in bytes.
    """
    buckets, cur = [], 0
    cap = first_cap_bytes if first_cap_bytes is not None else cap_bytes
    for b in reversed(param_bytes):
        cur += b
        if cur >= cap:
            buckets.append(cur)
            cur, cap = 0, cap_bytes
    if cur:
        buckets.append(cur)
    return buckets


def _entry(rank, world, store_path, out_dir, fn_module, fn_name, kwargs):
    import importlib
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // world))
    backend = "nccl" if kwargs.get("device", "cpu") == "cuda" else "gloo"
    dist.init_process_group(backend, init_method=Path(store_path).as_uri(), rank=rank, world_size=world)
    try:
        if backend == "nccl":
            torch.cuda.set_device(rank)
        fn = getattr(importlib.import_module(fn_module), fn_name)
        res = fn(rank, world, **kwargs)
        (Path(out_dir) / f"rank{rank}.json").write_text(json.dumps(res))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def spawn(fn, world_size: int, **kwargs) -> list[dict]:
    """Run ``fn(rank, world_size, **kwargs)`` in ``world_size`` processes; return each rank's dict."""
    tmp = tempfile.mkdtemp(prefix="flab-dist-")
    store = os.path.join(tmp, "store")
    mp.spawn(_entry, args=(world_size, store, tmp, fn.__module__, fn.__name__, kwargs), nprocs=world_size,
             join=True)
    return [json.loads((Path(tmp) / f"rank{r}.json").read_text()) for r in range(world_size)]


# ---- workers ---------------------------------------------------------------------------------

def allreduce_worker(rank, world, sizes_mb=(1, 4, 16), iters=10, device="cpu"):
    """Time all-reduce of fp32 buffers of the given sizes; return median seconds and bus bandwidth."""
    out = {}
    for mb in sizes_mb:
        t = torch.ones(int(mb * 2**20 / 4), device=device)
        for _ in range(3):
            dist.all_reduce(t)
        ts = []
        for _ in range(iters):
            dist.barrier()
            t0 = time.perf_counter()
            dist.all_reduce(t)
            if device == "cuda":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        med = sorted(ts)[len(ts) // 2]
        out[str(mb)] = {"seconds": med, "busbw_GBps": bus_bandwidth(t.numel() * 4, med, world) / 1e9}
    return out


def _toy_lm(preset: str, vocab: int, seed: int = 0):
    from frontierlab.model import LM, PRESETS
    torch.manual_seed(seed)
    return LM(PRESETS[preset](vocab_size=vocab))


def train_step_worker(rank, world, mode="ddp", preset="toy", vocab=8192, batch=8, seq=128, steps=8,
                      warmup=3, bucket_cap_mb=25.0, device="cpu", dtype="fp32", seed=0):
    """Time training steps of a data-parallel model, with and without gradient synchronisation.

    ``mode="ddp"``: ``DistributedDataParallel`` with ``bucket_cap_mb``; the no-sync arm runs inside
    ``model.no_sync()`` so no all-reduce happens. ``mode="fsdp2"``: ``fully_shard`` on every block and
    on the root; the no-sync arm calls ``set_requires_gradient_sync(False)``, which skips the gradient
    reduce-scatter (parameter all-gathers still happen in both arms). The difference of the medians
    is the *exposed* communication: the part the computation did not hide.

    ``dtype="bf16"`` (GPU): DDP runs the forward under ``torch.autocast``; FSDP2 uses
    ``MixedPrecisionPolicy(param_dtype=bfloat16, reduce_dtype=float32)`` instead.
    """
    bf16 = dtype == "bf16"
    dev = torch.device(device if device == "cpu" else f"cuda:{rank}")
    model = _toy_lm(preset, vocab, seed).to(dev)
    if mode == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP
        wrapped = DDP(model, bucket_cap_mb=bucket_cap_mb, device_ids=[rank] if dev.type == "cuda" else None)
        nosync = wrapped.no_sync
    elif mode == "fsdp2":
        from contextlib import contextmanager

        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
        mp = (MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32) if bf16
              else MixedPrecisionPolicy())
        for layer in model.model.layers:
            fully_shard(layer, mp_policy=mp)
        fully_shard(model, mp_policy=mp)
        bf16 = False                     # FSDP2 casts parameters itself; no autocast needed
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
    g = torch.Generator().manual_seed(seed + rank)
    x = torch.randint(0, vocab, (batch, seq), generator=g).to(dev)

    def fwd_bwd():
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=bf16):
            loss = wrapped(x, labels=x).loss
        loss.backward()

    def step(sync_grads: bool):
        opt.zero_grad(set_to_none=True)
        if sync_grads:
            fwd_bwd()
        else:
            with nosync():
                fwd_bwd()
        opt.step()

    res = {}
    for arm, flag in (("sync", True), ("nosync", False)):
        for _ in range(warmup):
            step(flag)
        ts = []
        for _ in range(steps):
            dist.barrier()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            step(flag)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        res[arm] = ts
        if arm == "sync" and mode == "ddp":
            # replicas must agree after synchronised steps (different inputs per rank, same update)
            res["param_checksum_after_sync"] = float(sum(p.detach().double().sum() for p in model.parameters()))
    n_params = sum(p.numel() for p in model.parameters())
    res["grad_bytes"] = n_params * 4
    if mode == "ddp":
        res["n_buckets"] = len(ddp_buckets([p.numel() * 4 for p in model.parameters() if p.requires_grad],
                                           int(bucket_cap_mb * 2**20)))   # explicit cap: first bucket same size
    return res
