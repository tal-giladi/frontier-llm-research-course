"""Fault injection and recovery with torch.distributed.checkpoint (DCP) (lesson 09.4).

A small data-parallel trainer (FSDP2 or DDP over ``gloo`` or NCCL) whose *full* training state survives a
crash:

* **model** and **optimizer** (AdamW) — sharded tensors under FSDP2, saved with ``dcp.save`` so every rank
  writes only its own shard;
* **LR scheduler** (``LambdaLR``, warm-up then cosine) and the **step**;
* **data-stream position** — one data generator shared by all ranks (each draws the whole global batch and
  keeps its slice), so its state is the same on every rank and does not depend on the number of ranks;
* **RNG** — the trainer drops 10% of input tokens at random, so randomness during training is real. Two modes:
  ``stateful``: a per-rank ``torch.Generator`` whose state is saved per rank (exact resume, but only into
  the same number of ranks); ``counter``: the noise for global sample ``g`` at step ``t`` comes from a
  generator seeded with ``(seed, t, g)``, so it is the same whatever the layout (counter-based RNG).

Checkpoints are committed atomically: all ranks write ``ckpt/step_NNNNNN/`` with DCP, then a barrier, then
rank 0 writes ``COMMITTED``. A restart loads the newest committed checkpoint; a half-written one (a crash
during the save) is skipped. ``die_at`` makes one rank call ``os._exit`` after a given step — a hard crash with
no clean-up, like a lost node — and ``die_in_save`` kills it between the shard writes and the commit.

Restoring into a different layout: DCP stores each tensor with its global shape and the offsets of every
shard, and ``dcp.load`` reads whatever pieces the *new* sharding needs. Save at FSDP2 on 2 ranks, load at
FSDP2 on 4 ranks or DDP on 1, and the full model and optimizer state is bit-identical after loading
(``labs/common/tests/test_dist.py``). Training on from there is *not* bitwise equal to the original run:
the gradient reduction sums in a different order, which changes the last bits (and with ``stateful`` RNG the
per-rank noise streams cannot be mapped onto a different number of ranks at all).

Every rank-0 step appends to ``<run>/metrics.jsonl``: loss, LR, step time, checkpoint time (blocking part),
and a ``start`` record per launch with the load time, so lost work and restart cost can be measured.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

DEFAULTS = dict(steps=20, ckpt_every=5, mode="fsdp2", global_batch=8, seq=32, vocab=97, hidden=64, layers=2,
                lr=3e-3, warmup=4, seed=0, dtype="float64", rng_mode="stateful", drop=0.1, die_at=None,
                die_rank=1, die_in_save=False, async_save=False, resume_from=None, data=None, device="cpu",
                keep=3, final_state=True, reshard_check=False)


# ---- data ------------------------------------------------------------------------------------------

def synthetic_tokens(n: int = 200_000, vocab: int = 97, seed: int = 1234):
    """A learnable token stream (a repeating pattern with 10% noise), the same on every rank."""
    g = torch.Generator().manual_seed(seed)
    base = torch.arange(n) % 32 + 1
    noise = torch.randint(1, vocab, (n,), generator=g)
    return torch.where(torch.rand(n, generator=g) < 0.1, noise, base) % vocab


class GlobalStream:
    """Resumable, layout-independent data stream: every rank draws the same global batch, keeps its slice."""

    def __init__(self, tokens: torch.Tensor, seq: int, seed: int):
        self.tokens, self.seq = tokens, seq
        self.gen = torch.Generator().manual_seed(seed)
        self.samples_seen = 0

    def next_global(self, global_batch: int) -> torch.Tensor:
        starts = torch.randint(0, len(self.tokens) - self.seq, (global_batch,), generator=self.gen)
        self.samples_seen += global_batch
        return torch.stack([self.tokens[s:s + self.seq] for s in starts.tolist()]).long()

    def state_dict(self) -> dict:
        return {"gen": self.gen.get_state(), "samples_seen": self.samples_seen}

    def load_state_dict(self, sd: dict) -> None:
        self.gen.set_state(sd["gen"])
        self.samples_seen = int(sd["samples_seen"])


def _noise_mask(shape, step: int, first_sample: int, spec: dict, gen: torch.Generator | None) -> torch.Tensor:
    if spec["rng_mode"] == "stateful":
        return torch.rand(shape, generator=gen) < spec["drop"]
    rows = []
    for j in range(shape[0]):                          # counter-based: depends only on (seed, step, sample)
        g = torch.Generator().manual_seed(hash((spec["seed"], step, first_sample + j)) % (2**63))
        rows.append(torch.rand(shape[1], generator=g) < spec["drop"])
    return torch.stack(rows)


# ---- model, optimizer, schedule ---------------------------------------------------------------------

def _model(spec: dict):
    from frontierlab.model import LM, toy
    torch.manual_seed(spec["seed"])
    cfg = toy(vocab_size=spec["vocab"]).with_(hidden_size=spec["hidden"], num_hidden_layers=spec["layers"],
                                               num_attention_heads=4, num_key_value_heads=2,
                                               head_dim=spec["hidden"] // 4, intermediate_size=3 * spec["hidden"])
    return LM(cfg).to(getattr(torch, spec["dtype"]))


def lr_lambda(spec: dict):
    w, n = spec["warmup"], spec["steps"]

    def f(step):
        if step < w:
            return (step + 1) / w
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, n - w)))
    return f


def _wrap(model, spec: dict, rank: int):
    if spec["mode"] == "fsdp2":
        from torch.distributed.fsdp import fully_shard
        for layer in model.model.layers:
            fully_shard(layer)
        fully_shard(model)
        return model
    if spec["mode"] == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP
        return DDP(model, device_ids=[rank] if spec["device"] == "cuda" else None)
    raise ValueError(spec["mode"])


def _inner(wrapped):
    return wrapped.module if hasattr(wrapped, "module") else wrapped


# ---- checkpoints -------------------------------------------------------------------------------------

def committed(ckpt_root: str | Path) -> list[Path]:
    """Committed checkpoint folders, oldest first (a folder without ``COMMITTED`` is an interrupted save)."""
    root = Path(ckpt_root)
    if not root.exists():
        return []
    return sorted(p for p in root.glob("step_*") if (p / "COMMITTED").exists())


def latest_committed(ckpt_root: str | Path) -> Path | None:
    c = committed(ckpt_root)
    return c[-1] if c else None


def _app_state(model, opt, sched, stream, step, noise_gen, rank, world):
    from torch.distributed.checkpoint.state_dict import get_state_dict
    msd, osd = get_state_dict(_inner(model), opt)
    extra = {"step": step, "world": world, "scheduler": sched.state_dict(), "data": stream.state_dict()}
    per_rank = {"torch_rng": torch.get_rng_state()}
    if noise_gen is not None:
        per_rank["noise_rng"] = noise_gen.get_state()
    return {"model": msd, "optim": osd, "extra": extra, f"rank{rank}": per_rank}


_CKPT_GROUP = None


def _ckpt_group():
    """A separate gloo process group for asynchronous saves: the background writer runs its collectives while the
    training loop runs its own, and two threads must not drive collectives on the same group at once (on the build
    machine that crashed the process)."""
    global _CKPT_GROUP
    if _CKPT_GROUP is None:
        _CKPT_GROUP = dist.new_group(backend="gloo")
    return _CKPT_GROUP


def _save(path: Path, state: dict, spec: dict, rank: int):
    import torch.distributed.checkpoint as dcp
    t0 = time.perf_counter()
    fut = None
    if spec["async_save"]:
        fut = dcp.async_save(state, checkpoint_id=str(path), process_group=_ckpt_group())
    else:
        dcp.save(state, checkpoint_id=str(path))
    blocking = time.perf_counter() - t0
    return fut, blocking


def _commit(path: Path, rank: int, spec: dict, step: int):
    if spec["die_in_save"] and spec["die_at"] == step and rank == spec["die_rank"]:
        os._exit(17)                                   # crash after writing shards, before the commit
    dist.barrier()
    if rank == 0:
        (path / "COMMITTED").write_text(json.dumps({"step": step, "time": time.time()}))
        old = committed(path.parent)[:-spec["keep"]] if spec["keep"] else []
        for p in old:
            shutil.rmtree(p, ignore_errors=True)
    dist.barrier()


def _load(path: Path, model, opt, sched, stream, noise_gen, rank, world, spec):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    msd, osd = get_state_dict(_inner(model), opt)
    meta = json.loads((path / "COMMITTED").read_text())
    state = {"model": msd, "optim": osd,
             "extra": {"step": 0, "world": 0, "scheduler": sched.state_dict(), "data": stream.state_dict()}}
    reader = dcp.FileSystemReader(str(path))
    saved_keys = set(reader.read_metadata().state_dict_metadata)
    rk = f"rank{rank}"
    saved_world = len({k.split(".")[0] for k in saved_keys if k.startswith("rank")})
    same_layout = saved_world == world
    if same_layout:
        state[rk] = {"torch_rng": torch.get_rng_state()}
        if noise_gen is not None:
            state[rk]["noise_rng"] = noise_gen.get_state()
    dcp.load(state, checkpoint_id=str(path))
    set_state_dict(_inner(model), opt, model_state_dict=state["model"], optim_state_dict=state["optim"])
    sched.load_state_dict(state["extra"]["scheduler"])
    stream.load_state_dict(state["extra"]["data"])
    notes = []
    if same_layout:
        torch.set_rng_state(state[rk]["torch_rng"])
        if noise_gen is not None:
            noise_gen.set_state(state[rk]["noise_rng"])
    elif spec["rng_mode"] == "stateful":
        notes.append(f"saved with {state['extra']['world']} ranks, loading into {world}: per-rank RNG streams "
                     "cannot be mapped, re-seeded (the run is no longer the same run)")
    return int(state["extra"]["step"]), meta, notes


def _full_state(model, opt, sched, stream, step, noise_gen):
    """Gathered full state on rank 0 (all ranks must call it): for bitwise comparison between runs."""
    from torch.distributed.checkpoint.state_dict import (StateDictOptions, get_model_state_dict,
                                                         get_optimizer_state_dict)
    o = StateDictOptions(full_state_dict=True, cpu_offload=True)
    msd = get_model_state_dict(_inner(model), options=o)
    osd = get_optimizer_state_dict(_inner(model), opt, options=o)
    rng = [None] * dist.get_world_size()
    mine = {"torch": torch.get_rng_state(), "noise": noise_gen.get_state() if noise_gen is not None else None}
    dist.all_gather_object(rng, mine)
    return {"model": msd, "optim": osd, "scheduler": sched.state_dict(), "data": stream.state_dict(),
            "step": step, "rng": rng}


# ---- the worker ----------------------------------------------------------------------------------------

def train_worker(rank, world, **kw):
    spec = {**DEFAULTS, **kw}
    t_launch = time.perf_counter()
    run = Path(spec["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    if spec["global_batch"] % world:
        raise ValueError("global_batch must be divisible by the number of ranks")
    local = spec["global_batch"] // world
    if spec["data"]:
        from frontierlab.data.loader import TokenData
        tokens = torch.from_numpy(TokenData("train", root=spec["data"]).tokens[:2_000_000].astype("int64"))
    else:
        tokens = synthetic_tokens(vocab=spec["vocab"])
    torch.manual_seed(spec["seed"])
    if spec["async_save"]:
        _ckpt_group()                                  # collective: every rank creates it at the same point
    model = _wrap(_model(spec), spec, rank)
    opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"], betas=(0.9, 0.95), weight_decay=0.1, foreach=False)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(spec))
    stream = GlobalStream(tokens, spec["seq"], spec["seed"] + 7)
    noise_gen = torch.Generator().manual_seed(spec["seed"] * 1000 + rank) if spec["rng_mode"] == "stateful" else None
    step, notes, load_s, loaded_from = 0, [], 0.0, None
    src = latest_committed(run / "ckpt")
    if src is None and spec["resume_from"]:
        src = Path(spec["resume_from"])
    if src is not None:
        t0 = time.perf_counter()
        step, _, notes = _load(src, model, opt, sched, stream, noise_gen, rank, world, spec)
        load_s, loaded_from = time.perf_counter() - t0, str(src)
        if spec["reshard_check"]:
            st = _full_state(model, opt, sched, stream, step, noise_gen)
            if rank == 0:
                torch.save(st, run / "loaded_state.pt")
    log = open(run / "metrics.jsonl", "a") if rank == 0 else None
    if log:
        log.write(json.dumps({"event": "start", "world": world, "mode": spec["mode"], "from_step": step,
                              "loaded_from": loaded_from, "load_s": load_s,
                              "startup_s": time.perf_counter() - t_launch, "notes": notes, "time": time.time()}) + "\n")
        log.flush()
    pending = None
    while step < spec["steps"]:
        t0 = time.perf_counter()
        batch = stream.next_global(spec["global_batch"])
        mine = batch[rank * local:(rank + 1) * local]
        first = stream.samples_seen - spec["global_batch"] + rank * local
        x = mine.clone()
        x[_noise_mask(x.shape, step, first, spec, noise_gen)] = 0
        if spec["device"] == "cuda":
            x, mine = x.cuda(), mine.cuda()
        opt.zero_grad(set_to_none=True)
        logits = model(x).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), mine[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        lsum = loss.detach().clone()
        dist.all_reduce(lsum)
        step_s = time.perf_counter() - t0
        ck_s = None
        if step % spec["ckpt_every"] == 0 or step == spec["steps"]:
            if pending is not None:                    # an async save in flight: finish and commit it first
                _finish(pending, rank, spec)
                pending = None
            path = run / "ckpt" / f"step_{step:06d}"
            state = _app_state(model, opt, sched, stream, step, noise_gen, rank, world)
            fut, ck_s = _save(path, state, spec, rank)
            if fut is None:
                _commit(path, rank, spec, step)
            else:
                pending = (fut, path, step)
        if log:
            log.write(json.dumps({"event": "step", "step": step, "loss": float(lsum) / world,
                                  "lr": sched.get_last_lr()[0], "step_s": step_s, "ckpt_s": ck_s}) + "\n")
            log.flush()
        if spec["die_at"] == step and not spec["die_in_save"] and rank == spec["die_rank"]:
            os._exit(17)                               # hard crash: no clean-up, no checkpoint
    if pending is not None:
        _finish(pending, rank, spec)
    if spec["final_state"]:
        st = _full_state(model, opt, sched, stream, step, noise_gen)
        if rank == 0:
            torch.save(st, run / "final_state.pt")
    if log:
        log.close()
    return {"step": step}


def _finish(pending, rank, spec):
    """Wait for this rank's async write to finish, then commit (an async checkpoint is committed later than
    it was started: a crash in between loses it, and the restart uses the previous one)."""
    fut, path, step = pending
    fut.result()
    _commit(path, rank, spec, step)


# ---- driver ------------------------------------------------------------------------------------------

def launch(world: int, **spec) -> dict:
    """Start ``world`` processes running :func:`train_worker`. Returns {"ok": bool, "error": str|None, "s": seconds}."""
    from frontierlab.perf.dist import spawn
    t0 = time.perf_counter()
    try:
        spawn(train_worker, world, **spec)
        return {"ok": True, "error": None, "s": time.perf_counter() - t0}
    except Exception as e:                         # noqa: BLE001 - a crashed rank surfaces as an exception
        return {"ok": False, "error": f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}",
                "s": time.perf_counter() - t0}


def read_metrics(run_dir) -> list[dict]:
    p = Path(run_dir) / "metrics.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def losses_by_step(run_dir) -> dict[int, float]:
    """Loss per step; a step that was run twice (after a crash) keeps its last value."""
    return {r["step"]: r["loss"] for r in read_metrics(run_dir) if r.get("event") == "step"}


def _flat(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _flat(v, f"{prefix}{k}.")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from _flat(v, f"{prefix}{i}.")
    else:
        yield prefix[:-1], x


def compare_states(a: dict, b: dict) -> dict:
    """Per component (model, optim, scheduler, data, step, rng): bitwise equal? and the largest difference."""
    out = {}
    for comp in ("model", "optim", "scheduler", "data", "step", "rng"):
        fa, fb = dict(_flat(a.get(comp))), dict(_flat(b.get(comp)))
        same, maxdiff = fa.keys() == fb.keys(), 0.0
        for k in fa.keys() & fb.keys():
            x, y = fa[k], fb[k]
            if torch.is_tensor(x) and torch.is_tensor(y):
                if x.shape != y.shape or x.dtype != y.dtype or not torch.equal(x, y):
                    same = False
                    if x.shape == y.shape and x.is_floating_point():
                        maxdiff = max(maxdiff, float((x.double() - y.double()).abs().max()))
                    else:
                        maxdiff = float("inf")
            elif x != y:
                same = False
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    maxdiff = max(maxdiff, abs(x - y))
        out[comp] = {"bitwise_equal": same, "max_abs_diff": maxdiff}
    return out


def load_final(run_dir, name: str = "final_state.pt") -> dict:
    return torch.load(Path(run_dir) / name, weights_only=False)


__all__ = ["DEFAULTS", "GlobalStream", "committed", "compare_states", "latest_committed", "launch", "load_final",
           "losses_by_step", "lr_lambda", "read_metrics", "synthetic_tokens", "train_worker"]
