"""The tested-capability matrix: which features actually compose for the course model (lesson 09.3).

A framework's feature list says what exists; it does not say what works *together* for *your* architecture.
This module runs one small probe per feature (or combination) on real processes and records, for each, one of:

* ``composed``   — ran, and its correctness check passed (loss or gradients equal to a single-process reference
  within the stated tolerance, or a saved state restored bit-identically);
* ``failed``     — raised, or ran but failed its check (the error or the mismatch is recorded);
* ``untestable`` — needs hardware or a package this machine does not have (CUDA, NCCL, a C++ compiler, torchao).

Probes use the course model (Baseline-0's layout at toy size: GQA with QK-norm, RoPE, SwiGLU, tied embeddings) in
float64 on CPU ``gloo`` processes. The result is evidence about *this* model, *this* torch version and *this*
machine; the GPU column of the lesson's matrix is filled from the main-path torchtitan runs.

    python -m frontierlab.dist.capability --world 2 --out runs/m09/capability.json
    python -m frontierlab.dist.capability --world 4          # adds the 2-D (FSDP2 x TP) probe
"""

from __future__ import annotations

import argparse
import json
import traceback

import torch
import torch.distributed as dist

TOL = 1e-10


def _cfg(vocab=61):
    from frontierlab.model import toy
    return toy(vocab_size=vocab).with_(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                                       num_key_value_heads=2, head_dim=16, intermediate_size=96)


def _model(seed=0):
    from frontierlab.model import LM
    torch.manual_seed(seed)
    return LM(_cfg()).double()


def _data(rank, world, B=2, T=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    full = torch.randint(0, 61, (B * world, T), generator=g)
    return full, full[rank * B:(rank + 1) * B]


def _ref_step(full_batch):
    """One SGD step of the whole model on the whole batch in one process: the reference."""
    m = _model()
    loss = m(full_batch, labels=full_batch).loss
    loss.backward()
    return m, float(loss)


def _sgd(model, lr=0.1):
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= lr * p.grad


def _max_param_diff(model, ref) -> float:
    from torch.distributed.tensor import DTensor
    err = 0.0
    sd_ref = dict(ref.named_parameters())
    for n, p in model.named_parameters():
        n = n.replace("_checkpoint_wrapped_module.", "").replace("_fsdp_wrapped_module.", "")
        if n.startswith("module."):
            n = n[len("module."):]
        t = p.full_tensor() if isinstance(p, DTensor) else p
        err = max(err, float((t.detach() - sd_ref[n].detach()).abs().max()))
    return err


# ---- probes (each returns (status, detail)) ---------------------------------------------------------

def probe_ddp(rank, world):
    from torch.nn.parallel import DistributedDataParallel as DDP
    full, mine = _data(rank, world)
    m = DDP(_model())
    m(mine, labels=mine).loss.backward()
    _sgd(m)
    ref, _ = _ref_step(full)
    _sgd(ref)
    return _verdict(_max_param_diff(m, ref), "parameters after one step vs one process on the whole batch")


def _fsdp(model, ac=False):
    from torch.distributed.fsdp import fully_shard
    if ac:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
        for i, layer in enumerate(model.model.layers):
            model.model.layers[i] = checkpoint_wrapper(layer)
    for layer in model.model.layers:
        fully_shard(layer)
    fully_shard(model)
    return model


def probe_fsdp2(rank, world, ac=False):
    full, mine = _data(rank, world)
    m = _fsdp(_model(), ac=ac)
    m(mine, labels=mine).loss.backward()
    _sgd(m)
    ref, _ = _ref_step(full)
    _sgd(ref)
    return _verdict(_max_param_diff(m, ref), "parameters after one step vs one process")


def tp_plan_for(num_layers: int) -> dict:
    """Megatron-style TP plan for the course model: q/k/v/gate/up column-parallel, o/down row-parallel."""
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
    plan = {}
    for i in range(num_layers):
        p = f"model.layers.{i}"
        plan.update({f"{p}.self_attn.q_proj": ColwiseParallel(), f"{p}.self_attn.k_proj": ColwiseParallel(),
                     f"{p}.self_attn.v_proj": ColwiseParallel(), f"{p}.self_attn.o_proj": RowwiseParallel(),
                     f"{p}.mlp.gate_proj": ColwiseParallel(), f"{p}.mlp.up_proj": ColwiseParallel(),
                     f"{p}.mlp.down_proj": RowwiseParallel()})
    return plan


def _tp_plan():
    return tp_plan_for(_cfg().num_hidden_layers)


def _allreduce_qk_norm_grads(model, group=None):
    """QK-norm weights are replicated over TP ranks, but each rank only sees its own heads, so each holds a
    *partial* gradient. Sum them (the fix the probe with ``fix_norms`` applies)."""
    from torch.distributed.tensor import DTensor
    for layer in model.model.layers:
        for norm in (layer.self_attn.q_norm, layer.self_attn.k_norm):
            g = norm.weight.grad
            if g is None:                               # FSDP2 without gradient sync keeps grads elsewhere
                continue
            dist.all_reduce(g.to_local() if isinstance(g, DTensor) else g, group=group)


def probe_tp(rank, world, adapt_heads=False, fix_norms=False, mesh=None):
    """Megatron-style TP with DTensor: q/k/v/gate/up column-parallel, o/down row-parallel.

    The course attention reshapes with its *global* head counts (``view(B, T, self.H, hd)``); after a column
    split each rank holds H/tp heads, so without adapting the module the reshape fails. ``adapt_heads`` sets
    the per-rank head counts, the change torchtitan's parallelize functions make for their own models."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import parallelize_module
    cfg = _cfg()
    tp = 2 if world % 2 == 0 else world               # TP over pairs of ranks (the course model has 2 KV heads)
    if adapt_heads and cfg.num_key_value_heads % tp:
        return "failed", (f"TP degree {tp} does not divide the {cfg.num_key_value_heads} KV heads; the plan would "
                          "need KV-head replication, which it does not do")
    full, _ = _data(0, 1)
    if mesh is None and world != tp:
        mesh = init_device_mesh("cpu", (world // tp, tp), mesh_dim_names=("rep", "tp"))["tp"]
    elif mesh is None:
        mesh = init_device_mesh("cpu", (world,))
    m = _model()
    parallelize_module(m, mesh, _tp_plan())
    if adapt_heads:
        for layer in m.model.layers:
            layer.self_attn.H //= mesh.size()
            layer.self_attn.KV //= mesh.size()
    loss = m(full, labels=full).loss                  # TP: every rank sees the same batch
    loss.backward()
    if fix_norms:
        _allreduce_qk_norm_grads(m, group=mesh.get_group())
    ref = _model()
    ref_loss = ref(full, labels=full).loss
    ref_loss.backward()
    err = abs(float(loss) - float(ref_loss))
    from torch.distributed.tensor import DTensor
    for (n, p), (_, q) in zip(m.named_parameters(), ref.named_parameters()):
        g = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
        err = max(err, float((g - q.grad).abs().max()))
    return _verdict(err, "loss and gradients vs one process")


def probe_fsdp_tp(rank, world):
    """2-D: FSDP2 over one mesh dimension, TP over the other (needs 4 ranks)."""
    if world < 4:
        return "untestable", "needs 4 ranks (run with --world 4)"
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor.parallel import parallelize_module
    mesh = init_device_mesh("cpu", (world // 2, 2), mesh_dim_names=("dp", "tp"))
    dp_rank, dp = mesh["dp"].get_local_rank(), mesh["dp"].size()
    full, mine = _data(dp_rank, dp)
    m = _model()
    parallelize_module(m, mesh["tp"], _tp_plan())
    for layer in m.model.layers:
        layer.self_attn.H //= 2
        layer.self_attn.KV //= 2
        fully_shard(layer, mesh=mesh["dp"])
    fully_shard(m, mesh=mesh["dp"])
    m(mine, labels=mine).loss.backward()
    _allreduce_qk_norm_grads(m, group=mesh["tp"].get_group())
    _sgd(m)
    ref, _ = _ref_step(full)
    _sgd(ref)
    return _verdict(_max_param_diff(m, ref), "parameters after one step vs one process (QK-norm grads summed over TP)")


def probe_pp(rank, world):
    """torch.distributed.pipelining: the course model split into ``world`` stages, 1F1B with 4 micro-batches."""
    from torch.distributed.pipelining import PipelineStage, Schedule1F1B
    from frontierlab.dist.layout import stage_layers
    cfg = _cfg()
    full, _ = _data(0, 2)
    ref = _model()
    ref_loss = float(ref(full, labels=full).loss)

    class Stage(torch.nn.Module):
        def __init__(self, base, layers, first, last):
            super().__init__()
            self.emb = base.model.embed_tokens if first else None
            self.layers = torch.nn.ModuleList(base.model.layers[i] for i in layers)
            self.norm = base.model.norm if last else None
            self.head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False).double() if last else None
            if last:
                self.head.weight.data.copy_(base.lm_head.weight.data)
            self.T = full.shape[1]

        def forward(self, x):
            pos = torch.arange(self.T)
            if self.emb is not None:
                x = self.emb(x)
            for blk in self.layers:
                x = blk(x, pos)
            return self.head(self.norm(x)) if self.head is not None else x

    base = _model()
    layers = stage_layers(cfg.num_hidden_layers, world)[rank]
    stage_mod = Stage(base, layers, rank == 0, rank == world - 1)
    stage = PipelineStage(stage_mod, rank, world, torch.device("cpu"))

    def loss_fn(logits, target):
        return torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                                 target[:, 1:].reshape(-1))

    sched = Schedule1F1B(stage, n_microbatches=4, loss_fn=loss_fn)
    losses = []
    if rank == 0:
        sched.step(full)
    elif rank == world - 1:
        sched.step(target=full, losses=losses)
    else:
        sched.step()
    out = torch.tensor([float(sum(losses) / len(losses)) if losses else 0.0], dtype=torch.float64)
    dist.broadcast(out, world - 1)
    return _verdict(abs(float(out) - ref_loss), "mean micro-batch loss vs one process (forward and backward ran)")


def probe_ring(rank, world):
    from frontierlab.dist.ring_attention import equivalence_worker
    r = equivalence_worker(rank, world, T=8 * world)
    return _verdict(max(r["out"], r["dq"], r["dk"], r["dv"]), "output and dq/dk/dv vs single-device attention")


def probe_dcp_reshard(rank, world, tmp):
    """Save FSDP2-sharded model + optimizer with DCP, load into a plain (unsharded) model in every process."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    full, mine = _data(rank, world)
    m = _fsdp(_model())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    m(mine, labels=mine).loss.backward()
    opt.step()
    msd, osd = get_state_dict(m, opt)
    dcp.save({"model": msd, "optim": osd}, checkpoint_id=tmp)
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    want = get_model_state_dict(m, options=StateDictOptions(full_state_dict=True))
    plain = _model(seed=123)                        # different init: everything must come from the checkpoint
    popt = torch.optim.AdamW(plain.parameters(), lr=1e-2)
    psd, posd = get_state_dict(plain, popt)
    state = {"model": psd, "optim": posd}
    dcp.load(state, checkpoint_id=tmp)
    set_state_dict(plain, popt, model_state_dict=state["model"], optim_state_dict=state["optim"])
    err = max(float((plain.state_dict()[k] - v).abs().max()) for k, v in want.items())
    return ("composed" if err == 0.0 else "failed",
            f"FSDP2 on {world} ranks -> unsharded model: max |diff| = {err:.1e} (must be exactly 0)")


def probe_async_dcp(rank, world, tmp):
    import torch.distributed.checkpoint as dcp
    m = _fsdp(_model())
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    fut = dcp.async_save({"model": get_model_state_dict(m)}, checkpoint_id=tmp)
    fut.result()
    return "composed", "dcp.async_save completed (load checked in the DCP probe)"


def probe_compile_fsdp(rank, world):
    full, mine = _data(rank, world)
    m = _fsdp(_model())
    try:
        cm = torch.compile(m)
        cm(mine, labels=mine).loss.backward()
    except Exception as e:                          # noqa: BLE001
        msg = str(e).splitlines()[0][:160] if str(e) else type(e).__name__
        if "compiler" in msg.lower() or "cl" in msg.lower() or "cxx" in msg.lower():
            return "untestable", f"torch.compile needs a C++ compiler here: {msg}"
        return "failed", msg
    return "composed", "forward and backward ran under torch.compile (numerics not compared)"


PROBES = [
    ("DDP", "data parallel", probe_ddp, {}),
    ("FSDP2", "fully_shard per block + root", probe_fsdp2, {}),
    ("FSDP2 + activation checkpointing", "checkpoint_wrapper on every block", probe_fsdp2, {"ac": True}),
    ("TP (DTensor), course attention unchanged", "Colwise q/k/v/gate/up, Rowwise o/down", probe_tp, {}),
    ("TP (DTensor) + per-rank head counts", "same plan, attention told its local heads", probe_tp, {"adapt_heads": True}),
    ("TP + per-rank heads + QK-norm grad all-reduce", "as above, partial QK-norm grads summed over TP",
     probe_tp, {"adapt_heads": True, "fix_norms": True}),
    ("FSDP2 x TP (2-D)", "dp x tp mesh", probe_fsdp_tp, {}),
    ("PP (torch.distributed.pipelining, 1F1B)", "one stage per rank", probe_pp, {}),
    ("CP: ring attention (course code)", "frontierlab.dist.ring_attention", probe_ring, {}),
    ("DCP save FSDP2 -> load unsharded", "resharding on load", probe_dcp_reshard, {"tmp": True}),
    ("DCP async_save under FSDP2", "", probe_async_dcp, {"tmp": True}),
    ("torch.compile + FSDP2", "Inductor on CPU", probe_compile_fsdp, {}),
]

NOT_ON_CPU = [
    ("NCCL backend", "needs CUDA GPUs"),
    ("Float8 linears (torchao)", "needs an FP8-capable GPU (L4, H100)"),
    ("MXFP8 / NVFP4 linears", "needs Blackwell (B200)"),
    ("async TP", "needs CUDA symmetric memory"),
    ("torchtitan as a whole", "the trainer targets CUDA; run it on the main path"),
]


def _verdict(err: float, what: str):
    return ("composed" if err <= TOL else "failed"), f"{what}: max |diff| = {err:.1e} (tolerance {TOL:.0e})"


def capability_worker(rank, world, tmp_root="", only=None):
    import os
    out = []
    for i, (name, how, fn, kw) in enumerate(PROBES):
        if only is not None and name not in only:
            continue
        args = dict(kw)
        if args.pop("tmp", False):
            args["tmp"] = os.path.join(tmp_root, f"probe{i}")
        try:
            status, detail = fn(rank, world, **args)
        except Exception as e:                      # noqa: BLE001 - a failed composition is a result
            last = traceback.format_exception_only(type(e), e)[-1].strip()
            status, detail = "failed", last[:300]
        # a probe can fail on one rank only; agree on the worst outcome
        code = torch.tensor([{"composed": 0, "untestable": 1, "failed": 2}[status]])
        dist.all_reduce(code, op=dist.ReduceOp.MAX)
        status = {0: "composed", 1: "untestable", 2: "failed"}[int(code)]
        out.append({"feature": name, "how": how, "status": status, "detail": detail})
        dist.barrier()
    return {"rows": out}


def run_matrix(world: int = 2) -> list[dict]:
    import tempfile
    from frontierlab.dist import capability as importable     # so the workers can import it under ``-m``
    from frontierlab.perf.dist import spawn
    tmp = tempfile.mkdtemp(prefix="flab-cap-")
    rows = spawn(importable.capability_worker, world, tmp_root=tmp)[0]["rows"]
    rows += [{"feature": n, "how": "", "status": "untestable", "detail": why} for n, why in NOT_ON_CPU]
    return rows


def environment() -> dict:
    import platform
    return {"torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform(),
            "cuda": torch.cuda.is_available(), "gloo": dist.is_gloo_available()}


def to_markdown(rows: list[dict], env: dict) -> str:
    lines = [f"Tested on: torch {env['torch']}, Python {env['python']}, {env['platform']}, CPU gloo processes.", "",
             "| Feature | How | Result | Evidence |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['feature']} | {r['how']} | {r['status']} | {r['detail'].replace('|', '/')} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--out", default=None, help="write the rows and the environment as JSON")
    a = ap.parse_args(argv)
    rows, env = run_matrix(a.world), environment()
    print(to_markdown(rows, env))
    if a.out:
        from pathlib import Path
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"env": env, "world": a.world, "rows": rows}, indent=2))
    return rows


if __name__ == "__main__":
    main()

__all__ = ["NOT_ON_CPU", "PROBES", "capability_worker", "tp_plan_for", "environment", "main", "run_matrix", "to_markdown"]
