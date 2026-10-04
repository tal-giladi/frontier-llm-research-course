"""Ring attention: causal attention over a sequence split across processes (context parallelism, lesson 09.1).

Each of ``n`` ranks holds the queries, keys and values of its own tokens (``T/n`` of them). Attention for
a query needs every earlier key, so the K/V blocks travel around a ring: at step ``i`` rank ``r`` holds the
block that started on rank ``(r - i) mod n``, computes attention of its queries against it, and merges the
partial result with the online-softmax rule (Liu et al., Ring Attention, arXiv 2310.01889; the merge is the
FlashAttention one):

    block result  o_j = softmax(S_j) V_j,  lse_j = log Σ exp(S_j)          (S_j = q K_jᵀ · scale, masked)
    merge         lse = log(exp(lse_a) + exp(lse_b))
                  o   = exp(lse_a - lse) · o_a + exp(lse_b - lse) · o_b

While a block is being computed, the next one is already in flight (``isend``/``irecv`` are started before the
compute), which is the overlap the paper relies on: communication is hidden when computing one block takes
longer than sending it.

Backward: with the saved output ``O`` and ``lse``, every block's gradient can be computed independently
(``P = exp(S - lse)``, ``dV_j = Pᵀ dO``, ``dS = P ∘ (dO Vᵀ - rowsum(dO ∘ O))``, ``dQ += dS K_j · scale``,
``dK_j = dSᵀ q · scale``). The K/V blocks make the same trip again, and each block's ``dK, dV`` accumulator
travels *with* it: after ``n`` hops it is back on its owner, holding the sum over all ranks' queries.

Masking uses absolute positions (key j visible to query i iff ``k_pos[j] <= q_pos[i]``), so any sequence
sharding works. A rank's tokens are split into contiguous segments, and only (query segment, key segment)
sub-blocks with at least one visible pair are computed — the block skipping a causal FlashAttention kernel does. :func:`shard_positions` gives the two used in practice: contiguous blocks, and the
load-balanced "two chunks per rank" layout (Llama 3 section 3.3.2; Meta's CP paper arXiv 2411.01783) that
gives every rank the same amount of causal work.

Grouped-query attention: ``k`` and ``v`` may have fewer heads than ``q`` (a divisor); blocks travel with the
smaller head count and are expanded only for the local compute.

The functions work on any backend; tests run them on ``gloo`` CPU processes in float64, where they agree
with single-device attention to about 1e-15 (forward and all three gradients).
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist


def shard_positions(T: int, world: int, rank: int, balanced: bool = True) -> torch.Tensor:
    """Token positions owned by ``rank``: contiguous ``T/world`` block, or chunks ``rank`` and ``2·world-1-rank``."""
    if balanced:
        if T % (2 * world):
            raise ValueError(f"T={T} must be divisible by 2*world={2 * world}")
        c = T // (2 * world)
        a, b = rank, 2 * world - 1 - rank
        return torch.cat((torch.arange(a * c, (a + 1) * c), torch.arange(b * c, (b + 1) * c)))
    if T % world:
        raise ValueError(f"T={T} must be divisible by world={world}")
    c = T // world
    return torch.arange(rank * c, (rank + 1) * c)


def causal_pairs(q_pos: torch.Tensor, k_pos: torch.Tensor) -> int:
    """Number of (query, key) pairs that are visible: the causal work of one block."""
    return int((k_pos[None, :] <= q_pos[:, None]).sum())


def work_per_rank(T: int, world: int, balanced: bool) -> list[int]:
    """Visible (query, key) pairs each rank computes over the whole ring (its share of causal attention)."""
    allpos = [shard_positions(T, world, r, balanced) for r in range(world)]
    return [sum(causal_pairs(allpos[r], allpos[j]) for j in range(world)) for r in range(world)]


def segments(pos: torch.Tensor) -> list[slice]:
    """Contiguous runs of positions as slices into ``pos`` (one for contiguous sharding, two for load-balanced)."""
    breaks = (torch.nonzero(pos[1:] != pos[:-1] + 1).flatten() + 1).tolist()
    edges = [0, *breaks, len(pos)]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]


def _visible_pairs(q_pos, kpos):
    """(query slice, key slice) pairs of segments with at least one visible (query, key) pair: the sub-blocks a
    kernel must compute. Fully masked sub-blocks are skipped, which is what makes load balancing pay off."""
    out = []
    for qs in segments(q_pos):
        for ks in segments(kpos):
            if int(kpos[ks].min()) <= int(q_pos[qs].max()):
                out.append((qs, ks))
    return out


def _expand(x: torch.Tensor, groups: int) -> torch.Tensor:
    return x if groups == 1 else x.repeat_interleave(groups, dim=1)


def block_attention(q, k, v, q_pos, k_pos, scale):
    """Attention of q (B, H, Tq, d) against one block k, v (B, H, Tk, d). Returns (o, lse); rows that see no
    key get o = 0 and lse = -inf, so merging them changes nothing."""
    s = (q @ k.transpose(-1, -2)) * scale
    visible = k_pos[None, :] <= q_pos[:, None]
    s = s.masked_fill(~visible, float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    p = torch.exp(s - lse[..., None])
    p = torch.nan_to_num(p, nan=0.0)                  # fully masked rows: exp(-inf - -inf)
    return p @ v, lse


def merge(o_a, lse_a, o_b, lse_b):
    """Combine two partial attention results over disjoint key sets (online softmax)."""
    lse = torch.logaddexp(lse_a, lse_b)
    wa = torch.nan_to_num(torch.exp(lse_a - lse), nan=0.0)[..., None]
    wb = torch.nan_to_num(torch.exp(lse_b - lse), nan=0.0)[..., None]
    return wa * o_a + wb * o_b, lse


def _ring_exchange(send: torch.Tensor, group=None):
    """Start sending ``send`` to the next rank and receiving the previous rank's tensor. Returns (buffer, reqs)."""
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    nxt = dist.get_global_rank(group, (rank + 1) % world) if group is not None else (rank + 1) % world
    prv = dist.get_global_rank(group, (rank - 1) % world) if group is not None else (rank - 1) % world
    recv = torch.empty_like(send)
    reqs = [dist.isend(send.contiguous(), nxt, group=group), dist.irecv(recv, prv, group=group)]
    return recv, reqs


class _RingAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, q_pos, k_pos, scale, group, stats):
        world = dist.get_world_size(group)
        g = q.shape[1] // k.shape[1]
        o = torch.zeros_like(q)
        lse = torch.full(q.shape[:-1], float("-inf"), dtype=q.dtype, device=q.device)
        kv = torch.stack((k, v))
        kpos = k_pos.to(torch.int64)
        for step in range(world):
            reqs = []
            if step < world - 1:                      # start moving the next block before computing this one
                kv_next, r1 = _ring_exchange(kv, group)
                kpos_next, r2 = _ring_exchange(kpos, group)
                reqs = r1 + r2
            pairs = _visible_pairs(q_pos, kpos)           # skip sub-blocks entirely in the future
            if pairs:
                t0 = _now()
                ke, ve = _expand(kv[0], g), _expand(kv[1], g)
                for qs, ks in pairs:
                    ob, lb = block_attention(q[:, :, qs], ke[:, :, ks], ve[:, :, ks], q_pos[qs], kpos[ks], scale)
                    o[:, :, qs], lse[:, :, qs] = merge(o[:, :, qs], lse[:, :, qs], ob, lb)
                if stats is not None:
                    stats["compute_s"] = stats.get("compute_s", 0.0) + _now() - t0
            t1 = _now()
            for r in reqs:
                r.wait()
            if stats is not None:
                stats["wait_s"] = stats.get("wait_s", 0.0) + _now() - t1
            if reqs:
                kv, kpos = kv_next, kpos_next
        ctx.save_for_backward(q, k, v, o, lse, q_pos, k_pos)
        ctx.scale, ctx.group = scale, group
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, q_pos, k_pos = ctx.saved_tensors
        scale, group = ctx.scale, ctx.group
        world = dist.get_world_size(group)
        B, KV, Tk, d = k.shape
        g = q.shape[1] // KV
        D = (do * o).sum(-1, keepdim=True)            # rowsum(dO ∘ O)
        dq = torch.zeros_like(q)
        kv = torch.stack((k, v))
        dkv = torch.zeros_like(kv)
        kpos = k_pos.to(torch.int64)
        for step in range(world):
            reqs = []
            if step < world - 1:
                kv_next, r1 = _ring_exchange(kv, group)
                kpos_next, r2 = _ring_exchange(kpos, group)
                reqs = r1 + r2
            pairs = _visible_pairs(q_pos, kpos)
            if pairs:
                ke, ve = _expand(kv[0], g), _expand(kv[1], g)
                dk_e, dv_e = torch.zeros_like(ke), torch.zeros_like(ve)
                for qs, ks in pairs:
                    qb, dob = q[:, :, qs], do[:, :, qs]
                    s = (qb @ ke[:, :, ks].transpose(-1, -2)) * scale
                    visible = kpos[ks][None, :] <= q_pos[qs][:, None]
                    p = torch.exp(s - lse[:, :, qs][..., None]).masked_fill(~visible, 0.0)
                    p = torch.nan_to_num(p, nan=0.0)
                    dv_e[:, :, ks] += p.transpose(-1, -2) @ dob
                    ds = p * (dob @ ve[:, :, ks].transpose(-1, -2) - D[:, :, qs])
                    dq[:, :, qs] += (ds @ ke[:, :, ks]) * scale
                    dk_e[:, :, ks] += (ds.transpose(-1, -2) @ qb) * scale
                if g > 1:                             # fold the expanded heads back onto their KV head
                    dk_e = dk_e.view(B, KV, g, Tk, d).sum(2)
                    dv_e = dv_e.view(B, KV, g, Tk, d).sum(2)
                dkv = dkv + torch.stack((dk_e, dv_e))
            for r in reqs:
                r.wait()
            # the accumulator follows its block: send it on (the last hop brings every block home)
            dkv, rd = _ring_exchange(dkv, group)
            for r in rd:
                r.wait()
            if reqs:
                kv, kpos = kv_next, kpos_next
        return dq, dkv[0], dkv[1], None, None, None, None, None


def _now() -> float:
    import time
    return time.perf_counter()


def ring_attention(q, k, v, q_pos, k_pos, scale: float | None = None, group=None, stats: dict | None = None):
    """Causal attention of this rank's queries against the keys of *all* ranks, by passing K/V around the ring.

    q (B, H, T_local, d); k, v (B, KV, T_local, d) with H a multiple of KV; q_pos, k_pos (T_local,) the absolute
    positions of this rank's tokens (usually equal). Differentiable in q, k, v. ``stats`` (optional dict)
    accumulates ``compute_s`` (block attention time) and ``wait_s`` (time spent waiting for the next block).
    """
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    return _RingAttention.apply(q, k, v, q_pos, k_pos, scale, group, stats)


def simulate_ring(q, k, v, positions: list[torch.Tensor], merge_fn=merge, scale: float | None = None):
    """Single-process model of the ring forward: ``positions[r]`` are rank r's token positions.

    Returns the list of per-rank outputs. Used to test a merge rule without starting processes.
    """
    n = len(positions)
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    g = q.shape[1] // k.shape[1]
    outs = []
    for r in range(n):
        qr = q[:, :, positions[r]]
        o = torch.zeros_like(qr)
        lse = torch.full(qr.shape[:-1], float("-inf"), dtype=q.dtype)
        for step in range(n):
            src = (r - step) % n
            kp = positions[src]
            ob, lb = block_attention(qr, _expand(k[:, :, kp], g), _expand(v[:, :, kp], g), positions[r], kp, scale)
            o, lse = merge_fn(o, lse, ob, lb)
        outs.append(o)
    return outs


# ---- worker for tests and the lab (run under frontierlab.perf.dist.spawn) ---------------------------

def equivalence_worker(rank, world, T=16, B=1, H=4, KV=2, d=8, balanced=True, seed=0, dtype="float64"):
    """Ring attention on this rank's shard vs single-device attention on the whole sequence (same inputs)."""
    dt = getattr(torch, dtype)
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, T, d, generator=g, dtype=dt)
    k = torch.randn(B, KV, T, d, generator=g, dtype=dt)
    v = torch.randn(B, KV, T, d, generator=g, dtype=dt)
    w = torch.randn(B, H, T, d, generator=g, dtype=dt)        # the loss is sum(o * w): a random upstream gradient
    qr, kr, vr = (x.clone().requires_grad_(True) for x in (q, k, v))
    ref = torch.nn.functional.scaled_dot_product_attention(qr, kr, vr, is_causal=True, enable_gqa=H != KV)
    (ref * w).sum().backward()
    pos = shard_positions(T, world, rank, balanced)
    ql, kl, vl = (x[:, :, pos].clone().requires_grad_(True) for x in (q, k, v))
    stats: dict = {}
    out = ring_attention(ql, kl, vl, pos, pos, stats=stats)
    (out * w[:, :, pos]).sum().backward()
    err = lambda a, b: float((a.detach() - b.detach()).abs().max())   # noqa: E731
    return {"out": err(out, ref[:, :, pos]), "dq": err(ql.grad, qr.grad[:, :, pos]),
            "dk": err(kl.grad, kr.grad[:, :, pos]), "dv": err(vl.grad, vr.grad[:, :, pos]),
            "pairs": causal_pairs(pos, torch.arange(T)), **stats}


def timing_worker(rank, world, T=4096, B=1, H=8, KV=8, d=64, balanced=True, repeats=5, seed=0, device="cpu"):
    """Forward + backward time of ring attention on this rank (float32 on CPU, bf16 on CUDA), with compute and
    wait split out. On CUDA the compute/wait split is only indicative (kernels are asynchronous); the step time
    is synchronised."""
    import time
    dev = torch.device("cpu" if device == "cpu" else f"cuda:{rank}")
    dt_ = torch.float32 if dev.type == "cpu" else torch.bfloat16
    g = torch.Generator().manual_seed(seed + rank)
    pos = shard_positions(T, world, rank, balanced).to(dev)
    Tl = len(pos)
    q = torch.randn(B, H, Tl, d, generator=g).to(dev, dt_).requires_grad_(True)
    k = torch.randn(B, KV, Tl, d, generator=g).to(dev, dt_).requires_grad_(True)
    v = torch.randn(B, KV, Tl, d, generator=g).to(dev, dt_).requires_grad_(True)

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    times, comp, wait = [], [], []
    for i in range(repeats + 1):
        stats: dict = {}
        dist.barrier()
        sync()
        t0 = time.perf_counter()
        out = ring_attention(q, k, v, pos, pos, stats=stats)
        out.sum().backward()
        sync()
        dt = time.perf_counter() - t0
        if i:                                            # first call is warm-up
            times.append(dt)
            comp.append(stats.get("compute_s", 0.0))
            wait.append(stats.get("wait_s", 0.0))
    med = lambda xs: sorted(xs)[len(xs) // 2]                    # noqa: E731
    return {"step_s": med(times), "fwd_compute_s": med(comp), "fwd_wait_s": med(wait), "samples": times,
            "pairs": causal_pairs(pos.cpu(), torch.arange(T))}


__all__ = ["block_attention", "causal_pairs", "segments", "equivalence_worker", "merge", "ring_attention", "shard_positions",
           "simulate_ring", "timing_worker", "work_per_rank"]
