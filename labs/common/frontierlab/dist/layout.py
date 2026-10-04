"""Parallelism layout planner: memory and communication per device (lesson 09.1).

A *layout* splits a training job over ``world = tp · cp · pp · dp`` devices:

* **TP** (tensor parallel) splits every weight matrix of a layer over ``tp`` devices; with sequence
  parallelism (SP) the activations between the matmuls are split along the sequence too.
* **CP** (context parallel) splits each sequence into ``cp`` pieces; attention needs the keys and values
  of the other pieces (ring attention or an all-gather of K/V).
* **PP** (pipeline parallel) gives each of ``pp`` stages ``L/pp`` consecutive layers.
* **DP** (data parallel) replicates the rest and splits the batch. With ZeRO stage ``z`` the optimizer
  state (z >= 1), the gradients (z >= 2) and the weights (z = 3, FSDP) are sharded over the DP group.
  CP ranks hold the same weights as DP ranks, so the weight-replication group is ``dp · cp``.
* **EP** (expert parallel) splits the routed experts of every MoE layer over ``ep`` devices. Here EP is
  carved out of the data-parallel dimension (``dp % ep == 0``) and experts are not tensor-split — the
  DeepSeek-V3 arrangement (no TP). Expert weights are replicated over ``dp · cp / ep`` devices.

Rank placement: ``order`` lists the dimensions from innermost (consecutive ranks) to outermost. Llama 3
(arXiv 2407.21783 section 3.3.2) orders them ``[TP, CP, PP, DP]`` because "the innermost parallelism
requires the highest network bandwidth and lowest latency"; :func:`group_span` says which groups stay
inside one node.

Every number this module returns is an **estimate from formulas** (bytes and FLOPs), labelled as such
wherever a lesson prints it. The formulas, per device, for one optimizer step of ``n_micro`` micro-batches
of ``b`` sequences of ``s`` tokens (``s_l = s / cp`` tokens per CP rank, ``h`` hidden size, ``e`` bytes
per activation element):

* weights + grads + master + optimizer: ``P_dense/tp · (w + g/z2 + (m + o)/z1)`` with the ZeRO divisors,
  plus the same for expert weights over their own group;
* activations, per layer and micro-batch (no dropout, FlashAttention so no s² term, SP on):
  ``s_l·b·e · (2h [2 norms] + h [attn in] + Hd_q + KV(d_q + d_v) + Hd_v [attention] + h [FFN in] + 4·I_act [SwiGLU])/tp``
  (the Korthikanti et al. 2022 accounting, arXiv 2205.05198, rewritten for GQA/MLA and SwiGLU; for a GPT
  layer with MHA and a 4h GeLU MLP the same bookkeeping gives their ``32·s·b·h`` without dropout masks);
  full recomputation keeps only the layer input, ``2·s_l·b·h/tp``;
* in-flight micro-batches on the first stage: ``pp`` (1F1B), ``pp·(1 + (pp-1)/(pp·v))`` (interleaved,
  v chunks), ``n_micro`` (GPipe), ``pp + 1`` chunks of ``L/pp`` layers (DualPipe, its README);
* communication bytes sent per device (ring algorithms): see :func:`comm_per_step`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from frontierlab.calc import ArchSpec, flops_per_token, param_counts
from frontierlab.calc.arith import attention_params, expert_params, ffn_params

DIMS = ("tp", "cp", "pp", "ep", "dp")   # "dp" in an order means the data-parallel ranks not used by EP


@dataclass(frozen=True)
class Layout:
    tp: int = 1
    cp: int = 1
    pp: int = 1
    dp: int = 1
    ep: int = 1
    vpp: int = 1                         # virtual pipeline chunks per PP rank (interleaved 1F1B)
    zero: int = 1                        # 0 = DDP, 1 = shard optimizer, 2 = + gradients, 3 = + weights (FSDP)
    order: tuple = ("tp", "cp", "pp", "ep", "dp")   # innermost first

    @property
    def world(self) -> int:
        return self.tp * self.cp * self.pp * self.dp

    def sizes(self) -> dict:
        return {"tp": self.tp, "cp": self.cp, "pp": self.pp, "ep": self.ep, "dp": self.dp // self.ep}

    def describe(self) -> str:
        s = f"TP{self.tp} CP{self.cp} PP{self.pp} DP{self.dp}"
        if self.ep > 1:
            s += f" (EP{self.ep} inside DP)"
        if self.vpp > 1:
            s += f" v={self.vpp}"
        return s + f" ZeRO-{self.zero}"


@dataclass(frozen=True)
class Train:
    seq: int = 8192                      # tokens per sequence
    micro_batch: int = 1                 # sequences per micro-batch per DP rank
    n_micro: int = 16                    # micro-batches per optimizer step per DP rank
    schedule: str = "1f1b"               # "1f1b" | "interleaved" | "gpipe" | "dualpipe"
    recompute: str = "none"              # "none" | "full"
    act_bytes: int = 2                   # bf16 activations
    w_bytes: float = 2                   # bf16 working weights
    g_bytes: float = 4                   # fp32 gradients (reduced in fp32)
    master_bytes: float = 4              # fp32 master weights
    optim_bytes: float = 8               # AdamW m and v in fp32 (DeepSeek-V3 keeps them in bf16: 4)
    comm_g_bytes: float = 4              # dtype of the gradient reduction
    sequence_parallel: bool = True

    @property
    def tokens_per_dp_rank(self) -> int:
        return self.seq * self.micro_batch * self.n_micro


@dataclass(frozen=True)
class Cluster:
    name: str = "H100 SXM, 8 per node"
    gpus_per_node: int = 8
    hbm_bytes: float = 80e9
    peak_flops: float = 989e12           # dense BF16
    intra_bw: float = 300e9              # achieved bus bandwidth inside a node (bytes/s, ASSUMED until measured)
    inter_bw: float = 40e9               # achieved per-GPU bandwidth between nodes (bytes/s, ASSUMED)
    notes: str = "bandwidths are assumptions: replace them with nccl-tests bus bandwidth measured on your cluster"


H100_NODE = Cluster()
H800_NODE = Cluster(name="H800, 8 per node (DeepSeek-V3 section 3.1)", intra_bw=160e9, inter_bw=50e9,
                    notes="DeepSeek-V3 section 3.1: NVLink 160 GB/s, InfiniBand 50 GB/s (link figures, not measured)")


# ---- the model --------------------------------------------------------------------------------

def llama3_405b() -> ArchSpec:
    """Llama 3 405B as described in arXiv 2407.21783 Table 3 (126 layers, width 16,384, FFN 53,248,
    128 heads, 8 KV heads, vocabulary 128,000 plus specials = 128,256)."""
    L = 126
    s = ArchSpec(name="Llama 3 405B", model_type="llama", vocab_size=128256, hidden_size=16384, num_layers=L,
                 layer_kinds=["full"] * L, num_heads=128, num_kv_heads=8, head_dim=128, v_head_dim=128,
                 dense_intermediate=53248, tie_embeddings=False, max_context=131072)
    s.dense_layers = list(range(L))
    return s


def _layer_params(spec: ArchSpec, i: int) -> dict:
    """Parameters of layer i split into ``dense`` (replicated under EP) and ``expert`` (routed experts)."""
    attn = attention_params(spec, spec.layer_kinds[i])
    f = ffn_params(spec, i)
    norms = spec.norms_per_layer * spec.hidden_size
    if spec.is_moe and i not in spec.dense_layers:
        routed = spec.n_experts * expert_params(spec)
        return {"dense": attn + norms + f["total"] - routed, "expert": routed}
    return {"dense": attn + norms + f["total"], "expert": 0}


def stage_layers(num_layers: int, pp: int) -> list[range]:
    """Contiguous layer ranges per stage; the first stages get the remainder."""
    base, extra = divmod(num_layers, pp)
    out, start = [], 0
    for r in range(pp):
        n = base + (1 if r < extra else 0)
        out.append(range(start, start + n))
        start += n
    return out


def stage_params(spec: ArchSpec, pp: int) -> list[dict]:
    """Dense and expert parameters held by each pipeline stage (embedding on the first, head on the last)."""
    pc = param_counts(spec)
    out = []
    for r, layers in enumerate(stage_layers(spec.num_layers, pp)):
        d = e = 0
        for i in layers:
            lp = _layer_params(spec, i)
            d, e = d + lp["dense"], e + lp["expert"]
        if r == 0:
            d += pc["embedding"]
        if r == pp - 1:
            d += pc["head"] + spec.hidden_size          # head (0 if tied) + final norm
            if spec.tie_embeddings and pp > 1:
                d += pc["embedding"]                    # the last stage needs its own copy of a tied head
        out.append({"layers": len(layers), "dense": d, "expert": e})
    return out


# ---- placement ----------------------------------------------------------------------------------

def strides(layout: Layout) -> dict:
    """Rank stride of each dimension for ``layout.order`` (innermost first)."""
    sz, out, st = layout.sizes(), {}, 1
    for d in layout.order:
        out[d] = st
        st *= sz[d]
    return out


def group_ranks(layout: Layout, dims: tuple, rank: int = 0) -> list[int]:
    """Ranks in the group that contains ``rank`` and varies along ``dims`` (e.g. ("ep", "dp") is the DP group)."""
    sz, st = layout.sizes(), strides(layout)
    coords = {d: (rank // st[d]) % sz[d] for d in DIMS}
    base = rank - sum(coords[d] * st[d] for d in dims)
    ranks = [base]
    for d in dims:
        ranks = [r + i * st[d] for r in ranks for i in range(sz[d])]
    return sorted(ranks)


def group_span(layout: Layout, dims: tuple, gpus_per_node: int) -> int:
    """Number of nodes the group of rank 0 along ``dims`` touches (1 = stays inside one node)."""
    return len({r // gpus_per_node for r in group_ranks(layout, dims)})


GROUPS = {"tp": ("tp",), "cp": ("cp",), "pp": ("pp",), "dp": ("ep", "dp"), "ep": ("ep",),
          "edp": ("dp",)}   # edp: replicas of one expert shard (DP ranks outside the EP group)


def check(spec: ArchSpec, layout: Layout, train: Train) -> list[str]:
    """Problems that make a layout invalid or wasteful, as readable strings (empty list = fine)."""
    p = []
    if layout.dp % layout.ep:
        p.append(f"ep={layout.ep} must divide dp={layout.dp} (EP is carved out of DP here)")
    if spec.num_heads % layout.tp or (spec.num_kv_heads and spec.attention != "mla" and spec.num_kv_heads % layout.tp):
        p.append(f"tp={layout.tp} does not divide the heads ({spec.num_heads} query, {spec.num_kv_heads} KV)")
    if spec.is_moe and spec.n_experts % layout.ep:
        p.append(f"ep={layout.ep} does not divide {spec.n_experts} experts")
    if train.seq % (2 * layout.cp) and layout.cp > 1:
        p.append(f"seq={train.seq} is not divisible into 2*cp={2 * layout.cp} load-balanced chunks")
    if train.schedule == "interleaved" and train.n_micro % layout.pp:
        p.append(f"interleaved 1F1B needs n_micro ({train.n_micro}) to be a multiple of pp ({layout.pp})")
    if train.schedule == "dualpipe" and (layout.pp % 2 or train.n_micro < 2 * layout.pp):
        p.append("DualPipe needs an even pp and at least 2*pp micro-batches (DualPipe README)")
    if layout.zero not in (0, 1, 2, 3):
        p.append("zero must be 0..3")
    return p


# ---- memory ---------------------------------------------------------------------------------------

def activation_bytes_per_layer(spec: ArchSpec, layout: Layout, train: Train, moe_layer: bool) -> float:
    """Saved activation bytes of one layer for one micro-batch on one device (estimate, see module docstring)."""
    s_l, b, e, h = train.seq / layout.cp, train.micro_batch, train.act_bytes, spec.hidden_size
    t = layout.tp
    if train.recompute == "full":
        return 2 * s_l * b * h / (t if train.sequence_parallel else 1)
    H, KV = spec.num_heads, (spec.num_heads if spec.attention == "mla" else spec.num_kv_heads)
    dq = spec.head_dim if spec.attention != "mla" else spec.qk_nope_dim + spec.qk_rope_dim
    dv = spec.v_head_dim or spec.head_dim
    if moe_layer:
        i_act = spec.top_k * spec.expert_intermediate + spec.shared_intermediate
        dispatched = spec.top_k * h            # each token's input is copied to its k experts
    else:
        i_act, dispatched = spec.dense_intermediate, 0
    seq_sharded = 2 * h + h + h                # two norm inputs + the attention and FFN matmul inputs
    tp_sharded = H * dq + KV * (dq + dv) + H * dv + 4 * i_act + dispatched
    per_elem = (seq_sharded / (t if train.sequence_parallel else 1)) + tp_sharded / t
    return s_l * b * e * per_elem


def in_flight(layout: Layout, train: Train) -> float:
    """Micro-batches whose activations the first stage holds at its peak (in units of one stage's layers)."""
    p = layout.pp
    if p == 1:
        return 1
    return {"1f1b": p, "interleaved": p * (1 + (p - 1) / (p * layout.vpp)), "gpipe": train.n_micro,
            "dualpipe": p + 1}[train.schedule]


def memory_per_device(spec: ArchSpec, layout: Layout, train: Train) -> dict:
    """Bytes per device on the most loaded pipeline stage, broken down (estimates)."""
    stages = stage_params(spec, layout.pp)
    dense_rep, expert_rep = layout.dp * layout.cp, layout.dp * layout.cp // layout.ep
    z = layout.zero

    def state(n_params, rep):
        w = n_params * train.w_bytes / (rep if z >= 3 else 1)
        g = n_params * train.g_bytes / (rep if z >= 2 else 1)
        mo = n_params * (train.master_bytes + train.optim_bytes) / (rep if z >= 1 else 1)
        return w, g, mo

    best = None
    for r, st in enumerate(stages):
        nd, ne = st["dense"] / layout.tp, st["expert"] / layout.ep
        wd, gd, md = state(nd, dense_rep)
        we, ge, me = state(ne, expert_rep)
        layers = stage_layers(spec.num_layers, layout.pp)[r]
        moe = [spec.is_moe and i not in spec.dense_layers for i in layers]
        act_one = sum(activation_bytes_per_layer(spec, layout, train, m) for m in moe)
        # stage r of 1F1B holds (pp - r) micro-batches; the first stage is the peak
        flight = in_flight(layout, train) if r == 0 else max(1, in_flight(layout, train) - r)
        act = act_one * flight
        gathered = 0.0
        if z >= 3 and layers:                   # FSDP: one gathered layer (plus one prefetched) in bf16
            per_layer = (nd + ne) / max(1, len(layers))
            gathered = 2 * per_layer * train.w_bytes
        row = {"stage": r, "layers": len(layers), "params_dense": nd, "params_expert": ne,
               "weights": wd + we, "grads": gd + ge, "optimizer": md + me, "activations": act,
               "fsdp_gathered": gathered}
        row["total"] = row["weights"] + row["grads"] + row["optimizer"] + act + gathered
        if best is None or row["total"] > best["total"]:
            best = row
    return best


# ---- communication ---------------------------------------------------------------------------------

def comm_per_step(spec: ArchSpec, layout: Layout, train: Train) -> dict:
    """Bytes each device sends per optimizer step, per parallelism dimension (ring algorithms).

    * TP with SP: per layer and micro-batch 2 all-gathers + 2 reduce-scatters forward and the same backward,
      each over a (s_l·b, h) activation: ``8·(t-1)/t · s_l·b·h·e`` (without SP: 4 all-reduces, the same bytes).
    * CP (ring, pass-KV): forward passes the local K and V around the ring ``cp-1`` times; backward passes
      K, V and their gradients: ``3·(cp-1) · s_l·b·KV·(d_q + d_v)·e`` per layer and micro-batch.
    * PP: each stage boundary sends the activation forward and its gradient back:
      ``2 · s_l·b·h·e / t`` per micro-batch and chunk (``v`` chunks when interleaved).
    * DP over n replicas, with G the local gradient bytes (``comm_g_bytes`` each) and W the local weight bytes:
      ZeRO-0 (DDP) all-reduces the gradients, ``2·(n-1)/n · G``; ZeRO-1/2 reduce-scatter the gradients and
      all-gather the updated weights, ``(n-1)/n · (G + W)``; ZeRO-3 reduce-scatters the gradients and all-gathers
      the weights in the forward and the backward of every micro-batch, ``(n-1)/n · (G + 2·n_micro·W)``. Experts
      reduce over their own replica group.
    * EP: per MoE layer and micro-batch, dispatch and combine all-to-all forward and backward:
      ``4 · (ep-1)/ep · tokens·k·h·e`` (uniform routing; the fraction (ep-1)/ep leaves the device).
    """
    t, c, p, ep = layout.tp, layout.cp, layout.pp, layout.ep
    s_l, b, e, h, m = train.seq / c, train.micro_batch, train.act_bytes, spec.hidden_size, train.n_micro
    stage = max(stage_params(spec, p), key=lambda r: r["dense"] + r["expert"])
    L_dev = stage["layers"]
    n_moe = sum(1 for i in stage_layers(spec.num_layers, p)[0] if spec.is_moe and i not in spec.dense_layers)
    out = {}
    out["tp"] = 8 * (t - 1) / t * s_l * b * h * e * L_dev * m if t > 1 else 0.0
    if c > 1:
        KV = spec.num_heads if spec.attention == "mla" else spec.num_kv_heads
        dq = spec.head_dim if spec.attention != "mla" else spec.qk_nope_dim + spec.qk_rope_dim
        dv = spec.v_head_dim or spec.head_dim
        kv_local = s_l * b * (KV / t) * (dq + dv) * e
        out["cp"] = 3 * (c - 1) * kv_local * L_dev * m
    else:
        out["cp"] = 0.0
    out["pp"] = 2 * s_l * b * h * e / t * m * layout.vpp if p > 1 else 0.0
    n_d, n_e = layout.dp * c, layout.dp * c // ep
    nd_, ne_ = stage["dense"] / t, stage["expert"] / ep          # parameters on this device

    def dp_volume(n, params):
        if n <= 1 or params == 0:
            return 0.0
        G, W, f = params * train.comm_g_bytes, params * train.w_bytes, (n - 1) / n
        if layout.zero == 0:
            return 2 * f * G                                       # all-reduce of the gradients
        if layout.zero in (1, 2):
            return f * G + f * W                                   # reduce-scatter grads, all-gather updated weights
        return f * G + 2 * m * f * W                               # ZeRO-3: weights gathered in fwd and bwd of every micro-batch

    dp_bytes = dp_volume(n_d, nd_) + dp_volume(n_e, ne_)
    out["dp"] = dp_bytes
    tokens = s_l * b / (t if train.sequence_parallel else 1)
    out["ep"] = 4 * (ep - 1) / ep * tokens * spec.top_k * h * e * n_moe * m if (ep > 1 and spec.is_moe) else 0.0
    return out


def plan(spec: ArchSpec, layout: Layout, train: Train, cluster: Cluster = H100_NODE, mfu: float = 0.40) -> dict:
    """Memory, communication and a compute-vs-communication time estimate for one optimizer step.

    ``compute_s`` = training FLOPs of the device's tokens ÷ (peak × ``mfu``), with the model's FLOPs split
    evenly over TP, CP and PP ranks. Communication time per dimension = bytes ÷ the bandwidth of the link
    that dimension's group uses (``intra_bw`` if it stays in one node, else ``inter_bw``). These are
    upper bounds on *exposed* communication: overlap hides part of it, which only a measurement shows.
    The pipeline bubble fraction uses lesson 09.2's formulas.
    """
    problems = check(spec, layout, train)
    notes = []
    if spec.num_layers % (layout.pp * layout.vpp):
        sizes = sorted({len(r) for r in stage_layers(spec.num_layers, layout.pp)})
        notes.append(f"{spec.num_layers} layers do not split evenly over pp={layout.pp} (v={layout.vpp}): "
                     f"stages hold {sizes} layers; the planner reports the most loaded stage")
    mem = memory_per_device(spec, layout, train)
    comm = comm_per_step(spec, layout, train)
    span = {d: group_span(layout, GROUPS[d], cluster.gpus_per_node) for d in ("tp", "cp", "pp", "dp", "ep")}
    bw = {d: (cluster.intra_bw if span[d] == 1 else cluster.inter_bw) for d in span}
    comm_s = {d: comm[d] / bw[d] for d in comm}
    tokens = train.tokens_per_dp_rank
    flops = flops_per_token(spec, train.seq) * tokens / (layout.tp * layout.cp * layout.pp)
    compute_s = flops / (cluster.peak_flops * mfu)
    p, v, m = layout.pp, layout.vpp, train.n_micro
    bubble = 0.0 if p == 1 else {"1f1b": (p - 1) / m, "gpipe": (p - 1) / m, "interleaved": (p - 1) / (v * m),
                                 "dualpipe": (p / 2 - 1) / m}[train.schedule]
    if mem["total"] > cluster.hbm_bytes:
        problems.append(f"needs {mem['total'] / 1e9:.1f} GB per device, more than {cluster.hbm_bytes / 1e9:.0f} GB")
    return {"layout": layout.describe(), "world": layout.world, "problems": problems, "notes": notes, "memory": mem,
            "comm_bytes": comm, "nodes_spanned": span, "comm_s": comm_s, "compute_s": compute_s,
            "bubble_fraction": bubble, "global_batch_tokens": tokens * layout.dp,
            "step_s_no_overlap": compute_s * (1 + bubble) + sum(comm_s.values())}


def format_plan(r: dict) -> str:
    gb = 1e9
    m = r["memory"]
    lines = [f"{r['layout']}  (world {r['world']}, global batch {r['global_batch_tokens'] / 1e6:.1f}M tokens)",
             f"  memory per device (stage {m['stage']}, {m['layers']} layers): weights {m['weights'] / gb:.1f} GB, "
             f"grads {m['grads'] / gb:.1f}, optimizer {m['optimizer'] / gb:.1f}, activations {m['activations'] / gb:.1f}, "
             f"FSDP gathered {m['fsdp_gathered'] / gb:.1f}  -> total {m['total'] / gb:.1f} GB  [ESTIMATE]"]
    parts = []
    for d in ("tp", "cp", "pp", "dp", "ep"):
        if r["comm_bytes"][d]:
            where = "intra-node" if r["nodes_spanned"][d] == 1 else f"{r['nodes_spanned'][d]} nodes"
            parts.append(f"{d.upper()} {r['comm_bytes'][d] / gb:.2f} GB ({where}, {r['comm_s'][d] * 1e3:.0f} ms)")
    lines.append("  communication per step: " + (", ".join(parts) if parts else "none"))
    lines.append(f"  compute {r['compute_s']:.2f} s at the assumed MFU, bubble {r['bubble_fraction']:.1%}, "
                 f"step without any overlap {r['step_s_no_overlap']:.2f} s  [PROJECTED]")
    for p in r["problems"]:
        lines.append(f"  ! {p}")
    for n in r.get("notes", []):
        lines.append(f"  note: {n}")
    return "\n".join(lines)


def with_(layout: Layout, **kw) -> Layout:
    return replace(layout, **kw)


__all__ = ["Cluster", "DIMS", "GROUPS", "H100_NODE", "H800_NODE", "Layout", "Train", "activation_bytes_per_layer",
           "check", "comm_per_step", "format_plan", "group_ranks", "group_span", "in_flight", "llama3_405b",
           "memory_per_device", "plan", "stage_layers", "stage_params", "strides"]
