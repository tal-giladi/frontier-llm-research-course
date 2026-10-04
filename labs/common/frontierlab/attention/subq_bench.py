"""Component-level cost of dense, linear and DSA-style attention, measured with the Module 2 method (05.3).

One attention layer's *mixing* step (projections excluded: they cost the same 2·N FLOPs per token in every
design and are measured separately) is split into components and each is timed on its own with
:func:`frontierlab.perf.timing.benchmark` (warm-up, synchronisation, repeats, bootstrap interval):

    dense      sdpa         ``F.scaled_dot_product_attention`` causal (FlashAttention on CUDA)
    linear     core         chunked gated delta rule (our PyTorch reference on CPU; fla ``chunk_kda`` on CUDA)
               step         one recurrent decode step
    dsa        indexer      I = sum_j w_j ReLU(q^I_j . k^I) for every (query, key) pair: the O(L^2) part
               topk         top-k over each query's scores
               gather       collecting the k selected keys and values per query: memory traffic
               attend       attention over the k gathered keys: the O(L k) part

Prefill (T = L new tokens, no cache) processes queries in blocks of ``block`` for the indexer, top-k and
gather, as a fused kernel would; nothing of size L x L is kept. Decode is one query against L cached
tokens. The whole-layer arms are also compared in interleaved rounds (:func:`frontierlab.perf.timing.interleaved`)
so drift hits all arms equally, with paired speed-ups.

``bytes_moved`` gives the minimum HBM traffic of each component (each input read once, output written once)
for the roofline projection of lesson 02.1. Op counts come from ``torch.profiler`` (CPU: aten ops; CUDA:
kernel launches).

Every timing depends on the machine and the kernel. A CPU number says nothing about a GPU; a PyTorch
gather path says nothing about a fused sparse kernel (DeepSeek's are not used here). The lesson labels
them accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from frontierlab.attention import deltanet
from frontierlab.perf.timing import benchmark, interleaved, speedup, sync


@dataclass
class Shape:
    H: int = 6          # query heads
    KV: int = 2         # key/value heads
    d: int = 64         # head dim (also linear d_k = d_v)
    HI: int = 4         # indexer heads
    dI: int = 32        # indexer head dim
    k: int = 256        # DSA top-k
    chunk: int = 16     # linear chunk size

    @classmethod
    def of(cls, cfg, k: int = 256, chunk: int = 16, HI: int = 4, dI: int = 32) -> "Shape":
        return cls(cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, HI, dI, k, chunk)


def make_inputs(sh: Shape, L: int, *, decode: bool, device="cpu", dtype=torch.float32, seed: int = 0) -> dict:
    """Random tensors of the right shapes for one layer at context L (B = 1)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    T = 1 if decode else L
    r = lambda *s: torch.randn(*s, generator=g).to(device=device, dtype=dtype)  # noqa: E731
    x = {"q": r(1, sh.H, T, sh.d), "k": r(1, sh.KV, L, sh.d), "v": r(1, sh.KV, L, sh.d),
         "iq": r(1, sh.HI, T, sh.dI), "ik": r(1, L, sh.dI), "w": r(1, T, sh.HI),
         "q_pos": torch.arange(L - T, L, device=device), "k_pos": torch.arange(L, device=device)}
    lq, lk = r(1, sh.H, T, sh.d) * sh.d ** -0.5, deltanet.l2norm(r(1, sh.H, T, sh.d))
    x.update(lq=lq, lk=lk, lv=r(1, sh.H, T, sh.d),
             lg=-torch.rand(1, sh.H, T, sh.d, generator=g).to(device=device, dtype=dtype) * 0.1,
             lbeta=torch.rand(1, sh.H, T, generator=g).to(device=device, dtype=dtype),
             state=torch.zeros(1, sh.H, sh.d, sh.d, device=device, dtype=torch.float32))
    return x


# ------------------------------------------------------------------------------------- components

def dense_sdpa(x, decode: bool):
    if decode:       # one query sees every cached key: no mask needed
        return F.scaled_dot_product_attention(x["q"], x["k"], x["v"], enable_gqa=True)
    return F.scaled_dot_product_attention(x["q"], x["k"], x["v"], is_causal=True, enable_gqa=True)


def linear_core(x, sh: Shape, mode: str = "chunked"):
    if mode == "fla":    # NOT RUN IN THIS BUILD (CUDA + flash-linear-attention 0.5.2)
        from fla.ops.kda import chunk_kda
        tr = (lambda t: t.transpose(1, 2).contiguous())
        o, _ = chunk_kda(tr(x["lq"]), tr(x["lk"]), tr(x["lv"]), tr(x["lg"].float()), tr(x["lbeta"]), scale=1.0,
                         initial_state=x["state"], output_final_state=True)
        return o
    if mode == "recurrent":
        return deltanet.delta_rule_recurrent(x["lq"], x["lk"], x["lv"], x["lg"], x["lbeta"], x["state"])[0]
    return deltanet.delta_rule_chunked(x["lq"], x["lk"], x["lv"], x["lg"], x["lbeta"], x["state"], sh.chunk)[0]


def _blocks(T: int, block: int):
    return [(s, min(s + block, T)) for s in range(0, T, block)]


def dsa_indexer(x, block: int):
    """Scores for every (query, key) pair, block by block; returns the list of (B, t, S) score blocks."""
    out = []
    for s, e in _blocks(x["iq"].shape[2], block):
        dots = torch.einsum("bjtd,bsd->bjts", x["iq"][:, :, s:e], x["ik"]).relu()
        sc = torch.einsum("btj,bjts->bts", x["w"][:, s:e], dots)
        allowed = x["k_pos"][None, :] <= x["q_pos"][s:e, None]
        out.append(sc.masked_fill(~allowed, float("-inf")))
    return out


def dsa_topk(scores_blocks, k: int):
    S = scores_blocks[0].shape[-1]
    return [sb.topk(min(k, S), dim=-1).indices for sb in scores_blocks]


def dsa_gather(x, idx_blocks, sh: Shape):
    """(B, KV, t, k, d) keys and values per block: the selected entries copied out of the cache."""
    out = []
    k, v = x["k"], x["v"]
    S, d = k.shape[2], k.shape[3]
    for idx in idx_blocks:
        t, kk = idx.shape[1], idx.shape[2]
        flat = idx.reshape(1, 1, t * kk, 1).expand(1, sh.KV, t * kk, d)
        kg = torch.gather(k, 2, flat).view(1, sh.KV, t, kk, d)
        vg = torch.gather(v, 2, flat).view(1, sh.KV, t, kk, d)
        out.append((kg, vg))
    return out


def dsa_attend(x, gathered, idx_blocks, scores_blocks, sh: Shape, block: int):
    outs = []
    rep = sh.H // sh.KV
    for (s, e), (kg, vg), idx, sb in zip(_blocks(x["q"].shape[2], block), gathered, idx_blocks, scores_blocks):
        valid = torch.isfinite(torch.gather(sb, -1, idx))                      # rows with < k allowed keys
        qb = x["q"][:, :, s:e].reshape(1, sh.KV, rep, e - s, sh.d)
        logits = torch.einsum("bgrtd,bgtkd->bgrtk", qb, kg) * sh.d ** -0.5
        logits = logits.masked_fill(~valid[:, None, None], float("-inf"))
        p = torch.softmax(logits.to(torch.promote_types(logits.dtype, torch.float32)), dim=-1).to(qb.dtype)
        outs.append(torch.einsum("bgrtk,bgtkd->bgrtd", p, vg).reshape(1, sh.H, e - s, sh.d))
    return torch.cat(outs, dim=2)


def dsa_full(x, sh: Shape, block: int):
    sb = dsa_indexer(x, block)
    idx = dsa_topk(sb, sh.k)
    return dsa_attend(x, dsa_gather(x, idx, sh), idx, sb, sh, block)


def dsa_reference(x, sh: Shape):
    """Same function with full masks (for the equality check of the blocked components)."""
    from frontierlab.attention.dsa import index_scores, topk_mask
    from frontierlab.attention.ops import _expand_kv
    sc = index_scores(x["iq"], x["ik"], x["w"])
    allowed = (x["k_pos"][None, :] <= x["q_pos"][:, None])[None]
    sel, _ = topk_mask(sc, allowed, sh.k)
    logits = (x["q"] @ _expand_kv(x["k"], sh.H).transpose(-1, -2)) * sh.d ** -0.5
    logits = logits.masked_fill(~sel[:, None], float("-inf"))
    p = torch.softmax(logits.to(torch.promote_types(logits.dtype, torch.float32)), -1).to(x["q"].dtype)
    return p @ _expand_kv(x["v"], sh.H)


# ------------------------------------------------------------------------------------- accounting

def component_flops_bytes(sh: Shape, L: int, decode: bool, b: int = 2, state_b: int = 4) -> dict:
    """FLOPs and minimum bytes moved for each component at context L (B = 1). b = bytes per element."""
    T = 1 if decode else L
    pairs = L if decode else L * (L + 1) / 2                     # causal (query, key) pairs
    kk = min(sh.k, L)
    sel_pairs = kk if decode else sum(min(t + 1, sh.k) for t in range(L)) if L <= 65536 else \
        sh.k * L - sh.k * sh.k / 2
    out = {
        "dense.sdpa": (4 * sh.H * sh.d * pairs, b * (2 * sh.KV * L * sh.d + 2 * sh.H * T * sh.d)),
        "dsa.indexer": (sh.HI * (2 * sh.dI + 2) * pairs, b * (L * sh.dI + T * sh.HI * (sh.dI + 1))),
        "dsa.topk": (pairs, 4 * pairs if not decode else 4 * L),          # read the fp32 scores once
        "dsa.gather": (0.0, b * 2 * 2 * sh.KV * sel_pairs * sh.d),       # read selected + write copies
        "dsa.attend": (4 * sh.H * sh.d * sel_pairs, b * (2 * sh.KV * sel_pairs * sh.d + 2 * sh.H * T * sh.d)),
    }
    if decode:
        out["linear.step"] = (7 * sh.H * sh.d * sh.d, state_b * 2 * sh.H * sh.d * sh.d + b * 4 * sh.H * sh.d)
    else:
        C = sh.chunk
        out["linear.core"] = (sh.H * L * (6 * sh.d * sh.d + C * 6.5 * sh.d),
                              b * 4 * sh.H * L * sh.d + state_b * 2 * sh.H * sh.d * sh.d * (L / C))
    return out


# ------------------------------------------------------------------------------------- measurement

def op_count(fn, device="cpu") -> int:
    """aten ops (CPU) or kernel launches (CUDA) in one call of fn, from torch.profiler."""
    from frontierlab.perf.profiling import profile_steps, summarize
    s = summarize(profile_steps(fn, steps=1, warmup=1, device=device))
    return s["n_kernels"] if torch.device(device).type == "cuda" else s["n_ops"]


def measure(sh: Shape, L: int, *, decode: bool, device="cpu", dtype=torch.float32, block: int = 512,
            warmup: int = 2, repeats: int = 7, rounds: int = 7, linear_mode: str | None = None,
            ops: bool = True) -> dict:
    """Times (median ms and 95% CI) of every component and of the whole layer per arm at context L."""
    x = make_inputs(sh, L, decode=decode, device=device, dtype=dtype)
    lin_mode = linear_mode or ("recurrent" if decode else "chunked")
    sb = dsa_indexer(x, block)
    idx = dsa_topk(sb, sh.k)
    gat = dsa_gather(x, idx, sh)
    comps = {
        "dense.sdpa": lambda: dense_sdpa(x, decode),
        ("linear.step" if decode else "linear.core"): lambda: linear_core(x, sh, lin_mode),
        "dsa.indexer": lambda: dsa_indexer(x, block),
        "dsa.topk": lambda: dsa_topk(sb, sh.k),
        "dsa.gather": lambda: dsa_gather(x, idx, sh),
        "dsa.attend": lambda: dsa_attend(x, gat, idx, sb, sh, block),
    }
    res = {"L": L, "mode": "decode" if decode else "prefill", "components": {}}
    fb = component_flops_bytes(sh, L, decode, b=torch.tensor([], dtype=dtype).element_size())
    for name, fn in comps.items():
        t = benchmark(fn, warmup=warmup, repeats=repeats, device=device)
        lo, hi = t.ci()
        flops, nbytes = fb[name]
        res["components"][name] = {"ms": t.median * 1e3, "ci_ms": (lo * 1e3, hi * 1e3), "cv": t.std / t.mean,
                                   "flops": flops, "bytes": nbytes}
    arms = {"dense": comps["dense.sdpa"], "linear": comps["linear.step" if decode else "linear.core"],
            "dsa": lambda: dsa_full(x, sh, block)}
    tim = interleaved(arms, warmup=warmup, rounds=rounds, device=device)
    res["arms"] = {}
    for name, t in tim.items():
        lo, hi = t.ci()
        row = {"ms": t.median * 1e3, "ci_ms": (lo * 1e3, hi * 1e3)}
        if name != "dense":
            row["speedup_vs_dense"] = speedup(tim["dense"], t)
        if ops:
            row["ops"] = op_count(arms[name], device)
        res["arms"][name] = row
    sync(device)
    return res
