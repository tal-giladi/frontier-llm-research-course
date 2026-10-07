"""Serving cost of architecture choices (lesson 15.3): prefill vs decode, KV capacity, disaggregation, RL rollouts.

**Two phases on the roofline** (Module 2, ``frontierlab.perf.roofline``). Prefill processes T prompt tokens in
one pass: about 2·N·T FLOPs against one read of the N weights, so its arithmetic intensity grows with T and
it is compute-bound. A decode step of a batch of B sequences does about 2·N·B FLOPs plus attention, but must
read all weights once and every sequence's KV cache once:

    t_decode(B, S) >= max( (2·N·B + B·attn(S)) / peak,  (weight_bytes + B·kv_bytes(S)) / bandwidth )

At small B the weight read dominates (memory-bound, cost almost flat in B); as B and S grow the KV read takes
over. So the KV bytes per sequence decide two things at once: how many sequences fit in memory (capacity) and
how long each decode step takes once they do.

**KV bytes.** For the course's own designs the exact functions of Modules 3–5 are reused
(``frontierlab.attention.accounting.m05_cache_bytes``, tested against ``Cache.nbytes()``); for released
models, the Module 1 calculator on pinned ``config.json`` snapshots (``frontierlab.calc``).
:func:`stage_d_designs` applies each Module 3–5 design to the *shape* of the Stage D model (Qwen3-1.7B: 28
layers, 2048 wide, 16 query heads, 8 KV heads of 128) — "what if this model had used MLA / windows / linear
layers" — which is a design calculation, not a released model.

**Serving simulator** (:func:`simulate`): an iteration-level model of one engine. Each iteration costs the
roofline time of the work it contains (prefill tokens and one decode token per running sequence). Three
policies, the ones the lesson compares:

* ``"prefill_first"`` — a waiting prompt is prefilled whole in the next iteration and running decodes wait
  for it (a decode stall; the behaviour Sarathi-Serve measures);
* ``"chunked"`` — every iteration carries the running decodes plus at most ``chunk`` prefill tokens
  (chunked prefill with stall-free batching, Sarathi-Serve);
* ``"disaggregated"`` — prefill on its own GPU(s), KV sent over a link, decode on its own GPU(s)
  (DistServe, Splitwise, Mooncake). Compared at the same total number of GPUs.

Metrics per request: TTFT (arrival to first token), TPOT (mean time between later tokens); goodput is the
fraction of requests meeting both SLOs. The simulator is a model with stated assumptions (no queueing
delays inside kernels, perfect overlap of KV transfer only if ``overlap_transfer``), meant for comparing
policies, not for predicting a real engine's numbers.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from frontierlab.attention import accounting as acc
from frontierlab.model.config import ModelConfig, baseline0
from frontierlab.perf.roofline import HARDWARE, Hardware

GiB = 2 ** 30

# The Stage D base model's shape (Qwen/Qwen3-1.7B-Base at ea980cb0, config.json): 28 layers, hidden 2048,
# 16 query heads, 8 KV heads, head_dim 128, intermediate 6144, vocabulary 151,936, tied embeddings.
QWEN3_1_7B = ModelConfig(vocab_size=151936, hidden_size=2048, num_hidden_layers=28, num_attention_heads=16,
                         num_key_value_heads=8, head_dim=128, intermediate_size=6144, rope_theta=1_000_000.0,
                         max_position_embeddings=32768, tie_word_embeddings=True)


def _designs(base: ModelConfig, window: int = 1024) -> dict[str, ModelConfig]:
    L, d = base.num_hidden_layers, base.head_dim
    lg = ["global" if (i + 1) % 6 == 0 else "local" for i in range(L)]
    return {
        "GQA (as released)": base,
        "MHA": base.with_(num_key_value_heads=base.num_attention_heads),
        "MQA": base.with_(num_key_value_heads=1),
        "MLA d_c=4d": base.with_(attention="mla", extra={"kv_lora_rank": 4 * d, "qk_rope_head_dim": d // 2}),
        f"local/global 5:1, w={window}": base.with_(attention="local_global",
                                                     extra={"window": window, "layer_types": lg}),
        "hybrid GDN 3:1 (GQA full)": base.with_(attention="hybrid", extra={"linear_kind": "gdn", "full_every": 4}),
        "DSA k=2048 (GQA main)": base.with_(attention="dsa", extra={"index_topk": 2048, "index_heads": 4,
                                                                    "index_head_dim": 64}),
    }


def stage_d_designs(window: int = 1024) -> dict[str, ModelConfig]:
    """The Module 3–5 designs on the Stage D model's shape (design calculation, not released models)."""
    return _designs(QWEN3_1_7B, window)


def baseline0_designs(window: int = 1024) -> dict[str, ModelConfig]:
    """The same designs on Baseline-0 (the course's own ~122M model)."""
    return _designs(baseline0(), window)


_PARAMS: dict = {}


def n_params(cfg: ModelConfig) -> int:
    """Total parameters (memoised: counting builds every attention module once)."""
    key = repr(cfg)
    if key not in _PARAMS:
        _PARAMS[key] = acc.param_counts(cfg)["total"]
    return _PARAMS[key]


def weight_bytes(cfg: ModelConfig, bytes_per: float = 2) -> float:
    return n_params(cfg) * bytes_per


def _is_m05(cfg: ModelConfig) -> bool:
    acc._import_m05()
    return cfg.attention in acc.M05_KINDS or acc._m05_family(cfg.attention) is not None


def kv_bytes_seq(cfg: ModelConfig, S: int, bytes_per: float = 2, state_bytes: float = 4) -> float:
    """Decode-cache bytes of one sequence after S tokens (K/V or latent, indexer keys, linear states), without
    the int64 position bookkeeping of the course's caches. Module 3 kinds (GQA family, MLA, windows) use
    ``accounting.kv_bytes``; Module 5 kinds (linear, hybrid, DSA) ``accounting.m05_cache_bytes``."""
    if _is_m05(cfg):
        return acc.m05_cache_bytes(cfg, S, bytes_per=bytes_per, state_bytes=state_bytes)
    return acc.kv_bytes(cfg, S, bytes_per)


_MATMUL: dict = {}


def _matmul_flops(cfg: ModelConfig) -> float:
    """2 x non-embedding parameters + the tied output head: the per-token matmul FLOPs (memoised)."""
    key = repr(cfg)
    if key not in _MATMUL:
        _MATMUL[key] = acc._fwd_matmul(cfg)
    return _MATMUL[key]


def decode_flops(cfg: ModelConfig, S: int) -> float:
    """Forward FLOPs to generate one token at context S: the same sums as ``accounting.decode_flops_per_token``
    (absorbed MLA: the serving path) and ``accounting.m05_decode_flops_per_token``, with the matmul part memoised
    (a test checks both agree)."""
    fwd = _matmul_flops(cfg)
    if _is_m05(cfg):
        for kind in acc.m05_layer_kinds(cfg):
            if kind == "linear":
                fwd += acc.linear_mix_flops(cfg, "recurrent")
            elif kind == "dsa":
                fwd += acc.indexer_flops_per_key(cfg) * S + acc._gqa_per_key(cfg) * min(S, acc.dsa_dims(cfg)["k"])
            else:
                w = acc.window(cfg) if kind == "local" else None
                fwd += acc._gqa_per_key(cfg) * (min(S, w) if w else S)
        return fwd
    w, per_key = acc.window(cfg), acc.flops_per_key(cfg, "absorbed" if cfg.attention == "mla" else "naive")
    for kind in acc.layer_kinds(cfg):
        fwd += per_key * (min(S, w) if (kind == "local" and w) else S)
    return fwd


def released_kv(snapshot: str, S: int, bytes_per: float = 2) -> dict:
    """KV bytes of a released model from its pinned config snapshot (``frontierlab.calc``)."""
    from frontierlab.calc import arith, hfconfig
    s = hfconfig.from_hf(snapshot)
    return {"model": snapshot, "kv_bytes": arith.kv_bytes(s, S, bytes_per),
            "params": arith.param_counts(s)["total"], "kinds": {k: s.layer_kinds.count(k) for k in set(s.layer_kinds)}}


def capacity(hbm_bytes: float, weights: float, kv_per_seq: float, reserve: float = 0.10) -> int:
    """Sequences whose cache fits next to the weights, keeping ``reserve`` of HBM for activations etc."""
    free = hbm_bytes * (1 - reserve) - weights
    return max(0, int(free // kv_per_seq)) if kv_per_seq > 0 else 10 ** 9


def kv_table(designs: dict[str, ModelConfig], S: int = 131072, bytes_per: float = 2, hbm_gib: float = 80.0,
             weight_bytes_per: float = 2) -> list[dict]:
    """Per design: KV GiB per sequence at S, KV bytes per token, sequences that fit on one GPU of ``hbm_gib``."""
    rows = []
    for name, cfg in designs.items():
        kv = kv_bytes_seq(cfg, S, bytes_per)
        w = weight_bytes(cfg, weight_bytes_per)
        rows.append({"design": name, "kv_gib_per_seq": kv / GiB, "kv_bytes_per_token": kv / S,
                     "weights_gib": w / GiB, "fit_at_S": capacity(hbm_gib * GiB, w, kv)})
    return rows


# --------------------------------------------------------------------------------------------- roofline times

def decode_step_time(cfg: ModelConfig, B: int, S: float, hw: Hardware, bytes_per: float = 2) -> float:
    """Lower bound for one decode step of B sequences at context S (weights read once, each cache once)."""
    flops = B * decode_flops(cfg, int(max(1, S)))
    nbytes = weight_bytes(cfg, bytes_per) + B * kv_bytes_seq(cfg, int(max(1, S)), bytes_per)
    return max(flops / hw.peak_flops, nbytes / hw.mem_bw)


def _fwd_flops(cfg: ModelConfig, T: int) -> float:
    """Average forward FLOPs per token of a T-token causal sequence (memoised by config and T)."""
    key = (repr(cfg), T)
    if key not in _MATMUL:
        _MATMUL[key] = (acc.m05_flops_per_token if _is_m05(cfg) else acc.flops_per_token)(cfg, max(1, T), training=False)
    return _MATMUL[key]


def prefill_time(cfg: ModelConfig, T: int, hw: Hardware, bytes_per: float = 2) -> float:
    """Lower bound for prefilling T tokens: forward FLOPs of a T-token sequence vs one weight read."""
    flops = T * _fwd_flops(cfg, T)
    return max(flops / hw.peak_flops, weight_bytes(cfg, bytes_per) / hw.mem_bw)


def intensity(cfg: ModelConfig, phase: str, B: int = 1, S: int = 1024, T: int = 1024, bytes_per: float = 2) -> float:
    """FLOP per byte of a prefill of T tokens or a decode step of B sequences at context S."""
    if phase == "prefill":
        return T * _fwd_flops(cfg, T) / weight_bytes(cfg, bytes_per)
    flops = B * decode_flops(cfg, S)
    return flops / (weight_bytes(cfg, bytes_per) + B * kv_bytes_seq(cfg, S, bytes_per))


def decode_throughput(cfg: ModelConfig, S: int, hw: Hardware, hbm_gib: float = 80.0, bytes_per: float = 2,
                      max_batch: int | None = None) -> dict:
    """Tokens/s at the largest batch that fits at context S (the KV-capacity-limited regime)."""
    w = weight_bytes(cfg, bytes_per)
    B = capacity(hbm_gib * GiB, w, kv_bytes_seq(cfg, S, bytes_per))
    if max_batch is not None:
        B = min(B, max_batch)
    if B == 0:
        return {"batch": 0, "step_s": float("inf"), "tokens_per_s": 0.0}
    t = decode_step_time(cfg, B, S, hw, bytes_per)
    return {"batch": B, "step_s": t, "tokens_per_s": B / t}


# --------------------------------------------------------------------------------------------- simulator

@dataclass
class Request:
    arrival: float
    prompt: int
    output: int
    rid: int = 0
    first: float = math.nan
    done: float = math.nan
    tokens: list = field(default_factory=list)       # time of every output token


def poisson_requests(rate: float, n: int, prompt: int, output: int, seed: int = 0, jitter: float = 0.0) -> list[Request]:
    """``n`` requests with exponential inter-arrival times (``rate`` per second); lengths optionally jittered."""
    rng = np.random.default_rng(seed)
    t, out = 0.0, []
    for i in range(n):
        t += rng.exponential(1.0 / rate)
        p = max(1, int(round(prompt * (1 + jitter * rng.uniform(-1, 1)))))
        o = max(1, int(round(output * (1 + jitter * rng.uniform(-1, 1)))))
        out.append(Request(t, p, o, i))
    return out


@dataclass
class Engine:
    """Roofline cost of one iteration on one GPU: ``prefill_tokens`` new tokens plus one decode token for each
    running sequence (with its context length)."""
    cfg: ModelConfig
    hw: Hardware
    bytes_per: float = 2
    hbm_gib: float = 80.0
    efficiency: float = 1.0          # fraction of the roofline actually reached (state it; 1 = the bound)

    def __post_init__(self):
        self.w = weight_bytes(self.cfg, self.bytes_per)
        self._fpt = {}

    def _prefill_flops(self, n: int, ctx: int) -> float:
        # per token: matmuls + attention over ~ctx/2 keys (one chunk of a longer prompt sees more context)
        return n * decode_flops(self.cfg, max(1, ctx))

    def iteration(self, prefill: list[tuple[int, int]], decode_ctx: list[int]) -> float:
        """prefill: list of (new tokens, context they attend to on average); decode_ctx: contexts of decodes."""
        flops = sum(self._prefill_flops(n, c) for n, c in prefill)
        flops += sum(decode_flops(self.cfg, max(1, c)) for c in decode_ctx)
        nbytes = self.w + sum(kv_bytes_seq(self.cfg, max(1, c), self.bytes_per) for c in decode_ctx)
        nbytes += sum(kv_bytes_seq(self.cfg, max(1, n), self.bytes_per) for n, _ in prefill)    # writing new KV
        return max(flops / self.hw.peak_flops, nbytes / self.hw.mem_bw) / self.efficiency

    def max_running(self, max_ctx: int) -> int:
        return capacity(self.hbm_gib * GiB, self.w, kv_bytes_seq(self.cfg, max_ctx, self.bytes_per))


def _run_colocated(reqs: list[Request], eng: Engine, policy: str, chunk: int, max_batch: int) -> None:
    pending = sorted(reqs, key=lambda r: r.arrival)
    max_ctx = max(r.prompt + r.output for r in reqs)
    cap = min(max_batch, eng.max_running(max_ctx))
    t, i = 0.0, 0
    waiting: list = []                                 # requests that arrived, prompt not (fully) prefilled
    prog: dict = {}                                    # rid -> prompt tokens prefilled so far
    running: list = []                                 # requests decoding
    while i < len(pending) or waiting or running:
        while i < len(pending) and pending[i].arrival <= t:
            waiting.append(pending[i]); prog[pending[i].rid] = 0; i += 1
        if not waiting and not running:
            t = pending[i].arrival
            continue
        pre: list = []
        if policy == "prefill_first":
            room = cap - len(running)
            batch = waiting[:max(0, room)]
            if batch:                                   # whole prompts; decodes stall this iteration
                pre = [(r.prompt, r.prompt // 2) for r in batch]
                dt = eng.iteration(pre, [])
                t += dt
                for r in batch:
                    waiting.remove(r)
                    r.first = t; r.tokens.append(t)
                    running.append(r)
                continue
        elif policy == "chunked":
            budget = chunk - len(running)
            for r in list(waiting):
                if budget <= 0 or len(running) + sum(1 for x in waiting if prog[x.rid] > 0) > cap:
                    break
                take = min(budget, r.prompt - prog[r.rid])
                pre.append((take, prog[r.rid] + take // 2))
                prog[r.rid] += take
                budget -= take
        else:
            raise ValueError(policy)
        ctx = [r.prompt + len(r.tokens) for r in running]
        dt = eng.iteration(pre, ctx)
        t += dt
        for r in list(running):
            r.tokens.append(t)
            if len(r.tokens) >= r.output:
                r.done = t
                running.remove(r)
        for r in list(waiting):
            if prog[r.rid] >= r.prompt and len(running) < cap:
                waiting.remove(r)
                r.first = t; r.tokens.append(t)
                running.append(r)
                if len(r.tokens) >= r.output:
                    r.done = t
                    running.remove(r)


def _prefill_only(reqs: list[Request], eng: Engine, link_bw: float, max_prefill_batch: int = 8) -> dict:
    """A prefill GPU: FCFS, batching up to ``max_prefill_batch`` prompts that have arrived. Sets ``first`` (the
    prefill pass produces the first token) and returns {rid: time its KV has reached the decode side}."""
    pending = sorted(reqs, key=lambda r: r.arrival)
    t, i, ready = 0.0, 0, {}
    while i < len(pending):
        t = max(t, pending[i].arrival)
        batch = []
        while i < len(pending) and pending[i].arrival <= t and len(batch) < max_prefill_batch:
            batch.append(pending[i]); i += 1
        t += eng.iteration([(r.prompt, r.prompt // 2) for r in batch], [])
        for r in batch:
            r.first = t
            ready[r.rid] = t + kv_bytes_seq(eng.cfg, r.prompt, eng.bytes_per) / link_bw
    return ready


def _decode_only(reqs: list[Request], eng: Engine, ready: dict, max_batch: int) -> None:
    """A decode GPU: continuous batching of the requests whose KV has arrived."""
    queue = sorted(((ready[r.rid], r.rid, r) for r in reqs), key=lambda x: (x[0], x[1]))
    heapq.heapify(queue)
    max_ctx = max(r.prompt + r.output for r in reqs)
    cap = min(max_batch, eng.max_running(max_ctx))
    t, running = 0.0, []
    while queue or running:
        while queue and queue[0][0] <= t and len(running) < cap:
            _, _, r = heapq.heappop(queue)
            r.tokens.append(r.first)
            running.append(r)
        if not running:
            t = queue[0][0]
            continue
        t += eng.iteration([], [r.prompt + len(r.tokens) for r in running])
        for r in list(running):
            r.tokens.append(t)
            if len(r.tokens) >= r.output:
                r.done = t
                running.remove(r)


def simulate(reqs: list[Request], cfg: ModelConfig, hw: Hardware | str = "H100-SXM", policy: str = "chunked",
             gpus: int = 2, chunk: int = 512, link_gbps: float = 400.0, max_batch: int = 256,
             efficiency: float = 1.0, bytes_per: float = 2, prefill_gpus: int | None = None) -> dict:
    """Serve ``reqs`` on ``gpus`` GPUs. Colocated policies split the requests round-robin over the GPUs;
    ``disaggregated`` uses ``prefill_gpus`` (default gpus // 2) prefill GPUs and the rest as decode GPUs, each
    request assigned round-robin to one of each. ``link_gbps`` is the KV transfer bandwidth (400 = one 400 Gb/s NIC)."""
    hw = HARDWARE[hw] if isinstance(hw, str) else hw
    rs = [Request(r.arrival, r.prompt, r.output, r.rid) for r in reqs]
    mk = lambda: Engine(cfg, hw, bytes_per, efficiency=efficiency)
    if policy == "disaggregated":
        n_pre = prefill_gpus or max(1, gpus // 2)
        n_dec = gpus - n_pre
        if n_dec < 1:
            raise ValueError("disaggregation needs at least one decode GPU")
        ready = {}
        for g in range(n_pre):
            ready.update(_prefill_only([r for k, r in enumerate(rs) if k % n_pre == g], mk(), link_gbps * 1e9 / 8))
        for g in range(n_dec):
            part = [r for k, r in enumerate(rs) if k % n_dec == g]
            if part:
                _decode_only(part, mk(), ready, max_batch)
    else:
        for g in range(gpus):
            part = [r for k, r in enumerate(rs) if k % gpus == g]
            if part:
                _run_colocated(part, mk(), policy, chunk, max_batch)
    return summarize(rs)


def summarize(rs: list[Request], ttft_slo: float | None = None, tpot_slo: float | None = None) -> dict:
    ttft = np.array([r.first - r.arrival for r in rs])
    tpot = np.array([(r.tokens[-1] - r.tokens[0]) / max(1, len(r.tokens) - 1) for r in rs])
    span = max(r.done for r in rs) - min(r.arrival for r in rs)
    out = {"requests": len(rs), "ttft_p50": float(np.median(ttft)), "ttft_p90": float(np.quantile(ttft, 0.9)),
           "tpot_p50": float(np.median(tpot)), "tpot_p90": float(np.quantile(tpot, 0.9)),
           "tokens_per_s": float(sum(r.output for r in rs) / span), "_ttft": ttft, "_tpot": tpot}
    if ttft_slo is not None and tpot_slo is not None:
        out["goodput_frac"] = goodput(out, ttft_slo, tpot_slo)
    return out


def goodput(summary: dict, ttft_slo: float, tpot_slo: float) -> float:
    """Fraction of requests meeting both the TTFT and the TPOT objective (DistServe's per-request attainment)."""
    return float(np.mean((summary["_ttft"] <= ttft_slo) & (summary["_tpot"] <= tpot_slo)))


# --------------------------------------------------------------------------------------------- RL rollouts

def rollout_time(cfg: ModelConfig, sequences: int, prompt: int, gen: int, hw: Hardware | str = "H100-SXM",
                 gpus: int = 1, hbm_gib: float = 80.0, bytes_per: float = 2, reserve: float = 0.10,
                 trainer_gib: float = 0.0, steps: int = 64) -> dict:
    """Roofline time to generate ``gen`` tokens for each of ``sequences`` rollouts (all the same length; real
    rollouts have a long tail that makes this an optimistic bound). Sequences run in waves of the largest
    batch whose caches fit at the final context (``trainer_gib`` of HBM is held by a colocated trainer).
    Decode time per wave is integrated over the context with ``steps`` points."""
    hw = HARDWARE[hw] if isinstance(hw, str) else hw
    w = weight_bytes(cfg, bytes_per)
    free_gib = hbm_gib - trainer_gib
    cap = capacity(free_gib * GiB, w, kv_bytes_seq(cfg, prompt + gen, bytes_per), reserve)
    if cap == 0:
        return {"batch": 0, "waves": math.inf, "seconds": math.inf, "tokens_per_s": 0.0}
    per_gpu = math.ceil(sequences / gpus)
    waves = math.ceil(per_gpu / cap)
    batch = min(cap, per_gpu)
    ctx = np.linspace(prompt, prompt + gen, steps)
    t_wave = prefill_time(cfg, prompt * batch, hw, bytes_per) + float(np.mean([decode_step_time(cfg, batch, c, hw, bytes_per)
                                                                                for c in ctx])) * gen
    total = waves * t_wave
    return {"batch": batch, "waves": waves, "seconds": total, "tokens_per_s": sequences * gen / total}
