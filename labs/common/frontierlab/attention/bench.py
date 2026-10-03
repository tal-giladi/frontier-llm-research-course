"""Decode memory and latency at a fixed context, measured with the Module 2 method (lesson 03.1, 03.2).

For each context length S and each arm (a model with its attention kind):

1. **Prefill** S tokens into a fresh cache, in chunks (so prefill memory stays bounded).
2. **Memory.** ``Cache.nbytes()`` (bytes held by the cache tensors, exact), split into K/V or latent
   bytes and position bookkeeping; on CUDA also the allocator's view: ``memory_allocated`` after the
   prefill minus before, and the peak during one decode step above that.
3. **Latency of one decode step at context S**, with :func:`frontierlab.perf.timing.interleaved`:
   warm-up rounds, synchronisation around every sample, rounds alternating between arms. Every timed
   call decodes one token and then *restores* the cache to its S-token state (the forward pass replaces
   the cache's tensors with new ones, so keeping references to the old ones is enough). Without the
   restore, each repeat runs at a longer context than the last.

Weights can be random: decode latency and cache size do not depend on their values. Quality is a
separate measurement (Eval v0).
"""

from __future__ import annotations

import time

import torch

from frontierlab.perf.timing import interleaved, speedup, sync


def cache_breakdown(cache) -> dict:
    """Bytes per cache entry name, summed over layers ({"k": ..., "v": ..., "pos": ...} or {"c_kv", "k_rope", "pos"})."""
    out: dict[str, int] = {}
    for layer in cache.layers:
        for name, t in layer.items():
            if isinstance(t, torch.Tensor):
                out[name] = out.get(name, 0) + t.numel() * t.element_size()
    return out


@torch.no_grad()
def prefill(model, S: int, B: int = 1, chunk: int = 1024, vocab: int | None = None, seed: int = 0, device="cpu"):
    """A cache holding S tokens of random ids, filled ``chunk`` tokens at a time."""
    g = torch.Generator().manual_seed(seed)
    V = vocab or model.config.vocab_size
    ids = torch.randint(0, V, (B, S), generator=g).to(device)
    cache = model.new_cache()
    for s in range(0, S, chunk):
        model(ids[:, s:s + chunk], cache=cache)
    return cache


class _Decoder:
    """One decode step at a fixed context: decode a token, then restore the cache."""

    def __init__(self, model, cache, B: int, device):
        self.model, self.cache = model, cache
        self.saved = [dict(layer) for layer in cache.layers]
        self.length = cache.length
        self.tok = torch.zeros(B, 1, dtype=torch.long, device=device)

    @torch.no_grad()
    def __call__(self):
        self.model(self.tok, cache=self.cache)
        for layer, saved in zip(self.cache.layers, self.saved):
            layer.clear()
            layer.update(saved)
        self.cache.length = self.length


def decode_benchmark(models: dict, contexts, *, B: int = 1, warmup: int = 3, rounds: int = 20,
                     chunk: int = 1024, device="cpu", log=print) -> list[dict]:
    """Rows of {arm, S, cache_bytes, kv_bytes, pos_bytes, step_ms (median), ci_ms, cv, [cuda_*]} plus speed-ups.

    ``models``: {arm name: model already on ``device`` and in eval mode}. The first arm is the reference
    for the paired speed-ups (``speedup_vs_first``: > 1 means faster than the first arm).
    """
    rows = []
    cuda = torch.device(device).type == "cuda"
    for S in contexts:
        decs, mem = {}, {}
        for name, m in models.items():
            if cuda:
                sync(device)
                before = torch.cuda.memory_allocated(device)
            t0 = time.perf_counter()
            cache = prefill(m, S, B, chunk, device=device)
            sync(device)
            t_pre = time.perf_counter() - t0
            br = cache_breakdown(cache)
            mem[name] = {"cache_bytes": cache.nbytes(), "pos_bytes": br.get("pos", 0),
                         "kv_bytes": cache.nbytes() - br.get("pos", 0), "prefill_s": t_pre, "breakdown": br}
            if cuda:
                mem[name]["cuda_cache_bytes"] = torch.cuda.memory_allocated(device) - before
            decs[name] = _Decoder(m, cache, B, device)
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
                base = torch.cuda.memory_allocated(device)
                decs[name]()
                sync(device)
                mem[name]["cuda_step_peak_bytes"] = torch.cuda.max_memory_allocated(device) - base
        res = interleaved(decs, warmup=warmup, rounds=rounds, device=device)
        first = next(iter(models))
        for name in models:
            t = res[name]
            lo, hi = t.ci()
            row = {"arm": name, "S": S, "B": B, **mem[name], "step_ms": t.median * 1e3,
                   "ci_ms": (lo * 1e3, hi * 1e3), "cv": t.std / t.mean}
            if name != first:
                row["speedup_vs_first"] = speedup(res[first], t)
            rows.append(row)
            log(format_row(row))
        for d in decs.values():                      # free this context's caches before the next one
            d.cache = d.saved = None
        if cuda:
            torch.cuda.empty_cache()
    return rows


def format_row(r: dict) -> str:
    from frontierlab.attention.accounting import human_bytes
    s = (f"S={r['S']:>7d}  {r['arm']:<14s} cache {human_bytes(r['cache_bytes']):>11s} "
         f"(K/V or latent {human_bytes(r['kv_bytes'])})  step {r['step_ms']:8.3f} ms "
         f"[{r['ci_ms'][0]:.3f}, {r['ci_ms'][1]:.3f}] cv {r['cv']:.1%}")
    if "speedup_vs_first" in r:
        sp = r["speedup_vs_first"]
        s += f"  speed-up vs first {sp['speedup']:.3f} [{sp['ci'][0]:.3f}, {sp['ci'][1]:.3f}]"
    if "cuda_cache_bytes" in r:
        s += f"  cuda: cache {human_bytes(r['cuda_cache_bytes'])}, step peak +{human_bytes(r['cuda_step_peak_bytes'])}"
    return s
