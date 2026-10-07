"""Speculative sampling from scratch (lesson 15.2): a cheap draft proposes, the target verifies in one pass.

**The step** (Leviathan et al. 2022, Algorithm 1; Chen et al. 2023). The draft proposes gamma tokens
d_1..d_gamma, sampling each from its own distribution q_i. One target forward over the last accepted token and
the drafts gives the target distributions p_1..p_{gamma+1}. Draft i is accepted with probability
min(1, p_i(d_i) / q_i(d_i)), in order, until the first rejection. At the first rejected position j a token is
drawn from the residual norm(max(0, p_j - q_j)); if all gamma are accepted a bonus token is drawn from
p_{gamma+1}. Every round therefore emits between 1 and gamma + 1 tokens.

**Why the output is distributed exactly as the target.** For one position, P(emit x) = q(x)·min(1, p(x)/q(x))
+ P(reject)·r(x) with r = max(0, p - q)/Z. The first term is min(q(x), p(x)). P(reject) = 1 - sum_x min(p, q)
= Z, because sum_x max(0, p - q) = sum_x (p - min(p, q)) = 1 - sum_x min(p, q). So the second term is
max(0, p(x) - q(x)), and min(q, p) + max(0, p - q) = p(x). Later positions are conditioned on the accepted
prefix, so the argument applies position by position (the lesson works it through; tests check it by
enumeration and by Monte Carlo). Greedy decoding is the special case where p and q are one-hot: a draft is
accepted exactly when it equals the target's argmax.

**Acceptance.** beta_i = sum_x min(p_i, q_i) = 1 - TV(p_i, q_i) is the probability that draft i is accepted given
its prefix; alpha = E[beta]. If acceptances were independent with rate alpha, a round would emit on average
(1 - alpha^(gamma+1)) / (1 - alpha) tokens (Leviathan et al. Eq. 1), and with c = draft step cost / target step
cost the expected walltime improvement is that number divided by (gamma·c + 1) (their Theorem 3.8).

**Drafts** share one interface (``start``, ``propose``, ``commit``):

* :class:`LMDraft` — an independent small language model with its own KV cache.
* :class:`HiddenDraft` — a one-block head that reads the *target's* last hidden state h_i and the embedding
  of token i+1 and predicts token i+2, as DeepSeek-V3's MTP module does (Eq. 21; ``frontierlab.blocks.mtp``)
  and as an EAGLE draft head does (EAGLE autoregresses on features; EAGLE-3 fuses features of several layers
  and predicts tokens directly). For the second and later drafts of a round the head feeds back its own
  output feature in place of the target's. Positions it drafted are recomputed from the target's real
  features once the target has verified them.

**Caches.** The target keeps a KV cache of every token but the last; after verification it is cut back to
the accepted prefix (:func:`truncate_cache`), so rejected drafts cost compute but leave no state.
"""

from __future__ import annotations

import time

import torch

from frontierlab.attention.base import Cache, LayerCache


# --------------------------------------------------------------------------------------------- the step

def probs_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Softmax at ``temperature``, or a one-hot of the argmax at temperature 0 (greedy as a distribution)."""
    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits.dtype)
    return torch.softmax(logits / temperature, dim=-1)


def residual(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """norm(max(0, p - q)) along the last axis. Where p == q everywhere (never rejected) it returns p."""
    r = (p - q).clamp_min(0)
    z = r.sum(-1, keepdim=True)
    return torch.where(z > 0, r / z.clamp_min(torch.finfo(r.dtype).tiny), p)


def accept_reject(p: torch.Tensor, q: torch.Tensor, drafts: torch.Tensor, u: torch.Tensor,
                  generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched verification of one round.

    p (B, g+1, V) target distributions at the g draft positions and the bonus position; q (B, g, V) draft
    distributions; drafts (B, g) int64; u (B, g) uniforms in [0, 1). Returns (n_accepted (B,) in 0..g,
    next_token (B,)): the residual sample at the first rejected position, or the bonus sample if none was."""
    B, g = drafts.shape
    if g == 0:
        return torch.zeros(B, dtype=torch.long), torch.multinomial(p[:, 0], 1, generator=generator).squeeze(-1)
    pd = p[:, :g].gather(-1, drafts[..., None]).squeeze(-1)
    qd = q.gather(-1, drafts[..., None]).squeeze(-1)
    ok = u < torch.clamp(pd / qd.clamp_min(torch.finfo(qd.dtype).tiny), max=1.0)
    n = ok.long().cumprod(-1).sum(-1)                                  # number of leading accepts
    ar = torch.arange(B)
    pj = p[ar, n]                                                      # (B, V) at the first rejected / bonus slot
    qj = torch.cat([q, torch.zeros_like(q[:, :1])], 1)[ar, n]          # q = 0 at the bonus slot: residual = p
    nxt = torch.multinomial(residual(pj, qj), 1, generator=generator).squeeze(-1)
    return n, nxt


def overlap(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """beta = sum_x min(p, q): the acceptance probability of a draft sampled from q, given its prefix."""
    return torch.minimum(p, q).sum(-1)


def expected_tokens_per_round(alpha: float, gamma: int) -> float:
    """(1 - alpha^(gamma+1)) / (1 - alpha), Leviathan et al. Eq. 1 (i.i.d. acceptances); gamma + 1 at alpha = 1."""
    if alpha >= 1.0:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def walltime_improvement(alpha: float, gamma: int, c: float) -> float:
    """Expected speed-up over plain decoding, Theorem 3.8: tokens per round / (gamma·c + 1)."""
    return expected_tokens_per_round(alpha, gamma) / (gamma * c + 1.0)


def best_gamma(alpha: float, c: float, max_gamma: int = 16) -> int:
    return max(range(1, max_gamma + 1), key=lambda g: walltime_improvement(alpha, g, c))


# --------------------------------------------------------------------------------------------- caches and forwards

def truncate_cache(cache: Cache, n: int) -> None:
    """Keep the first ``n`` cached positions of every layer (GQA-family caches: k, v, pos)."""
    for lc in cache.layers:
        _truncate_layer(lc, n)
    cache.length = min(cache.length, n)


def _truncate_layer(lc: LayerCache, n: int) -> None:
    if not lc:
        return
    if set(lc) - {"k", "v", "pos"}:
        raise NotImplementedError(f"rollback of cache entries {sorted(set(lc) - {'k', 'v', 'pos'})} is not implemented")
    lc["k"], lc["v"], lc["pos"] = lc["k"][:, :, :n], lc["v"][:, :, :n], lc["pos"][:n]


def forward_hidden(model, idx: torch.Tensor, cache: Cache | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``LM.forward`` that also returns the last layer's output before the final norm, h (B, T, C).

    Works for :class:`frontierlab.model.LM` and for a :class:`frontierlab.blocks.BlockLM` whose residual and
    FFN are plain (the layers are ordinary Blocks)."""
    B, T = idx.shape
    start = cache.length if cache is not None else 0
    pos = torch.arange(start, start + T, device=idx.device)
    x = model.model.embed_tokens(idx)
    for i, layer in enumerate(model.model.layers):
        x = layer(x, pos, cache.layers[i] if cache is not None else None)
    if cache is not None:
        cache.length += T
    logits = model.lm_head(model.model.norm(x))
    return (logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits), x


def _sample(pr: torch.Tensor, generator) -> int:
    return int(torch.multinomial(pr, 1, generator=generator))


# --------------------------------------------------------------------------------------------- drafts

class LMDraft:
    """An independent small LM as the draft (same tokenizer as the target)."""

    def __init__(self, model):
        self.model = model.eval()
        self.cache = None

    def start(self, seq: list[int], H):
        self.cache = self.model.new_cache()

    def propose(self, seq: list[int], H, gamma: int, temperature: float, generator):
        dev = next(self.model.parameters()).device
        feed = seq[self.cache.length:]                      # tokens not yet in the draft's cache (>= 1)
        logits = self.model(torch.tensor([feed], device=dev), cache=self.cache).logits[0, -1]
        toks, qs = [], []
        for i in range(gamma):
            q = probs_from_logits(logits.double() if logits.dtype == torch.float64 else logits.float(), temperature)
            t = _sample(q, generator)
            toks.append(t)
            qs.append(q)
            if i < gamma - 1:
                logits = self.model(torch.tensor([[t]], device=dev), cache=self.cache).logits[0, -1]
        return toks, (torch.stack(qs) if qs else None)

    def commit(self, new_len: int):
        truncate_cache(self.cache, min(self.cache.length, new_len - 1))


class HiddenDraft:
    """A DeepSeek-MTP / EAGLE-style head: ``module`` has ``combine(h, emb)``, ``block`` and ``norm`` (the
    :class:`frontierlab.blocks.mtp.DeepSeekModule` layout); ``embed`` and ``lm_head`` are the target's."""

    def __init__(self, module, embed, lm_head):
        self.module, self.embed, self.lm_head = module.eval(), embed, lm_head
        self.cache = None
        self.n_valid = 0                                     # entries computed from target features

    def start(self, seq: list[int], H):
        self.cache = LayerCache()
        self.n_valid = 0

    def _entries(self, h: torch.Tensor, toks: list[int], start: int) -> torch.Tensor:
        dev = h.device
        e = self.embed(torch.tensor([toks], device=dev))
        x = self.module.combine(h, e)
        pos = torch.arange(start, start + len(toks), device=dev)
        return self.module.block(x, pos, self.cache)

    def propose(self, seq: list[int], H: torch.Tensor, gamma: int, temperature: float, generator):
        """Entry i combines h_i with Emb(seq[i+1]) and predicts seq[i+2]. H holds h_0 .. h_{len(seq)-2}."""
        E, S = self.n_valid, len(seq)
        g = self._entries(H[:, E:S - 1], seq[E + 1:S], E)   # catch up with the target's real features
        self.n_valid = S - 1
        out = g[:, -1:]
        toks, qs = [], []
        for i in range(gamma):
            logits = self.lm_head(self.module.norm(out))[0, -1]
            q = probs_from_logits(logits.double() if logits.dtype == torch.float64 else logits.float(), temperature)
            t = _sample(q, generator)
            toks.append(t)
            qs.append(q)
            if i < gamma - 1:                                  # feed back the head's own feature (EAGLE-style)
                out = self._entries(out, [t], S - 1 + i)
        return toks, (torch.stack(qs) if qs else None)

    def commit(self, new_len: int):
        _truncate_layer(self.cache, self.n_valid)


# --------------------------------------------------------------------------------------------- generation

@torch.no_grad()
def autoregressive_generate(target, prompt: torch.Tensor, max_new: int, temperature: float = 1.0,
                            generator: torch.Generator | None = None) -> torch.Tensor:
    """Plain decoding with the KV cache, one target forward per token: the baseline (1, P + max_new)."""
    target.eval()
    cache = target.new_cache()
    logits, _ = forward_hidden(target, prompt, cache)
    seq = prompt[0].tolist()
    lg = logits[0, -1]
    for _ in range(max_new):
        t = _sample(probs_from_logits(lg, temperature), generator)
        seq.append(t)
        lg, _ = forward_hidden(target, torch.tensor([[t]], device=prompt.device), cache)
        lg = lg[0, -1]
    return torch.tensor([seq], device=prompt.device)


@torch.no_grad()
def speculative_generate(target, draft, prompt: torch.Tensor, max_new: int, gamma: int = 4,
                         temperature: float = 1.0, generator: torch.Generator | None = None,
                         track_overlap: bool = True) -> tuple[torch.Tensor, dict]:
    """Speculative sampling for one sequence (1, P), P >= 2. Returns (tokens (1, P + max_new), stats).

    stats: rounds, proposed, accepted, acceptance_rate, tokens_per_round, accepted_per_position (list),
    proposed_per_position, mean_overlap (alpha estimated as the mean of beta over proposed positions)."""
    target.eval()
    dev = prompt.device
    seq = prompt[0].tolist()
    P = len(seq)
    if P < 2:
        raise ValueError("prompt needs at least 2 tokens (the target caches all but the last)")
    target_len = P + max_new
    tcache = target.new_cache()
    _, H = forward_hidden(target, prompt[:, :-1], tcache)
    draft.start(seq, H)
    rounds = proposed = accepted = 0
    acc_pos, prop_pos, betas = [0] * gamma, [0] * gamma, []
    while len(seq) < target_len:
        g = min(gamma, target_len - len(seq) - 1)
        drafts, q = draft.propose(seq, H, g, temperature, generator) if g > 0 else ([], None)
        logits, h = forward_hidden(target, torch.tensor([[seq[-1]] + drafts], device=dev), tcache)
        lg = logits[0].double() if logits.dtype == torch.float64 else logits[0].float()
        p = probs_from_logits(lg, temperature)                     # (g+1, V)
        if g > 0:
            u = torch.rand(g, generator=generator, dtype=p.dtype)
            q = q.to(p.dtype)
            n, nxt = accept_reject(p[None], q[None], torch.tensor([drafts]), u[None], generator)
            n, nxt = int(n), int(nxt)
            if track_overlap:
                betas += overlap(p[:g], q).tolist()
        else:
            n, nxt = 0, _sample(p[0], generator)
        rounds += 1
        proposed += g
        accepted += n
        for i in range(g):
            prop_pos[i] += 1
            acc_pos[i] += int(i < n)
        H = torch.cat([H, h[:, :n + 1]], 1)[:, :len(seq) + n]
        truncate_cache(tcache, len(seq) + n)
        seq += drafts[:n] + [nxt]
        draft.commit(len(seq))
    seq = seq[:target_len]
    return torch.tensor([seq], device=dev), {
        "rounds": rounds, "proposed": proposed, "accepted": accepted,
        "acceptance_rate": accepted / proposed if proposed else float("nan"),
        "tokens_per_round": max_new / rounds, "accepted_per_position": acc_pos, "proposed_per_position": prop_pos,
        "mean_overlap": float(sum(betas) / len(betas)) if betas else float("nan")}


@torch.no_grad()
def step_cost_ratio(target, draft_step, repeats: int = 20, ctx: int = 64) -> dict:
    """Measured c = (one draft step) / (one target decode step) at batch 1 and context ``ctx`` (CPU or GPU),
    with ``frontierlab.perf.benchmark``. ``draft_step()`` must run one draft step."""
    from frontierlab.perf import benchmark
    dev = next(target.parameters()).device
    holder = {}

    def setup():
        c = target.new_cache()
        forward_hidden(target, torch.ones((1, ctx), dtype=torch.long, device=dev), c)
        holder["c"] = c

    tok = torch.ones((1, 1), dtype=torch.long, device=dev)
    t_target = benchmark(lambda: forward_hidden(target, tok, holder["c"]), warmup=3, repeats=repeats, setup=setup,
                         device=dev).median
    t_draft = benchmark(draft_step, warmup=3, repeats=repeats, device=dev).median
    return {"target_step_s": t_target, "draft_step_s": t_draft, "c": t_draft / t_target}


def timed(fn) -> tuple[object, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0
