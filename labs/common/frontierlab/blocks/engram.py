"""Conditional memory: an Engram-style hashed N-gram lookup module (lesson 06.4, extension).

Engram (DeepSeek-AI, arXiv 2601.07372, section 2) is a *second sparsity axis* next to MoE: MoE activates
a few experts (conditional computation); Engram looks up a few rows of a large embedding table
(conditional memory), indexed by the last few tokens. It is not a residual-stream design: it adds one
term to the residual at a few chosen layers, ``H <- H + Y``, before that layer's attention and FFN.

Per position t (paper Eqs. 1–5), for N-gram orders n = 2..N and K hash heads per order:

    g_{t,n}     = (x_{t-n+1}, ..., x_t)                       suffix N-gram of token ids (causal)
    z_{t,n,k}   = phi_{n,k}(g_{t,n})  in [0, M_{n,k})           multiplicative-XOR hash, M prime
    e_t         = concat_{n,k} E_{n,k}[z_{t,n,k}]              (d_mem,)  d_mem = (N-1)·K·d_head
    k_t, v_t    = W_K e_t, W_V e_t                              (C,) each
    alpha_t     = sigmoid( RMSNorm(h_t) · RMSNorm(k_t) / sqrt(C) )      scalar gate in (0, 1)
    v~_t        = alpha_t · v_t
    Y           = SiLU( Conv1D( RMSNorm(V~) ) ) + V~           depthwise causal conv, kernel w = 4,
                                                                dilation = N (the max order); weights 0 at init

What this course version leaves out (named in the lesson): the tokenizer compression P: V -> V'
(paper section 2.2: NFKC, lowercasing; a 23% smaller effective vocabulary for a 128K tokenizer), the
5x learning rate and no weight decay for the tables (section 4: "embedding parameters are updated using
Adam with a learning rate scaled by 5x and no weight decay"), and table sharding / host offload
(section 2.5). Multi-branch integration (section 2.4: one table and W_V shared, one W_K per residual
branch) is supported with ``branches = n`` for an n-stream model.

Positions with fewer than n-1 earlier tokens use a padding id (``vocab_size``) for the missing ones, so
the lookup is a function of real past tokens only — the module is causal and works with the decode
cache (it keeps the last N-1 token ids and the last (w-1)·N normalised gated values).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from frontierlab.layers.rmsnorm import RMSNorm

_MOD = 2_147_483_647          # 2^31 - 1, keeps every intermediate product inside int64
# Multipliers with good spectral properties modulo 2^31 - 1 (Park-Miller / L'Ecuyer MINSTD-type generators).
# A multiplier must not share structure with the table size: with mult = M every product of a small id is
# 0 mod M and almost every n-gram lands in a handful of buckets (lesson 06.4, common mistakes).
MULTIPLIERS = (48271, 69621, 16807, 39373, 742938285, 950706376, 1226874159, 62089911, 1343714438, 2027812808)


def primes_near(target: int, count: int) -> list[int]:
    """``count`` distinct primes at or just below ``target`` (table sizes M_{n,k})."""
    out, c = [], max(3, target)
    while len(out) < count:
        if c > 1 and all(c % p for p in range(2, int(c ** 0.5) + 1)):
            out.append(c)
        c -= 1
    return out


def hash_ngrams(ids: torch.Tensor, n: int, mult: int, mod: int, pad: int) -> torch.Tensor:
    """Multiplicative-XOR hash of the suffix n-gram ending at every position of ``ids`` (B, S) -> (B, S) in [0, mod).

    h = 0; for each token x of the n-gram (oldest first): h = ((h · mult) mod 2^31-1) XOR (x + 1); index = h mod M.
    Missing tokens before the sequence start are ``pad``. The arithmetic is exact in int64 (no overflow)."""
    B, S = ids.shape
    padded = torch.cat([torch.full((B, n - 1), pad, dtype=ids.dtype, device=ids.device), ids], 1)
    h = torch.zeros(B, S, dtype=torch.int64, device=ids.device)
    for j in range(n):
        tok = padded[:, j:j + S].to(torch.int64)
        h = torch.bitwise_xor((h * mult) % _MOD, tok + 1)
    return h % mod


class EngramModule(nn.Module):
    """Engram lookup + gating + causal conv for one layer (module docstring). ``forward(h, ids, cache)``.

    h: (B, T, C) or (B, T, n, C) with ``branches = n`` (one W_K per stream, shared table and W_V);
    ids: (B, T) long, the tokens at these positions; cache: a dict (the decode cache's slot for this layer).
    Returns Y with the shape of h; the caller adds it to the residual.
    """

    def __init__(self, hidden_size: int, vocab_size: int, max_n: int = 3, heads: int = 2, table_size: int = 4093,
                 head_dim: int = 32, kernel: int = 4, branches: int = 1, eps: float = 1e-6):
        super().__init__()
        self.C, self.max_n, self.heads, self.kernel, self.branches = hidden_size, max_n, heads, kernel, branches
        self.dilation = max_n
        self.pad = vocab_size
        orders = list(range(2, max_n + 1))
        primes = primes_near(table_size, len(orders) * heads)
        self.specs = []                     # (n, multiplier, table size) per table
        tables = []
        for oi, n in enumerate(orders):
            for k in range(heads):
                M = primes[oi * heads + k]
                self.specs.append((n, MULTIPLIERS[(oi * heads + k) % len(MULTIPLIERS)], M))
                tables.append(nn.Embedding(M, head_dim))
        self.tables = nn.ModuleList(tables)
        d_mem = len(tables) * head_dim
        self.d_mem = d_mem
        self.k_proj = nn.Linear(d_mem, hidden_size * branches, bias=False)
        self.v_proj = nn.Linear(d_mem, hidden_size, bias=False)
        self.h_norm = RMSNorm(hidden_size, eps=eps)
        self.k_norm = RMSNorm(hidden_size, eps=eps)
        self.v_norm = RMSNorm(hidden_size, eps=eps)
        self.conv = nn.Parameter(torch.zeros(hidden_size, kernel))        # depthwise; zero at initialisation
        self.last_gate: torch.Tensor | None = None

    def lookup(self, full_ids: torch.Tensor, T: int) -> torch.Tensor:
        """e_t for the last T positions of ``full_ids`` (B, S): (B, T, d_mem)."""
        parts = []
        for (n, mult, M), table in zip(self.specs, self.tables):
            idx = hash_ngrams(full_ids, n, mult, M, self.pad)[:, -T:]
            parts.append(table(idx))
        return torch.cat(parts, -1)

    def causal_conv(self, u: torch.Tensor, history: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Depthwise causal conv over time with dilation N: out_t = sum_j w_j · u_{t - j·N}, j = 0..w-1."""
        span = (self.kernel - 1) * self.dilation
        B, T, C = u.shape
        prev = history if history is not None else u.new_zeros(B, span, C)
        full = torch.cat([prev, u], 1)                                  # (B, span + T, C)
        out = torch.zeros_like(u)
        for j in range(self.kernel):
            s = span - j * self.dilation
            out = out + full[:, s:s + T] * self.conv[:, j]
        return out, full[:, -span:] if span else full[:, :0]

    def forward(self, h: torch.Tensor, full_ids: torch.Tensor, cache: dict | None = None, slot: str = "engram"):
        T = h.shape[1]
        e = self.lookup(full_ids, T).to(h.dtype)                       # (B, T, d_mem)
        v = self.v_proj(e)                                              # (B, T, C)
        k = self.k_norm(self.k_proj(e).view(*e.shape[:2], self.branches, self.C))     # (B, T, branches, C)
        streams = h.dim() == 4
        hh = h if streams else h.unsqueeze(-2)                          # (B, T, branches, C)
        alpha = torch.sigmoid((self.h_norm(hh) * k).sum(-1) / self.C ** 0.5)          # (B, T, branches)
        self.last_gate = alpha.detach()
        vt = alpha.unsqueeze(-1) * v.unsqueeze(-2)                       # (B, T, branches, C)
        Bsz = vt.shape[0]
        u = self.v_norm(vt).permute(0, 2, 1, 3).reshape(Bsz * self.branches, T, self.C)
        hist = cache.get(slot) if cache is not None else None
        conv, new_hist = self.causal_conv(u, hist)
        if cache is not None:
            cache[slot] = new_hist
        y = F.silu(conv).view(Bsz, self.branches, T, self.C).permute(0, 2, 1, 3) + vt
        return y if streams else y.squeeze(-2)


def engram_table_params(module: EngramModule) -> int:
    """Parameters in the lookup tables: stored, but only (N-1)·K rows of them are read per token."""
    return sum(t.weight.numel() for t in module.tables)


@torch.no_grad()
def collision_rate(ids: torch.Tensor, n: int, table_size: int, mult: int = MULTIPLIERS[0]) -> dict:
    """On a token stream ``ids`` (1-D): distinct n-grams, distinct hash buckets they occupy, and the share of
    distinct n-grams that share a bucket with another one. Lesson 06.4 uses it on Data-v0."""
    M = primes_near(table_size, 1)[0]
    s = ids.view(1, -1).long()
    h = hash_ngrams(s, n, mult, M, pad=-1)[0, n - 1:]
    grams = torch.stack([s[0, j:s.shape[1] - n + 1 + j] for j in range(n)], 1)
    uniq, inv = torch.unique(grams, dim=0, return_inverse=True)
    bucket_of = torch.full((uniq.shape[0],), -1, dtype=torch.long)
    bucket_of[inv] = h
    buckets, counts = torch.unique(bucket_of, return_counts=True)
    shared = counts[counts > 1].sum().item()
    return {"n": n, "table": M, "distinct_ngrams": int(uniq.shape[0]), "buckets_used": int(buckets.numel()),
            "share_colliding": shared / uniq.shape[0]}
