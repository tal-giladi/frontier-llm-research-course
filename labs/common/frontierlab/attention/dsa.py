"""DeepSeek Sparse Attention (DSA) style selection on top of Baseline-0's GQA (lesson 05.2).

What DeepSeek-V3.2 documents (arXiv 2512.02556, section 2.1; checked 2026-10-04):

* a **lightning indexer** computes, for query token t and preceding token s,

      I_{t,s} = sum_{j=1}^{H^I} w^I_{t,j} * ReLU(q^I_{t,j} . k^I_s)

  with q^I_{t,j} in R^{d^I} and w^I_{t,j} in R derived from h_t, and k^I_s in R^{d^I} from h_s (one
  indexer key per token, shared by the indexer heads); ReLU "for throughput consideration"; it "has a
  small number of heads and can be implemented in FP8";
* **fine-grained token selection**: the main attention of query t runs only over the key-value entries
  whose index scores are in the top-k of I_{t,:}; V3.2 uses k = 2048 (section 2.1.1);
* complexity of the main attention falls from O(L^2) to O(Lk), "although the lightning indexer still has
  a complexity of O(L^2)".

V3.2 instantiates this on MLA in its MQA mode; here the main attention is Baseline-0's GQA (RoPE,
QK-norm) so the only changed variable against Baseline-0 is the selection. ``index_heads`` (H^I),
``index_head_dim`` (d^I) and RoPE on the indexer are this implementation's settings: the report does not
state the head count; DeepSeek-V3.2's ``config.json`` has ``index_n_heads`` 64, ``index_head_dim`` 128,
``index_topk`` 2048.

The indexer is trained with its own objective (section 2.1.1), implemented by :func:`indexer_kl`:

* target: the main attention probabilities summed over heads, then L1-normalised over keys: p_{t,:};
* **dense warm-up**: dense attention, every parameter frozen except the indexer, loss
  L^I = sum_t KL(p_{t,:} || Softmax(I_{t,:}));
* **sparse training**: selection on, loss L^I = sum_t KL(p_{t,S_t} || Softmax(I_{t,S_t})) over the selected
  set S_t only; "we detach the indexer input from the computational graph", the indexer is trained only by
  L^I and the main model only by the language-modelling loss.

Here the indexer always reads ``x.detach()`` and the target p is computed under ``no_grad``, so L^I can
never reach the main model; the top-k mask is discrete, so the LM loss can never reach the indexer. The
two losses can therefore simply be added in one backward pass (``labs/module-05/train_arm.py``). We
average the KL over queries instead of summing (a constant factor) and renormalise p over S_t in the
sparse stage (the report writes p_{t,S_t} without saying whether it is renormalised; INFERENCE).

Two compute paths for the main attention, which must agree (tested):

* ``dsa_path="mask"`` [default] — full (T, S) logits with every unselected key masked. Simple and
  exact, used for training and the correctness suite; it does O(T S) work, so it is never faster than
  dense attention.
* ``dsa_path="gather"`` — gathers the k selected keys and values per query and attends over them only:
  O(T k) attention work plus the gather's memory traffic. This is the path 05.3 times. Queries are
  processed in blocks of ``gather_block`` to bound the (B, KV, block, k, d) gathered tensors.

Either way the indexer itself computes all T x S scores: that is the O(L^2) term.

LayerCache: ``k``, ``v``, ``pos`` as in GQA, plus ``idx_k`` (B, S, d^I), the cached indexer keys.

``cfg.extra``: ``index_topk`` [64], ``index_heads`` [4], ``index_head_dim`` [32], ``index_rope`` [True],
``dsa_mode`` ["sparse" | "dense"], ``dsa_path`` ["mask" | "gather"], ``gather_block`` [256].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from frontierlab.attention.base import LayerCache, RotaryEmbedding, apply_rope, causal_mask, register
from frontierlab.attention.gqa import GQAttention
from frontierlab.attention.ops import _expand_kv


def index_scores(iq: torch.Tensor, ik: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Lightning-indexer scores. iq (B, H_I, T, d_I), ik (B, S, d_I), w (B, T, H_I) -> I (B, T, S)."""
    dots = torch.einsum("bjtd,bsd->bjts", iq, ik).relu()
    return torch.einsum("btj,bjts->bts", w, dots)


def topk_mask(scores: torch.Tensor, allowed: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean selection mask (B, T, S) and indices (B, T, min(k, S)) of the top-k allowed scores.

    Rows with fewer than k allowed keys select all of them (the extra top-k picks land on disallowed
    entries and are removed by ``& allowed``). With k >= S every allowed key is selected: dense attention.
    """
    S = scores.shape[-1]
    masked = scores.masked_fill(~allowed, float("-inf"))
    kk = min(k, S)
    idx = masked.topk(kk, dim=-1).indices
    sel = torch.zeros_like(masked, dtype=torch.bool).scatter_(-1, idx, True) & allowed
    return sel, idx


def indexer_kl(p_target: torch.Tensor, scores: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Mean over queries of KL(p || Softmax(I)) restricted to ``support`` (B, T, S) bool.

    ``p_target`` (B, T, S): summed-over-heads main attention probabilities (no gradient). It is
    renormalised over the support (L1), as is the indexer softmax. Dense warm-up: support = causal set;
    sparse stage: support = selected set S_t.
    """
    p = p_target.masked_fill(~support, 0.0)
    p = p / p.sum(-1, keepdim=True).clamp_min(torch.finfo(p.dtype).tiny)
    logq = torch.log_softmax(scores.masked_fill(~support, float("-inf")), dim=-1)
    terms = torch.where(p > 0, p * (torch.log(p.clamp_min(torch.finfo(p.dtype).tiny)) - logq), torch.zeros_like(p))
    return terms.sum(-1).mean()


def attention_mass_recall(p_target: torch.Tensor, sel: torch.Tensor) -> float:
    """Share of the (L1-normalised, head-summed) main attention mass that falls on the selected keys."""
    p = p_target / p_target.sum(-1, keepdim=True)
    return float((p * sel).sum(-1).mean())


@register("dsa")
class DSAttention(GQAttention):
    def __init__(self, cfg):
        super().__init__(cfg)
        ex = cfg.extra
        C = cfg.hidden_size
        self.topk = int(ex.get("index_topk", 64))
        self.HI = int(ex.get("index_heads", 4))
        self.dI = int(ex.get("index_head_dim", 32))
        self.mode = ex.get("dsa_mode", "sparse")
        self.path = ex.get("dsa_path", "mask")
        self.gather_block = int(ex.get("gather_block", 256))
        self.idx_q = nn.Linear(C, self.HI * self.dI, bias=False)
        self.idx_k = nn.Linear(C, self.dI, bias=False)
        self.idx_w = nn.Linear(C, self.HI, bias=False)
        self.idx_rope = RotaryEmbedding(self.dI, cfg.rope_theta) if ex.get("index_rope", True) else None
        self.collect = False              # set by the training wrapper: compute L^I in forward
        self.indexer_loss = None          # last L^I (scalar tensor) when collect is True
        self.last_recall = None           # share of main attention mass on the selected keys (diagnostic)
        self.layer_type = "dsa"

    # --- the indexer ---------------------------------------------------------------------------
    def indexer(self, x: torch.Tensor, positions: torch.Tensor):
        """iq (B, H_I, T, d_I), ik (B, T, d_I), w (B, T, H_I) from x.detach() (the documented sparse-stage rule)."""
        xd = x.detach()
        B, T, _ = x.shape
        iq = self.idx_q(xd).view(B, T, self.HI, self.dI).transpose(1, 2)
        ik = self.idx_k(xd)
        if self.idx_rope is not None:
            cos, sin = self.idx_rope(positions)
            iq = apply_rope(iq, cos, sin)
            ik = apply_rope(ik.unsqueeze(1), cos, sin).squeeze(1)
        return iq, ik, self.idx_w(xd)

    def project(self, x, positions):
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.H, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.KV, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.KV, self.hd).transpose(1, 2)
        cos, sin = self.rope(positions)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def select(self, scores, allowed):
        if self.mode == "dense":
            return allowed.expand_as(scores), None
        return topk_mask(scores, allowed, self.topk)

    # --- main attention ------------------------------------------------------------------------
    def attend_masked(self, q, k, v, sel):
        """Explicit softmax over the selected keys. q (B,H,T,d), k/v (B,KV,S,d), sel (B,T,S) -> y, probs."""
        logits = (q @ _expand_kv(k, self.H).transpose(-1, -2)) * self.hd ** -0.5
        logits = logits.masked_fill(~sel[:, None], float("-inf"))
        work = torch.promote_types(logits.dtype, torch.float32)
        p = torch.softmax(logits.to(work), dim=-1).to(q.dtype)
        return p @ _expand_kv(v, self.H), p

    def attend_gather(self, q, k, v, scores, allowed):
        """O(T k) attention over gathered keys (no probabilities kept). Equal to attend_masked."""
        B, H, T, d = q.shape
        S = k.shape[2]
        kk = min(self.topk, S)
        masked = scores.masked_fill(~allowed, float("-inf"))
        rep = H // self.KV
        out = []
        for s in range(0, T, self.gather_block):
            e = min(s + self.gather_block, T)
            idx = masked[:, s:e].topk(kk, dim=-1).indices                       # (B, t, k)
            valid = torch.gather(allowed.expand(B, -1, -1)[:, s:e], -1, idx)     # (B, t, k)
            gi = idx[:, None, :, :, None].expand(B, self.KV, e - s, kk, d)
            kg = torch.gather(k[:, :, None].expand(B, self.KV, e - s, S, d), 3, gi)   # (B, KV, t, k, d)
            vg = torch.gather(v[:, :, None].expand(B, self.KV, e - s, S, d), 3, gi)
            qb = q[:, :, s:e].view(B, self.KV, rep, e - s, d)
            logits = torch.einsum("bgrtd,bgtkd->bgrtk", qb, kg) * d ** -0.5
            logits = logits.masked_fill(~valid[:, None, None], float("-inf"))
            work = torch.promote_types(logits.dtype, torch.float32)
            p = torch.softmax(logits.to(work), dim=-1).to(q.dtype)
            out.append(torch.einsum("bgrtk,bgtkd->bgrtd", p, vg).reshape(B, H, e - s, d))
        return torch.cat(out, dim=2)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, cache: LayerCache | None = None):
        B, T, _ = x.shape
        q, k, v = self.project(x, positions)
        iq, ik, w = self.indexer(x, positions)
        k_pos = positions
        if cache is not None:
            if "k" in cache:
                k = torch.cat((cache["k"], k), dim=2)
                v = torch.cat((cache["v"], v), dim=2)
                ik = torch.cat((cache["idx_k"], ik), dim=1)
                k_pos = torch.cat((cache["pos"], positions))
            cache["k"], cache["v"], cache["idx_k"], cache["pos"] = k, v, ik, k_pos
        scores = index_scores(iq, ik, w)                                         # (B, T, S): the O(L^2) part
        allowed = causal_mask(positions, k_pos)[None]                            # (1, T, S)
        if self.path == "gather" and self.mode == "sparse" and not self.collect:
            y = self.attend_gather(q, k, v, scores, allowed)
        else:
            sel, _ = self.select(scores, allowed)
            y, p = self.attend_masked(q, k, v, sel)
            if self.collect:
                with torch.no_grad():
                    # dense mode: p is the dense attention; sparse mode: p is already zero outside S_t
                    target = p.sum(1).detach().to(scores.dtype)
                    self.last_recall = None if self.mode == "sparse" else self._recall(target, scores, allowed)
                self.indexer_loss = indexer_kl(target, scores, sel)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.hd))

    def _recall(self, target, scores, allowed):
        """Attention-mass recall of three selections of the same size k (dense mode only):
        the indexer's top-k, the oracle top-k of the attention itself (the best any selector of k keys
        can do) and the k most recent keys (a sliding window: what selection must beat to be worth it)."""
        t = target.float()
        sel, _ = topk_mask(scores.detach().float(), allowed, self.topk)
        oracle, _ = topk_mask(t, allowed, self.topk)
        T, S = allowed.shape[-2:]
        q_pos = torch.arange(S - T, S, device=t.device)
        recent = allowed & (torch.arange(S, device=t.device)[None, :] > q_pos[:, None] - self.topk)[None]
        return {"indexer": attention_mass_recall(t, sel), "oracle": attention_mass_recall(t, oracle),
                "window": attention_mass_recall(t, recent.expand_as(t))}


def dsa_layers(model) -> list[DSAttention]:
    return [m for m in model.modules() if isinstance(m, DSAttention)]


def set_dsa(model, *, mode: str | None = None, path: str | None = None, topk: int | None = None,
            collect: bool | None = None) -> int:
    """Change mode ("sparse" / "dense"), path ("mask" / "gather"), top-k or loss collection on every DSA layer."""
    layers = dsa_layers(model)
    for m in layers:
        if mode is not None:
            m.mode = mode
        if path is not None:
            m.path = path
        if topk is not None:
            m.topk = int(topk)
        if collect is not None:
            m.collect = collect
    return len(layers)


def indexer_parameters(model):
    return [p for m in dsa_layers(model) for n, p in m.named_parameters() if n.startswith("idx_")]


def indexer_loss(model) -> torch.Tensor | None:
    """Sum of L^I over the DSA layers from the last forward with collect=True."""
    losses = [m.indexer_loss for m in dsa_layers(model) if m.indexer_loss is not None]
    return sum(losses) if losses else None


@torch.no_grad()
def selection_recall(model, idx: torch.Tensor, topk: int | None = None) -> list[dict]:
    """Per DSA layer: share of the DENSE main attention mass kept by the indexer's top-k, by the oracle
    top-k and by the k most recent keys ({"indexer", "oracle", "window"}; 1.0 = all of it).

    Runs the model in dense mode, so the attention being measured is the unselected one.
    """
    layers = dsa_layers(model)
    saved = [(m.mode, m.collect, m.topk) for m in layers]
    out = []
    try:
        for m in layers:
            m.mode, m.collect = "dense", True
            if topk is not None:
                m.topk = topk
        model(idx)
        out = [m.last_recall for m in layers]
    finally:
        for m, (mo, co, tk) in zip(layers, saved):
            m.mode, m.collect, m.topk = mo, co, tk
            m.indexer_loss = None
    return out
