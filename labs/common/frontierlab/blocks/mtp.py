"""Multi-token prediction (lesson 06.1): Meta's parallel heads and DeepSeek-V3's sequential modules.

Both add training targets beyond the next token; they differ in what each extra prediction may see.

**Meta (Gloeckle et al., arXiv 2404.19737, section 2).** A shared trunk produces z_t; n independent output
heads (each a transformer layer) and a shared unembedding predict x_{t+1}, ..., x_{t+n} from z_t alone:
loss L_n = -sum_t sum_{i=1..n} log P(x_{t+i} | z_t) (Eq. 2). Parameter-matched by moving layers: "when we
add n − 1 layers in future prediction heads, we remove n − 1 layers from the shared model trunk"
(section 3.1). Here head 1 is the model's own last layer, so the trunk is layers 0..L-2 and the next-token
path (and its decode cache) is the unchanged Baseline-0 path; heads 2..n are extra blocks on z, the
input of the last layer, each with its own final RMSNorm (a course choice) and the shared ``lm_head``.

**DeepSeek-V3 (arXiv 2412.19437, section 2.2, Eqs. 21–25).** D sequential modules. Module k combines
the previous depth's representation with the embedding of the token k positions ahead and runs one
transformer block:

    h'^k_i = M_k [ RMSNorm(h^{k-1}_i) ; RMSNorm(Emb(t_{i+k})) ]        M_k in R^{C x 2C}  (``eh_proj``)
    h^k_{1:T-k} = TRM_k(h'^k_{1:T-k}),   P^k_{i+k+1} = OutHead(h^k_i)    shared Emb and OutHead
    L_MTP = (lambda / D) · sum_k L^k_MTP

h^0 is the main model's final hidden state (before its final norm). Because module k sees t_{i+k}, the
prediction of t_{i+k+1} keeps "the complete causal chain" (section 2.2). DeepSeek-V3 uses D = 1, lambda
= 0.3 for the first 10T tokens and 0.1 for the remaining 4.8T (section 4.2), and discards the modules at
inference or reuses them for speculative decoding; it reports a second-token acceptance rate of 85–90%
and 1.8 times the tokens per second (section 5.4.3).

Module names follow DeepSeek-V3's checkpoint where one exists (``enorm``, ``hnorm``, ``eh_proj``, a
block, ``norm`` for its shared-head norm) under ``mtp.layers.{k}`` (the
layout Qwen3-Next's ``mtp.layers.0`` also uses); Meta heads are ``mtp.heads.{k}``.

Self-speculative greedy decoding (:func:`speculative_greedy`) drafts with the extra predictions and
verifies with the main head; it returns exactly the plain greedy output (tested) and counts how many
drafts were accepted — the acceptance rate Module 15 turns into latency.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from frontierlab.layers.rmsnorm import RMSNorm
from frontierlab.model.lm import Block

DEEPSEEK_V3_LAMBDA = ((0.0, 0.3), (10.0 / 14.8, 0.1))   # (fraction of training tokens, lambda) from V3 section 4.2


def lambda_at(step: int, steps: int, schedule=DEEPSEEK_V3_LAMBDA) -> float:
    """Piecewise-constant MTP loss weight: the last (fraction, value) pair whose fraction <= step / steps."""
    frac = step / max(1, steps)
    val = schedule[0][1]
    for f, v in schedule:
        if frac >= f:
            val = v
    return float(val)


class MetaHead(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.block = Block(cfg, layer_idx)
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)


class MetaHeads(nn.Module):
    """Heads 2..n of Meta's design: ``n_extra`` = n − 1 blocks on the trunk output z."""

    def __init__(self, cfg, n_extra: int):
        super().__init__()
        L = cfg.num_hidden_layers
        self.heads = nn.ModuleList(MetaHead(cfg, L + k) for k in range(n_extra))

    def forward(self, z: torch.Tensor, positions: torch.Tensor, lm_head) -> list[torch.Tensor]:
        """Logits of head k = 2..n for every position: list of (B, T, V); head k at position i predicts x_{i+k}."""
        return [lm_head(h.norm(h.block(z, positions))) for h in self.heads]


class DeepSeekModule(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        C = cfg.hidden_size
        self.enorm = RMSNorm(C, eps=cfg.rms_norm_eps)
        self.hnorm = RMSNorm(C, eps=cfg.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * C, C, bias=False)
        self.block = Block(cfg, layer_idx)
        self.norm = RMSNorm(C, eps=cfg.rms_norm_eps)

    def combine(self, h_prev: torch.Tensor, emb_ahead: torch.Tensor) -> torch.Tensor:
        """h' = M_k [RMSNorm(h^{k-1}); RMSNorm(Emb(t_{i+k}))], both (B, S, C) -> (B, S, C). Order as in V3 Eq. 21."""
        return self.eh_proj(torch.cat([self.hnorm(h_prev), self.enorm(emb_ahead)], dim=-1))


class DeepSeekMTP(nn.Module):
    """D sequential MTP modules (module docstring)."""

    def __init__(self, cfg, depth: int):
        super().__init__()
        L = cfg.num_hidden_layers
        self.layers = nn.ModuleList(DeepSeekModule(cfg, L + k) for k in range(depth))

    @property
    def depth(self) -> int:
        return len(self.layers)

    def forward(self, h0: torch.Tensor, idx: torch.Tensor, embed, lm_head) -> list[torch.Tensor]:
        """Logits of depth k = 1..D: list of (B, T-k, V); entry i of depth k predicts t_{i+k+1} (0-based ids)."""
        out, h = [], h0
        T = idx.shape[1]
        for k, mod in enumerate(self.layers, start=1):
            S = T - k
            if S <= 0:
                break
            hp = mod.combine(h[:, :S], embed(idx[:, k:k + S]))
            h = mod.block(hp, torch.arange(S, device=idx.device))
            out.append(lm_head(mod.norm(h)))
        return out


def shifted_ce(logits: torch.Tensor, labels: torch.Tensor, shift: int) -> torch.Tensor:
    """Mean cross-entropy of ``logits`` (B, S, V) at positions i against ``labels[:, i + shift]``."""
    S = min(logits.shape[1], labels.shape[1] - shift)
    if S <= 0:
        return logits.new_zeros(())
    lg = logits[:, :S].float() if logits.dtype in (torch.float16, torch.bfloat16) else logits[:, :S]
    return F.cross_entropy(lg.reshape(-1, lg.size(-1)), labels[:, shift:shift + S].reshape(-1))


def meta_losses(head_logits: list[torch.Tensor], labels: torch.Tensor) -> list[torch.Tensor]:
    """Head k = 2.. at position i predicts labels[i + k]."""
    return [shifted_ce(lg, labels, k) for k, lg in enumerate(head_logits, start=2)]


def deepseek_losses(depth_logits: list[torch.Tensor], labels: torch.Tensor) -> list[torch.Tensor]:
    """Depth k at position i predicts labels[i + k + 1] (V3 Eq. 24, averaged over valid positions)."""
    return [shifted_ce(lg, labels, k + 1) for k, lg in enumerate(depth_logits, start=1)]


@torch.no_grad()
def draft_tokens(model, seq: torch.Tensor) -> tuple[int, list[int]]:
    """(main head's greedy next token, greedy drafts for the tokens after it) for a (1, S) sequence.

    Meta: heads 2..n at the last position draft x_{S+1}, ..., x_{S+n-1} directly from z.
    DeepSeek: depth 1 needs Emb(t_{i+1}) = the main head's choice; depth k chains on depth k-1's choice."""
    out = model(seq, return_hidden=True)
    y1 = int(out.logits[0, -1].argmax())
    kind = model.blocks["mtp"]
    if kind == "meta":
        pos = torch.arange(seq.shape[1], device=seq.device)
        heads = model.mtp(out.hidden["z"], pos, model.lm_head)
        return y1, [int(h[0, -1].argmax()) for h in heads]
    if kind == "deepseek":
        h = out.hidden["h"][:, -1:]
        drafts, prev = [], y1
        for mod in model.mtp.layers:
            hp = mod.combine(h, model.model.embed_tokens(torch.tensor([[prev]], device=seq.device)))
            h = mod.block(hp, torch.zeros(1, dtype=torch.long, device=seq.device))
            prev = int(model.lm_head(mod.norm(h))[0, -1].argmax())
            drafts.append(prev)
        return y1, drafts
    return y1, []


@torch.no_grad()
def speculative_greedy(model, prompt: torch.Tensor, max_new: int) -> tuple[torch.Tensor, dict]:
    """Greedy decoding with self-drafted tokens, verified by the main head (recomputes without a cache; the
    acceptance count does not depend on caching). Returns (tokens (1, S + max_new), stats).

    Each round: draft y1 (main) + d_2..d_m (MTP); one verification forward over seq + [y1, d_2, ...] gives
    the main head's greedy choice at every draft position; drafts are accepted while they match. Output
    equals plain greedy decoding exactly."""
    was = model.training
    model.eval()
    seq = prompt.clone()
    target = prompt.shape[1] + max_new
    proposed = accepted = rounds = 0
    while seq.shape[1] < target:
        y1, drafts = draft_tokens(model, seq)
        cand = torch.cat([seq, torch.tensor([[y1] + drafts], device=seq.device)], 1)
        rounds += 1
        keep = [y1]
        if drafts:
            verify = model(cand).logits[0].argmax(-1)               # verify[j] = main greedy after cand[:j+1]
            base = seq.shape[1]
            for j, d in enumerate(drafts):
                if seq.shape[1] + len(keep) >= target:
                    break
                proposed += 1
                if int(verify[base + j]) == d:
                    accepted += 1
                    keep.append(d)
                else:
                    break
        seq = torch.cat([seq, torch.tensor([keep], device=seq.device)], 1)[:, :target]
    model.train(was)
    return seq, {"proposed": proposed, "accepted": accepted, "rounds": rounds,
                 "acceptance_rate": accepted / proposed if proposed else float("nan"),
                 "tokens_per_round": max_new / rounds}


@torch.no_grad()
def teacher_forced_acceptance(model, idx: torch.Tensor) -> dict:
    """First-draft agreement on real text, without generating (the cheap offline proxy). idx: (B, T).

    At position i the main head's greedy choice is y1_i = argmax p(. | x_{<=i}). The first draft d_i is
    head 2's prediction from z_i (Meta) or depth 1's prediction from h_i and Emb(x_{i+1}) (DeepSeek, teacher
    forced). It is compared with the main head's greedy choice one position later, given the real prefix
    x_{<=i+1}. Where y1_i equals the real x_{i+1} this is exactly the acceptance test of greedy speculative
    decoding (``exact_*`` fields); elsewhere it is a proxy. :func:`speculative_greedy` gives the exact
    generative number on a few prompts."""
    was = model.training
    model.eval()
    out = model(idx, return_hidden=True)
    main_next = out.logits.argmax(-1)                                      # (B, T): greedy for x_{i+1}
    T = idx.shape[1]
    pos = torch.arange(T, device=idx.device)
    kind = model.blocks["mtp"]
    if kind == "meta":
        draft = model.mtp(out.hidden["z"], pos, model.lm_head)[0].argmax(-1)[:, :T - 1]     # for x_{i+2}
    elif kind == "deepseek":
        mod = model.mtp.layers[0]
        hp = mod.combine(out.hidden["h"][:, :T - 1], model.model.embed_tokens(idx[:, 1:]))
        draft = model.lm_head(mod.norm(mod.block(hp, pos[:T - 1]))).argmax(-1)               # for x_{i+2}
    else:
        raise ValueError("model has no MTP")
    model.train(was)
    agree = draft == main_next[:, 1:]                                      # (B, T-1)
    exact = main_next[:, :T - 1] == idx[:, 1:]                             # main got x_{i+1} right
    n_exact = int(exact.sum())
    return {"positions": int(agree.numel()), "agreement": float(agree.float().mean()),
            "exact_positions": n_exact,
            "exact_acceptance": float((agree & exact).sum()) / n_exact if n_exact else float("nan")}
