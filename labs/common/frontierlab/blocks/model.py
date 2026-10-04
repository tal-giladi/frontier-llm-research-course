"""BlockLM: Baseline-0 with the Module 6 block changes, switched on from ``cfg.extra["blocks"]``.

``BlockLM`` subclasses :class:`frontierlab.model.LM`, so with every switch off it *is* Baseline-0 (same
modules, same Hugging Face Qwen3 names, same initialisation draws, bit-identical logits — tested). The
switches, all in ``cfg.extra["blocks"]`` (so they land in the run card and the checkpoint):

    mtp        "none" | "meta" | "deepseek"   multi-token prediction (lesson 06.1); ``mtp_depth`` extra
               predictions (Meta: heads 2..depth+1; DeepSeek: D = depth); ``mtp_lambda`` weight (default 1.0
               for Meta, Eq. 2; 0.3 for DeepSeek, V3 section 4.2)
    residual   "plain" | "hc" | "mhc"          n-stream residual (lesson 06.2); ``streams`` n, ``sinkhorn_iters``
    ffn        "dense" | "moe" | "matformer"   SwiGLU, fine-grained MoE (``moe`` dict), nested FFN (06.5)
    engram     {"layers": [...], ...}          conditional memory at those layers (06.4)
    ple        {"dim": d}                      Gemma 3n-style per-layer embeddings (06.5)

Training loss = next-token loss + mtp_lambda · MTP loss + moe_aux_coef · balance loss. ``per_token_loss``
is always the plain next-token loss of the main head, so Eval v0 (held-out windows) compares arms on the
same quantity. ``extras`` (on the output) and ``last_stats`` (on the model) hold the components.

The attention kind is still ``cfg.attention``: any Module 3–5 kind combines with these switches.
New parameter names (outside Qwen3): ``model.hyper.{i}.{attn,mlp}.*`` (HC/mHC), ``model.engram.{i}.*``,
``model.ple.*``, ``mtp.heads.{k}.*`` / ``mtp.layers.{k}.*``; MoE layers use Qwen3-MoE names
(``mlp.gate.weight``, ``mlp.experts.gate_up_proj``, ``mlp.experts.down_proj``, ``mlp.shared_expert.*``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from frontierlab.blocks.engram import EngramModule
from frontierlab.blocks.hyperconn import RESIDUALS, amax_gain, collapse_streams, expand_streams
from frontierlab.blocks.matformer import NestedMLP, PerLayerEmbedding, matformer_widths
from frontierlab.blocks.moe import MoEFFN
from frontierlab.blocks.mtp import DeepSeekMTP, MetaHeads, deepseek_losses, meta_losses
from frontierlab.model.config import ModelConfig
from frontierlab.model.lm import LM, LMOutput

DEFAULTS = {"mtp": "none", "mtp_depth": 1, "mtp_lambda": None, "residual": "plain", "streams": 4,
            "sinkhorn_iters": 20, "ffn": "dense", "moe": None, "matformer": None, "engram": None, "ple": None,
            "seed": 0}
MOE_DEFAULTS = {"experts": 8, "top_k": 2, "intermediate": None, "shared": 1, "aux_coef": 0.01, "first_dense": 1}
ENGRAM_DEFAULTS = {"layers": [1], "max_n": 3, "heads": 2, "table": 4093, "head_dim": 32, "kernel": 4}


def block_settings(cfg: ModelConfig) -> dict:
    b = {**DEFAULTS, **cfg.extra.get("blocks", {})}
    if b["mtp_lambda"] is None:              # Meta weights every head equally; DeepSeek-V3 starts at 0.3
        b["mtp_lambda"] = 1.0 if b["mtp"] == "meta" else 0.3
    if b["ffn"] == "moe":
        b["moe"] = {**MOE_DEFAULTS, **(b["moe"] or {})}
        if b["moe"]["intermediate"] is None:
            b["moe"]["intermediate"] = max(1, cfg.intermediate_size // (2 * b["moe"]["top_k"]))
    if b["ffn"] == "matformer":
        b["matformer"] = {"widths": matformer_widths(cfg.intermediate_size), **(b["matformer"] or {})}
    if b["engram"] is not None:
        b["engram"] = {**ENGRAM_DEFAULTS, **b["engram"]}
    return b


def with_blocks(cfg: ModelConfig, **settings) -> ModelConfig:
    """A copy of ``cfg`` with ``extra["blocks"]`` updated (``with_blocks(cfg, mtp="deepseek")``)."""
    blocks = {**cfg.extra.get("blocks", {}), **settings}
    return cfg.with_(extra={**cfg.extra, "blocks": blocks})


@dataclass
class BlockLMOutput(LMOutput):
    extras: dict = field(default_factory=dict)
    hidden: dict = field(default_factory=dict)


class BlockLM(LM):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)                       # Baseline-0 modules, names and initialisation, unchanged
        b = self.blocks = block_settings(cfg)
        L, C, std = cfg.num_hidden_layers, cfg.hidden_size, cfg.initializer_range
        new: list[nn.Module] = []
        if b["ffn"] == "moe":
            m = b["moe"]
            for i in range(m["first_dense"], L):
                layer = self.model.layers[i]
                layer.mlp = MoEFFN(C, m["experts"], m["top_k"], m["intermediate"], m["shared"])
                layer.mlp.reset_parameters(std)
                if layer.mlp.shared_expert is not None:
                    new.append(layer.mlp.shared_expert)
        elif b["ffn"] == "matformer":
            for layer in self.model.layers:
                old = layer.mlp
                layer.mlp = NestedMLP(C, cfg.intermediate_size)
                layer.mlp.load_state_dict(old.state_dict())            # same draws as Baseline-0
            self.register_buffer("mat_calls", torch.zeros((), dtype=torch.long))
            self.mat_eval_widths: list[int] | None = None
        self.n_streams = 1
        if b["residual"] != "plain":
            cls = RESIDUALS[b["residual"]]
            n = self.n_streams = int(b["streams"])
            kw = {"t_max": int(b["sinkhorn_iters"])} if b["residual"] == "mhc" else {}
            self.model.hyper = nn.ModuleList(
                nn.ModuleDict({"attn": cls(n, C, 2 * i, eps=cfg.rms_norm_eps, **kw),
                               "mlp": cls(n, C, 2 * i + 1, eps=cfg.rms_norm_eps, **kw)}) for i in range(L))
        if b["engram"] is not None:
            e = b["engram"]
            self.model.engram = nn.ModuleDict({
                str(i): EngramModule(C, cfg.vocab_size, e["max_n"], e["heads"], e["table"], e["head_dim"], e["kernel"],
                                     branches=self.n_streams, eps=cfg.rms_norm_eps) for i in e["layers"]})
            new.append(self.model.engram)
        if b["ple"] is not None:
            self.model.ple = PerLayerEmbedding(cfg.vocab_size, C, L, int(b["ple"]["dim"]), eps=cfg.rms_norm_eps)
            new.append(self.model.ple)
        if b["mtp"] == "meta":
            self.mtp = MetaHeads(cfg, int(b["mtp_depth"]))
            new.append(self.mtp)
        elif b["mtp"] == "deepseek":
            self.mtp = DeepSeekMTP(cfg, int(b["mtp_depth"]))
            new.append(self.mtp)
        elif b["mtp"] != "none":
            raise ValueError(f"unknown mtp {b['mtp']!r}")
        for mod in new:
            mod.apply(self._init)
        self.mtp_lambda = float(b["mtp_lambda"])
        self.track_hyper = False
        self.last_stats: dict = {}

    # --- MatFormer granularity -------------------------------------------------------------------------
    def set_widths(self, widths: list[int] | int | None):
        """Fix the FFN width of every layer (int) or per layer (list) for evaluation; None = full width."""
        if isinstance(widths, int):
            widths = [widths] * self.config.num_hidden_layers
        self.mat_eval_widths = widths

    def _apply_widths(self):
        b = self.blocks
        if b["ffn"] != "matformer":
            return None
        full = [self.config.intermediate_size] * self.config.num_hidden_layers
        if self.training:
            gw = b["matformer"]["widths"]
            g = torch.Generator().manual_seed(int(b["seed"]) * 1_000_003 + int(self.mat_calls))
            widths = [gw[int(torch.randint(len(gw), (1,), generator=g))]] * self.config.num_hidden_layers
            self.mat_calls += 1
        else:
            widths = self.mat_eval_widths or full
        for layer, w in zip(self.model.layers, widths):
            layer.mlp.width = int(w)
        return widths

    # --- forward -----------------------------------------------------------------------------------------
    def new_cache(self):
        cache = super().new_cache()
        cache.extra = {}
        return cache

    def forward(self, idx: torch.Tensor | None = None, labels: torch.Tensor | None = None, cache=None,
                reduction: str = "mean", inputs_embeds: torch.Tensor | None = None,
                return_hidden: bool = False) -> BlockLMOutput:
        b, cfg = self.blocks, self.config
        x = self.model.embed_tokens(idx) if inputs_embeds is None else inputs_embeds
        B, T = x.shape[:2]
        start = cache.length if cache is not None else 0
        positions = torch.arange(start, start + T, device=x.device)
        widths = self._apply_widths()
        full_ids = None
        if b["engram"] is not None:
            if idx is None:
                raise ValueError("Engram needs token ids (inputs_embeds alone is not enough)")
            prev = cache.extra.get("ids") if cache is not None else None
            full_ids = idx if prev is None else torch.cat([prev, idx], 1)
            if cache is not None:
                cache.extra["ids"] = full_ids[:, -(b["engram"]["max_n"] - 1):] if b["engram"]["max_n"] > 1 else full_ids[:, :0]
        ple_in = self.model.ple.inputs(idx, x) if b["ple"] is not None else None
        n = self.n_streams
        X = expand_streams(x, n) if b["residual"] != "plain" else x
        L = cfg.num_hidden_layers
        z = None
        res_mats = []
        for i, layer in enumerate(self.model.layers):
            lc = cache.layers[i] if cache is not None else None
            if b["engram"] is not None and str(i) in self.model.engram:
                slot = cache.extra.setdefault(f"engram{i}", {}) if cache is not None else None
                X = X + self.model.engram[str(i)](X, full_ids, slot)
            if i == L - 1 and b["mtp"] == "meta":
                z = collapse_streams(X) if X.dim() == 4 else X

            def fa(h, layer=layer, lc=lc):
                return layer.self_attn(layer.input_layernorm(h), positions, lc)

            def fm(h, layer=layer):
                return layer.mlp(layer.post_attention_layernorm(h))

            if b["residual"] == "plain":
                X = X + fa(X)
                X = X + fm(X)
            else:
                hy = self.model.hyper[i]
                for key, f in (("attn", fa), ("mlp", fm)):
                    hy[key].track = self.track_hyper
                    X = hy[key](X, f)
                    if self.track_hyper:
                        res_mats.append(hy[key].last_res)
            if ple_in is not None:
                p_l = ple_in[:, :, i]
                X = X + self.model.ple.layer(i, X, p_l.unsqueeze(-2) if X.dim() == 4 else p_l)
        h = collapse_streams(X) if X.dim() == 4 else X
        if cache is not None:
            cache.length += T
        logits = self.lm_head(self.model.norm(h))
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        extras: dict = {}
        if widths is not None:
            extras["matformer_width"] = widths[0]
        if res_mats:
            extras["hyper"] = amax_gain(res_mats)
        hidden = {"h": h, "z": z} if return_hidden else {}
        if labels is None:
            return BlockLMOutput(logits, extras=extras, hidden=hidden)
        tok = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
                              reduction="none").view(B, T - 1)
        main = tok.mean() if reduction == "mean" else tok.sum()
        loss = main
        if b["mtp"] != "none" and cache is None:
            if b["mtp"] == "meta":
                parts = meta_losses(self.mtp(z, positions, self.lm_head), labels)
                mtp_loss = torch.stack(parts).sum()                 # Meta Eq. 2: every head counts equally
            else:
                parts = deepseek_losses(self.mtp(h, labels, self.model.embed_tokens, self.lm_head), labels)
                mtp_loss = torch.stack(parts).mean()                # V3 Eq. 25: (lambda / D) sum_k
            loss = loss + self.mtp_lambda * mtp_loss
            extras["mtp_loss"] = float(mtp_loss.detach())
            extras["mtp_losses"] = [float(p.detach()) for p in parts]
        if b["ffn"] == "moe":
            auxes = [layer.mlp.last_aux for layer in self.model.layers if isinstance(layer.mlp, MoEFFN)]
            aux = torch.stack(auxes).mean()
            if self.training:
                loss = loss + b["moe"]["aux_coef"] * aux
            extras["moe_aux"] = float(aux.detach())
            extras["moe_max_load"] = max(_max_load(layer.mlp) for layer in self.model.layers
                                         if isinstance(layer.mlp, MoEFFN))
            for layer in self.model.layers:              # drop graph references (keeps the model deep-copyable)
                if isinstance(layer.mlp, MoEFFN):
                    layer.mlp.last_aux = None
        extras["main_loss"] = float(main.detach())
        self.last_stats = extras
        return BlockLMOutput(logits, loss, tok, extras=extras, hidden=hidden)


@torch.no_grad()
def _max_load(moe: MoEFFN) -> float:
    """Largest share of routing slots any expert received, times E (1.0 = perfectly balanced)."""
    r = moe.last_router
    E = moe.gate.num_experts
    counts = torch.bincount(r.topk_idx.reshape(-1), minlength=E).float()
    return float(counts.max() / counts.mean())


def build(cfg: ModelConfig) -> LM:
    """``BlockLM(cfg)`` if ``cfg.extra`` has a ``blocks`` entry, else the plain ``LM`` (identical anyway)."""
    return BlockLM(cfg) if "blocks" in cfg.extra else LM(cfg)


def load_model(checkpoint, map_location="cpu") -> BlockLM:
    """Rebuild a Module 6 model from a checkpoint written by the course loop (``frontierlab.blocks.train``)."""
    ck = torch.load(checkpoint, map_location=map_location, weights_only=False)
    cfg = ModelConfig(**ck["config"])
    model = BlockLM(cfg)
    model.load_state_dict(ck["model"])
    return model
