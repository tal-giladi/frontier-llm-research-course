"""Read a Hugging Face ``config.json`` into one normalised architecture description (lesson 01.2).

Every open model family names the same idea differently: DeepSeek says ``n_routed_experts``, Qwen
says ``num_experts``, gpt-oss says ``num_local_experts``; Gemma 3 hides its text model under
``text_config``; gpt-oss lists ``layer_types`` while Gemma 3 gives a ``sliding_window_pattern``.
:func:`from_hf` turns each supported family into an :class:`ArchSpec` and records, for every field,
which config key it came from (``spec.source``), so a number in a comparison table can always be
traced back to the file.

Supported ``model_type`` values: qwen3, qwen3_moe, qwen3_next, llama, olmo2, gemma3 / gemma3_text,
gpt_oss, deepseek_v3, kimi_k2, glm4_moe, minimax_m2. Anything else raises ``KeyError`` — add a reader
rather than guessing.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

SNAPSHOTS = Path(__file__).resolve().parent / "snapshots"


@dataclass
class ArchSpec:
    """Everything the calculator needs, in one vocabulary. Lengths are in elements, not bytes."""

    name: str
    model_type: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    layer_kinds: list[str]            # per layer: "full" | "sliding" | "linear"
    attention: str = "gqa"            # "gqa" (includes MHA when kv == heads) | "mla"
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0                 # query/key width per head (GQA)
    v_head_dim: int = 0               # value width per head
    q_gate: bool = False              # output gate fused into q_proj (Qwen3-Next gated attention)
    qkv_bias: bool = False
    o_bias: bool = False
    qk_norm: str = "none"             # "none" | "per_head" (RMSNorm over head_dim) | "full" (over all heads)
    sinks: bool = False               # one learned sink logit per head (gpt-oss)
    sliding_window: int | None = None
    # MLA (DeepSeek-V2/V3, Kimi K2)
    q_lora_rank: int | None = None
    kv_lora_rank: int = 0
    qk_nope_dim: int = 0
    qk_rope_dim: int = 0
    # linear attention (Gated DeltaNet in Qwen3-Next)
    lin_k_heads: int = 0
    lin_v_heads: int = 0
    lin_k_dim: int = 0
    lin_v_dim: int = 0
    lin_conv: int = 0
    # feed-forward
    dense_intermediate: int = 0
    dense_layers: list[int] = field(default_factory=list)   # layer indices with a dense MLP
    n_experts: int = 0
    top_k: int = 0
    expert_intermediate: int = 0
    n_shared: int = 0
    shared_intermediate: int = 0
    shared_gate: bool = False         # sigmoid gate on the shared expert (Qwen3-Next)
    router_bias: bool = False
    expert_bias: bool = False         # biases inside experts (gpt-oss)
    # norms, embeddings, misc
    norms_per_layer: int = 2          # RMSNorm vectors of width hidden_size per layer
    tie_embeddings: bool = False
    max_context: int = 0
    rope_theta: float = 0.0
    rope_scaling: str | None = None
    mtp_layers: int = 0               # multi-token-prediction modules (not counted in params)
    logit_controls: list[str] = field(default_factory=list)
    source: dict = field(default_factory=dict)   # field -> config key it was read from

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 0

    def moe_layers(self) -> list[int]:
        return [i for i in range(self.num_layers) if self.is_moe and i not in self.dense_layers]


def load_config(src) -> dict:
    """A config dict from a dict, a path, a snapshot name (``"Qwen/Qwen3-8B"``) or a JSON string."""
    if isinstance(src, dict):
        return src
    snap = SNAPSHOTS / (str(src).replace("/", "__") + ".json")
    if snap.exists():
        return json.loads(snap.read_text())
    p = Path(str(src))
    if p.exists():
        return json.loads(p.read_text())
    raise FileNotFoundError(f"{src}: not a file or a snapshot; use fetch_config() for the Hub")


def fetch_config(repo: str, revision: str = "main", timeout: float = 20.0) -> dict:
    """Download ``config.json`` from the Hugging Face Hub (gated repos need a token; use a snapshot)."""
    url = f"https://huggingface.co/{repo}/raw/{revision}/config.json"
    with urllib.request.urlopen(url, timeout=timeout) as r:   # noqa: S310 - fixed https host
        return json.loads(r.read().decode("utf-8"))


def snapshot_names() -> list[str]:
    return sorted(json.loads((SNAPSHOTS / "index.json").read_text())["models"])


def _get(d: dict, src: dict, field_name: str, *keys, default=None):
    """First present key among ``keys``; remember which one in ``src``."""
    for k in keys:
        if k in d and d[k] is not None:
            src[field_name] = k
            return d[k]
    if default is not None:
        src[field_name] = f"default ({default})"
    return default


def _common(d: dict, name: str, src: dict) -> dict:
    rs = d.get("rope_scaling") or {}
    return dict(
        name=name, model_type=d["model_type"],
        vocab_size=_get(d, src, "vocab_size", "vocab_size"),
        hidden_size=_get(d, src, "hidden_size", "hidden_size"),
        num_layers=_get(d, src, "num_layers", "num_hidden_layers"),
        num_heads=_get(d, src, "num_heads", "num_attention_heads"),
        num_kv_heads=_get(d, src, "num_kv_heads", "num_key_value_heads", "num_attention_heads"),
        max_context=_get(d, src, "max_context", "max_position_embeddings", default=0),
        rope_theta=float(_get(d, src, "rope_theta", "rope_theta", default=0.0)),
        rope_scaling=rs.get("rope_type") or rs.get("type"),
        tie_embeddings=bool(_get(d, src, "tie_embeddings", "tie_word_embeddings", default=False)),
    )


def _gqa_head_dim(d: dict, c: dict, src: dict) -> int:
    return _get(d, src, "head_dim", "head_dim") or c["hidden_size"] // c["num_heads"]


def _read_qwen3(d, name, src):
    c = _common(d, name, src)
    hd = _gqa_head_dim(d, c, src)
    s = ArchSpec(**c, layer_kinds=["full"] * c["num_layers"], head_dim=hd, v_head_dim=hd,
                 qkv_bias=bool(d.get("attention_bias", False)), qk_norm="per_head")
    src["qk_norm"] = "model_type (Qwen3Attention has q_norm/k_norm over head_dim)"
    if d["model_type"] == "qwen3":
        s.dense_intermediate = _get(d, src, "dense_intermediate", "intermediate_size")
        s.dense_layers = list(range(s.num_layers))
    else:
        s.n_experts = _get(d, src, "n_experts", "num_experts")
        s.top_k = _get(d, src, "top_k", "num_experts_per_tok")
        s.expert_intermediate = _get(d, src, "expert_intermediate", "moe_intermediate_size")
        s.dense_layers = list(d.get("mlp_only_layers") or [])
        s.dense_intermediate = d.get("intermediate_size", 0)
    return s


def _read_qwen3_next(d, name, src):
    c = _common(d, name, src)
    hd = _gqa_head_dim(d, c, src)
    every = _get(d, src, "layer_kinds", "full_attention_interval")
    kinds = ["full" if (i + 1) % every == 0 else "linear" for i in range(c["num_layers"])]
    return ArchSpec(**c, layer_kinds=kinds, head_dim=hd, v_head_dim=hd, q_gate=True, qk_norm="per_head",
                    lin_k_heads=_get(d, src, "lin_k_heads", "linear_num_key_heads"),
                    lin_v_heads=_get(d, src, "lin_v_heads", "linear_num_value_heads"),
                    lin_k_dim=_get(d, src, "lin_k_dim", "linear_key_head_dim"),
                    lin_v_dim=_get(d, src, "lin_v_dim", "linear_value_head_dim"),
                    lin_conv=_get(d, src, "lin_conv", "linear_conv_kernel_dim"),
                    n_experts=_get(d, src, "n_experts", "num_experts"),
                    top_k=_get(d, src, "top_k", "num_experts_per_tok"),
                    expert_intermediate=_get(d, src, "expert_intermediate", "moe_intermediate_size"),
                    n_shared=1, shared_gate=True,
                    shared_intermediate=_get(d, src, "shared_intermediate", "shared_expert_intermediate_size"),
                    dense_layers=list(d.get("mlp_only_layers") or []),
                    logit_controls=["partial RoPE " + str(d.get("partial_rotary_factor"))])


def _read_llama_like(d, name, src):
    """llama and olmo2: dense, GQA/MHA. OLMo 2 adds QK-norm over all heads and post-sublayer norms."""
    c = _common(d, name, src)
    hd = _gqa_head_dim(d, c, src)
    s = ArchSpec(**c, layer_kinds=["full"] * c["num_layers"], head_dim=hd, v_head_dim=hd,
                 qkv_bias=bool(d.get("attention_bias", False)), o_bias=bool(d.get("attention_bias", False)),
                 dense_intermediate=_get(d, src, "dense_intermediate", "intermediate_size"))
    s.dense_layers = list(range(s.num_layers))
    if d["model_type"] == "olmo2":
        s.qk_norm = "full"
        src["qk_norm"] = "model_type (Olmo2Attention normalises all heads together)"
        s.logit_controls = ["QK-norm", "norm after each sublayer"]
    return s


def _read_gemma3(d, name, src):
    if "text_config" in d:
        d = {**d["text_config"], "model_type": d["model_type"]}
        src["_note"] = "text_config of a multimodal config (vision tower not counted)"
    tied_default = "tie_word_embeddings" not in d
    d = {"tie_word_embeddings": True, **d}       # Gemma ties input and output embeddings by default
    c = _common(d, name, src)
    if tied_default:
        src["tie_embeddings"] = "absent; Gemma3 config class default (True)"
    src["qk_norm"] = "model_type (Gemma3Attention has q_norm/k_norm over head_dim)"
    hd = _gqa_head_dim(d, c, src)
    pattern = _get(d, src, "layer_kinds", "sliding_window_pattern")
    kinds = ["full" if (i + 1) % pattern == 0 else "sliding" for i in range(c["num_layers"])]
    ctl = ["QK-norm"]
    for k in ("attn_logit_softcapping", "final_logit_softcapping"):
        if d.get(k):
            ctl.append(f"{k}={d[k]}")
    s = ArchSpec(**c, layer_kinds=kinds, head_dim=hd, v_head_dim=hd, qk_norm="per_head",
                 sliding_window=_get(d, src, "sliding_window", "sliding_window"),
                 dense_intermediate=_get(d, src, "dense_intermediate", "intermediate_size"),
                 norms_per_layer=4, logit_controls=ctl)
    s.dense_layers = list(range(s.num_layers))
    src["norms_per_layer"] = "model_type (Gemma3 has pre- and post-norms around attention and MLP)"
    return s


def _read_gpt_oss(d, name, src):
    c = _common(d, name, src)
    hd = _gqa_head_dim(d, c, src)
    kinds = ["sliding" if t == "sliding_attention" else "full" for t in d["layer_types"]]
    src["layer_kinds"] = "layer_types"
    e = _get(d, src, "n_experts", "num_local_experts")
    return ArchSpec(**c, layer_kinds=kinds, head_dim=hd, v_head_dim=hd, qkv_bias=True, o_bias=True, sinks=True,
                    sliding_window=_get(d, src, "sliding_window", "sliding_window"), n_experts=e,
                    top_k=_get(d, src, "top_k", "num_experts_per_tok", "experts_per_token"),
                    expert_intermediate=_get(d, src, "expert_intermediate", "intermediate_size"),
                    router_bias=True, expert_bias=True,
                    logit_controls=["attention sinks", f"swiglu_limit={d.get('swiglu_limit')}"])


def _read_deepseek(d, name, src):
    """deepseek_v3 and kimi_k2: MLA attention, DeepSeekMoE with shared experts, first k layers dense."""
    c = _common(d, name, src)
    first_dense = _get(d, src, "dense_layers", "first_k_dense_replace", default=0)
    s = ArchSpec(**c, layer_kinds=["full"] * c["num_layers"], attention="mla",
                 q_lora_rank=_get(d, src, "q_lora_rank", "q_lora_rank"),
                 kv_lora_rank=_get(d, src, "kv_lora_rank", "kv_lora_rank"),
                 qk_nope_dim=_get(d, src, "qk_nope_dim", "qk_nope_head_dim"),
                 qk_rope_dim=_get(d, src, "qk_rope_dim", "qk_rope_head_dim"),
                 v_head_dim=_get(d, src, "v_head_dim", "v_head_dim"),
                 dense_intermediate=_get(d, src, "dense_intermediate", "intermediate_size"),
                 dense_layers=list(range(first_dense)),
                 n_experts=_get(d, src, "n_experts", "n_routed_experts"),
                 top_k=_get(d, src, "top_k", "num_experts_per_tok"),
                 expert_intermediate=_get(d, src, "expert_intermediate", "moe_intermediate_size"),
                 n_shared=_get(d, src, "n_shared", "n_shared_experts", default=0),
                 mtp_layers=_get(d, src, "mtp_layers", "num_nextn_predict_layers", default=0))
    s.head_dim = s.qk_nope_dim + s.qk_rope_dim
    s.shared_intermediate = s.expert_intermediate * s.n_shared
    src["head_dim"] = "qk_nope_head_dim + qk_rope_head_dim"
    src["attention"] = "model_type (DeepSeek-V3 modelling code: MLA)"
    return s


def _read_glm4_moe(d, name, src):
    c = _common(d, name, src)
    hd = _gqa_head_dim(d, c, src)
    n_shared = _get(d, src, "n_shared", "n_shared_experts", default=0)
    s = ArchSpec(**c, layer_kinds=["full"] * c["num_layers"], head_dim=hd, v_head_dim=hd,
                 qkv_bias=bool(d.get("attention_bias")), qk_norm="per_head" if d.get("use_qk_norm") else "none",
                 dense_intermediate=_get(d, src, "dense_intermediate", "intermediate_size"),
                 dense_layers=list(range(d.get("first_k_dense_replace", 0))),
                 n_experts=_get(d, src, "n_experts", "n_routed_experts"),
                 top_k=_get(d, src, "top_k", "num_experts_per_tok"),
                 expert_intermediate=_get(d, src, "expert_intermediate", "moe_intermediate_size"),
                 n_shared=n_shared, mtp_layers=d.get("num_nextn_predict_layers", 0),
                 logit_controls=["QK-norm" if d.get("use_qk_norm") else "no QK-norm",
                                 f"partial RoPE {d.get('partial_rotary_factor')}"])
    s.shared_intermediate = s.expert_intermediate * n_shared
    src["dense_layers"] = "first_k_dense_replace"
    return s


def _read_minimax_m2(d, name, src):
    c = _common(d, name, src)
    hd = _gqa_head_dim(d, c, src)
    kinds = ["full" if t == 1 else "linear" for t in d["attn_type_list"]]
    src["layer_kinds"] = "attn_type_list (1 = full attention)"
    return ArchSpec(**c, layer_kinds=kinds, head_dim=hd, v_head_dim=hd,
                    qk_norm="full" if d.get("use_qk_norm") else "none",
                    n_experts=_get(d, src, "n_experts", "num_local_experts"),
                    top_k=_get(d, src, "top_k", "num_experts_per_tok"),
                    expert_intermediate=_get(d, src, "expert_intermediate", "intermediate_size"),
                    mtp_layers=d.get("num_mtp_modules", 0) if d.get("use_mtp") else 0,
                    logit_controls=[f"QK-norm ({d.get('qk_norm_type')})", f"rotary_dim={d.get('rotary_dim')}"])


READERS = {"qwen3": _read_qwen3, "qwen3_moe": _read_qwen3, "qwen3_next": _read_qwen3_next,
           "llama": _read_llama_like, "olmo2": _read_llama_like, "gemma3": _read_gemma3,
           "gemma3_text": _read_gemma3, "gpt_oss": _read_gpt_oss, "deepseek_v3": _read_deepseek,
           "kimi_k2": _read_deepseek, "glm4_moe": _read_glm4_moe, "minimax_m2": _read_minimax_m2}


def from_hf(src, name: str | None = None) -> ArchSpec:
    """Normalise a Hugging Face config (dict, path or snapshot name) into an :class:`ArchSpec`."""
    d = load_config(src)
    mt = d.get("model_type")
    if mt not in READERS:
        raise KeyError(f"no reader for model_type {mt!r}; supported: {sorted(READERS)}")
    provenance: dict = {}
    spec = READERS[mt](d, name or (src if isinstance(src, str) else mt), provenance)
    provenance.setdefault("layer_kinds", "no per-layer field: every layer is full attention")
    provenance.setdefault("norms_per_layer", "model_type (input and post-attention RMSNorm)")
    spec.source = provenance
    return spec


def from_course(cfg) -> ArchSpec:
    """The course's own ``frontierlab.model.ModelConfig`` (Baseline-0 and its presets) as an ArchSpec."""
    s = ArchSpec(name="frontierlab", model_type="frontierlab", vocab_size=cfg.vocab_size,
                 hidden_size=cfg.hidden_size, num_layers=cfg.num_hidden_layers,
                 layer_kinds=["full"] * cfg.num_hidden_layers, num_heads=cfg.num_attention_heads,
                 num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim, v_head_dim=cfg.head_dim,
                 qk_norm="per_head" if cfg.qk_norm else "none", dense_intermediate=cfg.intermediate_size,
                 tie_embeddings=cfg.tie_word_embeddings, max_context=cfg.max_position_embeddings,
                 rope_theta=cfg.rope_theta)
    s.dense_layers = list(range(s.num_layers))
    return s
