"""Configuration of the course's reference model, Baseline-0 (dense Qwen3 layout).

Field names follow Hugging Face ``Qwen3Config`` so a checkpoint maps onto ``Qwen3ForCausalLM`` by
name. ``attention`` selects the attention module from :data:`frontierlab.attention.ATTENTION`;
Stage B modules register new kinds (``"mla"``, ``"sliding"``, ...) instead of editing the model.
``extra`` holds settings that only one attention kind (or block variant) reads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    head_dim: int = 64
    intermediate_size: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 2048
    tie_word_embeddings: bool = True
    qk_norm: bool = True
    initializer_range: float = 0.02
    attention: str = "gqa"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **kw) -> "ModelConfig":
        """A copy with some fields changed (``cfg.with_(attention="mla")``)."""
        return replace(self, **kw)


# Presets. Parameter counts are printed by ``python -m frontierlab.model.config``.
# The pilot ladder of plan section 12.1 is named by NON-embedding parameters (the N of scaling laws):
# pilot-10m ~9.4M, pilot-30m ~31.5M, pilot-70m ~70.8M, baseline0 ~96.8M (at vocab 32768).

def toy(vocab_size: int = 8192) -> ModelConfig:
    """CPU-sized model for correctness tests and the free CPU variant (~1.8M parameters at vocab 8192)."""
    return ModelConfig(vocab_size=vocab_size, hidden_size=128, num_hidden_layers=4, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=32, intermediate_size=384, max_position_embeddings=512)


def pilot_10m(vocab_size: int = 32768) -> ModelConfig:
    return ModelConfig(vocab_size=vocab_size, hidden_size=384, num_hidden_layers=6, num_attention_heads=6,
                       num_key_value_heads=2, intermediate_size=1024)


def pilot_30m(vocab_size: int = 32768) -> ModelConfig:
    return ModelConfig(vocab_size=vocab_size, hidden_size=512, num_hidden_layers=10, num_attention_heads=8,
                       num_key_value_heads=4, intermediate_size=1536)


def pilot_70m(vocab_size: int = 32768) -> ModelConfig:
    return ModelConfig(vocab_size=vocab_size, hidden_size=640, num_hidden_layers=16, num_attention_heads=10,
                       num_key_value_heads=2, intermediate_size=1792)


def baseline0(vocab_size: int = 32768) -> ModelConfig:
    """Baseline-0, the main-path reference model (~97M non-embedding, ~122M total with tied embeddings)."""
    return ModelConfig(vocab_size=vocab_size, intermediate_size=2816)


PRESETS = {"toy": toy, "pilot-10m": pilot_10m, "pilot-30m": pilot_30m, "pilot-70m": pilot_70m,
           "baseline0": baseline0}


def param_counts(cfg: ModelConfig) -> dict:
    """Parameter counts of the dense GQA model, split the way scaling-law papers split them.

    ``non_embedding`` excludes the token embedding (and the head when it is tied), which is the N
    used in Kaplan-style ``C ≈ 6·N·D`` estimates.
    """
    C, L, H, KV, hd, I = (cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads,
                          cfg.num_key_value_heads, cfg.head_dim, cfg.intermediate_size)
    attn = C * H * hd + 2 * C * KV * hd + H * hd * C + (2 * hd if cfg.qk_norm else 0)
    mlp = 3 * C * I
    norms = 2 * C
    per_layer = attn + mlp + norms
    emb = cfg.vocab_size * C
    head = 0 if cfg.tie_word_embeddings else cfg.vocab_size * C
    non_embedding = L * per_layer + C + head
    return {"total": non_embedding + emb, "non_embedding": non_embedding, "embedding": emb,
            "per_layer": per_layer, "attention_per_layer": attn, "mlp_per_layer": mlp}


if __name__ == "__main__":
    for name, fn in PRESETS.items():
        pc = param_counts(fn())
        print(f"{name:10s} total {pc['total'] / 1e6:7.2f}M   non-embedding {pc['non_embedding'] / 1e6:7.2f}M")
