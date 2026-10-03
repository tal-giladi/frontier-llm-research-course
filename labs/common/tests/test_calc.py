import json

import pytest

from frontierlab import flops
from frontierlab.calc import (decode_flops_per_token, flops_per_token, from_course, from_hf, kv_bytes,
                              kv_bytes_per_token, kv_cache_elements, param_counts, snapshot_names)
from frontierlab.calc.hfconfig import SNAPSHOTS
from frontierlab.model import baseline0

INDEX = json.loads((SNAPSHOTS / "index.json").read_text())["models"]


@pytest.mark.parametrize("name", sorted(INDEX))
def test_total_params_match_transformers_exactly(name):
    """Reference: AutoModelForCausalLM.from_config on the meta device, transformers 5.18.0 (see index.json)."""
    assert param_counts(from_hf(name))["total"] == INDEX[name]["hf_param_count"]


def test_active_params_match_published_numbers():
    # gpt-oss model card Table 1: active counts the unembedding but not the embedding.
    assert round(param_counts(from_hf("openai/gpt-oss-120b"))["active"] / 1e9, 2) == 5.13
    assert round(param_counts(from_hf("openai/gpt-oss-20b"))["active"] / 1e9, 2) == 3.61
    # DeepSeek-V3 section 4.2: "37B activated". The report does not say whether the embedding is
    # included: 36.6B without it, 37.6B with it; both are consistent with "37B".
    ds = param_counts(from_hf("deepseek-ai/DeepSeek-V3"))
    assert 36.5e9 < ds["active"] < ds["active_with_embedding"] < 37.6e9
    # Qwen3-30B-A3B model card: "30.5B in total and 3.3B activated", "Non-Embedding: 29.9B".
    # Qwen's non-embedding count leaves out both the embedding and the (untied) head.
    q = param_counts(from_hf("Qwen/Qwen3-30B-A3B"))
    assert round(q["total"] / 1e9, 1) == 30.5 and round((q["total"] - q["embedding"] - q["head"]) / 1e9, 1) == 29.9
    assert 3.3e9 <= q["active_with_embedding"] < 3.4e9
    assert round(param_counts(from_hf("Qwen/Qwen3-235B-A22B"))["active_with_embedding"] / 1e9) == 22


def test_agrees_with_course_model_arithmetic():
    cfg = baseline0()
    s = from_course(cfg)
    assert kv_bytes_per_token(s, 1024) == flops.kv_bytes_per_token(cfg) == 12288
    # the calculator leaves norm gains out of the matmul term; frontierlab.flops keeps them in N
    assert flops_per_token(s, 1024) == pytest.approx(flops.flops_per_token(cfg, 1024), rel=1e-3)


def test_mla_caches_the_latent_not_the_heads():
    s = from_hf("deepseek-ai/DeepSeek-V3")
    assert kv_cache_elements(s, 1)["total"] == 61 * (512 + 64)
    assert s.source["n_experts"] == "n_routed_experts" and s.source["kv_lora_rank"] == "kv_lora_rank"


def test_sliding_layers_stop_growing_at_the_window():
    s = from_hf("openai/gpt-oss-120b")
    per_layer = 8 * (64 + 64)
    assert kv_cache_elements(s, 128)["sliding"] == 18 * 128 * per_layer
    assert kv_cache_elements(s, 100_000)["sliding"] == 18 * 128 * per_layer      # capped
    assert kv_cache_elements(s, 100_000)["full"] == 18 * 100_000 * per_layer
    g = from_hf("unsloth/gemma-3-27b-pt")
    assert g.layer_kinds.count("full") == 10 and g.layer_kinds.count("sliding") == 52
    assert kv_bytes_per_token(g, 131072) < kv_bytes_per_token(g, 1024)


def test_linear_layers_have_constant_state():
    s = from_hf("Qwen/Qwen3-Next-80B-A3B-Instruct")
    assert s.layer_kinds.count("linear") == 36 and s.layer_kinds[3] == "full"
    a, b = kv_cache_elements(s, 1000), kv_cache_elements(s, 2000)
    assert a["linear"] == b["linear"] and b["full"] == 2 * a["full"]


def test_decode_cost_grows_with_context_only_in_full_layers():
    s = from_hf("Qwen/Qwen3-8B")
    d1, d2 = decode_flops_per_token(s, 1000), decode_flops_per_token(s, 2000)
    assert d2 - d1 == pytest.approx(36 * 2 * 32 * (128 + 128) * 1000)
    assert kv_bytes(s, 1000, bytes_per_elem=1) * 2 == kv_bytes(s, 1000)


def test_every_snapshot_reads():
    for n in snapshot_names():
        s = from_hf(n)
        assert len(s.layer_kinds) == s.num_layers and param_counts(s)["active"] <= param_counts(s)["total"]


def test_unknown_family_is_refused():
    with pytest.raises(KeyError):
        from_hf({"model_type": "mystery", "hidden_size": 8})
