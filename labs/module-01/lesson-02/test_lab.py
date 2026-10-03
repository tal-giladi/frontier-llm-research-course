import json

import pytest

from frontierlab.calc import from_course, from_hf, kv_cache_elements, param_counts, snapshot_names
from frontierlab.calc.hfconfig import SNAPSHOTS
from frontierlab.labkit import load_target
from frontierlab.model import baseline0

lab = load_target(__file__)
INDEX = json.loads((SNAPSHOTS / "index.json").read_text())["models"]


def test_attention_by_hand():
    # Baseline-0 (lesson 01.1 worked example): 1,179,648 + 393,216 + 128 = 1,572,992
    assert lab.attention_params(from_course(baseline0())) == 1_572_992
    # DeepSeek-V3 MLA block, worked in the lesson: 187,107,328
    assert lab.attention_params(from_hf("deepseek-ai/DeepSeek-V3")) == 187_107_328


@pytest.mark.parametrize("name", snapshot_names())
def test_totals_rebuild_transformers_counts(name):
    """Your attention count + the shared FFN/embedding arithmetic must give Transformers' exact total."""
    s = from_hf(name)
    pc = param_counts(s)
    full = sum(1 for k in s.layer_kinds if k != "linear")
    attn_ref = pc["attention"]
    from frontierlab.calc.arith import attention_params as ref
    linear = sum(ref(s, "linear") for k in s.layer_kinds if k == "linear")
    assert full * lab.attention_params(s) + linear == attn_ref
    assert pc["total"] == INDEX[name]["hf_param_count"]


@pytest.mark.parametrize("name", snapshot_names())
def test_kv_elements(name):
    s = from_hf(name)
    for S in (1, 100, 5000, 131072):
        ref = kv_cache_elements(s, S)
        assert lab.kv_elements(s, S) == ref["full"] + ref["sliding"], (name, S)


@pytest.mark.parametrize("name", snapshot_names())
def test_active(name):
    s = from_hf(name)
    assert lab.active_params(s) == param_counts(s)["active"]


def test_active_matches_gpt_oss_card():
    assert round(lab.active_params(from_hf("openai/gpt-oss-120b")) / 1e9, 2) == 5.13
