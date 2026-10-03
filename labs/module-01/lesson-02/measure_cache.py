"""Check the KV-cache formula against what Hugging Face Transformers actually stores (CPU, no download).

    python labs/module-01/lesson-02/measure_cache.py            # about 30 s on a laptop

For three real configs it builds a SHRUNK model (2-6 layers, tiny vocabulary and MLP, random weights;
attention shapes, window and layer pattern unchanged), runs one forward pass over S tokens with
``use_cache=True``, and counts the elements in each layer's cache. The calculator's prediction for the
same layers is printed next to it. Attention shapes are what the cache depends on, so shrinking the rest
does not change the per-layer numbers.
"""

import json
import warnings

import torch

from frontierlab.calc import from_hf
from frontierlab.calc.hfconfig import SNAPSHOTS

warnings.filterwarnings("ignore")
S = 1500


def shrink(name, model_type, **kw):
    from transformers import AutoConfig
    d = json.loads((SNAPSHOTS / (name.replace("/", "__") + ".json")).read_text())
    d.pop("quantization_config", None)
    d.pop("auto_map", None)
    if "text_config" in d:
        d, model_type = d["text_config"], "gemma3_text"
    d.pop("model_type", None)
    d.update(kw, vocab_size=1000, pad_token_id=None, bos_token_id=None, eos_token_id=None)
    return AutoConfig.for_model(model_type, **d)


CASES = {
    "openai/gpt-oss-20b": ("gpt_oss", dict(num_hidden_layers=2, num_local_experts=4,
                                           layer_types=["sliding_attention", "full_attention"])),
    "Qwen/Qwen3-8B": ("qwen3", dict(num_hidden_layers=2, intermediate_size=256)),
    "unsloth/gemma-3-27b-pt": ("gemma3", dict(num_hidden_layers=6, intermediate_size=256)),
}


def main():
    from transformers import AutoModelForCausalLM
    import transformers
    print(f"transformers {transformers.__version__}, S = {S} tokens")
    for name, (mt, kw) in CASES.items():
        spec = from_hf(name)
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(shrink(name, mt, **kw)).eval()
        with torch.no_grad():
            cache = model(torch.randint(0, 1000, (1, S)), use_cache=True).past_key_values
        kinds = spec.layer_kinds if mt != "gpt_oss" else ["sliding", "full"]
        for i, layer in enumerate(cache.layers):
            kind = kinds[i] if mt != "gemma3" else ("full" if (i + 1) % 6 == 0 else "sliding")
            stored = layer.keys.shape[2]
            measured = layer.keys.numel() + layer.values.numel()
            keep = S if kind == "full" else min(S, spec.sliding_window)
            predicted = keep * spec.num_kv_heads * (spec.head_dim + spec.v_head_dim)
            print(f"{name:24s} layer {i} {kind:7s} tokens stored {stored:5d}  elements {measured:9,d}"
                  f"  calculator {predicted:9,d}  ({type(layer).__name__})")


if __name__ == "__main__":
    main()
