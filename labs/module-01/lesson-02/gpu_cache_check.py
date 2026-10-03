"""Optional main-path step 6 of lab 01.2: measured KV-cache memory of a real model vs the calculator.

    python labs/module-01/lesson-02/gpu_cache_check.py --model Qwen/Qwen3-0.6B --context 32768

NOT RUN IN THIS BUILD; part of the Module 1 pilot. Needs a CUDA GPU with about 6 GB free and downloads
the model weights (about 1.5 GB for Qwen3-0.6B in BF16).

It loads the model in BF16, runs one prefill of ``--context`` random tokens with ``use_cache=True``,
and reports the bytes held by the returned cache tensors (exact) and the growth of
``torch.cuda.memory_allocated()`` across the call minus the logits (approximate: it also contains
allocator rounding). Both are printed next to ``frontierlab.calc.kv_bytes`` for the same config.
"""

import argparse

import torch

from frontierlab.calc import fetch_config, from_hf, kv_bytes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--context", type=int, default=32768)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM
    spec = from_hf(fetch_config(a.model), name=a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    ids = torch.randint(0, spec.vocab_size, (1, a.context), device="cuda")
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    with torch.no_grad():
        out = model(ids, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    grown = torch.cuda.memory_allocated() - before - out.logits.numel() * out.logits.element_size()
    held = sum(layer.keys.numel() * layer.keys.element_size() + layer.values.numel() * layer.values.element_size()
               for layer in out.past_key_values.layers)
    pred = kv_bytes(spec, a.context, bytes_per_elem=2)
    print(f"{a.model}, S = {a.context}: cache tensors {held / 2**20:.1f} MiB, allocator growth {grown / 2**20:.1f} MiB, "
          f"calculator {pred / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
