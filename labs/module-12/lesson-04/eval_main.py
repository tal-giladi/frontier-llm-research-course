"""Lab 12.4, main path: Eval Suite v2 for a Hugging Face checkpoint against the Stage D base model.

    python labs/module-12/lesson-04/eval_main.py --model runs/m12/l124-main/rl-s0/policy \\
        --base Qwen/Qwen3-1.7B-Base --out runs/m12/l124-main/eval            # 1x GPU; not run in this build
    python labs/module-12/lesson-04/eval_main.py --smoke --out runs/m12/l124-smoke   # CPU, tiny random models

Components (one score per item, the same items and seeds for both checkpoints):

* ``gsm8k_pass1`` (task) — 4 samples per question at temperature 1.0 on the first 500 GSM8K test questions,
  strict ``#### n`` reward; ``gsm8k_greedy`` (task).
* ``lambada_logprob`` (retention) — per-passage log-probability of the LAMBADA target word (Eval v0's
  pinned file), a general language-modelling skill RL on GSM8K does not train.
* ``ifeval_*`` (instruction) — the IFEval subset of :mod:`frontierlab.evals.suite_v2.ifeval`, greedy,
  prompt formatted as ``User: ...\\nAssistant:``; prompt- and instruction-level, strict and loose.

For a broader retention check also run lm-evaluation-harness 0.4.13 on both checkpoints, e.g.
``lm_eval --model hf --model_args pretrained=<path> --tasks arc_easy,hellaswag,ifeval --batch_size 16``,
and compare per-item with ``--log_samples``.

PROJECTED cost (pending the Module 12 pilot): GSM8K 2,500 generations x ~300 tokens + IFEval 215 x ~400
tokens + LAMBADA 5,153 forward passes, about 0.3-0.5 GPU-hours per checkpoint on an H100 with transformers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from frontierlab.evals.suite_v2 import core, ifeval
from frontierlab.posttrain import gsm8k
from frontierlab.posttrain.hf import left_pad


def load(path_or_repo: str, smoke: bool, revision: str | None = None):
    if smoke:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from frontierlab.posttrain.hf import SmokeTokenizer
        torch.manual_seed(0 if path_or_repo == "base" else 1)
        cfg = Qwen3Config(vocab_size=96, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=4096)
        return Qwen3ForCausalLM(cfg).eval(), SmokeTokenizer()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path_or_repo, revision=revision)
    tok.pad_token = tok.pad_token or tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(path_or_repo, revision=revision, dtype=torch.bfloat16).to(dev).eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, prompts, max_new, temperature, seed, batch=32):
    torch.manual_seed(seed)
    out = []
    for i in range(0, len(prompts), batch):
        ids, att = left_pad([tok.encode(p) for p in prompts[i:i + batch]], tok.pad_token_id)
        ids, att = ids.to(model.device), att.to(model.device)
        kw = dict(do_sample=True, temperature=temperature, top_k=0, top_p=1.0) if temperature > 0 else dict(do_sample=False)
        seq = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new, pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id, **kw)
        out += [tok.decode(s[ids.shape[1]:].tolist(), skip_special_tokens=True) for s in seq]
    return out


@torch.no_grad()
def lambada_logprobs(model, tok, texts):
    from frontierlab.evals.suite_v0 import split_last_word
    out = []
    for t in texts:
        ctx, tgt = split_last_word(t)
        c, g = tok.encode(ctx), tok.encode(tgt)
        x = torch.tensor([c + g], device=model.device)
        lp = torch.log_softmax(model(x).logits[0, len(c) - 1:-1].float(), -1)
        out.append(float(lp.gather(-1, torch.tensor(g, device=model.device)[:, None]).sum()))
    return out


def suite(model, tok, a) -> dict:
    if a.smoke:
        train = [{"question": "1+1", "solution": "#### 2", "answer": "2"}] * 4
        test = [{"question": f"{i}+1", "solution": f"#### {i + 1}", "answer": str(i + 1)} for i in range(6)]
        items = [{"prompt": "say hi", "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}] * 4
        lam = ["the cat sat on the mat", "a b c d"]
    else:
        train, test = gsm8k.load("train"), gsm8k.load("test")[:a.n_gsm8k]
        items = [it for it in ifeval.load(ifeval.download(a.ifeval_path)) if ifeval.supported(it)]
        from frontierlab.evals.suite_v0 import download_lambada, load_lambada
        lam = load_lambada(download_lambada(), a.n_lambada)
    shots = train[:4]
    prompts = [gsm8k.few_shot_prompt(q["question"], shots) for q in test]
    k = a.samples
    samp = generate(model, tok, [p for p in prompts for _ in range(k)], a.max_new, 1.0, seed=1234)
    greedy = generate(model, tok, prompts, a.max_new, 0.0, seed=0)
    pass1 = [sum(gsm8k.strict_reward(samp[i * k + j], q["answer"]) for j in range(k)) / k for i, q in enumerate(test)]
    comps = {"gsm8k_pass1": {"kind": "task", "items": pass1},
             "gsm8k_greedy": {"kind": "task", "items": [gsm8k.strict_reward(g, q["answer"]) for g, q in zip(greedy, test)]},
             "lambada_logprob": {"kind": "retention", "items": lambada_logprobs(model, tok, lam)}}
    resp = generate(model, tok, [f"User: {it['prompt']}\nAssistant:" for it in items], a.max_new, 0.0, seed=0)
    resp = [r.split("\nUser:")[0].strip() for r in resp]
    comps.update({k_: v for k_, v in ifeval.components(items, resp).items()})
    pins = {"gsm8k": gsm8k.REVISION, "n_gsm8k": len(test), "samples": k, "ifeval": ifeval.IFEVAL_REVISION,
            "n_lambada": len(lam), "max_new": a.max_new}
    return {"version": core.VERSION, "pins": pins, "components": comps}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="new")
    ap.add_argument("--base", default="base")
    ap.add_argument("--base-revision", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-gsm8k", type=int, default=500)
    ap.add_argument("--n-lambada", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--ifeval-path", default="labs/common/data/m12/ifeval_input_data.jsonl")
    ap.add_argument("--lambada-guard", type=float, default=0.05)
    a = ap.parse_args(argv)
    if a.smoke:
        a.samples, a.max_new = 2, 6
    a.out.mkdir(parents=True, exist_ok=True)
    rev = a.base_revision
    if not a.smoke and a.base == "Qwen/Qwen3-1.7B-Base" and rev is None:
        from frontierlab.posttrain import STAGE_D_BASE
        rev = STAGE_D_BASE["revision"]
    res = {}
    for name, path, r in (("base", a.base, rev), ("new", a.model, None)):
        f = a.out / f"{name}.json"
        if not f.exists():
            m, t = load(path, a.smoke, r)
            f.write_text(json.dumps(suite(m, t, a)))
        res[name] = json.loads(f.read_text())
    cmp = core.compare(res["base"], res["new"], guards={"lambada_logprob": a.lambada_guard})
    print(core.report(cmp))
    (a.out / "compare.json").write_text(json.dumps(cmp, indent=1, default=str))


if __name__ == "__main__":
    main()
