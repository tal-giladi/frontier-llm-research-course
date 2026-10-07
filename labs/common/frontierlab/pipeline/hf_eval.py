"""Eval Suite v2 for Hugging Face checkpoints with pinned revisions and an optional chat format (Module 13).

    python -m frontierlab.pipeline.hf_eval score --model allenai/OLMo-2-0425-1B-SFT \\
        --revision 0d85a3d037876ce6ac7d4311d994400fc66ac27f --chat --out runs/m13/l131-main/sft-chat.json
    python -m frontierlab.pipeline.hf_eval compare runs/m13/l131-main/sft-chat.json runs/m13/l131-main/dpo-chat.json
    python -m frontierlab.pipeline.hf_eval score --smoke --out runs/m13/hf-eval-smoke/a.json      # CPU, tiny random model

The components are Module 12's main-path suite (``labs/module-12/lesson-04/eval_main.py``), rebuilt here so
that Module 13's lab download is self-contained and every checkpoint can be loaded at a pinned revision:

* ``gsm8k_pass1`` / ``gsm8k_greedy`` (task): 4-shot GSM8K (training-split shots), strict ``#### n``;
* ``lambada_logprob`` (retention): log-probability of the LAMBADA target word (Eval v0's pinned file);
* ``ifeval_*`` (instruction): the 12-type IFEval subset of :mod:`frontierlab.evals.suite_v2.ifeval`.

``--chat`` wraps every prompt in the tokenizer's chat template (``apply_chat_template`` with
``add_generation_prompt=True``; ``--no-think`` passes ``enable_thinking=False`` for Qwen3 hybrid models).
Without it the prompts are plain text (``User: ...\\nAssistant:``) as in Module 12. The format is a pin:
:func:`frontierlab.evals.suite_v2.core.compare` refuses to pair results produced with different formats,
so a base model (plain) and a chat model (chat) are compared only in the same format.

Not run in this build except ``--smoke``; part of the Module 13 pilot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from frontierlab.evals.suite_v2 import core, ifeval
from frontierlab.posttrain import gsm8k
from frontierlab.posttrain.hf import left_pad


def load_hf(path_or_repo: str, revision: str | None = None, smoke: bool = False, dtype=torch.bfloat16, seed: int = 0):
    if smoke:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from frontierlab.posttrain.hf import SmokeTokenizer
        torch.manual_seed(seed)
        cfg = Qwen3Config(vocab_size=96, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=4096)
        return Qwen3ForCausalLM(cfg).eval(), SmokeTokenizer()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path_or_repo, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(path_or_repo, revision=revision, dtype=dtype).to(dev).eval()
    return model, tok


def render(tok, text: str, chat: bool, enable_thinking: bool | None = None) -> str:
    """One user turn, ready for generation."""
    if not chat or not hasattr(tok, "apply_chat_template"):
        return f"User: {text}\nAssistant:"
    kw = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, **kw)


def encode(tok, text: str) -> list[int]:
    """Token ids without adding a second BOS when the chat template already wrote one."""
    try:
        return tok.encode(text, add_special_tokens=False)
    except TypeError:                                 # the smoke tokenizer
        return tok.encode(text)


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new: int, temperature: float, seed: int, batch: int = 32,
             top_p: float = 1.0, top_k: int = 0) -> list[str]:
    torch.manual_seed(seed)
    out = []
    dev = next(model.parameters()).device
    for i in range(0, len(prompts), batch):
        ids, att = left_pad([encode(tok, p) for p in prompts[i:i + batch]], tok.pad_token_id)
        ids, att = ids.to(dev), att.to(dev)
        kw = dict(do_sample=True, temperature=temperature, top_k=top_k, top_p=top_p) if temperature > 0 else dict(do_sample=False)
        seq = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new, pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id, **kw)
        out += [tok.decode(s[ids.shape[1]:].tolist(), skip_special_tokens=True) for s in seq]
    return out


@torch.no_grad()
def lambada_logprobs(model, tok, texts: list[str]) -> list[float]:
    from frontierlab.evals.suite_v0 import split_last_word
    dev = next(model.parameters()).device
    out = []
    for t in texts:
        ctx, tgt = split_last_word(t)
        c, g = encode(tok, ctx), encode(tok, tgt)
        x = torch.tensor([c + g], device=dev)
        lp = torch.log_softmax(model(x).logits[0, len(c) - 1:-1].float(), -1)
        out.append(float(lp.gather(-1, torch.tensor(g, device=dev)[:, None]).sum()))
    return out


def suite(model, tok, *, chat: bool = False, enable_thinking: bool | None = None, n_gsm8k: int = 500,
          n_lambada: int = 1000, samples: int = 4, max_new: int = 384,
          ifeval_path: str = "labs/common/data/m12/ifeval_input_data.jsonl", smoke: bool = False) -> dict:
    if smoke:
        train = [{"question": "1+1", "solution": "#### 2", "answer": "2"}] * 4
        test = [{"question": f"{i}+1", "solution": f"#### {i + 1}", "answer": str(i + 1)} for i in range(6)]
        items = [{"prompt": "say hi", "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}] * 4
        lam = ["the cat sat on the mat", "a b c d"]
        samples, max_new = 2, 6
    else:
        train, test = gsm8k.load("train"), gsm8k.load("test")[:n_gsm8k]
        items = [it for it in ifeval.load(ifeval.download(ifeval_path)) if ifeval.supported(it)]
        from frontierlab.evals.suite_v0 import download_lambada, load_lambada
        lam = load_lambada(download_lambada(), n_lambada)
    shots = train[:4]
    gsm_text = [gsm8k.few_shot_prompt(q["question"], shots) for q in test]
    if chat:
        gsm_text = [t + "\n(Answer the last question in the same format, ending with #### and the number.)" for t in gsm_text]
    prompts = [render(tok, t, chat, enable_thinking) for t in gsm_text]
    k = samples
    samp = generate(model, tok, [p for p in prompts for _ in range(k)], max_new, 1.0, seed=1234)
    greedy = generate(model, tok, prompts, max_new, 0.0, seed=0)
    pass1 = [sum(gsm8k.strict_reward(samp[i * k + j], q["answer"]) for j in range(k)) / k for i, q in enumerate(test)]
    comps = {"gsm8k_pass1": {"kind": "task", "items": pass1},
             "gsm8k_greedy": {"kind": "task", "items": [gsm8k.strict_reward(g, q["answer"]) for g, q in zip(greedy, test)]},
             "lambada_logprob": {"kind": "retention", "items": lambada_logprobs(model, tok, lam)}}
    resp = generate(model, tok, [render(tok, it["prompt"], chat, enable_thinking) for it in items], max_new, 0.0, seed=0)
    resp = [r.split("\nUser:")[0].strip() for r in resp]
    comps.update(ifeval.components(items, resp))
    pins = {"gsm8k": gsm8k.REVISION, "n_gsm8k": len(test), "samples": k, "ifeval": ifeval.IFEVAL_REVISION,
            "n_lambada": len(lam), "max_new": max_new, "format": "chat" if chat else "plain",
            "enable_thinking": enable_thinking}
    return {"version": core.VERSION, "pins": pins, "components": comps,
            "summary": {n: sum(c["items"]) / max(1, len(c["items"])) for n, c in comps.items() if not n.startswith("_")}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--model", default="smoke")
    s.add_argument("--revision", default=None)
    s.add_argument("--chat", action="store_true")
    s.add_argument("--no-think", action="store_true")
    s.add_argument("--smoke", action="store_true")
    s.add_argument("--n-gsm8k", type=int, default=500)
    s.add_argument("--n-lambada", type=int, default=1000)
    s.add_argument("--samples", type=int, default=4)
    s.add_argument("--max-new", type=int, default=384)
    s.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("new")
    c.add_argument("--guard", type=float, default=0.02)
    c.add_argument("--lambada-guard", type=float, default=0.05)
    a = ap.parse_args(argv)
    if a.cmd == "score":
        model, tok = load_hf(a.model, a.revision, a.smoke)
        res = suite(model, tok, chat=a.chat, enable_thinking=False if a.no_think else None, n_gsm8k=a.n_gsm8k,
                    n_lambada=a.n_lambada, samples=a.samples, max_new=a.max_new, smoke=a.smoke)
        res["model"] = {"name": a.model, "revision": a.revision}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res))
        print(json.dumps(res["summary"], indent=1))
    else:
        base, new = json.loads(Path(a.base).read_text()), json.loads(Path(a.new).read_text())
        print(core.report(core.compare(base, new, guards={"lambada_logprob": a.lambada_guard}, default_guard=a.guard)))


if __name__ == "__main__":
    main()
