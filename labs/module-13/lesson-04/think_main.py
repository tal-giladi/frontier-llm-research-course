"""Lab 13.4, main path: Qwen3's thinking and non-thinking modes, thinking budgets and a cascade router on GSM8K.

    python labs/module-13/lesson-04/think_main.py --model Qwen/Qwen3-1.7B --n 500 \\
        --budgets 0,256,512,1024,2048,none --out runs/m13/l134-main          # 1x GPU; not run in this build
    python labs/module-13/lesson-04/think_main.py --model Qwen/Qwen3-1.7B --n 500 --route cascade --out runs/m13/l134-main
    python labs/module-13/lesson-04/think_main.py --smoke --out runs/m13/l134-smoke                      # CPU, tiny random model

* **Modes.** ``enable_thinking=False`` in ``apply_chat_template`` (the template writes an empty think block) or
  ``True`` (the model writes ``<think>...</think>`` first). Sampling as the model card recommends: thinking
  temperature 0.6, top-p 0.95, top-k 20; non-thinking 0.7, 0.8, 20 (Qwen/Qwen3-1.7B model card, checked 2026-10-07).
* **Budget forcing** (Qwen3 report section 4.3): generate at most B thinking tokens; if ``</think>`` has not
  appeared, append the report's stop-thinking text and let the model answer (up to 512 more tokens).
* **Cascade router**: answer in non-thinking mode first; if the mean log-probability of the answer tokens is
  below a threshold, answer again in thinking mode. Cost = all generated tokens, both passes counted.

Reward: the GSM8K gold number appears as the last number of the response after ``</think>`` (the model is asked
to end with ``#### <number>``; :func:`frontierlab.posttrain.gsm8k.strict_reward` is used when it does).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from frontierlab.pipeline.hf_eval import encode
from frontierlab.posttrain import gsm8k

STOP_THINKING = ("Considering the limited time by the user, I have to give the solution based on the thinking "
                 "directly now.\n</think>.\n\n")
QWEN3_1P7B = ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
SAMPLING = {True: dict(temperature=0.6, top_p=0.95, top_k=20), False: dict(temperature=0.7, top_p=0.8, top_k=20)}


def prompt(tok, question: str, thinking: bool, smoke: bool) -> str:
    text = f"{question}\nSolve it step by step and end with '#### <number>'."
    if smoke:
        return text
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=thinking)


def answer_part(text: str) -> str:
    return text.split("</think>")[-1]


def correct(text: str, gold: str) -> float:
    ans = answer_part(text)
    if gsm8k.strict_answer(ans) is not None:
        return gsm8k.strict_reward(ans, gold)
    return gsm8k.last_number_reward(ans, gold)


@torch.no_grad()
def gen(model, tok, prompts, max_new, sampling, seed, eos):
    torch.manual_seed(seed)
    from frontierlab.posttrain.hf import left_pad
    dev = next(model.parameters()).device
    ids, att = left_pad([encode(tok, p) for p in prompts], tok.pad_token_id)
    out = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), do_sample=True, max_new_tokens=max_new,
                         pad_token_id=tok.pad_token_id, eos_token_id=eos, output_scores=True,
                         return_dict_in_generate=True, **sampling)
    seqs = out.sequences[:, ids.shape[1]:]
    lp = torch.stack(out.scores, 1).float().log_softmax(-1).gather(-1, seqs[..., None]).squeeze(-1)
    texts, n_tok, mean_lp = [], [], []
    for row, l in zip(seqs.tolist(), lp):
        n = next((i + 1 for i, t in enumerate(row) if t in eos), len(row))
        texts.append(tok.decode(row[:n], skip_special_tokens=False))
        n_tok.append(n)
        mean_lp.append(float(l[:n].mean()))
    return texts, n_tok, mean_lp


def with_budget(model, tok, prompts, budget, seed, eos, smoke):
    """Thinking mode, at most ``budget`` thinking tokens, then the stop-thinking text and an answer."""
    if budget is None:
        texts, n, _ = gen(model, tok, prompts, 8192 if not smoke else 8, SAMPLING[True], seed, eos)
        return texts, n
    if budget == 0:                                  # no thinking at all: open and close the block at once
        texts, n = ["<think>\n"] * len(prompts), [0] * len(prompts)
    else:
        texts, n, _ = gen(model, tok, prompts, budget, SAMPLING[True], seed, eos)
    final, total = [], []
    for p, t, k in zip(prompts, texts, n):
        if "</think>" in t:
            final.append(p + t)
            total.append(k)
            continue
        final.append(p + t + STOP_THINKING)
        total.append(k)
    ans, m, _ = gen(model, tok, final, 512 if not smoke else 6, SAMPLING[True], seed + 1, eos)
    return [f + a for f, a in zip(final, ans)], [k + j for k, j in zip(total, m)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=QWEN3_1P7B[0])
    ap.add_argument("--revision", default=QWEN3_1P7B[1])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--budgets", default="0,256,512,1024,2048,none")
    ap.add_argument("--route", choices=["none", "cascade"], default="none")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        from frontierlab.pipeline.hf_stages import load
        model, tok = load("x", None, True, "cpu")
        items = [{"question": f"{i}+1", "answer": str(i + 1)} for i in range(4)]
        eos = [tok.eos_token_id]
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
        model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16).cuda().eval()
        items = gsm8k.load("test")[:a.n]
        eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    res = {}
    ps_nt = [prompt(tok, q["question"], False, a.smoke) for q in items]
    ps_t = [prompt(tok, q["question"], True, a.smoke) for q in items]

    def run(prompts, fn):
        texts, toks = [], []
        for i in range(0, len(prompts), a.batch):
            t, n = fn(prompts[i:i + a.batch], i)
            texts += t
            toks += n
        return texts, toks

    nt_texts, nt_toks, nt_lp = [], [], []
    for i in range(0, len(ps_nt), a.batch):
        t, n, lp = gen(model, tok, ps_nt[i:i + a.batch], 1024 if not a.smoke else 6, SAMPLING[False], i, eos)
        nt_texts += t; nt_toks += n; nt_lp += lp
    nt_c = [correct(t, q["answer"]) for t, q in zip(nt_texts, items)]
    res["nothink"] = {"accuracy": sum(nt_c) / len(nt_c), "tokens": sum(nt_toks) / len(nt_toks)}
    think_c, think_toks = None, None
    if a.route == "none":
        for b in a.budgets.split(","):
            B = None if b == "none" else int(b)
            texts, toks = run(ps_t, lambda ps, i: with_budget(model, tok, ps, B, i, eos, a.smoke))
            c = [correct(t, q["answer"]) for t, q in zip(texts, items)]
            res[f"think_B{b}"] = {"accuracy": sum(c) / len(c), "tokens": sum(toks) / len(toks)}
            print(b, res[f"think_B{b}"])
    else:
        texts, toks = run(ps_t, lambda ps, i: with_budget(model, tok, ps, None, i, eos, a.smoke))
        think_c = [correct(t, q["answer"]) for t, q in zip(texts, items)]
        for th in (-2.0, -1.0, -0.6, -0.4, -0.3, -0.2, -0.1, 0.0):
            esc = [lp < th for lp in nt_lp]
            acc = sum(tc if e else nc for e, tc, nc in zip(esc, think_c, nt_c)) / len(items)
            cost = sum(nt + (tt if e else 0) for e, nt, tt in zip(esc, nt_toks, toks)) / len(items)
            res[f"cascade_{th}"] = {"escalated": sum(esc) / len(esc), "accuracy": acc, "tokens": cost}
            print(th, res[f"cascade_{th}"])
    print(res["nothink"])
    (a.out / f"think_{a.route}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
