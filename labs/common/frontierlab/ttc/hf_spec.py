"""Main path of lesson 15.2: speculative decoding on the Stage D models.

    python -m frontierlab.ttc.hf_spec own  --out runs/m15/l152-main --gammas 1,2,4,6 --prompts 64     # Transformers, 1x GPU
    python -m frontierlab.ttc.hf_spec vllm --out runs/m15/l152-main --method draft_model --k 4         # vLLM 0.30.0
    python -m frontierlab.ttc.hf_spec vllm --out runs/m15/l152-main --method eagle3 --k 3
    python -m frontierlab.ttc.hf_spec own  --smoke --out runs/m15/l152-smoke                           # CPU, tiny random models

Not run in this build except ``--smoke``; part of the Module 15 pilot.

* ``own``: the course's rejection-sampling step (:func:`frontierlab.ttc.speculative.accept_reject`) on Hugging
  Face models. Target ``Qwen/Qwen3-1.7B-Base`` (``ea980cb``), draft ``Qwen/Qwen3-0.6B-Base`` (``da87bfb``): same
  tokenizer and vocabulary (151,936). Both use a ``DynamicCache``; after each round both caches are cut back to
  the accepted prefix with ``DynamicCache.crop``. Measured per prompt: acceptance rate, accepted tokens per
  round, and wall-clock (``torch.cuda.synchronize()`` before reading the clock) against plain decoding with the
  same code, interleaved. Batch 1: the latency setting where speculative decoding pays.
* ``vllm``: the production path. ``--method draft_model`` (the 0.6B draft), ``eagle3`` (``AngelSlim/Qwen3-1.7B_eagle3``
  at ``94441b4``, an EAGLE-3 head trained for ``Qwen/Qwen3-1.7B``, not the Base model: the target becomes
  ``Qwen/Qwen3-1.7B`` at ``70d244cc``; its licence is a custom AngelSlim licence, read it first) or ``ngram``.
  ``speculative_config={"method": ..., "model": ..., "num_speculative_tokens": k}``. Acceptance comes from the
  engine's counters ``vllm:spec_decode_num_accepted_tokens`` and ``vllm:spec_decode_num_draft_tokens``
  (``LLM.get_metrics()``; check the names at the pilot), throughput from tokens / wall-clock, with and without
  speculation, at batch 1 and at batch 32.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from frontierlab.ttc.speculative import accept_reject, probs_from_logits

TARGET = ("Qwen/Qwen3-1.7B-Base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1")
DRAFT = ("Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd")
CHAT = ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
EAGLE3 = ("AngelSlim/Qwen3-1.7B_eagle3", "94441b48acc5804677ae12259617c83323b543a9")


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class HFModel:
    """A Transformers causal LM with a croppable cache: ``feed(ids)`` returns the logits of the new tokens."""

    def __init__(self, model):
        self.model = model.eval()
        self.cache = None
        self.length = 0

    def reset(self):
        from transformers import DynamicCache
        self.cache, self.length = DynamicCache(), 0

    @torch.no_grad()
    def feed(self, ids: list[int]) -> torch.Tensor:
        dev = next(self.model.parameters()).device
        x = torch.tensor([ids], device=dev)
        out = self.model(input_ids=x, past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        self.length += len(ids)
        return out.logits[0].float()

    def crop(self, n: int):
        """Keep the first n cached tokens (Transformers 5.18: a negative argument removes that many tokens)."""
        if n < self.length:
            self.cache.crop(n - self.length)
            self.length = n


@torch.no_grad()
def hf_speculative(target: HFModel, draft: HFModel, prompt: list[int], max_new: int, gamma: int, temperature: float,
                   gen: torch.Generator, eos: int | None = None) -> tuple[list[int], dict]:
    """The same algorithm as :func:`frontierlab.ttc.speculative.speculative_generate`, with an LM draft."""
    target.reset(); draft.reset()
    seq = list(prompt)
    target.feed(seq[:-1])
    goal = len(seq) + max_new
    rounds = proposed = accepted = 0
    while len(seq) < goal:
        g = min(gamma, goal - len(seq) - 1)
        toks, qs = [], []
        if g > 0:
            lg = draft.feed(seq[draft.length:])[-1]
            for i in range(g):
                q = probs_from_logits(lg, temperature)
                t = int(torch.multinomial(q, 1, generator=gen))
                toks.append(t); qs.append(q)
                if i < g - 1:
                    lg = draft.feed([t])[-1]
        p = probs_from_logits(target.feed([seq[-1]] + toks), temperature)
        if g > 0:
            u = torch.rand(g, generator=gen, device=p.device)
            n, nxt = accept_reject(p[None], torch.stack(qs)[None], torch.tensor([toks], device=p.device), u[None], gen)
            n, nxt = int(n), int(nxt)
        else:
            n, nxt = 0, int(torch.multinomial(p[0], 1, generator=gen))
        rounds += 1; proposed += g; accepted += n
        target.crop(len(seq) + n)
        seq += toks[:n] + [nxt]
        draft.crop(min(draft.length, len(seq) - 1))
        if eos is not None and eos in toks[:n] + [nxt]:
            break
    return seq[len(prompt):max(len(prompt), goal)], {"rounds": rounds, "proposed": proposed, "accepted": accepted}


@torch.no_grad()
def hf_plain(target: HFModel, prompt: list[int], max_new: int, temperature: float, gen: torch.Generator) -> list[int]:
    target.reset()
    lg = target.feed(prompt)[-1]
    out = []
    for _ in range(max_new):
        t = int(torch.multinomial(probs_from_logits(lg, temperature), 1, generator=gen))
        out.append(t)
        lg = target.feed([t])[-1]
    return out


def _load(smoke: bool, device: str):
    if smoke:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(0)
        mk = lambda L, h: Qwen3ForCausalLM(Qwen3Config(vocab_size=96, hidden_size=h, intermediate_size=2 * h,
                                                       num_hidden_layers=L, num_attention_heads=4, num_key_value_heads=2,
                                                       head_dim=h // 4, max_position_embeddings=512)).eval()
        tgt, dr = mk(2, 64), mk(1, 32)
        prompts = [[int(x) for x in torch.randint(3, 96, (12,))] for _ in range(3)]
        return HFModel(tgt), HFModel(dr), prompts
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from frontierlab.posttrain import gsm8k
    tok = AutoTokenizer.from_pretrained(TARGET[0], revision=TARGET[1])
    tgt = AutoModelForCausalLM.from_pretrained(TARGET[0], revision=TARGET[1], dtype=torch.bfloat16).to(device)
    dr = AutoModelForCausalLM.from_pretrained(DRAFT[0], revision=DRAFT[1], dtype=torch.bfloat16).to(device)
    shots = gsm8k.load("train")[:4]
    qs = gsm8k.load("test")
    prompts = [tok.encode(gsm8k.few_shot_prompt(q["question"], shots)) for q in qs]
    return HFModel(tgt), HFModel(dr), prompts


def cmd_own(a):
    dev = "cpu" if a.smoke else "cuda"
    target, draft, prompts = _load(a.smoke, dev)
    prompts = prompts[:a.prompts]
    rows = []
    for T in (0.0, 1.0):
        for gamma in [int(x) for x in a.gammas.split(",")]:
            acc = prop = rnd = 0
            t_spec, t_plain = [], []
            for i, p in enumerate(prompts):
                for arm in ("plain", "spec"):                       # interleaved per prompt
                    g = torch.Generator(device=dev).manual_seed(i)
                    _sync(); t0 = time.perf_counter()
                    if arm == "plain":
                        hf_plain(target, p, a.max_new, T, g)
                    else:
                        _, st = hf_speculative(target, draft, p, a.max_new, gamma, T, g)
                        acc += st["accepted"]; prop += st["proposed"]; rnd += st["rounds"]
                    _sync()
                    (t_plain if arm == "plain" else t_spec).append(time.perf_counter() - t0)
            r = np.array(t_plain) / np.array(t_spec)
            rows.append({"temperature": T, "gamma": gamma, "acceptance": acc / max(1, prop),
                         "tokens_per_round": a.max_new * len(prompts) / max(1, rnd),
                         "speedup_median": float(np.median(r)),
                         "speedup_iqr": [float(np.quantile(r, 0.25)), float(np.quantile(r, 0.75))]})
            print(rows[-1])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "own.json").write_text(json.dumps(rows, indent=1))


def cmd_vllm(a):
    from vllm import LLM, SamplingParams
    from frontierlab.posttrain import gsm8k
    target = CHAT if a.method == "eagle3" else TARGET
    spec = None
    if a.method == "draft_model":
        spec = {"method": "draft_model", "model": DRAFT[0], "revision": DRAFT[1], "num_speculative_tokens": a.k}
    elif a.method == "eagle3":
        spec = {"method": "eagle3", "model": EAGLE3[0], "revision": EAGLE3[1], "num_speculative_tokens": a.k}
    elif a.method == "ngram":
        spec = {"method": "ngram", "num_speculative_tokens": a.k, "prompt_lookup_max": 4}
    res = {}
    for label, cfg in (("plain", None), (a.method, spec)):
        llm = LLM(model=target[0], revision=target[1], dtype="bfloat16", seed=0, speculative_config=cfg,
                  gpu_memory_utilization=0.6, max_model_len=4096)
        qs = gsm8k.load("test")[:a.prompts]
        prompts = [f"Question: {q['question']}\nAnswer:" for q in qs]
        sp = SamplingParams(temperature=a.temperature, max_tokens=a.max_new, seed=0)
        for bs in (1, 32):
            _sync(); t0 = time.perf_counter()
            n_tok = 0
            for i in range(0, len(prompts), bs):
                outs = llm.generate(prompts[i:i + bs], sp, use_tqdm=False)
                n_tok += sum(len(o.outputs[0].token_ids) for o in outs)
            _sync()
            res[f"{label}@bs{bs}"] = {"tokens_per_s": n_tok / (time.perf_counter() - t0)}
        try:
            m = {x.name: getattr(x, "value", getattr(x, "values", None)) for x in llm.get_metrics()}
            res[f"{label}_metrics"] = {k: v for k, v in m.items() if "spec_decode" in k}
        except Exception as e:                                    # noqa: BLE001 - metrics API checked at the pilot
            res[f"{label}_metrics"] = f"unavailable: {e}"
        del llm
        torch.cuda.empty_cache()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"vllm_{a.method}_k{a.k}.json").write_text(json.dumps(res, indent=1, default=str))
    print(res)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["own", "vllm"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gammas", default="1,2,4,6")
    ap.add_argument("--prompts", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--method", choices=["draft_model", "eagle3", "ngram"], default="draft_model")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.prompts, a.max_new, a.gammas = 2, 8, "2"
    {"own": cmd_own, "vllm": cmd_vllm}[a.command](a)


if __name__ == "__main__":
    main()
