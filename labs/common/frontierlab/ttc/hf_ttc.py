"""Main path of lesson 15.1 and the project: test-time compute on GSM8K with a Stage D reasoning model.

    python -m frontierlab.ttc.hf_ttc sample --out runs/m15/main --n-questions 500 --n 32           # vLLM 0.30.0, 1x H100
    python -m frontierlab.ttc.hf_ttc single --out runs/m15/main --budgets 256,512,1024,2048,4096
    python -m frontierlab.ttc.hf_ttc score  --out runs/m15/main                                     # ORM, Transformers
    python -m frontierlab.ttc.hf_ttc search --out runs/m15/main --width 4 --expand 4                # PRM beam search
    python -m frontierlab.ttc.hf_ttc latency --out runs/m15/main                                    # one question at a time
    python -m frontierlab.ttc.hf_ttc report --out runs/m15/main --budget 4096 --latency 20
    python -m frontierlab.ttc.hf_ttc smoke --out runs/m15/hf-smoke                                  # CPU, tiny random models

Not run in this build except ``smoke``; part of the Module 15 pilot.

* **Policy:** ``Qwen/Qwen3-1.7B`` at ``70d244cc`` (thinking and non-thinking modes, lesson 13.4), or ``--model`` a
  Module 14 RL checkpoint of Qwen3-1.7B-Base. Sampling as the model card recommends (thinking: temperature 0.6,
  top-p 0.95, top-k 20; non-thinking: 0.7, 0.8, 20).
* **Arms.** ``single``: one thinking chain per question with budget forcing at each budget (13.4's
  stop-thinking text). ``sample``: n candidates per question in non-thinking mode, and in thinking mode at a
  short budget (``--sample-budget``), from which majority vote, best-of-N and weighted vote are computed on
  subsets (``frontierlab.ttc.report``). ``search``: PRM-guided step-level beam search in non-thinking mode,
  steps separated by a blank line (the Hugging Face test-time-compute study's convention).
* **Verifiers.** Outcome: ``Skywork/Skywork-Reward-V2-Qwen3-1.7B`` at ``e51ea3e`` (a general reward model, Apache-2.0,
  pinned in Module 12); its scalar score is mapped to (0, 1) by a sigmoid for weighting. Process:
  ``Qwen/Qwen2.5-Math-PRM-7B`` at ``0610740`` (Qwen licence; ``trust_remote_code``: read the remote code at that
  revision before running it): every step is followed by ``<extra_0>``, and a step's reward is the
  positive-class probability of the two-way softmax at its ``<extra_0>`` position (model card).
* **Cost:** generated tokens from the engine's token ids, verifier tokens from the verifier's tokenizer, both
  saved per candidate; FLOPs and policy-token equivalents with ``frontierlab.ttc.budget`` (N_policy =
  2.03e9 with the separate output matrix of Qwen3-1.7B, N_ORM = 1.7e9, N_PRM = 7.6e9: check against the loaded
  models, the script prints them). **Latency:** wall-clock per question, one question at a time with
  ``torch.cuda.synchronize()`` around it, for every arm (``latency``).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from frontierlab.posttrain import gsm8k

POLICY = ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
ORM = ("Skywork/Skywork-Reward-V2-Qwen3-1.7B", "e51ea3e08fb81326c3b812a7ff0cb9cee83e59cc")
PRM = ("Qwen/Qwen2.5-Math-PRM-7B", "0610740060112df12585d00a1c5f4624d2f59051")
STOP_THINKING = ("Considering the limited time by the user, I have to give the solution based on the thinking "
                 "directly now.\n</think>.\n\n")
SAMPLING = {True: dict(temperature=0.6, top_p=0.95, top_k=20), False: dict(temperature=0.7, top_p=0.8, top_k=20)}
INSTR = "\nSolve it step by step, separating steps with a blank line, and end with '#### <number>'."


def answer_of(text: str) -> str | None:
    tail = text.split("</think>")[-1]
    a = gsm8k.strict_answer(tail)
    if a is not None:
        return a
    import re
    nums = re.findall(gsm8k._NUM, gsm8k.cut(tail))
    return gsm8k.normalise(nums[-1]) if nums else None


# --------------------------------------------------------------------------------------------- backends

class VLLMBackend:
    """vLLM 0.30.0 offline engine with prefix caching (the prompt of n samples is prefilled once)."""

    def __init__(self, model: str, revision: str, seed: int = 0, gpu_memory_utilization: float = 0.45,
                 max_model_len: int = 8192):
        from vllm import LLM
        self.llm = LLM(model=model, revision=revision, dtype="bfloat16", seed=seed, enable_prefix_caching=True,
                       gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len)
        self.tok = self.llm.get_tokenizer()

    def generate(self, prompts: list[str], n: int, max_tokens: int, sampling: dict, seed: int, stop=None):
        from vllm import SamplingParams
        sp = SamplingParams(n=n, max_tokens=max_tokens, seed=seed, stop=stop, **sampling)
        outs = self.llm.generate(prompts, sp, use_tqdm=False)
        return [[(c.text, len(c.token_ids), c.finish_reason) for c in o.outputs] for o in outs]

    def close(self):
        del self.llm
        torch.cuda.empty_cache()


class HFBackend:
    """Transformers ``generate`` (the smoke test and machines without vLLM). No prefix caching."""

    def __init__(self, model, tok, device="cpu"):
        self.model, self.tok, self.dev = model.eval(), tok, device

    @torch.no_grad()
    def generate(self, prompts: list[str], n: int, max_tokens: int, sampling: dict, seed: int, stop=None):
        from frontierlab.pipeline.hf_eval import encode
        from frontierlab.posttrain.hf import left_pad
        torch.manual_seed(seed)
        out = []
        for p in prompts:
            ids, att = left_pad([encode(self.tok, p)] * n, self.tok.pad_token_id)
            seq = self.model.generate(input_ids=ids.to(self.dev), attention_mask=att.to(self.dev), do_sample=True,
                                      max_new_tokens=max_tokens, pad_token_id=self.tok.pad_token_id,
                                      eos_token_id=self.tok.eos_token_id, **sampling)
            row = []
            for s in seq[:, ids.shape[1]:].tolist():
                k = next((i + 1 for i, t in enumerate(s) if t == self.tok.eos_token_id), len(s))
                text = self.tok.decode(s[:k], skip_special_tokens=True)
                reason = "stop" if k < len(s) else "length"
                if stop:
                    for st in stop:
                        if st in text:
                            text, reason = text.split(st)[0], "stop"
                row.append((text, k, reason))
            out.append(row)
        return out


def render(tok, question: str, thinking: bool, smoke: bool) -> str:
    text = question + INSTR
    if smoke or not hasattr(tok, "apply_chat_template"):
        return text
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=thinking)


def ntok(tok, text: str) -> int:
    try:
        return len(tok.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tok.encode(text))


# --------------------------------------------------------------------------------------------- arms

def budget_forced(be, prompts: list[str], budget: int, seed: int, answer_tokens: int = 512):
    """Thinking with at most ``budget`` tokens, then the stop-thinking text and an answer (lesson 13.4)."""
    first = be.generate(prompts, 1, budget, SAMPLING[True], seed)
    final, think_n = [], []
    for p, ((t, k, reason),) in zip(prompts, first):
        think_n.append(k)
        final.append(p + t + ("" if "</think>" in t else STOP_THINKING))
    ans = be.generate(final, 1, answer_tokens, SAMPLING[True], seed + 1)
    return [{"text": f[len(p):] + a[0][0], "gen_tokens": k + a[0][1]} for p, f, k, a in zip(prompts, final, think_n, ans)]


def beam_search(be, prm_score, tok, question: str, prompt: str, width: int, expand: int, max_steps: int = 12,
                step_tokens: int = 256, seed: int = 0) -> dict:
    """Step-level beam search (non-thinking mode): steps end at a blank line; the PRM scores every candidate
    prefix by its last step's reward; ``width`` beams survive; a beam whose step contains '####' is finished."""
    beams, finished = [""], []
    dec = ver = 0
    for step in range(max_steps):
        m = width * expand if step == 0 else expand
        outs = be.generate([prompt + b for b in beams], m, step_tokens, SAMPLING[False], seed + step, stop=["\n\n"])
        cands = []
        for b, row in zip(beams, outs):
            for text, k, _ in row:
                dec += k
                cands.append(b + text + "\n\n")
        sc, vt = prm_score(question, cands)
        ver += vt
        order = np.argsort(-np.asarray(sc))
        beams = []
        for j in order:
            (finished if "####" in cands[j] else beams).append((cands[j], sc[j]) if "####" in cands[j] else cands[j])
            if len(beams) >= width:
                break
        if not beams or len(finished) >= width:
            break
    pool = finished or [(b, 0.0) for b in beams]
    best = max(pool, key=lambda x: x[1])[0]
    return {"text": best, "decode_tokens": dec, "verifier_tokens": ver}


# --------------------------------------------------------------------------------------------- verifiers

class ORMScorer:
    def __init__(self, repo=ORM[0], revision=ORM[1], device="cuda", smoke=False):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if smoke:
            from transformers import Qwen3Config, Qwen3ForSequenceClassification
            from frontierlab.posttrain.hf import SmokeTokenizer
            torch.manual_seed(0)
            self.model = Qwen3ForSequenceClassification(Qwen3Config(vocab_size=96, hidden_size=32, intermediate_size=64,
                                                                    num_hidden_layers=1, num_attention_heads=4,
                                                                    num_key_value_heads=2, head_dim=8, num_labels=1,
                                                                    pad_token_id=0)).eval()
            self.tok, self.smoke, self.dev = SmokeTokenizer(), True, "cpu"
        else:
            self.tok = AutoTokenizer.from_pretrained(repo, revision=revision)
            self.model = AutoModelForSequenceClassification.from_pretrained(repo, revision=revision,
                                                                            dtype=torch.bfloat16, num_labels=1).to(device).eval()
            self.smoke, self.dev = False, device
        self.params = sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def __call__(self, question: str, responses: list[str]) -> tuple[list[float], int]:
        scores, toks = [], 0
        for r in responses:
            if self.smoke:
                ids = torch.tensor([self.tok.encode(question + r)[:256] or [2]])
            else:
                conv = [{"role": "user", "content": question}, {"role": "assistant", "content": r}]
                ids = self.tok.apply_chat_template(conv, tokenize=True, return_tensors="pt").to(self.dev)
            toks += ids.shape[1]
            s = float(self.model(ids).logits[0][0])
            scores.append(1 / (1 + math.exp(-s)))
        return scores, toks


class PRMScorer:
    """Qwen2.5-Math-PRM-7B per its model card; ``smoke`` = a random tiny classifier with the same interface."""

    def __init__(self, repo=PRM[0], revision=PRM[1], device="cuda", smoke=False):
        self.smoke = smoke
        if smoke:
            self.orm = ORMScorer(smoke=True)
            self.params = self.orm.params
            return
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(repo, revision=revision, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(repo, revision=revision, dtype=torch.bfloat16,
                                               trust_remote_code=True).to(device).eval()
        self.sep = self.tok.encode("<extra_0>")[0]
        self.dev = device
        self.params = sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def __call__(self, question: str, prefixes: list[str]) -> tuple[list[float], int]:
        if self.smoke:
            return self.orm(question, prefixes)
        scores, toks = [], 0
        for pre in prefixes:
            steps = [s for s in pre.split("\n\n") if s.strip()]
            msgs = [{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"}]
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            ids = self.tok.encode(text, return_tensors="pt").to(self.dev)
            toks += ids.shape[1]
            logits = self.model(input_ids=ids)[0]
            pos = (ids[0] == self.sep).nonzero().squeeze(-1)
            p = torch.softmax(logits[0, pos].float(), -1)[:, 1]
            scores.append(float(p[-1]) if len(p) else 0.0)          # last step's reward
        return scores, toks


# --------------------------------------------------------------------------------------------- commands

def _items(a):
    if a.smoke:
        return [{"question": f"{i} + {i + 1} = ?", "answer": str(2 * i + 1)} for i in range(3)]
    return gsm8k.load("test")[:a.n_questions]


def _backend(a):
    if a.smoke:
        from frontierlab.pipeline.hf_stages import load
        model, tok = load("x", None, True, "cpu")
        return HFBackend(model, tok), tok
    be = VLLMBackend(a.model, a.revision, a.seed)
    return be, be.tok


def _dump(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _load(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def cmd_sample(a):
    be, tok = _backend(a)
    items = _items(a)
    rows = []
    for mode, budget in (("nothink", None), ("think", a.sample_budget)):
        prompts = [render(tok, q["question"], mode == "think", a.smoke) for q in items]
        if mode == "nothink":
            outs = be.generate(prompts, a.n, a.max_tokens, SAMPLING[False], a.seed)
            cands = [[{"text": t, "gen_tokens": k} for t, k, _ in row] for row in outs]
        else:
            cands = [[] for _ in items]
            for j in range(a.n):
                for i, r in enumerate(budget_forced(be, prompts, budget, a.seed + 1000 * j,
                                                    a.max_tokens if not a.smoke else 4)):
                    cands[i].append(r)
        for q, p, cs in zip(items, prompts, cands):
            for c in cs:
                c["answer"] = answer_of(c["text"])
                c["correct"] = c["answer"] == q["answer"]
            rows.append({"mode": mode, "budget": budget, "question": q["question"], "gold": q["answer"],
                         "prompt_tokens": ntok(tok, p), "cands": cs})
    _dump(a.out / "samples.jsonl", rows)
    print(f"wrote {len(rows)} rows to {a.out / 'samples.jsonl'}")


def cmd_single(a):
    be, tok = _backend(a)
    items = _items(a)
    prompts = [render(tok, q["question"], True, a.smoke) for q in items]
    rows = []
    for b in [int(x) for x in a.budgets.split(",")]:
        res = budget_forced(be, prompts, b, a.seed, a.max_tokens if not a.smoke else 4)
        for q, p, r in zip(items, prompts, res):
            ans = answer_of(r["text"])
            rows.append({"budget": b, "gold": q["answer"], "answer": ans, "correct": ans == q["answer"],
                         "gen_tokens": r["gen_tokens"], "prompt_tokens": ntok(tok, p)})
    _dump(a.out / "single.jsonl", rows)


def cmd_score(a):
    orm = ORMScorer(smoke=a.smoke, device="cpu" if a.smoke else "cuda")
    rows = _load(a.out / "samples.jsonl")
    for r in rows:
        texts = [c["text"].split("</think>")[-1] for c in r["cands"]]
        for c, t in zip(r["cands"], texts):
            s, k = orm(r["question"], [t])
            c["score"], c["scored_tokens"] = s[0], k
    _dump(a.out / "samples.jsonl", rows)
    (a.out / "verifier.json").write_text(json.dumps({"orm": ORM, "orm_params": orm.params}))


def cmd_search(a):
    be, tok = _backend(a)
    prm = PRMScorer(smoke=a.smoke, device="cpu" if a.smoke else "cuda")
    items = _items(a)
    rows = []
    for q in items:
        p = render(tok, q["question"], False, a.smoke)
        t0 = time.perf_counter()
        r = beam_search(be, prm, tok, q["question"], p, a.width, a.expand, max_steps=2 if a.smoke else 12,
                        step_tokens=8 if a.smoke else 256, seed=a.seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        r["seconds"] = time.perf_counter() - t0
        r["answer"] = answer_of(r["text"])
        r["correct"] = r["answer"] == q["answer"]
        r["prompt_tokens"] = ntok(tok, p)
        rows.append(r)
    _dump(a.out / f"search_w{a.width}_e{a.expand}.jsonl", rows)
    (a.out / "prm.json").write_text(json.dumps({"prm": PRM, "prm_params": prm.params}))


def cmd_latency(a):
    """Wall-clock per question, one at a time, for N in --ns (non-thinking samples) and each single-chain budget."""
    be, tok = _backend(a)
    items = _items(a)[:a.latency_questions]
    out = {"parallel": {}, "single": {}}
    for N in [int(x) for x in a.ns.split(",")]:
        ts = []
        for q in items:
            p = render(tok, q["question"], False, a.smoke)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            be.generate([p], N, a.max_tokens if not a.smoke else 4, SAMPLING[False], a.seed)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        out["parallel"][N] = float(np.median(ts))
    for b in [int(x) for x in a.budgets.split(",")]:
        ts = []
        for q in items:
            p = render(tok, q["question"], True, a.smoke)
            t0 = time.perf_counter()
            budget_forced(be, [p], b, a.seed, a.max_tokens if not a.smoke else 4)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        out["single"][b] = float(np.median(ts))
    (a.out / "latency.json").write_text(json.dumps(out, indent=1))
    print(out)


def cmd_report(a):
    """Budget-matched table at --budget policy-token equivalents and the recommendation at --latency seconds."""
    from frontierlab.ttc import report as R
    rows_s = _load(a.out / "samples.jsonl")
    v = json.loads((a.out / "verifier.json").read_text()) if (a.out / "verifier.json").exists() else {"orm_params": 0}
    lat = json.loads((a.out / "latency.json").read_text()) if (a.out / "latency.json").exists() else None
    Np = a.policy_params
    rows = []
    for mode in ("nothink", "think"):
        rs = [r for r in rows_s if r["mode"] == mode]
        if not rs:
            continue
        pool = [r["cands"] for r in rs]
        gold = [r["gold"] for r in rs]
        P = float(np.mean([r["prompt_tokens"] for r in rs]))
        n = len(pool[0])
        Ns = [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= n]
        new = R.pool_rows(mode, pool, gold, Ns, policy_params=Np, verifier_params=v.get("orm_params", 0),
                          prompt_tokens=P, resamples=100)
        if lat and mode == "nothink":
            par = {int(k): t for k, t in lat["parallel"].items()}
            for r in new:
                if r["N"] in par:
                    r["latency_s"] = par[r["N"]]
        rows += new
    if (a.out / "single.jsonl").exists():
        rs = _load(a.out / "single.jsonl")
        for b in sorted({r["budget"] for r in rs}):
            sel = [r for r in rs if r["budget"] == b]
            row = R.single_row(f"thinking budget {b}", [r["correct"] for r in sel],
                               float(np.mean([r["gen_tokens"] for r in sel])), policy_params=Np,
                               prompt_tokens=float(np.mean([r["prompt_tokens"] for r in sel])))
            if lat and str(b) in lat["single"]:
                row["latency_s"] = lat["single"][str(b)]
            rows.append(row)
    for f in sorted(a.out.glob("search_w*_e*.jsonl")):
        res = _load(f)
        prm = json.loads((a.out / "prm.json").read_text())
        rows.append(R.search_row(f.stem, res, policy_params=Np, verifier_params=prm["prm_params"],
                                 prompt_tokens=float(np.mean([r["prompt_tokens"] for r in res])),
                                 latency_s=float(np.median([r["seconds"] for r in res]))))
    print(R.format_rows(sorted(rows, key=lambda r: r["pte"])))
    rec = R.recommend(rows, a.budget, a.latency)
    print("\nrecommendation:", rec["choice"]["label"] if rec["choice"] else None, "-", rec["why"])


def cmd_smoke(a):
    a.smoke = True
    for fn in (cmd_sample, cmd_single, cmd_score, cmd_search, cmd_latency):
        fn(a)
    a.policy_params = 10_000
    cmd_report(a)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["sample", "single", "score", "search", "latency", "report", "smoke"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=POLICY[0])
    ap.add_argument("--revision", default=POLICY[1])
    ap.add_argument("--n-questions", type=int, default=500)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--sample-budget", type=int, default=512)
    ap.add_argument("--budgets", default="256,512,1024,2048,4096")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--expand", type=int, default=4)
    ap.add_argument("--ns", default="1,4,16")
    ap.add_argument("--latency-questions", type=int, default=50)
    ap.add_argument("--budget", type=float, default=4096)
    ap.add_argument("--latency", type=float, default=None)
    ap.add_argument("--policy-params", type=int, default=2_031_739_904)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    {"sample": cmd_sample, "single": cmd_single, "score": cmd_score, "search": cmd_search, "latency": cmd_latency,
     "report": cmd_report, "smoke": cmd_smoke}[a.command](a)


if __name__ == "__main__":
    main()
