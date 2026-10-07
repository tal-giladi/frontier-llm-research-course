"""The short pipeline on Recipe-R (the learner's Module 11 checkpoint): SFT -> DPO -> RLVR, with an Eval-v2-style
suite after each stage (Module 13 project, part 2).

Recipe-R is a language model pretrained on FineWeb-Edu text with a byte-level BPE tokenizer (CPU variant: the
919K-parameter ``target-m11-r5-x10`` at vocabulary 1,024). Post-training it on the course's arithmetic task
asks the question every lab asks of its own base model: what does each stage buy, and what does it cost the
pretrained ability? The retention component is therefore the pretraining one: the model's loss on held-out
Data-v0 validation text.

* **Task.** Prompt ``"Q: 07+35="``, response ``" 42"`` then the end-of-text token (id 0) as EOS, both
  operations, training triples only (Module 12's split). The BPE tokenizer merges some digit pairs, so prompts
  are 6-9 tokens long: sampling and greedy decoding run per prompt-length bucket.
* **Stages.** SFT on gold targets (:func:`stage_sft`); DPO + NLL on the model's own samples ranked by exact match
  (:func:`stage_dpo`); a short GRPO with the exact-match reward (:func:`stage_rlvr`). There is no distillation
  stage: no teacher shares Recipe-R's tokenizer (lesson 13.2).
* **Suite** (:func:`suite`): ``add_greedy``, ``sub_greedy`` (task), ``val_nll`` (retention: minus the mean
  next-token loss of 64 windows of 256 tokens of Data-v0 validation text, one item per window).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from frontierlab.evals.suite_v2.core import VERSION
from frontierlab.model import LM, ModelConfig
from frontierlab.pipeline.compute import Ledger, n_params
from frontierlab.pipeline.seqs import Example, train_sft
from frontierlab.posttrain import advantages as A
from frontierlab.posttrain import losses as Lo
from frontierlab.posttrain.policy import token_logprobs
from frontierlab.posttrain.tasks import Problem, make_problems, problems_from, split_problems

DATA = Path("labs/common/data/m11-v1024")
DEFAULT_CKPT = Path("runs/m11/project/cpu/target-m11-r5-x10/checkpoint.pt")
EOT = 0


class RTok:
    """Recipe-R's tokenizer (``tokenizers`` file of the Data-v0 retokenization at vocabulary 1,024)."""

    def __init__(self, path: str | Path = DATA / "tokenizer.json"):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(path))

    def encode(self, s: str) -> list[int]:
        return self.tok.encode(s).ids

    def decode(self, ids) -> str:
        ids = [int(i) for i in ids]
        if EOT in ids:
            ids = ids[:ids.index(EOT)]
        return self.tok.decode(ids)


def load_recipe_r(path: str | Path = DEFAULT_CKPT) -> LM:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = LM(ModelConfig(**ck["config"]))
    m.load_state_dict(ck["model"])
    return m.eval()


def prompt_text(p: Problem) -> str:
    return f"Q: {p.a:02d}{p.op}{p.b:02d}="


def answer_text(p: Problem) -> str:
    return f" {p.result}"


def example(tok: RTok, p: Problem, text: str | None = None, finished: bool = True) -> Example:
    return Example.of([EOT] + tok.encode(prompt_text(p)), tok.encode(answer_text(p) if text is None else text)
                      + ([EOT] if finished else []))


@dataclass
class RRollout:
    tokens: torch.Tensor          # (B, P + R) prompt then response, EOT after the end
    response: torch.Tensor        # (B, R)
    mask: torch.Tensor            # (B, R) float: 1 up to and including the first EOT
    finished: torch.Tensor        # (B,) bool
    prompt_len: int


@torch.no_grad()
def sample(model, prompts: torch.Tensor, max_new: int, temperature: float, gen: torch.Generator) -> RRollout:
    """KV-cache sampling that ends a row at Recipe-R's end-of-text token (id 0), not the toy tokenizer's EOS.
    (Module 12's sampler assumes the toy tokenizer's EOS = 2, which is an ordinary BPE token here.)"""
    model.eval()
    B, P = prompts.shape
    cache = model.new_cache()
    logits = model(prompts, cache=cache).logits[:, -1].float()
    out = torch.full((B, max_new), EOT, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool)
    for t in range(max_new):
        if temperature > 0:
            nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=gen).squeeze(-1)
        else:
            nxt = logits.argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, EOT), nxt)
        out[:, t] = nxt
        done = done | (nxt == EOT)
        if bool(done.all()):
            break
        logits = model(nxt[:, None], cache=cache).logits[:, -1].float()
    is_end = out == EOT
    before = torch.cumsum(is_end.long(), -1) - is_end.long()
    return RRollout(torch.cat([prompts, out], 1), out, (before == 0).float(), is_end.any(-1), P)


def _buckets(tok, problems):
    by = {}
    for i, p in enumerate(problems):
        ids = [EOT] + tok.encode(prompt_text(p))
        by.setdefault(len(ids), []).append((i, ids))
    return by


@torch.no_grad()
def sample_texts(model, tok, problems, n: int = 1, temperature: float = 1.0, seed: int = 0, max_new: int = 5,
                 ledger: Ledger | None = None) -> list:
    """``n`` (text, finished) per problem; temperature 0 = greedy."""
    gen = torch.Generator().manual_seed(seed)
    out = [None] * len(problems)
    N = n_params(model)
    for L, rows in _buckets(tok, problems).items():
        ids = torch.tensor([r for _, r in rows for _ in range(n)], dtype=torch.long)
        ro = sample(model, ids, max_new, temperature, gen)
        resp, fin = ro.response, ro.finished
        if ledger is not None:
            ledger.forward("student_sample", N, float(ids.numel() + resp.numel()))
        texts = [(tok.decode(r.tolist()), bool(f)) for r, f in zip(resp, fin)]
        for j, (i, _) in enumerate(rows):
            out[i] = texts[j * n:(j + 1) * n]
    return out


def correct(p: Problem, text: str, finished: bool) -> bool:
    return finished and text.strip() == str(p.result)


def val_items(model, n_windows: int = 64, T: int = 256, seed: int = 0) -> list[float]:
    """Minus the mean next-token loss of ``n_windows`` fixed windows of Data-v0 validation text (one item each)."""
    data = np.memmap(DATA / "val.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data) - T - 1, size=n_windows)
    out = []
    with torch.no_grad():
        for s in starts:
            x = torch.tensor(np.asarray(data[s:s + T], dtype=np.int64))[None]
            out.append(-float(model(x, labels=x).loss))
    return out


def suite(model, tok, n_items: int = 200) -> dict:
    _, held = split_problems(2)
    comps = {}
    for name, op in (("add_greedy", "+"), ("sub_greedy", "-")):
        probs = problems_from(held, ".", 2, op)[:n_items]
        texts = sample_texts(model, tok, probs, 1, 0.0)
        comps[name] = {"kind": "task", "items": [float(correct(p, *t[0])) for p, t in zip(probs, texts)]}
    comps["val_nll"] = {"kind": "retention", "items": val_items(model)}
    pins = {"suite": "recipe-r", "n_items": n_items, "val_windows": 64, "T": 256, "split_seed": 1234}
    return {"version": VERSION, "pins": pins, "components": comps,
            "summary": {k: float(np.mean(v["items"])) for k, v in comps.items()}}


def train_problems(n: int, seed: int) -> list:
    _, held = split_problems(2)
    return make_problems(n, 2, ops="+-", tags=".", seed=seed, exclude=held)


def stage_sft(model, tok, steps: int = 600, seed: int = 0, lr: float = 1e-3, ledger: Ledger | None = None) -> dict:
    ex = [example(tok, p) for p in train_problems(20000, seed + 300)]
    return train_sft(model, ex, steps, batch=64, lr=lr, seed=seed, ledger=ledger)


def stage_dpo(model, tok, steps: int = 150, seed: int = 0, lr: float = 5e-5, ledger: Ledger | None = None) -> dict:
    import copy
    from frontierlab.pipeline import dpo as D
    probs = train_problems(4000, seed + 400)
    cands = sample_texts(model, tok, probs, 4, 1.0, seed, ledger=ledger)
    pairs = []
    for p, cs in zip(probs, cands):
        good = [c for c in cs if correct(p, *c)]
        bad = [c for c in cs if not correct(p, *c)]
        if good and bad:
            pairs.append((example(tok, p, *good[0]), example(tok, p, *bad[0])))
    ref = copy.deepcopy(model).eval()
    out = D.train_dpo(model, ref, pairs, steps, beta=0.1, nll_coef=0.2, lr=lr, seed=seed, ledger=ledger)
    out["pairs"] = len(pairs)
    return out


def stage_rlvr(model, tok, steps: int = 60, prompts: int = 16, group: int = 8, lr: float = 3e-5, seed: int = 0,
               control: str = "none", ledger: Ledger | None = None) -> dict:
    """A short GRPO (group-mean baseline, group std, token-mean loss, one update per batch, so the ratio is 1)
    with the exact-match reward; ``control="random"`` replaces the reward by Bernoulli(0.5)."""
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    N, t0, hist = n_params(model), time.perf_counter(), []
    for step in range(1, steps + 1):
        probs = train_problems(prompts, 10_000 * (seed + 1) + step)
        by = _buckets(tok, probs)
        loss_parts, n_tok, passes = [], 0.0, []
        opt.zero_grad(set_to_none=True)
        for L, rows in by.items():
            ids = torch.tensor([r for _, r in rows for _ in range(group)], dtype=torch.long)
            ro = sample(model, ids, 5, 1.0, gen)
            texts = [(tok.decode(r.tolist()), bool(f)) for r, f in zip(ro.response, ro.finished)]
            rew = torch.tensor([float(correct(probs[i], *texts[j * group + g]))
                                for j, (i, _) in enumerate(rows) for g in range(group)])
            passes.append(float(rew.mean()))
            if control == "random":
                rew = torch.tensor(rng.random(len(rew)) < 0.5, dtype=torch.float32)
            adv = A.group_advantages(rew.view(-1, group), "mean", "group").reshape(-1)
            model.train()
            logp = token_logprobs(model, ro.tokens, ro.prompt_len)
            loss, _ = Lo.policy_loss(logp, logp.detach(), adv, ro.mask, aggregation="token_mean")
            w = float(ro.mask.sum())
            (loss * w).backward()
            n_tok += w
            if ledger is not None:
                toks = float(ro.tokens.numel())
                ledger.forward("student_sample", N, toks)
                ledger.train("student_train", N, toks)
        for p_ in model.parameters():
            if p_.grad is not None:
                p_.grad /= max(1.0, n_tok)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        hist.append({"step": step, "pass": float(np.mean(passes))})
    return {"steps": steps, "history": hist[-5:], "seconds": round(time.perf_counter() - t0, 1)}
