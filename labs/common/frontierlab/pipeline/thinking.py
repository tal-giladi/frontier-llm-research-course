"""Thinking modes, budgets and routing in the toy world (lesson 13.4).

**The task.** Three-digit addition. A prompt is ``<mode><a:03d>+<b:03d>=`` with mode ``h`` (think, like
Qwen3's ``/think``) or ``n`` (no think, ``/no_think``). Every response has a thinking block that ends with
``#`` (the toy's ``</think>``) and then the answer, as in Qwen3's chat template, where a non-thinking
response carries an *empty* think block (Qwen3 report Table 9):

==========  =========================  ================================================
mode        response for 478+365       meaning
==========  =========================  ================================================
``h``       ``a13b14c08#843``          column sums with carries (units ``a``, tens ``b``,
                                       hundreds ``c``), then ``#``, then the answer
``n``       ``#843``                   empty thinking, answer at once
==========  =========================  ================================================

One model is trained on both (thinking-mode fusion in miniature, Qwen3 section 4.3).

**Budget forcing** (:func:`answer_with_budget`): let the model think for at most B tokens; if it has not
closed its thinking block, append ``#`` and let it answer from what it has (Qwen3's thinking budget:
"when the length of the model's thinking reaches a user-defined threshold, we manually halt the thinking
process and insert the stop-thinking instruction", section 4.3).

**Routing** (:func:`route_curve`): a router reads the prompt and sends it to thinking or non-thinking mode.
Cost is counted in generated tokens; quality is exact-match accuracy. GPT-5's disclosed design routes
between a fast model and a thinking model with a router "continuously trained on real signals, including
when users switch models, preference rates for responses, and measured correctness" (system card section 1;
company claim). The toy router is trained on one such signal: whether the non-thinking answer was right.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from frontierlab.posttrain.tokenizer import BOS, EOS, PAD, TOK

END_THINK = "#"
COLS = "abc"


@dataclass(frozen=True)
class AddProblem:
    a: int
    b: int
    mode: str = "h"          # "h" think, "n" no think

    @property
    def prompt(self) -> str:
        return f"{self.mode}{self.a:03d}+{self.b:03d}="

    @property
    def answer(self) -> str:
        return str(self.a + self.b)

    @property
    def trace(self) -> str:
        """Column sums from the units up, each with the incoming carry, two digits each."""
        out, carry = [], 0
        for i in range(3):
            s = (self.a // 10 ** i) % 10 + (self.b // 10 ** i) % 10 + carry
            out.append(f"{COLS[i]}{s:02d}")
            carry = s // 10
        return "".join(out)

    @property
    def target(self) -> str:
        return (self.trace if self.mode == "h" else "") + END_THINK + self.answer

    @property
    def carries(self) -> int:
        n, carry = 0, 0
        for i in range(3):
            s = (self.a // 10 ** i) % 10 + (self.b // 10 ** i) % 10 + carry
            carry = s // 10
            n += carry
        return n

    def with_mode(self, mode: str) -> "AddProblem":
        return AddProblem(self.a, self.b, mode)


def split(eval_frac: float = 0.1, seed: int = 2024) -> tuple[list, list]:
    """All (a, b) pairs split into training and held-out sets by a seeded draw."""
    rng = np.random.default_rng(seed)
    pairs = [(a, b) for a in range(1000) for b in range(1000)]
    held = rng.random(len(pairs)) < eval_frac
    return [p for p, h in zip(pairs, held) if not h], [p for p, h in zip(pairs, held) if h]


def problems(pairs, n: int, seed: int, mode: str | None = None) -> list:
    """``n`` problems from ``pairs``; mode None = each mode with probability 1/2."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pairs), size=n)
    modes = rng.choice(["h", "n"], size=n) if mode is None else [mode] * n
    return [AddProblem(*pairs[i], str(m)) for i, m in zip(idx, modes)]


def encode_prompts(probs: list) -> torch.Tensor:
    return torch.tensor([[BOS] + TOK.encode(p.prompt) for p in probs], dtype=torch.long)


def examples(probs: list):
    from frontierlab.pipeline.seqs import Example
    return [Example.of([BOS] + TOK.encode(p.prompt), TOK.encode(p.target) + [EOS]) for p in probs]


def parse(text: str) -> tuple[str, str | None]:
    """(thinking, answer); answer None if the thinking block never closed."""
    if END_THINK not in text:
        return text, None
    think, ans = text.split(END_THINK, 1)
    return think, ans


@torch.no_grad()
def _greedy(model, ids: torch.Tensor, steps: int, cache=None, logits=None):
    """Greedy continuation of a batch with the KV cache; returns (tokens (B, steps), cache, last logits)."""
    if cache is None:
        cache = model.new_cache()
        logits = model(ids, cache=cache).logits[:, -1]
    out = []
    done = torch.zeros(ids.shape[0], dtype=torch.bool)
    for _ in range(steps):
        nxt = torch.where(done, torch.full_like(logits[:, 0], PAD, dtype=torch.long), logits.argmax(-1))
        out.append(nxt)
        done |= nxt == EOS
        logits = model(nxt[:, None], cache=cache).logits[:, -1]
    return torch.stack(out, 1) if out else torch.zeros((ids.shape[0], 0), dtype=torch.long), cache, logits


@torch.no_grad()
def answer_with_budget(model, probs: list, budget: int | None, max_answer: int = 6) -> list[dict]:
    """Greedy responses with at most ``budget`` thinking tokens (None = unlimited, up to 12).

    Rows are decoded one by one after a batched pass: a row that closed its thinking block within the
    budget keeps its own answer; a row that did not gets ``#`` appended and a fresh greedy answer.
    Returns per problem: text, correct, think_tokens, total_tokens (generated, EOS included), forced."""
    model.eval()
    limit = 12 if budget is None else budget
    out: list = [None] * len(probs)
    toks, _, _ = _greedy(model, encode_prompts(probs), limit + max_answer + 1)
    to_force: dict = {}
    for i, (p, row) in enumerate(zip(probs, toks.tolist())):
        decoded = TOK.decode(row)                                   # stops at EOS
        think, ans = parse(decoded)
        if ans is not None and len(think) <= limit:
            out[i] = {"text": decoded, "correct": ans == p.answer, "think_tokens": len(think),
                      "total_tokens": len(decoded) + 1, "forced": False}
        else:                                                       # keep `limit` thinking tokens, then force '#'
            t = think[:limit]
            to_force.setdefault(len(t), []).append((i, t))
    for n, rows in to_force.items():                                # one batch per kept-thinking length
        ids = torch.tensor([[BOS] + TOK.encode(probs[i].prompt) + TOK.encode(t) + TOK.encode(END_THINK)
                            for i, t in rows], dtype=torch.long)
        ans_ids, _, _ = _greedy(model, ids, max_answer)
        for (i, t), a in zip(rows, ans_ids.tolist()):
            ans = TOK.decode(a)
            out[i] = {"text": t + END_THINK + ans, "correct": ans == probs[i].answer, "think_tokens": n,
                      "total_tokens": n + 1 + len(ans) + 1, "forced": True}
    return out


def route_curve(p_easy: np.ndarray, correct_nothink: np.ndarray, correct_think: np.ndarray,
                tokens_nothink: np.ndarray, tokens_think: np.ndarray, thresholds=None) -> list[dict]:
    """Accuracy and mean generated tokens when a prompt goes to thinking mode iff the router's estimate that
    the non-thinking answer is right, ``p_easy``, is below a threshold."""
    thresholds = np.linspace(0, 1.0001, 21) if thresholds is None else thresholds
    rows = []
    for th in thresholds:
        think = p_easy < th
        acc = np.where(think, correct_think, correct_nothink).mean()
        tok = np.where(think, tokens_think, tokens_nothink).mean()
        rows.append({"threshold": float(th), "think_frac": float(think.mean()), "accuracy": float(acc),
                     "tokens": float(tok)})
    return rows


def oracle_route(correct_nothink, correct_think, tokens_nothink, tokens_think) -> dict:
    """Think exactly when the non-thinking answer is wrong and thinking is right (the best any router can do)."""
    think = (~correct_nothink.astype(bool)) & correct_think.astype(bool)
    return {"think_frac": float(think.mean()),
            "accuracy": float(np.where(think, correct_think, correct_nothink).mean()),
            "tokens": float(np.where(think, tokens_think, tokens_nothink).mean())}


def random_route(frac: float, correct_nothink, correct_think, tokens_nothink, tokens_think, seed: int = 0) -> dict:
    """Send a random ``frac`` of prompts to thinking: the baseline a router has to beat at equal cost."""
    rng = np.random.default_rng(seed)
    think = rng.random(len(correct_think)) < frac
    return {"think_frac": float(think.mean()),
            "accuracy": float(np.where(think, correct_think, correct_nothink).mean()),
            "tokens": float(np.where(think, tokens_think, tokens_nothink).mean())}
