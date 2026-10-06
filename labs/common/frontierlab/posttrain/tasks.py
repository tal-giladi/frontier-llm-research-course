"""The toy verifiable world of Module 12's free CPU path.

**Arithmetic with instruction tags** (lessons 12.2-12.4, the project). A prompt is ``<tag><a><op><b>=``
with zero-padded ``D``-digit operands, for example ``.07+35=``. The response is the result followed by
EOS, written the way the tag asks:

====  ==========================================  ======================
tag   instruction                                 response for 07+35
====  ==========================================  ======================
``.`` plain answer                                ``42``
``P`` zero-pad the answer to D+1 digits           ``042``
``Q`` wrap the answer in ``#`` quotes             ``#42#``
``E`` end the answer with ``!``                   ``42!``
====  ==========================================  ======================

``op`` is ``+`` or ``-`` (subtraction always has ``a >= b``). RL trains on plain addition only; Eval
Suite v2 measures what happens to the rest (subtraction = capability retention, tags = instruction
following).

Verifiers (lesson 12.1): :func:`strict_verify` (finished with EOS and the text equals the answer),
:func:`last_number_verify` (the last number in the text, the usual "extract the final answer" rule) and
:func:`lenient_verify` (the answer appears anywhere: the kind of check that gets hacked).

**Letter strings with a hidden gold reward** (lesson 12.1). A known base sampler writes strings of
letters; :func:`gold_reward` is the "true human preference" that the lab hides from the reward model.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import torch

from frontierlab.posttrain.tokenizer import BOS, EOS, PAD, TOK

TAGS = ".PQE"


# --------------------------------------------------------------------------- arithmetic

@dataclass(frozen=True)
class Problem:
    tag: str
    a: int
    op: str
    b: int
    digits: int

    @property
    def result(self) -> int:
        return self.a + self.b if self.op == "+" else self.a - self.b

    @property
    def prompt(self) -> str:
        return f"{self.tag}{self.a:0{self.digits}d}{self.op}{self.b:0{self.digits}d}="

    @property
    def target(self) -> str:
        r = self.result
        return {".": str(r), "P": f"{r:0{self.digits + 1}d}", "Q": f"#{r}#", "E": f"{r}!"}[self.tag]


def make_problems(n: int, digits: int = 2, ops: str = "+", tags: str = ".", seed: int = 0,
                  exclude: set | None = None) -> list[Problem]:
    """``n`` problems drawn uniformly from the (op, tag, a, b) grid, from a seed; ``exclude`` = a set of
    (a, op, b) triples that must not appear (keeps RL training prompts disjoint from evaluation prompts)."""
    rng = np.random.default_rng(seed)
    out, hi = [], 10 ** digits
    while len(out) < n:
        a, b = int(rng.integers(hi)), int(rng.integers(hi))
        op = ops[int(rng.integers(len(ops)))]
        tag = tags[int(rng.integers(len(tags)))]
        if op == "-" and a < b:
            a, b = b, a
        if exclude and (a, op, b) in exclude:
            continue
        out.append(Problem(tag, a, op, b, digits))
    return out


def split_problems(digits: int = 2, eval_frac: float = 0.2, seed: int = 1234) -> tuple[set, set]:
    """Split every (a, op, b) triple into train and held-out sets by a seeded hash: no prompt leaks."""
    hi = 10 ** digits
    rng = np.random.default_rng(seed)
    train, held = set(), set()
    for op in "+-":
        for a in range(hi):
            for b in range(hi):
                if op == "-" and a < b:
                    continue
                (held if rng.random() < eval_frac else train).add((a, op, b))
    return train, held


def problems_from(triples, tags: str = ".", digits: int = 2, ops: str = "+") -> list[Problem]:
    """Every triple with an op in ``ops`` under every tag in ``tags``, in a fixed order (an evaluation set)."""
    return [Problem(t, a, op, b, digits) for (a, op, b) in sorted(triples) if op in ops for t in tags]


def encode_prompts(problems: list[Problem]) -> torch.Tensor:
    """(B, P) int64: BOS + prompt characters. All prompts of one digit count have the same length."""
    rows = [[BOS] + TOK.encode(p.prompt) for p in problems]
    if len({len(r) for r in rows}) != 1:
        raise ValueError("prompts must have equal length (same digit count)")
    return torch.tensor(rows, dtype=torch.long)


def encode_sft(problems: list[Problem], max_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forcing batch for SFT: ids (B, T) = BOS prompt target EOS PAD..., and ``loss_mask`` (B, T)
    that is 1 exactly on the target and EOS tokens (the positions the loss predicts)."""
    rows, masks = [], []
    for p in problems:
        pr, tg = [BOS] + TOK.encode(p.prompt), TOK.encode(p.target) + [EOS]
        rows.append(pr + tg)
        masks.append([0] * len(pr) + [1] * len(tg))
    T = max_len or max(len(r) for r in rows)
    ids = torch.full((len(rows), T), PAD, dtype=torch.long)
    mask = torch.zeros((len(rows), T), dtype=torch.long)
    for i, (r, m) in enumerate(zip(rows, masks)):
        ids[i, :len(r)] = torch.tensor(r)
        mask[i, :len(m)] = torch.tensor(m)
    return ids, mask


# --------------------------------------------------------------------------- verifiers

def response_text(ids: torch.Tensor) -> tuple[str, bool]:
    """(text up to EOS, finished?) for one response row."""
    row = ids.tolist()
    finished = EOS in row
    return TOK.decode(row), finished


def strict_verify(text: str, finished: bool, problem: Problem) -> bool:
    """Finished with EOS and the text is exactly the formatted answer."""
    return finished and text == problem.target


def last_number_verify(text: str, finished: bool, problem: Problem) -> bool:
    """The last run of digits in the text equals the answer (formatting and termination ignored)."""
    nums = re.findall(r"\d+", text)
    return bool(nums) and int(nums[-1]) == problem.result


def lenient_verify(text: str, finished: bool, problem: Problem) -> bool:
    """The answer's digits appear anywhere in the text. Easy to satisfy by writing many digits."""
    return str(problem.result) in text


VERIFIERS = {"strict": strict_verify, "last_number": last_number_verify, "lenient": lenient_verify}


def score(responses: torch.Tensor, problems: list[Problem], verifier: str = "strict") -> torch.Tensor:
    """(N,) float32 rewards in {0, 1} for responses (N, R) against ``problems`` (length N)."""
    fn = VERIFIERS[verifier]
    out = []
    for row, p in zip(responses, problems):
        text, fin = response_text(row)
        out.append(float(fn(text, fin, p)))
    return torch.tensor(out, dtype=torch.float32)


# --------------------------------------------------------------------------- instruction checks (Eval v2)

def follows_instruction(text: str, finished: bool, problem: Problem) -> bool:
    """Format only, like IFEval: does the response obey the tag, whatever the number?"""
    if not finished:
        return False
    if problem.tag == ".":
        return bool(re.fullmatch(r"\d+", text))
    if problem.tag == "P":
        return bool(re.fullmatch(r"\d{%d}" % (problem.digits + 1), text))
    if problem.tag == "Q":
        return bool(re.fullmatch(r"#\d+#", text))
    if problem.tag == "E":
        return bool(re.fullmatch(r"\d+!", text))
    raise ValueError(problem.tag)


# --------------------------------------------------------------------------- letter strings (12.1)

LETTERS = "abcdefghijklmn"


@dataclass
class LetterWorld:
    """A known base sampler over letter strings and the hidden gold reward of lesson 12.1.

    Base sampler: length L in 2..``max_len`` with P(L) proportional to exp(-(L - 5)^2 / 8); letters from a
    seeded first-order Markov chain. Gold reward of a string y:

        gold(y) = sum over distinct letters c in y of value[c] + 0.4 * min(L, 8) - 0.8 * max(0, L - 8)

    In the base distribution long strings are rare, so "longer is better" holds almost everywhere the
    reward model sees data, and fails past 8 letters, where the base sampler rarely goes.
    """
    seed: int = 0
    max_len: int = 16

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        k = len(LETTERS)
        self.trans = rng.dirichlet(np.full(k, 0.5), size=k)
        self.first = rng.dirichlet(np.full(k, 1.0))
        self.value = rng.uniform(-0.5, 1.0, size=k)
        Ls = np.arange(2, self.max_len + 1)
        p = np.exp(-((Ls - 5.0) ** 2) / 8.0)
        self.len_values, self.len_probs = Ls, p / p.sum()

    def sample(self, n: int, rng: np.random.Generator) -> list[str]:
        Ls = rng.choice(self.len_values, size=n, p=self.len_probs)
        out = []
        for L in Ls:
            c = rng.choice(len(LETTERS), p=self.first)
            s = [c]
            for _ in range(L - 1):
                c = rng.choice(len(LETTERS), p=self.trans[c])
                s.append(c)
            out.append("".join(LETTERS[i] for i in s))
        return out

    def gold(self, y: str) -> float:
        L = len(y)
        distinct = {LETTERS.index(ch) for ch in y}
        return float(sum(self.value[i] for i in distinct) + 0.4 * min(L, 8) - 0.8 * max(0, L - 8))

    def preference(self, y1: str, y2: str, rng: np.random.Generator, temperature: float = 0.5) -> int:
        """A noisy annotator: returns 1 if y1 is preferred, with P = sigmoid((gold1 - gold2) / temperature)."""
        z = (self.gold(y1) - self.gold(y2)) / temperature
        return int(rng.random() < 1.0 / (1.0 + math.exp(-z)))


def encode_strings(strings: list[str], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, max_len + 2) ids BOS s EOS PAD... and the index of each EOS (where a reward model reads its score)."""
    ids = torch.full((len(strings), max_len + 2), PAD, dtype=torch.long)
    last = torch.empty(len(strings), dtype=torch.long)
    for i, s in enumerate(strings):
        row = [BOS] + TOK.encode(s) + [EOS]
        ids[i, :len(row)] = torch.tensor(row)
        last[i] = len(row) - 1
    return ids, last
