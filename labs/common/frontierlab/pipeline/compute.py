"""Compute accounting for post-training pipelines (Module 13).

Every comparison in Module 13 counts *all* the compute an arm used, not only the student's training
step: a distillation arm pays for the teacher, a judge-trained arm pays for the judge, an RL arm pays
for sampling. The usual dense-transformer estimates (Kaplan et al. 2020, section 2.1):

* forward pass:                     2 N FLOPs per token
* forward + backward:               6 N FLOPs per token
* sampling one token with a cache:  about 2 N FLOPs (plus attention over the context, small here)

N is the number of parameters a token's activations are multiplied by. For the toy models it is the
full parameter count (the tied 35 x d embedding is under 2% of it). For a Hugging Face model with a tied
embedding the lookup is free but the output projection uses the same matrix, so the full count is again
right; :func:`hf_matmul_params` handles the untied case.

:class:`Ledger` collects (category, FLOPs) entries so a lab can print one line per arm. Categories:
``student_train``, ``student_sample``, ``reference`` (frozen reference forward passes in DPO),
``teacher_sample``, ``teacher_score``, ``judge_label`` (producing the judge's training labels),
``judge_train``, ``judge_score``, ``router``.
"""

from __future__ import annotations

from collections import defaultdict

CATEGORIES = ("student_train", "student_sample", "reference", "teacher_sample", "teacher_score",
              "judge_label", "judge_train", "judge_score", "router")


def n_params(model) -> int:
    """All parameters, each tied tensor counted once."""
    seen, n = set(), 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            n += p.numel()
    return n


def hf_matmul_params(n_total: int, vocab: int, d_model: int, tied: bool = True) -> int:
    """Parameters that multiply each token's activations. Tied embedding: the total already counts the
    table once and the output projection uses it, so N = n_total. Untied: the input table is a lookup,
    N = n_total - vocab * d_model."""
    return n_total if tied else n_total - vocab * d_model


def forward_flops(n: int, tokens: float) -> float:
    return 2.0 * n * tokens


def train_flops(n: int, tokens: float) -> float:
    return 6.0 * n * tokens


class Ledger:
    """A running tally of FLOPs (and tokens) by category."""

    def __init__(self):
        self.flops = defaultdict(float)
        self.tokens = defaultdict(float)

    def add(self, category: str, flops: float, tokens: float = 0.0) -> float:
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}; use one of {CATEGORIES}")
        self.flops[category] += float(flops)
        self.tokens[category] += float(tokens)
        return float(flops)

    def forward(self, category: str, n: int, tokens: float) -> float:
        return self.add(category, forward_flops(n, tokens), tokens)

    def train(self, category: str, n: int, tokens: float) -> float:
        return self.add(category, train_flops(n, tokens), tokens)

    def total(self, exclude: tuple = ()) -> float:
        return float(sum(v for k, v in self.flops.items() if k not in exclude))

    def merge(self, other: "Ledger") -> "Ledger":
        for k, v in other.flops.items():
            self.flops[k] += v
        for k, v in other.tokens.items():
            self.tokens[k] += v
        return self

    def to_dict(self) -> dict:
        return {"flops": dict(self.flops), "tokens": dict(self.tokens), "total": self.total()}

    @staticmethod
    def from_dict(d: dict) -> "Ledger":
        led = Ledger()
        for k, v in d.get("flops", {}).items():
            led.flops[k] = v
        for k, v in d.get("tokens", {}).items():
            led.tokens[k] = v
        return led

    def line(self) -> str:
        parts = [f"{k} {v:.2e}" for k, v in sorted(self.flops.items()) if v]
        return f"total {self.total():.2e} FLOPs (" + ", ".join(parts) + ")"
