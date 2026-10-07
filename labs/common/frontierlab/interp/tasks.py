"""Behaviours with a known mechanism, and prompt sets for open models (Module 17).

A causal-claim method is only trustworthy if it recovers a mechanism we already know. The course's
"known answer" is **induction** (Olsson et al. 2022): on a sequence ``... A B ... A`` a two-layer
transformer predicts ``B``. A layer-0 *previous-token head* writes "the token before me was A" into
position ``B``; a layer-1 *induction head* uses that as its key (K-composition), attends from the second
``A`` to ``B`` and copies ``B``. :func:`train_induction_model` trains Baseline-0's architecture at toy size
on repeated random sequences until it does this (about a minute on a laptop).

Clean/corrupt pairs (:func:`induction_pairs`) change one token so that the correct answer changes:

* ``"value"`` corruption: the first ``B`` becomes ``C``; the right answer becomes ``C``.
* ``"key"`` corruption: the first ``A`` becomes ``X``; the second ``A`` no longer has a match, so the
  model loses its reason to say ``B``. This is the corruption that tests K-composition.

The metric is the logit difference ``logit(B) - logit(C)`` at the second ``A`` (for the key corruption,
``C`` is a fixed distractor token that appears nowhere in the sequence).

For Hugging Face models the module also builds the indirect-object-identification prompts (Wang et al.
2022) and the A/B sycophancy items of lesson 17.4, split by template family so that a claim chosen on
one family can be tested on another.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import torch

from frontierlab.model import LM, ModelConfig


# --------------------------------------------------------------------------------------------- induction

def induction_config(vocab: int = 96) -> ModelConfig:
    """Two-layer Baseline-0 architecture (pre-norm, RoPE, QK-norm, SwiGLU), 4 heads of 16, no GQA."""
    return ModelConfig(vocab_size=vocab, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                       num_key_value_heads=4, head_dim=16, intermediate_size=128, max_position_embeddings=256)


def repeated_batch(B: int, half: int, vocab: int, gen: torch.Generator, reserved: int = 2, max_gap: int = 8,
                   gap: int | None = None):
    """(B, 2·half + max_gap) sequences: a segment of ``half`` distinct random tokens, ``g`` filler tokens,
    the same segment again, then ``max_gap - g`` filler tokens. ``g`` is random per row (or fixed to ``gap``),
    so the distance between the two copies varies and a head that just looks a fixed number of positions
    back cannot solve the task; only matching content (induction) can. Tokens ``0..reserved-1`` are never
    used (kept as distractors); filler tokens never occur in the segment. Returns (x, g) with g (B,)."""
    T = 2 * half + max_gap
    xs, gs = [], []
    for _ in range(B):
        perm = torch.randperm(vocab - reserved, generator=gen) + reserved
        seg, rest = perm[:half], perm[half:]
        g = int(torch.randint(0, max_gap + 1, (1,), generator=gen)) if gap is None else gap
        fill = rest[torch.randint(0, len(rest), (T - 2 * half,), generator=gen)]
        xs.append(torch.cat([seg, fill[:g], seg, fill[g:]]))
        gs.append(g)
    return torch.stack(xs), torch.tensor(gs)


def train_induction_model(steps: int = 600, half: int = 12, batch: int = 64, lr: float = 3e-3, seed: int = 0,
                          vocab: int = 96, max_gap: int = 8, path: str | Path | None = None, log_every: int = 100,
                          verbose: bool = True) -> LM:
    """Train the induction model (or load it from ``path`` if it exists). The loss counts only positions whose
    next token is in the second copy of the segment (predictable by induction); it falls close to 0 once the
    model has learnt induction (about 300 steps)."""
    if path is not None and Path(path).exists():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = LM(ModelConfig(**ck["config"]))
        m.load_state_dict(ck["model"])
        return m.eval()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    m = LM(induction_config(vocab))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    t_idx = torch.arange(2 * half + max_gap - 1)
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, step / 50) * (0.1 + 0.9 * (1 - step / steps))
        x, gaps = repeated_batch(batch, half, vocab, gen, max_gap=max_gap)
        out = m(x, labels=x)
        lo = (half + gaps)[:, None]
        mask = (t_idx[None] >= lo) & (t_idx[None] <= lo + half - 2)
        loss = (out.per_token_loss * mask).sum() / mask.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if verbose and (step % log_every == 0 or step == 1):
            print(f"induction step {step:5d}  induction loss {loss.item():.4f}")
    m.eval()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        from dataclasses import asdict
        torch.save({"model": m.state_dict(), "config": asdict(m.config)}, path)
    return m


@dataclass
class Pairs:
    """Clean and corrupt token ids (B, T), the position whose next-token logits are read, and the two
    answer ids per row: ``good`` (right on clean) and ``bad`` (right on corrupt, or a distractor)."""
    clean: torch.Tensor
    corrupt: torch.Tensor
    pos: int
    good: torch.Tensor
    bad: torch.Tensor
    meta: dict = field(default_factory=dict)


def induction_pairs(n: int, half: int = 12, vocab: int = 96, kind: str = "value", query: int | None = None,
                    gap: int = 4, max_gap: int = 8, seed: int = 0) -> Pairs:
    """``n`` clean/corrupt pairs with the same gap ``gap`` in every row (so one query position serves the
    batch). The query position is the second occurrence of ``A = x[query]``; the answer is ``B = x[query+1]``.
    A ``gap`` above the training ``max_gap`` is a held-out distance."""
    gen = torch.Generator().manual_seed(seed)
    x, _ = repeated_batch(n, half, vocab, gen, max_gap=max(max_gap, gap), gap=gap)
    q = half // 2 if query is None else query
    pos = half + gap + q
    clean, corrupt = x.clone(), x.clone()
    good = clean[:, q + 1].clone()
    if kind == "value":                            # first B -> C (a token not in the sequence)
        bad = torch.stack([_absent(row, vocab, gen) for row in x])
        corrupt[:, q + 1] = bad
    elif kind == "key":                            # first A -> X; B is no longer predicted
        corrupt[:, q] = torch.stack([_absent(row, vocab, gen) for row in x])
        bad = torch.zeros_like(good)               # distractor: reserved token 0 never occurs
    else:
        raise ValueError("kind is 'value' or 'key'")
    return Pairs(clean, corrupt, pos, good, bad, {"kind": kind, "query": q, "half": half, "gap": gap})


def _absent(row: torch.Tensor, vocab: int, gen: torch.Generator, reserved: int = 2) -> torch.Tensor:
    present = set(row.tolist())
    while True:
        t = int(torch.randint(reserved, vocab, (1,), generator=gen))
        if t not in present:
            return torch.tensor(t)


# --------------------------------------------------------------------------------------------- IOI (HF)

NAMES = [" Mary", " John", " Alice", " Bob", " Sarah", " David", " Emma", " James", " Laura", " Paul",
         " Kate", " Mark", " Anna", " Peter", " Lucy", " Tom", " Rachel", " Daniel", " Julia", " Mike"]
OBJECTS = [" drink", " book", " ball", " letter", " key", " gift", " pen", " ticket"]
PLACES = [" store", " park", " school", " office", " station", " beach", " market", " library"]

# Two template families: the claim is chosen on "train" and tested on "heldout" (other wording).
IOI_TEMPLATES = {
    "train": ["When{A} and{B} went to the{P},{S} gave a{O} to",
              "After{A} and{B} arrived at the{P},{S} handed a{O} to",
              "While{A} and{B} were at the{P},{S} passed a{O} to"],
    "heldout": ["Then,{A} and{B} had a long day at the{P}.{S} decided to give a{O} to",
                "Yesterday{A} and{B} visited the{P}, and{S} offered a{O} to"],
}


def ioi_prompts(tokenizer, family: str = "train", n: int = 64, seed: int = 0, names=None):
    """IOI prompts for a Hugging Face tokenizer. Returns a dict with ``clean`` / ``corrupt`` (B, T) ids
    (same length within the batch), ``good`` (the indirect object), ``bad`` (the subject) and ``pos`` (-1).

    Corruption is the ABC distribution of Wang et al.: the repeated subject is replaced by a third name,
    so the prompt no longer says who is the indirect object; patching clean activations into it restores
    the behaviour only through the components that carry the "which name was repeated" information.
    Only names that are a single token for this tokenizer are used; prompts whose clean and corrupt
    versions tokenize to different lengths are dropped."""
    rng = random.Random(seed)
    pool = [nm for nm in (names or NAMES) if len(tokenizer(nm, add_special_tokens=False)["input_ids"]) == 1]
    if len(pool) < 3:
        raise ValueError("need at least 3 single-token names for this tokenizer")
    rows, temps = [], IOI_TEMPLATES[family]
    tries = 0
    while len(rows) < n and tries < 50 * n:
        tries += 1
        t = rng.choice(temps)
        a, b, c = rng.sample(pool, 3)
        io, s = (a, b) if rng.random() < 0.5 else (b, a)
        p, o = rng.choice(PLACES), rng.choice(OBJECTS)
        clean = t.format(A=a, B=b, S=s, P=p, O=o)
        corrupt = t.format(A=a, B=b, S=c, P=p, O=o)
        ci = tokenizer(clean, add_special_tokens=False)["input_ids"]
        ki = tokenizer(corrupt, add_special_tokens=False)["input_ids"]
        if len(ci) != len(ki):
            continue
        rows.append((ci, ki, tokenizer(io, add_special_tokens=False)["input_ids"][0],
                     tokenizer(s, add_special_tokens=False)["input_ids"][0]))
    by_len: dict[int, list] = {}
    for r in rows:
        by_len.setdefault(len(r[0]), []).append(r)
    L = max(by_len, key=lambda k: len(by_len[k]))          # keep the most common length: one batch, no padding
    keep = by_len[L]
    return {"clean": torch.tensor([r[0] for r in keep]), "corrupt": torch.tensor([r[1] for r in keep]),
            "good": torch.tensor([r[2] for r in keep]), "bad": torch.tensor([r[3] for r in keep]), "pos": -1,
            "family": family}


def as_pairs(d: dict) -> Pairs:
    return Pairs(d["clean"], d["corrupt"], d["pos"], d["good"], d["bad"], {"family": d.get("family")})


# --------------------------------------------------------------------------------------------- the Module 17 LM

M17_RUN = Path("runs/m17/lm")


def m17_model(run: str | Path = M17_RUN, steps: int = 600) -> LM:
    """The Module 17 language model: Baseline-0's architecture at toy size (4 layers, width 128) trained on
    Data-v0 with the course loop for ``steps`` steps of 16 × 256 tokens (2.5M tokens; about 8 minutes on an
    idle laptop). Trained if ``run`` has no finished checkpoint; an interrupted run resumes exactly."""
    run = Path(run)
    ck_path = run / "checkpoint.pt"
    if not ck_path.exists() or torch.load(ck_path, weights_only=False)["step"] < steps:
        from frontierlab.train.loop import main as train
        train(["--run", str(run), "--preset", "toy", "--steps", str(steps), "--batch", "16", "--seq", "256",
               "--eval-every", "300", "--ckpt-every", "100", "--question", "Module 17 model for SAEs and graphs"])
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    m = LM(ModelConfig(**ck["config"]))
    m.load_state_dict(ck["model"])
    return m.eval()


def data_windows(split: str, n: int, T: int, seed: int) -> torch.Tensor:
    """(n, T) Data-v0 windows: random for ``train``, the fixed evaluation windows otherwise."""
    from frontierlab.data.loader import TokenData
    d = TokenData(split)
    if split == "train":
        return d.batch(n, T, torch.Generator().manual_seed(seed))
    return torch.stack([d.window(s, T) for s in d.eval_windows(n, T, seed)])
