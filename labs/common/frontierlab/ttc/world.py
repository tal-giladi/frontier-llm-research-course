"""The free-CPU test-time-compute world (lesson 15.1 and the Module 15 project).

**Task.** Four-digit addition with a thinking trace, the Module 13.4 format extended by one column. A prompt is
``h<a:04d>+<b:04d>=``; the response is the column sums from the units up, each with its incoming carry,
then ``#`` (the toy's end-of-thinking), the answer and EOS::

    h4783+3659=  ->  a12b14c14d08#8442

A step is one column (3 tokens: label and two digits). The answer is a deterministic function of a correct
trace, so a wrong answer comes from a wrong step or from a forced early answer.

**Policy.** The 308,400-parameter policy architecture of Modules 12–14 (``posttrain.sft.policy_config``),
trained with SFT on full traces and, for 30% of examples, traces cut after a random number of tokens
(the 13.4 "fusion+budget" recipe), and stopped early so that its steps are sometimes wrong. That makes
every test-time method in the lesson worth something and none of them perfect.

**Budget forcing** (:func:`sample`): at most B thinking tokens; if ``#`` has not appeared, it is inserted
and the model answers (s1's budget forcing; Qwen3's thinking budget; lesson 13.4).

**Verifiers.** Both use the Module 13 judge architecture (``pipeline.judge.Judge``: the policy network with
a scalar head at the last token).

* **ORM** (outcome verifier): scores a full response; trained on policy samples of *training* problems
  labelled by the exact checker (Cobbe et al. 2021's verifier, in miniature).
* **PRM** (process verifier): scores a prefix (prompt + the first j columns); every prefix of a sampled
  trajectory is labelled with that trajectory's final correctness. This is the one-rollout Monte Carlo
  estimate of Math-Shepherd's hard estimation (Wang et al. 2023): a value model, not human step labels.

**Guided search** (:func:`beam_search`): step-level beam search with the PRM, as in Snell et al. (2024,
section 5): sample ``width·expand`` first steps, keep the ``width`` best by PRM score, expand each
``expand`` times, repeat for every column; then each surviving beam writes ``#`` and an answer and the
ORM picks one. Every sampled token and every verifier token is counted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from frontierlab.pipeline.compute import n_params
from frontierlab.posttrain.tokenizer import BOS, EOS, PAD, TOK
from frontierlab.ttc.budget import Spend

COLS = "abcde"
END_THINK = "#"
END_ID = TOK.stoi[END_THINK]
DIGITS = 4


MODES = {"n": 0, "m": 3, "h": 9}          # reasoning effort -> thinking tokens per column (none, short, long)


@dataclass(frozen=True)
class SumProblem:
    a: int
    b: int
    digits: int = DIGITS
    mode: str = "m"

    @property
    def prompt(self) -> str:
        return f"{self.mode}{self.a:0{self.digits}d}+{self.b:0{self.digits}d}="

    def with_mode(self, mode: str) -> "SumProblem":
        return SumProblem(self.a, self.b, self.digits, mode)

    @property
    def step_len(self) -> int:
        return MODES[self.mode]

    @property
    def answer(self) -> str:
        return str(self.a + self.b)

    @property
    def trace(self) -> str:
        """Column sums from the units up. Mode m: ``a12`` (label, two-digit sum with carry); mode h:
        ``a3+9+0=12`` (label, both digits and the incoming carry written out, then the sum); mode n: empty."""
        if self.mode == "n":
            return ""
        out, carry = [], 0
        for i in range(self.digits):
            da, db = (self.a // 10 ** i) % 10, (self.b // 10 ** i) % 10
            s = da + db + carry
            out.append(f"{COLS[i]}{s:02d}" if self.mode == "m" else f"{COLS[i]}{da}+{db}+{carry}={s:02d}")
            carry = s // 10
        return "".join(out)

    @property
    def think_len(self) -> int:
        return self.step_len * self.digits

    def target(self, k: int | None = None) -> str:
        """Trace cut after ``k`` thinking tokens (None = full), then ``#`` and the answer."""
        t = self.trace if k is None else self.trace[:k]
        return t + END_THINK + self.answer

    @property
    def carries(self) -> int:
        n, carry = 0, 0
        for i in range(self.digits):
            carry = ((self.a // 10 ** i) % 10 + (self.b // 10 ** i) % 10 + carry) // 10
            n += carry
        return n


def is_heldout(a: int, b: int) -> bool:
    """About 10% of all (a, b) pairs are held out, by a fixed arithmetic hash (no table of 10^8 pairs)."""
    return (a * 1_000_003 + b * 7_919) % 97 < 10


def make_problems(n: int, seed: int, held: bool, digits: int = DIGITS, mode: str | None = "m") -> list[SumProblem]:
    """``n`` problems from the training (held=False) or held-out pairs; mode None = each mode with prob. 1/3."""
    rng = np.random.default_rng(seed)
    out, hi = [], 10 ** digits
    while len(out) < n:
        a, b = int(rng.integers(hi)), int(rng.integers(hi))
        m = mode if mode is not None else "nmh"[int(rng.integers(3))]
        if is_heldout(a, b) == held:
            out.append(SumProblem(a, b, digits, m))
    return out


def encode_prompts(probs: list[SumProblem]) -> torch.Tensor:
    return torch.tensor([[BOS] + TOK.encode(p.prompt) for p in probs], dtype=torch.long)


def prompt_tokens(p: SumProblem) -> int:
    return 1 + len(p.prompt)


# --------------------------------------------------------------------------------------------- policy

def training_examples(n: int, seed: int, trunc_frac: float = 0.3, digits: int = DIGITS):
    from frontierlab.pipeline.seqs import Example
    rng = np.random.default_rng(seed + 1)
    out = []
    for p in make_problems(n, seed, held=False, digits=digits, mode=None):
        k = int(rng.integers(0, p.think_len)) if (p.think_len and rng.random() < trunc_frac) else None
        out.append(Example.of([BOS] + TOK.encode(p.prompt), TOK.encode(p.target(k)) + [EOS]))
    return out


def train_policy(seed: int = 0, steps: int = 500, batch: int = 128, lr: float = 3e-3, n_train: int = 60000,
                 trunc_frac: float = 0.3):
    from frontierlab.model import LM
    from frontierlab.pipeline.seqs import train_sft
    from frontierlab.posttrain.sft import policy_config
    torch.manual_seed(seed)
    m = LM(policy_config())
    info = train_sft(m, training_examples(n_train, seed, trunc_frac), steps, batch, lr, seed=seed)
    return m.eval(), info


def ensure_policy(out: str | Path = "runs/m15/world/policy.pt", **kw) -> Path:
    """Train the world's policy once (about two minutes on a laptop) and reuse it."""
    from frontierlab.posttrain.sft import save_policy
    out = Path(out)
    if not out.exists():
        t0 = time.perf_counter()
        m, info = train_policy(**kw)
        save_policy(m, out, {"world": "m15-sum4", "train": {k: v for k, v in info.items() if k != "history"},
                             "seconds": round(time.perf_counter() - t0, 1), **kw})
    return out


# --------------------------------------------------------------------------------------------- sampling

def parse(text: str) -> tuple[str, str | None]:
    """(thinking, answer); answer None if thinking never ended or the answer is empty or not a number."""
    if END_THINK not in text:
        return text, None
    think, ans = text.split(END_THINK, 1)
    return think, (ans if ans.isdigit() else None)


@torch.no_grad()
def sample(model, probs: list[SumProblem], n: int = 1, think_budget: int | None = None, temperature: float = 1.0,
           generator: torch.Generator | None = None, chunk: int = 4096) -> list[list[dict]]:
    """``n`` responses per problem with at most ``think_budget`` thinking tokens (None = no limit).

    temperature 0 = greedy. Returns, per problem, ``n`` dicts: text, answer, correct, gen_tokens (tokens the
    model produced or was forced to take, EOS included), think_tokens, forced (``#`` was inserted)."""
    model.eval()
    rep = [p for p in probs for _ in range(n)]
    rows: list[dict] = []
    D = probs[0].digits
    full = max(p.think_len for p in probs)
    max_steps = full + 1 + (D + 1) + 1
    budget = full + 1 if think_budget is None else think_budget
    for s in range(0, len(rep), chunk):
        part = rep[s:s + chunk]
        B = len(part)
        cache = model.new_cache()
        logits = model(encode_prompts(part), cache=cache).logits[:, -1].float()
        toks = torch.full((B, max_steps), PAD, dtype=torch.long)
        thinking = torch.ones(B, dtype=torch.bool)
        done = torch.zeros(B, dtype=torch.bool)
        think_n = torch.zeros(B, dtype=torch.long)
        forced = torch.zeros(B, dtype=torch.bool)
        gen_n = torch.zeros(B, dtype=torch.long)
        for t in range(max_steps):
            if temperature == 0:
                nxt = logits.argmax(-1)
            else:
                nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=generator).squeeze(-1)
            force = thinking & ~done & (think_n >= budget) & (nxt != END_ID)
            nxt = torch.where(force, torch.full_like(nxt, END_ID), nxt)
            forced |= force
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            toks[:, t] = nxt
            gen_n += (~done).long()
            think_n += (thinking & ~done & (nxt != END_ID) & (nxt != EOS)).long()
            thinking &= nxt != END_ID
            done |= nxt == EOS
            if bool(done.all()):
                break
            logits = model(nxt[:, None], cache=cache).logits[:, -1].float()
        for i, p in enumerate(part):
            text, finished = TOK.decode(toks[i].tolist()), bool((toks[i] == EOS).any())
            _, ans = parse(text)
            rows.append({"text": text, "answer": ans if finished else None,
                         "correct": bool(finished and ans == p.answer), "gen_tokens": int(gen_n[i]),
                         "think_tokens": int(think_n[i]), "forced": bool(forced[i]), "finished": finished})
    return [rows[i * n:(i + 1) * n] for i in range(len(probs))]


# --------------------------------------------------------------------------------------------- verifiers

def judge_config():
    from frontierlab.posttrain.sft import policy_config
    return policy_config()


def new_judge(seed: int = 0):
    from frontierlab.pipeline.judge import Judge
    torch.manual_seed(seed)
    return Judge(judge_config())


def encode_scored(probs: list, texts: list[str], finished: list[bool], max_len: int | None = None):
    """(B, L) ids and last-token index; L = the longest row (at most 64, the policy's context)."""
    from frontierlab.pipeline.judge import encode_judged
    if max_len is None:
        max_len = min(64, max(1 + len(p.prompt) + len(t) + int(f) for p, t, f in zip(probs, texts, finished)))
    return encode_judged(probs, texts, finished, max_len=max_len)


def scored_tokens(p: SumProblem, text: str, finished: bool) -> int:
    """Tokens a verifier reads for one candidate: BOS + prompt + response (+ EOS)."""
    return prompt_tokens(p) + len(text) + int(finished)


@torch.no_grad()
def score(judge, probs: list, texts: list[str], finished: list[bool], batch: int = 4096) -> np.ndarray:
    """Verifier probabilities (sigmoid of the judge's logit) for candidates or prefixes."""
    judge.eval()
    out = []
    for s in range(0, len(probs), batch):
        ids, last = encode_scored(probs[s:s + batch], texts[s:s + batch], finished[s:s + batch])
        out.append(torch.sigmoid(judge(ids, last)).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def prefixes(text: str, digits: int = DIGITS, step_len: int = 3) -> list[str]:
    """Column-boundary prefixes of a response's thinking (after 1..D columns), for PRM training and scoring."""
    think, _ = parse(text)
    return [think[:step_len * j] for j in range(1, digits + 1) if len(think) >= step_len * j]


def verifier_data(policy, kind: str, n_problems: int, k: int, seed: int, mode: str = "m", temperature: float = 0.7):
    """(problems, texts, finished, labels) from k samples per training problem, labelled by the checker.
    Sample at the temperature the verifier will be used at: its training distribution should match."""
    g = torch.Generator().manual_seed(seed)
    probs = make_problems(n_problems, seed=seed + 500, held=False, mode=mode)
    samples = sample(policy, probs, n=k, temperature=temperature, generator=g)
    P, T, Fin, Y = [], [], [], []
    for p, rows in zip(probs, samples):
        for r in rows:
            P.append(p); T.append(r["text"]); Fin.append(r["finished"]); Y.append(float(r["correct"]))
            if kind == "prm":
                for pre in prefixes(r["text"], p.digits, p.step_len):
                    P.append(p); T.append(pre); Fin.append(False); Y.append(float(r["correct"]))
    return P, T, Fin, torch.tensor(Y)


def train_verifier(policy, kind: str = "orm", n_problems: int = 4000, k: int = 4, steps: int = 400,
                   batch: int = 256, lr: float = 2e-3, seed: int = 0, mode: str = "m", temperature: float = 0.7):
    """Train an ORM or PRM on the policy's own samples; returns (judge, info with positive rate and cost)."""
    from frontierlab.pipeline.seqs import batches
    P, T, Fin, Y = verifier_data(policy, kind, n_problems, k, seed, mode, temperature)
    ids, last = encode_scored(P, T, Fin)
    judge = new_judge(seed)
    opt = torch.optim.AdamW(judge.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    judge.train()
    t0 = time.perf_counter()
    for step, idx in enumerate(batches(len(Y), batch, steps, gen), start=1):
        for gr in opt.param_groups:
            gr["lr"] = lr * min(1.0, step / 20)
        loss = F.binary_cross_entropy_with_logits(judge(ids[idx], last[idx]), Y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    judge.eval()
    return judge, {"kind": kind, "examples": len(Y), "positive_rate": float(Y.mean()), "steps": steps,
                   "params": n_params(judge), "seconds": round(time.perf_counter() - t0, 1),
                   "label_samples": n_problems * k}


def ensure_verifier(policy, kind: str, out: str | Path, **kw):
    from frontierlab.pipeline.judge import Judge
    out = Path(out)
    if out.exists():
        ck = torch.load(out, map_location="cpu", weights_only=False)
        j = Judge(judge_config())
        j.load_state_dict(ck["model"])
        return j.eval(), ck["info"]
    j, info = train_verifier(policy, kind, **kw)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": j.state_dict(), "info": info}, out)
    return j, info


def rule_verify(p: SumProblem, text: str, finished: bool) -> bool:
    """A programmatic checker: recompute every column from the prompt's digits and the answer from the columns,
    and accept only a response that matches. This toy task happens to allow one, as unit tests do for code or a
    proof checker for Lean; most tasks do not. Its cost is negligible next to a model call (counted as 0)."""
    return finished and text == p.target()


def auc(scores, labels) -> float:
    """Area under the ROC curve (probability a random positive outscores a random negative; ties count 1/2)."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):                      # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# --------------------------------------------------------------------------------------------- guided search

@torch.no_grad()
def _extend(model, rows: list[list[int]], n_tokens: int, temperature: float, gen) -> torch.Tensor:
    """Sample ``n_tokens`` after each row (no cache: rows have different lengths). Returns (B, n_tokens)."""
    B = len(rows)
    L = torch.tensor([len(r) for r in rows])
    ids = torch.full((B, int(L.max()) + n_tokens), PAD, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r)
    out = torch.empty((B, n_tokens), dtype=torch.long)
    ar = torch.arange(B)
    for t in range(n_tokens):
        logits = model(ids[:, :int(L.max())]).logits[ar, L - 1].float()
        if temperature == 0:
            nxt = logits.argmax(-1)
        else:
            nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=gen).squeeze(-1)
        ids[ar, L] = nxt
        out[:, t] = nxt
        L = L + 1
    return out


@torch.no_grad()
def beam_search(policy, prm, orm, probs: list[SumProblem], width: int = 4, expand: int = 4,
                temperature: float = 1.0, generator: torch.Generator | None = None) -> list[dict]:
    """PRM-guided step-level beam search (module docstring). Per problem: answer, correct, spend counters.

    Steps containing ``#``, EOS or PAD are invalid (score -inf). The final answer of each beam is sampled at
    the same temperature (up to D + 2 tokens, cut at EOS) and the ORM picks among the ``width`` beams."""
    D = probs[0].digits
    beams = [[""] for _ in probs]
    dec = np.zeros(len(probs))
    ver = np.zeros(len(probs))
    for step in range(D):
        rows, owner, base = [], [], []
        for i, (p, bs) in enumerate(zip(probs, beams)):
            m = width * expand if step == 0 else expand
            for b in bs:
                for _ in range(m):
                    rows.append([BOS] + TOK.encode(p.prompt) + TOK.encode(b))
                    owner.append(i)
                    base.append(b)
        new = _extend(policy, rows, probs[0].step_len, temperature, generator)
        cand_text, valid = [], []
        for r, b in zip(new.tolist(), base):
            ok = all(t not in (END_ID, EOS, PAD) for t in r)
            cand_text.append(b + (TOK.decode(r) if ok else ""))
            valid.append(ok)
        for i in owner:
            dec[i] += probs[0].step_len
        sc = score(prm, [probs[i] for i in owner], cand_text, [False] * len(owner))
        for i, t in zip(owner, cand_text):
            ver[i] += scored_tokens(probs[i], t, False)
        sc = np.where(valid, sc, -np.inf)
        nb = [[] for _ in probs]
        for i in range(len(probs)):
            idx = [j for j, o in enumerate(owner) if o == i]
            order = sorted(idx, key=lambda j: -sc[j])[:width]
            keep = [cand_text[j] for j in order if np.isfinite(sc[j])]
            nb[i] = keep or beams[i]
        beams = nb
    # final answer per beam, ORM selection
    rows, owner, think = [], [], []
    for i, (p, bs) in enumerate(zip(probs, beams)):
        for b in bs:
            rows.append([BOS] + TOK.encode(p.prompt) + TOK.encode(b + END_THINK))
            owner.append(i)
            think.append(b)
    ans = _extend(policy, rows, D + 2, temperature, generator)
    texts, fin = [], []
    for r, b, i in zip(ans.tolist(), think, owner):
        f = EOS in r
        body = r[:r.index(EOS)] if f else r
        dec[i] += 1 + len(body) + int(f)                  # '#' (inserted, counted as a decode step) + answer + EOS
        texts.append(b + END_THINK + TOK.decode(body))
        fin.append(f)
    sc = score(orm, [probs[i] for i in owner], texts, fin)
    for i, t, f in zip(owner, texts, fin):
        ver[i] += scored_tokens(probs[i], t, f)
    out = []
    for i, p in enumerate(probs):
        idx = [j for j, o in enumerate(owner) if o == i]
        best = max(idx, key=lambda j: sc[j])
        _, a = parse(texts[best])
        a = a if fin[best] else None
        out.append({"answer": a, "correct": a == p.answer, "decode_tokens": float(dec[i]),
                    "verifier_tokens": float(ver[i]), "text": texts[best]})
    return out


def search_spend(res: list[dict], probs: list[SumProblem], policy_params: int, verifier_params: int) -> Spend:
    """Total spend of a beam-search run, prompt prefilled once per problem (prefix caching)."""
    return Spend(policy_params, verifier_params, float(sum(prompt_tokens(p) for p in probs)),
                 float(sum(r["decode_tokens"] for r in res)), float(sum(r["verifier_tokens"] for r in res)))


# --------------------------------------------------------------------------------------------- timing

def measure_step_times(policy, judge, batches=(1, 4, 16, 64, 256), repeats: int = 10, ctx: int = 20) -> dict:
    """Seconds per cached decode step of the policy (forward + softmax + sampling), and per verifier forward, at
    several batch sizes (CPU).

    Uses ``frontierlab.perf.benchmark`` (warm-up, repeats, median). The policy step is measured at a context
    of ``ctx`` tokens (the middle of a response here)."""
    from frontierlab.perf import benchmark
    dec, ver = {}, {}
    probs = make_problems(max(batches), seed=99, held=True)
    for b in batches:
        ids = encode_prompts(probs[:b])

        def setup_cache(b=b, ids=ids):
            c = policy.new_cache()
            with torch.no_grad():
                pad = torch.cat([ids, ids[:, :max(0, ctx - ids.shape[1])]], 1)
                policy(pad, cache=c)
            holder["c"] = c

        holder: dict = {}
        tok = torch.full((b, 1), TOK.stoi["1"], dtype=torch.long)

        def step():                            # one decode step as sample() runs it: forward, softmax, draw
            with torch.no_grad():
                lg = policy(tok, cache=holder["c"]).logits[:, -1].float()
                torch.multinomial(torch.softmax(lg / 0.7, -1), 1)

        dec[b] = benchmark(step, warmup=2, repeats=repeats, setup=setup_cache).median
        jids, jlast = encode_scored(probs[:b], [p.target() for p in probs[:b]], [True] * b)

        def vstep():
            with torch.no_grad():
                judge(jids, jlast)

        ver[b] = benchmark(vstep, warmup=2, repeats=repeats).median
    ids = encode_prompts(probs[:64])

    def prefill():
        with torch.no_grad():
            policy(ids, cache=policy.new_cache())

    pre = benchmark(prefill, warmup=2, repeats=repeats).median / ids.numel()
    return {"decode_step": dec, "verifier": ver, "prefill_per_token": pre}
