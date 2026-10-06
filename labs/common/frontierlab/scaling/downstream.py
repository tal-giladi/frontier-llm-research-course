"""Predicting downstream capability (lesson 11.2): a task tiny models can do, its metrics, and the fits.

**The task.** :func:`build_cloze` makes a four-way multiple-choice cloze from held-out text: a context of
``ctx`` tokens from one document, the true next ``cont`` tokens, and three distractors that are ``cont``-token
spans of *other* documents. Every option is real text of the same length, so a model only wins by linking the
continuation to the context. It is the construction of HellaSwag-style benchmarks reduced to what a 100K-parameter
model can show a trend on; it is not a capability benchmark.

**The metrics** (:func:`score_cloze`), from most to least continuous:

* ``nll_correct``: mean per-token negative log-likelihood of the correct continuation (Llama 3's first step,
  arXiv 2407.21783 section 3.2.1, uses the normalised NLL of the correct answer);
* ``p_correct``: the probability mass on the correct option after a softmax over the four options' total
  log-probabilities (Schaeffer et al., arXiv 2406.04391, follow how this mass moves);
* ``brier``: Σ_i (p_i − y_i)², a proper score;
* ``acc``: whether the correct option has the highest log-probability (a step function per item).

:func:`exact_match` greedy-decodes ``k`` tokens and compares them with the true continuation: an all-or-nothing
metric over k tokens. With per-token accuracy p it behaves roughly like p^k, the mechanism Schaeffer, Miranda and
Koyejo (arXiv 2304.15004) give for "emergent" curves that are smooth under a per-token metric.

**The fits.** :func:`fit_sigmoid` (accuracy as a function of task NLL: step 2 of the two-step method),
:func:`two_step_predict` (step 1 is any fit of task NLL against scale, e.g. ``fit.fit_parametric``), and
:func:`pca_capabilities` with :func:`fit_logistic` for observational scaling laws (Ruan et al., arXiv 2405.10938):
benchmark scores of many public models compressed to a few principal components, each benchmark a sigmoid of a
linear function of them.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def _spans(doc_starts: np.ndarray, n_tokens: int, length: int) -> list[tuple[int, int]]:
    """(doc_index, start) of every position where a span of ``length`` fits inside one document."""
    bounds = list(doc_starts.tolist()) + [n_tokens]
    out = []
    for d in range(len(bounds) - 1):
        lo, hi = bounds[d], bounds[d + 1] - length
        if hi > lo:
            out.append((d, lo, hi))
    return out


def build_cloze(data, n_items: int = 400, ctx: int = 48, cont: int = 8, n_choices: int = 4, seed: int = 0) -> dict:
    """Items from ``data`` (a :class:`frontierlab.data.loader.TokenData`, normally the test split).

    Returns ``context`` (n, ctx) int64, ``choices`` (n, n_choices, cont) int64, ``answer`` (n,) int64 (the
    position of the true continuation, uniform over options), all on CPU. Deterministic given ``seed``.
    """
    if data.doc_starts is None:
        raise ValueError("needs document boundaries (<split>_docs.npy)")
    tok = np.asarray(data.tokens)
    rng = np.random.default_rng(seed)
    item_docs = _spans(data.doc_starts, len(tok), ctx + cont)
    span_docs = _spans(data.doc_starts, len(tok), cont)
    if len(item_docs) < 2:
        raise ValueError("too few documents")
    contexts, choices, answers = [], [], []
    for _ in range(n_items):
        d, lo, hi = item_docs[rng.integers(len(item_docs))]
        s = int(rng.integers(lo, hi))
        true = tok[s + ctx:s + ctx + cont]
        opts = []
        while len(opts) < n_choices - 1:
            d2, lo2, hi2 = span_docs[rng.integers(len(span_docs))]
            if d2 == d:
                continue
            s2 = int(rng.integers(lo2, hi2))
            opts.append(tok[s2:s2 + cont])
        a = int(rng.integers(n_choices))
        opts.insert(a, true)
        contexts.append(tok[s:s + ctx])
        choices.append(np.stack(opts))
        answers.append(a)
    return {"context": torch.from_numpy(np.stack(contexts).astype(np.int64)),
            "choices": torch.from_numpy(np.stack(choices).astype(np.int64)),
            "answer": torch.tensor(answers, dtype=torch.int64), "ctx": ctx, "cont": cont}


@torch.no_grad()
def option_logprobs(model, items: dict, batch: int = 128, device="cpu") -> torch.Tensor:
    """(n, n_choices) float64: total log-probability of each option's tokens given the context."""
    was = model.training
    model.eval()
    ctx, cont = items["ctx"], items["cont"]
    n, k, _ = items["choices"].shape
    seqs = torch.cat([items["context"][:, None, :].expand(n, k, ctx), items["choices"]], dim=2).reshape(n * k, ctx + cont)
    out = []
    for i in range(0, seqs.shape[0], batch):
        x = seqs[i:i + batch].to(device)
        logits = model(x).logits.double()                         # (b, ctx+cont, V)
        lp = F.log_softmax(logits[:, ctx - 1:-1], dim=-1)         # predictions for the option tokens
        out.append(lp.gather(-1, x[:, ctx:, None]).squeeze(-1).sum(-1).cpu())
    model.train(was)
    return torch.cat(out).reshape(n, k)


def metrics_from_logprobs(lp: torch.Tensor, answer: torch.Tensor, cont: int) -> dict:
    """Accuracy, p_correct, Brier score and NLL of the correct option (per token), with per-item values kept."""
    lp = lp.double()
    p = torch.softmax(lp, dim=1)
    y = F.one_hot(answer, lp.shape[1]).double()
    acc_items = (lp.argmax(1) == answer).double()
    pc_items = p.gather(1, answer[:, None]).squeeze(1)
    brier_items = ((p - y) ** 2).sum(1)
    nll_items = -lp.gather(1, answer[:, None]).squeeze(1) / cont
    return {"acc": float(acc_items.mean()), "p_correct": float(pc_items.mean()), "brier": float(brier_items.mean()),
            "nll_correct": float(nll_items.mean()), "n": int(lp.shape[0]),
            "items": {"acc": acc_items.tolist(), "p_correct": pc_items.tolist(), "nll_correct": nll_items.tolist()}}


def score_cloze(model, items: dict, device="cpu", batch: int = 128) -> dict:
    return metrics_from_logprobs(option_logprobs(model, items, batch, device), items["answer"], items["cont"])


@torch.no_grad()
def exact_match(model, items: dict, k: int = 4, device="cpu", batch: int = 128) -> dict:
    """Greedy-decode ``k`` tokens after each context; exact match of all k with the true continuation, and the
    teacher-forced per-token top-1 accuracy on the same k positions."""
    was = model.training
    model.eval()
    n = items["context"].shape[0]
    true = items["choices"][torch.arange(n), items["answer"], :k]
    em, tok_acc = [], []
    for i in range(0, n, batch):
        x = items["context"][i:i + batch].to(device)
        t = true[i:i + batch].to(device)
        tf = model(torch.cat([x, t], 1)).logits[:, x.shape[1] - 1:-1].argmax(-1)    # teacher-forced predictions
        tok_acc.append((tf == t).double().cpu())
        seq = x
        for _ in range(k):
            nxt = model(seq).logits[:, -1].argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], 1)
        em.append((seq[:, -k:] == t).all(1).double().cpu())
    model.train(was)
    em, tok_acc = torch.cat(em), torch.cat(tok_acc)
    return {"exact_match": float(em.mean()), "token_acc": float(tok_acc.mean()), "k": k,
            "token_acc_pow_k": float(tok_acc.mean()) ** k}


def sigmoid_curve(x, x0: float, s: float, lo: float, hi: float) -> np.ndarray:
    """lo + (hi − lo) / (1 + exp(s·(x − x0))): decreasing in x for s > 0 (x = task NLL, lower is better)."""
    return lo + (hi - lo) / (1.0 + np.exp(s * (np.asarray(x, dtype=np.float64) - x0)))


def fit_sigmoid(x, y, lo: float | None = 0.25, hi: float | None = 1.0) -> dict:
    """Least-squares fit of :func:`sigmoid_curve`. ``lo``/``hi`` fixed (chance and perfect accuracy) unless None.

    Grid over (x0, s) for a start, then L-BFGS on all free parameters in float64.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    free_lo, free_hi = lo is None, hi is None
    lo0 = float(y.min()) - 1e-3 if free_lo else lo
    hi0 = float(y.max()) + 1e-3 if free_hi else hi
    best = None
    span = max(x.max() - x.min(), 1e-6)
    for x0 in np.linspace(x.min() - span, x.max() + span, 61):
        for s in np.geomspace(0.1 / span, 50 / span, 40):
            r = sigmoid_curve(x, x0, s, lo0, hi0) - y
            sse = float(r @ r)
            if best is None or sse < best[0]:
                best = (sse, x0, s)
    p = torch.tensor([best[1], math.log(best[2]), lo0, hi0], dtype=torch.float64, requires_grad=True)
    tx, ty = torch.tensor(x), torch.tensor(y)
    opt = torch.optim.LBFGS([p], lr=1.0, max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        l = p[2] if free_lo else torch.tensor(lo0, dtype=torch.float64)
        h = p[3] if free_hi else torch.tensor(hi0, dtype=torch.float64)
        f = ((l + (h - l) / (1 + torch.exp(torch.exp(p[1]) * (tx - p[0]))) - ty) ** 2).sum()
        f.backward()
        return f

    opt.step(closure)
    x0, ls, l, h = p.detach().tolist()
    out = {"x0": x0, "s": math.exp(ls), "lo": l if free_lo else lo0, "hi": h if free_hi else hi0}
    if not all(math.isfinite(v) for v in out.values()):
        out = {"x0": best[1], "s": best[2], "lo": lo0, "hi": hi0}
    r = sigmoid_curve(x, out["x0"], out["s"], out["lo"], out["hi"]) - y
    out["rms"] = float(np.sqrt(np.mean(r ** 2)))
    return out


def predict_sigmoid(fit: dict, x) -> np.ndarray:
    return sigmoid_curve(x, fit["x0"], fit["s"], fit["lo"], fit["hi"])


def two_step_predict(step1_predict, sigmoid_fit: dict, *args) -> dict:
    """Step 1: task NLL at the target (``step1_predict(*args)``); step 2: accuracy from that NLL."""
    nll = float(step1_predict(*args))
    return {"nll_correct": nll, "acc": float(predict_sigmoid(sigmoid_fit, nll))}


def pca_capabilities(scores, n_components: int = 3) -> dict:
    """Principal components of a (models × benchmarks) score matrix (Ruan et al., section 3).

    Columns are standardised (zero mean, unit variance) first; rows with a missing value (NaN) are dropped.
    Returns the per-model coordinates ``S`` (models × n_components), the ``loadings`` (n_components ×
    benchmarks), the fraction of variance each component explains and the kept row mask.
    """
    X = np.asarray(scores, dtype=np.float64)
    keep = ~np.isnan(X).any(1)
    Xk = X[keep]
    mu, sd = Xk.mean(0), Xk.std(0, ddof=0)
    Z = (Xk - mu) / np.where(sd > 0, sd, 1.0)
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    if Vt[0].sum() < 0:                     # sign convention: PC1 grows with the scores
        U[:, 0], Vt[0] = -U[:, 0], -Vt[0]
    S = U[:, :n_components] * s[:n_components]
    return {"S": S, "loadings": Vt[:n_components], "explained": (s ** 2 / (s ** 2).sum())[:n_components].tolist(),
            "mean": mu, "std": sd, "keep": keep}


def fit_logistic(S, y, lo: float = 0.0, hi: float = 1.0, l2: float = 1e-4) -> dict:
    """y ≈ lo + (hi − lo)·sigmoid(w·S + b) by L-BFGS on squared error (observational scaling law, Ruan Eq. 6 form)."""
    S, y = np.atleast_2d(np.asarray(S, dtype=np.float64)), np.asarray(y, dtype=np.float64)
    if S.shape[0] != y.shape[0]:
        S = S.T
    tS, ty = torch.tensor(S), torch.tensor(y)
    w = torch.zeros(S.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        f = ((lo + (hi - lo) * torch.sigmoid(tS @ w + b) - ty) ** 2).sum() + l2 * (w ** 2).sum()
        f.backward()
        return f

    opt.step(closure)
    return {"w": w.detach().numpy().copy(), "b": float(b.detach()), "lo": lo, "hi": hi}


def predict_logistic(fit: dict, S) -> np.ndarray:
    S = np.atleast_2d(np.asarray(S, dtype=np.float64))
    return fit["lo"] + (fit["hi"] - fit["lo"]) / (1 + np.exp(-(S @ fit["w"] + fit["b"])))
