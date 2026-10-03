"""Eval Suite v1, long-context component (Module 4, lesson 04.1).

Eval v0 (Module 1) asks "how good is the model at short context?". Eval v1 asks "does the model
*use* a long context?", and keeps v0 as its short-context regression check. Four components, all
scored per item so two models can be compared with a paired bootstrap:

1. **Natural-text position loss** (:func:`doc_position_losses`). The first ``T`` tokens of every
   held-out document with at least ``T`` tokens, so "position" means position in a real document.
   Loss by position bucket shows whether more context keeps helping, stops helping, or (past the
   trained length, without a working position scheme) hurts.
2. **Context gain** (:func:`context_gain`). For target tokens at document positions ``[lo, hi)``:
   loss with the context truncated to the last ``W`` tokens minus loss with the full context from the
   document start, per document. Positive = the model gets something from the tokens more than ``W``
   back. This is the direct "does it use the extra context" measurement on natural text.
3. **Synthetic key-value tasks with controlled position and distractors** (:func:`make_items`,
   :func:`score_items`, :func:`evidence_effect`). A haystack of held-out natural text with short statements inserted:
   ``Remember: <key> <value>.`` The query at the end is ``Remember: <key>`` and the model must put
   the right value first among the candidate values. Variables:

   * ``length`` — total tokens (the context-length sweep);
   * ``depth`` — where the answer statement sits (0 = start, 1 = just before the query): the
     position-sensitivity sweep ("lost in the middle");
   * ``distractors`` — other ``Remember`` statements with other keys and values;
   * ``hard`` — *lexical* distractors ``Ignore: <same key> <decoy>.``: a model that only matches
     the key token picks the decoy as often as the value;
   * ``hops`` — multi-hop chains ``Remember: k1 k2. Remember: k2 k3. ... Remember: kh value.``,
     statements shuffled through the haystack, query ``Remember: k1``, candidates = the final values
     of every chain. One hop is retrieval; two or more hops require *using* what was retrieved.

   Each item carries an *evidence-ablated* twin (the target chain replaced by filler) whose
   accuracy must fall to chance, and an exact *oracle* (:func:`oracle_answer`) that must score 100%:
   the two checks that the task measures what it claims.
4. **Short-context regression** (:func:`short_context_regression`): Eval v0 (held-out loss on the
   fixed windows at the original length, LAMBADA target log-probability) of the extended model
   against the model it came from, paired by item.

Scores per synthetic item: ``correct`` (the answer has the highest probability among the candidates),
``logp_cand`` (log-probability of the answer renormalised over the candidates: continuous, chance =
ln(1/K)) and ``logp`` (over the whole vocabulary). Chance accuracy is 1/K for K candidates.

:func:`effective_length` turns a length sweep into "the longest length at which the score's lower
confidence bound stays above a threshold at that length and every shorter one" — the RULER-style
definition (Hsieh et al. 2024, section 4), with the threshold stated by the user.

Version ``eval-v1.0``. Items are generated from a seed and the Data-v0 tokenizer and validation
split, so they are identical for every model scored with the same pins (stored in the result).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from frontierlab.data.loader import TokenData

VERSION = "eval-v1.0"


# --------------------------------------------------------------------------- task vocabulary

@dataclass
class TaskVocab:
    """Token ids the synthetic tasks are built from."""
    keys: list[int]
    values: list[int]
    remember: list[int]          # ids of " Remember:"
    ignore: list[int]            # ids of " Ignore:"
    period: list[int]            # ids of "."
    filler: np.ndarray           # natural-text token stream (no end-of-text tokens)
    pins: dict = field(default_factory=dict)


def vocab_from_tokenizer(tokenizer_path: str | Path, filler: TokenData, n_keys: int = 300, n_values: int = 300,
                         seed: int = 0, min_id: int = 1500) -> TaskVocab:
    """Keys and values: disjoint sets of single-token lowercase words (" table", " river", ...).

    Only tokens with id >= ``min_id`` are used: byte-level BPE assigns ids in merge order, so low ids
    are the most frequent pieces (" the", " more"), which would collide with the filler text.

    The words come from the tokenizer's own vocabulary, so every key and value is exactly one token
    and a statement can be built from ids without any BPE merge crossing a boundary.
    """
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_path))
    words = sorted(i for s, i in tok.get_vocab().items()
                   if s.startswith("Ġ") and s[1:].isascii() and s[1:].isalpha() and s[1:].islower()
                   and 5 <= len(s) - 1 <= 9 and i >= min_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(words)
    if len(words) < n_keys + n_values:
        raise ValueError(f"only {len(words)} single-token words; lower n_keys / n_values")
    eot = filler.meta.get("eot_id", 0)
    stream = np.asarray(filler.tokens, dtype=np.int64)
    import hashlib
    pins = {"tokenizer_sha256": hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest(),
            "filler_split": filler.split, "vocab_seed": seed, "n_keys": n_keys, "n_values": n_values}
    return TaskVocab(keys=sorted(words[:n_keys]), values=sorted(words[n_keys:n_keys + n_values]),
                     remember=tok.encode(" Remember:").ids, ignore=tok.encode(" Ignore:").ids,
                     period=tok.encode(".").ids, filler=stream[stream != eot], pins=pins)


# --------------------------------------------------------------------------- item generation

def _statement(prefix, a, b, period):
    return list(prefix) + [a, b] + list(period)


def make_items(vocab: TaskVocab, length: int, n: int, *, hops: int = 1, depth: float | None = None,
               distractors: int = 3, hard: int = 0, seed: int = 0) -> list[dict]:
    """``n`` items of exactly ``length`` tokens. ``depth=None`` places the answer statement at random.

    Every item is a dict with ``ids`` (list of token ids, query last), ``ids_ablated`` (same length,
    the target chain replaced by filler), ``answer``, ``candidates``, ``depth`` (actual, in [0, 1]) and
    the generation settings. The same arguments always give the same items.
    """
    if hops < 1:
        raise ValueError("hops >= 1")
    rng = np.random.default_rng([seed, length, hops, distractors, hard, 1001 if depth is None else int(depth * 1000)])
    items = []
    for _ in range(n):
        n_chains = 1 + distractors
        keys = rng.choice(vocab.keys, size=n_chains * hops, replace=False).reshape(n_chains, hops)
        vals = rng.choice(vocab.values, size=n_chains + hard, replace=False)
        chains = []                                     # list of statements per chain
        for c in range(n_chains):
            seq = list(keys[c]) + [vals[c]]
            chains.append([_statement(vocab.remember, int(seq[j]), int(seq[j + 1]), vocab.period) for j in range(hops)])
        decoys = [_statement(vocab.ignore, int(keys[0][0]), int(vals[n_chains + j]), vocab.period) for j in range(hard)]
        query = list(vocab.remember) + [int(keys[0][0])]
        target = chains[0]
        others = [s for c in chains[1:] for s in c] + decoys
        stmts = target + others
        body = length - len(query)
        n_fill = body - sum(len(s) for s in stmts)
        if n_fill < 1:
            raise ValueError(f"length {length} too short for {len(stmts)} statements")
        start = int(rng.integers(0, len(vocab.filler) - 2 * n_fill))
        fill = vocab.filler[start:start + n_fill].tolist()
        spare = vocab.filler[start + n_fill:start + n_fill + sum(len(s) for s in target)].tolist()
        # insertion offsets into the filler: answer statement (last of the target chain) at `depth`
        offs = rng.integers(0, n_fill + 1, size=len(stmts)).tolist()
        if depth is not None:
            offs[hops - 1] = int(round(depth * n_fill))
        # move each insertion point back to just after a sentence end, if there is one within 48 tokens
        ends = np.nonzero(np.asarray(fill) == vocab.period[-1])[0] + 1
        for j, off in enumerate(offs):
            p = ends[np.searchsorted(ends, off, side="right") - 1] if ends.size and ends[0] <= off else None
            if p is not None and off - p <= 48:
                offs[j] = int(p)
        order = list(range(len(stmts)))
        rng.shuffle(order)                               # statements at the same offset: random order
        placed = sorted(zip(offs, order, range(len(stmts))))
        ids, ab, pos_of, k, sp = [], [], {}, 0, 0
        for off, _, si in placed:
            ids += fill[k:off]
            ab += fill[k:off]
            k = off
            pos_of[si] = len(ids)
            ids += stmts[si]
            if si < hops:                                # target chain: ablated twin gets filler instead
                ab += spare[sp:sp + len(stmts[si])]
                sp += len(stmts[si])
            else:
                ab += stmts[si]
        ids += fill[k:] + query
        ab += fill[k:] + query
        assert len(ids) == length == len(ab)
        cands = sorted(int(c[-1][-1 - len(vocab.period)]) for c in chains) + sorted(int(v) for v in vals[n_chains:])
        items.append({"ids": ids, "ids_ablated": ab, "answer": int(vals[0]), "candidates": cands,
                      "depth": pos_of[hops - 1] / max(1, body - len(stmts[hops - 1])), "length": length,
                      "hops": hops, "distractors": distractors, "hard": hard})
    return items


def oracle_answer(item: dict, vocab: TaskVocab, field: str = "ids") -> int | None:
    """Solve an item exactly by reading its ``Remember`` statements (None if the evidence is missing)."""
    ids, P = item[field], list(vocab.remember)
    n, m = len(ids), len(P)
    q = ids[-1]
    link, keyset = {}, set(vocab.keys)
    for i in range(n - m - 1 - len(vocab.period)):
        if ids[i:i + m] == P and ids[i + m] in keyset:
            link.setdefault(ids[i + m], ids[i + m + 1])
    cur, seen = q, set()
    while cur in link and cur not in seen:
        seen.add(cur)
        cur = link[cur]
        if cur in item["candidates"]:
            return cur
    return None


# --------------------------------------------------------------------------- scoring

@torch.no_grad()
def score_items(model, items: list[dict], field: str = "ids", batch: int = 8, device="cpu",
                autocast_dtype=None) -> list[dict]:
    """Per item: correct (answer ranked first among candidates), logp_cand, logp, k (candidates)."""
    was = model.training
    model.eval()
    out = [None] * len(items)
    by_len: dict[int, list[int]] = {}
    for i, it in enumerate(items):
        by_len.setdefault(len(it[field]), []).append(i)
    for idx in by_len.values():
        for j in range(0, len(idx), batch):
            chunk = idx[j:j + batch]
            x = torch.tensor([items[i][field] for i in chunk], dtype=torch.long, device=device)
            with torch.autocast(device_type=x.device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                last = model(x).logits[:, -1].float()
            logp = torch.log_softmax(last, dim=-1)
            for r, i in enumerate(chunk):
                c = torch.tensor(items[i]["candidates"], device=device)
                lc = logp[r, c]
                a = items[i]["candidates"].index(items[i]["answer"])
                norm = lc - torch.logsumexp(lc, 0)
                out[i] = {"correct": bool(lc.argmax().item() == a and (lc == lc[a]).sum().item() == 1),
                          "logp_cand": float(norm[a]), "logp": float(logp[r, items[i]["answer"]]), "k": len(c)}
    model.train(was)
    return out


# --------------------------------------------------------------------------- natural text

def doc_starts_at_least(data: TokenData, T: int, max_docs: int | None = None) -> list[int]:
    from frontierlab.longctx.data import doc_start_windows
    return doc_start_windows(data, T, max_docs)


@torch.no_grad()
def doc_position_losses(model, data: TokenData, T: int, max_docs: int | None = None, batch: int = 4,
                        device="cpu", autocast_dtype=None) -> np.ndarray:
    """(n_docs, T-1) per-token losses on the first T tokens of every document with >= T tokens.

    Column p is the loss of predicting document token p+1 from tokens 0..p.
    """
    was = model.training
    model.eval()
    starts, rows = doc_starts_at_least(data, T, max_docs), []
    for i in range(0, len(starts), batch):
        x = torch.stack([data.window(s, T) for s in starts[i:i + batch]]).to(device)
        with torch.autocast(device_type=x.device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            rows.append(model(x, labels=x).per_token_loss.float().cpu().numpy())
    model.train(was)
    return np.concatenate(rows) if rows else np.zeros((0, T - 1))


def bucket_summary(losses: np.ndarray, edges: list[int], n_boot: int = 2000) -> list[dict]:
    """Mean loss per position bucket [edges[i], edges[i+1]) with a bootstrap CI over documents."""
    from frontierlab.stats import bootstrap_ci
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        per_doc = losses[:, lo:hi].mean(axis=1)
        m, a, b = bootstrap_ci(per_doc, n_boot=n_boot)
        out.append({"lo": lo, "hi": hi, "mean": m, "ci": (a, b), "docs": int(per_doc.size)})
    return out


@torch.no_grad()
def context_gain(model, data: TokenData, T: int, W: int, lo: int, hi: int | None = None,
                 max_docs: int | None = None, batch: int = 4, device="cpu", autocast_dtype=None) -> np.ndarray:
    """Per document: mean loss on targets at positions [lo, hi) with >= W tokens of context, minus with full context.

    Target token t (document position, lo <= t < hi <= T) is predicted (a) from document tokens
    0..t-1 and (b) from tokens lo-W..t-1 only. Returns (n_docs,) of mean(b) - mean(a): positive means the
    tokens more than W back helped.
    """
    hi = hi or T
    if not W <= lo < hi <= T:
        raise ValueError("need W <= lo < hi <= T")
    was = model.training
    model.eval()
    starts, gains = doc_starts_at_least(data, T, max_docs), []
    for i in range(0, len(starts), batch):
        full = torch.stack([data.window(s, hi) for s in starts[i:i + batch]]).to(device)
        cut = full[:, lo - W:]
        with torch.autocast(device_type=full.device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            a = model(full, labels=full).per_token_loss.float()[:, lo - 1:hi - 1]
            b = model(cut, labels=cut).per_token_loss.float()[:, W - 1:]
        gains.append((b.mean(1) - a.mean(1)).cpu().numpy())
    model.train(was)
    return np.concatenate(gains) if gains else np.zeros(0)


# --------------------------------------------------------------------------- summaries

def summarize_scores(scores: list[dict], n_boot: int = 2000) -> dict:
    from frontierlab.stats import bootstrap_ci
    acc = [float(s["correct"]) for s in scores]
    lc = [s["logp_cand"] for s in scores]
    k = np.mean([s["k"] for s in scores])
    return {"n": len(scores), "acc": bootstrap_ci(acc, n_boot=n_boot), "logp_cand": bootstrap_ci(lc, n_boot=n_boot),
            "chance_acc": float(np.mean([1 / s["k"] for s in scores])), "chance_logp_cand": float(-math.log(k))}


def effective_length(cells: list[dict], threshold: float, key: str = "acc") -> int | None:
    """Longest length L such that every swept length <= L has CI lower bound >= ``threshold``.

    ``cells``: [{"length": L, key: (mean, lo, hi)}, ...]. None if even the shortest length fails.
    """
    best = None
    for c in sorted(cells, key=lambda c: c["length"]):
        if c[key][1] >= threshold:
            best = c["length"]
        else:
            break
    return best


def short_context_regression(base_v0: dict, new_v0: dict, n_boot: int = 10000) -> dict:
    """Paired comparison of two Eval v0 results (``suite_v0.run_suite``) on the same items.

    Returns new - base for held-out loss (positive = worse) and LAMBADA target log-probability
    (negative = worse), with paired bootstrap intervals.
    """
    from frontierlab.stats import paired_bootstrap
    for k in ("windows", "T", "window_seed", "split"):
        if base_v0["heldout"][k] != new_v0["heldout"][k]:
            raise ValueError(f"Eval v0 held-out pins differ ({k}); results are not paired")
    if base_v0["lambada"]["sha256"] != new_v0["lambada"]["sha256"] or base_v0["lambada"]["n"] != new_v0["lambada"]["n"]:
        raise ValueError("LAMBADA pins differ; results are not paired")
    h = paired_bootstrap(new_v0["heldout"]["losses"], base_v0["heldout"]["losses"], n_boot=n_boot)
    lp = paired_bootstrap([i["logprob"] for i in new_v0["lambada"]["items"]],
                          [i["logprob"] for i in base_v0["lambada"]["items"]], n_boot=n_boot)
    return {"heldout_loss_diff": h, "lambada_logprob_diff": lp}


def run_synthetic(model, vocab: TaskVocab, lengths: list[int], *, depths=(0.0, 0.25, 0.5, 0.75, 1.0), n: int = 40,
                  hops=(1, 2), distractors: int = 3, hard: int = 1, seed: int = 0, device="cpu",
                  autocast_dtype=None, batch: int = 8, ablation: bool = True) -> dict:
    """The synthetic component over a grid: retrieval (hops=1) per length x depth, multi-hop per length."""
    cells = []
    for L in lengths:
        for h in hops:
            for d in (depths if h == 1 else (None,)):
                items = make_items(vocab, L, n, hops=h, depth=d, distractors=distractors, hard=hard if h == 1 else 0,
                                   seed=seed)
                sc = score_items(model, items, device=device, autocast_dtype=autocast_dtype, batch=batch)
                cell = {"length": L, "hops": h, "depth": d, "scores": sc}
                if ablation:
                    cell["scores_ablated"] = score_items(model, items, "ids_ablated", device=device,
                                                         autocast_dtype=autocast_dtype, batch=batch)
                cells.append(cell)
    return {"version": VERSION, "pins": {**vocab.pins, "seed": seed, "n": n, "distractors": distractors, "hard": hard},
            "cells": cells}


# --------------------------------------------------------------------------- the whole suite, and a CLI

def default_edges(L: int) -> list[int]:
    """Position buckets 0, 64, 128, 256, ... up to L-1 (the last column of per-token losses)."""
    e, b = [0], 64
    while b < L - 1:
        e.append(b)
        b *= 2
    return e + [L - 1]


def run_suite(model, vocab: TaskVocab, data: TokenData, lengths: list[int], train_len: int, *, n: int = 40,
              max_docs: int = 100, depths=(0.0, 0.25, 0.5, 0.75, 1.0), hops=(1, 2), distractors: int = 3,
              hard: int = 1, seed: int = 0, device="cpu", autocast_dtype=None, batch: int = 8) -> dict:
    """Components 1-3 of Eval v1 at each length (component 4 is Eval v0, run separately at ``train_len``).

    Context gain at length L uses targets in [L/2, L) and truncation lengths W = train_len/4 and
    W = train_len (only those with W <= L/2): "do tokens more than W back still help at length L?".
    """
    natural = []
    for L in lengths:
        losses = doc_position_losses(model, data, L, max_docs, device=device, autocast_dtype=autocast_dtype)
        if losses.shape[0] == 0:
            natural.append({"length": L, "docs": 0})
            continue
        edges = default_edges(L)
        gains = {}
        for W in sorted({max(1, train_len // 4), train_len}):
            if W <= L // 2:
                g = context_gain(model, data, L, W, L // 2, L, max_docs, device=device, autocast_dtype=autocast_dtype)
                gains[str(W)] = g.tolist()
        natural.append({"length": L, "docs": int(losses.shape[0]), "edges": edges,
                        "buckets": bucket_summary(losses, edges),
                        "doc_mean_by_bucket": [losses[:, lo:hi].mean(1).tolist() for lo, hi in zip(edges[:-1], edges[1:])],
                        "context_gain": gains})
    synth = run_synthetic(model, vocab, lengths, depths=depths, n=n, hops=hops, distractors=distractors, hard=hard,
                          seed=seed, device=device, autocast_dtype=autocast_dtype, batch=batch)
    return {"version": VERSION, "train_len": train_len, "lengths": list(lengths),
            "pins": {**synth["pins"], "data_split": data.split, "max_docs": max_docs,
                     "data_bin_sha256": data.meta.get(data.split, {}).get("bin_sha256")},
            "natural": natural, "synthetic": synth["cells"]}


def report(res: dict) -> str:
    """Plain-text tables of one Eval v1 result."""
    from frontierlab.stats import bootstrap_ci
    out = [f"Eval {res['version']}  (trained length {res['train_len']})", "",
           "1. Natural-text loss by document position (mean over documents)"]
    for nat in res["natural"]:
        if not nat.get("docs"):
            out.append(f"  L={nat['length']}: no held-out document this long")
            continue
        cells = "  ".join(f"[{b['lo']},{b['hi']}) {b['mean']:.3f}" for b in nat["buckets"])
        out.append(f"  L={nat['length']:>6} ({nat['docs']} docs): {cells}")
    out += ["", "2. Context gain = loss(context cut to last W tokens) - loss(full context), targets in [L/2, L)"]
    for nat in res["natural"]:
        for W, g in (nat.get("context_gain") or {}).items():
            m, lo, hi = bootstrap_ci(g, n_boot=2000)
            out.append(f"  L={nat['length']:>6} W={W:>5}: {m:+.4f} nats  95% CI [{lo:+.4f}, {hi:+.4f}]  ({len(g)} docs)")
    out += ["", "3. Synthetic tasks: accuracy [95% CI] | accuracy with the evidence removed | chance |",
            "   evidence effect = logp_cand(with evidence) - logp_cand(evidence removed), paired [95% CI]"]
    for c in res["synthetic"]:
        s = summarize_scores(c["scores"])
        where = f"depth {c['depth']:.2f}" if c["depth"] is not None else "shuffled"
        line = (f"  L={c['length']:>6} hops={c['hops']} {where:>11}: acc {s['acc'][0]:.2f} "
                f"[{s['acc'][1]:.2f}, {s['acc'][2]:.2f}]")
        if "scores_ablated" in c:
            ab = summarize_scores(c["scores_ablated"])["acc"][0]
            ev = evidence_effect(c)
            line += (f" | ablated {ab:.2f} | chance {s['chance_acc']:.2f} | evidence {ev['mean_diff']:+.3f} "
                     f"[{ev['ci'][0]:+.3f}, {ev['ci'][1]:+.3f}]")
        out.append(line)
    return "\n".join(out)


def evidence_effect(cell: dict, n_boot: int = 2000) -> dict:
    """Paired mean of logp_cand(with evidence) - logp_cand(evidence removed) over a cell's items.

    Positive = the model reads the statement. Unlike accuracy against chance, it cancels the model's
    prior preference among the candidate tokens, which is the same in both twins.
    """
    from frontierlab.stats import paired_bootstrap
    return paired_bootstrap([x["logp_cand"] for x in cell["scores"]], [x["logp_cand"] for x in cell["scores_ablated"]],
                            n_boot=n_boot)


def compare(a: dict, b: dict, n_boot: int = 4000) -> str:
    """Paired differences b - a for every component both results share (same items, same documents)."""
    from frontierlab.stats import paired_bootstrap
    if a["pins"] != b["pins"]:
        raise ValueError(f"Eval v1 pins differ; results are not paired:\n{a['pins']}\n{b['pins']}")
    out = ["Paired differences (second - first), 95% CI. Loss: negative = second better. "
           "Context gain and accuracy: positive = second better."]
    nat_b = {n["length"]: n for n in b["natural"]}
    for na in a["natural"]:
        nb = nat_b.get(na["length"])
        if not na.get("docs") or nb is None or not nb.get("docs"):
            continue
        for (lo, hi), da, db in zip(zip(na["edges"][:-1], na["edges"][1:]), na["doc_mean_by_bucket"],
                                    nb["doc_mean_by_bucket"]):
            r = paired_bootstrap(db, da, n_boot=n_boot)
            out.append(f"  loss         L={na['length']:>6} [{lo},{hi}): {r['mean_diff']:+.4f}  "
                       f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
        for W in na.get("context_gain", {}):
            if W in nb.get("context_gain", {}):
                r = paired_bootstrap(nb["context_gain"][W], na["context_gain"][W], n_boot=n_boot)
                out.append(f"  context gain L={na['length']:>6} W={W}: {r['mean_diff']:+.4f}  "
                           f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
    syn_b = {(c["length"], c["hops"], c["depth"]): c for c in b["synthetic"]}
    for ca in a["synthetic"]:
        cb = syn_b.get((ca["length"], ca["hops"], ca["depth"]))
        if cb is None:
            continue
        r = paired_bootstrap([float(s["correct"]) for s in cb["scores"]], [float(s["correct"]) for s in ca["scores"]],
                             n_boot=n_boot)
        where = f"depth {ca['depth']:.2f}" if ca["depth"] is not None else "shuffled"
        line = (f"  accuracy     L={ca['length']:>6} hops={ca['hops']} {where:>11}: {r['mean_diff']:+.3f}  "
                f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]")
        if "scores_ablated" in ca and "scores_ablated" in cb:
            ea = [x["logp_cand"] - y["logp_cand"] for x, y in zip(ca["scores"], ca["scores_ablated"])]
            eb = [x["logp_cand"] - y["logp_cand"] for x, y in zip(cb["scores"], cb["scores_ablated"])]
            e = paired_bootstrap(eb, ea, n_boot=n_boot)
            line += f"   evidence effect {e['mean_diff']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]"
        out.append(line)
    return "\n".join(out)


def load_model(run_or_ckpt: str | Path, device="cpu", rope: dict | None = None):
    """Model from ``<run>/checkpoint.pt`` (or a checkpoint path); ``rope`` = a RoPE rule to evaluate it with."""
    import frontierlab.longctx  # noqa: F401  (registers the Module 4 attention kinds)
    from frontierlab.longctx.attention import convert
    from frontierlab.model import LM, ModelConfig
    p = Path(run_or_ckpt)
    ck = torch.load(p / "checkpoint.pt" if p.is_dir() else p, map_location="cpu", weights_only=False)
    m = LM(ModelConfig(**ck["config"]))
    m.load_state_dict(ck["model"])
    if rope is not None:
        kind = m.config.attention if m.config.attention in ("gqa-rope-scaled", "gqa-irope") else "gqa-rope-scaled"
        m = convert(m, kind, rope=rope)
    return m.to(device).eval()


def main(argv=None):
    """``python -m frontierlab.evals.suite_v1 run RUN --train-len 256 --out RUN/eval_v1.json`` or ``compare A B``."""
    import argparse
    import json
    import time
    from frontierlab.data.prepare import DEFAULT_OUT
    ap = argparse.ArgumentParser(description="Eval Suite v1 (long context)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="score one model")
    r.add_argument("run_dir", help="run folder with checkpoint.pt, or a checkpoint path")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--lengths", type=int, nargs="+", default=[256, 512, 1024])
    r.add_argument("--train-len", type=int, required=True, help="the length the model was (last) trained at")
    r.add_argument("--rope", type=json.loads, default=None,
                   help="evaluate with this RoPE rule (JSON), e.g. {\"type\": \"yarn\", \"factor\": 4, "
                        "\"original_max_position_embeddings\": 256}")
    r.add_argument("--data", type=Path, default=None, help="prepared folder with long held-out documents "
                   "(default: labs/common/data/v0-long if it exists, else Data-v0)")
    r.add_argument("--split", default="val")
    r.add_argument("--n", type=int, default=40, help="items per synthetic cell")
    r.add_argument("--max-docs", type=int, default=100)
    r.add_argument("--hops", type=int, nargs="+", default=[1, 2])
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    r.add_argument("--bf16", action="store_true")
    c = sub.add_parser("compare", help="paired differences between two results")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        print(compare(json.loads(a.a.read_text()), json.loads(a.b.read_text())))
        return None
    long_root = DEFAULT_OUT.parent / "v0-long"
    root = a.data or (long_root if (long_root / "meta.json").exists() else DEFAULT_OUT)
    t0 = time.perf_counter()
    data = TokenData(a.split, root)
    vocab = vocab_from_tokenizer(DEFAULT_OUT / "tokenizer.json", TokenData("val", DEFAULT_OUT))
    model = load_model(a.run_dir, a.device, a.rope)
    res = run_suite(model, vocab, data, a.lengths, a.train_len, n=a.n, max_docs=a.max_docs, hops=tuple(a.hops),
                    seed=a.seed, device=a.device, autocast_dtype=torch.bfloat16 if a.bf16 else None)
    res["model"] = {"run": str(a.run_dir), "rope": a.rope, "attention": model.config.attention}
    res["pins"]["data_root"] = root.name
    res["seconds"] = round(time.perf_counter() - t0, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res))
    print(report(res))
    print(f"\n{res['seconds']} s -> {a.out}")
    return res


if __name__ == "__main__":
    main()
