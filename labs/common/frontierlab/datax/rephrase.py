"""Rephrasing harness: rewrite documents with a small open model, check the output, count the compute (lesson 10.3).

    python -m frontierlab.datax.rephrase run --source labs/common/data/m10/web --docs 300 --style wiki \\
        --model Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca --out runs/m10/reph-wiki
    python -m frontierlab.datax.rephrase build runs/m10/reph-wiki --out labs/common/data/m10/reph-wiki

``run`` takes the first ``--docs`` training documents of a source, truncates each to ``--max-src``
tokens of its own tokenizer (WRAP rephrases passages of at most 300 tokens, section 3.1), asks the
model to rewrite it in a style (:data:`STYLES`), and writes ``rephrased.jsonl`` (source id, source
text, output, token counts) and ``compute.json``. ``build`` writes the outputs as a training source
in the Data-v0 format, with the provenance of both the source documents and the generator.

Generator compute is counted, because it is part of the cost of the data (plan section 9). For a
decoder with N non-embedding parameters, L layers and attention width d_attn = heads × head_dim,
generating with a KV cache costs per sequence (forward only, one multiply-add = 2 FLOPs):

    prompt (prefill, P tokens):    2·N·P + 2·L·d_attn·P²                (P²/2 query-key pairs × 4·d_attn)
    each new token at context c:   2·N + 4·L·d_attn·c
    total:                         2·N·(P + G) + 2·L·d_attn·P² + 4·L·d_attn·Σ_{c=P}^{P+G-1} c

plus the output head 2·V·C per token if it is tied (N excludes it then). :func:`generation_flops`
implements this; the measured wall time and tokens per second are recorded next to it.

Checks on the output (all cheap, no second model):

* **fidelity**: share of the source's numbers present in the output; numbers in the output that are
  not in the source (*invented numbers*, a hallucination signal); recall of the source's content words;
* **novelty**: share of output word 4-grams that do not occur in the source (0 = a copy);
* **diversity** across outputs: distinct word bigrams over all outputs; the share of outputs that
  start with the same 6 words as another output (template collapse); mean pairwise MinHash Jaccard;
* **meta-text**: outputs that talk about the task ("Here is the rewritten text", "rephrased version");
* **contamination**: run :mod:`frontierlab.datax.leakage` on the outputs too, because the generator
  can reproduce text it memorised, including benchmark items (lesson 10.3).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

STYLES = {
    "wiki": ("Rewrite the following text as a short, clear encyclopedia-style passage. Keep every fact, "
             "name and number exactly. Do not add facts. Output only the rewritten passage.\n\nText:\n{text}"),
    "qa": ("Convert the following text into three to five question-and-answer pairs that together cover its "
           "facts. Keep names and numbers exactly. Do not add facts. Use the format 'Question: ...' then "
           "'Answer: ...'.\n\nText:\n{text}"),
    "simple": ("Rewrite the following text in simple, plain English that a twelve-year-old can follow. Keep every "
               "fact, name and number. Output only the rewritten text.\n\nText:\n{text}"),
}
META = re.compile(r"(here is|here's|rewritten|rephrased|i hope|as an ai|sure[,!])", re.I)
NUM = re.compile(r"\d+(?:[.,]\d+)*")
WORD = re.compile(r"[a-zA-Z]+")
STOP = set("the a an and or but if of to in on at for with by from as is are was were be been it its this that "
           "these those he she they we you i his her their our your not no yes do does did have has had will "
           "would can could should may might also than then there here which who whom what when where why how "
           "all any some more most such only into over under about after before between".split())


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in NUM.findall(text)}


def content_words(text: str) -> set[str]:
    return {w for w in (x.lower() for x in WORD.findall(text)) if len(w) > 3 and w not in STOP}


def word_ngrams(text: str, n: int) -> list[tuple]:
    w = [x.lower() for x in WORD.findall(text)]
    return [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]


def fidelity(src: str, out: str) -> dict:
    ns, no = numbers(src), numbers(out)
    cs, co = content_words(src), content_words(out)
    g_out = word_ngrams(out, 4)
    g_src = set(word_ngrams(src, 4))
    return {"numbers_kept": len(ns & no) / len(ns) if ns else None,
            "invented_numbers": len(no - ns),
            "content_recall": len(cs & co) / len(cs) if cs else None,
            "novel_4grams": (sum(g not in g_src for g in g_out) / len(g_out)) if g_out else None,
            "length_ratio": len(out.split()) / max(1, len(src.split())),
            "meta_text": bool(META.search(out[:200]))}


def diversity(outputs: list[str], seed: int = 0) -> dict:
    """Corpus-level diversity of the outputs (see the module docstring)."""
    from frontierlab.datax.neardup import MinHasher, doc_words, estimate_jaccard, shingles
    big = Counter(g for o in outputs for g in word_ngrams(o, 2))
    total = sum(big.values())
    heads = Counter(" ".join(o.lower().split()[:6]) for o in outputs)
    repeated_head = sum(c for h, c in heads.items() if c > 1 and h) / max(1, len(outputs))
    mh = MinHasher(64, seed)
    sigs = [mh.signature(shingles(doc_words(o), 3)) for o in outputs]
    rng = np.random.default_rng(seed)
    pairs = [(int(i), int(j)) for i, j in rng.integers(0, len(outputs), size=(min(2000, len(outputs) ** 2), 2)) if i != j]
    mean_j = float(np.mean([estimate_jaccard(sigs[i], sigs[j]) for i, j in pairs])) if pairs else 0.0
    return {"distinct_bigrams": len(big) / max(1, total), "repeated_opening": repeated_head,
            "mean_pairwise_jaccard": mean_j, "outputs": len(outputs)}


def generation_flops(n_nonemb: int, layers: int, d_attn: int, prompt: int, generated: int,
                     head_flops_per_token: int = 0) -> float:
    """Forward FLOPs to prefill ``prompt`` tokens and then generate ``generated`` tokens with a KV cache."""
    P, G = prompt, generated
    ctx_sum = G * P + G * (G - 1) / 2                     # Σ_{c=P}^{P+G-1} c
    return (2 * n_nonemb * (P + G) + 2 * layers * d_attn * P * P + 4 * layers * d_attn * ctx_sum
            + head_flops_per_token * (P + G))


def model_dims(model) -> dict:
    cfg = model.config
    emb = model.get_input_embeddings().weight.numel()
    tied = bool(getattr(cfg, "tie_word_embeddings", False))
    n_total = sum(p.numel() for p in model.parameters())
    n_nonemb = n_total - emb if tied else n_total - 2 * emb
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return {"n_total": n_total, "n_nonemb": n_nonemb, "layers": cfg.num_hidden_layers,
            "d_attn": cfg.num_attention_heads * head_dim, "vocab": cfg.vocab_size, "hidden": cfg.hidden_size,
            "head_flops_per_token": 2 * cfg.vocab_size * cfg.hidden_size}


class Rephraser:
    """A pinned Hugging Face chat model on CPU (or GPU), batched sampling with left padding."""

    def __init__(self, model_id: str, revision: str, device: str = "cpu", dtype: str = "float32"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id, self.revision, self.device = model_id, revision, device
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                                                          dtype=getattr(torch, dtype)).to(device).eval()
        self.dims = model_dims(self.model)

    def prompts(self, texts: list[str], style: str) -> list[str]:
        kw = {"enable_thinking": False} if "qwen3" in self.model_id.lower() else {}
        return [self.tok.apply_chat_template([{"role": "user", "content": STYLES[style].format(text=t)}],
                                             tokenize=False, add_generation_prompt=True, **kw) for t in texts]

    def generate(self, texts: list[str], style: str, max_new_tokens: int = 384, temperature: float = 0.7,
                 top_p: float = 0.9, seed: int = 0) -> tuple[list[str], list[dict]]:
        import torch
        torch.manual_seed(seed)
        enc = self.tok(self.prompts(texts, style), return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                      temperature=temperature, top_p=top_p, pad_token_id=self.tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        texts_out, stats = [], []
        for i in range(new.shape[0]):
            ids = new[i].tolist()
            g = len(ids)
            for j, t in enumerate(ids):
                if t in (self.tok.eos_token_id, self.tok.pad_token_id):
                    g = j
                    break
            p = int(enc["attention_mask"][i].sum())
            texts_out.append(self.tok.decode(ids[:g], skip_special_tokens=True).strip())
            stats.append({"prompt_tokens": p, "generated_tokens": g,
                          "flops": generation_flops(self.dims["n_nonemb"], self.dims["layers"], self.dims["d_attn"], p, g,
                                                    self.dims["head_flops_per_token"])})
        return texts_out, stats


def run(source: Path, out: Path, n_docs: int, style: str, model_id: str, revision: str, max_src: int = 256,
        batch: int = 8, max_new_tokens: int = 384, temperature: float = 0.7, top_p: float = 0.9, seed: int = 0,
        device: str = "cpu", skip: int = 0) -> dict:
    from frontierlab.data.loader import TokenData
    from frontierlab.datax.neardup import decode_docs
    from frontierlab.datax.sources import load_tokenizer
    out.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer()
    data = TokenData("train", source)
    texts = decode_docs(data, tok, skip + n_docs)[skip:]
    # truncate to max_src Data-v0 tokens at a word boundary
    srcs = []
    for t in texts:
        ids = tok.encode(t).ids[:max_src]
        s = tok.decode(ids)
        srcs.append(s[:s.rfind(" ")] if len(ids) == max_src and " " in s else s)
    done = []
    path = out / "rephrased.jsonl"
    if path.exists():
        done = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    reph = Rephraser(model_id, revision, device)
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as f:
        for i in range(len(done), len(srcs), batch):
            chunk = srcs[i:i + batch]
            tb = time.time()
            outs, stats = reph.generate(chunk, style, max_new_tokens, temperature, top_p, seed + i)
            per_doc = (time.time() - tb) / len(chunk)
            for j, (s, o, st) in enumerate(zip(chunk, outs, stats)):
                row = {"index": skip + i + j, "source": s, "output": o, **st, "seconds": round(per_doc, 3)}
                f.write(json.dumps(row) + "\n")
                done.append(row)
            f.flush()
            print(f"{len(done)}/{len(srcs)} ({time.time() - t0:.0f}s)", flush=True)
    seconds = time.time() - t0
    comp = {"model": model_id, "revision": revision, "style": style, "prompt": STYLES[style], "temperature": temperature,
            "top_p": top_p, "max_new_tokens": max_new_tokens, "seed": seed, "documents": len(done),
            "source": str(source), "skip": skip, "max_src_tokens": max_src, "dims": reph.dims,
            "prompt_tokens": sum(r["prompt_tokens"] for r in done),
            "generated_tokens": sum(r["generated_tokens"] for r in done),
            "flops": sum(r["flops"] for r in done), "seconds_this_session": round(seconds, 1),
            "seconds_total": round(sum(r.get("seconds", 0.0) for r in done), 1), "device": device}
    (out / "compute.json").write_text(json.dumps(comp, indent=2))
    return comp


def evaluate(out: Path) -> dict:
    rows = [json.loads(x) for x in (out / "rephrased.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    fid = [fidelity(r["source"], r["output"]) for r in rows]

    def mean(k):
        v = [f[k] for f in fid if f[k] is not None]
        return float(np.mean(v)) if v else None
    res = {"documents": len(rows), "numbers_kept": mean("numbers_kept"),
           "docs_with_invented_numbers": float(np.mean([f["invented_numbers"] > 0 for f in fid])),
           "content_recall": mean("content_recall"), "novel_4grams": mean("novel_4grams"),
           "length_ratio": mean("length_ratio"), "meta_text_rate": float(np.mean([f["meta_text"] for f in fid])),
           "empty_outputs": sum(not r["output"] for r in rows),
           "diversity_outputs": diversity([r["output"] for r in rows]),
           "diversity_sources": diversity([r["source"] for r in rows])}
    (out / "quality.json").write_text(json.dumps(res, indent=2))
    return res


def build(out: Path, dest: Path, include_source: bool = False) -> dict:
    """Write the outputs (optionally followed by their sources) as a training source; all go to ``train``."""
    from frontierlab.data.prepare import DEFAULT_OUT, EOT, sha256_file
    from frontierlab.datax.sources import load_tokenizer, write_source
    rows = [json.loads(x) for x in (out / "rephrased.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    comp = json.loads((out / "compute.json").read_text())
    tok = load_tokenizer()
    texts = [r["output"] for r in rows if r["output"]]
    prov = [{"source_index": r["index"], "kind": "rephrased"} for r in rows if r["output"]]
    if include_source:
        texts += [r["source"] for r in rows]
        prov += [{"source_index": r["index"], "kind": "original"} for r in rows]
    src_meta = json.loads((Path(comp["source"]) / "meta.json").read_text())
    meta = {"name": f"m10-reph-{comp['style']}", "vocab_size": tok.get_vocab_size(), "eot_id": tok.token_to_id(EOT),
            "tokenizer_sha256": sha256_file(DEFAULT_OUT / "tokenizer.json"),
            "split_rule": "all rephrased documents are training data; their sources are training documents",
            "provenance": {"dataset": f"rephrased from {src_meta.get('dataset')}", "config": src_meta.get("config"),
                           "revision": src_meta.get("revision"), "license": src_meta.get("license"),
                           "generator": {k: comp[k] for k in ("model", "revision", "style", "prompt", "temperature",
                                                              "top_p", "max_new_tokens", "seed")},
                           "generator_license": "Apache-2.0", "generator_flops": comp["flops"],
                           "command": f"python -m frontierlab.datax.rephrase build {out}"}}
    return write_source(dest, {"train": texts, "val": [], "test": []}, {"train": prov, "val": [], "test": []}, meta, tok)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--source", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--docs", type=int, default=300)
    r.add_argument("--skip", type=int, default=0)
    r.add_argument("--style", choices=sorted(STYLES), default="wiki")
    r.add_argument("--model", default="Qwen/Qwen3-0.6B")
    r.add_argument("--revision", default="c1899de289a04d12100db370d81485cdf75e47ca")
    r.add_argument("--max-src", type=int, default=256)
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--max-new-tokens", type=int, default=384)
    r.add_argument("--temperature", type=float, default=0.7)
    r.add_argument("--top-p", type=float, default=0.9)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--device", default="cpu")
    e = sub.add_parser("evaluate")
    e.add_argument("out", type=Path)
    b = sub.add_parser("build")
    b.add_argument("out", type=Path)
    b.add_argument("--dest", type=Path, required=True)
    b.add_argument("--include-source", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        print(json.dumps(run(a.source, a.out, a.docs, a.style, a.model, a.revision, a.max_src, a.batch,
                             a.max_new_tokens, a.temperature, a.top_p, a.seed, a.device, a.skip), indent=2))
    elif a.cmd == "evaluate":
        print(json.dumps(evaluate(a.out), indent=2))
    else:
        m = build(a.out, a.dest, a.include_source)
        print(json.dumps({k: m[k] for k in ("name", "train")}, indent=2))


if __name__ == "__main__":
    main()
