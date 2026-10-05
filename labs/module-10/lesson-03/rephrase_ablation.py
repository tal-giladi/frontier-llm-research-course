"""Lab 10.3: rephrase a small web corpus with a small open model, check it, and measure whether it is worth its compute.

    python labs/module-10/lesson-03/rephrase_ablation.py              # free CPU: generation ~30-40 min, training ~25 min

Stages (each cached; rerun the same command after an interruption):

1. **Generate.** The first ``--docs`` documents of the web pool (FineWeb, the training split), cut to
   ``--max-src`` Data-v0 tokens, rewritten twice by Qwen3-0.6B (pinned revision, temperature 0.7, top-p 0.9):
   in the ``wiki`` style (the training data) and in the ``qa`` style (a held-out *probe*: the same facts in a
   form none of the arms trains on). ``frontierlab.datax.rephrase`` records prompt and generated tokens,
   generator FLOPs and wall time.
2. **Check** the wiki outputs: numbers kept and invented, content-word recall, novelty, meta-text, diversity
   (your functions where the lab asks for them), and n-gram leakage of the outputs against LAMBADA.
3. **Train** four arms at equal tokens, 75% Data-v0 + 25% "slot" (3 seeds each): ``none`` (slot = more
   Data-v0), ``orig`` (slot = the original documents, repeated), ``reph`` (slot = the wiki rephrasings,
   repeated), ``both`` (slot = originals and rephrasings, 1:1).
4. **Score**: loss on the QA probe (primary: the facts in a new form), loss on the originals
   (the ``orig`` arm trained on them: an upper reference), Data-v0 and web validation loss, LAMBADA.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from frontierlab.data.prepare import DEFAULT_OUT
from frontierlab.datax import arms, leakage, rephrase
from frontierlab.datax.mixture import MixtureSpec, SourceRef
from frontierlab.datax.sources import M10_ROOT, load_tokenizer, write_source
from frontierlab.flops import flops_per_token
from frontierlab.labkit import load_path
from frontierlab.model import PRESETS

HERE = Path(__file__).resolve().parent
MODEL, REVISION = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"
VARIANTS = {
    "cpu": dict(preset="toy", steps=400, batch=16, seq=128, lr=3e-3, warmup=40, seeds=[0, 1, 2], device="cpu",
                docs=128, max_src=192, gen_batch=16, max_new=320, windows=128, lambada=1000),
    "t4": dict(preset="pilot-10m", steps=2000, batch=32, seq=512, lr=3e-3, warmup=100, seeds=[0, 1, 2], device="cuda",
               docs=2000, max_src=256, gen_batch=64, max_new=384, windows=256, lambada=5153),
    "main": dict(preset="pilot-30m", steps=4000, batch=64, seq=1024, lr=3e-3, warmup=200, seeds=[0, 1, 2],
                 device="cuda", docs=20000, max_src=256, gen_batch=128, max_new=384, windows=256, lambada=5153),
}
SLOT = 0.25


def folder(dest: Path, texts: list[str], split: str, name: str, tok, note: str) -> Path:
    if (dest / "meta.json").exists():
        return dest
    from frontierlab.data.prepare import EOT, sha256_file
    docs = {"train": [], "val": [], "test": []}
    docs[split] = texts
    prov = {s: [{"index": i} for i in range(len(v))] for s, v in docs.items()}
    meta = {"name": name, "vocab_size": tok.get_vocab_size(), "eot_id": tok.token_to_id(EOT),
            "tokenizer_sha256": sha256_file(DEFAULT_OUT / "tokenizer.json"), "note": note}
    write_source(dest, docs, prov, meta, tok)
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cpu")
    ap.add_argument("--out", type=Path, default=Path("runs/m10/l103"))
    ap.add_argument("--web", type=Path, default=M10_ROOT / "web")
    ap.add_argument("--generate-only", action="store_true")
    a = ap.parse_args(argv)
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    v = VARIANTS[a.variant]
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    tok = load_tokenizer()
    t0 = time.time()
    comp = {}
    for style in ("wiki", "qa"):
        g = a.out / f"gen-{style}"
        if not (g / "compute.json").exists() or json.loads((g / "compute.json").read_text())["documents"] < v["docs"]:
            rephrase.run(a.web, g, v["docs"], style, MODEL, REVISION, v["max_src"], v["gen_batch"], v["max_new"],
                         device=v["device"], seed=0 if style == "wiki" else 1)
        comp[style] = json.loads((g / "compute.json").read_text())
    rows = {s: [json.loads(x) for x in (a.out / f"gen-{s}" / "rephrased.jsonl").read_text(encoding="utf-8").splitlines()]
            for s in comp}
    if a.generate_only:
        return
    # 2. checks (the wiki outputs are the training data)
    src = [r["source"] for r in rows["wiki"]]
    out = [r["output"] for r in rows["wiki"]]
    nk = [lab.number_check(s, o) for s, o in zip(src, out)]
    kept = [k for k, _ in nk if k is not None]
    q = rephrase.evaluate(a.out / "gen-wiki")
    from frontierlab.evals import suite_v0
    lamb = suite_v0.load_lambada(suite_v0.download_lambada())
    idx = leakage.NgramIndex(13).add_texts(out).finish()
    leak = leakage.leakage_report(idx, lamb, 0.5, "lambada_vs_outputs")
    print("generation:", {s: {k: c[k] for k in ("documents", "prompt_tokens", "generated_tokens", "flops",
                                                    "seconds_this_session")} for s, c in comp.items()})
    print(f"wiki outputs: numbers kept {np.mean(kept):.3f}, docs with invented numbers "
          f"{np.mean([i > 0 for _, i in nk]):.3f}, novel 4-grams {np.mean([x for x in (lab.novel_ngram_share(s, o) for s, o in zip(src, out)) if x is not None]):.3f}, "
          f"distinct-2 outputs {lab.distinct_n(out):.3f} vs sources {lab.distinct_n(src):.3f}, content recall "
          f"{q['content_recall']:.3f}, meta-text {q['meta_text_rate']:.3f}, repeated openings {q['diversity_outputs']['repeated_opening']:.3f}")
    print(f"LAMBADA passages with >= 50% 13-gram overlap with the outputs: {leak['flagged']} of {leak['items']}")
    # 3. sources and probes
    srcs = {"orig": folder(a.out / "src-orig", src, "train", "l103-orig", tok, "original documents (truncated)"),
            "reph": folder(a.out / "src-reph", [o for o in out if o], "train", "l103-reph", tok, "wiki-style rephrasings"),
            "probe_qa": folder(a.out / "probe-qa", [r["output"] for r in rows["qa"] if r["output"]], "val", "l103-probe-qa",
                               tok, "QA-style rephrasings: held-out probe"),
            "probe_orig": folder(a.out / "probe-orig", src, "val", "l103-probe-orig", tok, "the originals as an eval set")}
    edu = str(DEFAULT_OUT)
    specs = {"none": MixtureSpec([SourceRef("edu", edu, 1.0)], name="l103-none"),
             "orig": MixtureSpec([SourceRef("edu", edu, 1 - SLOT), SourceRef("orig", str(srcs["orig"]), SLOT)], name="l103-orig"),
             "reph": MixtureSpec([SourceRef("edu", edu, 1 - SLOT), SourceRef("reph", str(srcs["reph"]), SLOT)], name="l103-reph"),
             "both": MixtureSpec([SourceRef("edu", edu, 1 - SLOT), SourceRef("orig", str(srcs["orig"]), SLOT / 2),
                                  SourceRef("reph", str(srcs["reph"]), SLOT / 2)], name="l103-both")}
    sets = {"probe_qa": str(srcs["probe_qa"]), "probe_orig": str(srcs["probe_orig"]), "edu": edu, "web": str(a.web)}
    results = {}
    for name, spec in specs.items():
        results[name] = {}
        for s in v["seeds"]:
            run = arms.train_arm(a.out / f"{name}-s{s}", spec, preset=v["preset"], steps=v["steps"], batch=v["batch"],
                                 seq=v["seq"], lr=v["lr"], warmup=v["warmup"], seed=s, device=v["device"],
                                 question="rephrase vs repeat a small corpus at equal tokens (10.3)")
            results[name][s] = arms.score(run, sets, windows=v["windows"], seq=v["seq"], n_lambada=v["lambada"], device=v["device"])
    acc = json.loads((a.out / f"orig-s{v['seeds'][0]}" / "mixture_accounting.json").read_text())
    print("slot epochs (orig arm):", {k: round(x["epochs"], 2) for k, x in acc["sources"].items()})
    rows_t = arms.table(results, "none", ["probe_qa", "probe_orig", "edu", "web", "lambada"], v["seeds"], decide=lab.decide)
    arms.print_table(rows_t)
    rows_o = arms.table({k: results[k] for k in ("orig", "reph", "both")}, "orig", ["probe_qa"], v["seeds"], decide=lab.decide)
    print("\nrephrase vs repeat (arm − orig) on the QA probe:")
    arms.print_table(rows_o)
    cfg = PRESETS[v["preset"]](vocab_size=8192)
    train_flops = flops_per_token(cfg, v["seq"]) * v["steps"] * v["batch"] * v["seq"]
    gen = comp["wiki"]["flops"]
    print(f"generator FLOPs (wiki set) {gen:.3e}; one training run {train_flops:.3e}; ratio {lab.cost_ratio(gen, train_flops):.1f}")
    (a.out / "summary.json").write_text(json.dumps({"compute": comp, "quality": q, "leak": {k: leak[k] for k in ("flagged", "items")},
                                                     "rows": rows_t, "rows_vs_orig": rows_o, "train_flops": train_flops}, indent=2))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
