"""Evaluate one trained model past its trained length under several RoPE rules, without any training.

    python labs/module-04/lesson-02/zero_shot.py --run runs/m04/base-cpu --train-len 256 --eval-len 1024
    LAB_TARGET=solution python labs/module-04/lesson-02/zero_shot.py --run runs/m04/base-cpu --train-len 256 --eval-len 1024

The frequencies come from YOUR lab.py (``--impl library`` uses frontierlab.longctx.rope instead). For
each rule it scores the first ``--eval-len`` tokens of every held-out document that long (Eval v1,
component 1) and prints:

* mean loss per document-position bucket;
* the paired difference to the unchanged model ("none") per bucket, with a 95% bootstrap interval
  over documents — the positions below the trained length show what the rule costs at short range,
  the positions above it show what it buys;
* the context gain beyond the trained length (Eval v1, component 2, W = --train-len).

s = eval-len / train-len. Results go to ``--out`` (JSON) for the write-up.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from frontierlab.data.loader import TokenData
from frontierlab.data.prepare import DEFAULT_OUT
from frontierlab.evals import suite_v1 as v1
from frontierlab.labkit import load_target
from frontierlab.longctx import rope as lib
from frontierlab.longctx.attention import convert
from frontierlab.longctx.rope import ScaledRotaryEmbedding
from frontierlab.stats import bootstrap_ci, paired_bootstrap

RULES = ["none", "pi", "ntk", "yarn", "yarn-notemp"]


def with_rule(model, rule: str, s: float, L: int, impl):
    """A copy of ``model`` whose RoPE layers use ``rule``'s frequencies (weights unchanged)."""
    kind = model.config.attention if model.config.attention in ("gqa-irope", "gqa-rope-scaled") else "gqa-rope-scaled"
    m = convert(model, kind)
    for layer in m.model.layers:
        rope = layer.self_attn.rope
        D, base = rope.rot_dim, m.config.rope_theta
        inv, af = {"none": (impl.default_inv_freq(D, base), 1.0),
                   "pi": (impl.pi_inv_freq(D, base, s), 1.0),
                   "ntk": (impl.ntk_inv_freq(D, base, s), 1.0),
                   "yarn": (impl.yarn_inv_freq(D, base, s, L), impl.yarn_attention_factor(s)),
                   "yarn-notemp": (impl.yarn_inv_freq(D, base, s, L), 1.0)}[rule]
        layer.self_attn.rope = ScaledRotaryEmbedding(inv, af)
    return m.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--train-len", type=int, required=True)
    ap.add_argument("--eval-len", type=int, default=1024)
    ap.add_argument("--rules", nargs="+", default=RULES, choices=RULES)
    ap.add_argument("--impl", choices=["lab", "library"], default="lab")
    ap.add_argument("--data", type=Path, default=None, help="default: labs/common/data/v0-long if prepared, else Data-v0")
    ap.add_argument("--max-docs", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    impl = load_target(str(Path(__file__).parent / "test_lab.py")) if a.impl == "lab" else lib
    long_root = DEFAULT_OUT.parent / "v0-long"
    root = a.data or (long_root if (long_root / "meta.json").exists() else DEFAULT_OUT)
    data = TokenData("val", root)
    base = v1.load_model(a.run, a.device)
    s, T, W = a.eval_len / a.train_len, a.eval_len, a.train_len
    edges = sorted({0, 64, W // 2, W, *[e for e in (2 * W, 4 * W, 8 * W) if e < T - 1], T - 1})
    dt = torch.bfloat16 if a.bf16 else None
    t0 = time.perf_counter()
    res = {"run": a.run, "train_len": W, "eval_len": T, "s": s, "data": root.name, "edges": edges, "rules": {}}
    for rule in a.rules:
        m = with_rule(base, rule, s, W, impl)
        L = v1.doc_position_losses(m, data, T, a.max_docs, device=a.device, autocast_dtype=dt)
        gain = v1.context_gain(m, data, T, W, T // 2, T, a.max_docs, device=a.device, autocast_dtype=dt) if W <= T // 2 else None
        res["rules"][rule] = {"doc_bucket_means": [L[:, lo:hi].mean(1).tolist() for lo, hi in zip(edges[:-1], edges[1:])],
                              "context_gain": None if gain is None else gain.tolist()}
    n_docs = len(res["rules"][a.rules[0]]["doc_bucket_means"][0])
    print(f"{a.run}: trained at {W}, evaluated at {T} (s = {s:g}), {n_docs} held-out documents from {root.name}\n")
    head = "".join(f"[{lo},{hi})".rjust(13) for lo, hi in zip(edges[:-1], edges[1:]))
    print(f"{'mean loss':<14}{head}{'gain W=' + str(W):>16}")
    for rule, r in res["rules"].items():
        g = r["context_gain"]
        gs = f"{bootstrap_ci(g, n_boot=2000)[0]:+.4f}" if g is not None else "-"
        print(f"{rule:<14}" + "".join(f"{sum(b) / len(b):13.4f}" for b in r["doc_bucket_means"]) + f"{gs:>16}")
    if "none" in res["rules"]:
        print("\npaired difference to 'none' (negative = better), 95% CI over documents")
        for rule, r in res["rules"].items():
            if rule == "none":
                continue
            cells = []
            for b, b0 in zip(r["doc_bucket_means"], res["rules"]["none"]["doc_bucket_means"]):
                d = paired_bootstrap(b, b0, n_boot=2000)
                cells.append(f"{d['mean_diff']:+.3f}[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]")
            print(f"{rule:<14}" + "  ".join(cells))
    res["seconds"] = round(time.perf_counter() - t0, 1)
    print(f"\n{res['seconds']} s")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
