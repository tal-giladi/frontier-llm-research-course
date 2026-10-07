"""Lab 17.3: transcoders, attribution graphs, and graph hypotheses tested by intervention.

    python labs/module-17/lesson-03/graph_lab.py             # free CPU: about 80 s after lab 17.1's model exists
    python labs/module-17/lesson-03/graph_lab.py --variant main --print

1. Your functions against the course's.
2. A TopK transcoder (1,024 features, k = 16) for each of the 4 MLPs of the Module 17 model: FVU, L0, and the
   global replacement model (every MLP replaced, no error terms): next-token loss and top-1 agreement with the
   model on validation windows.
3. Attribution graphs for three validation prompts: node counts, pruned size, the share of logit influence
   that flows from error nodes, and the most influential features.
4. Hypothesis tests. For the 6 most influential features of each graph: the graph's prediction of what removing
   the feature does to the top logit (local replacement model: frozen attention and norms, downstream features
   recomputed) against what it does in the original model; then the same for 12 random active features with
   similar activations (the control).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from frontierlab.interp import graphs as G
from frontierlab.interp import tasks as T
from frontierlab.labkit import load_path

HERE = Path(__file__).resolve().parent

MAIN = """# Main path (1x 48-80 GB GPU; not run in this build, part of the Module 17 pilot). Env B: circuit-tracer 0.5.0.
python -m frontierlab.interp.hf circuit --print          # prints the env setup and the commands below
circuit-tracer attribute --prompt "Fact: the capital of the state containing Dallas is" \\
    --transcoder_set mwhanna/qwen3-1.7b-transcoders-lowl0 --slug dallas --graph_file_dir runs/m17/graphs
# Then, in Python: ReplacementModel.from_pretrained("Qwen/Qwen3-1.7B", "mwhanna/qwen3-1.7b-transcoders-lowl0"),
# attribute(prompt, rm), prune_graph(g, 0.8, 0.98), compute_graph_scores(g) -> (replacement, completeness),
# and rm.feature_intervention(prompt, [(layer, pos, feature, 0.0)]) for the top-6 and 12 random active features.
# The transcoders were trained on Qwen/Qwen3-1.7B (post-trained), not -Base: run the graphs on that model.
# PROJECTED (pending pilot): one attribution graph = one forward + a backward per influential node in batches of 512;
# circuit-tracer's README targets a single GPU; budget ~1-3 min per prompt on an A100 for ~20-token prompts,
# 10 prompts + 60 interventions ~0.5 GPU-hours. Memory: read the transcoder set's config for its width; with
# lazy_decoder=True (the default) decoders are loaded on demand.
"""


def check_learner(lab, g):
    inf = G.influence(g)
    ours = (g.logit_probs / g.logit_probs.sum()) @ lab.total_influence(lab.normalise_rows(g.A))
    is_logit = np.array([nd[0] == "logit" for nd in g.nodes])
    print(f"  influence matches: {np.allclose(ours, inf)}; prune matches: "
          f"{lab.prune_nodes(inf, is_logit, 0.8) == G.prune(g, 0.8)}")


def pick_prompts(model, n=3, T_=14, seed=5):
    """Validation prefixes where the model is confident about the next token (top-1 probability >= 0.3)."""
    w = T.data_windows("val", 400, T_, seed)
    with torch.no_grad():
        p = torch.softmax(model(w).logits[:, -1], -1).max(-1).values
    idx = (p >= 0.3).nonzero().flatten()[:n]
    return [w[i:i + 1] for i in idx.tolist()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["cpu", "main"], default="cpu")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    if a.print or a.variant == "main":
        print(MAIN)
        return
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")     # Windows consoles: decoded tokens
    t0 = time.time()
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    try:
        from tokenizers import Tokenizer
        from frontierlab.data.prepare import DEFAULT_OUT
        tok = Tokenizer.from_file(str(Path(DEFAULT_OUT) / "tokenizer.json"))
        dec = lambda ids: tok.decode(ids)
    except Exception:                                     # pragma: no cover
        dec = lambda ids: str(ids)
    model = T.m17_model()
    L = model.config.num_hidden_layers

    print("2. Transcoders (TopK, 1,024 features, k = 16) on each MLP")
    tr = T.data_windows("train", 384, 128, 0)
    va = T.data_windows("val", 48, 128, 1234)
    tcs = []
    for l in range(L):
        t = time.time()
        x, y = G.collect_mlp(model, tr, l)
        tc = G.train_transcoder(x, y, 1024, "topk", k=16, steps=2000, seed=l)
        xv, yv = G.collect_mlp(model, va, l)
        st = G.transcoder_fvu(tc, xv, yv)
        print(f"  layer {l}: {len(x):,} training tokens, held-out FVU {st['fvu']:.3f}, L0 {st['l0']:.1f} ({time.time() - t:.0f} s)")
        tcs.append(tc)
    with torch.no_grad():
        lm = model(va).logits
        rp = G.replaced_logits(model, va, tcs)
    def nll(lg):
        return float(-torch.log_softmax(lg[:, :-1], -1).gather(-1, va[:, 1:, None]).mean())
    agree = float((lm.argmax(-1) == rp.argmax(-1)).float().mean())
    print(f"  replacement model (all 4 MLPs replaced, no error terms): loss {nll(rp):.4f} vs model {nll(lm):.4f}; "
          f"top-1 next-token agreement {agree:.3f}")

    print("\n3. Attribution graphs")
    prompts = pick_prompts(model)
    graphs = []
    for k, idx in enumerate(prompts):
        t = time.time()
        g = G.attribute(model, idx, tcs, n_logits=3)
        keep = G.prune(g, 0.8)
        inf = G.influence(g)
        nf = sum(nd[0] == "feat" for nd in g.nodes)
        kf = sum(g.nodes[i][0] == "feat" for i in keep)
        top = g.meta["top"][0]
        print(f"  prompt {k}: ...{dec(idx[0, -8:].tolist())!r} -> {dec([top])!r} (p = {g.meta['top_probs'][0]:.2f})")
        print(f"    {len(g.nodes)} nodes ({nf} active features); pruned at 0.8: {len(keep)} nodes ({kf} features); "
              f"error-node share of influence {G.error_share(g):.3f} ({time.time() - t:.0f} s)")
        feats = sorted([i for i, nd in enumerate(g.nodes) if nd[0] == "feat"], key=lambda i: -inf[i])
        for i in feats[:4]:
            _, l, p, f = g.nodes[i]
            print(f"    feature L{l}/{f} at position {p} ({dec([int(idx[0, p])])!r}): activation "
                  f"{g.activations[i]:.2f}, influence {inf[i]:.3f}")
        graphs.append((idx, g, feats))
        if k == 0:
            try:
                check_learner(lab, g)
            except NotImplementedError as e:
                print(f"  {e} - finish the TODOs (the script uses the course's code)")

    print("\n4. Graph predictions tested by intervention (remove one feature; change of the top logit minus mean)")
    rng = np.random.default_rng(0)
    rows = {"top": [], "random": []}
    for idx, g, feats in graphs:
        top = g.meta["top"][0]
        pos = g.meta["pos"]
        base = G.target_logit(G.trace(model, idx)["logits"], pos, top)
        acts = np.array([g.activations[i] for i in feats])
        chosen = feats[:6]
        lo, hi = np.quantile(acts[:6], [0.0, 1.0])
        pool = [i for i in feats[6:] if lo * 0.5 <= g.activations[i] <= hi * 1.5]
        rand = list(rng.choice(pool, size=min(12, len(pool)), replace=False)) if pool else []
        for name, sel in (("top", chosen), ("random", rand)):
            for i in sel:
                _, l, p, f = g.nodes[i]
                real = G.target_logit(G.intervene_feature(model, idx, tcs, l, p, f, 0.0), pos, top) - base
                pred = G.target_logit(G.predicted_feature_effect(model, idx, tcs, l, p, f, 0.0), pos, top) - base
                rows[name].append((real, pred))
    for name in ("top", "random"):
        r = np.array(rows[name])
        if len(r) == 0:
            continue
        corr = np.corrcoef(r[:, 0], r[:, 1])[0, 1] if len(r) > 2 else float("nan")
        print(f"  {name:6s} features (n = {len(r)}): mean |real change| {np.abs(r[:, 0]).mean():.3f}, mean |predicted| "
              f"{np.abs(r[:, 1]).mean():.3f}, correlation real vs predicted {corr:+.3f}, sign agreement "
              f"{(np.sign(r[:, 0]) == np.sign(r[:, 1])).mean():.2f}")
    if rows["random"]:
        p95 = np.quantile(np.abs(np.array(rows["random"])[:, 0]), 0.95)
        frac = (np.abs(np.array(rows["top"])[:, 0]) > p95).mean()
        print(f"  top features whose real effect exceeds the random features' 95th percentile ({p95:.3f}): {frac:.2f}")
    print(f"\ntotal {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
